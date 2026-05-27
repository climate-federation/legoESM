"""Multi-node MPI initialization and state management.

Supports two decomposition modes:

1. **Face-only** (1–6 MPI ranks): each rank owns full faces.
2. **Sub-face tiling** (N ranks where N is a multiple of 6, N > 6):
   each face is split into tiles, each rank owns one tile.

Usage
-----
::

    from legoesm.parallel.distributed import initialize_distributed

    config, layout = initialize_distributed(return_layout=True)
    # Ranks now hold only local data — no zero-masked global arrays.

Design
------
Ranks store **only their local data**.  ``layout.scatter()`` extracts
the local portion from a global array; ``layout.gather()`` reconstructs
the global array via MPI allgather (for I/O, plotting, restart only).

The legacy ``partition_state`` / ``gather_state`` API has been removed.
Use ``scatter_to_local`` / ``gather_to_global`` for all new code.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from legoesm.parallel.comm import CommTopology, build_comm_topology
from legoesm.parallel.layout import (
    DistributedLayout,
    SingleRankLayout,
    make_layout,
    scatter as layout_scatter,
    scatter_pytree as layout_scatter_pytree,
    gather as layout_gather,
    gather_pytree as layout_gather_pytree,
)
from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    set_active_config,
)
from legoesm.parallel.reductions import _require_mpi_stack, _mpi4jax_array_result

_active_topology: CommTopology | None = None
_active_layout: DistributedLayout | SingleRankLayout | None = None


def initialize_distributed(
    *,
    return_topology: bool = False,
    return_layout: bool = False,
    global_n: int | None = None,
    grid_type: str = "cubed_sphere",
) -> DeviceConfig | tuple:
    """Initialize JAX distributed runtime and set up MPI halo exchange.

    Automatically detects the number of MPI ranks and selects face-only
    or sub-face tiling decomposition:

    - 1, 2, 3, or 6 ranks → face-only.
    - Multiple of 6 and >6 → sub-face tiling.

    Parameters
    ----------
    return_topology : bool
        If ``True``, include the :class:`CommTopology` in the return.
    return_layout : bool
        If ``True``, include the :class:`DistributedLayout` in the return.
    global_n : int, optional
        Per-face grid resolution.  Required for ``return_layout=True``
        to construct the layout.  If ``None`` and ``return_layout`` is
        requested, the layout is not created (caller must call
        ``make_layout`` later).

    Returns
    -------
    DeviceConfig, or tuple including topology and/or layout as requested.
    """
    global _active_topology, _active_layout

    if _active_topology is not None:
        warnings.warn(
            "initialize_distributed() called more than once. "
            "Returning existing configuration.",
            RuntimeWarning,
            stacklevel=2,
        )
        from legoesm.parallel.mesh import get_active_config
        config = get_active_config()
        result = [config]
        if return_topology:
            result.append(_active_topology)
        if return_layout:
            result.append(_active_layout)
        return tuple(result) if len(result) > 1 else result[0]

    # Distributed mode currently only supports cubed-sphere decomposition.
    _SUPPORTED_DISTRIBUTED_GRIDS = {"cubed_sphere"}
    if grid_type not in _SUPPORTED_DISTRIBUTED_GRIDS:
        raise ValueError(
            f"Distributed (MPI) execution does not support grid_type={grid_type!r}. "
            f"Supported: {sorted(_SUPPORTED_DISTRIBUTED_GRIDS)}. "
            f"Use single-node execution for {grid_type} grids."
        )

    # Validate MPI dependencies before touching JAX distributed runtime.
    _mpi4jax, MPI = _require_mpi_stack()

    # Get rank/size from MPI BEFORE initializing JAX distributed runtime.
    # This is the authoritative source — MPI is always available under mpirun.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()

    # Determine whether this is a multi-node run.  When all ranks share
    # the same hostname, JAX's distributed runtime (gRPC coordinator) is
    # unnecessary — each rank can already see all local devices.  MPI halo
    # exchange and reductions work via mpi4jax regardless.
    _is_multi_node = False
    if n_processes > 1:
        import socket
        my_hostname = socket.gethostname()
        all_hostnames = comm.allgather(my_hostname)
        _is_multi_node = len(set(all_hostnames)) > 1

    if _is_multi_node:
        # Multi-node: initialize JAX distributed runtime with MPI-derived
        # coordinator info.  Rank 0's hostname serves as coordinator.
        coordinator_address = all_hostnames[0]
        coordinator_port = 1234
        coordinator_bind = f"{coordinator_address}:{coordinator_port}"

        try:
            jax.distributed.initialize(
                coordinator_address=coordinator_bind,
                num_processes=n_processes,
                process_id=rank,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"jax.distributed.initialize() failed on rank {rank}/{n_processes} "
                f"with coordinator={coordinator_bind}. "
                f"Ensure the coordinator port {coordinator_port} is not in use "
                f"and all ranks can reach {coordinator_address}. "
                f"Original error: {e}"
            ) from e

        # Validate JAX agrees with MPI.
        jax_rank = jax.process_index()
        jax_size = jax.process_count()
        if jax_rank != rank or jax_size != n_processes:
            warnings.warn(
                f"JAX process_index/count ({jax_rank}/{jax_size}) differs from "
                f"MPI rank/size ({rank}/{n_processes}). Using MPI values.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif n_processes > 1:
        # Single-node MPI: skip jax.distributed.initialize().
        # All ranks share the same local devices; MPI halo exchange and
        # reductions work via mpi4jax independently.
        pass

    # Build communication topology for this rank.
    topology = build_comm_topology(rank, n_processes)
    _active_topology = topology

    # Set up MPI halo exchange.
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("mpi", topology)

    # Create device mesh with local devices.
    local_devices = list(jax.local_devices())
    local_device_count = len(local_devices)
    if local_device_count < 1:
        raise RuntimeError("No local JAX devices found for this MPI rank.")

    # Per-rank GPU affinity for single-node MPI.
    # When multiple MPI ranks share the same host (no JAX distributed),
    # all ranks see the same local_devices().  Assign a non-overlapping
    # slice to each rank to prevent oversubscription.
    if not _is_multi_node and n_processes > 1 and local_device_count > 1:
        # Number of co-located ranks on this host.
        n_ranks_on_host = n_processes  # single-node: all ranks are local
        gpus_per_rank = max(1, local_device_count // n_ranks_on_host)
        local_rank = rank  # single-node: rank == local rank
        start = local_rank * gpus_per_rank
        end = min(start + gpus_per_rank, local_device_count)
        local_devices = local_devices[start:end]
        local_device_count = len(local_devices)
        if local_device_count < 1:
            raise RuntimeError(
                f"Rank {rank}: no GPUs assigned after affinity slicing "
                f"({n_ranks_on_host} ranks, {gpus_per_rank} GPUs/rank)."
            )

    # Each MPI rank uses its local devices.
    # For face-only: pick largest divisor of 6 ≤ local_device_count.
    # For sub-face: each rank owns one tile, so use 1 device per rank
    # (multiple local GPUs within a rank can be used for intra-rank parallelism).
    if n_processes <= 6:
        mesh_devices = next(
            (d for d in (6, 3, 2, 1) if d <= local_device_count),
            1,
        )
    else:
        # Sub-face tiling: each rank owns one tile, use all local devices
        mesh_devices = local_device_count

    if mesh_devices != local_device_count and n_processes <= 6:
        warnings.warn(
            "Local device count does not evenly divide 6 cubed-sphere faces. "
            f"Using {mesh_devices} of {local_device_count} local devices.",
            RuntimeWarning,
            stacklevel=2,
        )
    local_config = create_device_mesh(
        n_devices=mesh_devices,
        devices=local_devices,
    )

    # Return a new config marking distributed mode.
    tiling = topology.tiling
    config = DeviceConfig(
        mesh=local_config.mesh,
        face_sharding=local_config.face_sharding,
        replicated_sharding=local_config.replicated_sharding,
        n_devices=local_config.n_devices,
        backend=local_config.backend,
        is_distributed=True,
        tiling=tiling,
        grid_type=grid_type,
    )
    set_active_config(config)

    # Build the rank-local layout if grid resolution is known.
    layout = None
    if global_n is not None:
        layout = make_layout(rank, n_processes, global_n)
        _active_layout = layout

    result = [config]
    if return_topology:
        result.append(topology)
    if return_layout:
        result.append(layout)
    return tuple(result) if len(result) > 1 else result[0]


def initialize_distributed_latlon(
    *,
    global_n_lat: int,
    global_n_lon: int | None = None,
):
    """Initialize the MPI halo backend for a latitude-band lat-lon run.

    Parallel entry point to :func:`initialize_distributed` for the
    SCVT/cubed-sphere grids — separate because the lat-lon path
    needs neither the face-topology dance nor JAX's device-mesh
    SPMD machinery.  All the lat-lon MPI work happens through
    mpi4jax sendrecv (see
    :mod:`legoesm.parallel.latlon_mpi`) and the backend-dispatched
    halo helpers (see :mod:`legoesm.grids.halo_latlon`).

    What this does
    --------------
    1. Reads ``rank`` and ``n_processes`` from ``MPI.COMM_WORLD``.
    2. Builds a :class:`LatLonBandLayout` for the band this rank
       owns.
    3. Activates ``set_halo_backend("mpi", layout)`` — every
       subsequent ``pad_halo_latlon`` / ``pad_with_pole_bc_lat``
       call inside the dycore dispatches through MPI sendrecv at
       partition cuts + pole-fold / wall-BC constants at boundary
       ranks, and ``_is_distributed()`` returns True (gating
       conservation reductions).

    What this does NOT do
    ---------------------
    * Does not initialize ``jax.distributed`` (multi-node JAX
      coordinator).  Single-node CPU MPI runs don't need it; if
      you need multi-node JAX SPMD, call :func:`initialize_distributed`
      first (cubed-sphere path) or extend this helper to take the
      coordinator arguments.
    * Does not scatter state/forcing.  Callers slice the global
      grid + initial state themselves using the returned
      ``LatLonBandLayout`` (see ``scatter_state_latlon``).
    * Does not register an ``_active_layout`` of the cubed-sphere
      ``DistributedLayout`` type (those carry ``ownership.face_ids``
      etc. which have no lat-lon analog).  The
      ``LatLonBandLayout`` is exposed via ``get_mpi_topology()``
      from :mod:`legoesm.grids.halo` for code that needs to query
      "what part of the global lat axis do I own".

    Parameters
    ----------
    global_n_lat : int
        Global number of latitude rows of the grid this rank's band
        slices into.
    global_n_lon : int, optional
        Global number of longitude columns.  Defaults to
        ``2 * global_n_lat`` (the standard square-cell AMIP layout).

    Returns
    -------
    LatLonBandLayout
        This rank's band layout.  Use
        ``layout.lat_start`` / ``layout.lat_end`` to slice the
        global grid + initial state for this rank.
    """
    global _active_topology
    if _active_topology is not None:
        warnings.warn(
            "initialize_distributed_latlon() called more than once. "
            "Returning the existing topology.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _active_topology

    if global_n_lon is None:
        global_n_lon = 2 * global_n_lat

    # Validate MPI dependencies before touching JAX.
    _mpi4jax, MPI = _require_mpi_stack()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()

    from legoesm.parallel.latlon_mpi import make_latlon_band_layout
    layout = make_latlon_band_layout(
        rank=rank, n_ranks=n_processes,
        n_lat=global_n_lat, n_lon=global_n_lon,
    )
    _active_topology = layout

    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("mpi", layout)
    return layout


def get_active_topology() -> CommTopology | None:
    """Return the active MPI communication topology, if initialized."""
    return _active_topology


def get_active_layout():
    """Return the active distributed layout, if initialized."""
    return _active_layout


def set_active_layout(layout) -> None:
    """Set the active distributed layout (for deferred construction)."""
    global _active_layout
    _active_layout = layout


# =========================================================================
# Rank-local API (new — no zero-masked global arrays)
# =========================================================================

def scatter_to_local(global_state, layout=None):
    """Extract rank-local data from a global ``(6, n, n, ...)`` pytree.

    Unlike the legacy ``partition_state``, this returns arrays with shape
    ``(n_local_faces, n, n, ...)`` — **not** a zero-masked ``(6, n, n)``
    global array.  This saves memory proportional to ``6 / n_local_faces``.

    Parameters
    ----------
    global_state
        Any JAX pytree with face-indexed arrays.
    layout : DistributedLayout or SingleRankLayout, optional
        Layout for this rank.  Uses active layout if ``None``.

    Returns
    -------
    Rank-local pytree.
    """
    layout = layout or _active_layout
    if layout is None:
        raise ValueError(
            "No layout provided and no active layout is set. "
            "Call initialize_distributed(global_n=...) first."
        )
    return layout_scatter_pytree(global_state, layout)


def gather_to_global(local_state, layout=None):
    """Reconstruct a global ``(6, n, n, ...)`` pytree from rank-local data.

    This is the inverse of :func:`scatter_to_local`.  Uses MPI allgather
    under the hood.

    Use this **only** for I/O, plotting, restart writing, or debug.

    Parameters
    ----------
    local_state
        Rank-local pytree from :func:`scatter_to_local`.
    layout : DistributedLayout or SingleRankLayout, optional
        Layout for this rank.

    Returns
    -------
    Global pytree.
    """
    layout = layout or _active_layout
    if layout is None:
        raise ValueError(
            "No layout provided and no active layout is set. "
            "Call initialize_distributed(global_n=...) first."
        )
    return layout_gather_pytree(local_state, layout)
