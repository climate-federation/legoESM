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

from legoesm.parallel.comm import CommTopology, build_comm_topology
from legoesm.parallel.layout import (
    DistributedLayout,
    SingleRankLayout,
    make_layout,
    scatter_pytree as layout_scatter_pytree,
    gather_pytree as layout_gather_pytree,
)
from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    set_active_config,
)
from legoesm.parallel.reductions import require_mpi_stack

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
        # FV3_3D 2026-05-27: re-assert the MPI halo backend on every call.
        # The function contract is "set up MPI halo exchange"; an earlier
        # `set_halo_backend("local")` (used in reference-vs-MPI bit-for-bit
        # tests) must not silently outlive a follow-up `initialize_distributed`.
        from legoesm.grids.halo import get_halo_backend, set_halo_backend
        if get_halo_backend() != "mpi":
            set_halo_backend("mpi", _active_topology)
        # FV3_3D iter-1058 (codex iter-1056 WARN #2): rebuild
        # ``_active_layout`` when the caller's ``global_n`` differs
        # from the first-init layout's ``global_n``.  Previously the
        # re-entry branch silently returned the stale layout, which
        # would feed wrong tile sizes to ``scatter_to_local`` /
        # ``gather_to_global``.  The current FV3 step-fidelity tests
        # never call scatter, so the stale layout was latent, but any
        # future test or production code path that re-initializes
        # with a different grid size needs the layout refreshed.
        if global_n is not None:
            need_rebuild = (
                _active_layout is None
                or getattr(_active_layout, "global_n", None) != global_n
            )
            if need_rebuild:
                _active_layout = make_layout(
                    rank=_active_topology.rank,
                    n_ranks=_active_topology.n_processes,
                    global_n=global_n,
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
    _mpi4jax, MPI = require_mpi_stack()

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
    fold=None,
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
       ranks, and ``is_distributed()`` returns True (gating
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
    fold : FoldDescriptor, optional
        Tripolar north-fold descriptor (issue #353).  When supplied
        (and ``fold.is_active``), the returned layout carries it so the
        northernmost rank applies the permutation-based tripolar fold at
        the north boundary.  Pass ``geometry.fold`` for an ORCA / eORCA
        ocean run; omit (``None``) for regular lat-lon.

    Returns
    -------
    LatLonBandLayout
        This rank's band layout.  Use
        ``layout.lat_start`` / ``layout.lat_end`` to slice the
        global grid + initial state for this rank.
    """
    global _active_topology
    if _active_topology is not None:
        from legoesm.parallel.latlon_mpi import LatLonBandLayout
        if not isinstance(_active_topology, LatLonBandLayout):
            # A prior NON-lat-lon distributed init (cubed-sphere
            # CommTopology, Voronoi layout) left its topology in the
            # shared ``_active_topology`` slot.  Silently returning it
            # hands the lat-lon caller the wrong type
            # (``AttributeError: 'CommTopology' object has no
            # attribute 'lat_start'`` downstream — merge gate
            # 8460562, mixed-grid test process).  Re-arm for lat-lon
            # instead of the type-blind reuse.
            warnings.warn(
                "initialize_distributed_latlon() called after a "
                "non-lat-lon distributed init; replacing the active "
                "topology with a LatLonBandLayout and re-arming the "
                "MPI halo backend.",
                RuntimeWarning, stacklevel=2,
            )
            _active_topology = None
    if _active_topology is not None:
        _want_n_lon = (2 * global_n_lat if global_n_lon is None
                       else global_n_lon)
        if (_active_topology.n_lat_global != global_n_lat
                or _active_topology.n_lon_global != _want_n_lon):
            # Same hydra, third head (after the cross-grid TYPE reuse
            # above and the test backend leaks): reusing a layout for
            # a DIFFERENT global grid silently mis-slices every band
            # (and may carry a foreign tripolar fold) — the np=2 PCG
            # step-parity 1e-5 drift in merge gate 8460563 was exactly
            # a leaked smaller fold-active layout.  Re-arm fresh.
            warnings.warn(
                "initialize_distributed_latlon() re-called with a "
                f"different global grid ({global_n_lat}x{_want_n_lon} "
                f"vs active {_active_topology.n_lat_global}x"
                f"{_active_topology.n_lon_global}); replacing the "
                "active layout and re-arming the MPI halo backend.",
                RuntimeWarning, stacklevel=2,
            )
            _active_topology = None
    if _active_topology is not None:
        active_fold = getattr(_active_topology, "fold", None)
        active_on = (active_fold is not None
                     and getattr(active_fold, "is_active", False))
        requested_on = fold is not None and getattr(fold, "is_active", False)
        if active_on and not requested_on:
            # Mirror of the upgrade case below: a leaked TRIPOLAR layout
            # must not serve a REGULAR-grid request — the foreign north
            # fold permutes/sign-flips the northern band rows of every
            # subsequent pad (merge gate 8460563: 1e-5 step-parity drift
            # from exactly this same-dims stale-fold reuse).  Re-arm
            # without the fold.
            warnings.warn(
                "initialize_distributed_latlon() re-called WITHOUT a "
                "tripolar fold after a fold-active init; replacing the "
                "active layout with a fold-less one and re-arming the "
                "MPI halo backend.",
                RuntimeWarning, stacklevel=2,
            )
            updated = _active_topology._replace(fold=None)
            _active_topology = updated
            from legoesm.grids.halo import set_halo_backend
            set_halo_backend("mpi", updated)
            return updated
        if requested_on and not active_on:
            # A prior fold-less init must NOT mask a later tripolar (ORCA)
            # init — otherwise the ocean run would silently use the
            # geographic pole-fold.  Update the active layout to carry the
            # fold and re-arm the MPI halo backend.
            warnings.warn(
                "initialize_distributed_latlon() re-called with a tripolar "
                "fold after a fold-less init; updating the active layout to "
                "carry the fold.",
                RuntimeWarning, stacklevel=2,
            )
            updated = _active_topology._replace(fold=fold)
            _active_topology = updated
            from legoesm.grids.halo import set_halo_backend
            set_halo_backend("mpi", updated)
            return updated
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
    _mpi4jax, MPI = require_mpi_stack()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()

    from legoesm.parallel.latlon_mpi import make_latlon_band_layout
    layout = make_latlon_band_layout(
        rank=rank, n_ranks=n_processes,
        n_lat=global_n_lat, n_lon=global_n_lon,
        fold=fold,
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


def reset_distributed_topology() -> None:
    """Disarm the MPI halo backend and forget the active topology/layout.

    For test isolation and multi-run drivers: an armed backend leaks
    into EVERYTHING padded afterwards in the same process — a later
    test's 'serial reference' silently dispatches its pads through the
    leftover band layout (merge gate 8460566: a deterministic 1e-5
    step-parity 'drift' that was really a band-MPI-padded serial
    reference).  Mirrors the documented contract that callers of
    ``initialize_distributed_latlon`` deactivate with
    ``set_halo_backend('local')`` once a run finishes — this helper
    also clears the module topology so the next init starts fresh.
    """
    global _active_topology, _active_layout
    _active_topology = None
    _active_layout = None
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")
    from legoesm.parallel.voronoi_mpi import reset_voronoi_layout
    reset_voronoi_layout()


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
