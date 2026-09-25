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

# Default coordinator port for ``jax.distributed.initialize`` (the gRPC rendezvous
# socket rank 0 binds; every other rank dials ``rank0_host:PORT``).  A module
# constant (not a magic literal) so the SLURM launcher and any future caller agree.
_JAX_DIST_COORDINATOR_PORT = 1234


def _require_mpi4py():
    """Return ``mpi4py.MPI`` or raise a clear ImportError.

    The multi-controller SPMD multi-process path needs ONLY MPI rank/hostname
    discovery (to derive the ``jax.distributed`` coordinator) — its halo +
    reductions run through pure-JAX ``ppermute``/``psum`` inside ``shard_map``,
    NOT mpi4jax.  So this helper requires mpi4py only, unlike
    :func:`legoesm.parallel.reductions.require_mpi_stack`
    (which also requires mpi4jax for the cubed-sphere ``sendrecv`` halo).
    """
    import importlib.util

    if importlib.util.find_spec("mpi4py") is None:
        raise ImportError(
            "Multi-process jax.distributed bootstrap requires mpi4py (for rank / "
            "hostname discovery). Install it and launch under mpirun/srun "
            "(one process per device)."
        )
    from mpi4py import MPI

    return MPI


def initialize_jax_distributed_multiprocess(
    *,
    coordinator_port: int | None = None,
    local_device_ids=None,
):
    """Initialize the ``jax.distributed`` runtime for a multi-PROCESS run, deriving
    the coordinator from MPI rank/hostname — the mpi4jax-free bootstrap used by
    the multi-controller SPMD paths (cs_spmd production driver, lat-lon SPMD
    ocean step) to span GPUs across several nodes.

    This factors the SAME proven coordinator-discovery logic as the multi-node
    branch of :func:`initialize_distributed` (rank 0's hostname is the coordinator;
    every rank calls ``jax.distributed.initialize(addr, num_processes, process_id)``)
    but:

    * requires ONLY mpi4py for the MULTI-rank path (no mpi4jax — the SPMD halo is
      pure-JAX ppermute/psum); a genuinely single-process run (no MPI/PMI/SLURM
      launcher reporting >1 task) returns ``(0, 1)`` WITHOUT importing mpi4py, so
      the default single-controller path never depends on the optional dep;
    * does NOT arm the MPI halo backend (the SPMD step arms its own per-call
      halo backend around the ``shard_map``);
    * is a NO-OP for a single process — the default single-controller path stays
      byte-unchanged;
    * is idempotent — once ``jax.distributed`` is initialized (or the process is
      single), re-calling just returns the discovered ``(rank, n_processes)``.

    MUST be called BEFORE any ``jax`` array op / device query, because
    ``jax.distributed.initialize`` reconfigures the global device set so that
    ``jax.devices()`` returns ALL devices across ALL processes (and
    ``jax.local_devices()`` only this process' GPUs).

    Returns
    -------
    (rank, n_processes) : tuple[int, int]
        This process' global rank and the total process count (from MPI).
    """
    import os

    # Fast single-process short-circuit WITHOUT importing mpi4py: when no MPI/PMI/
    # SLURM launcher reports >1 task, this is a plain single-process run -> return
    # (0, 1) and do not even require mpi4py (the default path must not depend on an
    # optional dep).  Covers Open MPI (OMPI_COMM_WORLD_SIZE), MPICH/PMI
    # (PMI_SIZE), and bare srun (SLURM_NTASKS / SLURM_STEP_NUM_TASKS).
    _launcher_size = None
    for _var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
                 "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"):
        _v = os.environ.get(_var)
        if _v:
            try:
                _launcher_size = int(_v)
            except ValueError:
                _launcher_size = None
            break
    if _launcher_size is not None and _launcher_size <= 1:
        return 0, 1
    if _launcher_size is None:
        # NO launcher env at all (codex Medium): overwhelmingly a plain
        # ``python script.py`` — honour the documented no-optional-dep
        # single-process contract when mpi4py is absent.  When mpi4py IS
        # importable, still probe COMM_WORLD (an exotic launcher that
        # exports none of the four vars gets correct rank discovery
        # rather than a silent N-way replicated-serial run).
        import importlib.util
        if importlib.util.find_spec("mpi4py") is None:
            return 0, 1

    MPI = _require_mpi4py()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_processes = comm.Get_size()

    if n_processes <= 1:
        # Single process under a launcher (e.g. mpirun -np 1): the default single-
        # controller path.  jax sees all LOCAL devices; no coordinator needed.
        return rank, n_processes

    # Already initialized (idempotent re-entry): just report the topology.
    if jax.distributed.is_initialized():
        return rank, n_processes

    import socket

    my_hostname = socket.gethostname()
    all_hostnames = comm.allgather(my_hostname)
    coordinator_address = all_hostnames[0]
    if coordinator_port is None:
        # Env override / crc32(job-id)-derived / legacy fixed port: two jobs
        # sharing a node must not collide on the rendezvous socket
        # (EADDRINUSE on the second job's rank 0).
        from legoesm.parallel.early_init import resolve_coordinator_port

        coordinator_port = resolve_coordinator_port(
            default=_JAX_DIST_COORDINATOR_PORT)
    coordinator_bind = f"{coordinator_address}:{coordinator_port}"

    # Per-process LOCAL device id(s).  On a multi-GPU node with one process per
    # GPU, jax's auto-detect of the per-process device from the global CUDA
    # topology can DEADLOCK (DEADLINE_EXCEEDED on key cuda:global_topology) when
    # several co-located processes each see all the node's GPUs.  Declaring this
    # process' local GPU index explicitly (SLURM exposes it as SLURM_LOCALID under
    # srun) makes the topology exchange deterministic.  ONLY auto-derive on a GPU
    # platform: a CPU multi-process run (mpirun) wants MULTIPLE CPU devices per
    # process (xla_force_host_platform_device_count) -> pinning local_device_ids to
    # one would wrongly claim a single CPU device.  Falls back to None (jax
    # auto-detect) for single-GPU-per-node, CPU, or non-SLURM launches.
    if local_device_ids is None:
        _plats = (os.environ.get("JAX_PLATFORMS")
                  or os.environ.get("JAX_PLATFORM_NAME") or "")
        _is_gpu = ("cuda" in _plats.lower()) or ("gpu" in _plats.lower())
        _localid = os.environ.get("SLURM_LOCALID")
        if _is_gpu and _localid is not None:
            try:
                local_device_ids = int(_localid)
            except ValueError:
                local_device_ids = None

    _init_kwargs = dict(
        coordinator_address=coordinator_bind,
        num_processes=n_processes,
        process_id=rank,
    )
    if local_device_ids is not None:
        _init_kwargs["local_device_ids"] = local_device_ids

    try:
        jax.distributed.initialize(**_init_kwargs)
    except RuntimeError as e:
        raise RuntimeError(
            f"jax.distributed.initialize() failed on rank {rank}/{n_processes} "
            f"with coordinator={coordinator_bind}. Ensure the coordinator port "
            f"{coordinator_port} is free and every rank can reach "
            f"{coordinator_address}. Original error: {e}"
        ) from e
    from legoesm.parallel.early_init import pin_gloo_interface
    pin_gloo_interface()

    # Validate JAX agrees with MPI (same check as initialize_distributed).
    jax_rank = jax.process_index()
    jax_size = jax.process_count()
    if jax_rank != rank or jax_size != n_processes:
        warnings.warn(
            f"JAX process_index/count ({jax_rank}/{jax_size}) differs from MPI "
            f"rank/size ({rank}/{n_processes}). Using MPI values.",
            RuntimeWarning,
            stacklevel=2,
        )
    return rank, n_processes


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
        # Multi-node: initialize the JAX distributed runtime with MPI-derived
        # coordinator info (rank 0's hostname is the coordinator).  Delegate to
        # the shared bootstrap so the discovery + initialize + MPI-vs-JAX
        # validation live in ONE place (the SPMD paths use the same helper);
        # single-node MPI is handled by the elif below.  The helper is
        # idempotent via ``jax.distributed.is_initialized()`` — it never
        # probes ``jax.process_count()`` first (which would itself
        # initialise the XLA backend and guarantee a subsequent
        # ``initialize()`` raise — the #693 init-ordering bug class).
        initialize_jax_distributed_multiprocess()
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
    band_boundaries: tuple[int, ...] | None = None,
):
    """Initialize the MPI halo backend for a latitude-band lat-lon run.

    ``band_boundaries`` (optional): explicit band boundaries forwarded to
    :func:`legoesm.parallel.latlon_mpi.make_latlon_band_layout` — use
    :func:`legoesm.parallel.latlon_mpi.wet_band_boundaries` to balance WET
    cells across bands instead of row counts (land-heavy bands otherwise
    idle).  Every rank must pass the IDENTICAL boundaries (deterministic
    host computation from the global mask).  ``None`` keeps the even row
    split byte-identically.

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
        # A changed band-boundary request must not be served by a stale
        # layout (codex): a long-lived process that armed even bands and
        # later opts into wet-cell-balanced boundaries (or vice versa) would
        # silently keep the OLD decomposition — every slicer/scatter reads
        # lat_start/lat_end from the layout. Compare this rank's requested
        # span against the active one; re-arm fresh on mismatch.
        _r = _active_topology.rank
        _n = _active_topology.n_ranks
        if band_boundaries is not None:
            # Validate BEFORE the span comparison: an invalid request
            # (non-integral / overlong / bad span) must raise, never be
            # silently served by a coincidentally-matching stale layout
            # (codex round 2).
            from legoesm.parallel.latlon_mpi import validate_band_boundaries
            _b = validate_band_boundaries(band_boundaries, _n, global_n_lat)
            _want_span = (_b[_r], _b[_r + 1])
        else:
            _base, _rem = divmod(global_n_lat, _n)
            _s = (_r * (_base + 1) if _r < _rem
                  else _rem * (_base + 1) + (_r - _rem) * _base)
            _want_span = (_s, _s + _base + (1 if _r < _rem else 0))
        if (_active_topology.lat_start,
                _active_topology.lat_end) != _want_span:
            warnings.warn(
                "initialize_distributed_latlon() re-called with different "
                f"band boundaries (rank {_r}: requested rows "
                f"[{_want_span[0]}, {_want_span[1]}) vs active "
                f"[{_active_topology.lat_start}, "
                f"{_active_topology.lat_end})); replacing the active layout "
                "and re-arming the MPI halo backend.",
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
        boundaries=band_boundaries,
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
