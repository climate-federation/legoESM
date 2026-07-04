"""Early JAX distributed initialization.

Must be called before any legoESM or jax.numpy import that triggers XLA
backend discovery.  Importing this module is cheap: it only touches stdlib
and optionally mpi4py + jax (base package), never jax.numpy or legoESM.

Usage (at the very top of an entry-point script, before all other imports)::

    from legoesm.parallel.early_init import maybe_init_jax_distributed
    maybe_init_jax_distributed()

    # Safe to import legoESM / jax.numpy from here onward.
"""

from __future__ import annotations

import os
import socket
import zlib

# Idempotency flag — must NOT be a jax.process_count() probe: that call
# initialises the XLA backend, after which jax.distributed.initialize() raises
# "must be called before any JAX calls that might initialise the XLA backend"
# (issue #693: every multi-node run died here).
_INITIALIZED = False

# Legacy fixed coordinator port, kept as the last-resort fallback when no
# scheduler job id is present (matches the historical hardcoded value so
# launcher-less local runs keep working unchanged).
_LEGACY_COORDINATOR_PORT = 1234


def resolve_coordinator_port(default: int = _LEGACY_COORDINATOR_PORT) -> int:
    """Deterministic ``jax.distributed`` coordinator port for THIS job.

    Precedence:

    1. ``LEGOESM_COORDINATOR_PORT`` (explicit override);
    2. derived from the scheduler job id (``SLURM_JOB_ID`` / ``PBS_JOBID``)
       via crc32 into the dynamic-port range — every rank of one job
       computes the SAME port with no communication, while two different
       jobs sharing a node get different ports (a fixed port means the
       second job's coordinator dies with EADDRINUSE on shared clusters);
    3. ``default`` (the legacy fixed port) when no scheduler id exists.

    crc32 (not ``hash()``) because Python string hashing is randomized
    per process — ranks would disagree.
    """
    env = os.environ.get("LEGOESM_COORDINATOR_PORT")
    if env:
        return int(env)
    jobid = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
    if jobid:
        # 20000 + [0, 40000): stays inside the unprivileged range and clear
        # of the ephemeral-port ceiling on common Linux configs.
        return 20000 + zlib.crc32(jobid.encode()) % 40000
    return default


def resolve_local_device_ids() -> list[int]:
    """Local CUDA device index this process should claim, from the launcher env.

    Two supported launch conventions (#693):

    1. **One GPU visible per task** (``CUDA_VISIBLE_DEVICES`` pinned to a
       single device by the job script): local index 0 IS the pinned GPU.
    2. **All node GPUs visible to every task** (plain ``srun`` with a
       job-level ``--gres`` allocation, no ``--gpu-bind``): pick by the
       launcher's node-local rank (``SLURM_LOCALID`` /
       ``OMPI_COMM_WORLD_LOCAL_RANK`` / ``PALS_LOCAL_RANKID``).

    Convention 2 is the one that works with >1 GPU task per node: SLURM's
    ``--gpu-bind`` isolates each task's GPU in its own cgroup, which blocks
    the CUDA IPC that NCCL needs between on-node peers — every 2-node x
    3-GPU smoke died with ``Cuda failure 101 'invalid device ordinal'``
    inside ``ncclGroupEnd``/``ncclCommInitRankConfig`` (jobs 26030299,
    26030422). Un-isolated GPUs + local-rank binding is the standard
    JAX-on-SLURM recipe and what ``jax.distributed``'s own SLURM cluster
    auto-detection does.

    Falls back to ``[0]`` when no local-rank variable exists (serial or
    unknown launcher: claim the first visible device, the historical
    behaviour).
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        n_visible = len([d for d in visible.split(",") if d.strip()])
        if n_visible == 1:
            return [0]  # convention 1: the pinned GPU is local index 0
    local_rank = (
        os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("PALS_LOCAL_RANKID")
    )
    if local_rank is not None:
        return [int(local_rank)]
    return [0]


def init_jax_distributed_with_fallback() -> None:
    """``jax.distributed.initialize()`` with a PBS/PALS-safe fallback.

    Bare ``initialize()`` auto-detects SLURM / Open MPI (mpirun) / Cloud
    TPU only.  Under PBS + Cray PALS (Derecho ``mpiexec``) nothing is
    detected and it raises — so the transport is chosen by ENVIRONMENT,
    not by parsing exception text (real failures can mention
    "cluster"/"detect" too):

    - SLURM / Open MPI env present: bare ``initialize()`` (its cluster
      auto-detection also derives ``local_device_ids`` from the launcher's
      local-rank variable); any failure re-raises loudly.
    - PALS/PMI-only env (Derecho ``mpiexec``): the mpi4py bootstrap
      (``cluster_detection_method="mpi4py"``, the documented ALCF Cray-EX
      recipe) with :func:`resolve_local_device_ids` — the repo's PALS job
      shims pin ``CUDA_VISIBLE_DEVICES`` to ONE device per rank, so this
      resolves to local index 0 (the #693 device-binding convention).
      Plain-MPI
      bootstrap only; mpi4jax is never armed here, so the
      jax.distributed-vs-mpi4jax mixed-stack hazard does not apply.

    No-op if a previous call already initialized the runtime (shared
    ``_INITIALIZED`` flag with :func:`maybe_init_jax_distributed`; the
    "already initialized" RuntimeError from an out-of-band init is also
    treated as a no-op).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    import importlib.util

    import jax

    # Cross-path idempotency: an OUTER bootstrap
    # (initialize_jax_distributed_multiprocess / a launcher script) may have
    # federated the processes without setting THIS module's flag.
    # is_initialized() is the supported check (#749).
    if jax.distributed.is_initialized():
        _INITIALIZED = True
        return

    auto_detectable = any(
        v in os.environ for v in ("SLURM_JOB_ID", "OMPI_COMM_WORLD_SIZE")
    )
    pals_only = not auto_detectable and (
        "PALS_RANKID" in os.environ or "PMI_RANK" in os.environ
    )
    if pals_only and importlib.util.find_spec("mpi4py") is not None:
        jax.distributed.initialize(
            cluster_detection_method="mpi4py",
            local_device_ids=resolve_local_device_ids(),
        )
        _INITIALIZED = True
        return
    try:
        jax.distributed.initialize()
    except RuntimeError as e:
        if "already" in str(e).lower():
            _INITIALIZED = True
            return
        raise
    _INITIALIZED = True


def init_multicontroller_distributed(coordinator: str | None = None) -> None:
    """Initialize ``jax.distributed`` for a route-B multicontroller launch.

    Shared by every ``--multicontroller`` entry point (the ocean/atm SPMD
    benches and the ``run_omip`` route-B driver) so the launcher-env contract
    lives in ONE place.  MUST run before any other JAX use (backend init).

    - Explicit ``coordinator`` (``host:port``): read the launcher rank/size
      from Open MPI ``OMPI_COMM_WORLD_SIZE``/``RANK`` or Cray PALS
      ``PMI_SIZE``/``PMI_RANK`` and call ``jax.distributed.initialize``
      directly (the mpiexec path; also how the self-spawn tests inject rank).
    - No ``coordinator``: delegate to :func:`init_jax_distributed_with_fallback`
      (SLURM/OMPI auto-detect or the PALS mpi4py bootstrap).

    Real init failures re-raise loudly — a missing launcher rank env is a
    hard ``SystemExit``, never a silent single-process fallback (that would
    run N identical un-federated copies clobbering each other's output).
    """
    if coordinator is None:
        init_jax_distributed_with_fallback()
        return

    import jax

    n_procs = int(os.environ.get(
        "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "0")))
    proc_id = int(os.environ.get(
        "OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-1")))
    if n_procs < 1 or proc_id < 0:
        raise SystemExit(
            "--coordinator given but no launcher rank env found "
            "(OMPI_COMM_WORLD_SIZE/RANK or PMI_SIZE/PMI_RANK).")
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=n_procs, process_id=proc_id)
    global _INITIALIZED
    _INITIALIZED = True


def maybe_init_jax_distributed(coordinator_port: int | None = None) -> bool:
    """Initialize ``jax.distributed`` if running under multi-node MPI.

    Detects the MPI world size from the environment (SLURM_NTASKS /
    PMI_SIZE / OMPI_COMM_WORLD_SIZE).  When >1 rank and the ranks span
    more than one hostname, calls ``jax.distributed.initialize()`` with
    rank-0's hostname as coordinator.  On single-node MPI or serial runs,
    does nothing.

    Returns True if ``jax.distributed.initialize()`` was called, False
    otherwise.  Safe to call multiple times — idempotency is tracked via a
    module-level flag (NOT a ``jax.process_count()`` probe, which would
    initialise the XLA backend and then make ``initialize()`` raise; #693).

    ``coordinator_port=None`` (default) resolves the port via
    :func:`resolve_coordinator_port` (env override / job-id-derived /
    legacy 1234).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return False

    ntasks = int(
        os.environ.get(
            "SLURM_NTASKS",
            os.environ.get("PMI_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")),
        )
    )
    if ntasks <= 1:
        return False

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hosts = comm.allgather(socket.gethostname())
    if len(set(hosts)) <= 1:
        # Single-node MPI: JAX distributed not needed.
        return False

    import jax
    if coordinator_port is None:
        coordinator_port = resolve_coordinator_port()
    coordinator = f"{hosts[0]}:{coordinator_port}"
    # Device binding derived from the launcher env (resolve_local_device_ids):
    # [0] when the job script pins one GPU per task via CUDA_VISIBLE_DEVICES,
    # [SLURM_LOCALID] when all node GPUs are visible to every task. The former
    # hardcoded [0] broke >1-GPU-per-node launches (issue #693: either "no
    # supported devices found for platform CUDA" under rank-indexed
    # auto-assignment, or NCCL 'invalid device ordinal' under --gpu-bind cgroup
    # isolation — jobs 26030299/26030422).
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=size,
        process_id=rank,
        local_device_ids=resolve_local_device_ids(),
    )
    _INITIALIZED = True
    return True
