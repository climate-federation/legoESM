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


def launcher_world_size() -> int:
    """World size DECLARED by the job launcher's environment (0 = none).

    Reads the union of the launcher families every entry point supports:
    SLURM step (``SLURM_STEP_NUM_TASKS``), Open MPI
    (``OMPI_COMM_WORLD_SIZE``), PMI/PALS (``PMI_SIZE``), then the
    allocation-wide ``SLURM_NTASKS`` last.  Used by the post-init fallback
    guard — the launcher's declaration is the ground truth a federated
    runtime must match.
    """
    # Precedence = closeness to THIS process's launcher: the srun STEP
    # size, then the MPI launcher's own world (mpiexec inside a SLURM
    # allocation exports OMPI/PMI sizes — the truth), and only then the
    # allocation-wide SLURM_NTASKS (weakest: it describes the allocation,
    # not necessarily this launch; codex).
    for var in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE",
                "PMI_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return 0


def check_no_silent_process_fallback() -> None:
    """Fail LOUDLY when the federated process count disagrees with the
    launcher's declared world size.

    The route-B hazard this guards: ``jax.distributed.initialize`` (or an
    auto-detect miss) silently federates FEWER processes than the launcher
    started — N un-federated copies then run the same program, clobber each
    other's output, and a bench records a fake single-process row as an
    N-rank result.  Mirror of the route-A ``check_no_silent_mpi_fallback``
    (runtime.py); same override env for emergencies:
    ``LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI=1``.

    Call AFTER ``jax.distributed.initialize`` — ``jax.process_count()`` is
    then safe (the backend is already federated).
    """
    if os.environ.get("LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI") == "1":
        return
    declared = launcher_world_size()
    if declared <= 1:
        return
    import jax

    actual = int(jax.process_count())
    if actual != declared:
        raise RuntimeError(
            f"jax.distributed federated {actual} process(es) but the "
            f"launcher declared {declared} (SLURM_NTASKS / "
            f"OMPI_COMM_WORLD_SIZE / PMI_SIZE): a silent fallback would run "
            f"{declared} un-federated copies and record fake scaling rows. "
            f"Fix the launch (coordinator/port/env) or set "
            f"LEGOESM_ALLOW_SINGLE_PROCESS_UNDER_MPI=1 to override.")


def _launcher_local_rank() -> tuple[str, str] | None:
    """(env var, value) of the launcher's NODE-LOCAL rank, or ``None``.

    Union of the launcher families the init paths serve: Cray PALS,
    Open MPI, MVAPICH, and SLURM (SLURM_LOCALID only on a genuine
    multi-task launch — on a single-task sbatch step it is exported too
    and pinning on it would hide all but GPU 0 from a single-process
    multi-GPU run, the documented silent eff=0.5 bug).
    """
    for var in ("PALS_LOCAL_RANKID", "OMPI_COMM_WORLD_LOCAL_RANK",
                "MV2_COMM_WORLD_LOCAL_RANK"):
        v = os.environ.get(var)
        if v is not None and v.isdigit():
            return var, v
    slid = os.environ.get("SLURM_LOCALID")
    # Step-scoped multi-task guard (launcher_world_size prefers
    # SLURM_STEP_NUM_TASKS): a single-task step inside a larger
    # allocation must not bind on SLURM_LOCALID.
    if slid is not None and slid.isdigit() and launcher_world_size() > 1:
        return "SLURM_LOCALID", slid
    return None


def _pals_local_device_ids() -> list[int]:
    """Per-process ``local_device_ids`` for a multicontroller launch.

    When the job shim already pinned ``CUDA_VISIBLE_DEVICES`` to ONE device
    (the #693 convention), the process sees exactly one visible device →
    ``[0]``.  Otherwise (unpinned, or a MULTI-device visible list) index by
    the launcher's node-local rank (PALS / Open MPI / MVAPICH / guarded
    SLURM — see :func:`_launcher_local_rank`) so ranks sharing a node bind
    DIFFERENT devices instead of all contending for GPU 0 (the
    fake/contended-GPU row).  Falls back to ``[0]``.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_visible = len([x for x in cvd.split(",") if x.strip()]) if cvd else 0
    if n_visible == 1:
        return [0]  # shim-pinned: exactly one visible device
    local = _launcher_local_rank()
    if local is not None:
        var, val = local
        idx = int(val)
        if n_visible > 1 and idx >= n_visible:
            # More local ranks than visible devices is a LAUNCH error —
            # clamping would silently oversubscribe the last GPU (codex).
            raise RuntimeError(
                f"{var}={idx} but CUDA_VISIBLE_DEVICES exposes only "
                f"{n_visible} device(s): more local ranks than visible "
                f"GPUs. Fix the launcher ppn / CUDA_VISIBLE_DEVICES shim "
                f"(one rank per GPU).")
        return [idx]
    return [0]


def _nvidia_gpu_count() -> int:
    """Number of NVIDIA GPUs physically present on THIS host (0 = none).

    Reads the kernel driver's ``/proc/driver/nvidia/gpus`` listing — a pure
    filesystem probe that never initialises CUDA/XLA (this module's contract:
    cheap, pre-backend).  NOT ``nvidia-smi``: process-level tooling is
    forbidden here, and device LISTINGS ignore ``CUDA_VISIBLE_DEVICES``
    anyway (#1516's vacuous-guard trap).
    """
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


def _nvml_accessible_gpus() -> tuple[int, list[str] | None]:
    """(count, UUIDs) of GPUs THIS process can actually reach, or
    ``(-1, None)`` when NVML is unavailable.

    NVML is cgroup-aware — under SLURM ``ConstrainDevices=yes`` it
    enumerates exactly the devices the task's cgroup admits (measured on
    Levante, job 26815351: ``--gpus-per-task=1`` -> count 1 with a DISTINCT
    UUID per rank; job-level constraint only -> count 2 on every rank) —
    and it never initialises CUDA/XLA (management library; safe under this
    module's pre-backend contract).  It IGNORES ``CUDA_VISIBLE_DEVICES``,
    which is exactly what a binding decision needs: the accessible set,
    not the advertised one.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        n = int(pynvml.nvmlDeviceGetCount())
        uuids = []
        for i in range(n):
            u = pynvml.nvmlDeviceGetUUID(
                pynvml.nvmlDeviceGetHandleByIndex(i))
            uuids.append(u.decode() if isinstance(u, bytes) else u)
        return n, uuids
    except Exception:
        return -1, None


def _pin_local_rank_gpu_visibility() -> bool:
    """Per-rank GPU binding for a single-node multi-rank launch (#1516).

    The multi-NODE path binds ranks to devices inside
    ``jax.distributed.initialize(local_device_ids=_pals_local_device_ids())``;
    the single-node path skips that call entirely (jax.distributed is not
    needed), so NOTHING assigns rank -> device and every local rank boots
    on default GPU 0 — a silent 1-GPU run wearing an N-GPU costume that
    measures as a believable ~1.0x "speedup" (#1516).

    Binding decision (measured launcher contract, Levante job 26815351):

    - ``CUDA_VISIBLE_DEVICES`` multi-entry (plain ``srun -nN``: every task
      gets the full list ``0,1``): index the list by the launcher's
      node-local rank (:func:`_launcher_local_rank`, the same helper the
      multi-node path indexes with).
    - ``CUDA_VISIBLE_DEVICES`` single-entry or unset: trust it ONLY when
      the cgroup really isolates this task (NVML accessible count <= 1 —
      the ``--gpus-per-task=1`` / ``--gpu-bind=single:1`` / #693-shim
      shapes, where each task's ``0`` maps to a DIFFERENT physical GPU).
      When NVML sees MORE THAN ONE accessible device, an identical
      single-entry CVD on every rank is the collision itself (Levante
      ``--gpu-bind=none`` hands every task ``CVD=0`` while the job cgroup
      admits both GPUs — both ranks then compute on one physical device,
      measured twice: jobs 26806063 and 26815351) -> repin THIS rank to
      the NVML UUID at its node-local rank.  UUIDs, not indices: under a
      cgroup the accessible set need not start at physical index 0.

    No-ops (returns False) when: ``LEGOESM_NO_LOCAL_GPU_PIN=1``; an explicit
    non-GPU ``JAX_PLATFORMS``/``JAX_PLATFORM_NAME`` selection; an empty or
    disabled (``-1``/``NoDevFiles``) ``CUDA_VISIBLE_DEVICES``; a genuinely
    isolated / single-GPU task; a host with no NVIDIA GPUs (CPU MPI runs);
    or a single-entry CVD that cannot be cross-checked (no NVML).

    Raises RuntimeError — never a silent unpinned run — when GPUs are
    present but the rank cannot be bound: no launcher local-rank variable,
    or more local ranks than devices (mirrors ``_pals_local_device_ids``).
    """
    if os.environ.get("LEGOESM_NO_LOCAL_GPU_PIN") == "1":
        return False
    # Explicit non-GPU platform selection: nothing to pin (keeps CPU-backend
    # MPI test runs on GPU nodes working, e.g. JAX_PLATFORMS=cpu).
    plats = os.environ.get(
        "JAX_PLATFORMS", os.environ.get("JAX_PLATFORM_NAME", ""))
    if plats and not any(
            p.strip().lower() in ("cuda", "gpu")
            for p in plats.split(",")):
        return False
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = (
        [x.strip() for x in cvd.split(",") if x.strip()]
        if cvd is not None else None)
    if visible is not None and (
            not visible or visible[0] in ("-1", "NoDevFiles")):
        # Explicitly emptied or disabled: CUDA hides all GPUs itself.
        return False

    n_acc, acc_uuids = _nvml_accessible_gpus()
    if n_acc == 0:
        return False  # CPU-only host: nothing to pin.
    if visible is not None and len(visible) == 1:
        if 0 <= n_acc <= 1 or acc_uuids is None:
            # cgroup really isolates this task (gpus-per-task/#693 shim:
            # every rank's '0' is a DIFFERENT physical GPU), or no NVML to
            # tell — respect the external decision.
            return False
        # NVML sees >1 accessible device behind an identical single-entry
        # CVD: the measured Levante collision shape — repin below.
    if visible is None and n_acc < 0:
        # No CVD and no NVML: fall back to the physical /proc count
        # (workstation case, no cgroup to bias it).
        n_acc = _nvidia_gpu_count()
        if n_acc == 0:
            return False

    n_devices = (
        len(visible) if visible is not None and len(visible) > 1 else n_acc)
    local = _launcher_local_rank()
    if local is None:
        raise RuntimeError(
            f"single-node multi-rank launch with {n_devices} GPU(s) "
            "visible but no launcher node-local rank variable "
            "(PALS_LOCAL_RANKID / OMPI_COMM_WORLD_LOCAL_RANK / "
            "MV2_COMM_WORLD_LOCAL_RANK / SLURM_LOCALID): cannot bind "
            "ranks to distinct GPUs, and unpinned every rank boots on "
            "default GPU 0 (#1516: a silent 1-GPU run measured as an "
            "N-GPU result). Pin per rank before python starts (export "
            "CUDA_VISIBLE_DEVICES=$SLURM_LOCALID, the #693 shim) or set "
            "LEGOESM_NO_LOCAL_GPU_PIN=1 to accept unpinned devices.")
    var, val = local
    idx = int(val)
    if idx >= n_devices:
        # Mirrors _pals_local_device_ids: clamping would silently
        # oversubscribe the last GPU.
        where = ("in CUDA_VISIBLE_DEVICES"
                 if visible is not None and len(visible) > 1
                 else "accessible to this task")
        raise RuntimeError(
            f"{var}={idx} but only {n_devices} GPU(s) {where}: "
            "more local ranks than GPUs. Fix the launcher ppn / "
            "CUDA_VISIBLE_DEVICES shim (one rank per GPU) or set "
            "LEGOESM_NO_LOCAL_GPU_PIN=1 to accept unpinned devices.")
    if visible is not None and len(visible) > 1:
        pinned = visible[idx]
    elif acc_uuids is not None:
        pinned = acc_uuids[idx]
    else:
        pinned = str(idx)
    os.environ["CUDA_VISIBLE_DEVICES"] = pinned
    print(
        f"[early_init] single-node multi-rank: pinned "
        f"CUDA_VISIBLE_DEVICES={pinned} ({var}={idx}; #1516)",
        flush=True)
    return True


def nccl_transport_report() -> dict:
    """Best-effort NCCL transport facts for run metadata (route-B analog of
    the mpi4jax GPU-direct preflight).

    NCCL has no Python-queryable transport API; what IS knowable up front:
    the fabric env knobs and whether an OFI/net plugin library
    (``libnccl-net*``) is discoverable on ``LD_LIBRARY_PATH``/``LD_PRELOAD``
    / ``NCCL_NET_PLUGIN``.  ``missing_net_plugin_multi_node`` records the
    FACT of a multi-node launch with no net plugin visible — on OFI fabrics
    (Derecho Slingshot) that means NCCL silently runs correct-but-slow TCP
    sockets (git cba9715b2: 'route-B NCCL works cross-node but
    socket-bound'); native-IB fabrics run fine without a plugin, which is
    why the field states the fact, not the inference.  Advisory (the
    definitive check stays ``NCCL_DEBUG=INFO`` in the job log); recorded so
    a socket-bound row is falsifiable from the record.
    """
    import glob

    plugin_hit = None
    if os.environ.get("NCCL_NET_PLUGIN"):
        plugin_hit = os.environ["NCCL_NET_PLUGIN"]
    else:
        # LD_PRELOAD entries are separated by SPACES or colons (ld.so(8));
        # a colon-only split records nccl_net_plugin=None for the canonical
        # space-separated form and falsely flags a multi-node socket fallback
        # (pre-merge codex finding).  LD_LIBRARY_PATH stays colon-only.
        _preload = os.environ.get("LD_PRELOAD", "").replace(":", " ").split()
        paths = _preload + os.environ.get("LD_LIBRARY_PATH", "").split(":")
        for d in (p for p in paths if p):
            if os.path.isfile(d) and "libnccl-net" in os.path.basename(d):
                plugin_hit = d
                break
            hits = glob.glob(os.path.join(d, "libnccl-net*"))
            if hits:
                plugin_hit = hits[0]
                break
    n_nodes = 0
    for var in ("SLURM_NNODES", "SLURM_JOB_NUM_NODES", "PALS_NNODES"):
        v = os.environ.get(var)
        if v and v.isdigit():
            n_nodes = int(v)
            break
    multi_node = n_nodes > 1
    return {
        "nccl_net_plugin": plugin_hit,
        "nccl_net": os.environ.get("NCCL_NET"),
        "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
        "nccl_debug": os.environ.get("NCCL_DEBUG"),
        "n_nodes_declared": n_nodes,
        # The FACT (no net plugin visible on a multi-node launch), not an
        # inference: fabrics with native IB verbs run fine without a
        # plugin — 'sockets likely' is the DERECHO (Slingshot/OFI) reading
        # of this flag, stated in the warning text, not the field name
        # (codex).
        "missing_net_plugin_multi_node": bool(
            multi_node and plugin_hit is None),
    }


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
      recipe) with ``local_device_ids=[0]`` — the repo's PALS job shims pin
      ``CUDA_VISIBLE_DEVICES`` to ONE device per rank, so local index 0 is
      the pinned GPU (the #693 device-binding convention).  Plain-MPI
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
    # is_initialized() is the supported check (#749).  Still verify the
    # OUTER federation against the launcher's declared world size — a
    # pre-initialized 1-process runtime under N launcher ranks is the same
    # silent-fallback hazard (codex).
    if jax.distributed.is_initialized():
        _INITIALIZED = True
        check_no_silent_process_fallback()
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
            # Shim-pinned CUDA_VISIBLE_DEVICES -> [0]; unpinned bare
            # mpiexec -> index by PALS_LOCAL_RANKID so node-sharing ranks
            # bind DIFFERENT GPUs (contended-GPU-0 hazard).
            local_device_ids=_pals_local_device_ids(),
        )
        _INITIALIZED = True
        check_no_silent_process_fallback()
        return
    try:
        jax.distributed.initialize()
    except RuntimeError as e:
        if "already" in str(e).lower():
            _INITIALIZED = True
            check_no_silent_process_fallback()
            return
        raise
    _INITIALIZED = True
    check_no_silent_process_fallback()


def _warn_missing_nccl_plugin(rank: int | None = None) -> None:
    """Rank-0 warning when a multi-node launch has no NCCL net plugin
    visible — cross-node collectives then likely run correct-but-slow TCP
    sockets (the documented Derecho shape).  ``rank=None`` derives the rank
    from the federated runtime (safe post-init)."""
    report = nccl_transport_report()
    if not report["missing_net_plugin_multi_node"]:
        return
    if rank is None:
        try:
            import jax

            rank = int(jax.process_index())
        except Exception:
            rank = 0
    if rank == 0:
        print(
            "[early_init] WARNING: multi-node launch with no NCCL net "
            "plugin visible (libnccl-net*/NCCL_NET_PLUGIN): cross-node "
            "collectives will likely run on TCP SOCKETS (correct but "
            "slow — the documented Derecho socket-bound shape). Verify "
            "with NCCL_DEBUG=INFO; build/load the aws-ofi-nccl plugin "
            "for fabric speed.", flush=True)


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

    Idempotent: a no-op if the federation is already up (another entry point —
    e.g. run_amip's import-time ``maybe_init_jax_distributed`` on a real MPI
    launch — may have initialized first; a second ``jax.distributed.initialize``
    would raise "already initialized").
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    if coordinator is None:
        init_jax_distributed_with_fallback()
        _warn_missing_nccl_plugin()
        return

    import jax

    # Cross-path idempotency on the explicit-coordinator path too: an
    # out-of-band bootstrap may have federated already — a second
    # initialize() raises (codex).  Still verify the federation size.
    if jax.distributed.is_initialized():
        _INITIALIZED = True
        check_no_silent_process_fallback()
        return

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
        num_processes=n_procs, process_id=proc_id,
        # Same per-rank device binding as the PALS bootstrap path: a bare
        # multi-device CUDA_VISIBLE_DEVICES must not bind every local rank
        # to GPU 0 (codex).
        local_device_ids=_pals_local_device_ids())
    _INITIALIZED = True
    check_no_silent_process_fallback()
    _warn_missing_nccl_plugin(rank=proc_id)


def maybe_init_jax_distributed(coordinator_port: int | None = None) -> bool:
    """Initialize ``jax.distributed`` if running under multi-node MPI.

    Detects the MPI world size from the environment (SLURM_NTASKS /
    PMI_SIZE / OMPI_COMM_WORLD_SIZE).  When >1 rank and the ranks span
    more than one hostname, calls ``jax.distributed.initialize()`` with
    rank-0's hostname as coordinator.  On single-node multi-rank MPI it
    skips ``jax.distributed`` but still applies the per-rank GPU binding
    (:func:`_pin_local_rank_gpu_visibility`, #1516) — without it every
    local rank silently boots on default GPU 0.  On serial runs, does
    nothing.

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

    # Shared precedence (step > OMPI/PMI > allocation-wide NTASKS): an
    # `srun -n1` inside a larger allocation must NOT enter the MPI path
    # (codex).
    ntasks = launcher_world_size()
    if ntasks <= 1:
        return False

    # Per-rank GPU binding MUST happen BEFORE the mpi4py import below:
    # a CUDA-aware MPI stack (Levante: Open MPI/UCX) initialises the CUDA
    # driver during MPI_Init, and the driver snapshots CUDA_VISIBLE_DEVICES
    # at first initialisation — a pin applied after the import is silently
    # ignored (measured, job 26829100: post-pin CVD='0'/'1' per rank, yet
    # every rank still enumerated BOTH GPUs and computed on device 0).
    # The pin needs only launcher env + NVML, no communicator.  On the
    # multi-NODE lane the result equals the #693 shim convention
    # (_pals_local_device_ids then resolves the one visible device to [0]).
    _pin_local_rank_gpu_visibility()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hosts = comm.allgather(socket.gethostname())
    if len(set(hosts)) <= 1:
        # Single-node MPI: JAX distributed is not needed — the per-rank
        # GPU binding was already applied above (#1516); unpinned, every
        # local rank boots on default GPU 0 and the job completes with
        # most of the hardware idle.
        return False

    import jax
    if coordinator_port is None:
        coordinator_port = resolve_coordinator_port()
    coordinator = f"{hosts[0]}:{coordinator_port}"
    # Per-rank device binding via the shared launcher-family helper: a
    # shim-pinned CUDA_VISIBLE_DEVICES (the SLURM --gpu-bind=single:1
    # standard, #693) resolves to [0] exactly as before; an UNPINNED or
    # multi-device visible list indexes by the launcher's node-local rank
    # (guarded SLURM_LOCALID / OMPI / PALS) instead of piling every local
    # rank onto GPU 0 (codex).
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=size,
        process_id=rank,
        local_device_ids=_pals_local_device_ids(),
    )
    _INITIALIZED = True
    check_no_silent_process_fallback()
    return True
