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


# GPU-masking sentinels: SLURM/NVIDIA set these on a CPU-only allocation.
# Non-empty strings, so they must be rejected explicitly (mirrors the set in
# ``runtime/backend.py``, kept local because ``core/`` must not import
# ``runtime/`` at module scope).
_CUDA_DISABLED = frozenset({"-1", "NoDevFiles", ""})


def _host_visible_gpus() -> list[str]:
    """CUDA device ids visible to THIS process, without initialising CUDA.

    Returns the entries of ``CUDA_VISIBLE_DEVICES`` when it is set, else the
    device ordinals CUDA would assign, counted from the ``/dev/nvidiaN``
    character devices this process can actually open.  Empty list on a host
    with no NVIDIA driver.

    ``/dev`` rather than ``/proc/driver/nvidia/gpus``: under a device cgroup
    or container allocation ``/proc`` stays host-global, so a 4-GPU node with
    only GPU 0 granted still counts 4 and ranks get pinned to devices they
    cannot open (codex).  The ``/dev`` nodes track the allocation, and CUDA
    renumbers what it can see from 0 — hence ``range(n)``, not the minor
    numbers.

    Deliberately NOT ``nvidia-smi``: it ignores ``CUDA_VISIBLE_DEVICES``, so
    a gate built on it passes even with pinning removed (#1516).

    Limitations, stated rather than silently wrong: MIG instances (addressed
    by UUID, not ``/dev`` node) and ROCm/HIP (``HIP_VISIBLE_DEVICES``) are
    not enumerated — on those platforms pin explicitly in the launcher.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        if cvd in _CUDA_DISABLED or cvd.startswith("-"):
            # SLURM/NVIDIA mask GPUs on a CPU-only allocation with values
            # that are non-empty STRINGS, so a truthiness check reads them
            # as one visible device and "pins" a rank to device "-1"
            # (codex; same trap `runtime/backend.py` documents).
            return []
        return [x.strip() for x in cvd.split(",") if x.strip()]
    return [str(i) for i in range(len(openable_nvidia_nodes()))]


def openable_nvidia_nodes() -> list[str]:
    """Sorted ``/dev/nvidiaN`` paths this process can actually open.

    Doubles as the per-rank *device-set fingerprint*: under a
    ``--gpu-bind``-style cgroup each rank can open a DIFFERENT node, which is
    what distinguishes correct one-GPU-per-rank pinning from two ranks
    sharing one GPU (see :func:`maybe_init_jax_distributed`).
    """
    import glob
    return sorted(d for d in glob.glob("/dev/nvidia[0-9]*") if _openable(d))


def _openable(path: str) -> bool:
    """Can THIS process open ``path`` the way CUDA will?

    An ``open(O_RDWR|O_NONBLOCK)`` probe, not ``os.access``: ``access(2)``
    answers for the REAL uid/gid, so under a setuid launcher wrapper or a
    capability grant it reports "no" for a device the process can in fact use
    and the pin then raises a spurious "more ranks than GPUs" (GLM-5.2).
    O_RDWR, not O_RDONLY: CUDA needs write access to the device node, so a
    cgroup granting read-only would otherwise pass here and fail later inside
    backend creation (codex).  Opening the node does not initialise CUDA.
    """
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return False
    os.close(fd)
    return True


def _non_gpu_platform_selected() -> bool:
    """Explicit non-GPU JAX platform selection (``JAX_PLATFORMS=cpu`` etc.).

    Used to scope GPU-binding refusals: a run that will never create a CUDA
    backend must not die for want of a GPU pin.  An unset/empty selection
    counts as GPU-possible (JAX defaults to CUDA when it is available).
    """
    # ``get(A, get(B))`` does NOT fall back when A is set-but-EMPTY: a stale
    # ``export JAX_PLATFORMS=`` in the environment made an explicit
    # ``JAX_PLATFORM_NAME=cpu`` invisible, so a CPU-only run could still hit
    # the GPU-pin refusal.  Measured on Levante (job 26851115).
    plats = (os.environ.get("JAX_PLATFORMS")
             or os.environ.get("JAX_PLATFORM_NAME") or "")
    # ``rocm`` is a GPU platform too (``expand_platform_alias`` maps ``gpu`` ->
    # ``['cuda', 'rocm']``); classifying it as non-GPU would suppress the
    # refusal on exactly the hardware it protects.  Empty entries from a
    # trailing comma are dropped rather than counted as a non-GPU selection.
    entries = [p.strip().lower() for p in plats.split(",") if p.strip()]
    return bool(entries) and not any(
        p in ("cuda", "rocm", "gpu") for p in entries)


def _cuda_driver_fds_open() -> bool:
    """Does THIS process already hold open ``/dev/nvidia*`` descriptors?

    A CUDA-aware MPI (Open MPI/UCX) initialises the CUDA driver during
    ``MPI_Init`` and the driver keeps its device nodes open for the process
    lifetime — so open ``/dev/nvidia*`` fds after ``MPI_Init`` mean the
    driver has ALREADY snapshotted ``CUDA_VISIBLE_DEVICES`` and a narrowing
    applied now will be silently ignored (measured: Levante jobs 26829100 /
    26846811 — post-pin CVD read '0'/'1' per rank while every rank still
    enumerated and computed on BOTH GPUs).  Pure ``/proc`` scan; never
    initialises CUDA itself.  :func:`_openable`'s probe closes its fd
    immediately, so it cannot trip this.
    """
    try:
        fds = os.listdir("/proc/self/fd")
    except OSError:
        return False
    for fd in fds:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            return True
    return False


def pin_local_gpu(local_rank: int, n_local: int | None) -> str | None:
    """Bind this process to ONE GPU by narrowing ``CUDA_VISIBLE_DEVICES``.

    The #693 convention is that the job shim pins one device per rank before
    python starts.  When it has NOT (a bare single-node ``srun``/``mpirun``),
    every rank grabs the default device and the whole job runs on GPU 0 while
    the rest of the allocation idles — no error, ~1.0x "speedup", a result
    that reads as a clean refutation of multi-GPU scaling (#1516).

    Must be called before any JAX call that initialises the XLA backend, AND
    — on a CUDA-aware MPI stack — before ``MPI_Init``: the driver snapshots
    ``CUDA_VISIBLE_DEVICES`` at first initialisation, so a later narrowing is
    silently ignored (measured, Levante jobs 26829100 / 26846811).

    Args:
        local_rank: this process's rank among the processes on THIS host.
        n_local: how many processes share this host, or ``None`` when that is
            not yet knowable (the pre-``MPI_Init`` early pin, which has only
            the launcher environment).  With ``None`` the rank index itself
            is the oversubscription witness — a launcher-reported node-local
            rank ``i`` implies at least ``i+1`` ranks on this host — and the
            single-process no-op is skipped (the caller has already
            established a multi-rank launch and a launcher local-rank var).

    Returns the ``CUDA_VISIBLE_DEVICES`` value it set, or ``None`` when
    nothing needed changing (already pinned to a single device, single
    process on the host, or no GPU here).  The pre-pin value is preserved in
    ``LEGOESM_HOST_CUDA_VISIBLE_DEVICES`` so provenance/topology code can
    still recover how many GPUs the host offered (GLM-5.2).

    ``LEGOESM_ALLOW_SHARED_GPU=1`` opts into deliberate oversubscription
    (MPS): more ranks than GPUs are then spread round-robin instead of
    refused.

    Raises:
        RuntimeError: more local ranks than GPUs on the host (unless that
            override is set), or a node-local rank outside the visible
            devices.  Clamping would silently oversubscribe the last GPU.
            Callers running under MPI must make this raise COLLECTIVE — see
            :func:`maybe_init_jax_distributed`.
    """
    visible = _host_visible_gpus()
    if (n_local is not None and n_local <= 1) or len(visible) <= 1:
        # Single process per host, already shim-pinned, or CPU-only host.
        return None
    oversubscribed = (n_local > len(visible) if n_local is not None
                      else local_rank >= len(visible))
    if oversubscribed:
        if _non_gpu_platform_selected():
            # More ranks than GPUs, but the run selected a non-GPU platform
            # (JAX_PLATFORMS=cpu on a GPU node): no CUDA backend will ever be
            # created, so there is nothing to pin and nothing to refuse
            # (measured: the six-process CPU duo parity died here, job 9902131).
            return None
        if os.environ.get("LEGOESM_ALLOW_SHARED_GPU") == "1":
            # Deliberate oversubscription (MPS): spread round-robin instead of
            # refusing.  Without this the flag would be a lie — it gated only
            # the duplicate tripwire, so `LEGOESM_ALLOW_SHARED_GPU=1 mpirun
            # -np 4` on a 2-GPU node still aborted here (GLM-5.2).
            os.environ.setdefault("LEGOESM_HOST_CUDA_VISIBLE_DEVICES",
                                  ",".join(visible))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible[
                local_rank % len(visible)]
            return os.environ["CUDA_VISIBLE_DEVICES"]
        if n_local is None:
            # Pre-MPI_Init early pin: the launcher local-rank index already
            # exceeds the visible devices.  Either genuinely more local ranks
            # than GPUs, or a stale *_LOCAL_RANK/SLURM_LOCALID inherited from
            # an outer allocation — pre-MPI the two are indistinguishable.
            raise RuntimeError(
                f"launcher node-local rank {local_rank} but only "
                f"{len(visible)} GPU(s) visible: more local ranks than GPUs "
                f"(or a stale SLURM_LOCALID/*_LOCAL_RANK from an outer "
                f"allocation). Fix the launcher tasks-per-node, pin "
                f"CUDA_VISIBLE_DEVICES yourself (one rank per GPU), or set "
                f"LEGOESM_ALLOW_SHARED_GPU=1 to share deliberately.")
        raise RuntimeError(
            f"{n_local} ranks share this host but only {len(visible)} GPU(s) "
            f"are visible: more local ranks than GPUs. Fix the launcher "
            f"tasks-per-node, pin CUDA_VISIBLE_DEVICES yourself (one rank per "
            f"GPU), or set LEGOESM_ALLOW_SHARED_GPU=1 to share deliberately.")
    if local_rank >= len(visible):
        # A DIFFERENT failure from the one above (codex): the counts fit, but
        # this rank's node-local index does not — a stale/inherited
        # SLURM_LOCALID from an outer allocation, say.  Naming it "more ranks
        # than GPUs" would send the reader to the wrong knob.
        raise RuntimeError(
            f"node-local rank {local_rank} is out of range for the "
            f"{len(visible)} visible GPU(s) on a host running {n_local} "
            f"rank(s): inconsistent launcher rank metadata (a stale "
            f"SLURM_LOCALID / *_LOCAL_RANK inherited from an outer "
            f"allocation?). Unset it, or pin CUDA_VISIBLE_DEVICES yourself.")
    os.environ.setdefault("LEGOESM_HOST_CUDA_VISIBLE_DEVICES", ",".join(visible))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank]
    return visible[local_rank]


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
      recipe) with ``local_device_ids=_pals_local_device_ids()`` — which is
      ``[0]`` whenever the repo's PALS job shims have pinned
      ``CUDA_VISIBLE_DEVICES`` to ONE device per rank (the #693
      device-binding convention), and the launcher's node-local rank
      otherwise.  Plain-MPI
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
    rank-0's hostname as coordinator.

    On a SERIAL run this does nothing.  On MULTI-RANK runs — single-node
    included — it first binds each rank to one GPU via
    :func:`pin_local_gpu`; the single-node path takes no other action.  That
    binding is not optional bookkeeping: without it nothing assigns
    rank -> device on the single-node path and the whole job silently runs
    on GPU 0 (#1516).

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

    # Per-rank GPU pin BEFORE the mpi4py import (#1516 follow-up): a
    # CUDA-aware MPI stack (Levante: Open MPI/UCX) initialises the CUDA
    # driver during MPI_Init, and the driver snapshots CUDA_VISIBLE_DEVICES
    # at first initialisation — a pin applied after the import is silently
    # IGNORED while the env var reads correctly per rank (measured twice:
    # job 26829100 on the #1541 branch and job 26846811 on the merged tree —
    # post-pin CVD='0'/'1', yet both ranks enumerated BOTH GPUs and NVML
    # placed both PIDs on both devices).  The early pin needs only the
    # launcher environment: node-local rank from _launcher_local_rank();
    # n_local is unknowable pre-MPI, so pin_local_gpu(n_local=None) uses the
    # rank index itself as the oversubscription witness.  Failures are
    # DEFERRED as strings and raised through the COLLECTIVE gather below —
    # a per-rank raise before MPI_Init can leave healthy peers blocked
    # forever in the rendezvous (the GLM-5.2 lockstep-abort guarantee).
    # Launchers exporting no local-rank var fall through to the post-MPI pin
    # below, guarded by the _cuda_driver_fds_open() escalation.
    early_err: str | None = None
    early_pinned: str | None = None
    _lr_early = _launcher_local_rank()
    if _lr_early is not None:
        try:
            early_pinned = pin_local_gpu(
                local_rank=int(_lr_early[1]), n_local=None)
        except RuntimeError as exc:
            early_err = str(exc)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    hosts = comm.allgather(socket.gethostname())
    # Per-rank GPU binding, BEFORE the single-node early return: nothing
    # downstream assigns rank -> device on that path, so every rank grabbed
    # the default and the job silently ran entirely on GPU 0 (#1516).  A
    # shim-pinned CUDA_VISIBLE_DEVICES (#693) is left untouched; once pinned,
    # ``_pals_local_device_ids()`` resolves to [0] on the multi-node path
    # below exactly as it does for the shims.
    #
    # Node-local (rank, size) from MPI's own shared-memory split, which
    # catches what a hostname comparison misses: ranks on one physical node
    # can report different `socket.gethostname()` strings (short vs FQDN,
    # dual-NIC, split DNS), leaving one of them unpinned and colliding on
    # GPU 0 — the very failure being fixed (GLM-5.2).  Both groupings are
    # then reconciled a few lines below.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    shm_size, shm_rank = node_comm.Get_size(), node_comm.Get_rank()
    node_comm.Free()
    # ...but take whichever of the two groupings is LARGER.  Each fails in one
    # direction and the failures are opposite (GLM-5.2): hostnames UNDER-group
    # when one node's ranks report different strings; the shared-memory split
    # UNDER-groups when the MPI library does not place co-located ranks
    # together (containers without a shared /dev/shm).  Under-grouping means
    # n_local == 1, which makes the pin a silent no-op and every rank lands on
    # device 0 again — the exact defect this fixes.  Over-grouping is at worst
    # a loud error.
    host_size, host_rank = (hosts.count(hosts[rank]),
                            hosts[:rank].count(hosts[rank]))
    n_local, node_rank = ((host_size, host_rank) if host_size > shm_size
                          else (shm_size, shm_rank))
    # The LAUNCHER's node-local rank outranks MPI's ordering when it exports
    # one: under `srun --distribution=cyclic` the two disagree, and SLURM's
    # is the index the rest of the stack (and `_pals_local_device_ids`) binds
    # around (GLM-5.2).
    _lr = _launcher_local_rank()
    local_rank = int(_lr[1]) if _lr is not None else node_rank
    if early_err is not None:
        # The pre-MPI pin already failed for THIS rank; carry that into the
        # collective gather below instead of re-attempting a pin whose
        # narrowing the CUDA driver may no longer honour.
        pinned, device_ids, pin_err = None, None, early_err
    else:
        try:
            # No-op when the early pin already narrowed CVD to one device;
            # otherwise (no launcher local-rank var) this is the post-MPI
            # fallback pin, validated against the reconciled n_local.
            pinned = pin_local_gpu(local_rank=local_rank, n_local=n_local)
            # Resolved HERE, not at the `initialize()` call below, so that
            # its own "more local ranks than visible GPUs" raise also rides
            # the collective gather — otherwise one rank with a stale
            # SLURM_LOCALID dies while the rest block in the rendezvous
            # (codex).
            device_ids, pin_err = _pals_local_device_ids(), None
        except RuntimeError as exc:
            pinned, device_ids, pin_err = None, None, str(exc)
        if (pinned is not None and early_pinned is None
                and not _non_gpu_platform_selected()
                and os.environ.get("LEGOESM_ALLOW_SHARED_GPU") != "1"
                and _cuda_driver_fds_open()):
            # The narrowing was applied only NOW, after MPI_Init (no launcher
            # local-rank variable was available for the early pin), and this
            # process already holds open /dev/nvidia* fds: a CUDA-aware MPI
            # initialised the driver during MPI_Init, the driver snapshotted
            # the PRE-pin CUDA_VISIBLE_DEVICES, and the pin above will be
            # silently ignored — every local rank then computes on GPU 0
            # while the env var reads as correctly pinned (#1516, measured:
            # jobs 26829100 / 26846811).  Refusing loudly here is the
            # documented alternative to silently restoring that defect.
            pin_err = (
                f"CUDA_VISIBLE_DEVICES was narrowed to {pinned!r} only "
                f"AFTER MPI_Init, and this process already holds open "
                f"/dev/nvidia* file descriptors — a CUDA-aware MPI has "
                f"initialised the CUDA driver, which snapshots "
                f"CUDA_VISIBLE_DEVICES at first initialisation, so the pin "
                f"would be silently ignored and the local ranks would share "
                f"one GPU (#1516). Launch with a launcher that exports a "
                f"node-local rank (srun: SLURM_LOCALID; Open MPI mpirun: "
                f"OMPI_COMM_WORLD_LOCAL_RANK), pre-pin CUDA_VISIBLE_DEVICES "
                f"per rank before python starts, or set "
                f"LEGOESM_ALLOW_SHARED_GPU=1 if sharing is intended.")
            pinned = None
    # Duplicate-binding tripwire.  Two silent ways to end up with two ranks on
    # one GPU survive everything above, because from a single rank's
    # environment they are INDISTINGUISHABLE from correct pinning (codex):
    #   (a) `CUDA_VISIBLE_DEVICES=0 mpirun -np 2` — one visible device, which
    #       is exactly what a correct `--gpu-bind=single:1` shim also shows;
    #   (b) a stale SLURM_LOCALID inherited identically by every rank.
    # What separates them is the DEVICE SET the kernel exposes: under gpu-bind
    # each rank can open a different /dev/nvidiaN, so the (device set, device
    # order, chosen ordinal) triple differs per rank.  This is a TRIPWIRE, not
    # a proof of physical identity: a UUID-form CUDA_VISIBLE_DEVICES aliasing
    # an ordinal would slip through, and a deliberately shared GPU (MPS) is
    # rejected — set LEGOESM_ALLOW_SHARED_GPU=1 for that.
    fingerprint = (tuple(openable_nvidia_nodes()),
                   os.environ.get("CUDA_DEVICE_ORDER"),
                   os.environ.get("CUDA_VISIBLE_DEVICES"))
    # Gathered on the WORLD communicator and filtered by hostname, not on the
    # shared-memory sub-communicator: if that split under-groups, a per-rank
    # `node_comm` would make this check trivially find no duplicates and go
    # silent exactly when it is needed (GLM-5.2).
    node_fps = [f for h, f in comm.allgather((hosts[rank], fingerprint))
                if h == hosts[rank]]
    if (pin_err is None and _host_visible_gpus()
            and not _non_gpu_platform_selected()
            and os.environ.get("LEGOESM_ALLOW_SHARED_GPU") != "1"
            and node_fps.count(fingerprint) > 1):
        pin_err = (
            f"{node_fps.count(fingerprint)} of the {n_local} ranks on this "
            f"host resolve to the SAME GPU (openable devices "
            f"{list(fingerprint[0])}, CUDA_VISIBLE_DEVICES="
            f"{fingerprint[2]!r}). That is the silent half-idle-hardware "
            f"failure of #1516, not a pin. Give each rank its own GPU "
            f"(--gpu-bind=single:1, or one rank per device), run single-rank, "
            f"or set LEGOESM_ALLOW_SHARED_GPU=1 if sharing is intended.")
    # Raise in LOCKSTEP.  A per-rank raise on a heterogeneous allocation (one
    # node short of GPUs) kills those ranks while the healthy ones block
    # forever in the coordinator rendezvous below — the job hangs instead of
    # failing (GLM-5.2).  `allgather` is the collective already in use here.
    pin_errs = [(r, e) for r, e in enumerate(comm.allgather(pin_err)) if e]
    if pin_errs:
        raise RuntimeError(
            "GPU pinning failed, aborting every rank together: "
            + "; ".join(f"rank {r}: {e}" for r, e in pin_errs))
    effective_pin = early_pinned if early_pinned is not None else pinned
    if effective_pin is not None:
        # Name both indices: after the mask JAX numbers the pinned device 0,
        # so a reader correlating this line with `jax.devices()[0].id == 0`
        # would otherwise read a successful pin as a failed one (GLM-5.2).
        when = "pre-MPI_Init" if early_pinned is not None else "post-MPI_Init"
        print(f"[early_init] rank {rank} on {hosts[rank]}: pinned to host GPU "
              f"{effective_pin} ({when}; CUDA_VISIBLE_DEVICES="
              f"{effective_pin}; JAX will call it local device 0)",
              flush=True)
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
    # rank onto GPU 0 (codex).  Resolved above so its own launch-error raise
    # is collective; see the pinning block.
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=size,
        process_id=rank,
        local_device_ids=device_ids,
    )
    _INITIALIZED = True
    check_no_silent_process_fallback()
    return True
