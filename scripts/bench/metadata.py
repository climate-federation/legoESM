"""Shared, self-describing metadata for scaling-benchmark outputs (roadmap item 9).

SINGLE source of the metadata block that every scaling JSON carries, so that
each bench driver stops rolling its own record.  The goal is that a scaling row
is comparable and *falsifiable* from the record ALONE — a CPU fallback, a
host-staged halo, a replicated (non-SPMD) run, or an f32 ablation can no longer
masquerade as a valid GPU-direct f64 scaling point.

Every record answers the roadmap's item-9 questions:
  grid, component, resolution, levels; precision + precision knobs; backend,
  rank count, GPU count, devices per rank; decomposition type; MPI / GPU-direct
  settings; solver variant + residual; cells per rank/GPU.

Scientific descriptors (grid, resolution, solver, ...) are passed by the caller;
runtime facts (backend, device / process count, precision knobs, GPU-direct
mode) are auto-detected from the LIVE process so they cannot be mislabeled.

Consumers: ``run_levante_gpu_scaling.py``, ``run_cpu_mpi_scaling.py``,
``bench_atm_latlon_spmd_scaling.py``, ``bench_ocean_latlon_spmd_scaling.py``,
``bench_mpas_spmd_scaling.py``, ``bench_ocean_mpi_scaling.py``,
``bench_ocean_gpu_scaling.py`` (and any future bench driver) merge
``scaling_metadata(...)`` under the ``"metadata"`` key of their JSON payload.
Aggregators read ``payload["metadata"]``.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

#: Bump when the record shape changes so aggregators can branch on it.
#: v2: + ``transport`` / ``virtual_cpu_devices`` / ``launcher`` (anti-fake-
#: scaling audit) — a route-A mpi4jax row, a route-B NCCL row, a gloo/TCP
#: fabric row, and a CPU-virtual-device infra proxy are now distinguishable
#: from the record alone.
METADATA_SCHEMA_VERSION = 2

#: Halo/collective transports a scaling row may run on.  ``xla-local`` =
#: single-process multi-device SPMD (intra-process XLA collectives);
#: ``none`` = serial.  gloo/TCP multi-node is KNOWN anti-scaling fabric
#: (docs/performance/scaling/spectral_level_shard_cliff.md) — recording it
#: verbatim is what keeps such a row from masquerading as an NCCL result.
TRANSPORTS: tuple[str, ...] = ("mpi4jax", "nccl", "gloo", "xla-local", "none")

#: Env knobs that change a run's numerics / comparability.  Recorded VERBATIM so
#: an f32 (or TF32) ablation is never silently compared to an f64 baseline.
PRECISION_ENV_KNOBS: tuple[str, ...] = (
    "JAX_ENABLE_X64",
    "LEGOESM_VMIX_F32_SOLVE",
    "LEGOESM_BAROCLINIC_F32",
    "LEGOESM_ENABLE_TF32",
)

#: The env toggle requesting CUDA-aware (device-direct) mpi4jax halos.
GPU_DIRECT_ENV = "MPI4JAX_USE_CUDA_MPI"

#: Backends on which the GPU-direct toggle is meaningful.
_GPU_BACKENDS = ("gpu", "cuda", "rocm")

#: Keys a self-describing scaling record MUST carry with a NON-EMPTY value
#: (fail-fast hygiene).  ``0`` / ``False`` / a populated ``precision_knobs``
#: dict are legal values; only ``None`` / ``""`` fail.  These are the fields
#: without which a row cannot be compared or a fake-scaling run detected.
REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "grid",
    "component",
    "resolution",
    "n_levels",
    "precision",
    "precision_knobs",
    "backend",
    "decomposition",
    "n_ranks",
    "n_gpus",
    "device_count",
    "process_count",
    "gpu_direct_active",
    "host_staged_halo",
    "transport",
    "virtual_cpu_devices",
    "launcher",
)

#: Keys that MUST be PRESENT (the roadmap requires the field) but whose value
#: may legitimately be ``None`` — an atmosphere run has no barotropic-solver
#: residual; a single-device run has no partition metrics or cells-per-rank.
PRESENT_KEYS: tuple[str, ...] = (
    "devices_per_rank",
    "cells_per_rank",
    "solver_variant",
    "solver_residual",
    "scaling_kind",
)


def _env_flag_true(name: str) -> bool:
    """True iff env var ``name`` is a truthy flag ("1"/"true"/"yes"/"on")."""
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _is_empty(v: Any) -> bool:
    """True for a non-informative required value: ``None``, ``""``, or an EMPTY
    container (e.g. ``precision_knobs={}`` — which would hide an f32/TF32
    ablation).  Scalars ``0`` / ``0.0`` / ``False`` are NOT empty (a legitimate
    ``n_gpus=0`` / ``gpu_direct_active=False`` must pass)."""
    if v is None or v == "":
        return True
    if isinstance(v, (dict, list, tuple, set, frozenset)) and len(v) == 0:
        return True
    return False


def git_sha(short: bool = True) -> str:
    """Best-effort git commit SHA of THIS source tree (repro signature).

    Every scaling row must be attributable to an exact code state — a
    measurement without a SHA cannot be reproduced or compared across
    branches (audit item 7: no harness recorded one).  A dirty working
    tree is marked ``<sha>-dirty`` so the SHA never over-claims
    reproducibility.  Fail-open (``"unknown"``): a missing git binary or
    a non-repo run directory must never kill a benchmark.
    """
    import subprocess

    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=5, cwd=cwd)
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, cwd=cwd)
        if dirty.returncode == 0 and dirty.stdout.strip():
            return sha + "-dirty"
        return sha
    except Exception:
        return "unknown"


def detect_backend() -> str:
    """Live JAX backend ("cpu"/"gpu"/"tpu"), or "unknown" if JAX is absent."""
    try:
        import jax

        return str(jax.default_backend())
    except Exception:
        return "unknown"


def _jax_count(fn_name: str, default: int) -> int:
    try:
        import jax

        return int(getattr(jax, fn_name)())
    except Exception:
        return default


def mpi4jax_cuda_support() -> bool | None:
    """``True``/``False`` as reported by mpi4jax; ``None`` when mpi4jax (or the
    ``has_cuda_support`` predicate) is absent — i.e. CUDA support UNPROVEN."""
    try:
        import mpi4jax

        return bool(mpi4jax.has_cuda_support())
    except Exception:
        return None


def precision_knobs() -> dict[str, str]:
    """Snapshot of every precision-affecting env knob (default ``"0"`` = off)."""
    return {k: os.environ.get(k, "0") for k in PRECISION_ENV_KNOBS}


def detect_virtual_cpu_devices(backend: str | None = None) -> bool:
    """True when the run's parallelism comes from FORCED virtual CPU devices.

    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (N>1) on a CPU
    backend is the infra-validation mode: it characterizes communication
    overhead and correctness, NEVER hardware scaling.  A row with this flag
    True must not be reported as a device-count speedup (roadmap: "do not
    report CPU virtual devices as speedup").
    """
    b = (backend or detect_backend()).lower()
    if b != "cpu":
        return False
    m = re.search(
        r"xla_force_host_platform_device_count\s*=\s*(\d+)",
        os.environ.get("XLA_FLAGS", ""),
    )
    return bool(m and int(m.group(1)) > 1)


def detect_launcher() -> str:
    """Job launcher this process runs under, from launcher-specific env.

    ``slurm`` | ``pbs-pals`` (Cray PALS, e.g. Derecho ``mpiexec``) | ``pbs`` |
    ``openmpi`` (bare ``mpirun``) | ``none`` (local shell).  Order matters:
    PALS jobs also carry ``PBS_JOBID``, and SLURM steps may export OMPI vars.
    """
    env = os.environ
    if "SLURM_JOB_ID" in env:
        return "slurm"
    if "PALS_RANKID" in env or "PALS_NODEID" in env:
        return "pbs-pals"
    if "PBS_JOBID" in env:
        return "pbs"
    if "OMPI_COMM_WORLD_SIZE" in env or "PMI_SIZE" in env:
        return "openmpi"
    return "none"


def resolve_transport(
    transport: str | None,
    *,
    n_ranks: int,
    process_count: int,
    device_count: int,
    backend: str | None = None,
) -> str:
    """Resolve (and validate) the halo/collective transport for the record.

    Explicit ``transport`` wins (validated against :data:`TRANSPORTS`).
    Auto-resolution when ``None``:

    - ``n_ranks > process_count``: a route-A MPI world JAX cannot see (each
      rank is a separate single-process JAX) → ``"mpi4jax"``.
    - ``process_count > 1``: route-B multi-controller ``jax.distributed`` →
      ``"nccl"`` on a GPU backend, ``"gloo"`` on CPU (JAX's CPU collective
      transport).
    - ``device_count > 1``: single-process SPMD → ``"xla-local"``.
    - else ``"none"``.
    """
    if transport is not None:
        if transport not in TRANSPORTS:
            raise ValueError(
                f"unknown transport {transport!r}; expected one of "
                f"{TRANSPORTS} (a scaling row's transport must be a known, "
                "comparable fabric)."
            )
        return transport
    if n_ranks > process_count:
        return "mpi4jax"
    if process_count > 1:
        b = (backend or detect_backend()).lower()
        return "nccl" if b in _GPU_BACKENDS else "gloo"
    if device_count > 1:
        return "xla-local"
    return "none"


def gpu_direct_mode(
    backend: str | None = None, transport: str = "mpi4jax"
) -> dict[str, Any]:
    """Device-direct MPI configuration for the record.

    ``gpu_direct_active`` is ``True`` ONLY when a GPU backend is live, the
    ``MPI4JAX_USE_CUDA_MPI`` toggle is set, AND mpi4jax reports CUDA support —
    the exact conjunction that distinguishes a real GPU-direct halo from a
    silently host-staged one.  ``host_staged_halo`` flags the dangerous case: a
    GPU run whose halos are NOT device-direct (a scaling bug hiding as a valid
    row).  On CPU/TPU the toggle is moot, so both are ``False``.

    Both flags describe the **mpi4jax halo path only**: on any other
    ``transport`` (route-B NCCL, gloo, intra-process ``xla-local``, serial)
    there IS no mpi4jax halo to host-stage, so a GPU row must not be flagged
    ``host_staged_halo`` (codex: an NCCL row would otherwise look like a
    broken MPI row).  The raw request/support facts stay recorded verbatim.
    """
    b = (backend or detect_backend()).lower()
    on_gpu = b in _GPU_BACKENDS
    requested = _env_flag_true(GPU_DIRECT_ENV)
    cuda = mpi4jax_cuda_support()
    device_direct = bool(requested and cuda is True)
    mpi4jax_halo = transport == "mpi4jax"
    return {
        "gpu_direct_requested": requested,
        "mpi4jax_cuda_support": cuda,
        "gpu_direct_active": bool(on_gpu and mpi4jax_halo and device_direct),
        "host_staged_halo": bool(on_gpu and mpi4jax_halo and not device_direct),
    }


def scaling_metadata(
    *,
    grid: str,
    component: str,
    resolution: Any,
    n_levels: int,
    precision: str,
    n_ranks: int | None = None,
    n_gpus: int = 0,
    devices_per_rank: int | None = None,
    decomposition: str = "none",
    solver_variant: str = "n/a",
    solver_residual: float | None = None,
    conservation_drift: float | None = None,
    cells_per_rank: int | None = None,
    scaling_kind: str | None = None,
    transport: str | None = None,
    partition_metrics: dict[str, Any] | None = None,
    timestamp_utc: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete self-describing metadata block for one scaling row.

    Parameters
    ----------
    grid, component, resolution, n_levels, precision, decomposition
        Scientific descriptors of the case (caller-supplied).
    n_ranks, n_gpus, devices_per_rank, cells_per_rank
        Parallel layout.  ``devices_per_rank`` defaults to
        ``jax.device_count() // jax.process_count()`` when omitted.
    solver_variant, solver_residual, conservation_drift
        Solver identity + convergence/conservation evidence.  The roadmap
        forbids claiming a solver speedup without recording a residual /
        conservation drift, so these belong in the record.
    scaling_kind
        ``"weak"`` | ``"strong"`` | ``"throughput"`` | ``None`` — keeps
        weak/strong normalization (and single-device throughput-vs-size
        sweeps, which are NOT device-count scaling) from being conflated
        downstream.
    transport
        Halo/collective fabric (see :data:`TRANSPORTS`).  ``None`` →
        auto-resolved by :func:`resolve_transport`; route-A mpi4jax drivers
        that pass ``n_ranks`` explicitly resolve correctly, route-B /
        single-process SPMD auto-detect from the live process.
    partition_metrics
        MPAS / Voronoi partition-quality numbers when available
        (``edge_cut``, ``owned_halo_ratio``, ``cells_per_rank_min/max``,
        ``message_count``).
    extra
        Any other component-specific descriptors.
    """
    backend = detect_backend()
    device_count = _jax_count("device_count", -1)
    process_count = _jax_count("process_count", 1)
    # Resolve the transport BEFORE gpu_direct_mode: host_staged_halo /
    # gpu_direct_active are mpi4jax-halo semantics and must not fire on a
    # route-B NCCL / xla-local / serial row (codex finding 2).
    # ``n_ranks`` = number of MPI processes.  Default to the auto-detected
    # process count so a single-process SPMD run (1 process, N GPUs) records
    # ``n_ranks=1`` (truthful) while the device parallelism lives in
    # ``n_gpus`` / ``device_count``.  Route-A drivers (1 GPU/rank) pass the
    # real rank count explicitly.
    if n_ranks is None:
        n_ranks = process_count
    if devices_per_rank is None and device_count > 0 and process_count > 0:
        devices_per_rank = device_count // process_count
    resolved_transport = resolve_transport(
        transport,
        n_ranks=int(n_ranks),
        process_count=process_count,
        device_count=device_count,
        backend=backend,
    )

    md: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        # --- scientific descriptors ---
        "grid": grid,
        "component": component,
        "resolution": resolution,
        "n_levels": n_levels,
        "precision": precision,
        "decomposition": decomposition,
        "solver_variant": solver_variant,
        "solver_residual": solver_residual,
        "conservation_drift": conservation_drift,
        "scaling_kind": scaling_kind,
        # --- parallel layout ---
        "n_ranks": n_ranks,
        "n_gpus": n_gpus,
        "device_count": device_count,
        "process_count": process_count,
        "devices_per_rank": devices_per_rank,
        "cells_per_rank": cells_per_rank,
        # --- runtime facts (auto-detected) ---
        "backend": backend,
        "precision_knobs": precision_knobs(),
        "transport": resolved_transport,
        "virtual_cpu_devices": detect_virtual_cpu_devices(backend),
        "launcher": detect_launcher(),
        "hostname": os.environ.get("HOSTNAME")
        or os.environ.get("SLURMD_NODENAME", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "git_sha": git_sha(),
    }
    md.update(gpu_direct_mode(backend, transport=resolved_transport))
    if partition_metrics:
        md["partition_metrics"] = partition_metrics
    if extra:
        md["extra"] = extra
    return md


def validate_scaling_metadata(
    md: dict[str, Any], *, strict: bool = True
) -> list[str]:
    """Return the list of comparability problems (empty = self-describing).

    A REQUIRED key that is ``None``/``""`` and a PRESENT key that is absent
    both count.  Raises ``ValueError`` when ``strict`` and any problem exists —
    benchmark hygiene fail-fast, so a record that cannot be compared is never
    silently written.  (``decomposition == "none"`` and ``solver_residual is
    None`` are legal and do NOT count; ``0``/``False`` are legal values.)
    """
    missing = [k for k in REQUIRED_KEYS if _is_empty(md.get(k))]
    absent = [f"{k}(absent)" for k in PRESENT_KEYS if k not in md]
    problems = missing + absent
    if strict and problems:
        raise ValueError(
            f"scaling metadata not self-describing: {problems}; a record that "
            "cannot be compared must not be written (roadmap benchmark hygiene)."
        )
    return problems


def annotate_incomplete(md: dict[str, Any], *, warn: bool = True) -> dict[str, Any]:
    """Flag (do NOT discard) an incomplete record at write time.

    A benchmark record is built AFTER an expensive run has already completed,
    so a hard raise here would throw away real data.  Instead: validate
    non-strictly, and if the record is not self-describing, embed the problem
    list under ``md["_incomplete"]`` and emit a ``RuntimeWarning`` so the row
    is LOUDLY flagged and a downstream aggregator can skip/annotate it.
    Returns ``md`` (mutated) for chaining.  Use this at JSON-write time; use
    :func:`validate_scaling_metadata` (strict) where aborting is acceptable.
    """
    problems = validate_scaling_metadata(md, strict=False)
    if problems:
        md["_incomplete"] = problems
        if warn:
            import warnings

            warnings.warn(
                f"scaling record not fully self-describing: {problems}",
                RuntimeWarning,
                stacklevel=2,
            )
    return md


def tidy_throughput_fields(
    *,
    dt_seconds: float,
    time_per_step_ms: float,
    total_cells: int,
) -> dict[str, Any]:
    """Flat SYPD/throughput metrics for a bench record, aggregator-ready.

    The SPMD bench lanes (``bench_atm_latlon_spmd_scaling``,
    ``bench_mpas_spmd_scaling``, ``bench_ocean_latlon_spmd_scaling``,
    ``bench_cube_tiled_step_scaling``) historically recorded only
    ``steady_median_ms`` + ``cells`` — no top-level ``sypd`` — so
    ``aggregate_bcw_scaling.py`` dropped their rows and the CPU-vs-GPU
    plots showed empty SYPD panels for those lanes.  Merge this dict into
    the record (``rec.update(...)``) alongside the identity fields
    (``grid_type``, ``resolution``, ``n_levels``, ``precision``,
    ``physics_level``, ``mode``, ``backend``) the aggregator keys on.

    Formulas are the CANONICAL ones from ``run_cpu_mpi_scaling.py`` (365.25
    sim-days/yr) so SPMD rows and MPI rows are directly comparable:

        sypd         = (dt / t_step) / (365.25 * 86400) * 86400
        mcells_per_s = total_cells / t_step / 1e6
    """
    t_step = float(time_per_step_ms) * 1e-3
    if t_step > 0.0:
        sypd = (float(dt_seconds) / t_step) / (365.25 * 86400.0) * 86400.0
        mcells_per_s = float(total_cells) / t_step / 1e6
    else:  # degenerate timing (clock resolution) — flag, never divide by 0
        sypd = 0.0
        mcells_per_s = 0.0
    return {
        "dt_seconds": float(dt_seconds),
        "time_per_step_ms": float(time_per_step_ms),
        "total_cells": int(total_cells),
        "sypd": sypd,
        "mcells_per_s": mcells_per_s,
    }


#: Placeholder comm-fabric numbers for :func:`calibrated_bound` when the
#: caller passes no measured values.  Ballpark single-node GPU-interconnect
#: figures (order NVLink/PCIe), NOT measurements of THIS machine —
#: MACHINE-CALIBRATED-REQUIRED: any bound built on them is emitted with
#: ``bound_calibrated=False`` and must never be quoted as a hardware
#: roofline.  Calibrate with a ping-pong / allreduce microbenchmark on the
#: actual fabric and pass ``latency_us`` / ``bandwidth_GBs`` explicitly.
DEFAULT_COMM_LATENCY_US = 25.0
DEFAULT_COMM_BANDWIDTH_GBS = 10.0


def comm_accounting(
    *,
    halo_messages_per_step: int | None,
    n_lon: int | None = None,
    nlev: int = 1,
    dtype_bytes: int = 8,
    rows_per_message: int = 1,
    bytes_per_message: int | None = None,
    full_state_gathers_per_step: int = 0,
    scope_note: str | None = None,
) -> dict[str, Any]:
    """Per-step communication VOLUME fields for a scaling record (audit item 4).

    ``halo_bytes_per_step = messages x bytes_per_message`` where the
    per-message payload is either given explicitly (``bytes_per_message``)
    or sized as a lat-row slab ``n_lon * nlev * dtype_bytes *
    rows_per_message`` (the lat-band halo exchanges whole latitude rows;
    2-D barotropic slabs pass ``nlev=1``, 3-D baroclinic slabs the real
    level count).  ``halo_messages_per_step=None`` means the census is
    genuinely unknown for this configuration: every derived byte field is
    emitted ``None`` — never fabricated.

    ``full_state_gathers_per_step`` exists so drivers that DO gather the
    full state every step (e.g. run_omip-style host loops: 2 gathers) can
    report it; a fused-scan bench reports 0.  ``scope_note`` should say
    what the census covers (e.g. "barotropic solver only") so a partial
    count is never mistaken for total traffic.
    """
    out: dict[str, Any] = {
        "full_state_gathers_per_step": int(full_state_gathers_per_step),
    }
    if halo_messages_per_step is None:
        out.update(
            halo_messages_per_step=None,
            halo_bytes_per_message=None,
            halo_bytes_per_step=None,
        )
    else:
        if bytes_per_message is None:
            if n_lon is None:
                raise ValueError(
                    "comm_accounting: pass bytes_per_message OR the slab "
                    "dimensions (n_lon [+ nlev/dtype_bytes/rows_per_message]) "
                    "— a message count without a payload size cannot yield "
                    "bytes."
                )
            bytes_per_message = (
                int(n_lon) * int(nlev) * int(dtype_bytes)
                * int(rows_per_message)
            )
        out.update(
            halo_messages_per_step=int(halo_messages_per_step),
            halo_bytes_per_message=int(bytes_per_message),
            halo_bytes_per_step=(
                int(halo_messages_per_step) * int(bytes_per_message)
            ),
        )
    if scope_note:
        out["comm_scope_note"] = scope_note
    return out


def wet_cell_metrics(
    *,
    wet_columns: float,
    nlev: int,
    total_cells: int,
    n_devices: int,
    wet_columns_per_device: list[float] | None = None,
) -> dict[str, Any]:
    """Wet-cell weak-scaling metric (audit item 9): ACTIVE cell-levels.

    Weak-scaling rows historically report TOTAL cells; on a masked ocean
    the work is proportional to WET cell-levels, so per-device totals can
    look balanced while wet work is not.  z-star column-mask semantics
    (the lat-lon C-grid backend: ``land_mask`` is 2-D, a wet column is wet
    at all ``nlev`` levels — the same convention as
    ``bench_ocean_mpi_scaling``'s ``wet_band_boundaries`` accounting):
    ``wet_cell_levels = wet_columns * nlev``.

    ``wet_columns_per_device`` (optional, one entry per device in band
    order) adds min/max per-device wet cell-levels — the imbalance signal.
    ``wet_equals_total`` flags the metric as NON-INFORMATIVE (all-wet IC,
    e.g. a flat-bottom benchmark state); callers should print a loud note.
    """
    if int(nlev) <= 0:
        raise ValueError(f"wet_cell_metrics: nlev must be >= 1, got {nlev}")
    if int(n_devices) <= 0:
        raise ValueError(
            f"wet_cell_metrics: n_devices must be >= 1, got {n_devices}")
    wet_cols = int(round(float(wet_columns)))
    wcl = wet_cols * int(nlev)
    total = int(total_cells)
    out: dict[str, Any] = {
        "wet_cell_levels": wcl,
        "wet_cell_levels_per_device": wcl / int(n_devices),
        "wet_fraction": (wcl / total) if total > 0 else None,
        "wet_equals_total": bool(wcl == total),
    }
    if wet_columns_per_device is not None:
        per = [int(round(float(w))) * int(nlev)
               for w in wet_columns_per_device]
        if len(per) != int(n_devices):
            raise ValueError(
                "wet_cell_metrics: wet_columns_per_device has "
                f"{len(per)} entries for n_devices={n_devices}")
        out["wet_cell_levels_per_device_min"] = min(per)
        out["wet_cell_levels_per_device_max"] = max(per)
    return out


def calibrated_bound(
    *,
    measured_fused_step_ms: float | None = None,
    single_device_fused_step_ms: float | None = None,
    halo_messages_per_step: int | None = None,
    halo_bytes_per_step: int | None = None,
    n_reductions_per_step: int | None = None,
    rank_imbalance: float | None = None,
    latency_us: float | None = None,
    bandwidth_GBs: float | None = None,
    launch_host_ms: float = 0.0,
) -> dict[str, Any]:
    """Calibrated per-fused-step time bound (audit item 8).

        T_bound = max(compute, comm) + reduction + launch_host + imbalance

    Ingredients (all MEASURABLE, none fabricated):

    - ``compute``   = single-device ``fused_step_ms`` at the SAME
      per-device size (the caller passes its nd=1 row; ``None`` -> the
      bound is emitted null and flagged incomplete).
    - ``comm``      = ``messages x latency + bytes / bandwidth`` — halo
      traffic, modeled as overlappable with compute, hence the ``max``.
      A census that omits traffic (e.g. barotropic-only) UNDERestimates
      comm; the bound stays a valid LOWER bound on the step time.
    - ``reduction`` = ``n_reductions x latency`` — sequentially DEPENDENT
      allreduce-type collectives (CG dot products); latency-bound at
      bench scales, so bytes are neglected (small-message model).
    - ``imbalance`` = ``(rank_imbalance - 1) x compute`` from the
      MEASURED max/median block ratio.
    - ``launch_host`` — per-step dispatch overhead; ~0 inside a fused
      ``lax.scan`` block (amortized), so benches pass the default 0.0;
      drivers stepping one-at-a-time should pass their measured
      ``step_latency_ms - fused_step_ms``.

    ``latency_us`` / ``bandwidth_GBs`` default to the
    MACHINE-CALIBRATED-REQUIRED placeholders
    (:data:`DEFAULT_COMM_LATENCY_US` / :data:`DEFAULT_COMM_BANDWIDTH_GBS`);
    whenever either default is used the result carries
    ``bound_calibrated=False`` and must not be quoted as a machine
    roofline.  Any missing ingredient -> ``t_bound_ms=None`` +
    ``bound_incomplete_reason`` naming it — an incomplete bound is
    reported as incomplete, never invented.
    """
    calibrated = latency_us is not None and bandwidth_GBs is not None
    lat_us = (DEFAULT_COMM_LATENCY_US if latency_us is None
              else float(latency_us))
    bw_gbs = (DEFAULT_COMM_BANDWIDTH_GBS if bandwidth_GBs is None
              else float(bandwidth_GBs))
    if lat_us < 0.0:
        raise ValueError(f"calibrated_bound: latency_us must be >= 0, "
                         f"got {lat_us}")
    if bw_gbs <= 0.0:
        raise ValueError(f"calibrated_bound: bandwidth_GBs must be > 0, "
                         f"got {bw_gbs}")

    missing = [name for name, v in (
        ("single_device_fused_step_ms", single_device_fused_step_ms),
        ("halo_messages_per_step", halo_messages_per_step),
        ("halo_bytes_per_step", halo_bytes_per_step),
        ("n_reductions_per_step", n_reductions_per_step),
        ("rank_imbalance", rank_imbalance),
    ) if v is None]

    compute_ms = (None if single_device_fused_step_ms is None
                  else float(single_device_fused_step_ms))
    comm_ms = None
    if halo_messages_per_step is not None and halo_bytes_per_step is not None:
        comm_ms = (float(halo_messages_per_step) * lat_us * 1e-3
                   + float(halo_bytes_per_step) / (bw_gbs * 1e9) * 1e3)
    reduction_ms = (None if n_reductions_per_step is None
                    else float(n_reductions_per_step) * lat_us * 1e-3)
    imbalance_ms = None
    if rank_imbalance is not None and compute_ms is not None:
        imbalance_ms = max(float(rank_imbalance) - 1.0, 0.0) * compute_ms

    if missing:
        t_bound_ms = None
        measured_over_bound = None
    else:
        t_bound_ms = (max(compute_ms, comm_ms) + reduction_ms
                      + float(launch_host_ms) + imbalance_ms)
        measured_over_bound = (
            float(measured_fused_step_ms) / t_bound_ms
            if measured_fused_step_ms is not None and t_bound_ms > 0.0
            else None)

    return {
        "t_bound_ms": (None if t_bound_ms is None else round(t_bound_ms, 4)),
        "measured_over_bound": (None if measured_over_bound is None
                                else round(measured_over_bound, 4)),
        "bound_calibrated": bool(calibrated),
        "bound_incomplete_reason": (missing or None),
        "bound_ingredients": {
            "compute_ms": compute_ms,
            "comm_ms": (None if comm_ms is None else round(comm_ms, 6)),
            "reduction_ms": (None if reduction_ms is None
                             else round(reduction_ms, 6)),
            "imbalance_ms": (None if imbalance_ms is None
                             else round(imbalance_ms, 6)),
            "launch_host_ms": float(launch_host_ms),
            "latency_us": lat_us,
            "bandwidth_GBs": bw_gbs,
            "halo_messages_per_step": halo_messages_per_step,
            "halo_bytes_per_step": halo_bytes_per_step,
            "n_reductions_per_step": n_reductions_per_step,
            "rank_imbalance": rank_imbalance,
        },
    }


def timed_scan_blocks(
    advance,
    state,
    *,
    block_steps: int,
    n_blocks: int = 2,
    probe_steps: int = 3,
    sync_label: str = "timed_scan_blocks",
):
    """Measurement-contract timing: fused ``lax.scan`` blocks + probe latency.

    The trustworthy production-like number is a MULTI-STEP ``lax.scan`` block
    with device synchronization only AROUND the block (per-step host sync in a
    Python loop measures dispatch+sync latency, not fused device throughput —
    the audited anti-pattern in the SPMD benches).  Per-step dispatch latency
    is still physically meaningful (drivers that must step one-at-a-time pay
    it), so it is measured SEPARATELY by a short individually-synced probe and
    reported as ``step_latency_ms`` — never mixed into the fused number.

    Multi-controller runs additionally record the SLOWEST-process block time
    and the imbalance ratio (max/median across processes): with a single
    process's clock a straggler band is invisible and the reported time
    understates the true parallel step time.

    Parameters
    ----------
    advance
        ``advance(state) -> state`` — ONE production step with all static
        knobs (dt, forcing, ...) closed over.  May itself be jitted; it is
        re-traced INTO the fused scan (same graph, no double-jit penalty).
    state
        Initial (already sharded, post-seed) model state pytree.
    block_steps
        Steps per fused ``lax.scan`` block (the amortizing window).
    n_blocks
        Timed blocks; per-block times expose block-to-block drift.
    probe_steps
        Individually host-synced steps for the separate dispatch-latency
        probe (small: each one costs a full device round-trip).
    sync_label
        Base label for the multi-controller ``sync_global_devices`` fences.

    Returns
    -------
    (state, metrics) — final state (compile + probe + all blocks advanced)
    and a dict:
      ``compile_ms``           first-call cost of ``advance`` (trace+compile)
      ``scan_compile_ms``      first-call cost of the fused scan itself
      ``step_latency_ms``      median individually-synced per-step wall time
      ``block_ms``             per-block wall times, THIS process (list)
      ``fused_step_ms``        median(block_ms)/block_steps — the headline
      ``block_ms_max``/``block_ms_median``  slowest/median across processes
                               (equal to this process's for 1 process)
      ``rank_imbalance``       block_ms_max / block_ms_median  (>= 1.0)
      ``block_steps``/``n_blocks``/``probe_steps``  the schedule itself
    """
    import time

    import jax
    import numpy as np

    def _block(tree):
        jax.block_until_ready(jax.tree_util.tree_leaves(tree))

    multi = jax.process_count() > 1

    def _fence(tag: str):
        if multi:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(f"{sync_label}_{tag}")

    # --- 1. compile (first call of advance, separated from all timing) ---
    _fence("compile_start")
    t0 = time.perf_counter()
    state = advance(state)
    _block(state)
    compile_ms = (time.perf_counter() - t0) * 1e3

    # --- 2. dispatch-latency probe: individually synced steps, reported
    # separately (NEVER mixed into the fused number) ---
    probe_ms = []
    for _ in range(max(0, probe_steps)):
        t0 = time.perf_counter()
        state = advance(state)
        _block(state)
        probe_ms.append((time.perf_counter() - t0) * 1e3)
    step_latency_ms = float(np.median(probe_ms)) if probe_ms else float("nan")

    # --- 3. fused scan block (dtype-stable carry, the OM pattern) ---
    input_dtypes = jax.tree_util.tree_map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state)

    @jax.jit
    def _scan_run(st):
        def _body(carry, _):
            new = advance(carry)
            new = jax.tree_util.tree_map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes)
            return new, None
        return jax.lax.scan(_body, st, None, length=block_steps)[0]

    # Pre-compile the scan on cloned leaves (seed state untouched; every
    # process executes the same collective schedule — counts stay matched).
    _pre = jax.tree_util.tree_map(lambda x: x, state)
    t0 = time.perf_counter()
    _pre_out = _scan_run(_pre)
    _block(_pre_out)
    scan_compile_ms = (time.perf_counter() - t0) * 1e3
    del _pre, _pre_out

    block_ms = []
    for b in range(max(1, n_blocks)):
        _fence(f"block{b}_start")
        t0 = time.perf_counter()
        state = _scan_run(state)
        _block(state)
        block_ms.append((time.perf_counter() - t0) * 1e3)
    _fence("blocks_end")

    my_median = float(np.median(block_ms))
    if multi:
        from jax.experimental import multihost_utils
        all_medians = np.asarray(
            multihost_utils.process_allgather(np.float64(my_median)))
        block_ms_max = float(np.max(all_medians))
        block_ms_median = float(np.median(all_medians))
    else:
        block_ms_max = my_median
        block_ms_median = my_median
    rank_imbalance = (block_ms_max / block_ms_median
                      if block_ms_median > 0 else float("nan"))

    metrics = {
        "compile_ms": round(compile_ms, 1),
        "scan_compile_ms": round(scan_compile_ms, 1),
        "step_latency_ms": round(step_latency_ms, 3),
        "block_ms": [round(b, 2) for b in block_ms],
        "fused_step_ms": round(block_ms_max / max(1, block_steps), 4),
        "block_ms_max": round(block_ms_max, 2),
        "block_ms_median": round(block_ms_median, 2),
        "rank_imbalance": round(rank_imbalance, 4),
        "block_steps": int(block_steps),
        "n_blocks": int(n_blocks),
        "probe_steps": int(probe_steps),
    }
    return state, metrics
