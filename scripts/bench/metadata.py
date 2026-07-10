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
