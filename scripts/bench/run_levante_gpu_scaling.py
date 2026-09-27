#!/usr/bin/env python
"""GPU scaling benchmark for Levante (DKRZ) -- replicates CliMA Figures 11 & 12.

Runs the Jablonowski-Williamson baroclinic wave test case, measuring
wall-clock time per step and SYPD across varying GPU counts and resolutions.

Supported grids:
  spectral      -- Gaussian grid + spectral PE dycore (default)
  cubed-sphere  -- Cubed-sphere C-D grid + FV3 PE dycore
  icosahedral   -- MPAS Voronoi mesh + TRiSK PE dycore
  latlon        -- Lat-lon finite-volume grid + FV PE dycore

Multi-rank MPI scaling support (iter-13/14 honest-sweep guards):
  icosahedral   -- mesh graph-partition decomposition (any rank count)
  latlon        -- latitude-band decomposition, pole_bc='wall' +
                   diagnostic physics (make_latlon_mpi_step, #641 Grid 3)
  cubed-sphere  -- genuine <=6-face decomposition via --cs-mpi-scatter;
                   default (no flag) is replicated dynamics, refused
  spectral      -- single-rank only (no MPI path)

Two modes:
  weak   -- fix problem size per GPU, increase resolution with GPU count
            (replicates Yatunin et al. 2026 JAMES Figure 11 left panel)
  strong -- fix total problem size, increase GPU count
            (replicates Yatunin et al. 2026 JAMES Figure 12 left panel)

Usage
-----
Single-node SPMD (1-6 face-sharded devices on a single process,
``shard_map`` backend, no MPI)::

    python scripts/bench/run_levante_gpu_scaling.py --mode weak --precision float32
    python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere \\
        --mode strong --precision both

Multi-rank via MPI -- icosahedral / latlon (validated domain decomposition;
cubed-sphere needs --cs-mpi-scatter; see the support matrix above)::

    mpirun -np 4 python scripts/bench/run_levante_gpu_scaling.py \\
        --grid icosahedral --mode strong --n-gpus 4

The script auto-detects available GPUs when --n-gpus is not set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Shared, self-describing scaling metadata (roadmap item 9): merged per-result
# into the JSON so a host-staged / f32 / replicated GPU row is falsifiable from
# the record.  ``metadata.py`` imports JAX only lazily (safe before JAX init).
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from metadata import annotate_incomplete, scaling_metadata  # noqa: E402

# NOTE: do NOT import ``legoesm.constants`` at module load — it eagerly
# imports ``jax.numpy``, which initialises JAX before ``_configure_jax`` /
# ``_configure_mpi_gpu_affinity`` have a chance to set ``JAX_ENABLE_X64``,
# ``JAX_PLATFORMS``, ``CUDA_VISIBLE_DEVICES``, and ``XLA_FLAGS``.  Lazy
# imports inside the functions that consume ``constants.X`` keep the
# JAX-startup invariant intact.

# ---------------------------------------------------------------------------
# JAX configuration -- must happen before jax import
# ---------------------------------------------------------------------------

def _configure_mpi_gpu_affinity() -> None:
    """Pin one GPU per MPI rank via ``CUDA_VISIBLE_DEVICES``.

    Must be called **before** any JAX import so that JAX only sees the
    assigned GPU.  Uses MPI-launcher env vars (``OMPI_COMM_WORLD_LOCAL_RANK``
    for OpenMPI, ``MV2_COMM_WORLD_LOCAL_RANK`` for MVAPICH2, or
    ``SLURM_LOCALID`` for SLURM) to determine which GPU this rank should use.
    """
    local_rank = (
        os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
    )
    if local_rank is None:
        # SLURM_LOCALID + SLURM_NTASKS are exported even in the batch step
        # (where a single Python process runs and all GPUs must be visible).
        # Pinning on SLURM_LOCALID=0 there hid all but GPU 0 from
        # single-process multi-GPU runs (Ginsburg 8454397/8454737, 2026-06-10;
        # Levante 25842907, 2026-06-23).
        # Only honor SLURM_LOCALID when inside an actual srun step:
        # SLURM_STEP_NODELIST is set by srun for every task, but is absent
        # (or set to the batch magic) in the top-level batch step.
        slurm_localid = os.environ.get("SLURM_LOCALID")
        slurm_step_nodelist = os.environ.get("SLURM_STEP_NODELIST", "")
        if slurm_localid is not None and slurm_step_nodelist:
            local_rank = slurm_localid
    # Only set CUDA_VISIBLE_DEVICES when SLURM hasn't already restricted it.
    # With --gpus-per-node + srun, SLURM binds one GPU per task automatically
    # (CUDA_VISIBLE_DEVICES="0" for each task).  Overwriting with local_rank
    # (0,1,2,3 ...) then conflicts with SLURM's assignment and produces
    # "CUDA_ERROR_INVALID_DEVICE" on every task beyond the first
    # (Levante 25842908, 2026-06-23).
    if local_rank is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank


def _configure_jax(precision: str) -> None:
    """Set JAX env vars before import.

    Iter 21: stop *forcing* ``JAX_PLATFORMS=gpu,cpu`` as the default —
    JAX 0.10+ uses backend names ``cuda`` / ``rocm`` / ``cpu`` and
    rejects the generic ``gpu`` token, raising
    "Backend 'rocm' is not in the list of known backends" before any
    benchmark code runs.  Leave the variable unset by default and let
    JAX pick its default backend; respect any value the user / SLURM
    wrapper has already set.
    """
    # float64 and mixed both need x64 (mixed = fp32 storage/compute + fp64
    # accumulate; the fp64 accumulator requires x64).  The ocean precision
    # POLICY (storage dtype) is set per-arm by the ocean bench, not here.
    if precision in ("float64", "mixed"):
        os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    # Enable XLA GPU scheduling optimizations for multi-device scaling.
    # Apply when ``JAX_PLATFORMS`` is unset (auto-detect) or names a
    # GPU vendor — ``gpu`` (legacy alias), ``cuda`` (JAX 0.10 NVIDIA),
    # or ``rocm`` (AMD).  Skip when the user explicitly set ``cpu`` or
    # ``tpu`` to avoid pinging XLA flags that the chosen backend
    # rejects.
    platforms = os.environ.get("JAX_PLATFORMS", "")
    is_gpu_run = (
        not platforms
        or any(tok in platforms for tok in ("gpu", "cuda", "rocm"))
    )
    if is_gpu_run:
        xla_flags = os.environ.get("XLA_FLAGS", "")
        for flag in [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
        ]:
            if flag not in xla_flags:
                xla_flags = f"{xla_flags} {flag}" if xla_flags else flag
        os.environ["XLA_FLAGS"] = xla_flags


def _maybe_init_distributed(
    global_n: int | None = None,
    grid_type: str = "cubed-sphere",
) -> tuple[int, int]:
    """Detect and initialize distributed JAX if under MPI or SLURM.

    Returns (rank, world_size). For single-process, returns (0, 1).

    For cubed-sphere grids, uses ``initialize_distributed()`` from
    ``legoesm.parallel.distributed`` which sets the MPI halo backend,
    builds the CommTopology, creates the DeviceConfig, and (when
    ``global_n`` is provided) builds the :class:`DistributedLayout`.

    For lat-lon and icosahedral grids, performs lightweight MPI init
    (rank/size detection and multi-node JAX coordination) without
    cubed-sphere-specific topology.  Grid-specific MPI layout, scatter,
    and step functions are set up later in :func:`run_benchmark`.

    Parameters
    ----------
    global_n : int, optional
        Per-face grid resolution (cubed-sphere only).
    grid_type : str
        Grid type: ``"cubed-sphere"``, ``"latlon"``, or ``"icosahedral"``.
    """
    # Check for MPI environment
    if "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ:
        # Guard ONLY the import: mpi4py installed-but-unloadable (no libmpi;
        # ImportError, or the loader's RuntimeError) falls through to the
        # SLURM / serial detection below. Everything past a successful
        # import — including jax.distributed.initialize — must fail LOUDLY:
        # swallowing a real multi-node init failure would leave N ranks
        # running as independent serial programs (codex finding).
        MPI = None
        try:
            from mpi4py import MPI
        except (ImportError, RuntimeError):
            pass
        if MPI is not None:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            n_procs = comm.Get_size()

            if grid_type == "cubed-sphere":
                # Full cubed-sphere distributed init (topology, halo backend,
                # DeviceConfig).
                from legoesm.parallel.distributed import initialize_distributed
                initialize_distributed(global_n=global_n)
            elif n_procs > 1:
                # Lat-lon / icosahedral: lightweight MPI init.
                # Handle multi-node JAX coordination if needed.
                import socket
                hostnames = comm.allgather(socket.gethostname())
                if len(set(hostnames)) > 1:
                    import jax
                    from legoesm.parallel.early_init import (
                        resolve_coordinator_port,
                    )
                    jax.distributed.initialize(
                        coordinator_address=(
                            f"{hostnames[0]}:{resolve_coordinator_port()}"
                        ),
                        num_processes=n_procs,
                        process_id=rank,
                    )

            return rank, n_procs

    # Check for SLURM-launched multi-process jobs.  A plain sbatch allocation
    # sets SLURM_NTASKS>1 even when this script is executed as a single process
    # with multiple local GPUs (single-node case).  Only enter distributed init
    # when actually running on multiple nodes (SLURM_NNODES>1) under srun, where
    # SLURM_STEP_NODELIST is set and JAX can resolve the coordinator address.
    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    slurm_step_nodelist = os.environ.get("SLURM_STEP_NODELIST", "")
    if (slurm_ntasks and slurm_procid is not None
            and int(slurm_ntasks) > 1
            and slurm_nnodes > 1
            and slurm_step_nodelist):
        import jax
        from legoesm.parallel.early_init import (
            init_jax_distributed_with_fallback,
        )
        init_jax_distributed_with_fallback()
        return jax.process_index(), jax.process_count()

    return 0, 1


# ===========================================================================
# Dataclasses for results
# ===========================================================================

PHYSICS_CHOICES = ("none", "held_suarez", "moist", "gray_sbm", "rrtmg_full")

# Grid/physics support matrix.
#
#   "moist" = moisture (q_v/q_c/q_r) + Kessler warm-rain condensation, NO
#   radiation (the moist baroclinic-wave case).  The column physics is
#   grid-agnostic — it operates on a flattened ``(ncol, nlev)`` view and is
#   wired for EVERY grid via ``make_kessler_forcing_{mpas,cube,latlon,
#   spectral}`` + the moist tracer halo exchange the dycore already performs.
#   This is the tier to use for a *fair* cross-grid many-GPU scaling
#   comparison: the same column closure, embarrassingly parallel, on every
#   grid (it scales like dry plus a fixed per-column cost).  Mirrors the CPU
#   MPI driver ``run_cpu_mpi_scaling.py``.
#
#   "gray_sbm" / "rrtmg_full" add radiation + SBM convection and are routed
#   through the ModelDriver AMIP *segment* path (no MPI step yet), so they
#   stay cubed-sphere / lat-lon only (single-node SPMD).
_SUPPORTED_PHYSICS = {
    "cubed-sphere": {"none", "held_suarez", "moist", "gray_sbm", "rrtmg_full"},
    "latlon": {"none", "held_suarez", "moist", "gray_sbm", "rrtmg_full"},
    "icosahedral": {"none", "held_suarez", "moist"},
    "spectral": {"none", "held_suarez", "moist"},
}

# MPI distributed benchmark support.  Only grids with a validated
# local-domain (true-decomposition) MPI step are listed here — a grid
# absent from this set is refused under ``mpirun`` so replicated-dynamics
# wall-clock noise is never reported as scaling.
#
#   icosahedral -- full mesh graph-partition decomposition, any rank
#       count (``make_voronoi_mpi_step``).
#   latlon -- latitude-band decomposition with pole_bc='wall' +
#       diagnostic physics (``make_latlon_mpi_step``, issue #641 Grid 3).
#       Each rank owns a contiguous lat band; band-cut halos exchange via
#       sendrecv, the mass fixer allreduces the global sphere area.
#       Validated MPI == serial after gather in
#       ``tests/distributed/test_latlon_mpi_step.py`` (np=2/4).  Still
#       band-only: pole_bc='fold'/'tripole' and the PhysicsState carry
#       for prognostic physics (#405/#413) are NOT wired, and there is no
#       single-process multi-GPU (route-B SPMD) latlon step.
#
# Cubed-sphere is NOT in this set: the DEFAULT cube MPI path keeps full
# (6, n, n, ...) state on every rank (replicated dynamics, not scaling).
# The genuine face decomposition is the opt-in ``--cs-mpi-scatter`` path
# (``make_rank_local_cube_model`` slices the entire cdgrid metric set to
# owned faces; validated bit-for-bit incl. AD vs the single-process
# reference in ``tests/distributed/test_cube_face_scatter_mpi.py``,
# np=1/2/3/6).  It is allowed past the MPI guards via the ``_cube_scatter_ok``
# carve-out, capped at 6 faces (1 node); the sub-face TILED SPMD path for
# >6 GPUs (``LEGOESM_TILED_SPMD``) and the mandatory visual W2 v-wind
# cube-imprint regression + multi-node GPU run remain to be validated.
#
# Spectral has no MPI step (the level mesh shards state for memory only and
# falls through to ``model.step`` — single-process replication).
_MPI_SUPPORTED_GRIDS = {"icosahedral", "latlon"}


@dataclass
class TimingResult:
    n_gpus: int
    resolution: int
    n_levels: int
    precision: str
    mode: str
    physics_level: str
    dt_seconds: float
    n_warmup: int
    n_timing: int
    compile_time_s: float
    warmup_time_s: float
    timing_time_s: float
    time_per_step_ms: float
    sypd: float
    total_cells: int
    cells_per_gpu: int
    mcells_per_s: float
    # Grid family this row was measured on (cubed-sphere / latlon / icosahedral
    # / spectral).  Serialized into strong_scaling.json so the tidy aggregator
    # (aggregate_bcw_scaling.py) can tag each GPU row with its grid — without
    # it the nested atm report has no grid and the rows are silently dropped.
    grid_type: str = "cubed-sphere"
    scaling_efficiency: float = 1.0
    # Collective-permute op census of the compiled TIMED executable
    # (comm-minimisation step 1: measurement infrastructure for the
    # upcoming halo-fusion work).  ``-1`` = not measured — single
    # device, MPI, non-cubed-sphere, or the HLO guard was skipped via
    # LEGOESM_SPMD_FORCE_ALLGATHER.  Sync ops lower as
    # ``collective-permute``; async pairs as ``-start``/``-done``.
    hlo_collective_permute: int = -1
    hlo_collective_permute_start: int = -1
    hlo_collective_permute_done: int = -1


@dataclass
class ScalingReport:
    mode: str
    precisions: list[str]
    results: list[TimingResult]
    timestamp_utc: str = ""
    backend: str = ""
    hostname: str = ""

    def __post_init__(self):
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()


GRID_CHOICES = ("spectral", "cubed-sphere", "icosahedral", "latlon")

# ===========================================================================
# Resolution/GPU ladders
# ===========================================================================

# Weak scaling: problem size per GPU is roughly constant.
# Cell count per face = N^2, total = 6*N^2.
# With k GPUs we pick N such that 6*N^2 / k ~ 6 * N_base^2 / 1.
# => N = N_base * sqrt(k).  We round to the nearest even integer.
WEAK_SCALING_BASE_N = 24  # ~24x24 per face on 1 GPU => ~400 km

# Icosahedral weak scaling: subdivision level as base.
# nCells = 10*4^level + 2.  With k GPUs we want roughly k times as many
# cells, so pick the level whose cell count best matches k * base_cells.
WEAK_SCALING_BASE_LEVEL_ICO = 4  # 2562 cells on 1 GPU

def _weak_resolution(n_gpus: int, base_n: int = WEAK_SCALING_BASE_N) -> int:
    """Compute resolution for weak scaling at a given GPU count."""
    n_raw = base_n * math.sqrt(n_gpus)
    n_rounded = max(4, 2 * round(n_raw / 2))  # even number, min 4
    return n_rounded

MAX_SUBDIVISION_LEVEL = 8  # 10*4^8+2 = 655,362 cells — safe upper bound

def _weak_resolution_ico(n_gpus: int, base_level: int = WEAK_SCALING_BASE_LEVEL_ICO) -> int:
    """Compute icosahedral subdivision level for weak scaling.

    Weak scaling holds cells/GPU roughly constant.  With ``n_gpus`` devices
    the target total cell count is ``n_gpus * base_cells``.  We pick the
    subdivision level whose cells-per-GPU ratio is nearest to ``base_cells``
    (in log-space).  Ties are broken in favour of the level with *more*
    cells per GPU so that devices are not underutilised.

    Note: icosahedral levels jump by 4× in cell count, so perfect weak
    scaling at non-power-of-4 GPU counts is impossible.
    """
    base_level = min(base_level, MAX_SUBDIVISION_LEVEL)
    base_cells = 10 * 4 ** base_level + 2

    best_level = base_level
    best_ratio = float("inf")

    for lev in range(base_level, MAX_SUBDIVISION_LEVEL + 1):
        cells = 10 * 4 ** lev + 2
        cells_per_gpu = cells / n_gpus
        # How far from ideal cells/GPU (ratio ≥ 1, lower is better)
        ratio = max(cells_per_gpu / base_cells, base_cells / cells_per_gpu)
        if ratio < best_ratio or (
            ratio == best_ratio and cells_per_gpu >= base_cells
        ):
            best_ratio = ratio
            best_level = lev

    return best_level

# Lat-lon weak scaling: same approach as cubed-sphere.
WEAK_SCALING_BASE_N_LL = 64  # ~64x128 on 1 GPU

def _weak_resolution_ll(
    n_gpus: int,
    base_n: int = WEAK_SCALING_BASE_N_LL,
    n_ranks: int = 1,
) -> int:
    """Compute lat-lon resolution for weak scaling at a given GPU count.

    When ``n_ranks > 1`` (MPI), the result is rounded up to be divisible
    by ``n_ranks`` so that latitude bands divide evenly, and floored so
    every band carries at least the dycore halo (``_LATLON_MPI_HALO``):
    at very large rank counts ``base_n * sqrt(ng) / ng`` shrinks below the
    halo, which would make ``exchange_halo_latlon`` raise mid-sweep.
    """
    n_raw = base_n * math.sqrt(n_gpus)
    n_rounded = max(8, 2 * round(n_raw / 2))  # even number, min 8
    if n_ranks > 1:
        while n_rounded % n_ranks != 0:
            n_rounded += 2
        # Floor at halo rows per band (n_ranks * _LATLON_MPI_HALO is even and
        # divisible by n_ranks, so it preserves both invariants above).
        n_rounded = max(n_rounded, n_ranks * _LATLON_MPI_HALO)
    return n_rounded

# Strong scaling: fixed resolutions, sweep GPU counts.
STRONG_RESOLUTIONS_CS = [48, 96, 192]   # cubed-sphere: ~200, ~100, ~50 km
STRONG_RESOLUTIONS_SP = [42, 85, 170]   # spectral: T42, T85, T170
STRONG_RESOLUTIONS_ICO = [4, 5, 6, 7, 8]  # icosahedral: levels 4-8 (L8 = 655,362 cells, ~25 km; high levels need multi-GPU/multi-node)
STRONG_RESOLUTIONS_LL = [64, 128, 256]  # lat-lon: n_lat
# Lat-lon MPI halo width: the C-grid PPM / biharmonic operators pad
# ``halo=2`` rows (primitive_eq_latlon_cgrid: pad_halo_latlon_3d(halo=2);
# make_latlon_mpi_step default).  exchange_halo_latlon RAISES when a rank's
# band has fewer than ``halo`` rows (``halo > n_lat_local``), so the MPI
# band sweep skips a resolution whose smallest band would drop below it
# instead of crashing before timing.
_LATLON_MPI_HALO = 2

# Minimum total timing duration target (seconds).  When step times are
# sub-millisecond (e.g., I4 at ~0.5 ms/step), 100 steps yield only
# ~50 ms of timed work — within OS scheduling / NCCL jitter.
# Targeting ≥2 s greatly reduces relative noise.
_MIN_TIMING_SECONDS = 2.0


def _auto_n_timing(n_timing_base: int, total_cells: int, n_gpus: int,
                    max_timing: int = 1000) -> int:
    """Scale timing steps up for small grids to ensure stable measurements.

    For sub-millisecond step times (small cells/GPU), the default 100
    timing steps yield only ~50 ms of timed work which is within OS and
    NCCL jitter.  This helper estimates step time from cells/GPU and
    bumps n_timing so the timed window is ≥ _MIN_TIMING_SECONDS.

    Capped at *max_timing* to keep total benchmark runtime practical.
    A ``lax.scan(length=N)`` that times a single step compiles roughly
    proportional to *N*; pushing *max_timing* much past 1000 makes
    compile time dominate wall-clock for small grids without improving
    the precision of the timing measurement.  For sub-50-µs step times
    we further clamp to 200 steps — that's still ~10 ms of timed work,
    well above per-step jitter at that scale.
    """
    cells_per_gpu = total_cells // max(n_gpus, 1)
    # Rough model: step time ~ 0.01 ms per 1000 cells/GPU (from I4–I6 data)
    est_ms = max(cells_per_gpu / 100_000, 0.1)
    est_total_s = est_ms * n_timing_base / 1000.0
    if est_total_s >= _MIN_TIMING_SECONDS:
        return n_timing_base
    needed = int(math.ceil(_MIN_TIMING_SECONDS / (est_ms / 1000.0)))
    if est_ms < 0.05:
        # Sub-50µs steps: 200 scan iterations is enough timed work and
        # avoids paying a long XLA compile for a 2000-iteration scan.
        needed = min(needed, 200)
    needed = max(needed, n_timing_base)
    needed = min(needed, max_timing)
    # Round up to nearest 100 for clean reporting
    needed = ((needed + 99) // 100) * 100
    return needed

# GPU counts to sweep (must satisfy cubed-sphere tiling constraints).
GPU_COUNTS = [1, 2, 3, 6, 24, 54, 96]  # 1-6 divide faces; >6 must be 6*k^2


def _valid_gpu_counts(max_gpus: int, grid_type: str = "cubed-sphere") -> list[int]:
    """Return valid GPU counts up to max_gpus for the given grid type.

    Cubed-sphere requires divisors of 6 (face sharding) or 6*k^2 (tiling).
    Icosahedral supports any GPU count — its single-process multi-GPU path
    routes through ``make_voronoi_sharded_step``, a real domain-decomposed
    ``shard_map`` step.  Lat-lon and spectral are single-GPU only HERE
    (single-process): neither has a validated single-process multi-GPU
    sharded step.  Lat-lon's *multi-rank* scaling instead goes through MPI
    (``make_latlon_mpi_step``), where the count is pinned to world_size by
    ``fixed_gpu_count`` and bypasses this function entirely; single-process
    lat-lon raises NotImplementedError for n_gpus > 1.  Spectral creates a
    level mesh and shards the state but then falls through to plain
    ``model.step``, so a "multi-GPU" spectral point would just re-time the
    single-device program (the 1->2 GPU exactly-0.500-efficiency
    replication failure mode).
    """
    if grid_type in ("latlon", "spectral"):
        # Reason: no validated single-process multi-GPU sharded step.
        return [1]
    if grid_type == "icosahedral":
        return list(range(1, max_gpus + 1))

    # Cubed-sphere constraints: face-only divisors of 6. Sub-face tiled
    # counts (6*k^2 = 24, 54, ...) are EXCLUDED until the tiled SPMD halo
    # path is validated — make_sharded_step only activates the SPMD halo
    # backend for face-only tiling, so tiled runs would execute on an
    # unasserted path (codex review 2026-06-10).
    return [n for n in [1, 2, 3, 6] if n <= max_gpus]


# ===========================================================================
# Sharding tripwire (single-process multi-GPU cubed-sphere)
# ===========================================================================

def _assert_expected_sharding(state, dev_config, *, where: str) -> None:
    """Loud tripwire: face-leading leaves must actually be face-sharded.

    Guards against the 1->2 GPU replication bug (exactly 0.500 scaling
    efficiency at identical wall time): the timed executable silently
    ran with a fully replicated state, so every device computed the
    whole globe.  Verifies, leaf by leaf, that *state* carries the
    sharding the cubed-sphere SPMD policy expects — the SAME policy
    (``create_output_shardings``) used for the ``out_shardings`` of
    ``CompiledShardedStep`` and of the timed scan runner:

    * face-leading ``jax.Array`` leaves must be split across all
      ``dev_config.n_devices`` devices, with per-shard shape equal to
      ``NamedSharding.shard_shape`` (i.e. NOT replicated);
    * scalar/constant leaves (expected spec ``P()``) are allowed to be
      replicated;
    * non-``jax.Array`` leaves are skipped.

    Only meaningful for the cubed-sphere face-sharding policy — callers
    must not invoke it for voronoi/level/lat-lon meshes.  No-op when
    ``dev_config.mesh is None`` or ``n_devices == 1``.

    Raises ``RuntimeError`` on the first mismatch — never warn-only.
    """
    if (
        dev_config is None
        or getattr(dev_config, "mesh", None) is None
        or dev_config.n_devices == 1
    ):
        return

    import jax  # lazy: see top-of-file note on JAX init order
    from jax.sharding import PartitionSpec as _P
    from legoesm.parallel.sharded_dynamics import create_output_shardings

    expected = create_output_shardings(state, dev_config)
    # ``None`` marks non-array leaves in the policy pytree (and optional
    # ``None`` fields in the state map to ``None`` in the policy); flatten
    # BOTH sides with the same none-as-leaf convention so the two
    # flattenings stay aligned one-to-one.
    _none_leaf = lambda x: x is None
    state_leaves = jax.tree_util.tree_flatten_with_path(
        state, is_leaf=_none_leaf,
    )[0]
    expected_leaves = jax.tree_util.tree_leaves(expected, is_leaf=_none_leaf)
    if len(state_leaves) != len(expected_leaves):
        raise RuntimeError(
            f"{where}: sharding tripwire cannot align {len(state_leaves)} "
            f"state leaves with {len(expected_leaves)} expected shardings"
        )

    n_dev = dev_config.n_devices
    for (path, leaf), exp in zip(state_leaves, expected_leaves):
        if exp is None or not isinstance(leaf, jax.Array):
            continue
        name = jax.tree_util.keystr(path)
        actual = leaf.sharding
        try:
            matches = actual.is_equivalent_to(exp, leaf.ndim)
        except (AttributeError, TypeError):
            matches = actual == exp
        if not matches:
            raise RuntimeError(
                f"{where}: leaf {name} sharding {actual} does not match "
                f"expected {exp} — state lost its face sharding (every "
                f"device would compute the full globe)"
            )
        if exp.spec == _P():
            continue  # replicated scalars/constants: allowed
        shards = leaf.addressable_shards
        if len(shards) != n_dev:
            raise RuntimeError(
                f"{where}: leaf {name} has {len(shards)} addressable "
                f"shard(s), expected {n_dev}"
            )
        want = exp.shard_shape(leaf.shape)
        for shard in shards:
            if shard.data.shape != want:
                raise RuntimeError(
                    f"{where}: leaf {name} shard shape {shard.data.shape} "
                    f"!= expected {want} (full shape {leaf.shape}) — leaf "
                    f"is replicated, not face-sharded"
                )


# ===========================================================================
# Timed scan runner — shared between the dry-dycore and moist-segment paths
# ===========================================================================

def _count_collective_permute_ops(hlo_text: str) -> dict[str, int]:
    """Census of collective-permute ops in a compiled HLO module.

    Counts opcode *applications* (``<opcode>(``) so each op is counted
    once regardless of how many times its result name appears.  Sync
    halo exchanges lower to ``collective-permute``; the async form
    lowers to ``collective-permute-start`` / ``collective-permute-done``
    pairs.  Comm-minimisation sequencing step 1: this census is the
    before/after metric for the upcoming halo-fusion work.
    """
    import re
    return {
        "collective-permute": len(
            re.findall(r"\bcollective-permute\(", hlo_text)),
        "collective-permute-start": len(
            re.findall(r"\bcollective-permute-start\(", hlo_text)),
        "collective-permute-done": len(
            re.findall(r"\bcollective-permute-done\(", hlo_text)),
    }


def _hlo_census_fields(hlo_counts: dict[str, int] | None) -> dict[str, int]:
    """``TimingResult`` kwargs for the collective-permute census.

    Empty dict (→ the ``-1`` "not measured" defaults) when the HLO
    guard did not run.
    """
    if hlo_counts is None:
        return {}
    return {
        "hlo_collective_permute": hlo_counts["collective-permute"],
        "hlo_collective_permute_start": hlo_counts["collective-permute-start"],
        "hlo_collective_permute_done": hlo_counts["collective-permute-done"],
    }


def _build_timed_scan_runner(
    *,
    step_fn,
    state,
    dev_config,
    grid_type: str,
    n_timing: int,
    dt_static: float,
    n_grid: int,
    n_levels: int,
    n_gpus: int,
    precision: str,
    is_cs_distributed: bool,
    is_mpi: bool,
):
    """Build the timed ``lax.scan`` executable with the full sharded-path
    guard stack — ONE implementation for the dry-dycore and the
    moist/segment benchmark branches (codex BLOCKER: the segment branch
    bypassed every guard below and could record replicated-compute
    timings as multi-GPU scaling rows).

    Call AFTER warmup, with *state* being the post-warmup seed of the
    timed scan.  Applies, in order:

    1. the ``_scan_shardings`` gate — explicit in/out shardings for the
       OUTER timed jit, built once from the same shared leaf policy
       (``create_output_shardings``) as the inner compiled step.  Only
       for single-process multi-device cubed-sphere runs; ``None``
       (plain ``jax.jit``) everywhere else — zero behavior change for
       single-GPU / MPI / non-cubed-sphere rows;
    2. sharding tripwire #1 on the post-warmup seed state;
    3. the compiled-HLO hot-path guard: zero full-cube all-gathers in
       the timed executable (LEGOESM_SPMD_FORCE_ALLGATHER=1 skips with
       a loud warning), plus the collective-permute op census (printed
       and returned for the result row metadata).  The
       ``lower().compile()`` result is reused as the timed runner, so
       the guard adds no extra compilation;
    4. precompile against leaf-cloned state (so the timed run still
       starts from the post-warmup state, and queued XLA work cannot
       overlap the timed region — block on the precompile OUTPUT);
    5. sharding tripwire #2 on the precompile output (the actual timed
       executable's result).

    ``dt_static`` is captured in the scan-body closure as a Python
    float: several dycore step methods do Python ``==`` / ``if dt > 0``
    checks against a cached dt (spectral tracer filter, SI matrix
    cache, sponge factors), which require a concrete value, and a
    fixed-dt benchmark wants dt folded into compiled constants anyway.

    Returns ``(scan_runner, hlo_collective_counts)`` where the counts
    dict is ``None`` whenever the sharded gate is off or the HLO guard
    was skipped.
    """
    import jax

    # ------------------------------------------------------------------
    # State-sharding pytree for the *timed* executable.  The inner
    # compiled step constrains its own in/out shardings, but the
    # program actually measured is the OUTER scan-runner jit — without
    # explicit shardings on it XLA may run the whole scan replicated
    # (the 1->2 GPU "exactly 0.500 efficiency, identical wall time"
    # bug).  Built ONCE from the same policy as the inner step's
    # out_shardings; also gates the `_assert_expected_sharding`
    # tripwire calls below.
    # ------------------------------------------------------------------
    _scan_shardings = None
    if (
        grid_type == "cubed-sphere"
        and dev_config is not None
        and dev_config.mesh is not None
        and dev_config.n_devices > 1
        and not is_cs_distributed
        and not is_mpi
    ):
        from legoesm.parallel.sharded_dynamics import create_output_shardings
        _scan_shardings = create_output_shardings(state, dev_config)

    # Tripwire #1: after compile + warmup the state that seeds the timed
    # scan must still be face-sharded.  Loud RuntimeError, never a
    # warning — a replicated state here means every device computes the
    # full globe and the timing is meaningless.
    if _scan_shardings is not None:
        _assert_expected_sharding(state, dev_config, where="after warmup")

    # Build dtype-safe scan runner (prevents float32→float64 promotion
    # from breaking scan's type-matching requirement).
    input_dtypes = jax.tree.map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state)

    def _run(st):
        def _body(carry, _):
            new = step_fn(carry, dt_static)
            new = jax.tree.map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes,
            )
            return new, None
        return jax.lax.scan(_body, st, None, length=n_timing)[0]

    # Constrain the MEASURED executable too: explicit in/out shardings
    # on the outer scan jit (same pytree as the inner step's
    # out_shardings).  Without them the timed program is free to run
    # replicated even when the inner step is sharded.
    if _scan_shardings is not None:
        scan_runner = jax.jit(
            _run,
            in_shardings=(_scan_shardings,),
            out_shardings=_scan_shardings,
        )
    else:
        scan_runner = jax.jit(_run)

    # ------------------------------------------------------------------
    # Once-per-config compiled-HLO hot-path guard (codex BLOCKER 3): the
    # TIMED program must contain zero all-gather ops with full-cube face
    # extent.  The job-8456476 probe proved such gathers mean every
    # device computes the whole globe (per-device FLOPs ratio 1.00)
    # while the sharding tripwires above still pass — sharded state,
    # replicated compute.  Gated to single-process multi-device
    # cubed-sphere runs (the `_scan_shardings` gate); MPI and other
    # grids are untouched.  The lower().compile() result is reused as
    # the timed runner, so the guard adds no extra compilation.
    # ------------------------------------------------------------------
    hlo_counts = None
    if _scan_shardings is not None:
        if os.environ.get("LEGOESM_SPMD_FORCE_ALLGATHER", "") == "1":
            print(
                "    WARNING: LEGOESM_SPMD_FORCE_ALLGATHER=1 — the "
                "all_gather DIAGNOSTIC halo backend is active; skipping "
                "the full-cube all-gather HLO guard.  Timing rows from "
                "this run measure the compute-replicating backend and "
                "must not be quoted as scaling numbers.",
                flush=True,
            )
        else:
            from legoesm.parallel.cubesphere_exchange import (
                assert_no_fullcube_allgather,
            )
            _compiled_runner = scan_runner.lower(state).compile()
            _hlo_text = _compiled_runner.as_text()
            assert_no_fullcube_allgather(
                _hlo_text, n=n_grid,
                context=(
                    f"timed scan runner (C{n_grid}/L{n_levels}, "
                    f"n_gpus={n_gpus}, {precision})"
                ),
            )
            print(
                "    HLO guard: hot path clean — no full-cube all-gather "
                "ops in the timed executable",
                flush=True,
            )
            hlo_counts = _count_collective_permute_ops(_hlo_text)
            print(
                f"    HLO census: {hlo_counts['collective-permute']} "
                f"collective-permute, "
                f"{hlo_counts['collective-permute-start']} -start, "
                f"{hlo_counts['collective-permute-done']} -done op(s) "
                f"in the timed executable",
                flush=True,
            )
            scan_runner = _compiled_runner

    # Pre-compile the scan runner without mutating the timed state.
    # Re-binding ``state`` to the precompile output would start the
    # *timing* run from state advanced by ``n_timing`` extra steps —
    # biasing finite-time comparisons.  Clone the leaves so XLA still
    # compiles and warms caches against identical input
    # shapes/dtypes/sharding, but the original state remains the seed
    # for the timed scan.  IMPORTANT: block on the *output* leaves,
    # not the input — blocking the input does not wait for the queued
    # kernel to finish, so XLA work could overlap with the timed
    # region and bias measurements.
    _precompile_state = jax.tree.map(lambda x: x, state)
    _precompile_out = scan_runner(_precompile_state)
    jax.block_until_ready(jax.tree.leaves(_precompile_out))

    # Tripwire #2: the compiled scan runner's OUTPUT must be face-sharded
    # — this inspects the actual timed executable's result, catching the
    # "compiled loop feeds replicated output back" failure mode before
    # any timing is recorded.
    if _scan_shardings is not None:
        _assert_expected_sharding(
            _precompile_out, dev_config, where="after scan precompile",
        )

    return scan_runner, hlo_counts


# ===========================================================================
# CFL-safe timestep
# ===========================================================================

def _auto_dt(n_grid: int, grid_type: str = "cubed-sphere") -> float:
    """Choose a CFL-safe timestep for the hydrostatic PE at resolution n_grid.

    For explicit ssp_rk3 integration the CFL constraint must account for
    both the advective speed (~60 m/s) and the external gravity wave
    speed (~300 m/s):  dt < cfl * dx_min / (u_max + c_grav).
    """
    from legoesm import constants  # lazy: see top-of-file note on JAX init order

    R = constants.R_earth
    if grid_type == "spectral":
        # Gaussian grid: dx_min ~ pi * R / n_lon at equator, n_lon = 2*(n_max+1)
        n_lon = 2 * (n_grid + 1)
        dx_min = math.pi * R / n_lon
    elif grid_type == "icosahedral":
        # Voronoi SCVT: n_grid is subdivision level, nCells = 10*4^level + 2
        n_cells = 10 * 4 ** n_grid + 2
        dx_avg = R * math.sqrt(4.0 * math.pi / n_cells)
        dx_min = 0.9 * dx_avg
    elif grid_type == "latlon":
        # Lat-lon grid: dx_min ~ pi * R / n_lon at equator, n_lon = 2 * n_lat
        n_lon = 2 * n_grid
        dx_min = math.pi * R / n_lon
    else:
        # Cubed sphere: dx_min ~ (pi/2) * R / (n * sqrt(3))
        dx_min = (math.pi / 2) * R / (n_grid * math.sqrt(3))
    u_max = 60.0
    c_grav = 300.0  # external gravity wave speed [m/s]
    cfl = 0.7
    dt = cfl * dx_min / (u_max + c_grav)
    # Round down to a nice number
    dt = max(30.0, 30.0 * int(dt / 30.0))
    return dt


# ===========================================================================
# Hyperdiffusion scaling
# ===========================================================================

def hyperdiff_coeff(n_grid: int, grid_type: str = "cubed-sphere") -> float:
    """Scale \\nabla^4 hyperdiffusion coefficient with resolution."""
    if grid_type == "spectral":
        ref_n = 42
        ref_coeff = 2.5e16
    elif grid_type == "icosahedral":
        # For icosahedral, n_grid is a subdivision level.  Scale the
        # coefficient with dx^4 relative to level 5 (~120 km).
        from legoesm import constants  # lazy: see top-of-file note on JAX init order

        R = constants.R_earth
        ref_cells = 10 * 4 ** 5 + 2
        cur_cells = 10 * 4 ** n_grid + 2
        dx_ref = R * math.sqrt(4.0 * math.pi / ref_cells)
        dx_cur = R * math.sqrt(4.0 * math.pi / cur_cells)
        ref_coeff = 5e16
        return ref_coeff * (dx_cur / dx_ref) ** 4
    elif grid_type == "latlon":
        ref_n = 64
        ref_coeff = 5e16
    else:
        ref_n = 48
        ref_coeff = 5e16
    return ref_coeff * (ref_n / n_grid) ** 4


# ===========================================================================
# Physics helpers
# ===========================================================================

def _validate_physics(grid_type: str, physics_level: str) -> None:
    """Raise if the grid/physics combination is not supported."""
    supported = _SUPPORTED_PHYSICS.get(grid_type, set())
    if physics_level not in supported:
        raise ValueError(
            f"Physics level {physics_level!r} is not supported for "
            f"grid {grid_type!r}. Supported: {sorted(supported)}. "
            f"Radiative tiers (gray_sbm, rrtmg_full) run only on the "
            f"cubed-sphere / lat-lon AMIP segment path; the grid-agnostic "
            f"'moist' (Kessler) tier runs on every grid."
        )


def _build_physics_fn(physics_level: str, grid_type: str):
    """Build a Held-Suarez physics function for the given grid type.

    Returns None for 'none' and moist tiers (those use segment path).
    """
    if physics_level != "held_suarez":
        return None

    if grid_type == "spectral":
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_forcing_spectral,
        )
        return held_suarez_forcing_spectral
    elif grid_type == "latlon":
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_forcing_latlon,
        )
        return held_suarez_forcing_latlon
    elif grid_type == "icosahedral":
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_forcing_mpas,
        )
        return held_suarez_forcing_mpas
    else:  # cubed-sphere
        from legoesm.atmosphere.forcing.idealized.held_suarez import (
            held_suarez_forcing,
        )
        return held_suarez_forcing


def _build_moist_physics_fn(grid_type: str, dt: float):
    """Build the grid-specific Kessler warm-rain forcing for the 'moist' tier.

    Kessler is column-local: it reads ``q_v/q_c/q_r`` and ``T`` per column and
    returns condensation/evaporation tendencies, so it is the same closure on
    every grid (only the array layout the factory unflattens differs).  The
    timestep is bound into the forcing closure because the operator-split
    physics_fn convention itself passes no ``dt`` (mirrors
    ``run_cpu_mpi_scaling.py``).

    Raises ValueError on an unknown grid (dispatch hardening — a silent
    ``None`` would benchmark dycore-only under a 'moist' label).
    """
    from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
        make_kessler_forcing_cube,
        make_kessler_forcing_latlon,
        make_kessler_forcing_mpas,
        make_kessler_forcing_spectral,
    )

    if grid_type == "spectral":
        return make_kessler_forcing_spectral(dt)
    elif grid_type == "latlon":
        return make_kessler_forcing_latlon(dt)
    elif grid_type == "icosahedral":
        return make_kessler_forcing_mpas(dt)
    elif grid_type == "cubed-sphere":
        return make_kessler_forcing_cube(dt)
    raise ValueError(
        f"_build_moist_physics_fn: unknown grid_type {grid_type!r} "
        f"(expected one of {sorted(GRID_CHOICES)})"
    )


# ===========================================================================
# Segment-based benchmark (gray_sbm / rrtmg_full)
# ===========================================================================

def _grid_config_name(grid_type: str) -> str:
    """Map CLI grid name to ExperimentConfig grid_type."""
    return {"cubed-sphere": "cubed_sphere", "latlon": "latlon"}.get(grid_type, grid_type)


def _disc_name(grid_type: str) -> str:
    """Map CLI grid name to ExperimentConfig discretization."""
    return {
        "cubed-sphere": "cdgrid",
        "latlon": "finite_volume",
    }.get(grid_type, grid_type)


def _build_segment_benchmark(
    *,
    physics_level: str,
    grid_type: str,
    n_grid: int,
    n_levels: int,
    n_gpus: int,
    precision: str,
    dt: float | None = None,
):
    """Build a segment-based benchmark for moist physics tiers.

    Uses ModelDriver to construct the full AMIP pipeline (dynamics +
    radiation + convection + microphysics) with analytical forcing.
    Returns (step_fn, state_carry, dt, total_cells, cells_per_gpu,
    dev_config) where step_fn wraps run_segment(carry, 1, forcing) and
    dev_config is the driver's resolved DeviceConfig (consumed by the
    shared timed-scan-runner guard stack).
    """
    import jax
    import jax.numpy as jnp
    from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig, OutputConfig
    from legoesm.driver.model_driver import ModelDriver

    if dt is None:
        dt = _auto_dt(n_grid, grid_type)

    rad_scheme = "gray" if physics_level == "gray_sbm" else "rrtmgp"
    conv_scheme = "sbm"
    micro_scheme = "kessler" if physics_level == "rrtmg_full" else "none"

    config = ExperimentConfig(
        grid=GridConfig(
            grid_type=_grid_config_name(grid_type),
            resolution=n_grid,
            nlev=n_levels,
        ),
        dycore=DycoreConfig(
            model_type="hydrostatic",
            discretization=_disc_name(grid_type),
            dt=dt,
            fix_mass=True,
        ),
        output=OutputConfig(diag_days=999, checkpoint_days=0),
        dataset="analytical",
        radiation=rad_scheme,
        convection=conv_scheme,
        microphysics=micro_scheme,
        topography="flat",
        days=1,
        precision="fp64" if precision == "float64" else "fp32",
        fix_moisture=False,
        n_devices=n_gpus,
        # Forward-only benchmark: parallelise the RRTMGP g-point axis (the
        # sequential checkpointed scan is ~6x slower on GPU; block of 32
        # recovers most parallelism while bounding memory).  No effect on the
        # gray/none tiers.  Use 0 here only if benchmarking the training-AD path.
        rrtmgp_gpoint_batch_size=32,
    )

    driver = ModelDriver(config)
    driver.setup()

    # Tripwire (mirrors the dry path's create_device_mesh check): the
    # row below is labeled with n_gpus, so the resolved device config
    # MUST actually have that many devices.  The runtime silently
    # clamps when fewer devices are visible (e.g. CUDA_VISIBLE_DEVICES
    # mishap) — without this check the segment branch could record
    # "2-GPU" rows from de-facto 1-device (replicated) programs.
    # Skipped under MPI, where n_devices is the per-rank count.
    dev_config = driver._device_config
    if (
        dev_config is not None
        and not dev_config.is_distributed
        and dev_config.n_devices != n_gpus
    ):
        raise RuntimeError(
            f"requested n_gpus={n_gpus} but the driver's device config "
            f"resolved to {dev_config.n_devices} device(s) — "
            f"refusing to record a mislabeled scaling row "
            f"(visible devices: {len(jax.devices())})"
        )

    # Prepare run context (builds physics pipeline, solar, ozone, etc.)
    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)

    # Build segment function
    from legoesm.driver.compiled_segments import (
        build_segment_fn, pack_carry, pack_forcing, compute_segment_length,
    )

    sigma_full = ctx["sigma_full"]
    dsigma = ctx["dsigma"]
    step_unified = ctx["step_unified"]
    RAD_UPDATE_STEPS = ctx["RAD_UPDATE_STEPS"]

    segment_length = compute_segment_length(
        ctx["diag_interval"], ctx["checkpoint_interval"], RAD_UPDATE_STEPS,
    )

    run_segment_obj = build_segment_fn(
        model=driver.model,
        step_unified=step_unified,
        grid=driver.grid,
        sigma_full=sigma_full,
        dsigma=dsigma,
        dt=dt,
        rad_update_steps=RAD_UPDATE_STEPS,
        microphysics=config.microphysics,
        fix_moisture=config.fix_moisture,
        fix_mass=config.dycore.fix_mass,
        fric_decay=driver._fric_decay,
        qv_smooth_coeff=driver._qv_smooth_coeff,
        lat=driver._grid_lat,
        lon=driver._grid_lon,
        start_day=config.start_day,
        gradient_checkpoint=False,
        hyperdiffusion_3d_fn=driver._hyperdiffusion_3d_fn,
        tau_equator=config.tau_equator,
        tau_pole=config.tau_pole,
        sbm_tau_c=config.sbm_tau_c,
        sbm_RH_ref=config.sbm_RH_ref,
        C_H=config.C_H,
        C_E=config.C_E,
        albedo_ice=config.albedo_ice,
        albedo_ocean=config.albedo_ocean,
        ghg_vmr_override=ctx.get("ghg_vmr"),
        owned_face_ids=None,
        # Single-process multi-GPU: pin carry/forcing in/out shardings
        # on the segment JITs (third replication site).  The internal
        # mesh gate makes this a no-op for 1-GPU rows (mesh is None);
        # under MPI the per-rank config must NOT carry single-process
        # shardings, so pass None there.
        device_config=(
            dev_config
            if dev_config is not None and not dev_config.is_distributed
            else None
        ),
    )
    # Use the non-donating variant for benchmarking (safe with scan).
    # With a live mesh this is the non-donating SHARDED jit wrapper.
    run_segment = run_segment_obj.raw

    # Pack initial carry.  Iter 10: route the held_* arrays through
    # keyword args (``pack_carry`` makes them keyword-only after the
    # ``*,`` marker — passing them positionally collided with the
    # ``conv_prog`` slot and would raise) and seed ``target_mass`` so
    # the segment-level mass fixer has a non-zero anchor.  We compute
    # the global mass via the existing budget-aware
    # ``global_area_sum`` helper (fp64 accumulator when JAX has x64
    # enabled — see ``conservation_accumulator``).
    shape_2d = ctx["shape_2d"]
    shape_3d = ctx["shape_3d"]
    _sd = ctx["_sd"]

    from legoesm.core.conservation import global_area_sum
    _target_mass = global_area_sum(driver.state.p_s.data, driver.grid)

    carry = pack_carry(
        driver.state, driver.q_v, driver.q_c, driver.q_r,
        conv_prog=ctx.get("conv_prog"),
        held_dT_rad=ctx["held_dT_rad"],
        held_sw_net_sfc=ctx["held_sw_net_sfc"],
        held_lw_net_sfc=ctx["held_lw_net_sfc"],
        held_sw_up_toa=ctx["held_sw_up_toa"],
        held_lw_up_toa=ctx["held_lw_up_toa"],
        held_sw_down_toa=ctx["held_sw_down_toa"],
        step_index=0,
        target_mass=_target_mass,
    )

    # Pack forcing (constant during benchmark)
    sst, sic = driver.get_sst_sic(config.start_day)
    day_of_year = jnp.asarray(config.start_day % 365.25)
    seconds_of_day = jnp.asarray(0.0)

    forcing = pack_forcing(
        sst=sst, sic=sic,
        day_of_year=day_of_year, seconds_of_day=seconds_of_day,
        solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
        o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
    )

    # Single-process multi-GPU: shard the carry AND forcing across the
    # mesh BEFORE the segment, exactly as the production driver does
    # (``ModelDriver`` runs ``shard_pytree`` on the carry right before
    # ``run_segment``).  ``driver.state`` is already face-sharded (from
    # ``shard_state`` in setup), but the carry's auxiliary fields
    # (``held_*`` radiation, ``conv_prog``, accumulators, ``T_land``) and
    # the forcing (``o3_vmr``, ``sst``/``sic``) are built UNSHARDED here
    # — and the segment now MATCHES the input layout rather than
    # re-deriving one, so without this the face-leading aux/forcing
    # leaves would stay replicated (every device computing the full
    # globe for them) and the ``_assert_expected_sharding`` tripwire
    # below correctly fails.  Same gate as ``device_config`` above
    # (multi-device, live mesh, single-process — NOT MPI).
    if (
        dev_config is not None
        and not dev_config.is_distributed
        and dev_config.mesh is not None
        and dev_config.n_devices > 1
    ):
        from legoesm.parallel.mesh import shard_pytree
        carry = shard_pytree(carry, dev_config)
        forcing = shard_pytree(forcing, dev_config)

    # Total cells
    if grid_type == "cubed-sphere":
        total_cells = 6 * n_grid * n_grid * n_levels
    else:  # latlon
        total_cells = n_grid * (2 * n_grid) * n_levels

    cells_per_gpu = total_cells // max(1, n_gpus)

    # Wrap as (carry, dt) -> carry for the timing loop
    _forcing = forcing
    _run_seg = run_segment

    def step_fn(c, _dt):
        return _run_seg(c, 1, _forcing)

    return step_fn, carry, dt, total_cells, cells_per_gpu, dev_config


def _run_segment_benchmark(
    *,
    physics_level: str,
    grid_type: str,
    n_grid: int,
    n_levels: int,
    n_gpus: int,
    precision: str,
    mode: str,
    n_warmup: int,
    n_timing: int,
    dt: float | None = None,
) -> TimingResult:
    """Run a segment-based benchmark for moist physics tiers."""
    import jax
    import jax.numpy as jnp

    if precision == "float64":
        jax.config.update("jax_enable_x64", True)

    # RRTMG optics are preloaded inside _build_segment_benchmark() via
    # ModelDriver.setup() → _create_physics() → RRTMGP.preload().

    (step_fn, carry, dt_used, total_cells, cells_per_gpu,
     dev_config) = _build_segment_benchmark(
        physics_level=physics_level,
        grid_type=grid_type,
        n_grid=n_grid,
        n_levels=n_levels,
        n_gpus=n_gpus,
        precision=precision,
        dt=dt,
    )

    if grid_type == "latlon":
        res_label = f"LL{n_grid}"
    else:
        res_label = f"C{n_grid}"

    print(
        f"  [{precision}] {res_label}/L{n_levels} on {n_gpus} GPU(s) | "
        f"dt={dt_used:.0f}s | cells={total_cells:,} | cells/GPU={cells_per_gpu:,}"
        f" | physics={physics_level} (segment)",
        flush=True,
    )

    # JIT compilation
    t_compile_start = time.perf_counter()
    carry = step_fn(carry, dt_used)
    jax.block_until_ready(jax.tree.leaves(carry))
    compile_time = time.perf_counter() - t_compile_start
    print(f"    JIT compile: {compile_time:.2f}s", flush=True)

    # Warmup
    t_warmup_start = time.perf_counter()
    for _ in range(n_warmup):
        carry = step_fn(carry, dt_used)
    jax.block_until_ready(jax.tree.leaves(carry))
    warmup_time = time.perf_counter() - t_warmup_start

    # MPI detection (mirrors the dry path): the shared runner's sharded
    # guard stack must stay off for MPI ranks.
    _rank, _n_ranks = 0, 1
    try:
        from mpi4py import MPI as _MPI
        _rank, _n_ranks = _MPI.COMM_WORLD.Get_rank(), _MPI.COMM_WORLD.Get_size()
    except (ImportError, RuntimeError):
        # RuntimeError: mpi4py installed but no loadable libmpi (GPU-only
        # venv) — a single-process benchmark must not require MPI.
        pass
    _is_mpi = _n_ranks > 1

    # Timed steps via lax.scan — through the SAME shared runner +
    # sharding gate + tripwires + HLO guard as the dry-dycore path
    # (codex BLOCKER: this branch used to bypass all of them).  The
    # scan carries a SegmentCarry; ``create_output_shardings`` is the
    # shared leaf policy (face-leading leaves shard, scalars/diag
    # accumulators replicate).
    scan_runner, _hlo_counts = _build_timed_scan_runner(
        step_fn=step_fn,
        state=carry,
        dev_config=dev_config,
        grid_type=grid_type,
        n_timing=n_timing,
        dt_static=float(dt_used),
        n_grid=n_grid,
        n_levels=n_levels,
        n_gpus=n_gpus,
        precision=precision,
        is_cs_distributed=bool(
            dev_config is not None and dev_config.is_distributed
        ),
        is_mpi=_is_mpi,
    )

    # MPI barrier before timing
    try:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.Get_size() > 1:
            jax.block_until_ready(jax.tree.leaves(carry))
            _MPI.COMM_WORLD.Barrier()
    except (ImportError, RuntimeError):
        # broken/absent mpi4py must not break a single-process run
        pass

    t0 = time.perf_counter()
    carry = scan_runner(carry)
    jax.block_until_ready(jax.tree.leaves(carry))

    try:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.Get_size() > 1:
            _MPI.COMM_WORLD.Barrier()
    except (ImportError, RuntimeError):
        # broken/absent mpi4py must not break a single-process run
        pass

    t1 = time.perf_counter()

    timing_time = t1 - t0
    time_per_step = timing_time / n_timing
    time_per_step_ms = time_per_step * 1000.0
    sypd = (dt_used / time_per_step) / (365.25 * 86400) * 86400.0
    mcells_per_s = (total_cells / time_per_step) / 1e6

    print(
        f"    Timing: {time_per_step_ms:.2f} ms/step | "
        f"SYPD={sypd:.3f} | {mcells_per_s:.1f} Mcells/s",
        flush=True,
    )

    return TimingResult(
        n_gpus=n_gpus,
        resolution=n_grid,
        n_levels=n_levels,
        precision=precision,
        mode=mode,
        physics_level=physics_level,
        dt_seconds=dt_used,
        n_warmup=n_warmup,
        n_timing=n_timing,
        compile_time_s=compile_time,
        warmup_time_s=warmup_time,
        timing_time_s=timing_time,
        time_per_step_ms=time_per_step_ms,
        sypd=sypd,
        total_cells=total_cells,
        cells_per_gpu=cells_per_gpu,
        mcells_per_s=mcells_per_s,
        grid_type=grid_type,
        **_hlo_census_fields(_hlo_counts),
    )


# ===========================================================================
# Core benchmark runner
# ===========================================================================

def run_benchmark(
    *,
    n_grid: int,
    n_levels: int,
    n_gpus: int,
    precision: str,
    mode: str,
    n_warmup: int,
    n_timing: int,
    dt: float | None = None,
    grid_type: str = "spectral",
    no_conservation: bool = False,
    physics_level: str = "none",
    cs_mpi_scatter: bool = False,
) -> TimingResult:
    """Run the baroclinic wave benchmark and return timing results.

    When ``physics_level`` is ``"held_suarez"``, the appropriate
    Held-Suarez forcing function is passed to the model step so that
    physics tendencies are included in each RK stage.
    """
    _validate_physics(grid_type, physics_level)

    # Moist physics tiers use the ModelDriver segment path instead of
    # the bare dycore step.  This branch handles gray_sbm and rrtmg_full.
    if physics_level in ("gray_sbm", "rrtmg_full"):
        return _run_segment_benchmark(
            physics_level=physics_level,
            grid_type=grid_type,
            n_grid=n_grid,
            n_levels=n_levels,
            n_gpus=n_gpus,
            precision=precision,
            mode=mode,
            n_warmup=n_warmup,
            n_timing=n_timing,
            dt=dt,
        )

    # Grid-agnostic moist (Kessler warm-rain) tier: initial condition carries
    # q_v/q_c/q_r and the per-grid Kessler forcing is threaded into the step
    # below (single-device, SPMD, and Voronoi-MPI all supported).
    _moist = physics_level == "moist"

    import jax
    import jax.numpy as jnp

    # Spectral guard: the Gaussian/spectral pathway requires x64 (the
    # spherical-harmonic transforms operate on complex128).  A
    # float32 sweep step can land here after a float64 step has already
    # enabled x64, and JAX cannot disable x64 once it has been turned on
    # — so we silently get float64 internally regardless.  Surface the
    # mismatch immediately so benchmark CSVs do not record bogus
    # "float32 spectral" entries that are really running in float64.
    if grid_type == "spectral" and precision != "float64":
        import warnings
        warnings.warn(
            "Spectral dycore requires float64; coercing precision to float64 "
            "for this run.  Use --precision float64 to silence this warning.",
            stacklevel=2,
        )
        precision = "float64"

    # Set precision
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if precision == "float64" else jnp.float32

    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.mesh import (
        create_device_mesh, create_latlon_mesh, create_level_mesh,
        create_voronoi_device_mesh, shard_pytree,
    )

    # Choose timestep
    if dt is None:
        dt = _auto_dt(n_grid, grid_type)

    sigma = create_sigma_coordinate(n_levels)

    # Detect MPI for lat-lon and icosahedral grids (cubed-sphere uses
    # dev_config.is_distributed instead, set by initialize_distributed).
    _rank, _n_ranks = 0, 1
    try:
        from mpi4py import MPI as _MPI
        _rank, _n_ranks = _MPI.COMM_WORLD.Get_rank(), _MPI.COMM_WORLD.Get_size()
    except (ImportError, RuntimeError):
        # RuntimeError: mpi4py installed but no loadable libmpi (GPU-only
        # venv) — a single-process benchmark must not require MPI.
        pass
    _is_mpi = _n_ranks > 1

    # Grid-specific MPI layouts (populated in grid branches below).
    _voronoi_layout = None
    _latlon_layout = None

    if grid_type == "spectral":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPrimitiveEquationModel,
            SpectralPEConfig,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral

        grid = create_gaussian_grid(n_grid)
        hd = hyperdiff_coeff(n_grid, grid_type)
        config = SpectralPEConfig(
            hyperdiff_coeff=hd,
            hyperdiff_order=4,
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, config)
        state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True, moist=_moist)

        n_lat = grid.n_lat
        n_lon = grid.n_lon
        total_cells = n_lat * n_lon * n_levels
        dev_config = create_level_mesh(n_devices=n_gpus)
    elif grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
            MPASPrimitiveEquationConfig,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

        grid = create_voronoi_mesh(subdivision_level=n_grid)

        total_cells = grid.nCells * n_levels
        hd = hyperdiff_coeff(n_grid, grid_type)
        config = MPASPrimitiveEquationConfig(
            nu_del4=hd,
            nu_del4_ps=hd,
            fix_mass=True,
            pv_scheme="energy",
            time_integrator="ssp_rk3",
        )

        if _is_mpi:
            # MPI distributed: partition mesh across ranks.  State is
            # initialized from the global mesh and scattered to rank-local.
            # make_voronoi_mpi_step uses layout.local_mesh internally.
            from legoesm.parallel.voronoi_mpi import (
                make_voronoi_partition_layout,
            )
            _voronoi_layout = make_voronoi_partition_layout(
                grid, _rank, _n_ranks,
            )
            model = MPASPrimitiveEquationModel(grid, sigma, config)
            state = baroclinic_wave_init_mpas(grid, sigma, perturbed=True, moist=_moist)
            # Each MPI rank uses 1 GPU (affinity set by
            # _configure_mpi_gpu_affinity).
            dev_config = create_voronoi_device_mesh(
                nCells=_voronoi_layout.local_mesh.nCells,
                nEdges=_voronoi_layout.local_mesh.nEdges,
                nVertices=_voronoi_layout.local_mesh.nVertices,
                n_devices=1,
            )
        else:
            # Single-node: reorder for spatial locality and shard.
            if n_gpus > 1:
                from legoesm.parallel.voronoi_partition import (
                    reorder_voronoi_for_sharding,
                )
                grid = reorder_voronoi_for_sharding(grid, n_gpus)

            dev_config = create_voronoi_device_mesh(
                nCells=grid.nCells,
                nEdges=grid.nEdges,
                nVertices=grid.nVertices,
                n_devices=n_gpus,
            )

            # Replicate mesh on all devices so JIT-compiled operators
            # find connectivity arrays locally without cross-device gathers.
            if dev_config.n_devices > 1:
                from legoesm.parallel.mesh import replicate_pytree
                grid = replicate_pytree(grid, dev_config)

            model = MPASPrimitiveEquationModel(grid, sigma, config)
            state = baroclinic_wave_init_mpas(grid, sigma, perturbed=True, moist=_moist)
    elif grid_type == "latlon":
        # Lat-lon finite-volume C-grid primitive equations.  (The old A-grid
        # lat-lon dycore referenced by #115 was removed; this is the current
        # C-grid FV core, the same solver the driver resolves for
        # grid=latlon/discretization=finite_volume.)
        from legoesm import constants  # lazy: see top-of-file note on JAX init order
        from legoesm.grids.latlon import create_latlon_grid
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel,
            CGridLatLonPrimitiveEquationConfig,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon

        grid = create_latlon_grid(
            n_lat=n_grid, radius=constants.R_earth, omega=constants.Omega,
        )
        n_lat, n_lon = grid.n_lat, grid.n_lon
        total_cells = n_lat * n_lon * n_levels
        # CFL-safe Laplacian viscosity (same form as
        # driver.component_factory.compute_diffusion; inlined to avoid importing
        # the driver stack here, which trips a device_config circular import in
        # the benchmark subprocess).  grid.dx is the 2-cell zonal span, so the
        # min cell width is half of it.  The polar filter lifts the pole-cell
        # CFL, so dt is the equatorial CFL value that _auto_dt returns.
        dx_min = float(jnp.min(grid.dx)) / 2.0
        A_h = 0.05 * dx_min ** 2 / dt
        config = CGridLatLonPrimitiveEquationConfig(
            A_h=A_h,
            fix_mass=not no_conservation,
            use_polar_filter=True,
            time_integrator="ssp_rk3",
        )
        # The baroclinic-wave IC is always built on the GLOBAL grid; the MPI
        # path scatters it to the rank-local band below (--- MPI scatter ---).
        state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True, moist=_moist)
        if _is_mpi:
            # Latitude-band domain decomposition (issue #641 Grid 3): each rank
            # owns a contiguous lat band with pole_bc='wall' + diagnostic
            # physics — the validated make_latlon_mpi_step path
            # (tests/distributed/test_latlon_mpi_step.py: MPI == serial after
            # gather, np=2/4).  The full-sphere model is rebuilt per rank on the
            # band grid; slice_latlon_grid_to_band allreduces band areas so the
            # mass fixer divides by the global sphere area, not the band area.
            from legoesm.parallel.latlon_mpi import (
                make_latlon_band_layout,
                slice_latlon_grid_to_band,
            )
            _latlon_layout = make_latlon_band_layout(
                rank=_rank, n_ranks=_n_ranks,
                n_lat=grid.n_lat, n_lon=grid.n_lon,
            )
            band_grid = slice_latlon_grid_to_band(grid, _latlon_layout)
            model = CGridLatLonPrimitiveEquationModel(
                band_grid, sigma, config, dt=dt,
            )
            # One device per MPI rank (affinity set by
            # _configure_mpi_gpu_affinity), as for the icosahedral MPI path.
            dev_config = create_latlon_mesh(n_devices=1)
        else:
            if n_gpus > 1:
                raise NotImplementedError(
                    "Single-process multi-device SPMD (route-B) is not yet "
                    "wired for the lat-lon grid in this harness (there is no "
                    "make_latlon_sharded_step). For multi-GPU lat-lon scaling "
                    "use MPI domain decomposition (mpirun -np N ... --grid "
                    "latlon), which routes through "
                    "legoesm.parallel.latlon_mpi.make_latlon_mpi_step."
                )
            model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
            dev_config = create_latlon_mesh(n_devices=n_gpus)
    else:
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
            CDGridPrimitiveEquationConfig,
            hydrostatic_to_fv3,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        grid = create_cubed_sphere(n_grid)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        hd = hyperdiff_coeff(n_grid, grid_type)
        # When mass anchoring (``fix_mass_hydrostatic_target``) is active
        # we already get exact mass conservation via a single post-step
        # allreduce.  The per-stage ``zero_mean_ps_tendency`` correction
        # adds 3 allreduces per RK3 step (one per ``tendency_fn`` call)
        # for what is, with ``anchor_mass_to_initial=True``, a redundant
        # safety net at the cost of three extra latency-gated round-trips
        # per step.  Disable it in the scaling benchmark — the comment
        # in ``CDGridPrimitiveEquationConfig`` explicitly recommends this
        # for "pure performance benchmarks".
        config = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=hd,
            hyperdiff_ps_coeff=hd,
            use_conservation_fixer=not no_conservation,
            fix_mass=not no_conservation,
            anchor_mass_to_initial=not no_conservation,
            zero_mean_ps_tendency=False,
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, config)
        state_cc = baroclinic_wave_init(grid, sigma, perturbed=True, moist=_moist)
        state = hydrostatic_to_fv3(state_cc, cdgrid)

        total_cells = 6 * n_grid * n_grid * n_levels

        # MPI distributed: use per-rank local device count, not total
        from legoesm.parallel.mesh import get_active_config
        active_cfg = get_active_config()
        if active_cfg is not None and active_cfg.is_distributed:
            dev_config = active_cfg
        else:
            dev_config = create_device_mesh(n_devices=n_gpus)
            # Tripwire (codex ppermute-multiface review): the row below is
            # labeled with n_gpus, so the mesh MUST actually have that many
            # devices.  create_device_mesh silently clamps when fewer
            # devices are visible (e.g. CUDA_VISIBLE_DEVICES mishap) — the
            # old behavior recorded "2-GPU" rows from de-facto 1-device
            # (replicated) programs without complaint.
            if dev_config.n_devices != n_gpus:
                raise RuntimeError(
                    f"requested n_gpus={n_gpus} but the device mesh "
                    f"resolved to {dev_config.n_devices} device(s) — "
                    f"refusing to record a mislabeled scaling row "
                    f"(visible devices: {len(jax.devices())})"
                )

    backend = dev_config.backend

    # Guard: warn if MPI is active for a grid without validated MPI paths.
    # Cubed-sphere is validated ONLY with --cs-mpi-scatter (true face
    # decomposition); without it, cube MPI is replicated-dynamics noise.
    _is_cs_distributed = dev_config.is_distributed
    _cube_scatter_ok = grid_type == "cubed-sphere" and cs_mpi_scatter
    if (
        (_is_cs_distributed or _is_mpi)
        and grid_type not in _MPI_SUPPORTED_GRIDS
        and not _cube_scatter_ok
    ):
        print(
            f"  WARNING: MPI distributed benchmarks for grid_type={grid_type!r} "
            f"are not validated in this script. Only {sorted(_MPI_SUPPORTED_GRIDS)} "
            f"have validated MPI paths (cubed-sphere requires --cs-mpi-scatter).",
            flush=True,
        )

    # Cast to desired precision
    def _cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x
    state = jax.tree.map(_cast, state)

    # --- MPI scatter for rank-local grids ---
    # Cubed-sphere MPI still keeps full state on all ranks; icosahedral
    # scatters to rank-local domains.
    if _voronoi_layout is not None:
        from legoesm.parallel.voronoi_mpi import scatter_state_voronoi
        state = scatter_state_voronoi(state, _voronoi_layout.partition)
    elif _latlon_layout is not None:
        # Lat-lon MPI: slice the global baroclinic-wave IC to this rank's band.
        # baroclinic_wave_init_latlon yields a cell-centered, Field-wrapped
        # HydrostaticState (what model.step accepts and converts internally);
        # make_latlon_mpi_step delegates straight to _step_cgrid, which wants a
        # raw-array C-grid CGridLatLonHydrostaticState.  Convert on the GLOBAL
        # grid first — the cell->face v-wind interpolation needs the full
        # latitude column — then slice to the band.
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            hydrostatic_to_cgrid,
        )
        from legoesm.parallel.latlon_mpi import scatter_state_latlon
        state = hydrostatic_to_cgrid(state, grid)
        state = scatter_state_latlon(state, _latlon_layout)

    # Cubed-sphere MPI: keep full (6, n, n, ...) state on every rank
    # to match the production driver.  pad_halo_mpi requires the full
    # shape; MPI halo exchange keeps owned faces correct.
    from legoesm.parallel.distributed import get_active_layout
    if _is_cs_distributed and get_active_layout() is None:
        from legoesm.parallel.distributed import get_active_topology, set_active_layout
        from legoesm.parallel.layout import make_layout
        topo = get_active_topology()
        if topo is not None:
            set_active_layout(make_layout(topo.rank, topo.n_processes, n_grid))

    # Cubed-sphere MPI face-scatter (opt-in --cs-mpi-scatter): give each rank
    # ONLY its owned faces — TRUE domain decomposition instead of replicated
    # dynamics.  Slices the model's grid/cdgrid metrics to owned faces (the
    # cross-face halo tables stay full; see cube_face_scatter) and scatters the
    # state; model.step then runs on (n_local, n, n, ...) with the MPI face-only
    # halo exchanging owned-face edges.  Validated bit-for-bit (incl. AD) vs the
    # single-process reference in tests/distributed/test_cube_face_scatter_mpi.py.
    if _is_cs_distributed and cs_mpi_scatter and grid_type == "cubed-sphere":
        from legoesm.parallel.cube_face_scatter import make_rank_local_cube_model
        from legoesm.parallel.distributed import get_active_topology
        from legoesm.parallel.layout import make_layout, scatter_pytree
        topo = get_active_topology()
        if topo is not None and topo.n_processes > 1:
            owned = topo.local_face_ids
            make_rank_local_cube_model(model, owned)
            state = scatter_pytree(
                state, make_layout(topo.rank, topo.n_processes, n_grid)
            )

    # Shard across devices (SPMD for multi-GPU single-node, non-MPI)
    if dev_config.n_devices > 1 and not _is_cs_distributed and not _is_mpi:
        state = shard_pytree(state, dev_config)

    # Verify sharding is effective (not accidentally replicated).
    # ``NamedSharding`` has no ``.shape`` attribute, so the previous
    # ``getattr(..., 'shape', (1,))`` always returned ``(1,)`` and the
    # warning fired for every multi-device cubed-sphere run, masking any
    # real replication issue.  Use the existing diagnostic helper which
    # inspects ``sharding.spec``.
    if dev_config.n_devices > 1 and not _is_mpi:
        from legoesm.parallel.sharded_dynamics import check_sharding
        sr = check_sharding(state, dev_config)
        if sr["n_sharded"] == 0 and grid_type != "icosahedral":
            print(
                f"    WARNING: State appears fully replicated — sharding "
                f"may not be effective ({sr['n_replicated']} replicated, "
                f"{sr['n_unsharded']} unsharded leaves)",
                flush=True,
            )

    # Build physics function.  Held-Suarez and the grid-agnostic moist
    # (Kessler) tier both produce a column-local physics_fn; the radiative
    # tiers (gray_sbm/rrtmg_full) returned earlier via the segment path.
    if _moist:
        physics_fn = _build_moist_physics_fn(grid_type, dt)
    else:
        physics_fn = _build_physics_fn(physics_level, grid_type)

    # --- Step function selection ---
    # MPI distributed step functions take priority over SPMD sharded steps.
    if _voronoi_layout is not None:
        # Voronoi MPI: physics is applied operator-split INSIDE the step on
        # the rank-local mesh after a halo exchange, so physics_fn is passed
        # to the factory (NOT wrapped post-hoc like the other grids below).
        from legoesm.parallel.voronoi_mpi import make_voronoi_mpi_step
        step_fn = make_voronoi_mpi_step(
            model, _voronoi_layout, sigma, config, physics_fn=physics_fn,
        )
    elif _latlon_layout is not None:
        # Lat-lon MPI: the backend-dispatched pad_halo_latlon* operators inside
        # _step_cgrid fetch band-cut halos via sendrecv and apply the wall pole
        # BC at boundary ranks; physics_fn is forwarded per RK stage to
        # _step_cgrid (it REFUSES a stateful PhysicsState-carry scheme loudly —
        # only diagnostic physics is supported, issue #405/#413).
        from legoesm.parallel.latlon_mpi import make_latlon_mpi_step
        step_fn = make_latlon_mpi_step(
            model, _latlon_layout, physics_fn=physics_fn,
        )
    # SPMD sharded step functions (single-node multi-GPU).
    elif grid_type == "icosahedral" and dev_config.n_devices > 1:
        from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
        step_fn = make_voronoi_sharded_step(model, dev_config)
    elif grid_type == "cubed-sphere" and dev_config.n_devices > 1:
        from legoesm.parallel.sharded_dynamics import make_sharded_step
        step_fn = make_sharded_step(model, dev_config, n=n_grid, nlev=n_levels)
        # ``make_sharded_step`` itself activates the explicit SPMD halo
        # backend for face-sharded counts (2/3/6 devices, tiling (1,1));
        # do NOT re-activate it here — assert it actually happened.  If
        # the "local" backend were silently left active, ``pad_halo``
        # would fall back to implicit cross-shard reads that GSPMD
        # resolves by replicating compute on every device (HLO probe
        # job 8456476) — correct numbers, meaningless timings.
        if dev_config.tiling == (1, 1):
            from legoesm.grids.halo import get_halo_backend
            _active_halo = get_halo_backend()
            if _active_halo != "spmd":
                raise RuntimeError(
                    f"cubed-sphere single-process multi-GPU benchmark "
                    f"requires the SPMD halo backend, but the active "
                    f"backend is {_active_halo!r} — make_sharded_step did "
                    f"not activate it (n_devices={dev_config.n_devices}, "
                    f"tiling={dev_config.tiling})"
                )
    else:
        step_fn = model.step

    # Wrap step_fn to include physics for non-Voronoi-MPI grids.
    # Voronoi MPI already received physics_fn via make_voronoi_mpi_step above
    #   (operator-split inside the step), so it is excluded here to avoid a
    #   double application.
    # For cubed-sphere and icosahedral SPMD, physics_fn is passed to __call__.
    # For single-GPU all grids, physics_fn is passed to model.step.
    if physics_fn is not None and _voronoi_layout is None and _latlon_layout is None:
        if dev_config.n_devices > 1 and grid_type in ("cubed-sphere", "icosahedral"):
            # CompiledShardedStep / VoronoiShardedStep accept physics_fn
            _sharded_step = step_fn
            _phys = physics_fn
            step_fn = lambda state, dt: _sharded_step(state, dt, physics_fn=_phys)
        else:
            # Single-GPU: pass physics_fn to model.step
            _model_step = step_fn
            _phys = physics_fn
            step_fn = lambda state, dt: _model_step(state, dt, physics_fn=_phys)

    cells_per_gpu = total_cells // max(1, n_gpus)

    if grid_type == "spectral":
        res_label = f"T{n_grid}"
    elif grid_type == "icosahedral":
        res_label = f"I{n_grid}"
    elif grid_type == "latlon":
        res_label = f"LL{n_grid}"
    else:
        res_label = f"C{n_grid}"
    phys_tag = f" | physics={physics_level}" if physics_level != "none" else ""
    print(
        f"  [{precision}] {res_label}/L{n_levels} on {n_gpus} GPU(s) | "
        f"dt={dt:.0f}s | cells={total_cells:,} | cells/GPU={cells_per_gpu:,}"
        f"{phys_tag}",
        flush=True,
    )

    # ---------------------------------------------------------------
    # JIT compilation (first call)
    # ---------------------------------------------------------------
    t_compile_start = time.perf_counter()
    state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    compile_time = time.perf_counter() - t_compile_start
    print(f"    JIT compile: {compile_time:.2f}s", flush=True)

    # ---------------------------------------------------------------
    # Warmup (let caches settle, exclude from timing)
    # ---------------------------------------------------------------
    t_warmup_start = time.perf_counter()
    for _ in range(n_warmup):
        state = step_fn(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    warmup_time = time.perf_counter() - t_warmup_start

    # ---------------------------------------------------------------
    # Timed steps — use lax.scan to compile all timing steps into a
    # single XLA program, eliminating per-step host dispatch overhead
    # and enabling XLA's latency-hiding scheduler to pipeline
    # collectives across steps.  The sharding gate, both sharding
    # tripwires, the full-cube all-gather HLO guard + collective-
    # permute census, and the clone-precompile step all live in the
    # shared `_build_timed_scan_runner` (also used by the moist
    # segment branch).
    # ---------------------------------------------------------------
    scan_runner, _hlo_counts = _build_timed_scan_runner(
        step_fn=step_fn,
        state=state,
        dev_config=dev_config,
        grid_type=grid_type,
        n_timing=n_timing,
        dt_static=float(dt),
        n_grid=n_grid,
        n_levels=n_levels,
        n_gpus=n_gpus,
        precision=precision,
        is_cs_distributed=_is_cs_distributed,
        is_mpi=_is_mpi,
    )

    # Synchronize all ranks before timing for fair measurement
    try:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.Get_size() > 1:
            jax.block_until_ready(jax.tree.leaves(state))
            _MPI.COMM_WORLD.Barrier()
    except (ImportError, RuntimeError):
        # broken/absent mpi4py must not break a single-process run
        pass

    t0 = time.perf_counter()
    state = scan_runner(state)
    jax.block_until_ready(jax.tree.leaves(state))

    # Synchronize all ranks after timing for fair measurement
    try:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.Get_size() > 1:
            _MPI.COMM_WORLD.Barrier()
    except (ImportError, RuntimeError):
        # broken/absent mpi4py must not break a single-process run
        pass

    t1 = time.perf_counter()

    timing_time = t1 - t0
    time_per_step = timing_time / n_timing
    time_per_step_ms = time_per_step * 1000.0

    # SYPD: simulated years per wall-clock day
    simulated_seconds_per_step = dt
    wall_seconds_per_step = time_per_step
    sypd = (simulated_seconds_per_step / wall_seconds_per_step) / (365.25 * 86400) * 86400.0

    mcells_per_s = (total_cells / time_per_step) / 1e6

    print(
        f"    Timing: {time_per_step_ms:.2f} ms/step | "
        f"SYPD={sypd:.3f} | "
        f"{mcells_per_s:.1f} Mcells/s",
        flush=True,
    )

    return TimingResult(
        n_gpus=n_gpus,
        resolution=n_grid,
        n_levels=n_levels,
        precision=precision,
        mode=mode,
        physics_level=physics_level,
        dt_seconds=dt,
        n_warmup=n_warmup,
        n_timing=n_timing,
        compile_time_s=compile_time,
        warmup_time_s=warmup_time,
        timing_time_s=timing_time,
        time_per_step_ms=time_per_step_ms,
        sypd=sypd,
        total_cells=total_cells,
        cells_per_gpu=cells_per_gpu,
        mcells_per_s=mcells_per_s,
        grid_type=grid_type,
        **_hlo_census_fields(_hlo_counts),
    )


# ===========================================================================
# Weak scaling sweep
# ===========================================================================

def run_weak_scaling(
    *,
    n_gpus: int,
    precisions: list[str],
    n_levels: int,
    n_warmup: int,
    n_timing: int,
    base_n: int = WEAK_SCALING_BASE_N,
    base_level_ico: int = WEAK_SCALING_BASE_LEVEL_ICO,
    grid_type: str = "spectral",
    no_conservation: bool = False,
    physics_level: str = "none",
    cs_mpi_scatter: bool = False,
    fixed_gpu_count: int | None = None,
) -> list[TimingResult]:
    """Run weak scaling: fix cells/GPU, sweep GPU counts up to n_gpus.

    Parameters
    ----------
    fixed_gpu_count : int, optional
        When set (e.g. under MPI), run only at this GPU count instead
        of sweeping all valid counts up to *n_gpus*.
    """

    if fixed_gpu_count is not None:
        gpu_counts = [fixed_gpu_count]
    else:
        gpu_counts = _valid_gpu_counts(n_gpus, grid_type)
    results: list[TimingResult] = []

    if grid_type == "spectral":
        res_prefix = "T"
    elif grid_type == "icosahedral":
        res_prefix = "I"
    elif grid_type == "latlon":
        res_prefix = "LL"
    else:
        res_prefix = "C"
    if grid_type == "icosahedral":
        base_label = f"base level={base_level_ico} ({10 * 4**base_level_ico + 2} cells)"
    elif grid_type == "latlon":
        base_label = f"base N={WEAK_SCALING_BASE_N_LL}"
    else:
        base_label = f"base N={base_n}"
    print(f"\n{'='*72}")
    print(f"WEAK SCALING ({grid_type}, {base_label}, {n_levels} levels)")
    print(f"GPU counts: {gpu_counts}")
    print(f"{'='*72}")

    for prec in precisions:
        _configure_jax(prec)
        for ng in gpu_counts:
            if grid_type == "icosahedral":
                n_grid = _weak_resolution_ico(ng, base_level_ico)
            elif grid_type == "latlon":
                _n_mpi_ranks = fixed_gpu_count if fixed_gpu_count else 1
                n_grid = _weak_resolution_ll(ng, n_ranks=_n_mpi_ranks)
            else:
                n_grid = _weak_resolution(ng, base_n)
            print(f"\n--- {ng} GPU(s), {res_prefix}{n_grid} ---")
            try:
                result = run_benchmark(
                    n_grid=n_grid,
                    n_levels=n_levels,
                    n_gpus=ng,
                    precision=prec,
                    mode="weak",
                    n_warmup=n_warmup,
                    n_timing=n_timing,
                    grid_type=grid_type,
                    no_conservation=no_conservation,
                    physics_level=physics_level,
                    cs_mpi_scatter=cs_mpi_scatter,
                )
                results.append(result)
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)

    # Compute scaling efficiency relative to 1-GPU baseline.
    # For icosahedral grids the subdivision level jumps by 4× in cell
    # count, so cells/GPU is not constant across GPU counts.  Normalize
    # by the cells/GPU ratio to avoid misleading efficiency numbers.
    for prec in precisions:
        prec_results = [r for r in results if r.precision == prec]
        baseline = next((r for r in prec_results if r.n_gpus == 1), None)
        if baseline is not None:
            for r in prec_results:
                # Ideal weak scaling: constant throughput per cell per GPU.
                # efficiency = (t1 / tN) * (cells_per_gpu_N / cells_per_gpu_1)
                cell_ratio = r.cells_per_gpu / baseline.cells_per_gpu
                r.scaling_efficiency = (
                    baseline.time_per_step_ms / r.time_per_step_ms * cell_ratio
                )

    return results


# ===========================================================================
# Strong scaling sweep
# ===========================================================================

def run_strong_scaling(
    *,
    n_gpus: int,
    precisions: list[str],
    n_levels: int,
    n_warmup: int,
    n_timing: int,
    resolutions: list[int] | None = None,
    grid_type: str = "spectral",
    no_conservation: bool = False,
    physics_level: str = "none",
    cs_mpi_scatter: bool = False,
    fixed_gpu_count: int | None = None,
) -> list[TimingResult]:
    """Run strong scaling: fix resolution, sweep GPU counts up to n_gpus.

    Parameters
    ----------
    fixed_gpu_count : int, optional
        When set (e.g. under MPI), run only at this GPU count instead
        of sweeping all valid counts up to *n_gpus*.
    """

    if fixed_gpu_count is not None:
        gpu_counts = [fixed_gpu_count]
    else:
        gpu_counts = _valid_gpu_counts(n_gpus, grid_type)
    if resolutions is None:
        if grid_type == "spectral":
            defaults = STRONG_RESOLUTIONS_SP
        elif grid_type == "icosahedral":
            defaults = STRONG_RESOLUTIONS_ICO
        elif grid_type == "latlon":
            defaults = STRONG_RESOLUTIONS_LL
        else:
            defaults = STRONG_RESOLUTIONS_CS
        if grid_type == "icosahedral":
            resolutions = defaults  # levels don't need GPU-count filtering
        else:
            resolutions = [r for r in defaults if r >= 2 * min(gpu_counts)]
        if not resolutions:
            resolutions = [defaults[0]]

    results: list[TimingResult] = []

    if grid_type == "spectral":
        res_prefix = "T"
    elif grid_type == "icosahedral":
        res_prefix = "I"
    elif grid_type == "latlon":
        res_prefix = "LL"
    else:
        res_prefix = "C"
    print(f"\n{'='*72}")
    print(f"STRONG SCALING ({grid_type}, {n_levels} levels)")
    print(f"Resolutions: {[res_prefix+str(r) for r in resolutions]}")
    print(f"GPU counts:  {gpu_counts}")
    print(f"{'='*72}")

    for prec in precisions:
        _configure_jax(prec)
        for n_grid in resolutions:
            for ng in gpu_counts:
                # Skip if grid is too small for this many GPUs
                if grid_type == "icosahedral":
                    n_cells_ico = 10 * 4 ** n_grid + 2
                    if n_cells_ico < ng * 10:
                        print(
                            f"\n--- {ng} GPU(s), {res_prefix}{n_grid} --- SKIPPED "
                            f"(grid too small for {ng} GPUs)",
                            flush=True,
                        )
                        continue
                elif grid_type == "latlon":
                    # Lat-lon MPI uses make_latlon_band_layout, which supports
                    # UNEVEN bands (the first n_lat % n_ranks ranks get one
                    # extra row) — n_lat need NOT be divisible by the rank
                    # count.  The binding constraint is per-rank: every band
                    # must carry at least the halo width, else
                    # exchange_halo_latlon raises (halo > n_lat_local) before
                    # timing.  The smallest band is floor(n_lat / n_ranks), so
                    # skip when that drops below the halo.  (Single-process
                    # latlon runs at ng=1: band = n_lat >> halo.)
                    _min_band = n_grid // ng
                    if _min_band < _LATLON_MPI_HALO:
                        print(
                            f"\n--- {ng} GPU(s), {res_prefix}{n_grid} --- SKIPPED "
                            f"(n_lat={n_grid}/{ng} -> band {_min_band} < halo "
                            f"{_LATLON_MPI_HALO})",
                            flush=True,
                        )
                        continue
                elif grid_type == "cubed-sphere":
                    cells_per_face = n_grid * n_grid
                    if ng > 6 and cells_per_face < 16:
                        print(
                            f"\n--- {ng} GPU(s), {res_prefix}{n_grid} --- SKIPPED "
                            f"(grid too small for sub-face tiling)",
                            flush=True,
                        )
                        continue

                # Estimate total cells for auto-scaling timing steps
                if grid_type == "icosahedral":
                    _est_cells = (10 * 4 ** n_grid + 2) * n_levels
                elif grid_type == "latlon":
                    _est_cells = n_grid * 2 * n_grid * n_levels
                elif grid_type == "spectral":
                    _est_cells = n_grid * (n_grid + 1) * n_levels
                else:
                    _est_cells = 6 * n_grid * n_grid * n_levels
                _nt = _auto_n_timing(n_timing, _est_cells, ng)

                print(f"\n--- {ng} GPU(s), {res_prefix}{n_grid} [{prec}] ---")
                if _nt != n_timing:
                    print(f"    (auto-scaled n_timing: {n_timing} → {_nt})")
                try:
                    result = run_benchmark(
                        n_grid=n_grid,
                        n_levels=n_levels,
                        n_gpus=ng,
                        precision=prec,
                        mode="strong",
                        n_warmup=n_warmup,
                        n_timing=_nt,
                        grid_type=grid_type,
                        no_conservation=no_conservation,
                        physics_level=physics_level,
                        cs_mpi_scatter=cs_mpi_scatter,
                    )
                    results.append(result)
                except Exception as exc:
                    print(f"    FAILED: {exc}", flush=True)

    # Compute strong scaling efficiency relative to fewest-GPU baseline
    for prec in precisions:
        for n_grid in resolutions:
            group = [
                r for r in results
                if r.precision == prec and r.resolution == n_grid
            ]
            baseline = min(group, key=lambda r: r.n_gpus) if group else None
            if baseline is not None:
                for r in group:
                    ideal_speedup = r.n_gpus / baseline.n_gpus
                    actual_speedup = baseline.time_per_step_ms / r.time_per_step_ms
                    r.scaling_efficiency = actual_speedup / ideal_speedup

    return results


# ===========================================================================
# Output: CSV, JSON, summary table
# ===========================================================================

def _device_memory_stats() -> list[dict]:
    """Best-effort per-device memory stats for metadata.json.

    ``Device.memory_stats()`` is backend/version dependent (returns
    ``None`` on CPU, may raise on some platforms).  Returns ``{}`` or an
    error string per device when unavailable — NEVER fails the benchmark
    over memory accounting.
    """
    import jax  # lazy: see top-of-file note on JAX init order

    try:
        devices = jax.local_devices()
    except Exception as exc:  # pragma: no cover — defensive
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    stats = []
    for i, d in enumerate(devices):
        entry: dict[str, Any] = {
            "id": getattr(d, "id", i),
            "platform": getattr(d, "platform", "unknown"),
        }
        try:
            ms = d.memory_stats()
            entry["memory_stats"] = (
                {
                    str(k): (v if isinstance(v, (int, float, bool, str))
                             else str(v))
                    for k, v in ms.items()
                }
                if ms else {}
            )
        except Exception as exc:
            entry["memory_stats"] = f"unavailable: {type(exc).__name__}: {exc}"
        stats.append(entry)
    return stats


def write_csv(results: list[TimingResult], path: Path) -> None:
    """Write results to a CSV file."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"  CSV: {path}")


def write_json(
    report: ScalingReport,
    path: Path,
    *,
    n_ranks_true: int | None = None,
    component: str = "atmosphere",
    metadata_overrides: dict | None = None,
) -> None:
    """Write full report to JSON.

    ``component`` labels every row's metadata block — cross-script consumers
    (``bench_ocean_mpi_scaling.py``) MUST pass their own component so an
    ocean row is never stamped "atmosphere".  ``metadata_overrides`` merges
    extra ``scaling_metadata`` kwargs (e.g. ``solver_variant``) into each row.

    ``n_ranks_true`` is the real MPI world size from ``_maybe_init_distributed``
    (1 for single-process SPMD, N for the route-A MPI path).  It MUST be passed
    for the MPI route-A latlon/icosahedral path, where ``jax.distributed`` is
    NOT initialized on a single node, so ``jax.process_count()`` would report 1
    and the auto default would falsely record ``n_ranks=1`` for an
    ``mpirun -np N`` run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _live_process_count() -> int:
        try:
            import jax

            return int(jax.process_count())
        except Exception:
            return 1

    def _decomp(grid: str) -> str:
        # MPI route-A (world size > 1): grid-specific domain decomposition;
        # single-process runs shard via SPMD.
        if n_ranks_true and n_ranks_true > 1:
            if grid == "latlon":
                return "band"
            if grid == "icosahedral":
                return "cell_partition"
            return "mpi"
        return "spmd"

    def _row(r: TimingResult) -> dict:
        d = asdict(r)
        # Self-describing metadata (roadmap item 9): backend / precision knobs /
        # GPU-direct mode / decomposition / cells-per-GPU so this row is
        # comparable and a host-staged or f32 GPU row is falsifiable.  n_gpus is
        # the row's device count (scaling axis); n_ranks is the true MPI world
        # size (SPMD -> 1; route-A -> N), NOT jax.process_count() which is 1 on
        # single-node MPI where jax.distributed is not initialized.
        md_kwargs: dict = dict(
            grid=r.grid_type,
            component=component,
            resolution=r.resolution,
            n_levels=r.n_levels,
            precision=r.precision,
            n_ranks=n_ranks_true,
            # n_ranks_true>1 is by contract the route-A MPI path (mpi4jax
            # halos) — pin the transport explicitly, because a multi-node
            # route-A run may ALSO have jax.distributed initialized
            # (process_count == n_ranks), which would auto-resolve to
            # nccl/gloo and mislabel the fabric (codex finding 1).
            transport=("mpi4jax" if (n_ranks_true or 1) > 1 else None),
            n_gpus=r.n_gpus,
            decomposition=os.environ.get("LEGOESM_DECOMPOSITION")
            or _decomp(r.grid_type),
            # cells_per_rank is per PROCESS (n_ranks semantics): route-A
            # divides by the true MPI world; otherwise by the live process
            # count (1 for single-process SPMD — that one rank owns ALL
            # cells).  The per-device share stays in extra.cells_per_device.
            cells_per_rank=r.total_cells // max(
                n_ranks_true if (n_ranks_true and n_ranks_true > 1)
                else _live_process_count(), 1),
            scaling_kind=os.environ.get("LEGOESM_SCALING_KIND") or None,
            extra={
                "physics_level": r.physics_level,
                "mode": r.mode,
                "hlo_collective_permute": r.hlo_collective_permute,
                "cells_per_device": r.cells_per_gpu,
            },
        )
        if metadata_overrides:
            md_kwargs.update(metadata_overrides)
        d["metadata"] = annotate_incomplete(scaling_metadata(**md_kwargs))
        return d

    payload = {
        "mode": report.mode,
        "precisions": report.precisions,
        "timestamp_utc": report.timestamp_utc,
        "backend": report.backend,
        "hostname": report.hostname,
        "results": [_row(r) for r in report.results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON: {path}")


def print_summary_table(results: list[TimingResult], mode: str) -> None:
    """Print a summary table to stdout."""
    if not results:
        return

    print(f"\n{'='*90}")
    print(f"  {mode.upper()} SCALING SUMMARY")
    print(f"{'='*90}")

    header = (
        f"{'Prec':>7s} | {'GPUs':>5s} | {'Res':>5s} | {'L':>3s} | "
        f"{'dt(s)':>6s} | {'cells/GPU':>9s} | {'ms/step':>9s} | "
        f"{'SYPD':>8s} | {'Mcell/s':>9s} | {'Eff':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (x.precision, x.resolution, x.n_gpus)):
        print(
            f"{r.precision:>7s} | {r.n_gpus:>5d} | {r.resolution:<5d} | "
            f"{r.n_levels:>3d} | {r.dt_seconds:>6.0f} | "
            f"{r.cells_per_gpu:>9,d} | "
            f"{r.time_per_step_ms:>9.2f} | {r.sypd:>8.3f} | "
            f"{r.mcells_per_s:>9.1f} | {r.scaling_efficiency:>5.1%}"
        )
    print()


# ===========================================================================
# Plotting
# ===========================================================================

def plot_weak_scaling(results: list[TimingResult], output_dir: Path) -> None:
    """Generate weak scaling plot (time/step vs GPUs, CliMA Fig 11 style)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    colors = {"float32": "#2196F3", "float64": "#F44336"}
    markers = {"float32": "o", "float64": "s"}

    precisions = sorted(set(r.precision for r in results))
    all_gpus = sorted(set(r.n_gpus for r in results))

    for prec in precisions:
        prec_data = sorted(
            [r for r in results if r.precision == prec],
            key=lambda r: r.n_gpus,
        )
        if not prec_data:
            continue
        gpus = [r.n_gpus for r in prec_data]
        ms_per_step = [r.time_per_step_ms for r in prec_data]
        resolutions = [f"{r.resolution}" for r in prec_data]

        ax.plot(
            gpus, ms_per_step,
            color=colors.get(prec, "#333"),
            marker=markers.get(prec, "o"),
            markersize=7,
            linewidth=2,
            label=prec,
            zorder=5,
        )

        # Annotate resolution at each point
        for g, ms, res in zip(gpus, ms_per_step, resolutions):
            ax.annotate(
                res, (g, ms),
                textcoords="offset points", xytext=(0, 10),
                fontsize=7, ha="center", color=colors.get(prec, "#333"),
            )

    # Ideal weak scaling reference (horizontal line at 1-GPU value)
    for prec in precisions:
        baseline = next(
            (r for r in results if r.precision == prec and r.n_gpus == 1),
            None,
        )
        if baseline and len(all_gpus) > 1:
            ax.axhline(
                baseline.time_per_step_ms,
                color=colors.get(prec, "#333"),
                linestyle="--",
                alpha=0.4,
                linewidth=1,
                label=f"{prec} ideal",
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of GPUs", fontsize=12)
    ax.set_ylabel("Time per timestep [ms]", fontsize=12)
    ax.set_title("Weak Scaling -- Baroclinic Wave (Hydrostatic PE)", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    if all_gpus:
        ax.set_xticks(all_gpus)
        ax.set_xticklabels([str(g) for g in all_gpus])

    plt.tight_layout()
    path = output_dir / "weak_scaling.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Plot: {path}")


def _resolution_color_map(resolutions: list[int]) -> dict[int, str]:
    """Build a color map for an arbitrary set of resolution values.

    Uses a qualitative palette so every resolution gets a distinct,
    easy-to-read color regardless of grid type.
    """
    palette = [
        "#4CAF50",  # green
        "#2196F3",  # blue
        "#F44336",  # red
        "#9C27B0",  # purple
        "#FF9800",  # orange
        "#00BCD4",  # cyan
        "#795548",  # brown
        "#E91E63",  # pink
    ]
    return {res: palette[i % len(palette)] for i, res in enumerate(sorted(resolutions))}


def plot_strong_scaling(results: list[TimingResult], output_dir: Path) -> None:
    """Generate strong scaling plot (SYPD vs GPUs, CliMA Fig 12 style)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available -- skipping plots")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

    prec_linestyles = {"float32": "-", "float64": "--"}
    prec_markers = {"float32": "o", "float64": "s"}

    precisions = sorted(set(r.precision for r in results))
    resolutions = sorted(set(r.resolution for r in results))
    all_gpus = sorted(set(r.n_gpus for r in results))
    resolution_colors = _resolution_color_map(resolutions)

    for n_grid in resolutions:
        for prec in precisions:
            group = sorted(
                [r for r in results if r.resolution == n_grid and r.precision == prec],
                key=lambda r: r.n_gpus,
            )
            if not group:
                continue

            gpus = [r.n_gpus for r in group]
            sypd = [r.sypd for r in group]

            label = f"{n_grid} ({prec})"
            color = resolution_colors[n_grid]

            ax.plot(
                gpus, sypd,
                color=color,
                marker=prec_markers.get(prec, "o"),
                linestyle=prec_linestyles.get(prec, "-"),
                markersize=7,
                linewidth=2,
                label=label,
                zorder=5,
            )

    # Ideal strong scaling reference lines (dashed, from 1-GPU baseline)
    for n_grid in resolutions:
        baseline = next(
            (r for r in results if r.resolution == n_grid and r.n_gpus == min(
                r2.n_gpus for r2 in results if r2.resolution == n_grid
            )),
            None,
        )
        if baseline and len(all_gpus) > 1:
            gpu_range = np.array(sorted(
                set(r.n_gpus for r in results if r.resolution == n_grid)
            ))
            ideal_sypd = baseline.sypd * (gpu_range / baseline.n_gpus)
            ax.plot(
                gpu_range, ideal_sypd,
                color=resolution_colors[n_grid],
                linestyle=":",
                alpha=0.35,
                linewidth=1,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Number of GPUs", fontsize=12)
    ax.set_ylabel("SYPD (Simulated Years Per Day)", fontsize=12)
    ax.set_title("Strong Scaling -- Baroclinic Wave (Hydrostatic PE)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    if all_gpus:
        ax.set_xticks(all_gpus)
        ax.set_xticklabels([str(g) for g in all_gpus])

    plt.tight_layout()
    path = output_dir / "strong_scaling.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Plot: {path}")


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "GPU scaling benchmark for legoESM on Levante (DKRZ). "
            "Replicates CliMA JAMES 2026 Figures 11 and 12."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grid", choices=list(GRID_CHOICES), default="spectral",
        help="Grid / dycore type.",
    )
    p.add_argument(
        "--physics", choices=list(PHYSICS_CHOICES), default="none",
        help="Physics complexity level.  'none' = dycore only. "
             "'held_suarez' and 'moist' (Kessler warm-rain, no radiation) "
             "work on all grids — 'moist' is the grid-agnostic tier for a "
             "fair cross-grid many-GPU comparison (incl. icosahedral MPI). "
             "'gray_sbm' and 'rrtmg_full' require cubed-sphere or latlon.",
    )
    p.add_argument(
        "--mode", choices=["weak", "strong", "both"], default="both",
        help="Scaling mode: weak, strong, or both.",
    )
    p.add_argument(
        "--precision", choices=["float32", "float64", "both"], default="both",
        help="Floating-point precision(s) to benchmark.",
    )
    p.add_argument(
        "--n-gpus", type=int, default=0,
        help="Max number of GPUs to use (0 = auto-detect all available).",
    )
    p.add_argument(
        "--n-levels", type=int, default=26,
        help="Number of vertical levels.",
    )
    p.add_argument(
        "--n-warmup", type=int, default=3,
        help="Number of JIT warmup steps (excluded from timing).",
    )
    p.add_argument(
        "--n-timing", type=int, default=100,
        help="Number of steps to time (weak scaling mode).",
    )
    p.add_argument(
        "--weak-base-n", type=int, default=WEAK_SCALING_BASE_N,
        help="Base N for weak scaling (per 1 GPU).",
    )
    p.add_argument(
        "--strong-resolutions", type=str, default=None,
        help="Comma-separated resolutions for strong scaling "
             "(default: 42,85,170 spectral / 48,96,192 cubed-sphere / "
             "4,5,6 icosahedral).",
    )
    p.add_argument(
        "--output-dir", type=str, default="results/scaling",
        help="Base output directory for results and plots.",
    )
    p.add_argument(
        "--no-timestamp", action="store_true",
        help="Do not append a UTC timestamp sub-directory to --output-dir. "
             "Without this flag each run gets its own sub-directory, "
             "preventing accidental overwrite of previous results.",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation.",
    )
    p.add_argument(
        "--no-conservation", action="store_true",
        help="Disable conservation fixer and zero-mean tendency correction. "
             "Reduces global sync count for cleaner perf scaling measurement.",
    )
    p.add_argument(
        "--cs-mpi-scatter", action="store_true",
        help="Cubed-sphere MPI: scatter the 6 faces across ranks for TRUE "
             "domain decomposition (each rank owns 6/nranks faces) instead of "
             "replicating full state on every rank.  Face-only ranks (1/2/3/6). "
             "Without this, cubed-sphere MPI multi-rank timings are "
             "replicated-dynamics noise.  No effect on non-cubed-sphere grids.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    grid_type = args.grid

    # Resolve precision list
    if args.precision == "both":
        precisions = ["float32", "float64"]
    else:
        precisions = [args.precision]

    # GPU affinity must be set before JAX sees devices.
    _configure_mpi_gpu_affinity()

    # Configure JAX for first precision (will be reconfigured per run)
    _configure_jax(precisions[0])

    import jax

    # Initialize distributed runtime if under MPI/SLURM
    rank, world_size = _maybe_init_distributed(grid_type=grid_type)
    is_rank0 = (rank == 0)

    # Resolve GPU count
    if world_size > 1:
        # Distributed mode: total GPUs = local devices * world_size
        local_devices = jax.local_devices()
        local_gpu_count = len(local_devices)
        max_gpus = local_gpu_count * world_size
        if args.n_gpus > 0 and args.n_gpus != max_gpus:
            if is_rank0:
                print(
                    f"  WARNING: --n-gpus={args.n_gpus} overridden by "
                    f"distributed world: {local_gpu_count} local × "
                    f"{world_size} ranks = {max_gpus} GPUs"
                )
    else:
        try:
            available = len(jax.devices("gpu"))
        except RuntimeError:
            available = len(jax.devices())
        max_gpus = args.n_gpus
        if max_gpus <= 0:
            max_gpus = available
        elif max_gpus > available:
            # Fail LOUD: a silent clamp downstream produced months of
            # replicated "multi-GPU" rows with efficiency exactly 0.5.
            raise RuntimeError(
                f"--n-gpus={max_gpus} requested but JAX exposes only "
                f"{available} device(s) (CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}). Refusing "
                "to benchmark a clamped device count."
            )
        if max_gpus < 1:
            max_gpus = 1

    # Resolve strong scaling resolutions
    if args.strong_resolutions is not None:
        strong_res = [int(x.strip()) for x in args.strong_resolutions.split(",") if x.strip()]
    else:
        strong_res = None  # let run_strong_scaling pick defaults per grid type

    # Output directory — append a UTC timestamp sub-directory by default
    # so that successive runs never silently overwrite each other.
    output_dir = Path(args.output_dir)
    if not args.no_timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_dir / ts
    if is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine modes to run
    modes = []
    if args.mode in ("weak", "both"):
        modes.append("weak")
    if args.mode in ("strong", "both"):
        modes.append("strong")

    hostname = os.environ.get("HOSTNAME", os.environ.get("SLURM_NODELIST", "unknown"))
    backend = jax.default_backend().upper()

    physics_level = args.physics

    # Validate grid/physics combination early.
    _validate_physics(grid_type, physics_level)

    if is_rank0:
        print("=" * 72)
        print("  legoESM GPU Scaling Benchmark")
        print("=" * 72)
        print(f"  Grid:        {grid_type}")
        print(f"  Physics:     {physics_level}")
        print(f"  Backend:     {backend}")
        print(f"  Hostname:    {hostname}")
        print(f"  Max GPUs:    {max_gpus}")
        print(f"  Valid GPUs:  {_valid_gpu_counts(max_gpus, grid_type)}")
        print(f"  Precisions:  {precisions}")
        print(f"  Modes:       {modes}")
        print(f"  Levels:      {args.n_levels}")
        print(f"  Warmup:      {args.n_warmup} steps")
        print(f"  Timing:      {args.n_timing} steps")
        print(f"  Output:      {output_dir}")
        print("=" * 72)

    # Under MPI, the GPU count is fixed (= world_size * GPUs/rank).
    # Disable sweep by pinning to the actual count.
    fixed = max_gpus if world_size > 1 else None

    # Iter 13/17 honest-sweep guard: refuse multi-rank MPI for any
    # grid that is *not* in the validated MPI-supported set
    # (``icosahedral`` + ``latlon``).  A DEFAULT cubed-sphere MPI run keeps
    # full state per rank (replicated dynamics, not real scaling); spectral
    # has no MPI step.  This catches both with one branch and one message.
    #
    # EXCEPTION: ``--cs-mpi-scatter`` on cubed-sphere IS genuine face
    # decomposition (each rank owns a subset of the 6 faces; cross-face
    # halos exchange via mpi4jax), so it is allowed through here exactly
    # like the inner guard's ``_cube_scatter_ok`` carve-out.  Without
    # this the early guard aborts the run before the scatter path ever
    # executes — the two guards were inconsistent.
    _cube_scatter_ok = grid_type == "cubed-sphere" and args.cs_mpi_scatter
    if (world_size > 1 and grid_type not in _MPI_SUPPORTED_GRIDS
            and not _cube_scatter_ok):
        if is_rank0:
            print(
                f"ERROR: {grid_type} MPI multi-rank scaling is not "
                f"validated in this script. Supported MPI grids are "
                f"{sorted(_MPI_SUPPORTED_GRIDS)}.  Default cubed-sphere MPI "
                "is replicated-dynamics-only (pass ``--cs-mpi-scatter`` for "
                "genuine <=6-face decomposition); spectral has no MPI path. "
                "Run with a single MPI rank, add ``--cs-mpi-scatter``, or use "
                "``--grid icosahedral``/``--grid latlon`` for genuine "
                "multi-rank scaling.",
                flush=True,
            )
        raise SystemExit(2)

    all_results: list[TimingResult] = []

    # ---------------------------------------------------------------
    # Weak scaling
    # ---------------------------------------------------------------
    if "weak" in modes:
        weak_results = run_weak_scaling(
            n_gpus=max_gpus,
            precisions=precisions,
            n_levels=args.n_levels,
            n_warmup=args.n_warmup,
            n_timing=args.n_timing,
            base_n=args.weak_base_n,
            grid_type=grid_type,
            no_conservation=args.no_conservation,
            physics_level=physics_level,
            cs_mpi_scatter=args.cs_mpi_scatter,
            fixed_gpu_count=fixed,
        )
        all_results.extend(weak_results)
        if is_rank0:
            print_summary_table(weak_results, "weak")

            weak_report = ScalingReport(
                mode="weak",
                precisions=precisions,
                results=weak_results,
                backend=backend,
                hostname=hostname,
            )
            write_csv(weak_results, output_dir / "weak_scaling.csv")
            write_json(weak_report, output_dir / "weak_scaling.json",
                       n_ranks_true=world_size)

            if not args.no_plot:
                plot_weak_scaling(weak_results, output_dir)

    # ---------------------------------------------------------------
    # Strong scaling
    # ---------------------------------------------------------------
    if "strong" in modes:
        strong_results = run_strong_scaling(
            n_gpus=max_gpus,
            precisions=precisions,
            n_levels=args.n_levels,
            n_warmup=args.n_warmup,
            n_timing=args.n_timing,
            resolutions=strong_res,
            grid_type=grid_type,
            no_conservation=args.no_conservation,
            physics_level=physics_level,
            cs_mpi_scatter=args.cs_mpi_scatter,
            fixed_gpu_count=fixed,
        )
        all_results.extend(strong_results)
        if is_rank0:
            print_summary_table(strong_results, "strong")

            strong_report = ScalingReport(
                mode="strong",
                precisions=precisions,
                results=strong_results,
                backend=backend,
                hostname=hostname,
            )
            write_csv(strong_results, output_dir / "strong_scaling.csv")
            write_json(strong_report, output_dir / "strong_scaling.json",
                       n_ranks_true=world_size)

            if not args.no_plot:
                plot_strong_scaling(strong_results, output_dir)

    # ---------------------------------------------------------------
    # Combined output
    # ---------------------------------------------------------------
    if is_rank0:
        if all_results:
            write_csv(all_results, output_dir / "all_scaling.csv")

        # Save metadata
        meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "grid_type": grid_type,
            "backend": backend,
            "hostname": hostname,
            "max_gpus": max_gpus,
            "valid_gpu_counts": _valid_gpu_counts(max_gpus, grid_type),
            "precisions": precisions,
            "modes": modes,
            "n_levels": args.n_levels,
            "n_warmup": args.n_warmup,
            "n_timing": args.n_timing,
            "weak_base_n": args.weak_base_n,
            "strong_resolutions": strong_res,
            "jax_version": jax.__version__,
            "python_version": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_nodelist": os.environ.get("SLURM_NODELIST", ""),
            # Best-effort: {} / error string per device when the backend
            # has no memory accounting; never fails the run.
            "device_memory_stats": _device_memory_stats(),
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"  Metadata: {meta_path}")

        print(f"\nAll results saved to {output_dir}/")
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
