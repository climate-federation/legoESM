#!/usr/bin/env python
"""CPU MPI scaling benchmark for AMIP-like runs.

Measures wall-clock time per step and SYPD across varying MPI rank counts
and resolutions for the supported grid+physics combinations.

MPI-scalable grids (multi-rank weak/strong scaling, genuinely
domain-decomposed at the dycore level):
  icosahedral   -- MPAS Voronoi TRiSK PE dycore (cell partition)
  latlon        -- Lat-lon C-grid FV PE dycore (latitude-band
                   decomposition via ``make_latlon_mpi_step``;
                   needs >=2 lat rows per rank for the halo=2
                   PPM/biharmonic exchanges)

Single-rank only (listed but their MPI paths are not domain-decomposed
at the dycore level — see iter 13/14 honest-sweep guards):
  cubed-sphere  -- C-D grid + FV3 PE dycore (replicated dynamics
                   under MPI; iter 3 added scattered halo support but
                   driver-side state scatter is not yet implemented)
  spectral      -- Gaussian + spectral PE dycore (no MPI path at all)

Physics levels:
  held_suarez   -- Newtonian relaxation (cheapest, no I/O)
  gray_sbm      -- Gray radiation + SBM convection
  rrtmg_full    -- RRTMG radiation + Kessler microphysics + SBM

Design: one benchmark case per MPI process invocation (see F1 in plan).
Use --sweep to generate all cases for a SLURM array job.

Usage
-----
Multi-rank MPI scaling (icosahedral or latlon)::

    mpirun -np 8 python scripts/run_cpu_mpi_scaling.py \\
        --grid icosahedral --mode strong --physics held_suarez

    mpirun -np 4 python scripts/run_cpu_mpi_scaling.py \\
        --grid latlon --mode strong --physics held_suarez

Single-rank case (any grid)::

    python scripts/run_cpu_mpi_scaling.py \\
        --grid cubed-sphere --resolution 48 --physics held_suarez

Sweep mode (generate case list, no execution)::

    python scripts/run_cpu_mpi_scaling.py --sweep \\
        --grid icosahedral --mode strong --physics held_suarez
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

# NOTE: do NOT import ``legoesm.constants`` at module load — it eagerly
# imports ``jax.numpy``, which initialises JAX before ``_configure_jax_cpu``
# has a chance to set ``JAX_ENABLE_X64`` / ``JAX_PLATFORMS`` / thread flags.
# Use a lazy import inside the function that needs ``constants.R_earth``.

# ---------------------------------------------------------------------------
# JAX configuration -- must happen before JAX import
# ---------------------------------------------------------------------------

def _configure_jax_cpu(precision: str) -> None:
    """Force JAX to CPU, set precision, pin each rank to 1 thread."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    if precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Threads per rank = SLURM cpus-per-task (cpu-bind confines them to THIS
    # rank's cores).  ==1 (the default packing, one rank per core) => force
    # single-threaded Eigen so packed ranks never oversubscribe.  >1 (hybrid:
    # fewer ranks x more cores/rank) => let Eigen multi-thread so each rank uses
    # its allocated cores -- fewer ranks means fewer halo messages, the codex
    # MPI-improve lever, without idling cores.  Honors an explicit OMP override.
    n_thr = int(os.environ.get("SLURM_CPUS_PER_TASK", "1") or "1")
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, str(n_thr))
    if n_thr <= 1:
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_cpu_multi_thread_eigen=false" not in xla_flags:
            xla_flags = f"{xla_flags} --xla_cpu_multi_thread_eigen=false".strip()
        os.environ["XLA_FLAGS"] = xla_flags


def _configure_jax_gpu(precision: str) -> None:
    """Pin THIS MPI rank to one local GPU and run JAX on cuda (route-A).

    Single-node multi-GPU via mpi4jax (the SAME mpi4jax halo machinery as the
    CPU path — make_latlon_mpi_step / cube — just on cuda devices over the
    PCIe pair).  Must run BEFORE any JAX import.  Local rank from the launcher
    env (OpenMPI / SLURM).  Mirrors the ocean harness ``_configure_jax_gpu``
    (bench_ocean_mpi_scaling.py) so the atm lat-lon dycore gets a 2-GPU
    number via the proven overlay-venv route-A (cuda jax + CUDA-built
    mpi4jax)."""
    local = (os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
             or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK"))
    if local is None:
        # SLURM_LOCALID is exported even in a plain sbatch step (ntasks=1, no
        # srun); pinning on it THERE hides all but GPU 0 from a single-process
        # multi-GPU run (the documented silent eff=0.5 bug, jobs 8454397/
        # 8454737). Only honor it for a genuine multi-task launch (codex
        # capstone LOW).
        slid = os.environ.get("SLURM_LOCALID")
        nt = os.environ.get("SLURM_NTASKS", "1")
        if slid is not None and nt.isdigit() and int(nt) > 1:
            local = slid
    if local is None:
        local = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = local   # one GPU per rank
    os.environ["JAX_PLATFORMS"] = "cuda"
    if precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    # Two ranks share the node; do not let the allocator grab the whole GPU.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _init_mpi() -> tuple[int, int]:
    """Initialize MPI and return (rank, n_ranks)."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except ImportError:
        return 0, 1


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class TimingResult:
    n_ranks: int
    resolution: int
    n_levels: int
    precision: str
    mode: str
    grid_type: str
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
    cells_per_rank: int
    mcells_per_s: float
    scaling_efficiency: float = 1.0


@dataclass
class ScalingReport:
    mode: str
    grid_type: str
    physics_level: str
    precisions: list[str]
    results: list[TimingResult]
    timestamp_utc: str = ""
    hostname: str = ""

    def __post_init__(self):
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Grid choices and resolution ladders
# ===========================================================================

GRID_CHOICES = ("cubed-sphere", "latlon", "icosahedral", "spectral")
PHYSICS_CHOICES = ("none", "held_suarez", "gray_sbm", "rrtmg_full", "moist")

# Grid/physics support matrix.  Moist tiers require tracer storage
# that MPAS and spectral states do not have today.
_SUPPORTED_PHYSICS = {
    # Iter 40 honest-sweep: ``_build_physics_fn`` only knows how to
    # construct the Held-Suarez forcing.  ``gray_sbm`` /
    # ``rrtmg_full`` are routed through the AMIP segment driver
    # (``run_levante_gpu_scaling.py`` ``_run_segment_benchmark``),
    # not through this CPU-MPI script.  Listing them as supported
    # here let users pass ``--physics gray_sbm`` and silently
    # benchmark dycore-only with the moist-physics label.
    "cubed-sphere": {"none", "held_suarez"},
    "latlon": {"none", "held_suarez"},
    # "moist" = moisture (q_v/q_c/q_r) + Kessler warm-rain condensation,
    # NO radiation: the moist baroclinic-wave case.  Only wired for
    # icosahedral/MPAS, the sole multi-rank grid here — Kessler is
    # column-local so it adds NO horizontal halo coupling beyond the
    # dycore's tracer exchange, and the dycore already advects tracers
    # mass-consistently (so moist scales on the same ladder as dry).
    "icosahedral": {"none", "held_suarez", "moist"},
    "spectral": {"none", "held_suarez"},
}


def _validate_physics(grid_type: str, physics_level: str) -> None:
    supported = _SUPPORTED_PHYSICS.get(grid_type, set())
    if physics_level not in supported:
        raise ValueError(
            f"Physics level {physics_level!r} not supported for "
            f"grid {grid_type!r}. Supported: {sorted(supported)}."
        )


def _build_physics_fn(physics_level: str, grid_type: str):
    """Build a Held-Suarez physics function for the given grid type."""
    if physics_level != "held_suarez":
        return None
    if grid_type == "spectral":
        from legoesm.atmosphere.held_suarez import held_suarez_forcing_spectral
        return held_suarez_forcing_spectral
    elif grid_type == "latlon":
        from legoesm.atmosphere.held_suarez import held_suarez_forcing_latlon
        return held_suarez_forcing_latlon
    elif grid_type == "icosahedral":
        from legoesm.atmosphere.held_suarez import held_suarez_forcing_mpas
        return held_suarez_forcing_mpas
    else:
        from legoesm.atmosphere.held_suarez import held_suarez_forcing
        return held_suarez_forcing

# Weak scaling base values (constant cells/rank)
WEAK_BASE_CS = 24
WEAK_BASE_LL = 64
WEAK_BASE_ICO = 4  # subdivision level

# Strong scaling resolution sets
STRONG_RES_CS = [24, 48, 96]
STRONG_RES_LL = [64, 128, 256]
STRONG_RES_ICO = [4, 5, 6]
STRONG_RES_SP = [21, 42]


def _weak_resolution_cs(n_ranks: int, base_n: int = WEAK_BASE_CS) -> int:
    n_raw = base_n * math.sqrt(n_ranks)
    return max(4, 2 * round(n_raw / 2))


def _weak_resolution_ll(n_ranks: int, base_n: int = WEAK_BASE_LL) -> int:
    """Weak-scaling n_lat for the lat-lon band decomposition.

    Constant *cells per rank* (the icosahedral analog): total cells
    scale as ``n_lat * n_lon = 2 * n_lat**2``, so ``n_lat ~
    sqrt(n_ranks)`` keeps cells/rank fixed.  Constraints layered on
    top:

    * divisible by ``n_ranks`` (uniform bands → clean cells/rank),
    * at least 2 lat rows per rank — ``pad_halo_latlon_mpi`` raises
      when ``halo(=2 for PPM/biharmonic) > n_lat_local``, so a band
      must never be thinner than the deepest operator halo.

    The divisibility / band-floor rounding means the ACTUAL cells per
    rank drifts between rank counts (it is not exactly ``2 *
    base_n**2 * nlev``).  Each result row records the actual
    ``cells_per_rank`` (``TimingResult.cells_per_rank``, serialized to
    the per-case JSON), and the weak-efficiency computation in
    ``scripts/bench/aggregate_scaling_results.py`` normalizes by it —
    the same actual-cells normalization the icosahedral weak path
    needs for its discrete 4x subdivision-level jumps (see
    ``run_levante_gpu_scaling.run_weak_scaling``).
    """
    n_raw = base_n * math.sqrt(n_ranks)
    n_rounded = max(8, 2 * round(n_raw / 2))
    while n_rounded % n_ranks != 0 or n_rounded < 2 * n_ranks:
        n_rounded += 2
    return n_rounded


def _weak_resolution_ico(n_ranks: int, base_level: int = WEAK_BASE_ICO) -> int:
    base_cells = 10 * 4 ** base_level + 2
    best_level = base_level
    best_ratio = float("inf")
    for lev in range(base_level, 9):
        cells = 10 * 4 ** lev + 2
        cells_per_rank = cells / n_ranks
        ratio = max(cells_per_rank / base_cells, base_cells / cells_per_rank)
        if ratio < best_ratio or (ratio == best_ratio and cells_per_rank >= base_cells):
            best_ratio = ratio
            best_level = lev
    return best_level


def _valid_rank_counts(max_ranks: int, grid_type: str) -> list[int]:
    if grid_type == "spectral":
        return [1]
    if grid_type == "cubed-sphere":
        # Iter 13 honest-sweep guard: cubed-sphere MPI is not yet
        # domain-decomposed — every rank holds the full (6, n, n, ...)
        # state and runs the full dycore.  Multi-rank wall-clock
        # measurements are *not* real weak/strong scaling, just
        # rank-replicated computation plus halo overhead.  Until the
        # halo-side scattered indexing (iter 3) is plumbed through
        # ``model_driver.py`` and the scaling drivers actually scatter
        # per-rank state, restrict cubed-sphere MPI sweeps to rank 1
        # so the summary numbers reflect genuine single-rank
        # throughput rather than replicated-dynamics noise.
        return [1]
    # latlon and icosahedral: powers of 2 up to max
    counts = []
    n = 1
    while n <= max_ranks:
        counts.append(n)
        n *= 2
    return counts


# ===========================================================================
# CFL-safe timestep
# ===========================================================================

def _auto_dt(n_grid: int, grid_type: str) -> float:
    from legoesm import constants  # lazy: see top-of-file note on JAX init order

    R = constants.R_earth
    u_max = 60.0
    c_grav = 300.0
    cfl = 0.7

    if grid_type == "spectral":
        n_lon = 2 * (n_grid + 1)
        dx_min = math.pi * R / n_lon
    elif grid_type == "icosahedral":
        n_cells = 10 * 4 ** n_grid + 2
        dx_avg = R * math.sqrt(4.0 * math.pi / n_cells)
        dx_min = 0.9 * dx_avg
    elif grid_type == "latlon":
        # Pole-cell dx is the limiting spacing on lat-lon grids
        n_lat = n_grid
        dlat = math.pi / n_lat
        dlon = 2.0 * math.pi / (2 * n_lat)
        dx_min = R * dlon * math.cos(math.pi / 2.0 - dlat / 2.0)
    else:  # cubed-sphere
        dx_min = (math.pi / 2) * R / (n_grid * math.sqrt(3))

    dt = cfl * dx_min / (u_max + c_grav)
    # Round down to a "nice" value; no floor — lat-lon pole cells can
    # require sub-second timesteps at very high resolution.
    if dt >= 30.0:
        return 30.0 * int(dt / 30.0)
    elif dt >= 5.0:
        return 5.0 * int(dt / 5.0)
    elif dt >= 1.0:
        return float(int(dt))
    else:
        # Truncate down to 2 decimal places (never round up past CFL)
        return math.floor(dt * 100) / 100


# ===========================================================================
# Build benchmark step functions
# ===========================================================================

def _build_amip_step(
    *,
    grid_type: str,
    resolution: int,
    nlev: int,
    rank: int,
    n_ranks: int,
    precision: str,
    physics_level: str,
    dt: float | None = None,
    cs_spmd: bool = False,
):
    """Build a step function + initial state for one benchmark case.

    Returns (step_fn, state, dt, total_cells, cells_per_rank).
    """
    import jax
    import jax.numpy as jnp

    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if precision == "float64" else jnp.float32

    from legoesm.grids.vertical import create_sigma_coordinate
    sigma = create_sigma_coordinate(nlev)

    if dt is None:
        dt = _auto_dt(resolution, grid_type)

    def _cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    if grid_type == "cubed-sphere":
        if cs_spmd:
            return _build_cubed_sphere_spmd(
                resolution, nlev, dt, dtype, physics_level, _cast)
        return _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                                  physics_level, _cast)
    elif grid_type == "latlon":
        return _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                             physics_level, _cast)
    elif grid_type == "icosahedral":
        return _build_icosahedral(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                                  physics_level, _cast)
    elif grid_type == "spectral":
        return _build_spectral(resolution, nlev, sigma, dt, dtype,
                               physics_level, _cast)
    else:
        raise ValueError(f"Unsupported grid: {grid_type!r}")


def _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                        physics_level, cast_fn):
    import jax
    import jax.numpy as jnp

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
        hydrostatic_to_fv3,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init

    grid = create_cubed_sphere(resolution)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=0.0,
        hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
        zero_mean_ps_tendency=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    state = jax.tree.map(cast_fn, state)

    total_cells = 6 * resolution * resolution * nlev

    # MPI: initialise distributed but keep full (6, n, n, ...) state
    # on every rank, matching the production driver.  pad_halo_mpi
    # requires the full shape; each rank steps all faces and MPI halo
    # exchange keeps owned faces correct.
    if n_ranks > 1:
        from legoesm.parallel.distributed import initialize_distributed
        initialize_distributed(global_n=resolution, grid_type="cubed_sphere")

    physics_fn = _build_physics_fn(physics_level, "cubed-sphere")
    if physics_fn is not None:
        _phys = physics_fn
        step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
    else:
        step_fn = model.step
    cells_per_rank = total_cells // max(1, n_ranks)
    return step_fn, state, dt, total_cells, cells_per_rank


def _build_cubed_sphere_spmd(resolution, nlev, dt, dtype, physics_level,
                             cast_fn):
    """A1 path: TRUE cubed-sphere decomposition via jax.distributed.

    Multi-controller SPMD: the global face mesh is built from
    ``jax.devices()`` (THE multi-controller fix — ``jax.local_devices``
    would give each process a private 1-device mesh), the existing
    ``make_sharded_step`` + multiface-ppermute halo runs unchanged, and
    the state is sharded across the global device set.  Conservation
    (``fix_mass``) reduces on global-sharded arrays at the jnp level —
    SPMD-global by construction (parity receipt 6.7e-10 @5 steps, job
    8462928).  ``jax.distributed.initialize()`` must already have run
    (``main`` does it for ``--cs-spmd`` BEFORE any other JAX use).

    mpi4jax is NEVER armed in this mode: the replicated cubed-sphere
    path's mpi4jax halo machinery and jax.distributed collectives in
    one program is the documented mixed-stack deadlock hazard.
    """
    import jax

    from legoesm.parallel.mesh import create_device_mesh, shard_pytree
    from legoesm.parallel.sharded_dynamics import make_sharded_step
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
        hydrostatic_to_fv3,
        create_cubed_sphere_cdgrid,
    )

    if physics_level != "none":
        raise ValueError(
            "--cs-spmd currently benchmarks the dry dycore only "
            f"(physics={physics_level!r}); physics column scatter under "
            "the SPMD path is the next wiring step."
        )
    gdev = jax.devices()
    n_global = len(gdev)
    _ok = (6 % n_global == 0) if n_global <= 6 else (
        n_global % 6 == 0
        and (round((n_global // 6) ** 0.5)) ** 2 == n_global // 6
    )
    if not _ok:
        raise ValueError(
            f"--cs-spmd needs a device count dividing 6 or 6*kt^2 "
            f"(sub-face tiling), got {n_global} (srun -n "
            f"1|2|3|6|24|54|...).  np=24 parity receipt: 4.4e-10 "
            f"@5 steps, job 8465445."
        )

    cfg_mesh = create_device_mesh(n_devices=n_global, devices=gdev)
    grid = create_cubed_sphere(resolution)
    sigma = create_sigma_coordinate(nlev)
    state = hydrostatic_to_fv3(
        baroclinic_wave_init(grid, sigma, perturbed=True),
        create_cubed_sphere_cdgrid(grid))
    state = jax.tree.map(cast_fn, state)
    # IDENTICAL config to the serial cubed-sphere baseline above —
    # speedup comparisons are meaningless across different dynamics
    # settings.  The conservation fixers reduce via jnp-level global
    # sums on global-sharded arrays (SPMD-global psum by construction;
    # conservation-reduction audit + parity receipt).
    config = CDGridPrimitiveEquationConfig(
        hyperdiff_coeff=0.0,
        hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
        zero_mean_ps_tendency=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)

    step_fn = make_sharded_step(model, cfg_mesh, n=resolution, nlev=nlev)
    state = shard_pytree(state, cfg_mesh)

    total_cells = 6 * resolution * resolution * nlev
    cells_per_rank = total_cells // n_global
    return step_fn, state, dt, total_cells, cells_per_rank


def _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                   physics_level, cast_fn):
    import jax
    import jax.numpy as jnp

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )

    from legoesm.core.cfl import pole_cell_dx, cfl_max_dt

    n_lat = resolution
    n_lon = 2 * resolution
    grid = create_latlon_grid(n_lat, n_lon)

    # Clamp dt to pole-cell CFL limit (computed on the GLOBAL grid so
    # every rank derives the identical dt — only the boundary ranks
    # own the actual pole rows under band MPI).
    dx_pole = pole_cell_dx(grid)
    dt = min(dt, cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1))

    config = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0,
        fix_mass=True,
        use_polar_filter=False,
    )

    # Baroclinic wave init for lat-lon (cell-centred HydrostaticState).
    from tests.test_cases.baroclinic_wave import (
        baroclinic_wave_init_latlon,
    )
    state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True)
    state = jax.tree.map(cast_fn, state)

    total_cells = n_lat * n_lon * nlev

    physics_fn = _build_physics_fn(physics_level, "latlon")

    if n_ranks > 1:
        # Latitude-band MPI (mirrors the multi-rank icosahedral path):
        # 1. convert the global state to raw C-grid arrays *before*
        #    arming the MPI halo backend (the conversion's pole pads
        #    must run on the global array with the local backend),
        # 2. arm the band layout + MPI halo backend,
        # 3. slice the global grid to this rank's band, build the
        #    rank-local model on it,
        # 4. scatter the global state to the band,
        # 5. wrap the step;  ``make_latlon_mpi_step`` forwards
        #    ``physics_fn`` per RK stage exactly like the serial
        #    ``model.step(state, dt, physics_fn=...)`` path
        #    (Held-Suarez is column-local, so it adds no halo
        #    coupling beyond the dycore's own exchanges).
        from legoesm.parallel.distributed import initialize_distributed_latlon
        from legoesm.parallel.latlon_mpi import (
            make_latlon_mpi_step,
            scatter_state_latlon,
            slice_latlon_grid_to_band,
        )

        if n_lat // n_ranks < 2:
            raise ValueError(
                f"lat-lon band MPI needs >=2 lat rows per rank for the "
                f"halo=2 PPM/biharmonic exchange; got n_lat={n_lat} on "
                f"{n_ranks} ranks ({n_lat // n_ranks} rows/rank)."
            )

        # (1) global cell-centred -> global C-grid, still serial.
        cgrid_global = hydrostatic_to_cgrid(state, grid)

        # (2) band layout + MPI halo backend.
        layout = initialize_distributed_latlon(
            global_n_lat=n_lat, global_n_lon=n_lon,
        )

        # (3) rank-local band model (the wrapper re-instantiates it
        # with rank-aware pole_v_bc + allreduced total_area itself).
        band_grid = slice_latlon_grid_to_band(grid, layout)
        local_model = CGridLatLonPrimitiveEquationModel(
            band_grid, sigma, config, dt=dt,
        )

        # (4) + (5)
        state = scatter_state_latlon(cgrid_global, layout)
        step_fn = make_latlon_mpi_step(
            local_model, layout, physics_fn=physics_fn,
        )
        # ACTUAL rank-local cell count (Codex P1-fix review, MINOR):
        # ``total // n_ranks`` is only exact when n_lat divides evenly
        # (the sweep generator enforces that, a direct ``--resolution``
        # run does not — the first ``n_lat % n_ranks`` ranks then carry
        # one extra row).  Rank 0 is in that first group, so its count
        # is the bottleneck-rank load — the right weak-scaling
        # normalizer for the rank-0-written result JSON.
        cells_per_rank = layout.n_lat_local * n_lon * nlev
    else:
        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
        if physics_fn is not None:
            _phys = physics_fn
            step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
        else:
            step_fn = model.step
        cells_per_rank = total_cells

    return step_fn, state, dt, total_cells, cells_per_rank


def _build_icosahedral(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                        physics_level, cast_fn):
    import jax
    import jax.numpy as jnp

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
        MPASPrimitiveEquationConfig,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    mesh = create_voronoi_mesh(subdivision_level=resolution)
    # Mass fixer adds one global allreduce per step (267.8 us latency floor on
    # Ginsburg/Gloo). LEGOESM_NO_MASS_FIX=1 disables it for a scaling ABLATION
    # that isolates the dynamics+halo cost from the conservation allreduce
    # (codex MPI-improve #3). Production keeps it ON (conservation).
    _fix_mass = os.environ.get("LEGOESM_NO_MASS_FIX") != "1"
    config = MPASPrimitiveEquationConfig(
        nu_del4=0.0,
        nu_del4_ps=0.0,
        fix_mass=_fix_mass,
        time_integrator="ssp_rk3",
    )
    model = MPASPrimitiveEquationModel(mesh, sigma, config)
    _moist = physics_level == "moist"
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True, moist=_moist)
    state = jax.tree.map(cast_fn, state)

    total_cells = mesh.nCells * nlev

    if _moist:
        # Kessler warm-rain forcing bound to this step's dt (the dycore's
        # operator-split physics_fn convention passes no timestep).  Column-
        # local ⇒ no extra halo; applied once per step over dt.
        from legoesm.atmosphere.kessler_forcing import make_kessler_forcing_mpas
        physics_fn = make_kessler_forcing_mpas(dt)
    else:
        physics_fn = _build_physics_fn(physics_level, "icosahedral")

    if n_ranks > 1:
        # ``make_voronoi_mpi_step`` now forwards ``physics_fn`` via the
        # operator-split path (iter: HS MPI scaling), so multi-rank
        # icosahedral sweeps with ``--physics held_suarez`` benchmark the
        # genuine dynamics+physics step.  Held-Suarez is column-local
        # (Newtonian relaxation), so it adds no horizontal halo coupling
        # beyond the dycore's exchange.
        from legoesm.parallel.voronoi_mpi import (
            make_voronoi_partition_layout,
            scatter_state_voronoi,
            make_voronoi_mpi_step,
        )
        # Partition method A/B (audit #3): LEGOESM_VORONOI_PARTITION =
        # "geometric" (RCB, default) | "metis" (pymetis k-way edge-cut min).
        _pmethod = os.environ.get("LEGOESM_VORONOI_PARTITION", "geometric")
        layout = make_voronoi_partition_layout(mesh, rank, n_ranks,
                                               method=_pmethod)
        state = scatter_state_voronoi(state, layout.partition)
        step_fn = make_voronoi_mpi_step(
            model, layout, sigma, config, physics_fn=physics_fn,
        )
    else:
        if physics_fn is not None:
            _phys = physics_fn
            step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
        else:
            step_fn = model.step

    cells_per_rank = total_cells // max(1, n_ranks)
    return step_fn, state, dt, total_cells, cells_per_rank


def _build_spectral(resolution, nlev, sigma, dt, dtype, physics_level, cast_fn):
    import jax
    import jax.numpy as jnp

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel,
        SpectralPEConfig,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral

    grid = create_gaussian_grid(resolution)
    config = SpectralPEConfig(
        hyperdiff_coeff=0.0,
        hyperdiff_order=4,
        spectral_filter_strength=0.01,
        spectral_filter_order=8,
        time_integrator="ssp_rk3",
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config)
    state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)
    state = jax.tree.map(cast_fn, state)

    total_cells = grid.n_lat * grid.n_lon * nlev

    physics_fn = _build_physics_fn(physics_level, "spectral")
    if physics_fn is not None:
        _phys = physics_fn
        step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
    else:
        step_fn = model.step

    return step_fn, state, dt, total_cells, total_cells


# ===========================================================================
# Core benchmark runner
# ===========================================================================

def run_single_benchmark(
    *,
    grid_type: str,
    resolution: int,
    nlev: int,
    n_ranks: int,
    rank: int,
    precision: str,
    mode: str,
    physics_level: str,
    n_warmup: int,
    n_timing: int,
    dt: float | None = None,
    cs_spmd: bool = False,
) -> TimingResult:
    """Run a single benchmark case and return timing."""
    _validate_physics(grid_type, physics_level)

    import jax
    import jax.numpy as jnp

    step_fn, state, dt_used, total_cells, cells_per_rank = _build_amip_step(
        grid_type=grid_type,
        resolution=resolution,
        nlev=nlev,
        rank=rank,
        n_ranks=n_ranks,
        precision=precision,
        physics_level=physics_level,
        dt=dt,
        cs_spmd=cs_spmd,
    )

    if grid_type == "spectral":
        res_label = f"T{resolution}"
    elif grid_type == "icosahedral":
        res_label = f"I{resolution}"
    elif grid_type == "latlon":
        res_label = f"LL{resolution}"
    else:
        res_label = f"C{resolution}"

    if rank == 0:
        print(
            f"  [{precision}] {res_label}/L{nlev} on {n_ranks} rank(s) | "
            f"dt={dt_used:.0f}s | cells={total_cells:,} | "
            f"cells/rank={cells_per_rank:,} | physics={physics_level}",
            flush=True,
        )

    # --- JIT compilation ---
    t_compile_start = time.perf_counter()
    state = step_fn(state, dt_used)
    jax.block_until_ready(jax.tree.leaves(state))
    compile_time = time.perf_counter() - t_compile_start
    if rank == 0:
        print(f"    JIT compile: {compile_time:.2f}s", flush=True)

    # --- Warmup ---
    t_warmup_start = time.perf_counter()
    for _ in range(n_warmup):
        state = step_fn(state, dt_used)
    jax.block_until_ready(jax.tree.leaves(state))
    warmup_time = time.perf_counter() - t_warmup_start

    # --- Timed steps via lax.scan ---
    input_dtypes = jax.tree.map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state)

    @jax.jit
    def _scan_run(st, dt_val):
        def _body(carry, _):
            new = step_fn(carry, dt_val)
            new = jax.tree.map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes,
            )
            return new, None
        return jax.lax.scan(_body, st, None, length=n_timing)[0]

    # Pre-compile scan without mutating the state used for timing.
    # Previously we rebound ``state`` to the precompile output, so the
    # timed run started from state already advanced by ``n_timing``
    # steps and the benchmark was biased.  Use a leaf-cloned input so
    # XLA still warms compile + caches against identical layout but
    # ``state`` keeps its original (post-warmup) trajectory.
    _precompile_state = jax.tree.map(lambda x: x, state)
    _precompile_out = _scan_run(_precompile_state, dt_used)
    jax.block_until_ready(jax.tree.leaves(_precompile_out))

    # MPI barrier before timing
    try:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.Get_size() > 1:
            jax.block_until_ready(jax.tree.leaves(state))
            _MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass

    t0 = time.perf_counter()
    state = _scan_run(state, dt_used)
    jax.block_until_ready(jax.tree.leaves(state))

    # MPI barrier after timing
    try:
        from mpi4py import MPI as _MPI
        if _MPI.COMM_WORLD.Get_size() > 1:
            _MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass

    t1 = time.perf_counter()

    timing_time = t1 - t0
    time_per_step = timing_time / n_timing
    time_per_step_ms = time_per_step * 1000.0
    sypd = (dt_used / time_per_step) / (365.25 * 86400) * 86400.0
    mcells_per_s = (total_cells / time_per_step) / 1e6

    if rank == 0:
        print(
            f"    Timing: {time_per_step_ms:.2f} ms/step | "
            f"SYPD={sypd:.3f} | {mcells_per_s:.1f} Mcells/s",
            flush=True,
        )

    return TimingResult(
        n_ranks=n_ranks,
        resolution=resolution,
        n_levels=nlev,
        precision=precision,
        mode=mode,
        grid_type=grid_type,
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
        cells_per_rank=cells_per_rank,
        mcells_per_s=mcells_per_s,
    )


# ===========================================================================
# Sweep mode: generate case list
# ===========================================================================

@dataclass
class CaseSpec:
    grid: str
    resolution: int
    n_ranks: int
    mode: str
    physics: str
    nlev: int
    precision: str


def generate_sweep_cases(
    grid_type: str,
    mode: str,
    physics: str,
    max_ranks: int,
    nlev: int = 26,
    precision: str = "float64",
) -> list[CaseSpec]:
    """Generate all benchmark cases for a sweep."""
    cases = []
    rank_counts = _valid_rank_counts(max_ranks, grid_type)

    # icosahedral MPI multi-rank now applies ``physics_fn`` via the
    # operator-split path in ``make_voronoi_mpi_step``, so Held-Suarez
    # (column-local) multi-rank sweeps benchmark the genuine
    # dynamics+physics step.  No rank-1 restriction needed.

    if mode in ("weak", "both"):
        for n in rank_counts:
            if grid_type == "cubed-sphere":
                res = _weak_resolution_cs(n)
            elif grid_type == "latlon":
                res = _weak_resolution_ll(n)
            elif grid_type == "icosahedral":
                res = _weak_resolution_ico(n)
            else:
                res = 42
            cases.append(CaseSpec(grid_type, res, n, "weak", physics, nlev, precision))

    if mode in ("strong", "both"):
        if grid_type == "cubed-sphere":
            resolutions = STRONG_RES_CS
        elif grid_type == "latlon":
            resolutions = STRONG_RES_LL
        elif grid_type == "icosahedral":
            resolutions = STRONG_RES_ICO
        else:
            resolutions = STRONG_RES_SP

        for res in resolutions:
            for n in rank_counts:
                # Lat-lon bands: uniform decomposition (divisible) AND
                # >=2 lat rows per rank (halo=2 exchange minimum).
                if grid_type == "latlon" and (res % n != 0 or res // n < 2):
                    continue
                cases.append(CaseSpec(grid_type, res, n, "strong", physics, nlev, precision))

    return cases


# ===========================================================================
# Output
# ===========================================================================

def write_result_json(result: TimingResult, output_dir: Path) -> None:
    """Write a single result as a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"{result.grid_type}_{result.physics_level}_{result.mode}_"
        f"r{result.resolution}_n{result.n_ranks}_{result.precision}.json"
    )
    path = output_dir / fname
    payload = asdict(result)
    # Record the actual JAX backend so downstream aggregation does not have to
    # infer CPU-vs-GPU from the output-dir name (codex review): cpu/gpu/tpu.
    try:
        import jax
        payload["backend"] = jax.default_backend()
    except Exception:
        payload["backend"] = ""
    # Record the hybrid layout so scaling can be plotted vs CORES, not ranks:
    # a hybrid 8r x 4c run and a packed 32r x 1c run both report n_ranks but use
    # 32 vs 128 cores. cpus_per_task * n_ranks = the true resource count.
    _cpt = int(os.environ.get("SLURM_CPUS_PER_TASK", "1") or "1")
    payload["cpus_per_task"] = _cpt
    payload["n_cores"] = result.n_ranks * _cpt
    # Record conservation mode so a LEGOESM_NO_MASS_FIX ablation never dedups
    # with / is mislabeled as a production (mass-conserving) run (codex audit).
    payload["fix_mass"] = os.environ.get("LEGOESM_NO_MASS_FIX") != "1"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Result: {path}")


def print_summary(result: TimingResult) -> None:
    """Print a one-line summary."""
    print(
        f"  {result.grid_type:>14s} | {result.physics_level:>12s} | "
        f"{result.n_ranks:>5d} ranks | {result.resolution:<5d} | "
        f"L{result.n_levels:>2d} | dt={result.dt_seconds:>6.0f} | "
        f"{result.time_per_step_ms:>9.2f} ms/step | "
        f"SYPD={result.sypd:>8.3f} | {result.mcells_per_s:>9.1f} Mcells/s"
    )


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPU MPI scaling benchmark for legoESM AMIP-like runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grid", choices=list(GRID_CHOICES), default="latlon",
        help="Grid / dycore type.",
    )
    p.add_argument(
        "--resolution", type=int, default=0,
        help="Grid resolution (0 = auto from mode/ranks).",
    )
    p.add_argument(
        "--physics", choices=list(PHYSICS_CHOICES), default="held_suarez",
        help="Physics complexity level.",
    )
    p.add_argument(
        "--mode", choices=["weak", "strong", "both", "single"], default="single",
        help="Scaling mode: weak, strong, both, or single (one case).",
    )
    p.add_argument(
        "--precision", choices=["float32", "float64"], default="float64",
        help="Floating-point precision.",
    )
    p.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="cpu (default, CPU-MPI) or gpu (route-A: pin each rank to one "
             "local GPU, run the SAME mpi4jax dycore on the PCIe pair). "
             "Needs the overlay venv (cuda jax + CUDA-built mpi4jax).",
    )
    p.add_argument("--n-levels", type=int, default=26)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-timing", type=int, default=50)
    p.add_argument(
        "--cs-spmd", action="store_true",
        help="Cubed-sphere TRUE domain decomposition via jax.distributed "
             "multi-controller SPMD (global face mesh + multiface "
             "ppermute; the A1 path).  Replaces the replicated-dynamics "
             "refusal: launch with srun -n {2,3,6} (must divide 6).  "
             "Uses jax.distributed ONLY — the mpi4jax halo backend is "
             "never armed in this mode (mixed stacks deadlock).  "
             "Parity receipt: scripts/tmp/_probe_spmd_cube_parity.py "
             "(shard-local vs serial = 6.7e-10 @5 steps, job 8462928).",
    )
    p.add_argument(
        "--output-dir", type=str, default="results/cpu_scaling",
        help="Output directory for results.",
    )
    p.add_argument(
        "--sweep", action="store_true",
        help="Generate case list (JSON lines) and exit. No execution.",
    )
    p.add_argument(
        "--max-ranks", type=int, default=64,
        help="Maximum rank count for sweep mode.",
    )
    p.add_argument(
        "--case", type=str, default=None,
        help="JSON case spec (from sweep output).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    # --- Sweep mode: just print cases and exit ---
    if args.sweep:
        # Iter 41: validate the grid+physics combo *before* the
        # sweep generator runs, so unsupported tiers (e.g. moist
        # physics on cubed-sphere/lat-lon CPU MPI, see iter 40)
        # error out immediately with a clear message instead of
        # producing a JSON-lines list that subsequently fails at
        # runtime under the SLURM array.
        _validate_physics(args.grid, args.physics)
        cases = generate_sweep_cases(
            args.grid, args.mode if args.mode != "single" else "both",
            args.physics, args.max_ranks,
            args.n_levels, args.precision,
        )
        for c in cases:
            print(json.dumps(asdict(c)))
        return 0

    # --- Configure JAX for the target device ---
    if getattr(args, "device", "cpu") == "gpu":
        _configure_jax_gpu(args.precision)
        # Fail LOUD on CUDA fallback: without this, a GPU job whose CUDA init
        # failed (or that forgot --device gpu so _configure_jax_cpu pinned
        # JAX_PLATFORMS=cpu) silently records CPU numbers labeled as GPU
        # (bug: the whole g1..g16 ladder ran on CPU).  Refuse to mislabel.
        import jax as _jax
        _bk = _jax.default_backend()
        if _bk != "gpu":
            raise SystemExit(
                f"--device gpu requested but JAX default backend is {_bk!r} "
                f"(CUDA unavailable / not bound). Refusing to record "
                f"CPU-fallback numbers as GPU. Check CUDA_VISIBLE_DEVICES / "
                f"the cuda jax plugin on this node."
            )
    else:
        _configure_jax_cpu(args.precision)

    # --- A1 SPMD mode: federate processes into ONE multi-controller JAX
    # program BEFORE any other JAX use.  jax.distributed only — the
    # mpi4jax halo backend is never armed on this path (mixed stacks
    # deadlock; see --cs-spmd help).  Single-process launches skip the
    # init (jax.distributed requires a real multi-process environment).
    if args.cs_spmd:
        # Resolve the REQUESTED grid before touching jax.distributed:
        # every non-cubed-sphere builder arms the mpi4jax halo backend,
        # and mpi4jax + jax.distributed collectives in one program is
        # the documented mixed-stack deadlock.  (--case is parsed again
        # below; this early peek only needs the grid field.)
        _grid_req = (
            json.loads(args.case).get("grid", args.grid)
            if args.case else args.grid
        )
        if _grid_req != "cubed-sphere":
            print(
                f"ERROR: --cs-spmd supports only --grid cubed-sphere "
                f"(got {_grid_req!r}); lat-lon/icosahedral paths arm "
                "mpi4jax, which must never coexist with "
                "jax.distributed in one program.",
                flush=True,
            )
            return 2
        import jax as _jax
        import os as _os
        # Launcher-agnostic process count: SLURM (srun) or OpenMPI
        # (mpirun) — gating on SLURM_NTASKS alone would silently skip
        # initialize() under mpirun and leave N independent local
        # meshes all reporting n_ranks=N.
        _nproc = int(_os.environ.get(
            "SLURM_NTASKS", _os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
        if _nproc > 1:
            _jax.distributed.initialize()

    # --- MPI init ---
    rank, n_ranks = _init_mpi()
    is_rank0 = (rank == 0)

    # cs-spmd consistency gate: every launched process must have joined
    # ONE multi-controller program.  A mismatch means initialize() was
    # skipped (unknown launcher) or partially failed — measuring would
    # produce N independent serial runs labelled n_ranks=N.
    if args.cs_spmd:
        import jax as _jax
        if n_ranks > 1 and _jax.process_count() != n_ranks:
            if is_rank0:
                print(
                    f"ERROR: --cs-spmd launched with {n_ranks} MPI "
                    f"processes but jax.process_count()="
                    f"{_jax.process_count()} — jax.distributed did not "
                    "federate them (unsupported launcher?).  Refusing "
                    "to record replicated-serial numbers.",
                    flush=True,
                )
            return 3
        if n_ranks == 1 and is_rank0:
            print(
                "NOTE: --cs-spmd with a single process = sharded-on-1-"
                "device, NOT multi-controller SPMD; use the no-flag "
                "serial path for baselines.",
                flush=True,
            )

    # --- Parse case spec if provided ---
    if args.case:
        case = json.loads(args.case)
        grid_type = case["grid"]
        resolution = case["resolution"]
        physics_level = case["physics"]
        mode = case["mode"]
        nlev = case.get("nlev", args.n_levels)
        precision = case.get("precision", args.precision)
    else:
        grid_type = args.grid
        resolution = args.resolution
        physics_level = args.physics
        mode = args.mode
        nlev = args.n_levels
        precision = args.precision

    # Auto-resolve resolution
    if resolution == 0:
        if grid_type == "cubed-sphere":
            resolution = _weak_resolution_cs(n_ranks) if mode == "weak" else 48
        elif grid_type == "latlon":
            resolution = _weak_resolution_ll(n_ranks) if mode == "weak" else 64
        elif grid_type == "icosahedral":
            resolution = _weak_resolution_ico(n_ranks) if mode == "weak" else 5
        else:
            resolution = 42

    # Non-domain-decomposed grids only support 1 rank in this driver.
    if grid_type == "spectral" and n_ranks > 1:
        if is_rank0:
            print("ERROR: Spectral grid does not support MPI. Use 1 rank.")
        return 1
    # Iter 14 follow-up: defensive runtime guard for cubed-sphere MPI.
    # The iter 13 ``_valid_rank_counts`` guard prevents the sweep from
    # *generating* multi-rank cases, but a user could still pass an
    # explicit ``--case`` with ``n_ranks>1`` or invoke the script under
    # ``mpirun -np N`` with ``--mode single``.  Refuse the
    # configuration up-front instead of silently capturing replicated-
    # dynamics numbers.
    if grid_type == "cubed-sphere" and n_ranks > 1 and not args.cs_spmd:
        if is_rank0:
            print(
                "ERROR: cubed-sphere MPI is currently replicated-"
                "dynamics-only (every rank holds full state).  Use 1 "
                "rank or --grid icosahedral for genuine MPI scaling.",
                flush=True,
            )
        return 2
    # Lat-lon band MPI is real (latitude-band decomposition), but a
    # band must hold at least 2 lat rows for the halo=2 PPM /
    # biharmonic exchange (``pad_halo_latlon_mpi`` raises when
    # ``halo > n_lat_local``).  Refuse undersized configurations
    # up-front with a clear message instead of a mid-build traceback.
    if grid_type == "latlon" and n_ranks > 1 and resolution // n_ranks < 2:
        if is_rank0:
            print(
                f"ERROR: lat-lon band MPI needs >=2 lat rows per rank "
                f"(halo=2 exchange); resolution={resolution} on "
                f"{n_ranks} ranks gives {resolution // n_ranks} "
                f"rows/rank.  Increase --resolution or reduce ranks.",
                flush=True,
            )
        return 2

    if is_rank0:
        print("=" * 72)
        print("  legoESM CPU MPI Scaling Benchmark")
        print("=" * 72)
        print(f"  Grid:      {grid_type}")
        print(f"  Physics:   {physics_level}")
        print(f"  Precision: {precision}")
        print(f"  Ranks:     {n_ranks}")
        print(f"  Resolution:{resolution}")
        print(f"  Levels:    {nlev}")
        print(f"  Mode:      {mode}")
        print("=" * 72)

    result = run_single_benchmark(
        grid_type=grid_type,
        resolution=resolution,
        nlev=nlev,
        n_ranks=n_ranks,
        rank=rank,
        precision=precision,
        mode=mode,
        physics_level=physics_level,
        n_warmup=args.n_warmup,
        n_timing=args.n_timing,
        cs_spmd=bool(args.cs_spmd),
    )

    if is_rank0:
        print_summary(result)
        output_dir = Path(args.output_dir) / f"{grid_type}_{physics_level}_{mode}"
        write_result_json(result, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
