#!/usr/bin/env python
"""CPU MPI scaling benchmark for AMIP-like runs.

Measures wall-clock time per step and SYPD across varying MPI rank counts
and resolutions for four grid types and three physics levels.

Supported grids (MPI-scalable):
  cubed-sphere  -- C-D grid + FV3 PE dycore
  latlon        -- Lat-lon FV PE dycore
  icosahedral   -- MPAS Voronoi TRiSK PE dycore

Single-rank baseline:
  spectral      -- Gaussian + spectral PE dycore (no MPI)

Physics levels:
  held_suarez   -- Newtonian relaxation (cheapest, no I/O)
  gray_sbm      -- Gray radiation + SBM convection
  rrtmg_full    -- RRTMG radiation + Kessler microphysics + SBM

Design: one benchmark case per MPI process invocation (see F1 in plan).
Use --sweep to generate all cases for a SLURM array job.

Usage
-----
Single case::

    mpirun -np 4 python scripts/run_cpu_mpi_scaling.py \\
        --grid latlon --resolution 64 --physics held_suarez

Sweep mode (generate case list, no execution)::

    python scripts/run_cpu_mpi_scaling.py --sweep \\
        --grid icosahedral --mode strong --physics held_suarez

From JSON case spec::

    mpirun -np 6 python scripts/run_cpu_mpi_scaling.py \\
        --case '{"grid":"cubed-sphere","resolution":48,...}'
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

# ---------------------------------------------------------------------------
# JAX configuration -- must happen before JAX import
# ---------------------------------------------------------------------------

def _configure_jax_cpu(precision: str) -> None:
    """Force JAX to CPU, set precision, pin each rank to 1 thread."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    if precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Pin each MPI rank to a single CPU thread
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_cpu_multi_thread_eigen=false" not in xla_flags:
        xla_flags = f"{xla_flags} --xla_cpu_multi_thread_eigen=false".strip()
    os.environ["XLA_FLAGS"] = xla_flags


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
PHYSICS_CHOICES = ("none", "held_suarez", "gray_sbm", "rrtmg_full")

# Grid/physics support matrix.  Moist tiers require tracer storage
# that MPAS and spectral states do not have today.
_SUPPORTED_PHYSICS = {
    "cubed-sphere": {"none", "held_suarez", "gray_sbm", "rrtmg_full"},
    "latlon": {"none", "held_suarez", "gray_sbm", "rrtmg_full"},
    "icosahedral": {"none", "held_suarez"},
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
    n_raw = base_n * math.sqrt(n_ranks)
    n_rounded = max(8, 2 * round(n_raw / 2))
    # Must be divisible by n_ranks
    while n_rounded % n_ranks != 0:
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
    if grid_type == "latlon":
        # Lat-lon MPI step raises NotImplementedError (the latitude-band
        # decomposition infrastructure exists but the C-grid operators
        # have not been adapted to local domains).  Iter 1 stripped
        # lat-lon from the GPU sweep; iter 5 mirrors that here so the
        # CPU MPI driver does not generate multi-rank cases that
        # immediately error out and pollute the sweep summary.  See #115.
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
    R = 6.371229e6
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


def _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
                   physics_level, cast_fn):
    import jax
    import jax.numpy as jnp

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
    )

    from legoesm.core.cfl import pole_cell_dx, cfl_max_dt

    n_lat = resolution
    n_lon = 2 * resolution
    grid = create_latlon_grid(n_lat, n_lon)

    # Clamp dt to pole-cell CFL limit
    dx_pole = pole_cell_dx(grid)
    dt = min(dt, cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1))

    config = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0,
        fix_mass=True,
        use_polar_filter=False,
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)

    # Baroclinic wave init for lat-lon
    from tests.test_cases.baroclinic_wave import (
        baroclinic_wave_init_latlon,
    )
    state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True)
    state = jax.tree.map(cast_fn, state)

    total_cells = n_lat * n_lon * nlev

    physics_fn = _build_physics_fn(physics_level, "latlon")

    if n_ranks > 1:
        from legoesm.parallel.latlon_mpi import make_latlon_mpi_step
        # make_latlon_mpi_step raises NotImplementedError — lat-lon
        # MPI local-compute requires operator adaptation not yet done.
        make_latlon_mpi_step(model, grid, None, sigma, config)
    else:
        if physics_fn is not None:
            _phys = physics_fn
            step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
        else:
            step_fn = model.step

    cells_per_rank = total_cells // max(1, n_ranks)
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
    config = MPASPrimitiveEquationConfig(
        nu_del4=0.0,
        nu_del4_ps=0.0,
        fix_mass=True,
        time_integrator="ssp_rk3",
    )
    model = MPASPrimitiveEquationModel(mesh, sigma, config)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    state = jax.tree.map(cast_fn, state)

    total_cells = mesh.nCells * nlev

    physics_fn = _build_physics_fn(physics_level, "icosahedral")

    if n_ranks > 1:
        from legoesm.parallel.voronoi_mpi import (
            make_voronoi_partition_layout,
            scatter_state_voronoi,
            make_voronoi_mpi_step,
        )
        layout = make_voronoi_partition_layout(mesh, rank, n_ranks)
        state = scatter_state_voronoi(state, layout.partition)
        step_fn = make_voronoi_mpi_step(model, layout, sigma, config)
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
                if grid_type == "latlon" and res % n != 0:
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
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
    p.add_argument("--n-levels", type=int, default=26)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-timing", type=int, default=50)
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
        cases = generate_sweep_cases(
            args.grid, args.mode if args.mode != "single" else "both",
            args.physics, args.max_ranks,
            args.n_levels, args.precision,
        )
        for c in cases:
            print(json.dumps(asdict(c)))
        return 0

    # --- Configure JAX for CPU ---
    _configure_jax_cpu(args.precision)

    # --- MPI init ---
    rank, n_ranks = _init_mpi()
    is_rank0 = (rank == 0)

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
    if grid_type == "cubed-sphere" and n_ranks > 1:
        if is_rank0:
            print(
                "ERROR: cubed-sphere MPI is currently replicated-"
                "dynamics-only (every rank holds full state).  Use 1 "
                "rank or --grid icosahedral for genuine MPI scaling.",
                flush=True,
            )
        return 2
    if grid_type == "latlon" and n_ranks > 1:
        if is_rank0:
            print(
                "ERROR: lat-lon MPI step is not implemented "
                "(``make_latlon_mpi_step`` raises NotImplementedError, "
                "see #115).  Use 1 rank.",
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
    )

    if is_rank0:
        print_summary(result)
        output_dir = Path(args.output_dir) / f"{grid_type}_{physics_level}_{mode}"
        write_result_json(result, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
