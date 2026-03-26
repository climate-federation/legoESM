#!/usr/bin/env python
"""GPU scaling benchmark for Levante (DKRZ) -- replicates CliMA Figures 11 & 12.

Runs the Jablonowski-Williamson baroclinic wave test case, measuring
wall-clock time per step and SYPD across varying GPU counts and resolutions.

Supported grids:
  spectral      -- Gaussian grid + spectral PE dycore (default)
  cubed-sphere  -- Cubed-sphere C-D grid + FV3 PE dycore
  icosahedral   -- MPAS Voronoi mesh + TRiSK PE dycore

Two modes:
  weak   -- fix problem size per GPU, increase resolution with GPU count
            (replicates Yatunin et al. 2026 JAMES Figure 11 left panel)
  strong -- fix total problem size, increase GPU count
            (replicates Yatunin et al. 2026 JAMES Figure 12 left panel)

Usage
-----
Single-node (4 A100s)::

    python scripts/run_levante_gpu_scaling.py --mode weak --precision float32
    python scripts/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong --precision both

Multi-node via MPI (set up by the companion SLURM script)::

    mpirun -np 8 python scripts/run_levante_gpu_scaling.py \\
        --mode strong --precision float64 --n-gpus 8

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

# ---------------------------------------------------------------------------
# JAX configuration -- must happen before jax import
# ---------------------------------------------------------------------------

def _configure_jax(precision: str) -> None:
    """Set JAX env vars before import."""
    if precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "gpu,cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")


# ===========================================================================
# Dataclasses for results
# ===========================================================================

@dataclass
class TimingResult:
    n_gpus: int
    resolution: int
    n_levels: int
    precision: str
    mode: str
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
    scaling_efficiency: float = 1.0


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


GRID_CHOICES = ("spectral", "cubed-sphere", "icosahedral")

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

def _weak_resolution_ico(n_gpus: int, base_level: int = WEAK_SCALING_BASE_LEVEL_ICO) -> int:
    """Compute icosahedral subdivision level for weak scaling."""
    base_cells = 10 * 4 ** base_level + 2
    target_cells = base_cells * n_gpus
    # Find level whose cell count is closest to target
    best_level = base_level
    for lev in range(2, 10):
        cells = 10 * 4 ** lev + 2
        if cells >= target_cells * 0.7:
            best_level = lev
            break
    return best_level

# Strong scaling: fixed resolutions, sweep GPU counts.
STRONG_RESOLUTIONS_CS = [48, 96, 192]   # cubed-sphere: ~200, ~100, ~50 km
STRONG_RESOLUTIONS_SP = [42, 85, 170]   # spectral: T42, T85, T170
STRONG_RESOLUTIONS_ICO = [4, 5, 6]      # icosahedral: levels 4, 5, 6

# GPU counts to sweep (must satisfy cubed-sphere tiling constraints).
GPU_COUNTS = [1, 2, 3, 6, 24, 54, 96]  # 1-6 divide faces; >6 must be 6*k^2


def _valid_gpu_counts(max_gpus: int, grid_type: str = "cubed-sphere") -> list[int]:
    """Return valid GPU counts up to max_gpus for the given grid type.

    Cubed-sphere requires divisors of 6 (face sharding) or 6*k^2 (tiling).
    Icosahedral and spectral grids support any GPU count.
    """
    if grid_type in ("icosahedral", "spectral"):
        return list(range(1, max_gpus + 1))

    # Cubed-sphere constraints
    valid = []
    # face-only: divisors of 6
    for n in [1, 2, 3, 6]:
        if n <= max_gpus:
            valid.append(n)
    # sub-face tiling: 6*k^2
    k = 2
    while True:
        n = 6 * k * k
        if n > max_gpus:
            break
        valid.append(n)
        k += 1
    return sorted(set(valid))


# ===========================================================================
# CFL-safe timestep
# ===========================================================================

def _auto_dt(n_grid: int, grid_type: str = "cubed-sphere") -> float:
    """Choose a CFL-safe timestep for the hydrostatic PE at resolution n_grid.

    For a jet speed ~50 m/s and the PE semi-implicit scheme, the advective
    CFL constraint is dt < 0.8 * dx_min / u_max.  We use a conservative
    estimate.
    """
    R = 6.371229e6
    if grid_type == "spectral":
        # Gaussian grid: dx_min ~ pi * R / n_lon at equator, n_lon = 2*(n_max+1)
        n_lon = 2 * (n_grid + 1)
        dx_min = math.pi * R / n_lon
    elif grid_type == "icosahedral":
        # Voronoi SCVT: n_grid is subdivision level, nCells = 10*4^level + 2
        n_cells = 10 * 4 ** n_grid + 2
        dx_avg = R * math.sqrt(4.0 * math.pi / n_cells)
        dx_min = 0.9 * dx_avg
    else:
        # Cubed sphere: dx_min ~ (pi/2) * R / (n * sqrt(3))
        dx_min = (math.pi / 2) * R / (n_grid * math.sqrt(3))
    u_max = 60.0
    cfl = 0.7
    dt = cfl * dx_min / u_max
    # Round down to a nice number
    dt = max(30.0, 30.0 * int(dt / 30.0))
    return dt


# ===========================================================================
# Hyperdiffusion scaling
# ===========================================================================

def _hyperdiff_coeff(n_grid: int, grid_type: str = "cubed-sphere") -> float:
    """Scale \\nabla^4 hyperdiffusion coefficient with resolution."""
    if grid_type == "spectral":
        ref_n = 42
        ref_coeff = 2.5e16
    elif grid_type == "icosahedral":
        # For icosahedral, n_grid is a subdivision level.  Scale the
        # coefficient with dx^4 relative to level 5 (~120 km).
        R = 6.371229e6
        ref_cells = 10 * 4 ** 5 + 2
        cur_cells = 10 * 4 ** n_grid + 2
        dx_ref = R * math.sqrt(4.0 * math.pi / ref_cells)
        dx_cur = R * math.sqrt(4.0 * math.pi / cur_cells)
        ref_coeff = 5e16
        return ref_coeff * (dx_cur / dx_ref) ** 4
    else:
        ref_n = 48
        ref_coeff = 5e16
    return ref_coeff * (ref_n / n_grid) ** 4


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
) -> TimingResult:
    """Run the baroclinic wave dycore benchmark and return timing results."""

    import jax
    import jax.numpy as jnp

    # Set precision
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if precision == "float64" else jnp.float32

    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.mesh import (
        create_device_mesh, create_voronoi_device_mesh, shard_pytree,
    )

    # Choose timestep
    if dt is None:
        dt = _auto_dt(n_grid, grid_type)

    sigma = create_sigma_coordinate(n_levels)

    if grid_type == "spectral":
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel,
            SpectralPEConfig,
            baroclinic_wave_init_spectral,
        )

        grid = create_gaussian_grid(n_grid)
        hd = _hyperdiff_coeff(n_grid, grid_type)
        config = SpectralPEConfig(
            hyperdiff_coeff=hd,
            hyperdiff_order=4,
            spectral_filter_strength=0.01,
            spectral_filter_order=8,
            time_integrator="ssp_rk3",
        )
        model = SpectralPrimitiveEquationModel(grid, sigma, config)
        state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)

        n_lat = grid.n_lat
        n_lon = grid.n_lon
        total_cells = n_lat * n_lon * n_levels
        dev_config = create_device_mesh(n_devices=n_gpus)
    elif grid_type == "icosahedral":
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
            MPASPrimitiveEquationConfig,
        )
        from legoesm.atmosphere.physics.held_suarez_mpas import baroclinic_wave_init_mpas

        grid = create_voronoi_mesh(subdivision_level=n_grid)

        # Reorder mesh for spatial locality when sharding across GPUs.
        if n_gpus > 1:
            from legoesm.parallel.voronoi_partition import (
                reorder_voronoi_for_sharding,
            )
            grid = reorder_voronoi_for_sharding(grid, n_gpus)

        hd = _hyperdiff_coeff(n_grid, grid_type)
        config = MPASPrimitiveEquationConfig(
            nu_del4=hd,
            fix_mass=True,
            pv_scheme="energy",
            time_integrator="ssp_rk3",
        )
        model = MPASPrimitiveEquationModel(grid, sigma, config)
        state = baroclinic_wave_init_mpas(grid, sigma, perturbed=True)

        total_cells = grid.nCells * n_levels
        dev_config = create_voronoi_device_mesh(
            nCells=grid.nCells,
            nEdges=grid.nEdges,
            nVertices=grid.nVertices,
            n_devices=n_gpus,
        )
    else:
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
            CDGridPrimitiveEquationConfig,
        )
        from legoesm.atmosphere.physics.baroclinic_wave import baroclinic_wave_init

        grid = create_cubed_sphere(n_grid)
        hd = _hyperdiff_coeff(n_grid, grid_type)
        config = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=hd,
            hyperdiff_ps_coeff=hd,
            use_conservation_fixer=True,
            fix_mass=True,
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, config)
        state = baroclinic_wave_init(grid, sigma, perturbed=True)

        total_cells = 6 * n_grid * n_grid * n_levels
        dev_config = create_device_mesh(n_devices=n_gpus)

    backend = dev_config.backend

    # Cast to desired precision
    def _cast(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x
    state = jax.tree.map(_cast, state)

    # Shard across devices
    if dev_config.n_devices > 1:
        state = shard_pytree(state, dev_config)

    cells_per_gpu = total_cells // max(1, n_gpus)

    if grid_type == "spectral":
        res_label = f"T{n_grid}"
    elif grid_type == "icosahedral":
        res_label = f"I{n_grid}"
    else:
        res_label = f"C{n_grid}"
    print(
        f"  [{precision}] {res_label}/L{n_levels} on {n_gpus} GPU(s) | "
        f"dt={dt:.0f}s | cells={total_cells:,} | cells/GPU={cells_per_gpu:,}",
        flush=True,
    )

    # ---------------------------------------------------------------
    # JIT compilation (first call)
    # ---------------------------------------------------------------
    t_compile_start = time.perf_counter()
    state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    compile_time = time.perf_counter() - t_compile_start
    print(f"    JIT compile: {compile_time:.2f}s", flush=True)

    # ---------------------------------------------------------------
    # Warmup (let caches settle, exclude from timing)
    # ---------------------------------------------------------------
    t_warmup_start = time.perf_counter()
    for _ in range(n_warmup):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    warmup_time = time.perf_counter() - t_warmup_start

    # ---------------------------------------------------------------
    # Timed steps
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(n_timing):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
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
    grid_type: str = "spectral",
) -> list[TimingResult]:
    """Run weak scaling: fix cells/GPU, sweep GPU counts up to n_gpus."""

    gpu_counts = _valid_gpu_counts(n_gpus, grid_type)
    results: list[TimingResult] = []

    if grid_type == "spectral":
        res_prefix = "T"
    elif grid_type == "icosahedral":
        res_prefix = "I"
    else:
        res_prefix = "C"
    print(f"\n{'='*72}")
    print(f"WEAK SCALING ({grid_type}, base N={base_n}, {n_levels} levels)")
    print(f"GPU counts: {gpu_counts}")
    print(f"{'='*72}")

    for prec in precisions:
        _configure_jax(prec)
        for ng in gpu_counts:
            if grid_type == "icosahedral":
                n_grid = _weak_resolution_ico(ng, base_n)
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
                )
                results.append(result)
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)

    # Compute scaling efficiency relative to 1-GPU baseline
    for prec in precisions:
        prec_results = [r for r in results if r.precision == prec]
        baseline = next((r for r in prec_results if r.n_gpus == 1), None)
        if baseline is not None:
            for r in prec_results:
                # Ideal weak scaling: time/step stays constant
                r.scaling_efficiency = baseline.time_per_step_ms / r.time_per_step_ms

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
) -> list[TimingResult]:
    """Run strong scaling: fix resolution, sweep GPU counts up to n_gpus."""

    gpu_counts = _valid_gpu_counts(n_gpus, grid_type)
    if resolutions is None:
        if grid_type == "spectral":
            defaults = STRONG_RESOLUTIONS_SP
        elif grid_type == "icosahedral":
            defaults = STRONG_RESOLUTIONS_ICO
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
                elif grid_type == "cubed-sphere":
                    cells_per_face = n_grid * n_grid
                    if ng > 6 and cells_per_face < 16:
                        print(
                            f"\n--- {ng} GPU(s), {res_prefix}{n_grid} --- SKIPPED "
                            f"(grid too small for sub-face tiling)",
                            flush=True,
                        )
                        continue

                print(f"\n--- {ng} GPU(s), {res_prefix}{n_grid} [{prec}] ---")
                try:
                    result = run_benchmark(
                        n_grid=n_grid,
                        n_levels=n_levels,
                        n_gpus=ng,
                        precision=prec,
                        mode="strong",
                        n_warmup=n_warmup,
                        n_timing=n_timing,
                        grid_type=grid_type,
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


def write_json(report: ScalingReport, path: Path) -> None:
    """Write full report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": report.mode,
        "precisions": report.precisions,
        "timestamp_utc": report.timestamp_utc,
        "backend": report.backend,
        "hostname": report.hostname,
        "results": [asdict(r) for r in report.results],
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
        f"{'dt(s)':>6s} | {'ms/step':>9s} | {'SYPD':>8s} | "
        f"{'Mcell/s':>9s} | {'Eff':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: (x.precision, x.resolution, x.n_gpus)):
        print(
            f"{r.precision:>7s} | {r.n_gpus:>5d} | {r.resolution:<5d} | "
            f"{r.n_levels:>3d} | {r.dt_seconds:>6.0f} | "
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

    resolution_colors = {
        48: "#4CAF50",
        96: "#2196F3",
        192: "#F44336",
        384: "#9C27B0",
    }
    prec_linestyles = {"float32": "-", "float64": "--"}
    prec_markers = {"float32": "o", "float64": "s"}

    precisions = sorted(set(r.precision for r in results))
    resolutions = sorted(set(r.resolution for r in results))
    all_gpus = sorted(set(r.n_gpus for r in results))

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
            color = resolution_colors.get(n_grid, "#666")

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
                color=resolution_colors.get(n_grid, "#666"),
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
        "--output-dir", type=str, default="output/scaling",
        help="Output directory for results and plots.",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation.",
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

    # Configure JAX for first precision (will be reconfigured per run)
    _configure_jax(precisions[0])

    import jax

    # Resolve GPU count
    max_gpus = args.n_gpus
    if max_gpus <= 0:
        try:
            max_gpus = len(jax.devices("gpu"))
        except RuntimeError:
            max_gpus = len(jax.devices())
    if max_gpus < 1:
        max_gpus = 1

    # Resolve strong scaling resolutions
    if args.strong_resolutions is not None:
        strong_res = [int(x.strip()) for x in args.strong_resolutions.split(",") if x.strip()]
    else:
        strong_res = None  # let run_strong_scaling pick defaults per grid type

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine modes to run
    modes = []
    if args.mode in ("weak", "both"):
        modes.append("weak")
    if args.mode in ("strong", "both"):
        modes.append("strong")

    hostname = os.environ.get("HOSTNAME", os.environ.get("SLURM_NODELIST", "unknown"))
    backend = jax.default_backend().upper()

    print("=" * 72)
    print("  legoESM GPU Scaling Benchmark")
    print("=" * 72)
    print(f"  Grid:        {grid_type}")
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
        )
        all_results.extend(weak_results)
        print_summary_table(weak_results, "weak")

        weak_report = ScalingReport(
            mode="weak",
            precisions=precisions,
            results=weak_results,
            backend=backend,
            hostname=hostname,
        )
        write_csv(weak_results, output_dir / "weak_scaling.csv")
        write_json(weak_report, output_dir / "weak_scaling.json")

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
        )
        all_results.extend(strong_results)
        print_summary_table(strong_results, "strong")

        strong_report = ScalingReport(
            mode="strong",
            precisions=precisions,
            results=strong_results,
            backend=backend,
            hostname=hostname,
        )
        write_csv(strong_results, output_dir / "strong_scaling.csv")
        write_json(strong_report, output_dir / "strong_scaling.json")

        if not args.no_plot:
            plot_strong_scaling(strong_results, output_dir)

    # ---------------------------------------------------------------
    # Combined output
    # ---------------------------------------------------------------
    if all_results:
        write_csv(all_results, output_dir / "all_scaling.csv")

    # Save metadata
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "grid_type": grid_type,
        "backend": backend,
        "hostname": hostname,
        "max_gpus": max_gpus,
        "valid_gpu_counts": _valid_gpu_counts(max_gpus),
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
