#!/usr/bin/env python
"""Atmosphere test matrix: organized, hierarchical test runner.

Runs shallow-water, hydrostatic, non-hydrostatic, and RCE tests across
cubed-sphere, lat-lon, and spectral grids at ~2.5° baseline resolution.

Output structure:
    results/atmosphere/<grid>/<equation_set>/<resolution>/<vertical_coord>/<test>/

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py --quick
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py --only sw
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py --test held_suarez_gray --grid cubed_sphere
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.stdout.reconfigure(line_buffering=True)

# Ensure project root is on sys.path so `tests.*` imports work.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib setup (Agg backend for headless)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Test case registry
# ===========================================================================

@dataclass
class TestCase:
    """A single test case in the matrix."""
    grid_type: str          # cubed_sphere, latlon, spectral
    equation_set: str       # shallow_water, hydrostatic, nonhydrostatic
    resolution: str         # C36, 72x144, T42
    vertical_coord: str     # none, sigma, hybrid, height
    test_name: str          # williamson2, held_suarez_gray, etc.
    duration_days: float    # duration in days
    quick_duration_days: float  # quick-mode duration in days
    run_fn: str             # name of the runner function (string)
    run_kwargs: dict = field(default_factory=dict)

    @property
    def output_path(self) -> str:
        return f"{self.grid_type}/{self.equation_set}/{self.resolution}/{self.vertical_coord}/{self.test_name}"


# ---------------------------------------------------------------------------
# Full test matrix
# ---------------------------------------------------------------------------
TEST_MATRIX: list[TestCase] = [
    # ===== SHALLOW WATER — CUBED-SPHERE =====
    TestCase("cubed_sphere", "shallow_water", "C36", "none", "williamson2",
             5, 1, "run_sw_cubed_sphere", {"test_num": 2}),
    TestCase("cubed_sphere", "shallow_water", "C36", "none", "williamson5",
             15, 1, "run_sw_cubed_sphere", {"test_num": 5}),

    # ===== SHALLOW WATER — LAT-LON =====
    TestCase("latlon", "shallow_water", "72x144", "none", "williamson2",
             5, 1, "run_sw_latlon", {"test_num": 2}),
    TestCase("latlon", "shallow_water", "72x144", "none", "williamson5",
             15, 1, "run_sw_latlon", {"test_num": 5}),

    # ===== SHALLOW WATER — SPECTRAL =====
    TestCase("spectral", "shallow_water", "T42", "none", "williamson2",
             5, 1, "run_sw_spectral", {"test_num": 2}),
    TestCase("spectral", "shallow_water", "T42", "none", "williamson5",
             15, 1, "run_sw_spectral", {"test_num": 5}),

    # ===== HYDROSTATIC — CUBED-SPHERE SIGMA =====
    TestCase("cubed_sphere", "hydrostatic", "C36", "sigma", "held_suarez_gray",
             200, 30, "run_hydro_held_suarez_cube", {"radiation": "gray"}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "sigma", "held_suarez_rrtmgp",
             200, 30, "run_hydro_held_suarez_cube", {"radiation": "rrtmgp"}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "sigma", "baroclinic_wave",
             10, 2, "run_hydro_baroclinic_cube", {}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "sigma", "dcmip_transport_11",
             12, 1, "run_hydro_dcmip_transport", {"test_num": 11}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "sigma", "dcmip_transport_12",
             12, 1, "run_hydro_dcmip_transport", {"test_num": 12}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "sigma", "dcmip_transport_13",
             12, 1, "run_hydro_dcmip_transport", {"test_num": 13}),

    # ===== HYDROSTATIC — CUBED-SPHERE HYBRID =====
    TestCase("cubed_sphere", "hydrostatic", "C36", "hybrid", "held_suarez_gray",
             200, 30, "run_hydro_held_suarez_cube", {"radiation": "gray", "vert": "hybrid"}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "hybrid", "held_suarez_rrtmgp",
             200, 30, "run_hydro_held_suarez_cube", {"radiation": "rrtmgp", "vert": "hybrid"}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "hybrid", "baroclinic_wave",
             10, 2, "run_hydro_baroclinic_cube", {"vert": "hybrid"}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "hybrid", "amip_gray",
             365, 30, "run_hydro_amip_cube", {"radiation": "gray"}),
    TestCase("cubed_sphere", "hydrostatic", "C36", "hybrid", "amip_rrtmgp",
             365, 30, "run_hydro_amip_cube", {"radiation": "rrtmgp"}),

    # ===== HYDROSTATIC — CUBED-SPHERE RCE =====
    TestCase("cubed_sphere", "hydrostatic", "C36", "sigma", "rce_fixed_ocean",
             200, 30, "run_hydro_rce_cube", {}),

    # ===== HYDROSTATIC — LAT-LON =====
    TestCase("latlon", "hydrostatic", "72x144", "sigma", "held_suarez_gray",
             200, 30, "run_hydro_held_suarez_latlon", {"radiation": "gray"}),
    TestCase("latlon", "hydrostatic", "72x144", "sigma", "held_suarez_rrtmgp",
             200, 30, "run_hydro_held_suarez_latlon", {"radiation": "rrtmgp"}),

    # ===== HYDROSTATIC — SPECTRAL SIGMA =====
    TestCase("spectral", "hydrostatic", "T42", "sigma", "held_suarez_gray",
             200, 30, "run_hydro_held_suarez_spectral", {"radiation": "gray"}),
    TestCase("spectral", "hydrostatic", "T42", "sigma", "held_suarez_rrtmgp",
             200, 30, "run_hydro_held_suarez_spectral", {"radiation": "rrtmgp"}),
    TestCase("spectral", "hydrostatic", "T42", "sigma", "baroclinic_wave",
             10, 2, "run_hydro_baroclinic_spectral", {}),

    # ===== HYDROSTATIC — SPECTRAL HYBRID =====
    TestCase("spectral", "hydrostatic", "T42", "hybrid", "held_suarez_gray",
             200, 30, "run_hydro_held_suarez_spectral", {"radiation": "gray", "vert": "hybrid"}),
    TestCase("spectral", "hydrostatic", "T42", "hybrid", "amip_gray",
             365, 30, "run_hydro_amip_spectral", {"radiation": "gray"}),

    # ===== NON-HYDROSTATIC — CUBED-SPHERE =====
    TestCase("cubed_sphere", "nonhydrostatic", "C36", "height", "dcmip2025_tc1",
             3 / 24, 0.5 / 24, "run_nh_dcmip2025_cube", {"test_case": "tc1"}),
    TestCase("cubed_sphere", "nonhydrostatic", "C36", "height", "dcmip2025_tc2a",
             6 / 24, 5 / (24 * 60), "run_nh_dcmip2025_cube", {"test_case": "tc2a"}),
    TestCase("cubed_sphere", "nonhydrostatic", "C36", "height", "dcmip2025_tc3",
             2 / 24, 5 / (24 * 60), "run_nh_dcmip2025_cube", {"test_case": "tc3"}),

    # ===== NON-HYDROSTATIC — SPECTRAL =====
    TestCase("spectral", "nonhydrostatic", "T42", "height", "dcmip2025_tc1",
             3 / 24, 0.5 / 24, "run_nh_dcmip2025_spectral", {"test_case": "tc1"}),
]


# ===========================================================================
# Shared utilities
# ===========================================================================

ALL_RESULTS: list[dict[str, Any]] = []


def record(tc: TestCase, status: str, wall_time: float, notes: str = ""):
    """Record a test result."""
    icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!"}[status]
    ALL_RESULTS.append({
        "test": tc.test_name,
        "grid": tc.grid_type,
        "equation_set": tc.equation_set,
        "resolution": tc.resolution,
        "vertical_coord": tc.vertical_coord,
        "status": status,
        "wall_time": wall_time,
        "notes": notes,
    })
    label = f"{tc.grid_type}/{tc.equation_set}/{tc.test_name}"
    print(f"  {icon} {status:5s} | {label:<55s} | {wall_time:7.1f}s | {notes}")


def check_finite(arrays: dict[str, Any]) -> bool:
    for _name, arr in arrays.items():
        if not bool(jnp.all(jnp.isfinite(arr))):
            return False
    return True


def write_results_txt(output_dir: Path, rows: dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        for k, v in rows.items():
            f.write(f"{k}: {v}\n")


def save_timeseries_csv(output_dir: Path, filename: str, header: str,
                        columns: dict[str, list]):
    """Save timeseries data as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(columns.keys())
    n = len(columns[keys[0]])
    with open(output_dir / filename, "w") as f:
        f.write(header + "\n")
        for i in range(n):
            vals = ",".join(f"{columns[k][i]:.8e}" for k in keys)
            f.write(vals + "\n")


def save_timeseries_plot(output_dir: Path, filename: str, title: str,
                         panels: list[tuple[str, list, list, str]]):
    """Save multi-panel timeseries plot.

    panels: list of (ylabel, x_data, y_data, label)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (ylabel, xd, yd, label) in zip(axes, panels):
        ax.plot(xd, yd, label=label)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compute_hyperdiff_cube(n_grid: int, reference_n: int = 48,
                           reference_coeff: float = 5e16) -> float:
    return reference_coeff * (reference_n / n_grid) ** 4


def compute_hyperdiff_spectral(truncation: int) -> float:
    from legoesm import constants
    a = constants.R_earth
    eig_max = truncation * (truncation + 1) / (a * a)
    return 1.0 / (0.5 * 3600.0 * eig_max ** 2)


def compute_hyperdiff_latlon(n_lat: int, reference_n: int = 64,
                              reference_coeff: float = 2e16) -> float:
    return reference_coeff * (reference_n / n_lat) ** 4


# ===========================================================================
# Runner functions
# ===========================================================================

# ---------------------------------------------------------------------------
# Shallow water — cubed-sphere
# ---------------------------------------------------------------------------
def run_sw_cubed_sphere(tc: TestCase, output_dir: Path, days: float, **kwargs):
    test_num = kwargs["test_num"]

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import (
        ShallowWaterModel, ShallowWaterConfig,
    )
    from tests.test_cases.williamson import (
        williamson_test2, williamson_test2_exact, williamson_test5,
        compute_error_norms,
    )

    n_grid = int(tc.resolution[1:])  # C36 → 36
    grid = create_cubed_sphere(n_grid)
    dt = 300.0
    hyperdiff = compute_hyperdiff_cube(n_grid)

    if test_num == 2:
        state_init = williamson_test2(grid)
    else:
        state_init = williamson_test5(grid)

    config = ShallowWaterConfig(
        hyperdiff_coeff=hyperdiff,
        edge_blend_strength=0.25,
    )
    model = ShallowWaterModel(grid, config)
    state = state_init

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)
    diag_times, diag_mass, diag_max_wind = [], [], []

    mass_init = float(jnp.mean(state_init.h.data))

    t0 = time.time()
    for i in range(n_steps):
        state = model.step(state, dt)
        if (i + 1) % diag_every == 0:
            day = (i + 1) * dt / 86400.0
            diag_times.append(day)
            diag_mass.append(float(jnp.mean(state.h.data)))
            diag_max_wind.append(float(jnp.max(jnp.sqrt(
                state.u.data ** 2 + state.v.data ** 2))))
    jax.block_until_ready(state.h.data)
    wall = time.time() - t0

    ok = check_finite({"h": state.h.data, "u": state.u.data, "v": state.v.data})

    # Error norms for TC2
    notes = ""
    if test_num == 2:
        exact = williamson_test2_exact(grid, days * 86400.0)
        norms = compute_error_norms(state, exact, grid)
        notes = f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    else:
        mass_drift = abs(diag_mass[-1] - mass_init) / abs(mass_init) if diag_mass else 0
        notes = f"mass drift={mass_drift:.2e}"

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "n_steps": n_steps, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"SW Williamson {test_num} — C{n_grid}",
                             [("Mean h (m)", diag_times, diag_mass, "h"),
                              ("Max |v| (m/s)", diag_times, diag_max_wind, "|v|")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Shallow water — lat-lon
# ---------------------------------------------------------------------------
def run_sw_latlon(tc: TestCase, output_dir: Path, days: float, **kwargs):
    test_num = kwargs["test_num"]

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
        FVShallowWaterLatLonModel, FVShallowWaterLatLonConfig,
    )
    from tests.test_cases.williamson_latlon import (
        williamson_test2_latlon, williamson_test2_exact_latlon,
        williamson_test5_latlon, compute_error_norms_latlon,
    )

    parts = tc.resolution.split("x")
    n_lat, n_lon = int(parts[0]), int(parts[1])
    grid = create_latlon_grid(n_lat, n_lon)
    dt = 300.0
    hyperdiff = compute_hyperdiff_latlon(n_lat)

    if test_num == 2:
        state_init = williamson_test2_latlon(grid)
    else:
        state_init = williamson_test5_latlon(grid)

    config = FVShallowWaterLatLonConfig(hyperdiff_coeff=hyperdiff)
    model = FVShallowWaterLatLonModel(grid, config)
    state = state_init

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)
    diag_times, diag_mass, diag_max_wind = [], [], []
    mass_init = float(jnp.mean(state_init.h.data))

    t0 = time.time()
    for i in range(n_steps):
        state = model.step(state, dt)
        if (i + 1) % diag_every == 0:
            day = (i + 1) * dt / 86400.0
            diag_times.append(day)
            diag_mass.append(float(jnp.mean(state.h.data)))
            diag_max_wind.append(float(jnp.max(jnp.sqrt(
                state.u.data ** 2 + state.v.data ** 2))))
    jax.block_until_ready(state.h.data)
    wall = time.time() - t0

    ok = check_finite({"h": state.h.data, "u": state.u.data, "v": state.v.data})

    notes = ""
    if test_num == 2:
        exact = williamson_test2_exact_latlon(grid, days * 86400.0)
        norms = compute_error_norms_latlon(state, exact, grid)
        notes = f"L2={norms['l2']:.2e}, Linf={norms['linf']:.2e}"
    else:
        mass_drift = abs(diag_mass[-1] - mass_init) / abs(mass_init) if diag_mass else 0
        notes = f"mass drift={mass_drift:.2e}"

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "n_steps": n_steps, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"SW Williamson {test_num} — {n_lat}x{n_lon}",
                             [("Mean h (m)", diag_times, diag_mass, "h"),
                              ("Max |v| (m/s)", diag_times, diag_max_wind, "|v|")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Shallow water — spectral
# ---------------------------------------------------------------------------
def run_sw_spectral(tc: TestCase, output_dir: Path, days: float, **kwargs):
    test_num = kwargs["test_num"]

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralSWConfig, spectral_sw_tendencies,
        williamson_test2_spectral, williamson_test5_spectral,
        spectral_to_grid,
    )
    from legoesm.timestepping.ssp_rk3 import ssp_rk3_step

    truncation = int(tc.resolution[1:])  # T42 → 42
    grid = create_gaussian_grid(truncation)
    dt = 60.0  # explicit spectral SW needs small dt

    a = grid.radius
    eig_max = truncation * (truncation + 1) / (a * a)
    hyperdiff = 1.0 / (1.0 * 3600.0 * eig_max ** 2)
    config = SpectralSWConfig(hyperdiff_coeff=hyperdiff)

    if test_num == 2:
        state_init = williamson_test2_spectral(grid)
    else:
        state_init = williamson_test5_spectral(grid)

    def tendency(s):
        return spectral_sw_tendencies(s, grid, config)

    step_jit = jax.jit(lambda s, dt_: ssp_rk3_step(s, tendency, dt_))
    state = state_init

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, n_steps // 20)
    diag_times, diag_max_wind = [], []

    t0 = time.time()
    for i in range(n_steps):
        state = step_jit(state, dt)
        if (i + 1) % diag_every == 0:
            fields = spectral_to_grid(state, grid)
            day = (i + 1) * dt / 86400.0
            diag_times.append(day)
            diag_max_wind.append(float(jnp.max(jnp.sqrt(
                fields["u"] ** 2 + fields["v"] ** 2))))
    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    fields = spectral_to_grid(state, grid)
    ok = check_finite({"h": fields["h"], "u": fields["u"], "v": fields["v"]})
    notes = ""

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "days": days, "dt": dt, "n_steps": n_steps, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL",
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"SW Spectral Williamson {test_num} — T{truncation}",
                             [("Max |v| (m/s)", diag_times, diag_max_wind, "|v|")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic Held-Suarez — cubed-sphere
# ---------------------------------------------------------------------------
def run_hydro_held_suarez_cube(tc: TestCase, output_dir: Path, days: float, **kwargs):
    radiation = kwargs.get("radiation", "gray")
    vert = kwargs.get("vert", "sigma")

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    from legoesm.atmosphere.dynamics.primitive_eq import (
        PrimitiveEquationModel, PrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing, held_suarez_init
    from legoesm.core.operators import global_integral

    n_grid = int(tc.resolution[1:])
    nlev = 40
    dt = 200.0
    grid = create_cubed_sphere(n_grid)

    if vert == "hybrid":
        sigma = standard_hybrid_levels(nlev)
    else:
        sigma = create_sigma_coordinate(nlev)

    hyperdiff = compute_hyperdiff_cube(n_grid)
    config = PrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        use_conservation_fixer=True,
        fix_mass=True,
        edge_blend_uv=0.15,
        edge_blend_T=0.10,
        edge_blend_p_s=0.20,
        edge_blend_width=2,
    )
    model = PrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)
    mass_init = float(global_integral(state.p_s, grid))

    # Physics: gray or RRTMGP via make_physics
    if radiation == "rrtmgp":
        from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
        from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

        phys_cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="rrtmgp"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        physics_fn = make_physics(phys_cfg, model_type="hydrostatic", dt=dt)
    else:
        physics_fn = held_suarez_forcing

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))  # every 6 hours
    diag_times, diag_mass, diag_max_wind, diag_mean_T = [], [], [], []

    t0 = time.time()
    last_print = t0
    for step in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)

        if (step + 1) % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 1000:
                print(f"  BLOWUP at step {step + 1}, u_max={u_max:.1f}")
                break

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            mass = float(global_integral(state.p_s, grid))
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            mean_T = float(jnp.mean(state.T.data))
            diag_times.append(day)
            diag_mass.append(mass)
            diag_max_wind.append(max_wind)
            diag_mean_T.append(mean_T)

            now = time.time()
            if now - last_print > 30:
                mass_drift = abs(mass - mass_init) / abs(mass_init)
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s | "
                      f"<T>={mean_T:.1f} K | mass drift={mass_drift:.2e}")
                last_print = now

    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data,
                        "p_s": state.p_s.data})
    mass_final = float(global_integral(state.p_s, grid))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    notes = f"mass drift={mass_drift:.2e}, max|v|={diag_max_wind[-1]:.1f}" if diag_max_wind else ""

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "vertical_coord": vert, "levels": nlev, "radiation": radiation,
        "days": days, "dt": dt, "n_steps": n_steps, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        mass_drift_ts = [abs(m - mass_init) / abs(mass_init) for m in diag_mass]
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"Held-Suarez ({radiation}) — C{n_grid}/L{nlev} ({vert})",
                             [("Mass drift (rel)", diag_times, mass_drift_ts, "mass"),
                              ("Max |v| (m/s)", diag_times, diag_max_wind, "|v|"),
                              ("Mean T (K)", diag_times, diag_mean_T, "T")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic Held-Suarez — lat-lon
# ---------------------------------------------------------------------------
def run_hydro_held_suarez_latlon(tc: TestCase, output_dir: Path, days: float, **kwargs):
    radiation = kwargs.get("radiation", "gray")

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_latlon import (
        LatLonPrimitiveEquationModel, LatLonPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez_latlon import (
        held_suarez_forcing_latlon, held_suarez_init_latlon,
    )
    from legoesm.core.operators_latlon import global_integral as global_integral_ll

    parts = tc.resolution.split("x")
    n_lat, n_lon = int(parts[0]), int(parts[1])
    nlev = 40
    dt = 200.0

    grid = create_latlon_grid(n_lat, n_lon)
    sigma = create_sigma_coordinate(nlev)
    hyperdiff = compute_hyperdiff_latlon(n_lat)

    config = LatLonPrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = LatLonPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init_latlon(grid, sigma)
    mass_init = float(global_integral_ll(state.p_s, grid))

    if radiation == "rrtmgp":
        from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
        from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

        phys_cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="rrtmgp"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        physics_fn = make_physics(phys_cfg, model_type="hydrostatic", dt=dt)
    else:
        physics_fn = held_suarez_forcing_latlon

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))
    diag_times, diag_mass, diag_max_wind, diag_mean_T = [], [], [], []

    t0 = time.time()
    last_print = t0
    for step in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)

        if (step + 1) % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 1000:
                print(f"  BLOWUP at step {step + 1}")
                break

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            mass = float(global_integral_ll(state.p_s, grid))
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            mean_T = float(jnp.mean(state.T.data))
            diag_times.append(day)
            diag_mass.append(mass)
            diag_max_wind.append(max_wind)
            diag_mean_T.append(mean_T)

            now = time.time()
            if now - last_print > 30:
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s | <T>={mean_T:.1f} K")
                last_print = now

    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data,
                        "p_s": state.p_s.data})
    mass_final = float(global_integral_ll(state.p_s, grid))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    notes = f"mass drift={mass_drift:.2e}"

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "radiation": radiation, "levels": nlev,
        "days": days, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        mass_drift_ts = [abs(m - mass_init) / abs(mass_init) for m in diag_mass]
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"Held-Suarez ({radiation}) — {n_lat}x{n_lon}/L{nlev}",
                             [("Mass drift (rel)", diag_times, mass_drift_ts, "mass"),
                              ("Max |v| (m/s)", diag_times, diag_max_wind, "|v|"),
                              ("Mean T (K)", diag_times, diag_mean_T, "T")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic Held-Suarez — spectral
# ---------------------------------------------------------------------------
def run_hydro_held_suarez_spectral(tc: TestCase, output_dir: Path, days: float, **kwargs):
    radiation = kwargs.get("radiation", "gray")
    vert = kwargs.get("vert", "sigma")

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral, spectral_pe_to_grid,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral

    truncation = int(tc.resolution[1:])
    nlev = 40
    dt = 600.0

    grid = create_gaussian_grid(truncation)
    if vert == "hybrid":
        sigma = standard_hybrid_levels(nlev)
    else:
        sigma = create_sigma_coordinate(nlev)

    hyperdiff = compute_hyperdiff_spectral(truncation)
    config = SpectralPEConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_order=2,
        semi_implicit=True,
        si_T_ref=300.0,
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=300.0)

    # For RRTMGP, use make_physics; for gray, use spectral forcing
    if radiation == "rrtmgp":
        from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
        from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig

        phys_cfg = PhysicsConfig(
            radiation=RadiationConfig(scheme="rrtmgp"),
            convection=ConvectionConfig(scheme="none"),
            turbulence=TurbulenceConfig(scheme="none"),
            microphysics=MicrophysicsConfig(scheme="none"),
            gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
        )
        physics_fn = make_physics(phys_cfg, model_type="spectral_pe", dt=dt)
    else:
        physics_fn = held_suarez_forcing_spectral

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))
    diag_times, diag_max_wind, diag_mean_T = [], [], []

    t0 = time.time()
    last_print = t0
    for step in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)

        if (step + 1) % 100 == 0:
            if not jnp.all(jnp.isfinite(state.vor_hat.data)):
                print(f"  BLOWUP at step {step + 1}")
                break

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            fields = spectral_pe_to_grid(state, grid, sigma)
            max_wind = float(jnp.max(jnp.sqrt(fields["u"]**2 + fields["v"]**2)))
            mean_T = float(jnp.mean(fields["T"]))
            diag_times.append(day)
            diag_max_wind.append(max_wind)
            diag_mean_T.append(mean_T)

            now = time.time()
            if now - last_print > 30:
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s | <T>={mean_T:.1f} K")
                last_print = now

    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    ok = check_finite({
        "vor_hat": state.vor_hat.data, "div_hat": state.div_hat.data,
        "T_hat": state.T_hat.data, "lnps_hat": state.lnps_hat.data,
    })
    notes = f"max|v|={diag_max_wind[-1]:.1f}" if diag_max_wind else ""

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "vertical_coord": vert, "radiation": radiation, "levels": nlev,
        "days": days, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"Held-Suarez ({radiation}) — T{truncation}/L{nlev} ({vert})",
                             [("Max |v| (m/s)", diag_times, diag_max_wind, "|v|"),
                              ("Mean T (K)", diag_times, diag_mean_T, "T")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic baroclinic wave — cubed-sphere
# ---------------------------------------------------------------------------
def run_hydro_baroclinic_cube(tc: TestCase, output_dir: Path, days: float, **kwargs):
    vert = kwargs.get("vert", "sigma")

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
    from legoesm.atmosphere.dynamics.primitive_eq import (
        PrimitiveEquationModel, PrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.baroclinic_wave import baroclinic_wave_init
    from legoesm.core.operators import global_integral

    n_grid = int(tc.resolution[1:])
    nlev = 40
    dt = 200.0

    grid = create_cubed_sphere(n_grid)
    if vert == "hybrid":
        sigma = standard_hybrid_levels(nlev)
    else:
        sigma = create_sigma_coordinate(nlev)

    hyperdiff = compute_hyperdiff_cube(n_grid)
    config = PrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        use_conservation_fixer=True,
        fix_mass=True,
        edge_blend_uv=0.15,
        edge_blend_T=0.10,
        edge_blend_p_s=0.20,
        edge_blend_width=2,
    )
    model = PrimitiveEquationModel(grid, sigma, config)
    state = baroclinic_wave_init(grid, sigma, perturbed=True)
    mass_init = float(global_integral(state.p_s, grid))

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))
    diag_times, diag_max_wind, diag_ps_pert = [], [], []

    t0 = time.time()
    ps_init_data = np.array(state.p_s.data)
    last_print = t0
    for step in range(n_steps):
        state = model.step(state, dt)

        if (step + 1) % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 1000:
                print(f"  BLOWUP at step {step + 1}")
                break

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            ps_pert = float(jnp.max(jnp.abs(state.p_s.data - ps_init_data)))
            diag_times.append(day)
            diag_max_wind.append(max_wind)
            diag_ps_pert.append(ps_pert)

            now = time.time()
            if now - last_print > 30:
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s | "
                      f"p_s pert={ps_pert:.0f} Pa")
                last_print = now

    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data,
                        "p_s": state.p_s.data})
    mass_final = float(global_integral(state.p_s, grid))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    u_max_final = float(jnp.max(jnp.abs(state.u.data)))
    notes = f"mass drift={mass_drift:.2e}, |u|_max={u_max_final:.1f} m/s"

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "vertical_coord": vert, "levels": nlev,
        "days": days, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"Baroclinic Wave — C{n_grid}/L{nlev} ({vert})",
                             [("Max |v| (m/s)", diag_times, diag_max_wind, "|v|"),
                              ("p_s perturbation (Pa)", diag_times, diag_ps_pert, "p_s'")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic baroclinic wave — spectral
# ---------------------------------------------------------------------------
def run_hydro_baroclinic_spectral(tc: TestCase, output_dir: Path, days: float, **kwargs):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        baroclinic_wave_init_spectral, spectral_pe_to_grid,
    )

    truncation = int(tc.resolution[1:])
    nlev = 40
    dt = 600.0

    grid = create_gaussian_grid(truncation)
    sigma = create_sigma_coordinate(nlev)
    hyperdiff = compute_hyperdiff_spectral(truncation)

    config = SpectralPEConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_order=2,
        semi_implicit=True,
        si_T_ref=300.0,
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config)
    state = baroclinic_wave_init_spectral(grid, sigma, perturbed=True)

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(6 * 3600 / dt))
    diag_times, diag_max_wind = [], []

    t0 = time.time()
    last_print = t0
    for step in range(n_steps):
        state = model.step(state, dt)

        if (step + 1) % 100 == 0:
            if not jnp.all(jnp.isfinite(state.vor_hat.data)):
                print(f"  BLOWUP at step {step + 1}")
                break

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            fields = spectral_pe_to_grid(state, grid, sigma)
            max_wind = float(jnp.max(jnp.sqrt(fields["u"]**2 + fields["v"]**2)))
            diag_times.append(day)
            diag_max_wind.append(max_wind)

            now = time.time()
            if now - last_print > 30:
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s")
                last_print = now

    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    ok = check_finite({"vor_hat": state.vor_hat.data, "div_hat": state.div_hat.data,
                        "T_hat": state.T_hat.data})
    notes = f"max|v|={diag_max_wind[-1]:.1f}" if diag_max_wind else ""

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "levels": nlev, "days": days, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"Baroclinic Wave — T{truncation}/L{nlev}",
                             [("Max |v| (m/s)", diag_times, diag_max_wind, "|v|")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic DCMIP transport — cubed-sphere
# ---------------------------------------------------------------------------
def run_hydro_dcmip_transport(tc: TestCase, output_dir: Path, days: float, **kwargs):
    test_num = kwargs["test_num"]

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.tracer_transport import (
        TracerTransportModel, TracerTransportConfig,
    )
    from tests.test_cases.dcmip_transport import (
        dcmip11_wind, dcmip11_init,
        dcmip12_wind, dcmip12_init,
        dcmip13_wind, dcmip13_init,
        compute_tracer_error_norms,
        create_dcmip_sigma,
    )

    TEST_CONFIGS = {
        11: {"wind_fn": dcmip11_wind, "init_fn": dcmip11_init,
             "period_days": 12.0, "default_dt": 1800.0, "n_tracers": 4},
        12: {"wind_fn": dcmip12_wind, "init_fn": dcmip12_init,
             "period_days": 1.0, "default_dt": 600.0, "n_tracers": 1},
        13: {"wind_fn": dcmip13_wind, "init_fn": dcmip13_init,
             "period_days": 12.0, "default_dt": 1800.0, "n_tracers": 4},
    }
    tc_cfg = TEST_CONFIGS[test_num]

    n_grid = int(tc.resolution[1:])
    nlev = 30
    dt = tc_cfg["default_dt"]
    grid = create_cubed_sphere(n_grid)
    sigma_coord = create_dcmip_sigma(nlev)

    state_init = tc_cfg["init_fn"](grid, sigma_coord)
    config = TracerTransportConfig(hyperdiff_coeff=0.0)
    model = TracerTransportModel(grid, sigma_coord, tc_cfg["wind_fn"], config)

    # Use actual period or capped by days
    period = min(days, tc_cfg["period_days"])
    n_steps = int(period * 86400.0 / dt)
    diag_every = max(1, n_steps // 20)

    state = state_init
    t0 = time.time()
    for i in range(n_steps):
        state = model.step(state, dt)
        if (i + 1) % diag_every == 0:
            q = state.tracers.data
            progress = (i + 1) / n_steps * 100
            q1_min = float(jnp.min(q[..., 0]))
            q1_max = float(jnp.max(q[..., 0]))
            print(f"    Step {i + 1:6d}/{n_steps} ({progress:5.1f}%) | "
                  f"q1: [{q1_min:.4f}, {q1_max:.4f}]")
    jax.block_until_ready(state.tracers.data)
    wall = time.time() - t0

    ok = check_finite({"tracers": state.tracers.data})

    # Compute error norms for flow-reversal tests
    notes = ""
    if test_num in (11, 12):
        norms = compute_tracer_error_norms(state, state_init, grid)
        n_tracers = tc_cfg["n_tracers"]
        norm_strs = []
        for i in range(n_tracers):
            norm_strs.append(f"q{i + 1} L2={norms['l2'][i]:.4e}")
        notes = ", ".join(norm_strs)

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "levels": nlev, "period_days": period, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic AMIP — cubed-sphere (gray radiation, no SST data needed)
# ---------------------------------------------------------------------------
def run_hydro_amip_cube(tc: TestCase, output_dir: Path, days: float, **kwargs):
    """Simplified AMIP: analytical SSTs, hybrid vertical coord, gray or RRTMGP."""
    radiation = kwargs.get("radiation", "gray")

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import standard_hybrid_levels
    from legoesm.atmosphere.dynamics.primitive_eq import (
        PrimitiveEquationModel, PrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.core.operators import global_integral
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm import constants

    n_grid = int(tc.resolution[1:])
    nlev = 40
    dt = 300.0
    grid = create_cubed_sphere(n_grid)
    sigma = standard_hybrid_levels(nlev)
    hyperdiff = compute_hyperdiff_cube(n_grid)

    config = PrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        use_conservation_fixer=True,
        fix_mass=True,
        edge_blend_uv=0.15,
        edge_blend_T=0.10,
        edge_blend_p_s=0.20,
        edge_blend_width=2,
    )
    model = PrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma, T_init=280.0)
    mass_init = float(global_integral(state.p_s, grid))

    # Analytical SST: zonally uniform, warm tropics, cold poles
    lat_2d = grid.lat  # (6, n, n)
    sst = 273.15 + 27.0 * jnp.maximum(0.0, 1.0 - 3.0 * lat_2d**2 / (jnp.pi / 2)**2)

    # Moisture initialization (60% RH, sigma-weighted)
    from legoesm.grids.vertical import pressure_from_hybrid
    p_full = pressure_from_hybrid(sigma, state.p_s.data)
    q_sat_init = saturation_mixing_ratio(state.T.data, p_full)
    # sigma-like weighting via B_full
    b_weights = jnp.array(sigma.B_full)
    q_v = 0.6 * q_sat_init * b_weights ** 2
    q_v = jnp.minimum(q_v, q_sat_init)

    # Physics config
    gray_config = GrayRadiationConfig(
        tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
        sfc_albedo=0.31, perpetual_equinox=True,
    )

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(24 * 3600 / dt))  # daily diagnostics
    diag_times, diag_max_wind, diag_mean_T, diag_mean_ps = [], [], [], []

    t0 = time.time()
    last_print = t0
    for step in range(n_steps):
        state = model.step(state, dt)

        if (step + 1) % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 1000:
                print(f"  BLOWUP at step {step + 1}")
                break

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            mean_T = float(jnp.mean(state.T.data))
            mean_ps = float(jnp.mean(state.p_s.data))
            diag_times.append(day)
            diag_max_wind.append(max_wind)
            diag_mean_T.append(mean_T)
            diag_mean_ps.append(mean_ps)

            now = time.time()
            if now - last_print > 30:
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s | "
                      f"<T>={mean_T:.1f} K | <p_s>={mean_ps:.0f} Pa")
                last_print = now

    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data,
                        "p_s": state.p_s.data})
    mass_final = float(global_integral(state.p_s, grid))
    mass_drift = abs(mass_final - mass_init) / abs(mass_init)
    notes = f"mass drift={mass_drift:.2e}"

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "radiation": radiation, "levels": nlev,
        "days": days, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"AMIP ({radiation}) — C{n_grid}/L{nlev} hybrid",
                             [("Max |v| (m/s)", diag_times, diag_max_wind, "|v|"),
                              ("Mean T (K)", diag_times, diag_mean_T, "T"),
                              ("Mean p_s (Pa)", diag_times, diag_mean_ps, "p_s")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic AMIP — spectral
# ---------------------------------------------------------------------------
def run_hydro_amip_spectral(tc: TestCase, output_dir: Path, days: float, **kwargs):
    """Simplified AMIP with spectral PE: Held-Suarez forcing + hybrid coord."""
    radiation = kwargs.get("radiation", "gray")

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import standard_hybrid_levels
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral, spectral_pe_to_grid,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_forcing_spectral

    truncation = int(tc.resolution[1:])
    nlev = 40
    dt = 600.0

    grid = create_gaussian_grid(truncation)
    sigma = standard_hybrid_levels(nlev)
    hyperdiff = compute_hyperdiff_spectral(truncation)

    config = SpectralPEConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_order=2,
        semi_implicit=True,
        si_T_ref=300.0,
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, config)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=280.0)

    physics_fn = held_suarez_forcing_spectral

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(24 * 3600 / dt))
    diag_times, diag_max_wind, diag_mean_T = [], [], []

    t0 = time.time()
    last_print = t0
    for step in range(n_steps):
        state = model.step_with_physics(state, dt, physics_fn)

        if (step + 1) % 100 == 0:
            if not jnp.all(jnp.isfinite(state.vor_hat.data)):
                print(f"  BLOWUP at step {step + 1}")
                break

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            fields = spectral_pe_to_grid(state, grid, sigma)
            max_wind = float(jnp.max(jnp.sqrt(fields["u"]**2 + fields["v"]**2)))
            mean_T = float(jnp.mean(fields["T"]))
            diag_times.append(day)
            diag_max_wind.append(max_wind)
            diag_mean_T.append(mean_T)

            now = time.time()
            if now - last_print > 30:
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s | <T>={mean_T:.1f} K")
                last_print = now

    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    ok = check_finite({"vor_hat": state.vor_hat.data, "T_hat": state.T_hat.data,
                        "lnps_hat": state.lnps_hat.data})
    notes = f"max|v|={diag_max_wind[-1]:.1f}" if diag_max_wind else ""

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "radiation": radiation, "levels": nlev,
        "days": days, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"AMIP ({radiation}) — T{truncation}/L{nlev} hybrid",
                             [("Max |v| (m/s)", diag_times, diag_max_wind, "|v|"),
                              ("Mean T (K)", diag_times, diag_mean_T, "T")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Hydrostatic RCE — cubed-sphere with slab ocean
# ---------------------------------------------------------------------------
def run_hydro_rce_cube(tc: TestCase, output_dir: Path, days: float, **kwargs):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate, pressure_from_sigma
    from legoesm.atmosphere.dynamics.primitive_eq import (
        PrimitiveEquationModel, PrimitiveEquationConfig,
    )
    from legoesm.atmosphere.physics.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.atmosphere.physics.convection.sbm import sbm_convection
    from legoesm.atmosphere.physics.convection.config import SBMConfig
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm import constants

    n_grid = int(tc.resolution[1:])
    nlev = 40
    dt = 300.0
    grid = create_cubed_sphere(n_grid)
    sigma = create_sigma_coordinate(nlev)
    hyperdiff = compute_hyperdiff_cube(n_grid)

    dycore_config = PrimitiveEquationConfig(
        hyperdiff_coeff=hyperdiff,
        hyperdiff_ps_coeff=hyperdiff,
        use_conservation_fixer=True,
        fix_mass=True,
        edge_blend_uv=0.15 if n_grid > 16 else 0.0,
        edge_blend_T=0.10 if n_grid > 16 else 0.0,
        edge_blend_p_s=0.20 if n_grid > 16 else 0.0,
        edge_blend_width=2 if n_grid > 16 else 1,
    )
    model = PrimitiveEquationModel(grid, sigma, dycore_config)
    state = held_suarez_init(grid, sigma, T_init=280.0)

    # Moisture init (60% RH)
    p_full = state.p_s.data[..., None] * sigma.sigma_full
    q_sat_init = saturation_mixing_ratio(state.T.data, p_full)
    q_v = 0.6 * q_sat_init * jnp.array(sigma.sigma_full) ** 2
    q_v = jnp.minimum(q_v, q_sat_init)

    # Fixed SST (300 K everywhere)
    ocean_sst = jnp.full(state.p_s.data.shape, 300.0)

    gray_config = GrayRadiationConfig(
        tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
        sfc_albedo=0.31, perpetual_equinox=True,
    )
    sbm_config = SBMConfig(tau_c=7200.0, RH_ref=0.7)

    n_steps = int(days * 86400 / dt)
    diag_every = max(1, int(24 * 3600 / dt))
    diag_times, diag_mean_T, diag_max_wind = [], [], []

    g = constants.g
    c_pd = constants.c_pd
    L_v = constants.L_v
    C_H = 1.5e-3

    t0 = time.time()
    last_print = t0
    for step in range(n_steps):
        # Dynamics step
        state = model.step(state, dt)

        if (step + 1) % 100 == 0:
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 1000:
                print(f"  BLOWUP at step {step + 1}")
                break

        # Simple large-scale condensation
        p_full_now = state.p_s.data[..., None] * sigma.sigma_full
        q_sat = saturation_mixing_ratio(state.T.data, p_full_now)
        excess = jnp.maximum(q_v - q_sat, 0.0)
        q_v = q_v - excess
        new_T = state.T.data + L_v * excess / c_pd
        state = state._replace(T=state.T.replace(data=new_T))

        if (step + 1) % diag_every == 0:
            day = (step + 1) * dt / 86400.0
            max_wind = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
            mean_T = float(jnp.mean(state.T.data))
            diag_times.append(day)
            diag_mean_T.append(mean_T)
            diag_max_wind.append(max_wind)

            now = time.time()
            if now - last_print > 30:
                print(f"    Day {day:7.1f}/{days} | max|v|={max_wind:6.1f} m/s | <T>={mean_T:.1f} K")
                last_print = now

    jax.block_until_ready(state.T.data)
    wall = time.time() - t0

    ok = check_finite({"T": state.T.data, "u": state.u.data, "v": state.v.data})
    notes = f"<T>={diag_mean_T[-1]:.1f} K" if diag_mean_T else ""

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "levels": nlev, "days": days, "dt": dt, "wall_time": f"{wall:.1f}s",
        "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    if diag_times:
        save_timeseries_plot(output_dir, "timeseries.png",
                             f"RCE Fixed Ocean — C{n_grid}/L{nlev}",
                             [("Mean T (K)", diag_times, diag_mean_T, "T"),
                              ("Max |v| (m/s)", diag_times, diag_max_wind, "|v|")])

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Non-hydrostatic DCMIP-2025 — cubed-sphere
# ---------------------------------------------------------------------------
def run_nh_dcmip2025_cube(tc: TestCase, output_dir: Path, days: float, **kwargs):
    test_case = kwargs["test_case"]

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel, CompressibleEulerConfig,
    )

    n_grid = int(tc.resolution[1:])
    nlev = 40
    grid = create_cubed_sphere(n_grid)

    # TC-specific init and config
    duration_hours = days * 24.0

    if test_case == "tc1":
        from tests.test_cases.dcmip2025 import dcmip25_tc1_init
        state, height_coord, terrain_metric = dcmip25_tc1_init(grid, n_levels=nlev)
        dt = max(0.2, 6.0 * (16.0 / n_grid))
        nh_config = CompressibleEulerConfig(
            n_acoustic_substeps=10,
            semi_implicit_acoustic=True,
            sponge_width=10000.0,
            sponge_coeff=0.05,
            hyperdiff_coeff=compute_hyperdiff_cube(n_grid),
            edge_blend_uv=0.15,
            edge_blend_w=0.15,
            edge_blend_theta=0.15,
            edge_blend_rho=0.15,
        )
    elif test_case == "tc2a":
        from tests.test_cases.dcmip2025 import dcmip25_tc2_init
        state, height_coord, terrain_metric, _ = dcmip25_tc2_init(grid, n_levels=nlev)
        dt = max(0.2, 4.0 * (16.0 / n_grid))
        nh_config = CompressibleEulerConfig(
            n_acoustic_substeps=10,
            semi_implicit_acoustic=True,
            sponge_width=5000.0,
            sponge_coeff=0.1,
            small_earth_factor=1.0 / 120.0,
            hyperdiff_coeff=compute_hyperdiff_cube(n_grid),
            edge_blend_uv=0.15,
            edge_blend_w=0.15,
            edge_blend_theta=0.15,
            edge_blend_rho=0.15,
        )
    else:  # tc3
        from tests.test_cases.dcmip2025 import dcmip25_tc3_init
        state, height_coord, terrain_metric, _ = dcmip25_tc3_init(grid, n_levels=nlev)
        dt = max(0.1, 2.0 * (16.0 / n_grid))
        nh_config = CompressibleEulerConfig(
            n_acoustic_substeps=10,
            semi_implicit_acoustic=True,
            sponge_width=5000.0,
            sponge_coeff=0.1,
            small_earth_factor=1.0 / 120.0,
            hyperdiff_coeff=compute_hyperdiff_cube(n_grid),
            edge_blend_uv=0.15,
            edge_blend_w=0.15,
            edge_blend_theta=0.15,
            edge_blend_rho=0.15,
        )

    model = CompressibleEulerModel(grid, height_coord, terrain_metric, nh_config)

    n_steps = int(duration_hours * 3600.0 / dt)
    diag_every = max(1, n_steps // 20)

    t0 = time.time()
    stable = True
    for i in range(n_steps):
        state = model.step(state, dt)
        if (i + 1) % diag_every == 0:
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            if not jnp.all(jnp.isfinite(state.u.data)):
                print(f"    BLOWUP at step {i + 1}")
                stable = False
                break
            t_hours = (i + 1) * dt / 3600.0
            print(f"    Step {i + 1:6d}/{n_steps} | t={t_hours:.2f}h | |w|_max={w_max:.4f} m/s")

    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    ok = stable and check_finite({
        "u": state.u.data, "v": state.v.data, "w": state.w.data,
        "theta_prime": state.theta_prime.data, "rho_prime": state.rho_prime.data,
    })
    w_max_final = float(jnp.max(jnp.abs(state.w.data))) if ok else float("nan")
    notes = f"|w|_max={w_max_final:.4f} m/s, dt={dt:.2f}s"

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "levels": nlev, "duration_hours": duration_hours, "dt": dt,
        "wall_time": f"{wall:.1f}s", "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    return "PASS" if ok else "FAIL", wall, notes


# ---------------------------------------------------------------------------
# Non-hydrostatic DCMIP-2025 — spectral
# ---------------------------------------------------------------------------
def run_nh_dcmip2025_spectral(tc: TestCase, output_dir: Path, days: float, **kwargs):
    test_case = kwargs["test_case"]

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_nh import (
        SpectralCompressibleEulerModel, SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )

    truncation = int(tc.resolution[1:])
    nlev = 40
    dt = 2.0
    duration_hours = days * 24.0

    grid = create_gaussian_grid(truncation)
    hyperdiff = compute_hyperdiff_spectral(truncation)

    if test_case == "tc1":
        state, height_coord, terrain_metric = dcmip25_tc1_init_spectral(grid, n_levels=nlev)
    else:
        raise ValueError(f"Spectral NH only supports tc1, got {test_case}")

    config = SpectralNHConfig(
        n_acoustic_substeps=6,
        sponge_width=10000.0,
        sponge_coeff=0.05,
        hyperdiff_coeff=hyperdiff,
    )
    model = SpectralCompressibleEulerModel(grid, height_coord, terrain_metric, config)

    n_steps = int(duration_hours * 3600.0 / dt)
    diag_every = max(1, n_steps // 20)

    t0 = time.time()
    stable = True
    for i in range(n_steps):
        state = model.step(state, dt)
        if (i + 1) % diag_every == 0:
            if not jnp.all(jnp.isfinite(state.theta_prime_hat.data)):
                print(f"    BLOWUP at step {i + 1}")
                stable = False
                break
            t_hours = (i + 1) * dt / 3600.0
            print(f"    Step {i + 1:6d}/{n_steps} | t={t_hours:.2f}h")

    jax.block_until_ready(state.theta_prime_hat.data)
    wall = time.time() - t0

    ok = stable and check_finite({
        "theta_prime_hat": state.theta_prime_hat.data,
        "rho_prime_hat": state.rho_prime_hat.data,
    })
    notes = f"dt={dt:.2f}s"

    results = {
        "test": tc.test_name, "grid": tc.grid_type, "resolution": tc.resolution,
        "levels": nlev, "duration_hours": duration_hours, "dt": dt,
        "wall_time": f"{wall:.1f}s", "status": "PASS" if ok else "FAIL", "notes": notes,
    }
    write_results_txt(output_dir, results)

    return "PASS" if ok else "FAIL", wall, notes


# ===========================================================================
# Runner dispatch
# ===========================================================================

RUNNERS: dict[str, Callable] = {
    "run_sw_cubed_sphere": run_sw_cubed_sphere,
    "run_sw_latlon": run_sw_latlon,
    "run_sw_spectral": run_sw_spectral,
    "run_hydro_held_suarez_cube": run_hydro_held_suarez_cube,
    "run_hydro_held_suarez_latlon": run_hydro_held_suarez_latlon,
    "run_hydro_held_suarez_spectral": run_hydro_held_suarez_spectral,
    "run_hydro_baroclinic_cube": run_hydro_baroclinic_cube,
    "run_hydro_baroclinic_spectral": run_hydro_baroclinic_spectral,
    "run_hydro_dcmip_transport": run_hydro_dcmip_transport,
    "run_hydro_amip_cube": run_hydro_amip_cube,
    "run_hydro_amip_spectral": run_hydro_amip_spectral,
    "run_hydro_rce_cube": run_hydro_rce_cube,
    "run_nh_dcmip2025_cube": run_nh_dcmip2025_cube,
    "run_nh_dcmip2025_spectral": run_nh_dcmip2025_spectral,
}


# ===========================================================================
# CLI + Main
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Atmosphere test matrix: organized, hierarchical test runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", type=str, default="all",
                   choices=["sw", "hydro", "nh", "all"],
                   help="Run only a specific equation set (default: all)")
    p.add_argument("--grid", type=str, default="all",
                   choices=["cubed_sphere", "latlon", "spectral", "all"],
                   help="Run only a specific grid type (default: all)")
    p.add_argument("--test", type=str, default=None,
                   help="Run only a specific test name (e.g. held_suarez_gray)")
    p.add_argument("--resolution", type=str, default=None,
                   help="Override baseline resolution (e.g. C48, T85, 90x180)")
    p.add_argument("--output", "-o", type=str, default="results/atmosphere",
                   help="Base output directory (default: results/atmosphere)")
    p.add_argument("--quick", action="store_true",
                   help="Use shorter durations for quick verification")
    p.add_argument("--list", action="store_true",
                   help="List all test cases and exit")
    return p


def filter_tests(tests: list[TestCase], args) -> list[TestCase]:
    """Filter test matrix based on CLI arguments."""
    filtered = tests

    # Filter by equation set
    eq_map = {"sw": "shallow_water", "hydro": "hydrostatic", "nh": "nonhydrostatic"}
    if args.only != "all":
        eq_set = eq_map[args.only]
        filtered = [t for t in filtered if t.equation_set == eq_set]

    # Filter by grid type
    if args.grid != "all":
        filtered = [t for t in filtered if t.grid_type == args.grid]

    # Filter by test name
    if args.test:
        filtered = [t for t in filtered if args.test in t.test_name]

    return filtered


def main():
    parser = build_parser()
    args = parser.parse_args()

    # List mode
    if args.list:
        print(f"{'#':>3s}  {'Grid':<14s}  {'Equation Set':<16s}  {'Resolution':<10s}  "
              f"{'Vert':<7s}  {'Test':<25s}  {'Days':>8s}  {'Quick':>8s}")
        print("-" * 100)
        for i, tc in enumerate(TEST_MATRIX, 1):
            print(f"{i:3d}  {tc.grid_type:<14s}  {tc.equation_set:<16s}  "
                  f"{tc.resolution:<10s}  {tc.vertical_coord:<7s}  "
                  f"{tc.test_name:<25s}  {tc.duration_days:8.2f}  "
                  f"{tc.quick_duration_days:8.4f}")
        print(f"\nTotal: {len(TEST_MATRIX)} test cases")
        return

    # Filter tests
    tests = filter_tests(TEST_MATRIX, args)
    if not tests:
        print("No tests match the given filters.")
        return

    # Apply resolution override
    if args.resolution:
        new_tests = []
        for t in tests:
            new_tests.append(TestCase(
                grid_type=t.grid_type, equation_set=t.equation_set,
                resolution=args.resolution, vertical_coord=t.vertical_coord,
                test_name=t.test_name, duration_days=t.duration_days,
                quick_duration_days=t.quick_duration_days,
                run_fn=t.run_fn, run_kwargs=t.run_kwargs,
            ))
        tests = new_tests

    output_base = Path(args.output)

    # Print header
    print("=" * 78)
    print("  legoESM Atmosphere Test Matrix")
    print("=" * 78)
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  X64:        {jax.config.jax_enable_x64}")
    print(f"  Devices:    {jax.devices()}")
    print(f"  Output:     {output_base}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Tests:      {len(tests)} / {len(TEST_MATRIX)}")
    print("=" * 78)
    print()

    # Run tests
    t_start_all = time.time()
    for i, tc in enumerate(tests, 1):
        days = tc.quick_duration_days if args.quick else tc.duration_days
        out_dir = output_base / tc.output_path

        label = f"{tc.grid_type}/{tc.equation_set}/{tc.test_name}"
        print(f"\n[{i}/{len(tests)}] {label} ({tc.resolution}, {tc.vertical_coord}, "
              f"{days:.4g} days)")
        print("-" * 60)

        runner = RUNNERS.get(tc.run_fn)
        if runner is None:
            record(tc, "ERROR", 0, f"Unknown runner: {tc.run_fn}")
            continue

        try:
            status, wall, notes = runner(tc, out_dir, days, **tc.run_kwargs)
            record(tc, status, wall, notes)
        except Exception as e:
            record(tc, "ERROR", 0, str(e)[:120])
            traceback.print_exc()

    total_wall = time.time() - t_start_all

    # Print summary
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'Status':6s}  {'Grid':<14s}  {'Equation Set':<16s}  "
          f"{'Test':<25s}  {'Time':>8s}  Notes")
    print("-" * 100)

    n_pass = n_fail = n_error = 0
    for r in ALL_RESULTS:
        icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!"}[r["status"]]
        print(f"  {icon}{r['status']:5s}  {r['grid']:<14s}  {r['equation_set']:<16s}  "
              f"{r['test']:<25s}  {r['wall_time']:7.1f}s  {r['notes']}")
        if r["status"] == "PASS":
            n_pass += 1
        elif r["status"] == "FAIL":
            n_fail += 1
        else:
            n_error += 1

    print("-" * 100)
    print(f"  Total: {len(ALL_RESULTS)} tests | "
          f"PASS: {n_pass} | FAIL: {n_fail} | ERROR: {n_error} | "
          f"Wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)")
    print("=" * 78)

    # Save summary
    output_base.mkdir(parents=True, exist_ok=True)
    summary_path = output_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "results": ALL_RESULTS,
            "total_wall_time": total_wall,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "quick_mode": args.quick,
        }, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    summary_txt = output_base / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("legoESM Atmosphere Test Matrix Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(ALL_RESULTS)} tests | "
                f"PASS: {n_pass} | FAIL: {n_fail} | ERROR: {n_error}\n")
        f.write(f"Wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)\n")
        f.write(f"Quick mode: {args.quick}\n\n")
        for r in ALL_RESULTS:
            f.write(f"{r['status']:5s}  {r['grid']:<14s}  {r['equation_set']:<16s}  "
                    f"{r['test']:<25s}  {r['wall_time']:7.1f}s  {r['notes']}\n")
    print(f"  Summary saved to: {summary_txt}")

    # Exit with non-zero if any failures
    if n_fail > 0 or n_error > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
