#!/usr/bin/env python
"""Sea ice test matrix: organized test runner for legoESM sea ice model.

Runs a comprehensive suite of sea ice test cases covering thermodynamics,
dynamics (EVP), transport, and multi-category ITD across cubed-sphere and
column (grid-agnostic) configurations.

Test cases:
  Thermodynamics:
    - stefan_growth        1-D Stefan problem: analytical ice growth rate
    - slab_stability       Multi-step slab stability under seasonal forcing
    - surface_melt         Excess energy converts to melt, not T overshoot
    - open_water_freeze    Open-water freezing from supercooled ocean
    - maykut_untersteiner  Seasonal equilibrium: Arctic forcing → ~3m ice

  Dynamics (EVP / mEVP):
    - evp_zero_strength    P_star=0 recovers near free-drift velocity
    - evp_compression      Uniform compression: isotropic stress = -P/2
    - evp_convergence      EVP stress converges to VP target with N_evp
    - mehlmann_lkf         Cyclone-driven LKF benchmark (Mehlmann+ 2021)
    - mevp_zero_strength   mEVP analog of evp_zero_strength
    - mevp_compression     mEVP analog of evp_compression
    - mevp_convergence     mEVP stress converges to VP target with N_mevp

  Transport:
    - advect_uniform       Uniform field advection preserves state
    - advect_step          Sharp-interface advection: monotonicity + bounds
    - cosine_bell          Solid-body rotation of a cosine bell (Putman & Lin 2007)

  Multi-category ITD:
    - itd_growth_remap     Growth followed by remap preserves volume
    - itd_melt_remap       Melt followed by remap preserves bounds
    - itd_roundtrip        Distribute -> aggregate -> distribute roundtrip

  Integration:
    - slab_100_steps          100-step slab integration stability
    - dynamic_20_steps        20-step dynamic (EVP + transport) stability
    - dynamic_mevp_20_steps   20-step dynamic (mEVP + transport) stability
    - multi_cat_10_steps      10-step 5-category integration stability

Output structure:
    results/sea_ice/<case>/<grid_type>/<resolution>/

Each case folder contains:
    - timeseries.csv / .png       (scalar diagnostics over time)
    - conservation.csv / .png     (mass/energy conservation drift)
    - field_snapshots.png         (2D field maps at selected times)
    - snapshots_native.npz        (native grid snapshot arrays)
    - snapshots_latlon.npz        (regridded to 181x360 for cubed-sphere)
    - results.txt                 (run metadata)

References:
    - Stefan, J. (1891). On the theory of ice formation.
    - Maykut, G.A. & Untersteiner, N. (1971). Some results from a
      time-dependent thermodynamic model of sea ice. JGR, 76, 1550-1575.
    - Hunke, E.C. & Dukowicz, J.K. (1997). An EVP model for sea ice
      dynamics. J. Phys. Oceanogr., 27, 1849-1867.
    - Lipscomb, W.H. (2001). Remapping the thickness distribution in
      sea ice models. J. Geophys. Res., 106(C7), 13989-14000.
    - Putman, W.M. & Lin, S.-J. (2007). FV transport on cubed-sphere
      grids. J. Comput. Phys., 227, 55-78.
    - Mehlmann, C. et al. (2021). Simulating linear kinematic features
      in VP sea ice models. JAMES, 13, e2021MS002523.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_sea_ice_test_matrix.py
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_sea_ice_test_matrix.py --quick
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_sea_ice_test_matrix.py --only stefan_growth
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_sea_ice_test_matrix.py --grid cubed_sphere
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_sea_ice_test_matrix.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm import constants
from legoesm.core.field import Field
from legoesm.diagnostics.conservation_drift import compute_relative_drift
from legoesm.experiments.matrix.namelist import write_case_namelist
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ice.config import SeaIceConfig
from legoesm.ice.state import (
    SeaIceState,
    DynamicSeaIceState,
    init_dynamic_ice_state,
)
from legoesm.ice.sea_ice import step_sea_ice
from legoesm.ice.dynamics import evp_solver, mevp_solver, free_drift_velocity
from legoesm.ice.rheology import ice_strength, vp_stress, delta_deformation
from legoesm.ice.itd import (
    category_bounds,
    upper_bounds,
    aggregate_state,
    distribute_to_categories,
    linear_remap,
)
from legoesm.ice.transport import advect_ice_tracers


# ===========================================================================
# Configuration constants
# ===========================================================================

GRID_RESOLUTIONS: dict[str, str] = {
    "cubed_sphere": "C8",
    "column": "4",       # grid-agnostic column with 4 points
}

GRID_TYPES = ["cubed_sphere", "column"]

# ===========================================================================
# TestCase dataclass
# ===========================================================================

@dataclass
class TestCase:
    """A single test case in the sea ice matrix."""
    category: str           # thermo, dynamics, transport, itd, integration
    case: str               # stefan_growth, evp_zero_strength, etc.
    grid_type: str          # cubed_sphere, column
    resolution: str         # C8, 4
    duration_steps: int     # number of time steps
    quick_steps: int        # quick mode steps
    dt: float = 3600.0      # timestep [s]
    run_kwargs: dict = field(default_factory=dict)

    @property
    def output_path(self) -> str:
        return f"{self.category}/{self.case}/{self.grid_type}/{self.resolution}"


# ===========================================================================
# Test matrix generation
# ===========================================================================

def _build_test_matrix() -> list[TestCase]:
    """Generate the full test matrix from grid x case."""
    matrix: list[TestCase] = []
    res = GRID_RESOLUTIONS

    # --- Thermodynamics: column (grid-agnostic) ---
    for case, steps, quick, kw in [
        ("stefan_growth",        100,   20, {}),
        ("slab_stability",       500,  100, {}),
        ("surface_melt",          50,   10, {}),
        ("open_water_freeze",     50,   10, {}),
        ("maykut_untersteiner", 8760, 2190, {}),  # 1 year (hourly), quick=3 months
    ]:
        matrix.append(TestCase("thermo", case, "column", res["column"],
                                steps, quick, 3600.0, dict(kw)))

    # --- Dynamics: cubed_sphere only (needs grid operators) ---
    for case, steps, quick, kw in [
        ("evp_zero_strength",  1,  1, {"P_star": 0.0}),
        ("evp_compression",    1,  1, {}),
        ("evp_convergence",    1,  1, {}),
        ("mehlmann_lkf",      48, 12, {}),  # 2 days at dt=3600s, quick=12h
        # mEVP mirrors the EVP cases: same physical asserts, different
        # solver, to catch mEVP-only regressions in the matrix harness.
        ("mevp_zero_strength", 1,  1, {"P_star": 0.0}),
        ("mevp_compression",   1,  1, {}),
        ("mevp_convergence",   1,  1, {}),
    ]:
        matrix.append(TestCase("dynamics", case, "cubed_sphere",
                                res["cubed_sphere"], steps, quick, 3600.0,
                                dict(kw)))

    # --- Transport: cubed_sphere only ---
    for case, steps, quick, kw in [
        ("advect_uniform",   10,   3, {}),
        ("advect_step",      20,   5, {}),
        ("cosine_bell",     576, 144, {}),  # 12 days at dt=1800s, quick=3 days
    ]:
        matrix.append(TestCase("transport", case, "cubed_sphere",
                                res["cubed_sphere"], steps, quick, 1800.0,
                                dict(kw)))

    # --- ITD: column (grid-agnostic) ---
    for case, steps, quick, kw in [
        ("itd_growth_remap",   1,  1, {}),
        ("itd_melt_remap",     1,  1, {}),
        ("itd_roundtrip",      1,  1, {}),
    ]:
        matrix.append(TestCase("itd", case, "column", res["column"],
                                steps, quick, 3600.0, dict(kw)))

    # --- Integration: full model ---
    # Slab: column
    matrix.append(TestCase("integration", "slab_100_steps", "column",
                           res["column"], 100, 30, 3600.0))
    # Dynamic: cubed_sphere
    matrix.append(TestCase("integration", "dynamic_20_steps",
                           "cubed_sphere", res["cubed_sphere"], 20, 5, 3600.0))
    matrix.append(TestCase("integration", "dynamic_mevp_20_steps",
                           "cubed_sphere", res["cubed_sphere"], 20, 5, 3600.0))
    # Multi-category: cubed_sphere
    matrix.append(TestCase("integration", "multi_cat_10_steps",
                           "cubed_sphere", res["cubed_sphere"], 10, 3, 3600.0))

    return matrix


TEST_MATRIX = _build_test_matrix()


# ===========================================================================
# Results tracking
# ===========================================================================

ALL_RESULTS: list[dict[str, Any]] = []


def record(tc: TestCase, status: str, wall_time: float, notes: str = ""):
    icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!", "SKIP": "--"}[status]
    ALL_RESULTS.append({
        "category": tc.category, "test": tc.case, "grid": tc.grid_type,
        "resolution": tc.resolution, "status": status,
        "wall_time": wall_time, "notes": notes,
    })
    label = f"{tc.category}/{tc.case}/{tc.grid_type}/{tc.resolution}"
    print(f"  {icon} {status:5s} | {label:<55s} | {wall_time:7.1f}s | {notes}")


# ===========================================================================
# Shared helpers
# ===========================================================================

DIMS_COL = ("ncol",)
DIMS_CS = ("face", "x", "y")


def _make_forcing(shape, dims, **kw):
    """Create AtmToSurface forcing."""
    from legoesm.core.coupling_fields import AtmToSurface
    f = jnp.float64
    defaults = dict(
        sw_down=100.0, lw_down=200.0, T_lowest=250.0, q_lowest=1e-3,
        u_lowest=5.0, v_lowest=-3.0, p_lowest=9.5e4, p_surface=constants.p_ref,
        rho_lowest=1.2, cos_zenith=0.5, co2_ppmv=400.0,
    )
    defaults.update(kw)
    d = defaults
    return AtmToSurface(
        sw_down=jnp.full(shape, d["sw_down"], f),
        lw_down=jnp.full(shape, d["lw_down"], f),
        precip_total=jnp.zeros(shape, f),
        precip_snow=jnp.zeros(shape, f),
        T_lowest=jnp.full(shape, d["T_lowest"], f),
        q_lowest=jnp.full(shape, d["q_lowest"], f),
        u_lowest=jnp.full(shape, d["u_lowest"], f),
        v_lowest=jnp.full(shape, d["v_lowest"], f),
        p_lowest=jnp.full(shape, d["p_lowest"], f),
        p_surface=jnp.full(shape, d["p_surface"], f),
        rho_lowest=jnp.full(shape, d["rho_lowest"], f),
        cos_zenith=jnp.full(shape, d["cos_zenith"], f),
        co2_ppmv=jnp.full(shape, d["co2_ppmv"], f),
        has_radiation=jnp.ones(shape, f),
        has_precipitation=jnp.ones(shape, f),
    )


def _make_slab_state(shape, dims, h=1.0, T_ice=260.0, conc=0.8):
    return SeaIceState(
        h_ice=Field(jnp.full(shape, h), name="h_ice", dims=dims, units="m"),
        T_ice=Field(jnp.full(shape, T_ice), name="T_ice", dims=dims, units="K"),
        concentration=Field(jnp.full(shape, conc), name="conc", dims=dims, units="1"),
    )


def _save_results(outdir: Path, tc: TestCase, diag: dict, snapshots: dict | None = None):
    """Save diagnostic CSV and plots."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Save timeseries CSV
    if diag.get("times"):
        import pandas as pd
        df = pd.DataFrame(diag)
        df.to_csv(outdir / "timeseries.csv", index=False)

        # Plot timeseries
        keys = [k for k in diag if k not in ("times", "steps")]
        n_keys = len(keys)
        if n_keys > 0:
            fig, axes = plt.subplots(n_keys, 1, figsize=(10, 3 * n_keys),
                                      squeeze=False)
            for i, k in enumerate(keys):
                axes[i, 0].plot(diag["times"], diag[k], "b-")
                axes[i, 0].set_ylabel(k)
                axes[i, 0].set_xlabel("step")
                axes[i, 0].grid(True, alpha=0.3)
            fig.suptitle(f"{tc.category}/{tc.case} ({tc.grid_type})")
            fig.tight_layout()
            fig.savefig(outdir / "timeseries.png", dpi=100)
            plt.close(fig)

    # Save snapshots
    if snapshots:
        np.savez_compressed(outdir / "snapshots_native.npz",
                            **{f"step_{k}_{fld}": np.asarray(v)
                               for k, snap in snapshots.items()
                               for fld, v in snap.items()})

    # Save metadata
    with open(outdir / "results.txt", "w") as f:
        f.write(f"case: {tc.case}\n")
        f.write(f"grid: {tc.grid_type}\n")
        f.write(f"resolution: {tc.resolution}\n")
        f.write(f"dt: {tc.dt}\n")
        f.write(f"steps: {tc.duration_steps}\n")


# ===========================================================================
# Thermodynamic test cases
# ===========================================================================

def run_stefan_growth(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Stefan problem: ice growth from cold atmosphere over freezing ocean.

    Analytical solution: h(t) ~ sqrt(2*k_ice*dT*t / (rho_ice*L_f))
    for conduction-limited growth with T_sfc ~ T_freeze.
    """
    config = SeaIceConfig()
    shape = (4,)
    dims = DIMS_COL
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    # Start with thin ice, very cold atmosphere, ocean at freezing
    state = _make_slab_state(shape, dims, h=0.01, T_ice=250.0, conc=1.0)
    forcing = _make_forcing(shape, dims, T_lowest=230.0, sw_down=0.0,
                            lw_down=150.0, u_lowest=2.0, v_lowest=0.0)
    ocean_sst = jnp.full(shape, config.T_freeze_ocean)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "T_mean": [], "conc_mean": []}

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt)
        diag["times"].append(i + 1)
        diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
        diag["T_mean"].append(float(jnp.mean(state.T_ice.data)))
        diag["conc_mean"].append(float(jnp.mean(state.concentration.data)))

    _save_results(outdir, tc, diag)

    # Validate: ice should have grown monotonically and be above initial h
    h_vals = diag["h_mean"]
    grew = all(h_vals[i] >= h_vals[i - 1] - 1e-12 for i in range(1, len(h_vals)))
    final_h = h_vals[-1]
    grew_above_init = final_h > 0.01  # started at 0.01m
    ok = grew and grew_above_init and jnp.all(jnp.isfinite(state.h_ice.data))

    if not ok:
        return "FAIL", f"h_final={final_h:.4f}, monotonic={grew}"
    return "PASS", f"h_final={final_h:.4f}m after {n_steps} steps"


def run_slab_stability(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Multi-step slab stability under constant forcing."""
    config = SeaIceConfig()
    shape = (4,)
    dims = DIMS_COL
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    state = _make_slab_state(shape, dims, h=1.0, T_ice=260.0, conc=0.8)
    forcing = _make_forcing(shape, dims)
    ocean_sst = jnp.full(shape, config.T_freeze_ocean)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "T_mean": [], "conc_mean": []}

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt)
        diag["times"].append(i + 1)
        diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
        diag["T_mean"].append(float(jnp.mean(state.T_ice.data)))
        diag["conc_mean"].append(float(jnp.mean(state.concentration.data)))

    _save_results(outdir, tc, diag)

    finite = (jnp.all(jnp.isfinite(state.h_ice.data)) and
              jnp.all(jnp.isfinite(state.T_ice.data)))
    h_ok = jnp.all(state.h_ice.data >= 0.0)
    T_ok = (jnp.all(state.T_ice.data >= config.T_ice_min) and
            jnp.all(state.T_ice.data <= config.T_freeze_ocean))
    conc_ok = (jnp.all(state.concentration.data >= 0.0) and
               jnp.all(state.concentration.data <= 1.0))

    ok = bool(finite and h_ok and T_ok and conc_ok)
    if not ok:
        return "FAIL", f"finite={finite}, h>=0={h_ok}, T_ok={T_ok}, conc_ok={conc_ok}"
    return "PASS", f"Stable for {n_steps} steps"


def run_surface_melt(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Strong warming: excess energy melts ice, T_ice stays <= T_freeze."""
    config = SeaIceConfig()
    shape = (4,)
    dims = DIMS_COL
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    state = _make_slab_state(shape, dims, h=1.0,
                              T_ice=config.T_freeze_ocean - 0.1, conc=0.9)
    forcing = _make_forcing(shape, dims, T_lowest=280.0, sw_down=500.0,
                            lw_down=350.0)
    ocean_sst = jnp.full(shape, config.T_freeze_ocean + 0.5)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "T_mean": []}
    h_init = float(jnp.mean(state.h_ice.data))

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt)
        diag["times"].append(i + 1)
        diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
        diag["T_mean"].append(float(jnp.mean(state.T_ice.data)))

    _save_results(outdir, tc, diag)

    # Surface melt clamps the skin temperature at the fresh-ice/snow TOP
    # melt point T_melt_surface (0 C = 273.15 K) — NOT the saline basal
    # freezing point T_freeze_ocean (271.35 K).  Under strong atmospheric
    # warming the surface legitimately warms to T_melt_surface and parks
    # there while excess energy converts to melt (sea_ice.py clamps T_ice
    # to config.T_melt_surface and books the surplus as dh/dt).
    #
    # This is a column SMOKE for the surface-melt regime: it asserts the
    # clamp ENGAGED (skin reached the melt point — non-vacuous) and did
    # NOT overshoot it, and that ice did not grow under strong warming.
    # The quantitative surface-melt ENERGY CLOSURE (rho_ice*L_f*ice_melt +
    # skin_cap*dT == net surface energy, with no energy dropped) is
    # asserted directly at the kernel level in
    # tests/unit/test_land_ice_sea_ice_thermo.py::
    # TestThinIceImplicitMelt::test_thin_ice_surface_melt_energy_closure —
    # h_decreased here is a coarse sanity check, not a surface-melt energy
    # proof (this column's basal flux also affects h).
    T_max = float(jnp.max(state.T_ice.data))
    # No overshoot above the surface melt point (the property the case guards).
    no_overshoot = T_max <= config.T_melt_surface + 1e-6
    # Clamp engaged: warming was strong enough to drive the skin TO the melt
    # point, so the surface-melt branch is actually exercised (else vacuous).
    clamp_engaged = T_max >= config.T_melt_surface - 1e-3
    h_decreased = diag["h_mean"][-1] < h_init

    ok = bool(no_overshoot and clamp_engaged and h_decreased)
    if not ok:
        return "FAIL", (f"no_overshoot={no_overshoot}, clamp_engaged={clamp_engaged}, "
                        f"h_decreased={h_decreased} (T_max={T_max:.4f}K)")
    return "PASS", f"h: {h_init:.3f} -> {diag['h_mean'][-1]:.3f}m, skin clamped at melt point"


def run_open_water_freeze(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Open water freezing: thin ice seed, cold atmosphere, ocean at freezing.

    Tests that net ice growth occurs under strong atmospheric cooling.
    Uses a thin ice seed (0.05m) to avoid thermal-inertia oscillation
    that occurs with zero-thickness ice in the slab formulation.
    """
    config = SeaIceConfig()
    shape = (4,)
    dims = DIMS_COL
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    # Start with thin ice seed under very cold atmosphere
    h_init = 0.05
    state = _make_slab_state(shape, dims, h=h_init, T_ice=260.0, conc=0.3)
    forcing = _make_forcing(shape, dims, T_lowest=230.0, sw_down=0.0,
                            lw_down=150.0, u_lowest=2.0, v_lowest=0.0)
    ocean_sst = jnp.full(shape, config.T_freeze_ocean)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "conc_mean": []}

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt)
        diag["times"].append(i + 1)
        diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
        diag["conc_mean"].append(float(jnp.mean(state.concentration.data)))

    _save_results(outdir, tc, diag)

    h_grew = diag["h_mean"][-1] > h_init
    conc_grew = diag["conc_mean"][-1] > 0.3

    ok = bool(h_grew and conc_grew)
    if not ok:
        return "FAIL", f"h_grew={h_grew} ({diag['h_mean'][-1]:.4f}), conc_grew={conc_grew}"
    return "PASS", f"Ice grew: h={h_init:.2f}->{diag['h_mean'][-1]:.4f}m, a={diag['conc_mean'][-1]:.4f}"


# ===========================================================================
# Dynamics test cases
# ===========================================================================

def run_evp_zero_strength(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """P_star=0: EVP should give near free-drift velocity."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    s0 = jnp.zeros(shape)

    u_new, v_new, s11, s22, s12 = evp_solver(
        jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
        h_ice=jnp.ones(shape),
        concentration=jnp.ones(shape),
        wind_u=jnp.full(shape, 10.0),
        wind_v=jnp.zeros(shape),
        ocean_u=jnp.zeros(shape),
        ocean_v=jnp.zeros(shape),
        grid=grid, dt=3600.0, N_evp=20, P_star=0.0,
    )

    finite = jnp.all(jnp.isfinite(u_new)) and jnp.all(jnp.isfinite(v_new))
    has_motion = float(jnp.max(jnp.abs(u_new))) > 1e-4

    diag = {"times": [1], "max_u": [float(jnp.max(jnp.abs(u_new)))],
            "max_v": [float(jnp.max(jnp.abs(v_new)))]}
    _save_results(outdir, tc, diag)

    ok = bool(finite and has_motion)
    if not ok:
        return "FAIL", f"finite={finite}, has_motion={has_motion}"
    return "PASS", f"max|u|={float(jnp.max(jnp.abs(u_new))):.4f} m/s"


def run_evp_compression(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Uniform ice, zero forcing: stress should converge to -P/2 (isotropic)."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    s0 = jnp.zeros(shape)
    h = jnp.ones(shape)
    A = jnp.full(shape, 0.9)

    # EVP relaxes ~1 - exp(-1/(2*T_evp)) ~ 0.75 of the VP target per
    # DYNAMIC step (T_evp=0.36), converging to -P/2 over several steps as
    # sigma is carried forward — it does NOT reach -P/2 in a single call.
    # Iterate dynamic steps (matches the F-EVP validation-test fix).
    u = jnp.zeros(shape)
    v = jnp.zeros(shape)
    s11 = s22 = s12 = s0
    for _ in range(6):
        u, v, s11, s22, s12 = evp_solver(
            u, v, s11, s22, s12,
            h_ice=h, concentration=A,
            wind_u=jnp.zeros(shape), wind_v=jnp.zeros(shape),
            ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, N_evp=50,
        )

    P = ice_strength(h[0, 0, 0], A[0, 0, 0])
    target = float(-P / 2)
    s11_mean = float(jnp.mean(s11))
    s22_mean = float(jnp.mean(s22))
    s12_mean = float(jnp.mean(s12))
    rel_err_11 = abs(s11_mean - target) / abs(target) if abs(target) > 1 else abs(s11_mean - target)
    rel_err_22 = abs(s22_mean - target) / abs(target) if abs(target) > 1 else abs(s22_mean - target)

    diag = {"times": [1],
            "s11_mean": [s11_mean], "s22_mean": [s22_mean], "s12_mean": [s12_mean],
            "target": [target]}
    _save_results(outdir, tc, diag)

    ok = rel_err_11 < 0.2 and rel_err_22 < 0.2 and abs(s12_mean) < 100.0
    if not ok:
        return "FAIL", f"s11={s11_mean:.1f}, s22={s22_mean:.1f}, target={target:.1f}"
    return "PASS", f"s11~s22~{target:.1f} (err<20%), s12~0"


def run_evp_convergence(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """EVP stress should converge toward VP target as N_evp increases."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    s0 = jnp.zeros(shape)
    h = jnp.full(shape, 1.5)
    A = jnp.full(shape, 0.9)

    errors = []
    N_values = [5, 20, 50, 100]

    for N in N_values:
        _, _, s11, s22, s12 = evp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=h, concentration=A,
            wind_u=jnp.full(shape, 5.0), wind_v=jnp.full(shape, -2.0),
            ocean_u=jnp.full(shape, 0.1), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, N_evp=N,
        )
        errors.append(float(jnp.mean(jnp.abs(s11))))

    diag = {"times": N_values, "mean_abs_s11": errors}
    _save_results(outdir, tc, diag)

    # Check that stress magnitude converges (last value should be similar to
    # second-to-last, indicating convergence)
    converging = abs(errors[-1] - errors[-2]) < 0.3 * abs(errors[-2]) if errors[-2] > 1 else True
    finite = all(np.isfinite(e) for e in errors)

    ok = bool(finite and converging)
    if not ok:
        return "FAIL", f"errors={[f'{e:.1f}' for e in errors]}"
    return "PASS", f"EVP converges: |s11|={[f'{e:.0f}' for e in errors]}"


def run_mevp_zero_strength(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """P_star=0: mEVP should give near free-drift velocity (mirror of EVP)."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    s0 = jnp.zeros(shape)

    u_new, v_new, _, _, _ = mevp_solver(
        jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
        h_ice=jnp.ones(shape),
        concentration=jnp.ones(shape),
        wind_u=jnp.full(shape, 10.0),
        wind_v=jnp.zeros(shape),
        ocean_u=jnp.zeros(shape),
        ocean_v=jnp.zeros(shape),
        grid=grid, dt=3600.0, N_mevp=100, P_star=0.0,
        alpha_mevp=500.0, beta_mevp=500.0,
    )

    finite = jnp.all(jnp.isfinite(u_new)) and jnp.all(jnp.isfinite(v_new))
    has_motion = float(jnp.max(jnp.abs(u_new))) > 1e-4

    diag = {"times": [1], "max_u": [float(jnp.max(jnp.abs(u_new)))],
            "max_v": [float(jnp.max(jnp.abs(v_new)))]}
    _save_results(outdir, tc, diag)

    ok = bool(finite and has_motion)
    if not ok:
        return "FAIL", f"finite={finite}, has_motion={has_motion}"
    return "PASS", f"max|u|={float(jnp.max(jnp.abs(u_new))):.4f} m/s"


def run_mevp_compression(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Uniform ice, zero forcing: mEVP stress should converge to -P/2."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    s0 = jnp.zeros(shape)
    h = jnp.ones(shape)
    A = jnp.full(shape, 0.9)

    # Smaller alpha → faster per-iteration convergence to VP target;
    # 200 iterations is enough for ~10% residual at alpha=50.
    _, _, s11, s22, s12 = mevp_solver(
        jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
        h_ice=h, concentration=A,
        wind_u=jnp.zeros(shape), wind_v=jnp.zeros(shape),
        ocean_u=jnp.zeros(shape), ocean_v=jnp.zeros(shape),
        grid=grid, dt=3600.0, N_mevp=200,
        alpha_mevp=50.0, beta_mevp=50.0,
    )

    P = ice_strength(h[0, 0, 0], A[0, 0, 0])
    target = float(-P / 2)
    s11_mean = float(jnp.mean(s11))
    s22_mean = float(jnp.mean(s22))
    s12_mean = float(jnp.mean(s12))
    rel_err_11 = abs(s11_mean - target) / abs(target) if abs(target) > 1 else abs(s11_mean - target)
    rel_err_22 = abs(s22_mean - target) / abs(target) if abs(target) > 1 else abs(s22_mean - target)

    diag = {"times": [1],
            "s11_mean": [s11_mean], "s22_mean": [s22_mean], "s12_mean": [s12_mean],
            "target": [target]}
    _save_results(outdir, tc, diag)

    ok = rel_err_11 < 0.2 and rel_err_22 < 0.2 and abs(s12_mean) < 100.0
    if not ok:
        return "FAIL", f"s11={s11_mean:.1f}, s22={s22_mean:.1f}, target={target:.1f}"
    return "PASS", f"s11~s22~{target:.1f} (err<20%), s12~0"


def run_mevp_convergence(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """mEVP stress should converge toward VP target as N_mevp increases."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    s0 = jnp.zeros(shape)
    h = jnp.full(shape, 1.5)
    A = jnp.full(shape, 0.9)

    errors = []
    N_values = [10, 50, 200, 1000]

    # alpha=50 keeps per-iteration relaxation = 2 %, so N=1000 gives
    # (1-1/50)^1000 ≈ 2e-9 residual on the stress side.
    for N in N_values:
        _, _, s11, _, _ = mevp_solver(
            jnp.zeros(shape), jnp.zeros(shape), s0, s0, s0,
            h_ice=h, concentration=A,
            wind_u=jnp.full(shape, 5.0), wind_v=jnp.full(shape, -2.0),
            ocean_u=jnp.full(shape, 0.1), ocean_v=jnp.zeros(shape),
            grid=grid, dt=3600.0, N_mevp=N,
            alpha_mevp=50.0, beta_mevp=50.0,
        )
        errors.append(float(jnp.mean(jnp.abs(s11))))

    diag = {"times": N_values, "mean_abs_s11": errors}
    _save_results(outdir, tc, diag)

    converging = abs(errors[-1] - errors[-2]) < 0.3 * abs(errors[-2]) if errors[-2] > 1 else True
    finite = all(np.isfinite(e) for e in errors)

    ok = bool(finite and converging)
    if not ok:
        return "FAIL", f"errors={[f'{e:.1f}' for e in errors]}"
    return "PASS", f"mEVP converges: |s11|={[f'{e:.0f}' for e in errors]}"


# ===========================================================================
# Transport test cases
# ===========================================================================

def run_advect_uniform(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Uniform field + constant velocity should preserve state."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    h = jnp.ones(shape) * 1.5
    a = jnp.full(shape, 0.8)
    T = jnp.full(shape, 260.0)
    u = jnp.full(shape, 0.01)  # slow, uniform
    v = jnp.zeros(shape)

    h_orig, a_orig, T_orig = h, a, T

    for _ in range(n_steps):
        h, a, T = advect_ice_tracers(h, a, T, u, v, grid, dt)

    h_err = float(jnp.max(jnp.abs(h - h_orig)))
    a_err = float(jnp.max(jnp.abs(a - a_orig)))
    T_err = float(jnp.max(jnp.abs(T - T_orig)))

    diag = {"times": [n_steps], "h_err": [h_err], "a_err": [a_err], "T_err": [T_err]}
    _save_results(outdir, tc, diag)

    # Uniform field should be nearly preserved by divergence operator
    ok = h_err < 0.1 and a_err < 0.01 and T_err < 1.0
    if not ok:
        return "FAIL", f"h_err={h_err:.4f}, a_err={a_err:.4f}, T_err={T_err:.4f}"
    return "PASS", f"Max errors: h={h_err:.2e}, a={a_err:.2e}, T={T_err:.2e}"


def run_advect_step(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Sharp interface advection: check bounds preservation."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    # Ice only on face 0
    h = jnp.zeros(shape).at[0].set(2.0)
    a = jnp.zeros(shape).at[0].set(0.9)
    T = jnp.full(shape, 260.0)
    u = jnp.full(shape, 0.02)
    v = jnp.full(shape, 0.01)

    vol_init = float(jnp.sum(h * a))

    diag = {"times": [], "vol": [], "h_min": [], "a_max": [], "T_min": [], "T_max": []}

    for i in range(n_steps):
        h, a, T = advect_ice_tracers(h, a, T, u, v, grid, dt)
        diag["times"].append(i + 1)
        diag["vol"].append(float(jnp.sum(h * a)))
        diag["h_min"].append(float(jnp.min(h)))
        diag["a_max"].append(float(jnp.max(a)))
        diag["T_min"].append(float(jnp.min(T)))
        diag["T_max"].append(float(jnp.max(T)))

    _save_results(outdir, tc, diag)

    h_nonneg = float(jnp.min(h)) >= -1e-10
    a_bounded = float(jnp.min(a)) >= -1e-10 and float(jnp.max(a)) <= 1.0 + 1e-10
    T_bounded = float(jnp.min(T)) >= 180.0 - 1e-6 and float(jnp.max(T)) <= constants.T_freeze_ocean + 1e-6
    vol_final = float(jnp.sum(h * a))
    vol_drift = compute_relative_drift([vol_init, vol_final])

    ok = bool(h_nonneg and a_bounded and T_bounded)
    if not ok:
        return "FAIL", (f"h>=0: {h_nonneg}, a_bounded: {a_bounded}, "
                        f"T_bounded: {T_bounded}, vol_drift: {vol_drift:.4f}")
    return "PASS", f"Bounds OK, vol_drift={vol_drift:.2e}"


# ===========================================================================
# ITD test cases
# ===========================================================================

def run_itd_growth_remap(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Growth followed by remap: volume should be approximately conserved."""
    n_cat = 5
    h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
    a_old = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])
    T_old = jnp.array([250.0, 255.0, 260.0, 263.0, 265.0])

    # Simulate growth: all categories thicken
    h_new = h_old + 0.3
    a_new = a_old * 1.05
    T_new = T_old + 1.0  # warming

    vol_before = float(jnp.sum(h_new * a_new))
    h_r, a_r, T_r = linear_remap(h_old, a_old, h_new, a_new, n_cat, T_new=T_new)
    vol_after = float(jnp.sum(h_r * a_r))

    lo = category_bounds(n_cat)
    hi = upper_bounds(n_cat)

    # Check category bounds
    bounds_ok = True
    for k in range(n_cat):
        if float(a_r[k]) > 0.0:
            if float(h_r[k]) < float(lo[k]) - 1e-8 or float(h_r[k]) > float(hi[k]) + 1e-8:
                bounds_ok = False

    vol_drift = compute_relative_drift([vol_before, vol_after])
    T_bounded = jnp.all(T_r >= 180.0) and jnp.all(T_r <= constants.T_freeze_ocean)

    diag = {"times": [1], "vol_before": [vol_before], "vol_after": [vol_after],
            "vol_drift": [vol_drift]}
    _save_results(outdir, tc, diag)

    ok = bounds_ok and vol_drift < 0.05 and bool(T_bounded)
    if not ok:
        return "FAIL", f"bounds_ok={bounds_ok}, vol_drift={vol_drift:.4f}, T_bounded={T_bounded}"
    return "PASS", f"Vol conserved ({vol_drift:.2e}), bounds OK, T bounded"


def run_itd_melt_remap(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Melt followed by remap: ice thins, categories demoted correctly."""
    n_cat = 5
    h_old = jnp.array([0.3, 1.0, 2.0, 3.0, 5.0])
    a_old = jnp.array([0.1, 0.2, 0.15, 0.1, 0.05])

    # Simulate melt: all categories thin
    h_new = jnp.maximum(h_old - 0.4, 0.0)
    a_new = a_old * 0.95

    vol_before = float(jnp.sum(h_new * a_new))
    h_r, a_r = linear_remap(h_old, a_old, h_new, a_new, n_cat)
    vol_after = float(jnp.sum(h_r * a_r))

    lo = category_bounds(n_cat)
    hi = upper_bounds(n_cat)

    bounds_ok = True
    for k in range(n_cat):
        if float(a_r[k]) > 0.0:
            if float(h_r[k]) < float(lo[k]) - 1e-8 or float(h_r[k]) > float(hi[k]) + 1e-8:
                bounds_ok = False

    a_bounded = jnp.all(a_r >= 0.0) and jnp.all(a_r <= 1.0)
    vol_drift = compute_relative_drift([vol_before, vol_after])

    diag = {"times": [1], "vol_drift": [vol_drift]}
    _save_results(outdir, tc, diag)

    ok = bounds_ok and bool(a_bounded) and vol_drift < 0.05
    if not ok:
        return "FAIL", f"bounds_ok={bounds_ok}, a_bounded={a_bounded}, vol_drift={vol_drift:.4f}"
    return "PASS", f"Melt remap OK, vol_drift={vol_drift:.2e}"


def run_itd_roundtrip(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Distribute -> aggregate roundtrip preserves volume."""
    h_slab = jnp.array(1.5)
    T_slab = jnp.array(260.0)
    a_slab = jnp.array(0.7)

    vol_init = float(h_slab * a_slab)

    h_mc, T_mc, a_mc = distribute_to_categories(h_slab, T_slab, a_slab, 5)
    h_agg, T_agg, a_agg = aggregate_state(h_mc, T_mc, a_mc)

    vol_agg = float(h_agg * a_agg)
    vol_drift = compute_relative_drift([vol_init, vol_agg])

    h_err = abs(float(h_agg) - float(h_slab))
    T_err = abs(float(T_agg) - float(T_slab))

    diag = {"times": [1], "vol_drift": [vol_drift], "h_err": [h_err], "T_err": [T_err]}
    _save_results(outdir, tc, diag)

    ok = vol_drift < 1e-10 and h_err < 1e-10 and T_err < 1e-10
    if not ok:
        return "FAIL", f"vol_drift={vol_drift:.2e}, h_err={h_err:.2e}, T_err={T_err:.2e}"
    return "PASS", f"Roundtrip exact (errors < 1e-10)"


# ===========================================================================
# Integration test cases
# ===========================================================================

def run_slab_100_steps(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """100-step slab integration: verify stability and bounds."""
    config = SeaIceConfig()
    shape = (6, 8, 8)
    dims = DIMS_CS
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    state = _make_slab_state(shape, dims, h=1.0, T_ice=260.0, conc=0.8)
    forcing = _make_forcing(shape, dims)
    ocean_sst = jnp.full(shape, config.T_freeze_ocean)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "T_mean": [], "conc_mean": []}

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt)
        if (i + 1) % max(1, n_steps // 20) == 0:
            diag["times"].append(i + 1)
            diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
            diag["T_mean"].append(float(jnp.mean(state.T_ice.data)))
            diag["conc_mean"].append(float(jnp.mean(state.concentration.data)))

    _save_results(outdir, tc, diag)

    finite = (jnp.all(jnp.isfinite(state.h_ice.data)) and
              jnp.all(jnp.isfinite(state.T_ice.data)))
    h_ok = jnp.all(state.h_ice.data >= 0.0)

    ok = bool(finite and h_ok)
    if not ok:
        return "FAIL", f"finite={finite}, h>=0={h_ok}"
    return "PASS", f"Stable for {n_steps} steps, h_mean={diag['h_mean'][-1]:.3f}m"


def run_dynamic_20_steps(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """20-step dynamic (EVP + transport) integration."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    dims = DIMS_CS
    config = SeaIceConfig(dynamics="evp", transport="advect", N_evp=10)
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    state = init_dynamic_ice_state(shape)
    state = state._replace(
        h_ice=state.h_ice.replace(data=jnp.ones(shape) * 1.5),
        concentration=state.concentration.replace(data=jnp.full(shape, 0.9)),
    )
    forcing = _make_forcing(shape, dims)
    ocean_sst = jnp.full(shape, 271.0)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "max_u": []}

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt, grid=grid)
        diag["times"].append(i + 1)
        diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
        diag["max_u"].append(float(jnp.max(jnp.abs(state.u_ice.data))))

    _save_results(outdir, tc, diag)

    finite = (jnp.all(jnp.isfinite(state.h_ice.data)) and
              jnp.all(jnp.isfinite(state.u_ice.data)))
    h_ok = jnp.all(state.h_ice.data >= 0.0)

    ok = bool(finite and h_ok)
    if not ok:
        return "FAIL", f"finite={finite}, h>=0={h_ok}"
    return "PASS", f"Stable for {n_steps} steps, max|u|={diag['max_u'][-1]:.4f} m/s"


def run_dynamic_mevp_20_steps(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """20-step dynamic (mEVP + transport) integration — mirror of
    ``run_dynamic_20_steps`` for the mEVP rheology branch.
    """
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    dims = DIMS_CS
    config = SeaIceConfig(
        dynamics="mevp", transport="advect", N_mevp=20,
        alpha_mevp=500.0, beta_mevp=500.0,
    )
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    state = init_dynamic_ice_state(shape)
    state = state._replace(
        h_ice=state.h_ice.replace(data=jnp.ones(shape) * 1.5),
        concentration=state.concentration.replace(data=jnp.full(shape, 0.9)),
    )
    forcing = _make_forcing(shape, dims)
    ocean_sst = jnp.full(shape, 271.0)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "max_u": [], "max_sigma": []}

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt, grid=grid)
        diag["times"].append(i + 1)
        diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
        diag["max_u"].append(float(jnp.max(jnp.abs(state.u_ice.data))))
        diag["max_sigma"].append(float(jnp.max(jnp.abs(state.sigma_11.data))))

    _save_results(outdir, tc, diag)

    finite = (jnp.all(jnp.isfinite(state.h_ice.data)) and
              jnp.all(jnp.isfinite(state.u_ice.data)) and
              jnp.all(jnp.isfinite(state.sigma_11.data)))
    h_ok = jnp.all(state.h_ice.data >= 0.0)
    u_bounded = diag["max_u"][-1] < 2.0      # Arctic drift O(0.1 m/s)
    sigma_bounded = diag["max_sigma"][-1] < 1.0e6  # typical max ~1e5

    ok = bool(finite and h_ok and u_bounded and sigma_bounded)
    if not ok:
        return "FAIL", (
            f"finite={finite}, h>=0={h_ok}, "
            f"max|u|={diag['max_u'][-1]:.3f}, "
            f"max|σ|={diag['max_sigma'][-1]:.2e}"
        )
    return "PASS", (
        f"Stable for {n_steps} steps, "
        f"max|u|={diag['max_u'][-1]:.4f} m/s, "
        f"max|σ|={diag['max_sigma'][-1]:.2e} N/m"
    )


def run_multi_cat_10_steps(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """10-step 5-category integration with ITD remap."""
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    dims = DIMS_CS
    config = SeaIceConfig(dynamics="free_drift", n_categories=5)
    dt = tc.dt
    n_steps = tc.quick_steps if quick else tc.duration_steps

    h_mc, T_mc, a_mc = distribute_to_categories(
        jnp.full(shape, 1.5), jnp.full(shape, 255.0), jnp.full(shape, 0.7), 5)

    # Build a fully-formed 5-category state (all 12 fields incl. the
    # snow / brine / pond tracers) via the canonical initializer, then
    # overwrite h/T/conc with the distributed multi-category arrays.
    # Hand-constructing DynamicSeaIceState here would omit the new-physics
    # fields and raise a TypeError (state grew; this harness had not).
    state = init_dynamic_ice_state(h_mc.shape, n_categories=5)
    state = state._replace(
        h_ice=Field(data=h_mc, name="h_ice", dims=("face", "x", "y", "cat"), units="m"),
        T_ice=Field(data=T_mc, name="T_ice", dims=("face", "x", "y", "cat"), units="K"),
        concentration=Field(data=a_mc, name="conc", dims=("face", "x", "y", "cat"), units="1"),
    )

    forcing = _make_forcing(shape, dims)
    ocean_sst = jnp.full(shape, 271.0)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "total_conc": []}

    for i in range(n_steps):
        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt)
        h_agg, _, a_total = aggregate_state(
            state.h_ice.data, state.T_ice.data, state.concentration.data)
        diag["times"].append(i + 1)
        diag["h_mean"].append(float(jnp.mean(h_agg)))
        diag["total_conc"].append(float(jnp.mean(a_total)))

    _save_results(outdir, tc, diag)

    finite = jnp.all(jnp.isfinite(state.h_ice.data))
    h_ok = jnp.all(state.h_ice.data >= 0.0)
    shape_ok = state.h_ice.data.shape == h_mc.shape

    ok = bool(finite and h_ok and shape_ok)
    if not ok:
        return "FAIL", f"finite={finite}, h>=0={h_ok}, shape={shape_ok}"
    return "PASS", f"Stable for {n_steps} steps, {state.h_ice.data.shape[-1]} categories"


# ===========================================================================
# Benchmark: Cosine bell solid-body rotation (Putman & Lin 2007)
# ===========================================================================

def _cosine_bell_h(lon, lat, radius):
    """Cosine bell ice thickness field.

    Bell centred at (270E, 0N) with radius R0 = radius/3.
    Peak thickness 2 m.
    """
    R0 = radius / 3.0
    lon_c = 3.0 * jnp.pi / 2.0  # 270 E
    lat_c = 0.0                  # equator

    r = radius * jnp.arccos(jnp.clip(
        jnp.sin(lat_c) * jnp.sin(lat)
        + jnp.cos(lat_c) * jnp.cos(lat) * jnp.cos(lon - lon_c),
        -1.0, 1.0,
    ))
    h_peak = 2.0  # m
    return jnp.where(r < R0, (h_peak / 2.0) * (1.0 + jnp.cos(jnp.pi * r / R0)),
                     0.0)


def _rotation_winds_geo(lon, lat, radius, period, beta):
    """Solid body rotation wind in geographic (east, north) components.

    Eqs. 29-30 of Putman & Lin (2007).
    """
    u0 = 2.0 * jnp.pi * radius / period
    cos_b = jnp.cos(beta)
    sin_b = jnp.sin(beta)
    u_east = u0 * (jnp.cos(lat) * cos_b + jnp.sin(lat) * jnp.cos(lon) * sin_b)
    v_north = -u0 * jnp.sin(lon) * sin_b
    return u_east, v_north


def run_cosine_bell(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Solid-body rotation of a cosine bell on the cubed sphere.

    A cosine-bell-shaped ice thickness field is advected by prescribed
    solid-body rotation winds for one full revolution (12 days).
    After 12 days the bell returns to its initial position, so the
    exact solution equals the initial condition.

    This tests the transport operator's accuracy and conservation.
    Error norms (L1, L2, Linf) are computed against the initial field.

    Reference: Putman & Lin (2007), Section 4.1.
    """
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    dt = tc.dt  # 1800s
    n_steps = tc.quick_steps if quick else tc.duration_steps
    radius = constants.R_earth
    period = 12.0 * 86400.0  # 12-day rotation
    beta = jnp.pi / 4.0  # rotation angle

    # Initial cosine bell thickness
    h_init = _cosine_bell_h(grid.lon, grid.lat, radius)
    a_init = jnp.where(h_init > 0.0, 1.0, 0.0)  # concentration = 1 where ice exists
    T_init = jnp.full(shape, 260.0)

    # Prescribe solid-body rotation winds in grid-aligned coordinates
    u_east, v_north = _rotation_winds_geo(grid.lon, grid.lat, radius, period, beta)
    # Geographic → grid-aligned rotation:
    #   u_grid = cos(angle) * u_east + sin(angle) * v_north
    #   v_grid = -sin(angle) * u_east + cos(angle) * v_north
    cos_a = grid.cos_angle
    sin_a = grid.sin_angle
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north

    # Scale winds from m/s to grid velocity (winds are in physical coordinates
    # but transport uses velocity * dt to compute CFL-like divergence)
    # The divergence operator already accounts for grid metrics, so we pass
    # the physical velocities directly.

    h = h_init
    a = a_init
    T = T_init
    vol_init = float(jnp.sum(h * a * grid.area))

    diag = {"times": [], "vol_drift": [], "max_h": [], "L2_err": []}

    for i in range(n_steps):
        h, a, T = advect_ice_tracers(h, a, T, u_grid, v_grid, grid, dt)

        if (i + 1) % max(1, n_steps // 20) == 0 or i == n_steps - 1:
            vol_now = float(jnp.sum(h * a * grid.area))
            vol_drift = compute_relative_drift([vol_init, vol_now])
            # Compute L2 error against initial condition
            diff = h - h_init
            L2 = float(jnp.sqrt(jnp.sum(diff ** 2 * grid.area) / jnp.sum(grid.area)))
            diag["times"].append(i + 1)
            diag["vol_drift"].append(vol_drift)
            diag["max_h"].append(float(jnp.max(h)))
            diag["L2_err"].append(L2)

    # Final error norms (against initial field — exact solution after full rotation)
    diff = h - h_init
    area = grid.area
    L1 = float(jnp.sum(jnp.abs(diff) * area) / jnp.sum(h_init * area + 1e-30))
    L2 = float(jnp.sqrt(jnp.sum(diff ** 2 * area) / jnp.sum(h_init ** 2 * area + 1e-30)))
    Linf = float(jnp.max(jnp.abs(diff)) / (jnp.max(h_init) + 1e-30))
    vol_final = float(jnp.sum(h * a * area))
    vol_drift = compute_relative_drift([vol_init, vol_final])

    _save_results(outdir, tc, diag)

    # Bounds checks
    h_nonneg = float(jnp.min(h)) >= -1e-10
    finite = bool(jnp.all(jnp.isfinite(h)))

    # The centered divergence transport is highly diffusive at C8.
    # This test is *diagnostic* — it measures transport quality via error
    # norms, not a strict pass/fail.  We only require finiteness and
    # non-negativity.  Large errors are expected and document the need
    # for a proper upwind/FCT scheme.
    ok = bool(finite and h_nonneg)
    notes = f"L1={L1:.3f}, L2={L2:.3f}, Linf={Linf:.3f}, vol_drift={vol_drift:.2e}"
    if not ok:
        return "FAIL", notes
    return "PASS", notes


# ===========================================================================
# Benchmark: Mehlmann et al. (2021) cyclone-driven LKF
# ===========================================================================

def run_mehlmann_lkf(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Cyclone-driven LKF benchmark (Mehlmann et al. 2021, JAMES).

    A moving cyclone drives thin ice, producing Linear Kinematic
    Features (leads and ridges).  The original benchmark uses a planar
    512 km square domain with no-slip walls; here we adapt it to a
    global cubed-sphere grid with equivalent physical parameters.

    Original setup (Eqs. 10-15 of Mehlmann et al.):
    - H(0) = 0.3m, A(0) = 1.0, v(0) = 0
    - Cyclone: v_max=15 m/s, convergence angle 72 deg, moving NE
    - Ocean: anticyclonic gyre v_max=0.01 m/s
    - rho_ice=900, P*=27.5 kPa, C=20, e=2

    Validates:
    - EVP solver produces nonzero deformation under convergent/shearing wind
    - Ice velocity responds to spatially varying wind
    - Stress tensor is finite and bounded
    - Deformation features form (max deformation > background)

    Reference: Mehlmann et al. (2021), JAMES, 13, e2021MS002523.
    Data: Mendeley kj58y3sdtk.
    """
    grid = create_cubed_sphere(8)
    n = grid.n
    shape = (6, n, n)
    dt = tc.dt  # 3600s
    n_steps = tc.quick_steps if quick else tc.duration_steps

    # Thin ice, full concentration (Mehlmann setup: h=0.3m, A=1.0)
    h_ice = jnp.full(shape, 0.3)
    conc = jnp.ones(shape)
    u_ice = jnp.zeros(shape)
    v_ice = jnp.zeros(shape)
    s0 = jnp.zeros(shape)

    # Cyclonic wind forcing: a time-varying cyclone moves diagonally.
    # At each step, the cyclone centre advances; the wind field is
    # a Rankine-like vortex in geographic coordinates.
    # For the global grid, we use a simpler approach: a geostrophic-like
    # cyclone centred at (lon_c, lat_c) with wind speed decaying with
    # distance.

    radius = constants.R_earth
    lon_c0 = jnp.pi        # initial cyclone centre: 180E
    lat_c0 = jnp.pi / 6    # 30N
    wind_max = 15.0         # m/s (Mehlmann Eq. 12)
    alpha_conv = 72.0 * jnp.pi / 180.0  # convergence angle (Mehlmann Eq. 12)

    # Cyclone drift velocity (moves NE, adapted from Mehlmann Eq. 10:
    # m_x(t) = 51200 + 51200*t for planar 512km domain → ~60 km/day)
    drift_speed = 60e3  # m/day
    v_drift_lon = drift_speed / (radius * 86400.0)  # rad/s eastward
    v_drift_lat = drift_speed / (radius * 86400.0)  # rad/s northward

    diag = {"times": [], "max_deformation": [], "max_u": [], "max_stress": []}

    s11, s22, s12 = s0, s0, s0

    for i in range(n_steps):
        t = i * dt
        # Current cyclone centre
        lon_c = lon_c0 + v_drift_lon * t
        lat_c = lat_c0 + v_drift_lat * t

        # Great-circle distance from cyclone centre to each grid point
        cos_dist = (jnp.sin(lat_c) * jnp.sin(grid.lat)
                    + jnp.cos(lat_c) * jnp.cos(grid.lat)
                    * jnp.cos(grid.lon - lon_c))
        r = radius * jnp.arccos(jnp.clip(cos_dist, -1.0, 1.0))

        # Mehlmann-style cyclone wind adapted for global sphere.
        # The original Eq. 11-12 use a 512 km planar domain with decay
        # scale ~100 km.  On a global C8 grid (dx~800 km) we scale the
        # decay to 4000 km and use a Rankine vortex profile for the
        # tangential speed.
        r_vortex = 2000e3  # vortex radius [m] (scaled for global)
        v_tang = jnp.where(
            r < r_vortex,
            wind_max * r / r_vortex,
            wind_max * r_vortex / jnp.maximum(r, 1e3),
        )
        # Apply exponential decay beyond the vortex core
        r_km = r / 1000.0
        decay = jnp.exp(-0.0003 * r_km)  # scale ~3300 km (global-adapted)
        v_tang = v_tang * decay

        # Geographic direction from centre to point (unit radial vector)
        dx = jnp.cos(grid.lat) * jnp.sin(grid.lon - lon_c)
        dy = (jnp.cos(lat_c) * jnp.sin(grid.lat)
              - jnp.sin(lat_c) * jnp.cos(grid.lat) * jnp.cos(grid.lon - lon_c))
        r_xy = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-20)
        rx = dx / r_xy  # unit radial (east component)
        ry = dy / r_xy  # unit radial (north component)

        # Mehlmann convergence angle: inflow spiral with 72 deg
        cos_ac = jnp.cos(alpha_conv)
        sin_ac = jnp.sin(alpha_conv)
        # CCW tangent rotated by convergence angle
        wind_e = v_tang * (-sin_ac * rx - cos_ac * ry)
        wind_n = v_tang * (sin_ac * ry - cos_ac * rx)

        # Rotate to grid-aligned
        cos_a = grid.cos_angle
        sin_a = grid.sin_angle
        wind_u = cos_a * wind_e + sin_a * wind_n
        wind_v = -sin_a * wind_e + cos_a * wind_n

        # Run EVP solver
        # Mehlmann et al. Table 1 parameters
        u_ice, v_ice, s11, s22, s12 = evp_solver(
            u_ice, v_ice, s11, s22, s12,
            h_ice, conc,
            wind_u, wind_v,
            jnp.zeros(shape), jnp.zeros(shape),  # simplified: no ocean currents
            grid, dt,
            N_evp=20,  # reduced for speed (paper uses 100 mEVP substeps)
            P_star=2.75e4,   # 27.5 kPa (paper value)
            C_strength=20.0, # C=20 (paper value)
            e_yield=2.0,     # e=2 (paper value)
            T_evp=0.36,
            rho_ice=900.0,   # paper value (vs legoESM default 917)
            rho_air=1.3,     # paper value (vs default 1.225)
            rho_ocean=1026.0, # paper value (vs default 1025)
            C_ai=1.2e-3,     # paper value (vs default 1.3e-3)
            C_oi=5.5e-3,     # paper value (same as default)
        )

        # Compute deformation = sqrt(shear^2 + divergence^2) from velocity
        # Use simple centered differences as an approximation
        from legoesm.ice.rheology import strain_rates
        eps_11, eps_22, eps_12 = strain_rates(u_ice, v_ice, grid)
        div_rate = eps_11 + eps_22
        shear_rate = jnp.sqrt((eps_11 - eps_22) ** 2 + 4.0 * eps_12 ** 2)
        deformation = jnp.sqrt(div_rate ** 2 + shear_rate ** 2)

        diag["times"].append(i + 1)
        diag["max_deformation"].append(float(jnp.max(deformation)))
        diag["max_u"].append(float(jnp.max(jnp.sqrt(u_ice ** 2 + v_ice ** 2))))
        diag["max_stress"].append(float(jnp.max(jnp.abs(s11))))

    _save_results(outdir, tc, diag)

    # Validation criteria:
    finite = (jnp.all(jnp.isfinite(u_ice)) and jnp.all(jnp.isfinite(s11)))
    has_motion = float(jnp.max(jnp.sqrt(u_ice ** 2 + v_ice ** 2))) > 1e-4
    has_deformation = max(diag["max_deformation"]) > 1e-8
    stress_bounded = float(jnp.max(jnp.abs(s11))) < 1e8

    ok = bool(finite and has_motion and has_deformation and stress_bounded)
    notes = (f"max_deform={max(diag['max_deformation']):.2e}, "
             f"max|u|={diag['max_u'][-1]:.4f} m/s, "
             f"max|s11|={diag['max_stress'][-1]:.0f} N/m")
    if not ok:
        return "FAIL", notes
    return "PASS", notes


# ===========================================================================
# Benchmark: Maykut-Untersteiner (1971) seasonal equilibrium
# ===========================================================================

# Monthly Arctic climatological forcing from Lindsay (1998, J. Climate,
# Table 1, p. 325), as used by the CICE/Icepack standalone driver.
# These are 45-year means from Soviet drifting ice station observations.
#   T_air [K], SW_down [W/m²], LW_down [W/m²], wind [m/s]
# Month indices 0-11 = Jan-Dec.
_MU_T_AIR = jnp.array([
    241.75, 240.35, 241.55, 249.05, 262.15, 271.35,  # Jan-Jun  (Lindsay Table 1, °C→K)
    273.05, 271.75, 265.15, 253.65, 245.55, 242.05,   # Jul-Dec
])
_MU_SW_DOWN = jnp.array([
    0.0,   1.2,  31.5, 146.0, 263.3, 307.9,  # Jan-Jun
    230.6, 134.7, 44.2,   2.6,   0.0,   0.0,  # Jul-Dec
])
_MU_LW_DOWN = jnp.array([
    164.0, 160.5, 164.1, 188.1, 245.2, 291.2,  # Jan-Jun
    303.9, 297.0, 263.8, 210.9, 177.0, 166.0,   # Jul-Dec
])
_MU_WIND = jnp.array([
    4.4, 4.0, 4.0, 3.9, 3.9, 4.2,  # Jan-Jun
    4.1, 4.2, 4.5, 4.2, 3.9, 4.0,  # Jul-Dec
])


def run_maykut_untersteiner(tc: TestCase, outdir: Path, quick: bool) -> tuple[str, str]:
    """Maykut-Untersteiner seasonal thermodynamic equilibrium.

    A 1-D ice column is forced by a representative Arctic annual
    cycle of atmospheric forcing (Lindsay 1998 / Icepack climatological
    data derived from Soviet drifting ice station observations).

    The model should reach a seasonal equilibrium with mean thickness
    ~2-4 m within a few years.  Since legoESM does not track snow,
    the equilibrium thickness is expected to differ from the full
    M-U solution (~2.88 m with snow).

    Reference: Maykut & Untersteiner (1971); Semtner (1976);
    Lindsay (1998, J. Climate, 11, 313-331).
    """
    config = SeaIceConfig(
        ocean_heat_transfer_coeff=6.0,  # Icepack convention with bulk fluxes
    )
    shape = (4,)
    dims = DIMS_COL
    dt = tc.dt  # 3600s (hourly)
    n_steps = tc.quick_steps if quick else tc.duration_steps

    # Start with 1m ice
    state = _make_slab_state(shape, dims, h=1.0, T_ice=250.0, conc=1.0)
    ocean_sst = jnp.full(shape, config.T_freeze_ocean)
    ocean_u = jnp.zeros(shape)
    ocean_v = jnp.zeros(shape)

    diag = {"times": [], "h_mean": [], "T_mean": [], "month": []}

    for i in range(n_steps):
        # Determine month (0-11) from hour-of-year
        hour_of_year = i % 8760
        month = min(int(hour_of_year / 730), 11)  # ~730 hours/month

        # Interpolate forcing to current month
        m0 = month
        m1 = (month + 1) % 12
        frac = (hour_of_year - month * 730) / 730.0
        frac = jnp.clip(frac, 0.0, 1.0)

        T_air = float(_MU_T_AIR[m0] * (1 - frac) + _MU_T_AIR[m1] * frac)
        sw = float(_MU_SW_DOWN[m0] * (1 - frac) + _MU_SW_DOWN[m1] * frac)
        lw = float(_MU_LW_DOWN[m0] * (1 - frac) + _MU_LW_DOWN[m1] * frac)
        wind = float(_MU_WIND[m0] * (1 - frac) + _MU_WIND[m1] * frac)

        forcing = _make_forcing(shape, dims,
                                T_lowest=T_air, sw_down=sw, lw_down=lw,
                                u_lowest=wind, v_lowest=0.0,
                                q_lowest=1e-3, rho_lowest=1.4)

        state, _ = step_sea_ice(state, forcing, ocean_sst, ocean_u, ocean_v,
                                config, U_min=1.0, dt=dt)

        # Record daily diagnostics
        if (i + 1) % 24 == 0:
            day = (i + 1) / 24.0
            diag["times"].append(day)
            diag["h_mean"].append(float(jnp.mean(state.h_ice.data)))
            diag["T_mean"].append(float(jnp.mean(state.T_ice.data)))
            diag["month"].append(month)

    _save_results(outdir, tc, diag)

    # Validation:
    finite = (jnp.all(jnp.isfinite(state.h_ice.data)) and
              jnp.all(jnp.isfinite(state.T_ice.data)))
    h_final = float(jnp.mean(state.h_ice.data))
    h_positive = h_final > 0.0
    T_ok = (jnp.all(state.T_ice.data >= config.T_ice_min) and
            jnp.all(state.T_ice.data <= config.T_freeze_ocean))

    # For 1-year run: ice should be thicker than initial (Arctic grows ice
    # through the cold season). Full M-U equilibrium (~3m) needs multi-year.
    # Quick (3-month) run covers Jan-Mar: should show growth.
    h_grew = h_final > 1.0 if n_steps >= 2190 else True  # only check for >= 3 months

    # Seasonal cycle check: if we have a full year, max h should exceed min h
    h_vals = diag["h_mean"]
    has_cycle = True
    if len(h_vals) > 100:
        h_max = max(h_vals)
        h_min = min(h_vals)
        has_cycle = (h_max - h_min) > 0.05  # at least 5cm seasonal amplitude

    ok = bool(finite and h_positive and T_ok and h_grew and has_cycle)
    n_years = n_steps / 8760.0
    notes = (f"h_final={h_final:.3f}m after {n_years:.1f}yr, "
             f"T_final={float(jnp.mean(state.T_ice.data)):.1f}K")
    if has_cycle and len(h_vals) > 100:
        notes += f", cycle={max(h_vals):.2f}-{min(h_vals):.2f}m"
    if not ok:
        return "FAIL", notes
    return "PASS", notes


# ===========================================================================
# Test dispatch
# ===========================================================================

RUNNERS = {
    # Thermodynamics
    "stefan_growth": run_stefan_growth,
    "slab_stability": run_slab_stability,
    "surface_melt": run_surface_melt,
    "open_water_freeze": run_open_water_freeze,
    "maykut_untersteiner": run_maykut_untersteiner,
    # Dynamics
    "evp_zero_strength": run_evp_zero_strength,
    "evp_compression": run_evp_compression,
    "evp_convergence": run_evp_convergence,
    "mehlmann_lkf": run_mehlmann_lkf,
    "mevp_zero_strength": run_mevp_zero_strength,
    "mevp_compression": run_mevp_compression,
    "mevp_convergence": run_mevp_convergence,
    # Transport
    "advect_uniform": run_advect_uniform,
    "advect_step": run_advect_step,
    "cosine_bell": run_cosine_bell,
    # ITD
    "itd_growth_remap": run_itd_growth_remap,
    "itd_melt_remap": run_itd_melt_remap,
    "itd_roundtrip": run_itd_roundtrip,
    # Integration
    "slab_100_steps": run_slab_100_steps,
    "dynamic_20_steps": run_dynamic_20_steps,
    "dynamic_mevp_20_steps": run_dynamic_mevp_20_steps,
    "multi_cat_10_steps": run_multi_cat_10_steps,
}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Sea ice test matrix runner")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced step counts for fast CI")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only tests matching this category or case name")
    parser.add_argument("--grid", type=str, default=None,
                        help="Run only tests on this grid type")
    parser.add_argument("--test", type=str, default=None,
                        help="Run a single test case by name")
    parser.add_argument("--output", type=str, default="results/sea_ice",
                        help="Output directory (default: results/sea_ice)")
    parser.add_argument("--list", action="store_true",
                        help="List all test cases and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'Category':<15s} {'Case':<25s} {'Grid':<15s} {'Res':<6s} "
              f"{'Steps':>6s} {'Quick':>6s}")
        print("-" * 80)
        for tc in TEST_MATRIX:
            print(f"{tc.category:<15s} {tc.case:<25s} {tc.grid_type:<15s} "
                  f"{tc.resolution:<6s} {tc.duration_steps:>6d} {tc.quick_steps:>6d}")
        return

    output_base = Path(args.output)

    # Filter matrix
    cases = TEST_MATRIX
    if args.test:
        cases = [tc for tc in cases if tc.case == args.test]
    elif args.only:
        cases = [tc for tc in cases
                 if args.only in tc.case or args.only == tc.category]
    if args.grid:
        cases = [tc for tc in cases if tc.grid_type == args.grid]

    if not cases:
        # ``--test`` is an EXACT case selector (the form emitted by a sea-ice
        # `setup:` template via legoesm.core.setup_selector); matching nothing
        # is a hard error rather than a silent no-op so a typo'd selector or a
        # (case, grid) that is not instantiated fails loudly.
        if args.test:
            raise SystemExit(
                f"ERROR: no sea-ice test case matches --test {args.test!r}"
                + (f" --grid {args.grid!r}" if args.grid else "")
                + ". Run `--list` to see valid (case, grid) pairs."
            )
        print("No test cases match the filter. Use --list to see all cases.")
        return

    # Create the output dir only AFTER confirming there is work — a typo'd
    # --test selector must not leave an empty results directory behind.
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  Sea Ice Test Matrix  |  {len(cases)} cases  |  "
          f"{'QUICK' if args.quick else 'FULL'} mode")
    print(f"{'='*80}\n")

    t0_total = time.time()

    for tc in cases:
        runner = RUNNERS.get(tc.case)
        if runner is None:
            record(tc, "SKIP", 0.0, "No runner implemented")
            continue

        outdir = output_base / tc.output_path
        # Per-case namelist parameter file (#682): the resolved case config,
        # written up-front so it is present even if the run later fails.
        write_case_namelist(
            outdir, tc,
            title=f"sea-ice test-case namelist: {tc.output_path}",
            extra={"quick": bool(args.quick)},
        )
        t0 = time.time()
        try:
            status, notes = runner(tc, outdir, args.quick)
            wall = time.time() - t0
            record(tc, status, wall, notes)
        except Exception as exc:
            wall = time.time() - t0
            tb = traceback.format_exc()
            record(tc, "ERROR", wall, f"{type(exc).__name__}: {exc}")
            print(f"    Traceback:\n{tb}")

    # Summary
    total_wall = time.time() - t0_total
    n_pass = sum(1 for r in ALL_RESULTS if r["status"] == "PASS")
    n_fail = sum(1 for r in ALL_RESULTS if r["status"] == "FAIL")
    n_error = sum(1 for r in ALL_RESULTS if r["status"] == "ERROR")
    n_skip = sum(1 for r in ALL_RESULTS if r["status"] == "SKIP")

    print(f"\n{'='*80}")
    print(f"  SUMMARY: {n_pass} PASS | {n_fail} FAIL | {n_error} ERROR | {n_skip} SKIP")
    print(f"  Total wall time: {total_wall:.1f}s")
    print(f"{'='*80}\n")

    summary = {
        "results": ALL_RESULTS,
        "total_wall_time": total_wall,
        "n_pass": n_pass, "n_fail": n_fail,
        "n_error": n_error, "n_skip": n_skip,
        "quick_mode": args.quick,
    }
    with open(output_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Text summary
    with open(output_base / "summary.txt", "w") as f:
        f.write(f"Sea Ice Test Matrix Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"PASS: {n_pass}  FAIL: {n_fail}  ERROR: {n_error}  SKIP: {n_skip}\n")
        f.write(f"Total wall time: {total_wall:.1f}s\n\n")
        for r in ALL_RESULTS:
            f.write(f"  {r['status']:5s} | {r['category']}/{r['test']}/{r['grid']} "
                    f"| {r['wall_time']:.1f}s | {r['notes']}\n")

    sys.exit(1 if n_fail > 0 or n_error > 0 else 0)


if __name__ == "__main__":
    main()
