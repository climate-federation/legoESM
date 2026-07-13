#!/usr/bin/env python
"""Cubed-sphere FV3 atmosphere validation suite.

Exercises the canonical C-D grid / FV3 path only.
Produces machine-readable summary + diagnostic plots.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/validate_cubed_sphere_fv3_atmos.py \
        --output results/fv3_cube_validation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm import constants

# ---------------------------------------------------------------------------
# Edge-jump metric
# ---------------------------------------------------------------------------
from legoesm.grids.halo import CONNECTIVITY


def cube_edge_jump(field, grid):
    """Compute mean and max discontinuity across cube-face edges.

    Parameters
    ----------
    field : jax.Array, shape (6, n, n[, ...])
    grid : CubedSphereGrid

    Returns
    -------
    dict with 'mean_jump', 'max_jump', 'rms_jump'.
    """
    n = grid.n
    # CONNECTIVITY uses integer keys: 0=W, 1=E, 2=S, 3=N
    def _get_strip(f, direction):
        if direction == 0:    # West
            return f[0, :]
        elif direction == 1:  # East
            return f[-1, :]
        elif direction == 2:  # South
            return f[:, 0]
        else:                 # North
            return f[:, -1]

    jumps = []
    for face, edges in CONNECTIVITY.items():
        for direction, (nbr_face, nbr_edge, is_reversed) in edges.items():
            strip = _get_strip(field[face], direction)
            nbr_strip = _get_strip(field[nbr_face], nbr_edge)

            if is_reversed:
                nbr_strip = nbr_strip[..., ::-1] if strip.ndim == 1 else jnp.flip(nbr_strip, axis=0)

            diff = jnp.abs(strip - nbr_strip)
            jumps.append(diff)

    all_jumps = jnp.concatenate([j.ravel() for j in jumps])
    return {
        "mean_jump": float(jnp.mean(all_jumps)),
        "max_jump": float(jnp.max(all_jumps)),
        "rms_jump": float(jnp.sqrt(jnp.mean(all_jumps ** 2))),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sw_to_cdgrid(state, cdgrid):
    """Convert A-grid ShallowWaterState to CDGridShallowWaterState."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import CDGridShallowWaterState
    from legoesm.grids.halo import pad_halo_vector
    h, u, v, h_s = state.h.data, state.u.data, state.v.data, state.h_s.data
    u_pad, v_pad = pad_halo_vector(
        u, v, cdgrid.base.cos_angle, cdgrid.base.sin_angle,
        cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
        interp_offsets=cdgrid.base.halo_interp_offsets,
    )
    u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                   u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
    v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                   v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
    return CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


# ---------------------------------------------------------------------------
# Test A: Shallow-water FV3
# ---------------------------------------------------------------------------

def run_sw_validation(output_dir):
    """Williamson TC2 + TC5 on native CDGrid shallow water."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterModel, CDGridShallowWaterConfig,
    )
    from tests.test_cases.williamson import williamson_test2, williamson_test5

    results = []

    for case, init_fn, days, dt_val in [
        ("williamson2", williamson_test2, 5, 600.0),
        ("williamson5", williamson_test5, 5, 600.0),
    ]:
        grid = create_cubed_sphere(16)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        config = CDGridShallowWaterConfig(hyperdiff_coeff=1e15, use_conservation_fixer=True)
        model = CDGridShallowWaterModel(grid, config)
        sw = init_fn(grid)
        state = _sw_to_cdgrid(sw, cdgrid)

        area = grid.area
        mass_init = float(jnp.sum(state.h.astype(jnp.float64) * area.astype(jnp.float64)))
        nsteps = int(days * 86400 / dt_val)

        t0 = time.time()
        mass_series = [mass_init]
        for step in range(nsteps):
            state = model.step(state, dt_val)
            if (step + 1) % max(1, nsteps // 10) == 0:
                mass_series.append(float(jnp.sum(state.h.astype(jnp.float64) * area.astype(jnp.float64))))
        wall = time.time() - t0

        mass_final = mass_series[-1]
        # iter-159: centralized drift helper (NaN-aware, 1.0
        # baseline floor).  Same migration as iter-157.
        from legoesm.diagnostics import compute_relative_drift
        mass_drift = compute_relative_drift([mass_init, mass_final])
        ok = bool(jnp.all(jnp.isfinite(state.h)))
        edge = cube_edge_jump(state.h, grid)

        # Error norms for TC2
        norms = {}
        if case == "williamson2":
            h_err = state.h - sw.h.data
            l2 = float(jnp.sqrt(jnp.sum(h_err**2 * area) / jnp.sum(sw.h.data**2 * area)))
            linf = float(jnp.max(jnp.abs(h_err)) / jnp.max(jnp.abs(sw.h.data)))
            norms = {"l2": l2, "linf": linf}

        r = {
            "case": case, "grid": f"C16", "dt": dt_val, "days": days,
            "model_path": "shallow_water_fv3_cdgrid.CDGridShallowWaterModel",
            "finite": ok, "mass_drift": mass_drift,
            "edge_mean_jump": edge["mean_jump"], "edge_max_jump": edge["max_jump"],
            "wall_time": wall, **norms,
            "status": "PASS" if ok and mass_drift < 1e-4 else "FAIL",
        }
        results.append(r)
        print(f"  SW {case}: {'PASS' if r['status']=='PASS' else 'FAIL'} "
              f"mass_drift={mass_drift:.2e} edge_max={edge['max_jump']:.2e} "
              f"{f'L2={l2:.2e}' if norms else ''} ({wall:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Test B: Hydrostatic FV3
# ---------------------------------------------------------------------------

def run_hydro_validation(output_dir):
    """Rest-state + Held-Suarez + AMIP dt sweep."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_forcing, held_suarez_init
    from legoesm.core.operators import global_integral
    from legoesm.core.operators_fv_cubed import default_div_damp_coeffs

    results = []

    # B1: Rest-state invariance C8/L10, 50 steps
    print("  Hydro rest-state C8/L10...")
    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(10)
    nu2, _ = default_div_damp_coeffs(grid, dt=300.0)
    config = CDGridPrimitiveEquationConfig(
        div_damp_coeff=nu2, hyperdiff_coeff=1e14,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)

    mass_init = float(global_integral(state.p_s, grid))
    t0 = time.time()
    s = state
    for _ in range(50):
        s = model.step(s, 300.0)
    wall = time.time() - t0
    ok = bool(jnp.all(jnp.isfinite(s.T.data)))
    mass_final = float(global_integral(s.p_s, grid))
    # iter-159: centralized drift helper.
    from legoesm.diagnostics import compute_relative_drift
    mass_drift = compute_relative_drift([mass_init, mass_final])
    ps_drift = float(jnp.max(jnp.abs(s.p_s.data - state.p_s.data))) / constants.p_ref
    edge_T = cube_edge_jump(s.T.data[..., -1], grid)

    r = {
        "case": "rest_state_50steps", "grid": "C8", "nlev": 10, "dt": 300.0,
        "model_path": "primitive_eq_cdgrid.CDGridPrimitiveEquationModel",
        "finite": ok, "mass_drift": mass_drift, "ps_drift_pct": ps_drift * 100,
        "edge_T_max_jump": edge_T["max_jump"],
        "wall_time": wall,
        "status": "PASS" if ok and ps_drift < 0.01 else "FAIL",
    }
    results.append(r)
    print(f"    rest_state: {r['status']} ps_drift={ps_drift:.2e} "
          f"mass_drift={mass_drift:.2e} edge_T_max={edge_T['max_jump']:.2e}")

    # B2: AMIP dt sweep C8/L10
    print("  Hydro AMIP dt sweep C8/L10...")
    for dt_val in [150.0, 300.0, 450.0, 600.0]:
        grid = create_cubed_sphere(8)
        sigma = create_sigma_coordinate(10)
        nu2, _ = default_div_damp_coeffs(grid, dt=dt_val)
        config = CDGridPrimitiveEquationConfig(
            div_damp_coeff=nu2, hyperdiff_coeff=1e14, A_h=1e5,
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, config)
        state = held_suarez_init(grid, sigma)

        nsteps = int(1 * 86400 / dt_val)  # 1 day
        t0 = time.time()
        s = state
        blew_up = False
        for step in range(nsteps):
            s = model.step(s, dt_val, physics_fn=held_suarez_forcing)
            if not bool(jnp.all(jnp.isfinite(s.T.data))):
                blew_up = True
                break
        wall = time.time() - t0
        ok = not blew_up

        r = {
            "case": f"amip_dt{int(dt_val)}", "grid": "C8", "nlev": 10, "dt": dt_val,
            "model_path": "primitive_eq_cdgrid.CDGridPrimitiveEquationModel",
            "finite": ok, "steps_completed": step + 1, "steps_total": nsteps,
            "wall_time": wall,
            "status": "PASS" if ok else "FAIL",
        }
        if ok:
            mass_final = float(global_integral(s.p_s, grid))
            mass_init_val = float(global_integral(state.p_s, grid))
            # iter-159: centralized drift helper.
            from legoesm.diagnostics import compute_relative_drift
            r["mass_drift"] = compute_relative_drift(
                [mass_init_val, mass_final])
            r["T_range"] = [float(jnp.min(s.T.data)), float(jnp.max(s.T.data))]
            edge_T = cube_edge_jump(s.T.data[..., -1], grid)
            r["edge_T_max_jump"] = edge_T["max_jump"]

        results.append(r)
        status = "PASS" if ok else f"FAIL (step {step+1}/{nsteps})"
        print(f"    dt={dt_val:5.0f}s: {status} ({wall:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Test C: Non-hydrostatic FV3
# ---------------------------------------------------------------------------

def run_nh_validation(output_dir):
    """Short DCMIP TC1 on CDGrid compressible Euler."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
    )
    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1 import dcmip25_tc1_init

    results = []
    print("  NH DCMIP TC1 C8...")
    grid = create_cubed_sphere(8)
    state, hcoord, tmetric = dcmip25_tc1_init(grid, n_levels=10)

    config = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=6, sponge_width=10000.0, sponge_coeff=0.05,
    )
    model = CDGridCompressibleEulerModel(grid, hcoord, tmetric, config)

    dt = 5.0
    nsteps = 60  # 5 minutes
    t0 = time.time()
    s = state
    for step in range(nsteps):
        s = model.step(s, dt)
    wall = time.time() - t0

    ok = bool(jnp.all(jnp.isfinite(s.u.data)) and jnp.all(jnp.isfinite(s.w.data)))
    w_max = float(jnp.max(jnp.abs(s.w.data)))

    r = {
        "case": "dcmip_tc1_short", "grid": "C8", "nlev": 10, "dt": dt,
        "model_path": "compressible_euler_cdgrid.CDGridCompressibleEulerModel",
        "finite": ok, "w_max": w_max, "wall_time": wall,
        "status": "PASS" if ok else "FAIL",
    }
    results.append(r)
    print(f"    TC1: {r['status']} |w|_max={w_max:.4f} m/s ({wall:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# Test D: No legacy wrapper usage
# ---------------------------------------------------------------------------

def run_legacy_check():
    """Verify deleted wrapper modules cannot be imported."""
    results = []
    for mod in [
        "legoesm.atmosphere.dynamics.shallow_water_fv",
        "legoesm.atmosphere.dynamics.shallow_water",
        "legoesm.atmosphere.dynamics.primitive_eq",
        "legoesm.atmosphere.dynamics.primitive_eq_fv",
    ]:
        try:
            __import__(mod)
            results.append({"module": mod, "status": "FAIL", "note": "Module still importable"})
            print(f"    FAIL: {mod} is still importable")
        except (ImportError, ModuleNotFoundError):
            results.append({"module": mod, "status": "PASS", "note": "Correctly removed"})
            print(f"    PASS: {mod} correctly removed")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FV3 cubed-sphere atmosphere validation")
    parser.add_argument("--output", "-o", default="results/fv3_cube_validation")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FV3 Cubed-Sphere Atmosphere Validation Suite")
    print("=" * 70)
    print(f"  Backend: {jax.default_backend()}")
    print(f"  X64: {jax.config.jax_enable_x64}")
    print(f"  Output: {out}")
    print()

    all_results = []

    print("A. Shallow-Water FV3 Native Tests")
    print("-" * 40)
    all_results.extend(run_sw_validation(out))
    print()

    print("B. Hydrostatic FV3 Cubed-Sphere Tests")
    print("-" * 40)
    all_results.extend(run_hydro_validation(out))
    print()

    print("C. Non-Hydrostatic FV3 Cubed-Sphere Tests")
    print("-" * 40)
    all_results.extend(run_nh_validation(out))
    print()

    print("D. Legacy Wrapper Removal Check")
    print("-" * 40)
    all_results.extend(run_legacy_check())
    print()

    # Summary
    n_pass = sum(1 for r in all_results if r.get("status") == "PASS")
    n_fail = sum(1 for r in all_results if r.get("status") == "FAIL")
    print("=" * 70)
    print(f"  SUMMARY: {n_pass} PASS, {n_fail} FAIL out of {len(all_results)} tests")
    print("=" * 70)

    # Write JSON summary
    summary = {
        "results": all_results,
        "n_pass": n_pass, "n_fail": n_fail,
        "canonical_implementations": {
            "shallow_water": "shallow_water_fv3_cdgrid.CDGridShallowWaterModel",
            "hydrostatic_pe": "primitive_eq_cdgrid.CDGridPrimitiveEquationModel",
            "nonhydrostatic_ce": "compressible_euler_cdgrid.CDGridCompressibleEulerModel",
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary: {out / 'summary.json'}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
