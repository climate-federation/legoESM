#!/usr/bin/env python
"""Mixed-precision validation campaign for the OMIP ocean (A2 follow-up).

The cheap unit gate (tests/validation/test_precision_omip.py) proves the
``--precision mixed`` policy is WIRED. This is the next tier: an actual
fp64-vs-mixed integration that quantifies the numerical cost of mixed
precision on a closed (forcing-free) ocean channel, where volume / heat /
salt are conserved quantities, so any end-vs-start drift is pure
conservation/precision error.

It is NOT the full century-scale production campaign (global SST/SSS drift,
AMOC, bias maps over O(100 yr) — that is a long OMIP-faithful sbatch). It is
the reusable, deterministic harness that gates mixed precision before such a
run: build the SAME closed channel under fp64 and under the mixed policy,
step K times, and compare:

  0. ENGAGEMENT (mixed actually ran fp32 storage, fp64 ran fp64, and the two
     diverged) — so the agreement checks can never pass vacuously,
  1. finiteness (both modes stay finite — no blow-up),
  2. self-conservation per mode (closed basin: |heat_end - heat_start| small).
     HEAT and SALT are the meaningful raw-conservation signal here (the tracer
     conservation fixer defaults OFF). VOLUME is partly enforced by the default
     ``fix_eta_drift`` free-surface correction, so its check validates the
     production CORRECTED path, not raw free-surface conservation,
  3. mixed-vs-fp64 agreement (tracer budgets + RMS(T) field diff bounded).

Returns a structured verdict and (CLI) writes a JSON report; exits non-zero if
any gate fails, so it can run as an sbatch validation job.

Scope/honesty: this exercises the mixed-precision COMPUTE path + the fp64
ocean-kernel overrides (barotropic/EOS/PGF/coriolis). Whether the prognostic
state is actually stored fp32 depends on the rest-state builder; the report
surfaces the integrated state dtype per mode (``*_state_dtype``) so the result
never overclaims. Full fp32-storage century drift remains the production run.

Run on a COMPUTE node (JAX compile) — never the login node.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def _build_channel(n_lat: int, n_lon: int, n_levels: int = 4):
    """Closed forcing-free channel with a thermal front + sheared jet.

    Same construction the ocean unit tests use (tests/ocean/unit/
    test_ab2_scope.py::_channel): a meridional thermal front makes lateral
    diffusion nonzero and a depth-sheared zonal jet makes friction + bottom
    drag nonzero, so the step exercises the precision-sensitive kernels. No
    surface forcing -> volume/heat/salt are conserved.
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid = create_latlon_grid(n_lat, n_lon)
    z_coord = create_ocean_z_star(n_levels=n_levels, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=4000.0, land_lat_threshold=80.0)
    lat = np.degrees(np.asarray(grid.lat))
    T = np.asarray(state.T.data) + 4.0 * np.tanh(lat / 15.0)[:, None, None]
    state = state._replace(T=state.T.replace(data=jnp.asarray(T)))
    nlev = z_coord.n_levels
    shear = np.linspace(1.0, 0.1, nlev)[None, None, :]
    u = 0.2 * np.cos(np.radians(lat))[:, None, None] * shear
    u = np.broadcast_to(u, state.u.data.shape).copy()
    u *= np.asarray(state.u_mask.data)[..., None]
    u[:, -1] = u[:, 0]
    state = state._replace(u=state.u.replace(data=jnp.asarray(u)))
    cfg = LatLonCGridOceanConfig.from_flat(
        n_barotropic_substeps=8, enable_runtime_checks=False,
        implicit_vertical_mixing=True, A_h=2.0e4, A_v=1.0e-3, K_v=1.0e-4,
        K_h=500.0, bottom_drag_r=1.0e-3)
    return grid, z_coord, state, LatLonCGridOceanModel(grid, z_coord, cfg)


def _run_mode(mode: str, n_lat: int, n_lon: int, n_steps: int, dt: float):
    """Build + step the channel under ``mode`` precision; return budgets+state."""
    from legoesm.runtime.precision import apply_precision
    from legoesm.ocean.budgets import compute_tracer_budget

    apply_precision(mode)
    grid, z_coord, state, model = _build_channel(n_lat, n_lon)
    b0 = compute_tracer_budget(state, z_coord, grid_type="latlon", grid=grid)
    s = state
    for _ in range(n_steps):
        s = model.step(s, dt)
    b1 = compute_tracer_budget(s, z_coord, grid_type="latlon", grid=grid)
    return {
        "vol0": float(b0.volume), "vol1": float(b1.volume),
        "heat0": float(b0.heat_content), "heat1": float(b1.heat_content),
        "salt0": float(b0.salt_mass), "salt1": float(b1.salt_mass),
        "T": np.asarray(s.T.data, dtype=np.float64),
        "state_dtype": str(s.T.data.dtype),  # transparency: did storage go fp32?
        "finite": bool(np.all(np.isfinite(np.asarray(s.T.data)))
                       and np.all(np.isfinite(np.asarray(s.u.data)))),
    }


def run_precision_validation(
    n_lat: int = 16, n_lon: int = 32, n_steps: int = 40, dt: float = 1800.0,
    *, self_tol: float = 1e-6, cross_tol: float = 5e-3,
) -> dict:
    """fp64-vs-mixed validation on the closed channel. Returns a verdict dict.

    self_tol  : max allowed |budget_end-budget_start|/|budget_start| per mode
                (closed-basin conservation, near machine-eps with the fixers).
    cross_tol : max allowed |budget_mixed-budget_fp64|/|budget_fp64| and a
                bounded RMS(T) field divergence between the two modes.
    """
    ref = _run_mode("fp64", n_lat, n_lon, n_steps, dt)
    mix = _run_mode("mixed", n_lat, n_lon, n_steps, dt)
    # restore fp64 baseline (apply_precision is global)
    from legoesm.runtime.precision import apply_precision
    apply_precision("fp64")

    def _rel(a, b):
        return abs(a - b) / max(abs(b), 1e-30)

    checks = {}
    checks["fp64_finite"] = ref["finite"]
    checks["mixed_finite"] = mix["finite"]
    # Engagement gates (codex 2026-06-22): without these the agreement checks
    # below could PASS VACUOUSLY if the mixed policy silently did not engage
    # (mixed == fp64). Assert that mixed actually ran fp32 STORAGE, fp64 ran
    # fp64, and that the two modes genuinely diverged (non-zero field diff).
    checks["mixed_storage_is_fp32"] = mix["state_dtype"] == "float32"
    checks["fp64_storage_is_fp64"] = ref["state_dtype"] == "float64"
    for q in ("vol", "heat", "salt"):
        checks[f"fp64_{q}_conserved"] = _rel(ref[f"{q}1"], ref[f"{q}0"]) <= self_tol
        checks[f"mixed_{q}_conserved"] = _rel(mix[f"{q}1"], mix[f"{q}0"]) <= self_tol
        checks[f"{q}_cross_agree"] = _rel(mix[f"{q}1"], ref[f"{q}1"]) <= cross_tol
    t_scale = float(np.sqrt(np.mean(ref["T"] ** 2))) or 1.0
    t_rms = float(np.sqrt(np.mean((mix["T"] - ref["T"]) ** 2)))
    checks["T_field_agree"] = (t_rms / t_scale) <= cross_tol
    # ...AND mixed must genuinely DIFFER from fp64 (else "agreement" is vacuous
    # because the policy did nothing). A real fp32-compute run always diverges.
    checks["mixed_differs_from_fp64"] = t_rms > 0.0

    metrics = {
        "n_lat": n_lat, "n_lon": n_lon, "n_steps": n_steps, "dt": dt,
        "fp64_heat_drift_rel": _rel(ref["heat1"], ref["heat0"]),
        "mixed_heat_drift_rel": _rel(mix["heat1"], mix["heat0"]),
        "heat_cross_rel": _rel(mix["heat1"], ref["heat1"]),
        "salt_cross_rel": _rel(mix["salt1"], ref["salt1"]),
        "T_rms_diff": t_rms, "T_rms_diff_rel": t_rms / t_scale,
        # Transparency: the prognostic-state dtype actually integrated under
        # each policy. If both are float64 the run validated the mixed COMPUTE
        # path + the fp64 ocean-kernel overrides (not full fp32 storage); a
        # fp32 mixed_state_dtype means storage was exercised too. Either way
        # the agreement metrics above are honest for what was run.
        "fp64_state_dtype": ref["state_dtype"],
        "mixed_state_dtype": mix["state_dtype"],
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-lat", type=int, default=16)
    p.add_argument("--n-lon", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--dt", type=float, default=1800.0)
    p.add_argument("--self-tol", type=float, default=1e-6)
    p.add_argument("--cross-tol", type=float, default=5e-3)
    p.add_argument("--out", type=str, default=None,
                   help="Optional JSON report path.")
    args = p.parse_args()
    result = run_precision_validation(
        args.n_lat, args.n_lon, args.n_steps, args.dt,
        self_tol=args.self_tol, cross_tol=args.cross_tol)
    print(json.dumps(result, indent=2))
    print("=== precision-validation verdict:",
          "PASS" if result["passed"] else "FAIL", "===")
    for k, v in result["checks"].items():
        if not v:
            print(f"  FAILED: {k}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
