"""Phase G tier-3 (measure-first): Veros-ACC FREE-RUN comparison.

Runs the legoESM Veros-ACC recipe FORWARD (the recipe now carries Veros's wind
stress + T* restoring; see ``veros_acc_recipe.build_acc_recipe(with_surface_
forcing=True)``) and compares bulk statistics against a Veros ACC free run of the
same physical duration. Both sides are reduced with the SAME legoESM diagnostic
code (the Veros snapshot is bridged onto the legoESM grid first), so the
comparison is apples-to-apples.

This is a MEASURE-FIRST harness: it runs with the CURRENT recipe (SSP-RK3 time
stepping, EKE off / constant GM, single dt) and reports deltas — it does NOT
gate pass/fail. Known model-formulation differences are printed before the
table so the deltas are read in context (see the strategy doc §8 ledger).

Run in float64 (``JAX_ENABLE_X64=1`` + an fp64 precision policy) to match Veros.

Usage::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/ocean_fidelity/run_acc_freerun.py --years 1.0
    # quick smoke (10 days, no Veros):
    ... run_acc_freerun.py --smoke --no-veros
"""

from __future__ import annotations

import argparse
from functools import partial

import numpy as np

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.0
# ACC channel re-entrant band for the Drake-passage transport (the default
# diagnostic band -65..-45 is OUTSIDE this domain; see plan R6).
_ACC_DRAKE_LAT_S = -40.0
_ACC_DRAKE_LAT_N = -22.0


def _layer_thickness_np(state, z_coord):
    from legoesm.ocean.vertical import compute_layer_thickness
    return np.asarray(compute_layer_thickness(
        state.eta.data, state.H_bathy.data, z_coord))


def _bulk_stats(state, z_coord, grid):
    """Compute apples-to-apples bulk diagnostics from a (legoESM-shaped) state."""
    from legoesm.ocean.diagnostics_streamfunction import barotropic_streamfunction
    from legoesm.ocean.budgets import compute_energy_budget

    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    T = np.asarray(state.T.data)
    mask = np.asarray(state.land_mask.data)
    h = _layer_thickness_np(state, z_coord)               # (n_lat, n_lon, nlev)
    lat = np.degrees(np.asarray(grid.lat))

    # ACC (Drake) transport from the barotropic streamfunction [Sv].
    psi = np.asarray(barotropic_streamfunction(u, h, mask, grid))   # (n_lat, n_lon) [Sv]
    band = (lat >= _ACC_DRAKE_LAT_S) & (lat <= _ACC_DRAKE_LAT_N)
    acc_T = float(psi[band, :].max() - psi[band, :].min()) if band.any() else float("nan")

    # Total kinetic energy [J] via the shared energy budget.
    eb = compute_energy_budget(state, z_coord, grid_type="latlon", grid=grid)
    KE = float(eb.KE)

    # Wet-cell temperature statistics [degC]. Volume-weighted mean.
    wet3d = (h > 0.0) & (mask[:, :, None] > 0.5)
    area = np.asarray(getattr(grid, "area", np.ones(mask.shape)))
    vol = (area[:, :, None] * h) * wet3d
    Tw = T[wet3d]
    vol_mean_T = float(np.sum(T * vol) / np.sum(vol)) if np.sum(vol) > 0 else float("nan")

    # Surface-layer max |u| [m/s] (jet strength proxy).
    umax = float(np.nanmax(np.abs(u)))

    return {
        "ACC_transport_Sv": acc_T,
        "total_KE_J": KE,
        "vol_mean_T_C": vol_mean_T,
        "T_min_C": float(Tw.min()) if Tw.size else float("nan"),
        "T_max_C": float(Tw.max()) if Tw.size else float("nan"),
        "max_abs_u_ms": umax,
    }


def _run_legoesm(years, dt, *, snapshot_every_days=None):
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    recipe = build_acc_recipe(with_surface_forcing=True)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)
    sf = recipe.wind_forcing
    state = recipe.initial_state

    total_steps = int(round(years * _DAYS_PER_YEAR * _SECONDS_PER_DAY / dt))
    # Integrate in 1-day blocks for granular NaN-checking; jit the inner scan.
    steps_per_block = max(1, int(round(_SECONDS_PER_DAY / dt)))

    def _body(st, _):
        return model.step(st, dt, surface_forcing=sf), None

    @partial(jax.jit, static_argnames=("n",))
    def _block(st, n):
        st, _ = jax.lax.scan(_body, st, None, length=n)
        return st

    done = 0
    while done < total_steps:
        n = min(steps_per_block, total_steps - done)
        state = _block(state, n)
        jax.block_until_ready(state.u.data)
        done += n
        if not bool(jnp.all(jnp.isfinite(state.u.data))):
            raise RuntimeError(f"legoESM blew up after {done} steps "
                               f"({done * dt / _SECONDS_PER_DAY:.1f} days)")
        day = done * dt / _SECONDS_PER_DAY
        if done % (steps_per_block * 30) == 0 or done == total_steps:
            umax = float(jnp.max(jnp.abs(state.u.data)))
            print(f"  legoESM day {day:7.1f}: max|u|={umax:.4f} m/s")
    return state, recipe


def _run_veros_bridged(years, template_state):
    """Run Veros ACC for the same physical time and bridge to a legoESM state."""
    from legoesm.ocean.fidelity.veros_runner import run_veros
    from legoesm.ocean.fidelity.veros_state_bridge import (
        veros_snapshot_to_legoesm_state,
    )
    runlen_s = years * _DAYS_PER_YEAR * _SECONDS_PER_DAY
    result = run_veros("acc_channel", runlen_s=runlen_s)
    bridged = veros_snapshot_to_legoesm_state(result, template_state)
    return bridged.state, result


_KNOWN_DIFFERENCES = """\
Known model-formulation differences (the deltas below should be read in this
context; see docs/ocean_fidelity/oracle_recipe_strategy.md §8):
  1. Time integrator: legoESM SSP-RK3 vs Veros leapfrog+AB2+Robert-Asselin.
  2. Timestep: legoESM single dt=4800 s; Veros dt_mom=4800 / dt_tracer=43200 s.
  3. EKE OFF (constant GM kappa=1000) vs Veros prognostic Eden-Greatbatch EKE
     -> largest expected divergence in stratification / ACC transport.
  4. GM/Redi isoneutral discretization differs (tier-2 T_iso corr ~0.17).
These are NOT bugs; they are the documented gaps a measure-first run quantifies.\
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=float, default=1.0,
                    help="Physical integration time [years] (default 1).")
    ap.add_argument("--dt", type=float, default=4800.0,
                    help="legoESM timestep [s] (default 4800 = Veros dt_mom; "
                         "43200 violates the barotropic CFL).")
    ap.add_argument("--smoke", action="store_true",
                    help="10-day stability smoke run (overrides --years).")
    ap.add_argument("--no-veros", action="store_true",
                    help="Skip the Veros run (legoESM-only diagnostics).")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy

    years = (10.0 / _DAYS_PER_YEAR) if args.smoke else args.years

    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        print(f"== legoESM ACC free run: {years*_DAYS_PER_YEAR:.0f} days, "
              f"dt={args.dt:.0f} s (fp64) ==")
        lego_state, recipe = _run_legoesm(years, args.dt)
        lego = _bulk_stats(lego_state, recipe.z_coord, recipe.grid)

        veros = None
        if not args.no_veros:
            print(f"== Veros ACC reference: {years*_DAYS_PER_YEAR:.0f} days ==")
            try:
                veros_state, _ = _run_veros_bridged(years, recipe.initial_state)
                veros = _bulk_stats(veros_state, recipe.z_coord, recipe.grid)
            except Exception as exc:  # noqa: BLE001 — report, don't crash
                print(f"  Veros run/bridge failed: {type(exc).__name__}: {exc}")
    finally:
        set_policy(_prev)

    print("\n" + _KNOWN_DIFFERENCES + "\n")
    print(f"{'metric':22s} {'legoESM':>14s} {'Veros':>14s} {'abs diff':>12s} {'% diff':>9s}")
    print("-" * 74)
    for k in lego:
        lv = lego[k]
        if veros is not None and k in veros and np.isfinite(veros[k]):
            vv = veros[k]
            d = lv - vv
            pct = 100.0 * d / vv if vv != 0 else float("nan")
            print(f"{k:22s} {lv:14.4g} {vv:14.4g} {d:12.4g} {pct:8.1f}%")
        else:
            print(f"{k:22s} {lv:14.4g} {'(n/a)':>14s} {'':>12s} {'':>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
