"""Phase G tier-3 (measure-first): Veros-ACC FREE-RUN comparison.

Runs the legoESM Veros-ACC recipe FORWARD (the recipe now carries Veros's wind
stress + T* restoring; see ``veros_acc_recipe.build_acc_recipe(with_surface_
forcing=True)``) and compares bulk statistics against a Veros ACC free run of the
same physical duration. Both sides are reduced with the SAME legoESM diagnostic
code (the Veros snapshot is bridged onto the legoESM grid first), so the
comparison is apples-to-apples.

This is a MEASURE-FIRST harness: it runs with the CURRENT recipe (forward-Euler /
split-explicit time stepping, prognostic EKE + K_iso=K_gm matching Veros, single
dt) and reports deltas — it does NOT gate pass/fail. Known model-formulation
differences are printed before the table so the deltas are read in context (see
the strategy doc §8 ledger).

Run in float64 (``JAX_ENABLE_X64=1`` + an fp64 precision policy) to match Veros.

Usage::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        .venv/bin/python scripts/validate/ocean_fidelity/run_acc_freerun.py --years 1.0
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


def _run_legoesm(years, dt, *, snapshot_every_days=None, outer_integrator=None,
                 bottom_drag_r=None, barotropic_solver=None, dt_mom_ratio=None,
                 momentum_friction_additive=False, coriolis_scheme=None,
                 ab2_scope=None):
    import jax
    import jax.numpy as jnp
    from legoesm.core.field import Field
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    recipe = build_acc_recipe(with_surface_forcing=True)
    cfg = recipe.model_config
    if outer_integrator is not None:
        cfg = cfg._replace(outer_integrator=outer_integrator)
    if bottom_drag_r is not None:
        # Dissipation-audit knob: Veros applies -r_bot·u (a RATE, 1e-5/s) at the
        # bottom cell with NO /dz; legoESM applies -r·u/h_bot. To match Veros's
        # ~28h drag, r ≈ r_bot·h_bot ≈ 1e-5·276 ≈ 2.8e-3 (flat-bottom ACC).
        cfg = cfg._replace(bottom_drag_r=bottom_drag_r)
    if barotropic_solver is not None:
        # Rigid-lid fidelity: match Veros's barotropic FORMULATION (streamfunction
        # rather than the legoESM split-explicit free surface). The dissipation
        # audit isolated the barotropic formulation as the genuine ACC gap.
        cfg = cfg._replace(barotropic_solver=barotropic_solver)
    if dt_mom_ratio is not None:
        # Veros dt_mom≠dt_tracer asynchronous stepping (#44 Stage B): the `dt`
        # passed here IS dt_tracer (the clock); momentum + barotropic + implicit
        # friction use dt_mom = dt / dt_mom_ratio. Veros ACC = DT_TRACER_S/DT_MOM_S
        # = 43200/4800 = 9.0. Requires barotropic_solver="rigid_lid" (fixed depth ⇒
        # exact tracer conservation). Pass `--dt 43200 --dt-mom-ratio 9
        # --barotropic-solver rigid_lid --outer-integrator ab2` for the faithful run.
        cfg = cfg._replace(dt_mom_ratio=dt_mom_ratio)
    if momentum_friction_additive:
        # Veros ADDITIVE momentum vertical-friction placement (friction.py +
        # solve_stream.py): the implicit friction increment is evaluated on the
        # pre-step u^n and added alongside the AB2-extrapolated explicit
        # tendency (weight 1.0), instead of backward-Euler on the AB2 state.
        # Requires --outer-integrator ab2 (validated at config construction).
        cfg = cfg._replace(momentum_friction_additive=True)
    if ab2_scope is not None:
        # Veros-faithful AB2 scope (D2): dissipative tendencies (lateral
        # friction + bottom drag; tracer diffusion + GM/Redi) at weight 1.0
        # like Veros's du_mix / tr[tau]-diffusion placement, with their
        # depth-mean routed through the barotropic forcing (solve_stream.py
        # uloc structure). Requires outer_integrator="ab2". Climate-neutral
        # on the ACC (218 vs 221 Sv) — faithfulness/stability option.
        cfg = cfg._replace(ab2_scope=ab2_scope)
    if coriolis_scheme is not None:
        # Veros explicit-AB2 Coriolis placement (dycore-audit D1): the plain f×u
        # enters du_dt (so the outer AB2 extrapolates it and its depth-mean feeds
        # the barotropic rigid-lid slow forcing = solve_stream.py uloc/vloc), the
        # Matsuno rotation sub-step is skipped, and the rigid-lid solver's own
        # Coriolis addition is gated off (no double count). Requires
        # --outer-integrator ab2 + --barotropic-solver rigid_lid (validated).
        cfg = cfg._replace(coriolis_scheme=coriolis_scheme)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    if coriolis_scheme == "explicit_ab2":
        # Surface the conditional-stability margin for the configured domain.
        model.check_coriolis_stability(dt)
    sf = recipe.wind_forcing
    state = recipe.initial_state
    # AB2 needs the prior-increment carry seeded (to zero) so the scan keeps a
    # constant pytree; the first step is then a 1.6x forward-Euler seed.
    if cfg.outer_integrator == "ab2" and state.T_incr_prev is None:
        _z = lambda d: Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                             dims=d.dims, units=d.units)
        state = state._replace(
            T_incr_prev=_z(state.T), S_incr_prev=_z(state.S),
            u_incr_prev=_z(state.u), v_incr_prev=_z(state.v))
    # Rigid-lid: pre-build the static island/depth data (host-side flood-fill)
    # and seed the streamfunction carry (ψ, dψ, dψ_prev, dpsin, dpsin_prev) to
    # zero so the jitted scan keeps a constant pytree (None -> array would crash).
    if cfg.barotropic_solver == "rigid_lid" and state.psi is None:
        rl = model._ensure_rigid_lid_data(state)
        _zV = jnp.zeros((recipe.grid.n_lat + 1, recipe.grid.n_lon + 1),
                        dtype=state.u.data.dtype)
        _zI = jnp.zeros((rl.nisle,), dtype=state.u.data.dtype)
        state = state._replace(psi=_zV, dpsi=_zV, dpsi_prev=_zV,
                               dpsin=_zI, dpsin_prev=_zI)

    # Reflect the ACTUAL config used by the model (with all the _replace knobs:
    # outer_integrator, dt_mom_ratio, coriolis_scheme, ...) on the returned recipe
    # so callers that read recipe.model_config see what ran, not the bare default.
    recipe = recipe._replace(model_config=cfg)

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
  0. Bottom drag: now FAITHFULLY mapped (dissipation audit, 2026-05-29). Veros applies
     r_bot=1e-5 as a RATE on the bottom cell (no /dz); legoESM uses -r*u/h_bot, so the
     recipe now sets bottom_drag_r=r_bot*h_bot~2.76e-3 (the prior r=1e-5 was ~276x too
     weak). BUT the faithful drag OVER-DAMPS the transport: @1yr it gives -83% (18 vs
     110 Sv) vs the mis-mapped r=1e-5's +58% -> legoESM's barotropic transport
     over-responds to bottom drag vs Veros. The genuine deeper difference is the
     BAROTROPIC FORMULATION (free surface vs Veros rigid-lid/streamfunction); the old
     r=1e-5 masked it via compensating errors. (30d looked like a fix: KE +233%->+27%;
     1yr revealed the over-damping.) CONFIRMED: a rigid-lid option (--barotropic-solver
     rigid_lid) is now built and RESTORES an O(100 Sv) ACC vs the free surface's collapsed
     18 Sv at the SAME faithful drag -> the barotropic formulation was the dominant control.
  1. Time integrator: NOW MATCHED (#44). --outer-integrator ab2 is the faithful Veros
     scheme — AB2 on the EXPLICIT tendency + implicit vertical mixing applied ONCE
     (tracers temp[taup1]=temp[tau]+dt_tracer*((1.5+eps)*dtemp[tau]-(0.5+eps)*dtemp[taum1]),
     AB2 eps-offset; core/thermodynamics.py + core/external/solve_stream.py). Stable +
     compatible with convective adjustment. Measured NOT the climate lever (the residual
     is the eddy-mean equilibration, not the dycore numerics).
  2. Timestep: NOW MATCHED (#44). --dt-mom-ratio 9 gives Veros's dt_mom=4800 /
     dt_tracer=43200 asynchronous ("distorted-physics") stepping (--dt IS dt_tracer;
     dt_mom = dt/ratio). Requires --barotropic-solver rigid_lid (fixed depth ⇒ exact
     tracer conservation). Faithful run: --dt 43200 --dt-mom-ratio 9 --barotropic-solver
     rigid_lid --outer-integrator ab2.
  3. EKE GM coefficient: ON, prognostic Eden-Greatbatch with the Rhines `eke_len`
     (form + eke_len reproduce Veros's K_gm/eke_len to machine precision, gates
     E9 + L4) -> the GM *skew* coefficient MATCHES Veros. EKE cold-starts at e_min.
  4. EKE Redi K_iso=K_gm: now MATCHED (prognostic Redi, commit 4c1ec219) -- the
     step drives kappa_Redi from the prognostic kappa (Veros
     enable_eke_isopycnal_diffusion), oracle machine-exact (gate R3).
  5. GM/Redi isoneutral slopes: NOW use the Veros-faithful NEUTRAL
     (locally-referenced) density gradient ∂ρ/∂T·∇T+∂ρ/∂S·∇S
     (GMRediConfig.slope_density="neutral", isoneutral.py:40-41), removing the
     in-situ compressibility bias that collapsed K_33. Tier-2 T_iso interior
     corr 0.20 (in-situ) -> 0.37 (neutral, Veros K_iso fed); neutral K_33 now
     tracks Veros (ratio ~1.0 at the thermocline). Residual = the per-triad
     drodzb kr-sum + exact metric factors (dxu/dxt/dyu/dyt/cost), documented as
     a follow-up; inert on this uniform channel.
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
    ap.add_argument("--outer-integrator", default=None,
                    choices=("forward_euler", "ab2"),
                    help="Override the legoESM outer time integrator (default: the "
                         "recipe's forward_euler). 'ab2' = Veros Adams-Bashforth-2.")
    ap.add_argument("--bottom-drag-r", type=float, default=None,
                    help="Override the linear bottom-drag coefficient [m/s] "
                         "(dissipation audit; recipe default 1e-5). Veros's effective "
                         "rate is r_bot·h_bot ≈ 2.8e-3 since legoESM divides by h_bot.")
    ap.add_argument("--barotropic-solver", default=None,
                    choices=("explicit_substep", "implicit_cn", "rigid_lid"),
                    help="Override the barotropic solver. 'rigid_lid' matches Veros's "
                         "barotropic FORMULATION (streamfunction) — the dissipation "
                         "audit isolated the free-surface-vs-rigid-lid difference as "
                         "the genuine ACC transport gap.")
    ap.add_argument("--dt-mom-ratio", type=float, default=None,
                    help="Veros dt_mom≠dt_tracer asynchronous stepping (#44): dt_mom = "
                         "dt / ratio (--dt IS dt_tracer). Veros ACC = 9 (dt_tracer=43200, "
                         "dt_mom=4800). Requires --barotropic-solver rigid_lid. Faithful "
                         "run: --dt 43200 --dt-mom-ratio 9 --barotropic-solver rigid_lid "
                         "--outer-integrator ab2.")
    ap.add_argument("--momentum-friction-additive", action="store_true",
                    help="Veros ADDITIVE momentum vertical-friction placement "
                         "(friction.py + solve_stream.py): the implicit friction "
                         "increment is evaluated on the pre-step u^n and added "
                         "alongside the AB2 explicit tendency (weight 1.0), instead "
                         "of backward-Euler on the AB2 state. Requires "
                         "--outer-integrator ab2.")
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy

    years = (10.0 / _DAYS_PER_YEAR) if args.smoke else args.years

    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        _oi = args.outer_integrator or "recipe default (forward_euler)"
        _bs = args.barotropic_solver or "recipe default (free surface)"
        _dr = f"dt_mom={args.dt/args.dt_mom_ratio:.0f}s (ratio {args.dt_mom_ratio:g})" \
            if args.dt_mom_ratio else "dt_mom=dt_tracer (synchronous)"
        print(f"== legoESM ACC free run: {years*_DAYS_PER_YEAR:.0f} days, "
              f"dt_tracer={args.dt:.0f} s (fp64), integrator={_oi}, barotropic={_bs}, "
              f"{_dr} ==")
        lego_state, recipe = _run_legoesm(
            years, args.dt, outer_integrator=args.outer_integrator,
            bottom_drag_r=args.bottom_drag_r,
            barotropic_solver=args.barotropic_solver,
            dt_mom_ratio=args.dt_mom_ratio,
            momentum_friction_additive=args.momentum_friction_additive)
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
