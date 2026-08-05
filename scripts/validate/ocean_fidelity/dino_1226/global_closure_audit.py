"""#1492 item 0.1 — GLOBAL CLOSED-DOMAIN CONSERVATION-CLOSURE AUDIT.

Different question from every prior #1226 instrument. Those measured heat
DELIVERY into an open box (attribution against NEMO). This measures whether
legoESM CONSERVES what it must, over the WHOLE domain, against ITSELF -- no
oracle needed. Per conserved quantity Q in {volume, salt, heat}, per step::

    residual(t) = d(integral of Q over whole domain)/dt
                  - (boundary fluxes the model ACTUALLY APPLIED)

Correct model -> residual ~1e-15 relative. A systematic nonzero residual is
an internal source/sink that should not exist.

Rationale (FESOM2-JAX, arXiv:2608.01546): every operator here is already
verified near-clean in isolation (fidelity_bar_gate.py), so a climate-
timescale drift must arise where operators are JOINED; a budget-closure
audit locates it directly. Their own bug (freshwater leak in the z* flux
normalization) was visible only as a decadal salinity drift -- invisible to
any single-operator check.

APPLIED FLUX, NOT RE-DERIVED (mandatory, see plan item 0.1 / oracle-fidelity
Rule 5): the T/S applied flux is read back as
``(state_after_forcing - state_before_forcing) / dt`` from the SAME call
the production twin drivers make
(``apply_dino_lat_lon_surface_forcing``, kamm_twin_90d.py:339-340 /
twin_y20_box_budget.py:141-142) -- i.e. the function's own output diffed
against its own input, not a re-derivation from ``DINOConfig`` restoring
constants. This is the one function that changes T/S from outside the
dynamical step for this recipe (confirmed by direct code read, see
DINO SURFACE-FORCING BUDGET note below); volume has NO applied boundary
flux at all for ``nemo_dino_kamm_mlf`` (no freshwater/EMP -- ``model.step``
is called with ``freshwater=None`` in every #1226 twin driver, gating out
both ``freshwater_eta_tendency`` (ocean_model_latlon_cgrid.py:3174) and
``virtual_salt_flux``/``normalized_virtual_salt_flux`` (:4228-4289); DINO's
only salt path is the virtual-salt RESTORING term inside
``apply_dino_lat_lon_surface_forcing``, captured above like T).

DINO SURFACE-FORCING BUDGET (confirmed by direct read, not inferred): the
production twin loop (kamm_twin_90d.py/twin_y20_box_budget.py) calls
``apply_dino_lat_lon_surface_forcing(...)`` THEN ``model.step(st, DT,
surface_forcing=sf)`` where ``sf`` (``dino_step_surface_forcing``, dino.py:
3393-3410) carries WIND ONLY ("Heat/salt/SW stay on the analytic post-step
applicator" -- sf's own docstring). So the closed-domain heat/salt budget's
ONLY applied source is captured by diffing ``apply_dino_lat_lon_surface_
forcing``'s own input/output; ``model.step`` itself (dynamics, GM/Redi,
vertical mixing, Asselin, barotropic/z*) must be a PURE REDISTRIBUTION with
zero net global source for a correct model.

HIDDEN INTERNAL VOLUME CORRECTOR (found during this audit, reported not
fixed -- lane rule): ``LatLonCGridOceanConfig.fix_eta_drift`` defaults
``True`` (state.py:1392) and DINO's ``dino_lat_lon_model_config`` never
overrides it, so by DEFAULT every ``_step_impl`` call (ocean_model_latlon_
cgrid.py:3408-3474) projects a uniform eta correction that forces
``vol(eta_new) == vol(eta_old) + dt*integral(F_slow_eta)`` EXACTLY, every
step -- i.e. production DINO runs never expose their own raw volume drift;
it is silently patched every step. ``ocean_conservation_fixer``
(``use_conservation_fixer``, a SEPARATE flag, default False, matches DINO)
is confirmed OFF for this recipe, so heat/salt are NOT correspondingly
patched -- only volume has a live, always-on internal corrector. This audit
runs BOTH the default (fixer ON, matches every existing #1226 twin
artifact) and a ``--no-fix-eta-drift`` ablation (the true unpatched
residual) so the correction magnitude itself is measured rather than
silently absorbed into a spuriously-clean volume closure.

Usage
-----
    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
        python global_closure_audit.py run <out.npz> [--days 90] \\
            [--no-fix-eta-drift] [--poison Q:MAG]

    python global_closure_audit.py selftest   # non-vacuity self-test only,
                                               # no NEMO restart needed
"""
from __future__ import annotations

import argparse
import dataclasses
import sys

import numpy as np

_THIS_DIR = __file__.rsplit("/", 1)[0]
sys.path.insert(0, _THIS_DIR)
from kamm_twin_90d import _build_twin_state, RUN_TRAJ, DT, STEPS_PER_DAY  # noqa: E402

from legoesm.ocean.experiments.dino import apply_dino_lat_lon_surface_forcing  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64,
    require_explicit_e3t_mode,
    describe_float_leaves,
)
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402
from legoesm import constants  # noqa: E402

RHO0, CP = constants.rho_ocean, constants.c_sw
Y20_RESTART = "DINO_00230400_restart.nc"
# Y20 restart lives in the rebuilt/stitched twin-budget dir, NOT the default
# RUN_STEPDUMP kamm_twin_90d.py points at (that holds the day-180 restart) --
# see twin_y20_box_budget.py's own RUN_STEPDUMP_Y20 (identical path, this
# probe reuses it rather than re-deriving).
RUN_STEPDUMP_Y20 = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TWIN_Y20_BUDGET_STITCHED"


def _global_integrals(state, z_coord, area, min_water_column_m):
    """(volume, heat_J, salt_psu_m3) over the WHOLE wet domain, fp64.

    heat_J = rho0*cp*sum(h*T*area); salt in psu.m^3 (sum(h*S*area)) -- same
    convention as tracer_content_conservation.py's ``content()``, reused
    (not re-derived): a catastrophic-cancellation-safe fp64 accumulation
    over ~372k cells.
    """
    h = np.asarray(compute_layer_thickness(
        np.asarray(state.eta.data, dtype=np.float64),
        np.asarray(state.H_bathy.data, dtype=np.float64), z_coord,
        min_water_column_m=min_water_column_m), dtype=np.float64)
    mask = (np.asarray(state.land_mask.data) > 0.5)[:, :, None]
    vol_cell = h * area[..., None] * mask
    T = np.asarray(state.T.data, dtype=np.float64)
    S = np.asarray(state.S.data, dtype=np.float64)
    V = float(np.sum(vol_cell, dtype=np.float64))
    H = RHO0 * CP * float(np.sum(vol_cell * T, dtype=np.float64))
    Sc = float(np.sum(vol_cell * S, dtype=np.float64))
    return V, H, Sc


def _applied_forcing_flux(state_before, forcing, z_coord, cfg, dt, t_seconds, area,
                          min_water_column_m):
    """APPLIED (not re-derived) dV/dH/dS contribution [m^3/s, W, psu.m^3/s]
    from ``apply_dino_lat_lon_surface_forcing`` for this ONE step, by
    diffing the function's own output against its own input -- the same
    function every #1226 twin driver calls, at the SAME call site
    (kamm_twin_90d.py:339-340).

    Volume: this function never touches ``state.eta`` (T/S/u only, see its
    docstring) -- checked by assertion below rather than assumed, so a
    future change that starts moving eta here cannot silently go
    unaccounted.
    """
    state_after = apply_dino_lat_lon_surface_forcing(
        state_before, forcing, z_coord, cfg, dt, t_seconds=t_seconds)
    d_eta = np.asarray(state_after.eta.data) - np.asarray(state_before.eta.data)
    assert float(np.max(np.abs(d_eta))) == 0.0, (
        "apply_dino_lat_lon_surface_forcing moved eta -- volume-flux "
        "accounting in this probe assumed it never does (docstring: "
        "'Returns a new state (immutable update of T, S, u)'); update "
        "_applied_forcing_flux to capture the eta change too."
    )
    # EXACT (v2 instrument fix): the applied content increment is the SAME
    # integral function differenced ACROSS the forcing call, not a
    # thickness-reweighted tendency.  eta is unchanged (asserted above), so h
    # is bit-identical on both sides and the difference is exactly
    # rho0*cp*integral(h*dT) under the IDENTICAL mask/floor convention the
    # closed-domain integral uses.  v1 evaluated h with
    # ``min_water_column_m=None`` while ``_global_integrals`` used the config
    # floor -- an inconsistency on the applied side; the difference form
    # removes that whole class of convention error by construction, and makes
    # the reported residual EXACTLY the net global content change caused by
    # ``model.step`` alone (forcing contribution cancels identically).
    V_b, H_b, S_b = _global_integrals(state_before, z_coord, area, min_water_column_m)
    V_a, H_a, S_a = _global_integrals(state_after, z_coord, area, min_water_column_m)
    return state_after, (V_a - V_b) / dt, (H_a - H_b) / dt, (S_a - S_b) / dt


def run_audit(out_path: str, n_days: int, *, fix_eta_drift: bool = True,
              poison: str | None = None, restart_file: str = Y20_RESTART,
              asselin_gamma: float | None = None):
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    e3t_mode = require_explicit_e3t_mode(context="global_closure_audit")
    print(f"LEGOESM_NEMO_E3T={e3t_mode}")

    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        "nemo_dino_kamm_mlf", RUN_TRAJ, RUN_STEPDUMP_Y20, bridge_tke=True,
        bridge_before=True, restart_file=restart_file,
    )
    require_fp64(br.geometry, br.z_coord, st, context="global_closure_audit")
    print("DTYPES (state, first 5 float leaves):", describe_float_leaves(st)[:5])
    print("DTYPES (z_coord, first 5 float leaves):", describe_float_leaves(br.z_coord)[:5])

    if not fix_eta_drift:
        mc = mc._replace(fix_eta_drift=False)
    # ATF/leapfrog discriminator: gamma=0 removes the Robert-Asselin filter
    # entirely, so a residual that is carried by the filter's time-level
    # bookkeeping collapses while a genuine flux-form leak does not.
    if asselin_gamma is not None:
        mc = mc._replace(asselin_gamma=asselin_gamma)
        print(f"ABLATION: asselin_gamma={asselin_gamma}")
    print(f"fix_eta_drift={mc.fix_eta_drift}  use_conservation_fixer={mc.use_conservation_fixer}")
    assert mc.use_conservation_fixer is False, (
        "use_conservation_fixer=True would force-conserve heat/salt every "
        "step (ocean_conservation_fixer) and make this audit vacuous for "
        "those two quantities; DINO's recipe default is False -- if that "
        "changed, this probe must be updated, not silently pass.")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)

    import jax
    import jax.numpy as jnp

    area = np.asarray(br.geometry.area, dtype=np.float64)
    dyn = jax.jit(lambda st, t: model.step(st, DT, surface_forcing=sf, t_seconds=t))

    # --- Non-vacuity self-test injection (mandatory, item 4): perturb ONE
    # tracer field by a known relative amount right after day 0 and confirm
    # the SAME accounting machinery below detects it at the injected
    # magnitude, in the SAME units. Applied to the state the integral is
    # read from -- not a separate code path -- so a real leak of this shape
    # is provably not silently absorbed.
    poison_q, poison_mag = None, 0.0
    if poison:
        poison_q, mag_str = poison.split(":")
        poison_mag = float(mag_str)

    V0, H0, S0 = _global_integrals(st, br.z_coord, area, mc.min_water_column_m)
    print(f"day 0 integrals: V={V0:.10e} m^3  H={H0:.10e} J  S={S0:.10e} psu.m^3")

    nsteps = STEPS_PER_DAY * n_days
    t_seconds = 0.0
    rows = []  # per-day: t, V,H,S, applied dV/dH/dS integrated, residuals
    accum_dV_applied = accum_dH_applied = accum_dS_applied = 0.0
    V_prev, H_prev, S_prev = V0, H0, S0

    for k in range(nsteps):
        t_next = (k + 1) * DT
        st, dV_dt, dH_dt, dS_dt = _applied_forcing_flux(
            st, forcing, br.z_coord, cfg, DT, t_next, area,
            mc.min_water_column_m)
        accum_dV_applied += dV_dt * DT
        accum_dH_applied += dH_dt * DT
        accum_dS_applied += dS_dt * DT

        st = dyn(st, jnp.asarray(t_next))
        t_seconds = t_next

        if k == 0 and poison_q is not None:
            # Inject AFTER day-0 integrals are recorded, ONCE, then let the
            # normal loop continue -- the injected quantity's OWN next
            # integral sample must show the poison as a step discontinuity
            # of the expected magnitude.
            if poison_q == "salt":
                st = st._replace(S=st.S.replace(
                    data=st.S.data * (1.0 + poison_mag)))
            elif poison_q == "heat":
                st = st._replace(T=st.T.replace(
                    data=st.T.data * (1.0 + poison_mag)))
            elif poison_q == "volume":
                st = st._replace(eta=st.eta.replace(
                    data=st.eta.data + poison_mag))
            else:
                raise SystemExit(f"unknown --poison quantity {poison_q!r}")
            print(f"POISON INJECTED at end of step 1: {poison_q} *= "
                  f"(1+{poison_mag:.3e})" if poison_q != "volume" else
                  f"POISON INJECTED at end of step 1: eta += {poison_mag:.3e} m")

        Td = np.asarray(st.T.data)
        m = np.asarray(st.land_mask.data) > 0.5
        if not np.isfinite(Td[m]).all():
            raise SystemExit(f"NaN/Inf at day {t_seconds/86400:.2f} -- aborting")

        if (k + 1) % STEPS_PER_DAY == 0:
            V, H, S = _global_integrals(st, br.z_coord, area, mc.min_water_column_m)
            dV_actual, dH_actual, dS_actual = V - V_prev, H - H_prev, S - S_prev
            res_V = dV_actual - accum_dV_applied
            res_H = dH_actual - accum_dH_applied
            res_S = dS_actual - accum_dS_applied
            day = (k + 1) // STEPS_PER_DAY
            rows.append((t_seconds, V, H, S, accum_dV_applied, accum_dH_applied,
                         accum_dS_applied, res_V, res_H, res_S))
            print(f"  day {t_seconds/86400:5.1f}  "
                  f"res_V={res_V:+.6e} m^3 (rel {res_V/max(abs(V0),1e-30):+.3e})  "
                  f"res_H={res_H:+.6e} J (rel {res_H/max(abs(H0),1e-30):+.3e})  "
                  f"res_S={res_S:+.6e} psu.m^3 (rel {res_S/max(abs(S0),1e-30):+.3e})",
                  flush=True)
            accum_dV_applied = accum_dH_applied = accum_dS_applied = 0.0
            V_prev, H_prev, S_prev = V, H, S

    rows = np.array(rows, dtype=np.float64)
    np.savez(out_path,
              t=rows[:, 0], V=rows[:, 1], H=rows[:, 2], S=rows[:, 3],
              applied_dV=rows[:, 4], applied_dH=rows[:, 5], applied_dS=rows[:, 6],
              residual_V=rows[:, 7], residual_H=rows[:, 8], residual_S=rows[:, 9],
              V0=V0, H0=H0, S0=S0, fix_eta_drift=fix_eta_drift,
              e3t_mode=e3t_mode, restart_file=restart_file,
              poison=poison or "", n_days=n_days,
              asselin_gamma=(np.nan if asselin_gamma is None else asselin_gamma))
    print(f"SAVED {out_path}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("out")
    p_run.add_argument("--days", type=int, default=90)
    p_run.add_argument("--no-fix-eta-drift", action="store_true",
                        help="ablate the always-on volume corrector -- the "
                             "TRUE unpatched closure residual")
    p_run.add_argument("--poison", default=None,
                        help="non-vacuity injection, e.g. salt:1e-6, "
                             "heat:1e-6, volume:0.01 (eta shift in metres)")
    p_run.add_argument("--restart-file", default=Y20_RESTART)
    p_run.add_argument("--asselin-gamma", type=float, default=None,
                        help="override the Robert-Asselin coefficient (0.0 = "
                             "filter OFF) -- the ATF/leapfrog discriminator")

    sub.add_parser("selftest")
    return p.parse_args(argv)


def _selftest():
    """Runnable, NEMO-free check that the residual accounting arithmetic
    itself is correct (not a claim about the model): synthetic V/H/S series
    with a KNOWN injected leak must reproduce that leak in the residual.
    """
    rng = np.random.default_rng(0)
    n = 50
    V = 1000.0 + np.cumsum(rng.normal(0, 1e-6, n))
    applied = np.diff(V, prepend=V[0])
    applied[0] = 0.0
    leak_step, leak_mag = 20, 5.0
    V_leaky = V.copy()
    V_leaky[leak_step:] += leak_mag
    residual = np.diff(V_leaky, prepend=V_leaky[0]) - applied
    residual[0] = 0.0
    detected = residual[leak_step]
    assert abs(detected - leak_mag) < 1e-9, (
        f"self-test FAILED: injected {leak_mag}, detected {detected}")
    print(f"SELFTEST OK: injected leak {leak_mag} -> detected residual "
          f"{detected:.6f} at step {leak_step}")


def main(argv=None):
    args = _parse_args(argv)
    if args.mode == "selftest":
        _selftest()
        return
    run_audit(args.out, args.days, fix_eta_drift=not args.no_fix_eta_drift,
              poison=args.poison, restart_file=args.restart_file,
              asselin_gamma=args.asselin_gamma)


if __name__ == "__main__":
    main()
