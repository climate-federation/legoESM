"""#1029 probe: term-by-term temperature budget of ONE model step in the hot column.

Takes the saved pre-trip states of ``burst_anatomy_probe.py`` (job 28432774,
``loc4_*/probe_s<seed>/clean_<step>.npz``), rebuilds the SAME model
(``burst_anatomy_probe.build`` with the matrix del-4 coefficient), and steps
each saved state once through the model's own ``_step_cgrid_impl``.  Nothing
is re-derived: the terms are captured from the code path while it runs.

Terms (K per step, column of the hottest cell of the last saved state):
  H    horizontal advection     captured output of cgrid_fv_scalar_advection_latlon_3d
  V    vertical advection       captured output of vertical_advection_hybrid (T call)
  OM   omega/p conversion       kappa*max(T,T_min)*captured sb81_omega_over_p_dyn
  G    v.grad(ln p) conversion  remainder dT(nu4=0,A_h=0) - H - V - OM (the only other
                                summand on the code's ``dT_dt = horiz + vert + adiabatic``)
  LAP  del-2 diffusion          dT(nu4=0) - dT(nu4=0, A_h=0)
  D4   del-4 diffusion          dT(full) - dT(nu4=0)
  PF   polar filter             recorded stage tendency - dT(full)
each combined over the SSP-RK3 stages with the scheme's net weights
dt*(1/6, 1/6, 2/3); then
  HS   Held-Suarez relaxation   T change across _apply_physics_increment
  FL   50 K floor (+rails)      T change across _apply_safety_rails

CONTROLS (budget read only if both hold):
  C1 internal closure: RK-weighted sum of the recorded stage tendencies equals
     the integrator's T increment, and H+V+OM+G+LAP+D4+PF+HS+FL equals the
     model's T change, to 1e-9 x max|dT| in the column.
  C2 instrument: this CPU step of saved state n reproduces saved state n+1
     (written by the GPU run): max|T_cpu - T_saved| in the column <= 1e-6 x
     max|T_saved(n+1) - T(n)| in the column.  The step out of the LAST saved
     state has no saved successor (it is the trip step) and is labelled so.

READING RULE (written before the run).  Column levels: COLD = at the floor
(T <= T_min + 0.01 K) in state n; HOT = the other levels between the first and
last COLD level.  Sums are enthalpy-weighted, c_p dT dp/g [J/m2 per step].
  A  "vertical-motion heating runaway in a statically unstable column":
     CONFIRMED in an interval if the adiabatic term (OM+G) summed over HOT is
     positive AND >= 50% of the sum of all positive terms over HOT;
     REFUTED if it is <= 0 or < 25% of that sum.
  B  "two-level vertical mode pumped by the one-sided 50 K floor":
     CONFIRMED in an interval if FL over the column is >= 50% of the column's
     net enthalpy change AND the net change is positive; REFUTED if FL is
     < 10% of |net|.  The pre-floor minimum T (hidden negative T) is printed.
  Verdict per mechanism needs both seeds and both validated intervals;
  otherwise the per-interval numbers are reported with no mechanism claimed.

FOLLOW-UP CHECKS (user-approved 2026-10-26; rules written after the budget
run above, before these two were run -- so not blind to its numbers).
  C  vertical Courant number of the T vertical advection, as the scheme sees
     it (``vertical_courant``: |F| dt / upwind p_full spacing, F = the model's
     own interface mass flux captured from the T call), at RK stage 1 (the
     saved state) and max over the 3 stages, over BAND = levels first..last
     COLD inclusive.
     INSTRUMENT CONTROLS (C read only if both hold): (i) identity
     |V| dt == C |upwind dT| at every cell of the domain, to 1e-9 x max |V| dt;
     (ii) quiet column (same row, 180 deg of longitude away): max C over the
     column < 0.2 and its T has no level at the floor.
     CONFIRMED "explicit vertical overshoot (C > 1) in the hot column" if the
     stage-1 BAND max C >= 1 at both validated intervals of both seeds;
     REFUTED if the BAND max C over all 3 stages < 0.5 at all validated
     intervals; otherwise reported, no claim.
  F  floor-free re-step: from each seed's EARLIEST saved state, 3 steps of the
     unchanged model with the floor's added heat (the captured FL term)
     subtracted after every step (``floor_free_run``).  Swing amplitude
     A = max_k |T_k - (T_{k-1}+T_{k+1})/2| in the hot column.
     CONFIRMED "the swing grows without the floor" if A(2 steps)/A(0) >= 1.5
     for both seeds; REFUTED ("the floor is needed for growth") if
     A(2)/A(0) <= 1.0 for both seeds; otherwise reported, no claim.  Step 3
     corresponds to the trip step and is reported only.  Reference: the same
     ratio on the floored run's own saved states.

Usage: python temperature_budget.py <probe_dir> [--out budget.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from legoesm import constants
import legoesm.grids.vertical as _vert
import legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid as _pe
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState, cgrid_latlon_hydrostatic_tendencies)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_forcing_latlon)
from legoesm.grids.vertical import dp_from_hybrid, pressure_from_hybrid

TERMS = ("H", "V", "OM", "G", "LAP", "D4", "PF", "HS", "FL")
RK3_W = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)   # ssp_rk3_step net stage weights


@contextmanager
def _patched(obj, name, wrapper):
    own = name in vars(obj)          # False for a method seen through an instance
    orig = getattr(obj, name)
    setattr(obj, name, wrapper(orig))
    try:
        yield
    finally:
        if own:
            setattr(obj, name, orig)
        else:
            delattr(obj, name)


def step_budget(model, state, target_mass, physics_fn=held_suarez_forcing_latlon):
    """One model step of ``state`` with every T term captured (full 3-D fields).

    Returns ``(state_out, terms, info)``: ``terms[name]`` is the K-per-step
    contribution of each term; ``info`` carries the closure residuals and the
    pre-floor / per-stage minimum temperatures.
    """
    cfg, dt = model.config, model.dt
    if cfg.time_integrator != "ssp_rk3" or cfg.energy_fixer:
        raise ValueError("budget weights assume ssp_rk3 and energy_fixer off")
    cap = {}

    def rec_hadv(f):
        def w(*a, **k):
            out = f(*a, **k)
            cap["H"] = out
            return out
        return w

    def rec_vadv(f):
        def w(q, *a, **k):
            out = f(q, *a, **k)
            if q.shape == state.T.shape:     # u, v, T in that order: T wins
                cap["V"], cap["Q"], cap["MF"], cap["PS"] = out, q, a[0], a[1]
            return out
        return w

    def rec_om(f):
        def w(*a, **k):
            out = f(*a, **k)
            cap["X"] = out
            return out
        return w

    stages = []

    def tend(s, c):
        return cgrid_latlon_hydrostatic_tendencies(
            s, model.grid, model.sigma_coord, c, dt=dt)[2]

    def rec_dispatch(f):
        def w(s0, tendency_fn, dt_, name):
            def tf(s):
                full = tend(s, cfg)
                no4 = tend(s, cfg._replace(nu_del4=0.0))
                cap.clear()
                dyn = tend(s, cfg._replace(nu_del4=0.0, A_h=0.0))
                H, V = cap["H"], cap["V"]
                OM = constants.kappa * jnp.maximum(s.T, cfg.T_min) * cap["X"]
                # cap["Q"]: the T the call received (floor-clipped inside the
                # tendency), not the raw stage T, which goes negative.
                C, dfu = vertical_courant(cap["Q"], cap["MF"], cap["PS"],
                                          model.sigma_coord, dt)
                r = tendency_fn(s)
                stages.append(dict(H=H, V=V, OM=OM, G=dyn - H - V - OM,
                                   LAP=no4 - dyn, D4=full - no4, PF=r.T - full,
                                   rec=r.T, Tmin=float(jnp.min(s.T)),
                                   C=C, C_ident=jnp.abs(V) * dt - C * jnp.abs(dfu)))
                return r
            out = f(s0, tf, dt_, name)
            stages.append(dict(dyn_incr=out.T - s0.T))
            return out
        return w

    rails, phys = {}, {}

    def rec_phys(f):
        def w(st, *a, **k):
            out = f(st, *a, **k)
            phys["HS"] = out.T - st.T
            return out
        return w

    def rec_rails(f):
        def w(st, *a, **k):
            out = f(st, *a, **k)
            rails["FL"] = out.T - st.T
            rails["Tmin_prefloor"] = float(jnp.min(st.T))
            return out
        return w

    with _patched(_pe, "cgrid_fv_scalar_advection_latlon_3d", rec_hadv), \
            _patched(_pe, "vertical_advection_hybrid", rec_vadv), \
            _patched(_vert, "sb81_omega_over_p_dyn", rec_om), \
            _patched(_pe, "dispatch_integrator", rec_dispatch), \
            _patched(model, "_apply_physics_increment", rec_phys), \
            _patched(model, "_apply_safety_rails", rec_rails):
        out, _ = model._step_cgrid_impl(state, dt, target_mass, physics_fn)

    dyn_incr = stages.pop()["dyn_incr"]
    if len(stages) != 3:
        raise RuntimeError(f"expected 3 RK stages, recorded {len(stages)}")
    terms = {n: dt * sum(w * st[n] for w, st in zip(RK3_W, stages))
             for n in TERMS[:7]}
    terms["HS"] = phys.get("HS", jnp.zeros_like(state.T))
    terms["FL"] = rails["FL"]
    rk_rec = dt * sum(w * st["rec"] for w, st in zip(RK3_W, stages))
    info = dict(rk_vs_integrator=rk_rec - dyn_incr,
                sum_vs_step=sum(terms.values()) - (out.T - state.T),
                Tmin_stages=[st["Tmin"] for st in stages],
                Tmin_prefloor=rails["Tmin_prefloor"],
                courant_stages=[st["C"] for st in stages],
                courant_ident=[st["C_ident"] for st in stages],
                courant_scale=[dt * float(jnp.max(jnp.abs(st["V"]))) for st in stages])
    return out, terms, info


def vertical_courant(field, mass_flux, p_s, coord, dt):
    """Upwind vertical Courant number of ``vertical_advection_hybrid``.

    Mirrors that scheme exactly: full-level flux F = mean of the two interface
    fluxes, upwind side backward where F > 0, spacing = |p_full difference| on
    that side; C = |F| dt / spacing (0 where the upwind gradient is the
    zero pad).  Returns ``(C, dfu)`` with dfu the upwind field difference, so
    |V| dt == C |dfu| is the identity that ties C to the captured tendency V.
    """
    F = 0.5 * (mass_flux[..., :-1] + mass_flux[..., 1:])
    dp = jnp.clip(jnp.abs(jnp.diff(pressure_from_hybrid(coord, p_s, full=True),
                                   axis=-1)), 1e-10, None)
    df = jnp.diff(field, axis=-1)
    pad = ((0, 0),) * (df.ndim - 1)
    bwd = F > 0
    sp = jnp.where(bwd, jnp.pad(dp, (*pad, (1, 0))), jnp.pad(dp, (*pad, (0, 1))))
    dfu = jnp.where(bwd, jnp.pad(df, (*pad, (1, 0))), jnp.pad(df, (*pad, (0, 1))))
    C = jnp.where(sp > 0, jnp.abs(F) * dt / jnp.where(sp > 0, sp, 1.0), 0.0)
    return C, dfu


def swing_amplitude(T_col):
    """Level-alternating (2-level) amplitude: max_k |T_k - (T_{k-1}+T_{k+1})/2|."""
    T_col = np.asarray(T_col)
    return float(np.max(np.abs(T_col[1:-1] - 0.5 * (T_col[:-2] + T_col[2:]))))


def floor_free_run(model, state, target, n, ij):
    """``n`` model steps of ``state`` with the floor's added heat set aside.

    Each step is the unchanged model step; afterwards its captured floor term
    FL is subtracted, i.e. T is carried on as it was just before the 50 K
    floor (the in-tendency max(T, T_min) of the omega/p term is untouched).
    Returns per-step dicts for column ``ij``.
    """
    i, j = ij
    rows = [dict(step=0, A=swing_amplitude(np.asarray(state.T)[i, j]),
                 Tmax=float(np.max(np.asarray(state.T)[i, j])),
                 Tmin=float(np.min(np.asarray(state.T)[i, j])))]
    s = state
    for k in range(1, n + 1):
        out, terms, _ = step_budget(model, s, target)
        s = out._replace(T=out.T - terms["FL"])
        Tc = np.asarray(s.T)[i, j]
        rows.append(dict(step=k, A=swing_amplitude(Tc), Tmax=float(np.max(Tc)),
                         Tmin=float(np.min(Tc)), finite=bool(np.all(np.isfinite(Tc)))))
    return rows


def load_state(npz):
    z = np.load(npz)          # phis: the run's own (GPU-built) mountain
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(z["u"]), v=jnp.asarray(z["v"]), T=jnp.asarray(z["T"]),
        p_s=jnp.asarray(z["p_s"]), phis=jnp.asarray(z["phis"]), tracers={})


def column_summary(T0, terms, info, dp, T_min):
    """Per-level column numbers + the reading-rule sums for one interval."""
    T0 = np.asarray(T0)
    w = constants.c_pd * np.asarray(dp) / constants.g        # J/m2/K per level
    tc = {n: np.asarray(v) for n, v in terms.items()}
    cold = np.where(T0 <= T_min + 0.01)[0]
    hot = (np.array([k for k in range(cold[0], cold[-1] + 1) if k not in cold])
           if cold.size else np.array([], int))
    net = float(np.sum(w * sum(tc.values())))
    hotsum = {n: float(np.sum((w * tc[n])[hot])) for n in TERMS}
    pos = sum(v for v in hotsum.values() if v > 0)
    adi = hotsum["OM"] + hotsum["G"]
    fl = float(np.sum(w * tc["FL"]))
    return dict(
        cold=cold.tolist(), hot=hot.tolist(), net_Jm2=net,
        hot_Jm2=hotsum, column_Jm2={n: float(np.sum(w * tc[n])) for n in TERMS},
        A_adiabatic_share=(adi / pos if pos > 0 else float("nan")),
        A_adiabatic_hot_Jm2=adi,
        B_floor_share=(fl / net if net != 0 else float("nan")),
        B_floor_Jm2=fl,
        levels={n: np.round(tc[n], 3).tolist() for n in TERMS},
        T0=np.round(T0, 2).tolist())


def load_probe(probe_dir):
    """Rebuild the probe's model and list its saved states (checked vs meta)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from burst_anatomy_probe import build, runner

    meta = json.loads((probe_dir / "meta.json").read_text())
    nu4 = runner._biharmonic_visc_latlon(72)
    if abs(nu4 - meta["nu_del4"]) > 1e-9 * nu4:
        raise SystemExit(f"nu_del4 {nu4} != probe's {meta['nu_del4']}")
    grid, sigma, model, st0, dt, ah = build(meta["h0"], meta["seed"], nu4=nu4)
    if abs(dt - meta["dt"]) > 0 or abs(ah - meta["A_h"]) > 1e-9 * ah:
        raise SystemExit("dt / A_h differ from the probe run")
    target = model.compute_mass(st0)                 # anchor_mass_to_initial
    clean = sorted(probe_dir.glob("clean_*.npz"), key=lambda p: int(p.stem[6:]))
    return meta, grid, sigma, model, st0, target, clean


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    meta, grid, sigma, model, st0, target, clean = load_probe(a.probe_dir)
    states = [load_state(c) for c in clean]
    # The CPU rebuild of the mountain differs from the run's float32 phis by
    # rounding only (seed 42: 0.07 of 19278 m2/s2); the saved field is used,
    # and the difference is printed as provenance.
    dphis = float(np.max(np.abs(np.asarray(states[0].phis, float)
                                - np.asarray(st0.phis, float))))
    print(f"max |phis_saved - phis_rebuilt| = {dphis:.3g} m2/s2")
    i, j, _ = np.unravel_index(int(np.argmax(np.asarray(states[-1].T))),
                               states[-1].T.shape)
    lat = float(np.degrees(np.asarray(grid.lat))[i])
    lon = float(np.degrees(np.asarray(grid.lon))[j])
    print(f"seed {meta['seed']} hot column ({i},{j}) = {lat:.2f}N {lon:.2f}E")
    res = dict(meta=meta, column=[int(i), int(j), lat, lon], intervals=[])
    for n, (c, s) in enumerate(zip(clean, states)):
        out, terms, info = step_budget(model, s, target)
        col = lambda f: np.asarray(f)[i, j]  # noqa: E731
        dT_col = col(out.T) - col(s.T)
        scale = float(np.max(np.abs(dT_col)))
        c1 = max(float(np.max(np.abs(col(info["rk_vs_integrator"])))),
                 float(np.max(np.abs(col(info["sum_vs_step"])))))
        row = dict(step_from=int(c.stem[6:]), max_abs_dT_col=scale,
                   C1_resid=c1, C1_pass=c1 <= 1e-9 * scale,
                   Tmin_stages_global=info["Tmin_stages"],
                   Tmin_prefloor_global=info["Tmin_prefloor"])
        if n + 1 < len(states):
            ref = col(states[n + 1].T)
            sref = float(np.max(np.abs(ref - col(s.T))))
            c2 = float(np.max(np.abs(col(out.T) - ref)))
            row.update(C2_resid=c2, C2_pass=c2 <= 1e-6 * sref)
        else:
            row.update(C2_resid=None, C2_pass=None, note="trip step: no saved successor")
        dp = dp_from_hybrid(sigma, s.p_s)[i, j]
        row.update(column_summary(col(s.T), {k: col(v) for k, v in terms.items()},
                                  info, dp, model.config.T_min))
        Cs = [np.asarray(c) for c in info["courant_stages"]]
        band = (list(range(row["cold"][0], row["cold"][-1] + 1))
                if row["cold"] else [])
        jq = (j + Cs[0].shape[1] // 2) % Cs[0].shape[1]
        ident = max(float(np.max(np.abs(np.asarray(x)))) / sc
                    for x, sc in zip(info["courant_ident"], info["courant_scale"]))
        row.update(
            C_band_stage1=float(np.max(Cs[0][i, j, band])) if band else None,
            C_band_all_stages=(float(max(np.max(c[i, j, band]) for c in Cs))
                               if band else None),
            C_col_stage1=np.round(Cs[0][i, j], 3).tolist(),
            C_quiet_max=float(np.max(Cs[0][i, jq])),
            quiet_floor_levels=int(np.sum(np.asarray(s.T)[i, jq]
                                          <= model.config.T_min + 0.01)),
            C_ident_rel=ident, C_ident_pass=ident <= 1e-9)
        print(json.dumps({k: row[k] for k in (
            "step_from", "C_band_stage1", "C_band_all_stages", "C_quiet_max",
            "quiet_floor_levels", "C_ident_rel", "C_ident_pass")}), flush=True)
        print("  C(st1) ", row["C_col_stage1"])
        res["intervals"].append(row)
        print(json.dumps({k: row[k] for k in (
            "step_from", "C1_pass", "C1_resid", "C2_pass", "C2_resid", "cold", "hot",
            "net_Jm2", "column_Jm2", "hot_Jm2", "A_adiabatic_share", "B_floor_share",
            "Tmin_prefloor_global")}), flush=True)
        print("  T0     ", row["T0"])
        for t in TERMS:
            print(f"  {t:<6} ", row["levels"][t])
    ff = floor_free_run(model, states[0], target, 3, (i, j))
    ref = [swing_amplitude(np.asarray(x.T)[i, j]) for x in states]
    res["floor_free"] = dict(rows=ff, A_ratio_2=ff[2]["A"] / ff[0]["A"],
                             floored_A=ref, floored_A_ratio_2=ref[2] / ref[0])
    print("floor-free:", json.dumps(res["floor_free"]), flush=True)
    if a.out:
        a.out.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
