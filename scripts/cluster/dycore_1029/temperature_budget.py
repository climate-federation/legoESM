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
from legoesm.grids.vertical import dp_from_hybrid

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
            if q.shape == state.T.shape:
                cap["V"] = out
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
                r = tendency_fn(s)
                stages.append(dict(H=H, V=V, OM=OM, G=dyn - H - V - OM,
                                   LAP=no4 - dyn, D4=full - no4, PF=r.T - full,
                                   rec=r.T, Tmin=float(jnp.min(s.T))))
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
                Tmin_prefloor=rails["Tmin_prefloor"])
    return out, terms, info


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


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from burst_anatomy_probe import build, runner

    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    meta = json.loads((a.probe_dir / "meta.json").read_text())
    nu4 = runner._biharmonic_visc_latlon(72)
    if abs(nu4 - meta["nu_del4"]) > 1e-9 * nu4:
        raise SystemExit(f"nu_del4 {nu4} != probe's {meta['nu_del4']}")
    grid, sigma, model, st0, dt, ah = build(meta["h0"], meta["seed"], nu4=nu4)
    if abs(dt - meta["dt"]) > 0 or abs(ah - meta["A_h"]) > 1e-9 * ah:
        raise SystemExit("dt / A_h differ from the probe run")
    target = model.compute_mass(st0)                 # anchor_mass_to_initial
    clean = sorted(a.probe_dir.glob("clean_*.npz"), key=lambda p: int(p.stem[6:]))
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
        res["intervals"].append(row)
        print(json.dumps({k: row[k] for k in (
            "step_from", "C1_pass", "C1_resid", "C2_pass", "C2_resid", "cold", "hot",
            "net_Jm2", "column_Jm2", "hot_Jm2", "A_adiabatic_share", "B_floor_share",
            "Tmin_prefloor_global")}), flush=True)
        print("  T0     ", row["T0"])
        for t in TERMS:
            print(f"  {t:<6} ", row["levels"][t])
    if a.out:
        a.out.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
