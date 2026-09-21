"""Does the formation-profile rain debit's capacity cap bind, and what drives
the day-1 rain change?  One offline call of the production Bechtold kernel
(un-jitted) on a checkpoint's tropical-ocean columns, with the sink helper
wrapped to RECORD its inputs and outputs, so the cap scale, the binding
levels and the excess are read directly rather than inferred from the
returned rain (which the downdraft/sub-cloud evaporation rescale further).

Modes
  default            both schemes on the same state: rain, sink, heat, and the
                     recorded scale/binding statistics over ALL columns
  --swap-with RUN    formation scheme on six states: each run's full state
                     (known-answer rows), then T, q, T+q, or the carried
                     convection state (mass-flux memory + stochastic state)
                     taken from RUN into this run's state.  Each row mixes
                     trigger, closure, conversion and evaporation responses;
                     none isolates the closure.

Limits (state them next to any number): winds, dynamics tendencies and
surface fluxes are zeros/dummies (not in the checkpoint); the diurnal
correction and the closure's moisture convergence therefore see dummies,
identically in every call of one invocation.

Usage: rain_sink_scale_offline.py <run> [--day 135] [--sftlf-from run] [--box BOX]
                                  [--swap-with run] [--stride N]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloud_layers import mesh_coords  # noqa: E402
from ledger_regional import BOXES, SEC_PER_DAY, _sftlf_on_mesh  # noqa: E402
from legoesm import constants  # noqa: E402
from legoesm.driver.config import experiment_config_from_dict  # noqa: E402
from legoesm.driver.physics_pipeline import _resolve_convection  # noqa: E402
import legoesm.atmosphere.physics.convection.bechtold as B  # noqa: E402

ROOT = os.environ.get("AMIP_ROOT", "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")


class _Recorder:
    """Wraps distribute_rain_vapor_sink; keeps the formation call's arrays.
    The capacity mirror runs in the helper's own dtype so borderline
    want > cap classifications match the kernel's."""
    def __init__(self):
        self.orig = B.distribute_rain_vapor_sink
        self.last = None
        self.calls = []

    def __call__(self, form, q_v, dq_v_dt, dp, dt, scheme):
        sink, scale = self.orig(form, q_v, dq_v_dt, dp, dt, scheme)
        self.calls.append(scheme)
        if scheme == "formation":
            q_v, dq_v_dt = np.asarray(q_v), np.asarray(dq_v_dt)
            cap = (B._RAIN_SINK_CAPACITY_FRAC
                   * np.maximum(q_v + q_v.dtype.type(dt) * dq_v_dt, 0.0) / q_v.dtype.type(dt))
            self.last = dict(want=np.maximum(np.asarray(form), 0.0), cap=cap,
                             sink=np.asarray(sink), scale=np.asarray(scale), dp=np.asarray(dp))
        return sink, scale


def _state(run, day, sel):
    z = np.load(f"{ROOT}/{run}/checkpoint_day_{day:04d}.npz", allow_pickle=True)
    return dict(T=np.asarray(z["T"])[sel], q=np.asarray(z["trc_q_v"])[sel],
                ps=np.asarray(z["p_s"])[sel], vg=np.asarray(z["meta_vgrid"]),
                prog=np.asarray(z["physstate_conv_prog_profile"])[sel],
                stoch=np.asarray(z["physstate_conv_stoch_state"])[sel],
                rad=np.asarray(z["physstate_rad_heating"])[sel])


def _call(bcfg, st, fl_sel, dt, scheme, T=None, q=None):
    T = st["T"] if T is None else T
    q = st["q"] if q is None else q
    p_half = st["vg"][0][None, :] * constants.p_ref + st["vg"][1][None, :] * st["ps"][:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    ncol, nlev = T.shape
    zz = jnp.zeros((ncol, nlev))
    out, M_u, _ = B.bechtold_convection(
        jnp.asarray(T), jnp.asarray(q), jnp.asarray(p_full), jnp.asarray(p_half), zz, zz,
        jnp.asarray(st["prog"]), jnp.asarray(st["stoch"]), None, dt,
        config=bcfg._replace(rain_vapor_sink=scheme),
        shf_w_m2=jnp.full((ncol,), -20.0), lhf_w_m2=jnp.full((ncol,), -100.0),
        dT_dt_rad=jnp.asarray(st["rad"]), land_frac=jnp.asarray(fl_sel),
        dT_dt_dyn=zz, dq_dt_dyn=zz)
    m = (p_half[:, 1:] - p_half[:, :-1]) / constants.g
    rain = np.sum(np.asarray(out.dq_r_conv_dt, np.float64) * m, axis=1) * SEC_PER_DAY
    sink = -np.sum(np.asarray(out.dq_v_dt, np.float64) * m, axis=1) * SEC_PER_DAY
    heat = (np.sum(np.asarray(out.dT_dt, np.float64) * m, axis=1)
            * constants.c_pd / constants.L_v * SEC_PER_DAY)
    mu_max = np.max(np.asarray(M_u, np.float64), axis=1)   # relaxed profile's peak, NOT the closure's M_b
    return rain, sink, heat, mu_max


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--day", type=int, default=135)
    ap.add_argument("--sftlf-from"); ap.add_argument("--box", default="ITCZ 10S-10N")
    ap.add_argument("--swap-with"); ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args(argv)
    exp = json.load(open(f"{ROOT}/{a.run}/experiment_config.json"))
    lat, lon, area = mesh_coords(exp)
    fl = _sftlf_on_mesh(a.sftlf_from or a.run, lat, lon)
    lo, hi = BOXES[a.box]
    sel = np.where((lat >= lo) & (lat <= hi) & (fl < 0.01))[0][::a.stride]
    w = area[sel] / area[sel].sum()
    dt = float(exp["dycore"]["dt"])
    _, bcfg = _resolve_convection(experiment_config_from_dict(exp))
    rec = _Recorder(); B.distribute_rain_vapor_sink = rec
    try:
        _run(a, rec, bcfg, lat, fl, sel, w, dt)
    finally:
        B.distribute_rain_vapor_sink = rec.orig


def _run(a, rec, bcfg, lat, fl, sel, w, dt):
    st = _state(a.run, a.day, sel)
    tag = f"{a.box} ocean, {len(sel)} cols, day {a.day}"
    if a.swap_with:
        so = _state(a.swap_with, a.day, sel)
        assert so["T"].shape == st["T"].shape and np.array_equal(so["vg"], st["vg"]), "different grids"
        # Rows 1-2: known-answer checks (each run's FULL state incl. its carried
        # mass-flux memory).  Rows 3-5: swap T, q, or both into this run's
        # state keeping its memory (instantaneous thermodynamic response,
        # memory fixed).  Row 6: only the memory swapped.  Each row mixes
        # trigger, closure, conversion and evaporation responses; none isolates
        # the closure -- that needs the pre-relaxation closure mass flux, which
        # the kernel does not return.  Swapped T rides this run's ps.
        st_mem = dict(st, prog=so["prog"], stoch=so["stoch"])   # memory AND stochastic state
        rows = [(f"full state {a.run}", st, None, None), (f"full state {a.swap_with}", so, None, None),
                (f"T from {a.swap_with}, rest {a.run}", st, so["T"], None),
                (f"q from {a.swap_with}, rest {a.run}", st, None, so["q"]),
                (f"T+q from {a.swap_with}, rest {a.run}", st, so["T"], so["q"]),
                (f"conv state (memory+stoch) from {a.swap_with}", st_mem, None, None)]
        print(f"=== formation scheme on swapped states ({tag}) [kg/m2/day; peak M_u kg/m2/s] ===")
        for name, base, T, q in rows:
            rain, sink, heat, mu = _call(bcfg, base, fl[sel], dt, "formation", T, q)
            print(f"{name:44s} rain {np.sum(w*rain):.3f}  sink {np.sum(w*sink):.3f}  "
                  f"peak M_u {np.sum(w*mu):.4f}  raining {np.mean(rain > 1e-6):.0%}")
        return
    res = {}
    for s in ("vapour_mass", "formation"):
        rain, sink, heat, _ = _call(bcfg, st, fl[sel], dt, s)
        res[s] = rain
        print(f"{s:12s} ({tag}): rain {np.sum(w*rain):.3f}  vapour sink {np.sum(w*sink):.3f}  "
              f"heat/L_v {np.sum(w*heat):.3f} kg/m2/day; rain>0 in {np.mean(rain > 1e-6):.0%} of columns")
    assert rec.calls.count("formation") == 1, rec.calls
    r = rec.last
    assert r["scale"].shape[0] == len(sel)
    binding = (r["want"] > r["cap"]) & (r["want"] > 0)
    g = constants.g
    f64 = lambda x: np.asarray(x, np.float64)       # native-dtype capacity, float64 sums
    generated = np.sum(f64(np.maximum(r["want"] - r["cap"], 0)) * f64(r["dp"]), axis=1) / g
    realized = np.sum(f64(r["sink"]) * f64(r["dp"]), axis=1) / g
    taken = np.sum(f64(np.minimum(r["want"], r["cap"])) * f64(r["dp"]), axis=1) / g
    rain_total = np.sum(f64(r["want"]) * f64(r["dp"]), axis=1) / g
    routed = realized - taken                       # excess that found slack
    discarded = generated - routed                  # rain deleted: (1-scale)*rain (sub-floor: all of it)
    # Pairing contract, evaluated on the returned (post-mask) sink.  The
    # bound is proportional to the column's rain, not to the discard: scale
    # is a recorded float32 quotient (~1 ulp of rain_total), so a fixed atol
    # below that noise fails healthy scale~1 columns (codex + GLM round 5).
    np.testing.assert_array_less(np.abs(discarded - (1.0 - f64(r["scale"])) * rain_total),
                                 5e-7 * rain_total + 1e-20)
    print(f"recorded cap scale over the selected columns: min {r['scale'].min():.6f}  "
          f"columns with scale<1: {np.sum(r['scale'] < 1.0)}  "
          f"levels with want>cap: {binding.sum()} of {np.sum(r['want'] > 0)} formation levels  "
          f"excess generated {np.sum(w*generated)*SEC_PER_DAY:.4f} / routed {np.sum(w*routed)*SEC_PER_DAY:.4f} "
          f"/ discarded {np.sum(w*discarded)*SEC_PER_DAY:.4f} kg/m2/day")
    on = res["vapour_mass"] > 1e-6
    print(f"returned-rain ratio formation/vapour_mass (post-evaporation, NOT the scale): "
          f"rain-weighted {np.sum(w[on]*res['formation'][on])/np.sum(w[on]*res['vapour_mass'][on]):.3f}")


if __name__ == "__main__":
    main()
