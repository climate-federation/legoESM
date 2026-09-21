"""Does the formation-profile rain debit's capacity cap bind?  Call the
production Bechtold kernel ONCE, offline, on a checkpoint's tropical-ocean
columns under both ``rain_vapor_sink`` schemes and compare the rain each
returns on IDENTICAL input state.  ``vapour_mass`` never rescales, so the
ratio formation/vapour_mass is the realized cap scale (downstream
evaporation acts on both alike).  Winds, dynamics tendencies and surface
fluxes are the same dummies for both schemes (they cancel in the ratio).

Usage: rain_sink_scale_offline.py <run> [--day 135] [--sftlf-from run]
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

ROOT = os.environ.get("AMIP_ROOT", "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--day", type=int, default=135)
    ap.add_argument("--sftlf-from"); ap.add_argument("--box", default="ITCZ 10S-10N")
    a = ap.parse_args(argv)
    rundir = f"{ROOT}/{a.run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    lat, lon, area = mesh_coords(exp)
    fl = _sftlf_on_mesh(a.sftlf_from or a.run, lat, lon)
    z = np.load(f"{rundir}/checkpoint_day_{a.day:04d}.npz", allow_pickle=True)
    lo, hi = BOXES[a.box]
    sel = np.where((lat >= lo) & (lat <= hi) & (fl < 0.01))[0]
    T, q, ps = np.asarray(z["T"])[sel], np.asarray(z["trc_q_v"])[sel], np.asarray(z["p_s"])[sel]
    vg = np.asarray(z["meta_vgrid"])
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    dp = p_half[:, 1:] - p_half[:, :-1]
    ncol, nlev = T.shape
    cfg = experiment_config_from_dict(exp)
    kernel, bcfg = _resolve_convection(cfg)
    zz = jnp.zeros((ncol, nlev))
    kw = dict(u=zz, v=zz, conv_prog_profile=jnp.asarray(z["physstate_conv_prog_profile"][sel]),
              conv_stoch_state=jnp.asarray(z["physstate_conv_stoch_state"][sel]), prng_key=None,
              dt=float(exp["dycore"]["dt"]), land_frac=jnp.asarray(fl[sel]),
              shf_w_m2=jnp.full((ncol,), -20.0), lhf_w_m2=jnp.full((ncol,), -100.0),
              dT_dt_dyn=zz, dq_dt_dyn=zz,
              dT_dt_rad=jnp.asarray(z["physstate_rad_heating"][sel]))
    w = area[sel] / area[sel].sum()
    m = dp / constants.g
    res = {}
    for s in ("vapour_mass", "formation"):
        out, _, _ = kernel(jnp.asarray(T), jnp.asarray(q), jnp.asarray(p_full), jnp.asarray(p_half),
                           config=bcfg._replace(rain_vapor_sink=s), **kw)
        rain = np.sum(np.asarray(out.dq_r_conv_dt, np.float64) * m, axis=1) * SEC_PER_DAY
        sink = -np.sum(np.asarray(out.dq_v_dt, np.float64) * m, axis=1) * SEC_PER_DAY
        heat = np.sum(np.asarray(out.dT_dt, np.float64) * m, axis=1) * constants.c_pd / constants.L_v * SEC_PER_DAY
        res[s] = (rain, sink, heat)
        print(f"{s:12s} {a.box} ocean ({ncol} cols, day {a.day}): rain {np.sum(w*rain):.3f}  "
              f"vapour sink {np.sum(w*sink):.3f}  heat/L_v {np.sum(w*heat):.3f} kg/m2/day; "
              f"rain>0 in {np.mean(rain > 0):.0%} of columns")
    r0, r1 = res["vapour_mass"][0], res["formation"][0]
    on = r0 > 1e-6
    ratio = r1[on] / r0[on]
    print(f"formation/vapour_mass rain ratio over raining columns: mean {ratio.mean():.3f} "
          f"p10 {np.percentile(ratio,10):.3f} p50 {np.percentile(ratio,50):.3f} p90 {np.percentile(ratio,90):.3f}; "
          f"rain-weighted {np.sum(w[on]*r1[on])/np.sum(w[on]*r0[on]):.3f}")


if __name__ == "__main__":
    main()
