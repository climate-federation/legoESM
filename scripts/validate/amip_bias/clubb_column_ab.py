"""Offline CLUBB column A/B on saved production columns (CPU, fp64).

Reuses ``top_press_scm_ab.load_run_columns`` (rebuilds the checkpoint's run and
refuses any CLUBB-visible config drift). Each arm calls the production entry
``clubb_turbulence_prognostic`` N times at the macmic sub-step on the columns of
one ocean box, with the surface heat and moisture fluxes held at the
checkpoint's lagged values and no other physics or advection. Arms:

  current    the code as is (host gets the post-advance closure's rcm and
             cloud fraction)
  pr2317     host liquid from the core's ``rcm_out`` (PR #2317, emulated here by
             swapping the core's diagnostics; model code untouched)
  cam21_pre  ``rcm_out`` and the PRE-advance closure's cloud fraction
             (CLUBB_CESM r8099 timing, per the #2317 review)
  omega      current + a prescribed pressure velocity (CAM passes wm from the
             host omega every call; our MPAS lane passes none)
  omega_pr2317  pr2317 + the same omega

The omega profile is CESM2 AMIP's box-mean ``wap`` for the month (a stand-in:
the checkpoint carries no omega). Reported per arm, box mean: cloud fraction
and grid-mean liquid over 950-850 hPa (time mean over the last half of the
calls and at the end), and the inversion pressure (interface of largest
-dtheta/dp below 700 hPa).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
CESM2 = "/pool/data/CMIP6/data/CMIP/NCAR/CESM2/amip/r1i1p1f1"
BOXES = {"peru_sc": (-20.0, -10.0, -90.0, -80.0),
         "namibia_sc": (-20.0, -10.0, 0.0, 10.0),
         "trades_S": (-25.0, -10.0, -180.0, 180.0)}
ARMS = ("current", "pr2317", "cam21_pre", "omega", "omega_pr2317")


def _tool():
    f = _ROOT / "scripts/validate/oracle/clubb_glue/top_press_scm_ab.py"
    spec = importlib.util.spec_from_file_location("top_press_scm_ab", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def box_columns(mesh_npz, box, max_cols):
    m = np.load(mesh_npz)
    lat, lon = np.asarray(m["lat"]), ((np.asarray(m["lon"]) + 180) % 360) - 180
    sel = np.flatnonzero((lat >= box[0]) & (lat <= box[1]) & (lon >= box[2])
                         & (lon <= box[3]) & (np.asarray(m["land"]) == 0))
    if sel.size > max_cols:
        sel = sel[np.linspace(0, sel.size - 1, max_cols).astype(int)]
    return sel


def inversion_pressure(T, p, p_min=70000.0):
    """Top-down (ncol, nlev): pressure of the layer interface with the largest
    potential-temperature lapse -dtheta/dp between p_min and the surface."""
    from legoesm import constants
    th = T * (constants.p_ref / p) ** constants.kappa
    s = -(th[:, 1:] - th[:, :-1]) / (p[:, 1:] - p[:, :-1])   # > 0 where theta falls with p
    pm = 0.5 * (p[:, 1:] + p[:, :-1])
    s = np.where(pm >= p_min, s, -np.inf)
    return pm[np.arange(len(pm)), np.argmax(s, axis=1)]


def cesm2_omega(month, box, p_target):
    import xarray as xr
    d = xr.open_dataset(glob.glob(f"{CESM2}/Amon/wap/*/*/*.nc")[0]).sel(time=month).isel(time=0)
    sf = xr.open_dataset(glob.glob(f"{CESM2}/fx/sftlf/*/*/*.nc")[0]).sftlf
    lon = ((sf.lon + 180) % 360) - 180
    sel = ((sf.lat >= box[0]) & (sf.lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
           & (sf == 0)).values
    w = np.cos(np.radians(np.broadcast_to(sf.lat.values[:, None], sel.shape)[sel]))
    x = np.moveaxis(d.wap.values, 0, -1)[sel]
    ok = np.isfinite(x)
    prof = np.nansum(x * w[:, None], 0) / np.sum(w[:, None] * ok, 0)
    plev = d.plev.values
    o = np.argsort(plev)
    return np.stack([np.interp(pc, plev[o], prof[o]) for pc in p_target])


def run_box(tool, a, box_name):
    pick = box_columns(a.mesh_npz, BOXES[box_name], a.max_cols)
    L = tool.load_run_columns(a.checkpoint, a.deck, a.mesh_level, pick=pick)
    jax, jnp, C = L.jax, L.jnp, L.C  # noqa: N806
    from legoesm import constants
    from legoesm.thermo import saturation_mixing_ratio
    d = np.load(a.checkpoint, allow_pickle=True)
    shf = jnp.asarray(np.asarray(d["sfc_lag_shflx"], float)[pick])
    lhf = jnp.asarray(np.asarray(d["sfc_lag_lhflx"], float)[pick])
    p_full = np.asarray(L.p_full)
    omega = jnp.asarray(cesm2_omega(a.month, BOXES[box_name], p_full))
    orig_pdf, orig_core = C.compute_pdf_closure, C.advance_clubb_core

    def make_step(arm):
        rec = []

        def pdf(*x, **k):
            r = orig_pdf(*x, **k)
            rec.append(r)
            return r

        def core(*x, **k):
            rec.clear()
            ns, dg = orig_core(*x, **k)
            if len(rec) != 2:
                raise RuntimeError(f"expected 2 closure calls per core call, got {len(rec)}")
            dg = dict(dg)
            if arm in ("pr2317", "cam21_pre", "omega_pr2317"):
                dg["rcm"] = dg["rcm_out"]
            if arm == "cam21_pre":
                dg["cloud_frac"] = rec[0]["cloud_frac"]
            return ns, dg

        om = omega if arm.startswith("omega") else None

        def step(u, v, T, qv, qc, mom):
            z_full, z_half = L.compute_heights_from_sigma(T, L.p_half, q_v=qv)
            rho = L.compute_rho(T, L.p_full, q_v=qv)
            # kinematic surface fluxes, positive upward, at the lowest level's density
            wpthlp = shf / (rho[:, -1] * constants.c_pd)
            wprtp = lhf / (rho[:, -1] * constants.L_v)
            o, mom_n = C.clubb_turbulence_prognostic(
                u, v, T, qv, mom, L.p_full, L.p_half, z_full, z_half, T[:, -1],
                qv[:, -1], rho, L.dt, L.ccfg, sfc_wpthlp=wpthlp, sfc_wprtp=wprtp,
                sfc_upwp=jnp.zeros_like(shf), sfc_vpwp=jnp.zeros_like(shf),
                q_c=qc, lat=L.lat, omega=om)
            return (u + L.dt * o.du_dt, v + L.dt * o.dv_dt, T + L.dt * o.dT_dt,
                    qv + L.dt * o.dq_v_dt, qc + L.dt * o.dq_c_dt, mom_n, o.cloud_fraction)

        jstep = jax.jit(step)

        def first(*s):
            C.compute_pdf_closure, C.advance_clubb_core = pdf, core
            try:
                return jstep(*s)
            finally:
                C.compute_pdf_closure, C.advance_clubb_core = orig_pdf, orig_core
        return first, jstep

    win = (p_full >= 85000.0) & (p_full <= 95000.0)
    dp = np.diff(np.asarray(L.p_half), axis=1)
    wdp = np.where(win, dp, 0.0)

    def lay(x):
        return float(np.mean((x * wdp).sum(1) / wdp.sum(1)))

    def rh_of(T, qv):
        return np.asarray(qv) / np.asarray(saturation_mixing_ratio(jnp.asarray(T), L.p_full))

    res = {"ncol": int(len(pick)), "init": {
        "cf_950_850": lay(np.asarray(d["physstate_cloud_fraction"], float)[pick]),
        "qc_950_850_gkg": lay(np.asarray(L.qc0)) * 1e3,
        "inversion_hPa": float(np.mean(inversion_pressure(np.asarray(L.T0), p_full))) / 100}}
    for arm in ARMS:
        first, jstep = make_step(arm)
        s = (L.u0, L.v0, L.T0, L.qv0, L.qc0, L.mom0)
        cfs, qcs, tops = [], [], []
        for n in range(a.steps):
            *s, cf = (first if n == 0 else jstep)(*s)
            if not all(np.all(np.isfinite(np.asarray(x))) for x in (*s, cf)):
                res[arm] = {"first_nonfinite_call": n}
                break
            if n >= a.steps // 2:
                cfs.append(lay(np.asarray(cf)))
                qcs.append(lay(np.asarray(s[4])) * 1e3)
                tops.append(float(np.mean(inversion_pressure(np.asarray(s[2]), p_full))) / 100)
        else:
            res[arm] = {"cf_950_850": float(np.mean(cfs)), "qc_950_850_gkg": float(np.mean(qcs)),
                        "inversion_hPa": float(np.mean(tops)), "cf_end": cfs[-1],
                        "inversion_end_hPa": tops[-1]}
        print(box_name, arm, json.dumps(res[arm]), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("deck")
    ap.add_argument("--mesh-level", type=int, required=True)
    ap.add_argument("--mesh-npz", required=True, help="lat/lon/land per cell, degrees")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--max-cols", type=int, default=40)
    ap.add_argument("--month", default="1979-01")
    ap.add_argument("--boxes", nargs="+", default=list(BOXES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tool = _tool()
    out = {b: run_box(tool, a, b) for b in a.boxes}
    sha = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    out["meta"] = dict(git=sha, checkpoint=str(Path(a.checkpoint).resolve()), deck=a.deck,
                       steps=a.steps, max_cols=a.max_cols, month=a.month,
                       omega_source="CESM2 AMIP r1i1p1f1 Amon wap box mean")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
