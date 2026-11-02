"""Box-mean profiles (ocean only) of T, theta, RH, cloud fraction, liquid: legoESM vs CESM2.

legoESM from checkpoint snapshots (mean over given files), CESM2 from one
month of CMIP6 AMIP plev19 ta/hur and model-level cls/clw interpolated in
log-p to the same pressure levels. Also lower-tropospheric stability
LTS = theta(700) - theta(1000). RH: legoESM via legoesm.thermo (liquid).
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

CESM2 = "/pool/data/CMIP6/data/CMIP/NCAR/CESM2/amip/r1i1p1f1"
PLEVS = np.array([1000, 975, 950, 925, 900, 875, 850, 800, 750, 700, 600, 500.0]) * 100.0


def interp_logp(x, p, plevs):
    """x, p: (ncol, nlev) any vertical order -> (ncol, nplev)."""
    o = np.argsort(p, axis=1)
    p, x = np.take_along_axis(p, o, 1), np.take_along_axis(x, o, 1)
    lp = np.log(p)
    return np.stack([np.array([np.interp(np.log(q), lp[i], x[i], left=np.nan, right=np.nan)
                               for i in range(p.shape[0])]) for q in plevs], 1)


def theta(T, p):
    return T * (constants.p_ref / p) ** constants.kappa


def ours(ckpts, mesh, box):
    m = np.load(mesh)
    lat, lon = np.asarray(m["lat"]), ((np.asarray(m["lon"]) + 180) % 360) - 180   # mesh file is in degrees
    sel = ((lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
           & (m["land"] == 0))
    w = m["area"][sel]
    acc = {}
    for f in ckpts:
        z = np.load(f, allow_pickle=True)
        a, b = z["meta_vgrid"]
        ph = a[None] * 1e5 + b[None] * z["p_s"][sel, None]
        pm = 0.5 * (ph[:, 1:] + ph[:, :-1])
        T, qv = z["T"][sel], z["trc_q_v"][sel]
        rh = qv / np.asarray(saturation_mixing_ratio(T, pm))
        for k, x in (("T", T), ("RH", rh), ("cf", z["physstate_cloud_fraction"][sel]),
                     ("q_c_gkg", z["trc_q_c"][sel] * 1e3), ("q_v_gkg", qv * 1e3)):
            acc.setdefault(k, []).append(interp_logp(x, pm, PLEVS))
    return {k: np.nansum(np.mean(v, 0) * w[:, None], 0) / w.sum() for k, v in acc.items()}


def cesm2(month, box):
    def ds(t, v):
        return xr.open_dataset(glob.glob(f"{CESM2}/{t}/{v}/*/*/*.nc")[0]).sel(time=month).isel(time=0)
    sf = xr.open_dataset(glob.glob(f"{CESM2}/fx/sftlf/*/*/*.nc")[0]).sftlf
    lon = ((sf.lon + 180) % 360) - 180
    sel = ((sf.lat >= box[0]) & (sf.lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
           & (sf == 0)).values
    lat2 = np.broadcast_to(sf.lat.values[:, None], sel.shape)
    w = np.cos(np.radians(lat2[sel]))
    out = {}
    for v, name, scale in (("ta", "T", 1.0), ("hur", "RH", 0.01), ("hus", "q_v_gkg", 1e3)):
        d = ds("Amon", v)[v].interp(plev=PLEVS)
        x = np.moveaxis(d.values, 0, -1)[sel] * scale
        out[name] = np.nansum(x * w[:, None], 0) / np.sum(w[:, None] * np.isfinite(x), 0)
    for t, v, name, scale in (("CFmon", "cls", "cf", 0.01), ("Amon", "clw", "q_c_gkg", 1e3)):
        d = ds(t, v)
        pm = (d.a * d.p0 + d.b * d.ps).transpose("lat", "lon", "lev").values[sel]
        x = np.moveaxis(d[v].values, 0, -1)[sel] * scale
        x = interp_logp(np.nan_to_num(x), pm, PLEVS)
        out[name] = np.nansum(x * w[:, None], 0) / np.sum(w[:, None] * np.isfinite(x), 0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--box", nargs=4, type=float, required=True, help="lat0 lat1 lon0 lon1")
    ap.add_argument("--month", default="1979-01")
    a = ap.parse_args(argv)
    o, c = ours(a.ckpt, a.mesh, a.box), cesm2(a.month, a.box)
    print(f"box {a.box}  month {a.month}  ckpts {len(a.ckpt)}")
    print(f"{'hPa':>5s} | {'T':>6s} {'T2':>6s} | {'RH':>5s} {'RH2':>5s} | {'qv':>5s} {'qv2':>5s}"
          f" | {'cf':>5s} {'cls2':>5s} | {'qc':>6s} {'clw2':>6s}   (2 = CESM2)")
    for i, p in enumerate(PLEVS):
        print(f"{p/100:5.0f} | {o['T'][i]:6.1f} {c['T'][i]:6.1f} | {o['RH'][i]:5.2f} {c['RH'][i]:5.2f}"
              f" | {o['q_v_gkg'][i]:5.2f} {c['q_v_gkg'][i]:5.2f} | {o['cf'][i]:5.3f} {c['cf'][i]:5.3f}"
              f" | {o['q_c_gkg'][i]:6.4f} {c['q_c_gkg'][i]:6.4f}")
    i7, i10 = list(PLEVS).index(70000.0), 0
    for lab, d in (("ours", o), ("CESM2", c)):
        lts = theta(d["T"][i7], 70000.0) - theta(d["T"][i10], 100000.0)
        print(f"LTS {lab}: {lts:.2f} K")


if __name__ == "__main__":
    main()
