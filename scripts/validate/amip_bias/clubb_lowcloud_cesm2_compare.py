"""Low-cloud fraction and liquid, legoESM vs CESM2 AMIP, 950-700 hPa, ocean boxes.

legoESM: the CLUBB PDF cloud fraction carried in the checkpoints
(``physstate_cloud_fraction``, CAM6's ``alst``) and grid-mean ``trc_q_c``,
averaged over the given checkpoints (snapshots). CESM2: CMIP6 ``cl`` (CAM
CLOUD = min(max(alst, aist) + deepcu, 1), clubb_intr.F90:2575-2586),
CFmon ``cls`` (stratiform, AST) and ``clc`` (convective, CONCLD), and
Amon ``clw``, for one month. Layer mean is pressure-thickness weighted over
model levels whose mid pressure lies in [p_lo, p_hi]; box mean is area
weighted, ocean only (legoESM land == 0, CESM2 sftlf == 0).
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import xarray as xr

BOXES = {
    "trades_N": (10.0, 25.0, -180.0, 180.0),
    "trades_S": (-25.0, -10.0, -180.0, 180.0),
    "southern_ocean": (-65.0, -40.0, -180.0, 180.0),
    "peru_sc": (-20.0, -10.0, -90.0, -80.0),
    "namibia_sc": (-20.0, -10.0, 0.0, 10.0),
}
CESM2 = "/pool/data/CMIP6/data/CMIP/NCAR/CESM2/amip/r1i1p1f1"


def layer_mean(x, p_mid, dp, p_lo, p_hi):
    w = np.where((p_mid >= p_lo) & (p_mid <= p_hi), dp, 0.0)
    return (x * w).sum(-1) / w.sum(-1)


def box_mask(lat, lon, box):
    la0, la1, lo0, lo1 = box
    lon = ((lon + 180.0) % 360.0) - 180.0
    return (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)


def ours(ckpts, mesh, p_lo, p_hi):
    m = np.load(mesh)
    lat, lon, area, land = (np.asarray(m[k], float) for k in ("lat", "lon", "area", "land"))
    if np.abs(lat).max() < 3.2:
        lat, lon = np.degrees(lat), np.degrees(lon)
    cf, qc = [], []
    for f in ckpts:
        z = np.load(f, allow_pickle=True)
        a, b = z["meta_vgrid"]
        ph = a[None, :] * 1e5 + b[None, :] * z["p_s"][:, None]
        pm, dp = 0.5 * (ph[:, 1:] + ph[:, :-1]), np.diff(ph, axis=1)
        cf.append(layer_mean(z["physstate_cloud_fraction"], pm, dp, p_lo, p_hi))
        qc.append(layer_mean(z["trc_q_c"], pm, dp, p_lo, p_hi))
    cf, qc = np.mean(cf, 0), np.mean(qc, 0)
    out = {}
    for name, box in BOXES.items():
        sel = box_mask(lat, lon, box) & (land == 0)
        w = area[sel]
        out[name] = {"cf_clubb": float((cf[sel] * w).sum() / w.sum()),
                     "q_c_gkg": float((qc[sel] * w).sum() / w.sum() * 1e3),
                     "ncol": int(sel.sum())}
    return out


def cesm2(month, p_lo, p_hi):
    def one(table, var):
        f = glob.glob(f"{CESM2}/{table}/{var}/*/*/*.nc")[0]
        d = xr.open_dataset(f).sel(time=month).isel(time=0)
        return d
    d = one("Amon", "cl")
    ph = (d.a_bnds.values[None, None, :, :] * float(d.p0) + d.b_bnds.values[None, None, :, :]
          * d.ps.values[:, :, None, None])                      # lat, lon, lev, 2
    pm = (d.a.values[None, None, :] * float(d.p0) + d.b.values[None, None, :] * d.ps.values[:, :, None])
    dp = np.abs(ph[..., 0] - ph[..., 1])
    sf = xr.open_dataset(glob.glob(f"{CESM2}/fx/sftlf/*/*/*.nc")[0]).sftlf.values
    lat2, lon2 = np.meshgrid(d.lat.values, d.lon.values, indexing="ij")
    wcos = np.cos(np.radians(lat2))
    fields = {"cl": d.cl.values}
    for table, var in (("CFmon", "cls"), ("CFmon", "clc"), ("Amon", "clw")):
        fields[var] = one(table, var)[var].values
    out = {}
    for name, box in BOXES.items():
        sel = box_mask(lat2, lon2, box) & (sf == 0)
        w = wcos[sel]
        r = {}
        for var, x in fields.items():
            x = np.moveaxis(x, 0, -1)                           # lev last
            v = layer_mean(np.nan_to_num(x), pm, dp, p_lo, p_hi)[sel]
            scale = 1e3 if var == "clw" else 0.01                # % -> fraction, kg/kg -> g/kg
            r[var] = float((v * w).sum() / w.sum() * scale)
        r["ncol"] = int(sel.sum())
        out[name] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--month", default="1979-01")
    ap.add_argument("--p-lo", type=float, default=70000.0)
    ap.add_argument("--p-hi", type=float, default=95000.0)
    a = ap.parse_args(argv)
    res = {"args": vars(a), "ours": ours(a.ckpt, a.mesh, a.p_lo, a.p_hi),
           "cesm2": cesm2(a.month, a.p_lo, a.p_hi)}
    print(json.dumps(res, indent=1))
    print(f"{'box':16s} {'ours_clubb':>10s} {'cesm2_cl':>9s} {'cesm2_cls':>9s} {'cesm2_clc':>9s}"
          f" {'ours_qc':>8s} {'cesm2_clw':>9s}  (fraction; g/kg)")
    for b in BOXES:
        o, c = res["ours"][b], res["cesm2"][b]
        print(f"{b:16s} {o['cf_clubb']:10.3f} {c['cl']:9.3f} {c['cls']:9.3f} {c['clc']:9.3f}"
              f" {o['q_c_gkg']:8.4f} {c['clw']:9.4f}")


if __name__ == "__main__":
    main()
