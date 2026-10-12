#!/usr/bin/env python
"""nino3 layer-band heat budget, ours vs NEMO, between consecutive 5-day windows.

For each band (cells whose centre lies in [z_lo, z_hi)) and each pair of
consecutive 5-day windows (k, k+1):

    S    = (Tband_{k+1} - Tband_k) / 5 d                     storage rate [K/day]
    ADV  = mean over the two windows of the box-wall advective convergence
           (equatorial_box_wall_heat_transport, upwind wall fluxes, ONE code path
           for both models)
    SFC  = surface input into the band: top band qns + f_b*qsr, deeper bands f_b*qsr,
           with qsr/qns the window means from nino3_surface_flux_windows_vs_nemo
           (NEMO: its SBC file; ours: production-bulk rebuild) and f_b the fraction
           of qsr absorbed in the band (nino3_sw_absorption_vs_nemo: ours == NEMO to
           1e-4, so ONE profile serves both; passed explicitly with --abs-frac)
    RES  = S - ADV - SFC  = everything else (vertical mixing, lateral diffusion,
           and the error of the two reconstructions).  A RESIDUAL, never an attribution.

NEMO INSTRUMENT CHECK: RUN_GATEWAY_D30_TRD (bitwise RUN_GATEWAY) also writes 5-day
band means of ttrd_zdf + ttrd_ldf; its RES must match that measured sum before
our RES can be read as "mixing".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import netCDF4 as nc
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import equatorial_box_wall_heat_transport as ebw  # noqa: E402

BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 50.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir", required=True, type=Path)
    ap.add_argument("--nemo-dir", required=True, type=Path)
    ap.add_argument("--nemo-prefix", default="ORCA1_5d_20000101_20000130_")
    ap.add_argument("--flux-json", required=True, type=Path)
    ap.add_argument("--abs-frac", required=True,
                    help="fraction of qsr absorbed in each band, comma list (from nino3_sw_absorption_vs_nemo)")
    ap.add_argument("--mesh-mask", default="/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc")
    ap.add_argument("--windows", default="0,1,2,3,4,5")
    a = ap.parse_args()
    sha = subprocess.run(["git", "-C", str(_HERE.parents[3]), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"[provenance] git {sha} | {' '.join(sys.argv)}")
    fb = [float(x) for x in a.abs_frac.split(",")]
    flux = json.loads(a.flux_json.read_text())["nino3_5S5N_210E270E"]
    rc = {"ours": 1025.0 * 3994.0, "nemo": 1026.0 * 3991.86795711963}   # each model's own rho0*cp
    nd = a.nemo_dir; px = a.nemo_prefix
    trd = nc.Dataset(nd / f"{px}trd5d_T.nc")

    # ours: snapshot_final carries day 30 -> expose it under the day-stamped name the wall probe globs
    tmp = Path(tempfile.mkdtemp())
    for f in a.snapshot_dir.glob("snapshot_day*.npz"):
        os.symlink(f, tmp / f.name)
    if (a.snapshot_dir / "snapshot_final.npz").exists():
        z = np.load(a.snapshot_dir / "snapshot_final.npz")
        os.symlink(a.snapshot_dir / "snapshot_final.npz", tmp / f"snapshot_day{int(round(float(z['time_days']))):04d}.npz")

    box = dict(lon_lo=210.0, lon_hi=270.0, lat_halfwidth=5.0, mesh_mask=a.mesh_mask)
    m = nc.Dataset(a.mesh_mask)
    A = np.asarray(m["e1t"][0] * m["e2t"][0], dtype=np.float64)
    lat = np.asarray(m["gphit"][0], dtype=np.float64); lon = np.asarray(m["glamt"][0], dtype=np.float64)
    j0, j1, i0, i1 = ebw._box_indices(lat, lon, 210.0, 270.0, 5.0)
    area_box = float(A[j0:j1 + 1, i0:i1 + 1].sum())

    wins = [int(x) for x in a.windows.split(",")]
    rows = {}
    for b, (zlo, zhi) in enumerate(BANDS):
        for k in wins:
            day = 5 * k + 5
            r = {}
            if (tmp / f"snapshot_day{day:04d}.npz").exists():
                r["ours"] = ebw.ours(SimpleNamespace(snapshot_dir=str(tmp), hours=f"{24 * day - 1}-{24 * day}",
                                                    z_lo=zlo, z_hi=zhi, **box))
            r["nemo"] = ebw.nemo(SimpleNamespace(nemo_t=str(nd / f"{px}grid_T.nc"), nemo_u=str(nd / f"{px}grid_U.nc"),
                                                 nemo_v=str(nd / f"{px}grid_V.nc"), nemo_w=str(nd / f"{px}grid_W.nc"),
                                                 nemo_trd=None, recs=f"{k}-{k}", z_lo=zlo, z_hi=zhi, **box))
            # NEMO's own zdf + ldf band mean (thickness x area weighted), same window
            t = nc.Dataset(nd / f"{px}grid_T.nc")
            lat_o = np.asarray(t["nav_lat"][:]); lon_o = np.asarray(t["nav_lon"][:])
            jo0, jo1, io0, io1 = ebw._box_indices(lat_o, lon_o, 210.0, 270.0, 5.0)
            Ao = A[0:lat_o.shape[0], 1:1 + lat_o.shape[1]][jo0:jo1 + 1, io0:io1 + 1]
            zc = np.asarray(t["deptht"][:]).ravel()
            ks = [kk for kk in range(zc.size) if zlo <= zc[kk] < zhi]
            num = 0.0; den = 0.0
            for kk in ks:
                def rd(v):  # NEMO writes land as 1e20 fill, not NaN: drop both (ebw.nemo's guard)
                    x = np.asarray(trd[v][k, kk], dtype=np.float64)[jo0:jo1 + 1, io0:io1 + 1]
                    return np.where(np.isfinite(x) & (np.abs(x) < 1e15), x, 0.0)
                e3 = rd("e3t")
                tt = rd("ttrd_zdf") + rd("ttrd_ldf")
                num += (tt * e3 * Ao).sum(); den += (e3 * Ao).sum()
            r["nemo"]["mix_trd"] = num / den * 86400.0
            for mod in ("ours", "nemo"):
                if mod not in r:
                    continue
                f = flux[str(k)][mod]
                H = r[mod]["vol"] / area_box
                q = fb[b] * f["qsr"] + (f["qns"] if b == 0 else 0.0)
                r[mod]["sfc"] = q / rc[mod] / H * 86400.0
            rows[(b, k)] = r

    print(f"\n{'band':>8s} {'pair(d)':>9s} | {'model':5s} {'S':>8s} {'ADV':>8s} {'SFC':>8s} {'RES':>8s} | NEMO zdf+ldf (meas)   [K/day]")
    for b, (zlo, zhi) in enumerate(BANDS):
        for k in wins:
            if k + 1 not in wins:
                continue
            for mod in ("nemo", "ours"):
                r0 = rows[(b, k)].get(mod); r1 = rows[(b, k + 1)].get(mod)
                if r0 is None or r1 is None:
                    continue
                S = (r1["T_band"] - r0["T_band"]) / 5.0
                ADV = 0.5 * (r0["adv_K_day"] + r1["adv_K_day"])
                SFC = 0.5 * (r0["sfc"] + r1["sfc"])
                RES = S - ADV - SFC
                meas = (f"{0.5 * (r0['mix_trd'] + r1['mix_trd']):+8.4f}" if mod == "nemo" else "")
                print(f"{zlo:3.0f}-{zhi:<4.0f} {5 * k + 2.5:4.1f}-{5 * k + 7.5:<4.1f} | {mod:5s} {S:+8.4f} {ADV:+8.4f} "
                      f"{SFC:+8.4f} {RES:+8.4f} | {meas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
