#!/usr/bin/env python
"""Box-wall volume and heat transports of the equatorial box, layer by layer, ONE code path
for legoESM (window-mean snapshots) and NEMO (hourly means): which wall carries the heat that
warms 20-40 m in the first days?

For a layer band [z_lo, z_hi] (cells whose centre lies in the band) the probe sums, over the
window, the volume inflow through the WEST, EAST, SOUTH, NORTH walls and the TOP and BOTTOM
interfaces, and the heat inflow through each (flux x UPWIND temperature).  It prints, per wall,
the volume inflow in Sv and the heat convergence in K/day of the band-mean temperature
(heat inflow / band volume), plus two controls:
  * volume closure: the six volume inflows must sum to ~0 (z-star thickness change is small);
    a residual comparable to a wall's inflow means a staggering convention is wrong;
  * NEMO only: the reconstructed advective tendency vs NEMO's own ``ttrd_totad`` band mean
    (same window) -- the upwind reconstruction of an FCT scheme is approximate, so agreement
    is expected to order and sign, not to the digit.

Conventions (read from the code, not assumed):
  legoESM snapshot: u_mean (ny, nx+1, nz) at faces, face f = WEST face of cell f; v_mean
  (ny+1, nx, nz), face g = SOUTH face of cell g; mass_flux_w_mean (ny, nx, nz+1) at interfaces,
  index k = TOP interface of cell k, m/s, positive UP; dy_u/dx_v face lengths; h_mean thickness.
  NEMO: uo at the U point = EAST face of T cell i; vo at the V point = NORTH face of T cell j;
  wocetr_eff [m3/s] at the TOP face of cell k, positive UP; hourly files are mesh[0:331, 1:361]
  (checked on wet cells); e2u/e1v from the mesh; e3t from the hourly T file.
"""
import argparse
import glob
import re
import subprocess
import sys

import netCDF4 as nc
import numpy as np


def _box_indices(lat, lon, lon_lo, lon_hi, halfwidth):
    jj, ii = np.where((np.abs(lat) <= halfwidth) & (lon % 360.0 >= lon_lo) & (lon % 360.0 <= lon_hi))
    j0, j1, i0, i1 = jj.min(), jj.max(), ii.min(), ii.max()
    if jj.size != (j1 - j0 + 1) * (i1 - i0 + 1):
        raise SystemExit("box is not a full rectangle in index space")
    return j0, j1, i0, i1


def _upwind(flux, t_from_neg, t_from_pos):
    """T carried by a face flux: positive flux carries t_from_neg (the cell on the negative side)."""
    return np.where(flux >= 0.0, t_from_neg, t_from_pos)


def band_budget(name, layers, vol_in, heat_in, band_vol, t_band, extra=None):
    print(f"\n## {name}: layers {layers}")
    tot_v = 0.0; tot_h = 0.0
    for wall in ("W", "E", "S", "N", "BOT", "TOP"):
        v = vol_in[wall]; h = heat_in[wall]
        tot_v += v; tot_h += h
        print(f"  {wall:>3s}: volume in {v / 1e6:+8.3f} Sv | heat conv {h / band_vol * 86400:+8.4f} K/day")
    print(f"  SUM: volume {tot_v / 1e6:+8.3f} Sv (closure control; |sum|/max|wall| = "
          f"{abs(tot_v) / max(abs(vol_in[w]) for w in vol_in):.3f}) | advective tendency "
          f"{tot_h / band_vol * 86400:+8.4f} K/day; minus T_band x div: "
          f"{(tot_h - t_band * tot_v) / band_vol * 86400:+8.4f} K/day")
    if extra:
        print("  " + extra)


def ours(a):
    files = sorted(glob.glob(f"{a.snapshot_dir}/snapshot_day*.npz"),
                   key=lambda f: float(re.search(r"day([0-9]+\.[0-9]+)", f).group(1)))
    h_lo, h_hi = (float(x) for x in a.hours.split("-"))
    sel = [f for f in files if h_lo < float(re.search(r"day([0-9]+\.[0-9]+)", f).group(1)) * 24.0 <= h_hi + 0.05]
    if not sel:
        raise SystemExit(f"no snapshot in hours ({h_lo}, {h_hi}]")
    z0 = np.load(sel[0])
    lat = z0["lat_T"]; lon = z0["lon_T"]
    j0, j1, i0, i1 = _box_indices(lat, lon, a.lon_lo, a.lon_hi, a.lat_halfwidth)
    rows = list(range(j0 - 3, j1 + 5)); rowv = {gg: 0.0 for gg in rows}
    zc = np.abs(np.asarray(z0["z_center_ref"]))
    ks = [k for k in range(zc.size) if a.z_lo <= zc[k] < a.z_hi]
    dy_u = z0["dy_u"]; dx_v = z0["dx_v"]; A = z0["cell_area"]
    acc_v = {w: 0.0 for w in ("W", "E", "S", "N", "BOT", "TOP")}; acc_h = dict(acc_v); vol = 0.0; tb = 0.0
    for f in sel:
        z = np.load(f)
        u = z["u_mean"]; v = z["v_mean"]; W = z["mass_flux_w_mean"]; T = z["T_mean"]; h = z["h_mean"]
        for k in ks:
            # west wall: face f=i0 (west face of cell i0); flux>0 = eastward = INTO the box
            hf = 0.5 * (h[j0:j1 + 1, i0 - 1, k] + h[j0:j1 + 1, i0, k])
            fw = u[j0:j1 + 1, i0, k] * hf * dy_u[j0:j1 + 1, i0]
            tw = _upwind(fw, T[j0:j1 + 1, i0 - 1, k], T[j0:j1 + 1, i0, k])
            acc_v["W"] += fw.sum(); acc_h["W"] += (fw * tw).sum()
            # east wall: face f=i1+1; flux>0 = eastward = OUT of the box
            hf = 0.5 * (h[j0:j1 + 1, i1, k] + h[j0:j1 + 1, i1 + 1, k])
            fe = u[j0:j1 + 1, i1 + 1, k] * hf * dy_u[j0:j1 + 1, i1 + 1]
            te = _upwind(fe, T[j0:j1 + 1, i1, k], T[j0:j1 + 1, i1 + 1, k])
            acc_v["E"] -= fe.sum(); acc_h["E"] -= (fe * te).sum()
            # south wall: face g=j0; flux>0 = northward = INTO the box
            hf = 0.5 * (h[j0 - 1, i0:i1 + 1, k] + h[j0, i0:i1 + 1, k])
            fs = v[j0, i0:i1 + 1, k] * hf * dx_v[j0, i0:i1 + 1]
            ts = _upwind(fs, T[j0 - 1, i0:i1 + 1, k], T[j0, i0:i1 + 1, k])
            acc_v["S"] += fs.sum(); acc_h["S"] += (fs * ts).sum()
            # north wall: face g=j1+1; flux>0 = northward = OUT
            hf = 0.5 * (h[j1, i0:i1 + 1, k] + h[j1 + 1, i0:i1 + 1, k])
            fn = v[j1 + 1, i0:i1 + 1, k] * hf * dx_v[j1 + 1, i0:i1 + 1]
            tn = _upwind(fn, T[j1, i0:i1 + 1, k], T[j1 + 1, i0:i1 + 1, k])
            acc_v["N"] -= fn.sum(); acc_h["N"] -= (fn * tn).sum()
            for gg in rows:
                hf = 0.5 * (h[gg - 1, i0:i1 + 1, k] + h[gg, i0:i1 + 1, k])
                rowv[gg] += (v[gg, i0:i1 + 1, k] * hf * dx_v[gg, i0:i1 + 1]).sum()
            vol += (h[j0:j1 + 1, i0:i1 + 1, k] * A[j0:j1 + 1, i0:i1 + 1]).sum()
            tb += (T[j0:j1 + 1, i0:i1 + 1, k] * h[j0:j1 + 1, i0:i1 + 1, k] * A[j0:j1 + 1, i0:i1 + 1]).sum()
        kt, kb = ks[0], ks[-1]
        # top interface of the band (index kt), w>0 = up = OUT; bottom interface (index kb+1), w>0 = up = IN
        wt = W[j0:j1 + 1, i0:i1 + 1, kt] * A[j0:j1 + 1, i0:i1 + 1]
        tt = _upwind(wt, T[j0:j1 + 1, i0:i1 + 1, kt], T[j0:j1 + 1, i0:i1 + 1, max(kt - 1, 0)])
        acc_v["TOP"] -= wt.sum(); acc_h["TOP"] -= (wt * tt).sum()
        wb = W[j0:j1 + 1, i0:i1 + 1, kb + 1] * A[j0:j1 + 1, i0:i1 + 1]
        tbt = _upwind(wb, T[j0:j1 + 1, i0:i1 + 1, min(kb + 1, T.shape[2] - 1)], T[j0:j1 + 1, i0:i1 + 1, kb])
        acc_v["BOT"] += wb.sum(); acc_h["BOT"] += (wb * tbt).sum()
    n = len(sel)
    for w in acc_v:
        acc_v[w] /= n; acc_h[w] /= n
    vol /= n; tb /= n
    print(f"  [v-rows] northward transport through each v-face row, band {a.z_lo:.0f}-{a.z_hi:.0f} m, zonal sum {a.lon_lo:.0f}-{a.lon_hi:.0f}E (Sv):")
    for gg in rows:
        print(f"    face lat {0.5 * (lat[gg - 1, i0] + lat[gg, i0]):+6.2f}  {rowv[gg] / n / 1e6:+7.3f}")
    band_budget(f"legoESM {a.snapshot_dir.rstrip('/').split('/')[-1]} hours {a.hours} ({n} windows)",
                [f"{zc[k]:.1f}" for k in ks], acc_v, acc_h, vol, tb / vol)


def nemo(a):
    t = nc.Dataset(a.nemo_t); u = nc.Dataset(a.nemo_u); v = nc.Dataset(a.nemo_v); d = nc.Dataset(a.nemo_trd)
    m = nc.Dataset(a.mesh_mask)
    lat = np.asarray(t.variables["nav_lat"][:]); lon = np.asarray(t.variables["nav_lon"][:])
    gphit = np.asarray(m.variables["gphit"][0])[0:lat.shape[0], 1:1 + lat.shape[1]]
    wet = np.asarray(m.variables["tmask"][0, 0])[0:lat.shape[0], 1:1 + lat.shape[1]] == 1
    if np.max(np.abs(gphit - lat)[wet]) > 1e-3:
        raise SystemExit("hourly file is not mesh[0:ny, 1:nx+1]; refusing")
    e2u = np.asarray(m.variables["e2u"][0])[0:lat.shape[0], 1:1 + lat.shape[1]]
    e1v = np.asarray(m.variables["e1v"][0])[0:lat.shape[0], 1:1 + lat.shape[1]]
    e1t = np.asarray(m.variables["e1t"][0])[0:lat.shape[0], 1:1 + lat.shape[1]]
    e2t = np.asarray(m.variables["e2t"][0])[0:lat.shape[0], 1:1 + lat.shape[1]]
    A = e1t * e2t
    j0, j1, i0, i1 = _box_indices(lat, lon, a.lon_lo, a.lon_hi, a.lat_halfwidth)
    rows = list(range(j0 - 3, j1 + 5)); rowv = {gg: 0.0 for gg in rows}
    zc = np.asarray(t.variables["deptht"][:]).ravel()
    ks = [k for k in range(zc.size) if a.z_lo <= zc[k] < a.z_hi]
    r_lo, r_hi = (int(x) for x in a.recs.split("-"))
    # control: the trend file and the state files must be the same run
    _tv = "votemper" if "votemper" in d.variables else "to"
    dchk = float(np.nanmax(np.abs(np.asarray(d.variables[_tv][r_hi, 5]) - np.asarray(t.variables["to"][r_hi, 5]))))
    print(f"[control] trend-file T vs state-file T at rec {r_hi}, level 5: max|diff| = {dchk:.2e} (must be ~0)")
    acc_v = {w: 0.0 for w in ("W", "E", "S", "N", "BOT", "TOP")}; acc_h = dict(acc_v); vol = 0.0; tb = 0.0; trd = 0.0
    def g(var, r, k):
        x = np.asarray(var[r, k], dtype=np.float64)
        return np.where(np.isfinite(x) & (np.abs(x) < 1e15), x, 0.0)
    for r in range(r_lo, r_hi + 1):
        for k in ks:
            T = g(t.variables["to"], r, k); e3 = g(t.variables["e3t"], r, k); uo = g(u.variables["uo"], r, k); vo = g(v.variables["vo"], r, k)
            # west wall = U point i0-1 (east face of cell i0-1)
            hf = 0.5 * (e3[j0:j1 + 1, i0 - 1] + e3[j0:j1 + 1, i0])
            fw = uo[j0:j1 + 1, i0 - 1] * hf * e2u[j0:j1 + 1, i0 - 1]
            acc_v["W"] += fw.sum(); acc_h["W"] += (fw * _upwind(fw, T[j0:j1 + 1, i0 - 1], T[j0:j1 + 1, i0])).sum()
            hf = 0.5 * (e3[j0:j1 + 1, i1] + e3[j0:j1 + 1, i1 + 1])
            fe = uo[j0:j1 + 1, i1] * hf * e2u[j0:j1 + 1, i1]
            acc_v["E"] -= fe.sum(); acc_h["E"] -= (fe * _upwind(fe, T[j0:j1 + 1, i1], T[j0:j1 + 1, i1 + 1])).sum()
            hf = 0.5 * (e3[j0 - 1, i0:i1 + 1] + e3[j0, i0:i1 + 1])
            fs = vo[j0 - 1, i0:i1 + 1] * hf * e1v[j0 - 1, i0:i1 + 1]
            acc_v["S"] += fs.sum(); acc_h["S"] += (fs * _upwind(fs, T[j0 - 1, i0:i1 + 1], T[j0, i0:i1 + 1])).sum()
            hf = 0.5 * (e3[j1, i0:i1 + 1] + e3[j1 + 1, i0:i1 + 1])
            fn = vo[j1, i0:i1 + 1] * hf * e1v[j1, i0:i1 + 1]
            acc_v["N"] -= fn.sum(); acc_h["N"] -= (fn * _upwind(fn, T[j1, i0:i1 + 1], T[j1 + 1, i0:i1 + 1])).sum()
            for gg in rows:  # same face convention as ours: face gg = south face of cell gg = NEMO V point gg-1
                hf = 0.5 * (e3[gg - 1, i0:i1 + 1] + e3[gg, i0:i1 + 1])
                rowv[gg] += (vo[gg - 1, i0:i1 + 1] * hf * e1v[gg - 1, i0:i1 + 1]).sum()
            vol += (e3[j0:j1 + 1, i0:i1 + 1] * A[j0:j1 + 1, i0:i1 + 1]).sum()
            tb += (T[j0:j1 + 1, i0:i1 + 1] * e3[j0:j1 + 1, i0:i1 + 1] * A[j0:j1 + 1, i0:i1 + 1]).sum()
            trd += (g(d.variables["ttrd_totad"], r, k)[j0:j1 + 1, i0:i1 + 1] * e3[j0:j1 + 1, i0:i1 + 1] * A[j0:j1 + 1, i0:i1 + 1]).sum()
        kt, kb = ks[0], ks[-1]
        Tt = g(t.variables["to"], r, kt); Ta = g(t.variables["to"], r, max(kt - 1, 0)); Tb = g(t.variables["to"], r, kb); Tbb = g(t.variables["to"], r, min(kb + 1, zc.size - 1))
        wt = g(d.variables["wocetr_eff"], r, kt)[j0:j1 + 1, i0:i1 + 1]
        acc_v["TOP"] -= wt.sum(); acc_h["TOP"] -= (wt * _upwind(wt, Tt[j0:j1 + 1, i0:i1 + 1], Ta[j0:j1 + 1, i0:i1 + 1])).sum()
        wb = g(d.variables["wocetr_eff"], r, kb + 1)[j0:j1 + 1, i0:i1 + 1]
        acc_v["BOT"] += wb.sum(); acc_h["BOT"] += (wb * _upwind(wb, Tbb[j0:j1 + 1, i0:i1 + 1], Tb[j0:j1 + 1, i0:i1 + 1])).sum()
    n = r_hi - r_lo + 1
    for w in acc_v:
        acc_v[w] /= n; acc_h[w] /= n
    vol /= n; tb /= n; trd /= n
    print(f"  [v-rows] northward transport through each v-face row, band {a.z_lo:.0f}-{a.z_hi:.0f} m, zonal sum {a.lon_lo:.0f}-{a.lon_hi:.0f}E (Sv):")
    for gg in rows:
        print(f"    face lat {0.5 * (lat[gg - 1, i0] + lat[gg, i0]):+6.2f}  {rowv[gg] / n / 1e6:+7.3f}")
    band_budget(f"NEMO {a.nemo_t.split('/')[-2]} recs {a.recs} ({n} hourly means)", [f"{zc[k]:.1f}" for k in ks],
                acc_v, acc_h, vol, tb / vol,
                extra=f"CONTROL NEMO's own ttrd_totad band mean: {trd / vol * 86400:+8.4f} K/day")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-dir"); ap.add_argument("--hours", default="24-36")
    ap.add_argument("--nemo-t"); ap.add_argument("--nemo-u"); ap.add_argument("--nemo-v"); ap.add_argument("--nemo-trd")
    ap.add_argument("--mesh-mask", default="/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc")
    ap.add_argument("--recs", default="24-35")
    ap.add_argument("--z-lo", type=float, default=20.0); ap.add_argument("--z-hi", type=float, default=40.0)
    ap.add_argument("--lon-lo", type=float, default=220.0); ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    a = ap.parse_args()
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"[provenance] git {sha} | {' '.join(sys.argv)}")
    if a.snapshot_dir:
        ours(a)
    if a.nemo_t:
        nemo(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
