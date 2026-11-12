#!/usr/bin/env python
"""How much of the equatorial zonal variance sits at grid scale (<= 4 dx)?

WHY.  With the Smagorinsky backstop off, the day-30 nino3 MEAN bias vanishes but
the SST pattern correlation against NEMO collapses (0.85 -> 0.63).  That is
either resolved variability NEMO also has (TIW-like, > 4 dx) or a grid-scale
mode NEMO does not have.  Row-wise zonal FFT over the equatorial band, share of
variance at wavelengths <= 4 dx, plus the 2-dx zigzag share (dino_1226
two_dx_share), for ours (snapshot npz) and NEMO (5-day grid_T/grid_U file).

Self-check (--selftest): a pure checkerboard scores ~1 in both metrics; a
smooth 20-dx cosine scores ~0.  Run it before quoting a number.

Conventions: NEMO hourly/5-day fields are mesh[0:331, 1:361]; ours (332, 362).
Our u is at west faces (ny, nx+1, nz); averaged to T points.  NEMO uo is at
east faces (U point i = east face of cell i); averaged to T points the same way.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "baro_fixed_bias_wall_map", os.path.join(_HERE, "dino_1226", "baro_fixed_bias_wall_map.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
two_dx_share = _mod.two_dx_share


def high_band_share(row: np.ndarray, max_wavelength_dx: float = 4.0) -> float:
    """Share of a row's zonal variance at wavelengths <= max_wavelength_dx grid cells."""
    v = row - row.mean()
    if v.size < 8 or not np.any(v):
        return float("nan")
    p = np.abs(np.fft.rfft(v)) ** 2
    k = np.arange(p.size)                     # cycles per row
    wl = np.where(k > 0, v.size / np.maximum(k, 1), np.inf)
    return float(p[wl <= max_wavelength_dx].sum() / p[1:].sum())


def band_report(name: str, field: np.ndarray, wet: np.ndarray, rows: list[int]) -> None:
    hb, tdx, std = [], [], []
    for j in rows:
        m = wet[j]
        if m.sum() < 16:
            continue
        v = field[j][m]
        hb.append(high_band_share(v)); std.append(float(np.std(v)))
        s = two_dx_share(field, wet, j)
        if s is not None:
            tdx.append(s)
    print(f"  {name:36s} rows={len(hb):2d}  zonal std={np.mean(std):8.4f}  share(<=4dx)={np.mean(hb):6.3f}  share(2dx)={np.mean(tdx):6.3f}")


def ours(path: str, k: int, lon_lo: float, lon_hi: float, halfwidth: float):
    z = np.load(path)
    lat = z["lat_T"]; lon = z["lon_T"] % 360.0; wet = np.asarray(z["land_mask"]) > 0.5
    T = z["T"]; u = z["u"]
    uT = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    sel = (lon >= lon_lo) & (lon <= lon_hi)
    wetw = wet & sel
    rows = [j for j in range(lat.shape[0]) if np.any(np.abs(lat[j]) <= halfwidth) and wetw[j].sum() >= 16]
    print(f"[ours] {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}  level {k} ({float(z['z_center_ref'][k]):.0f} m)  rows {rows[0]}..{rows[-1]}")
    band_report("SST", T[:, :, 0], wetw, rows)
    band_report(f"T level {k}", T[:, :, k], wetw, rows)
    band_report(f"u level {k}", uT[:, :, k], wetw, rows)
    band_report("u surface", uT[:, :, 0], wetw, rows)


def nemo(t_path: str, u_path: str, rec: int, k: int, lon_lo: float, lon_hi: float, halfwidth: float):
    import netCDF4
    t = netCDF4.Dataset(t_path); u = netCDF4.Dataset(u_path)
    lat = np.asarray(t.variables["nav_lat"][:]); lon = np.asarray(t.variables["nav_lon"][:]) % 360.0
    def g(var, *idx):
        x = np.asarray(var[idx], dtype=np.float64)
        return np.where(np.isfinite(x) & (np.abs(x) < 1e15), x, np.nan)
    tos = g(t.variables["tos"], rec); Tk = g(t.variables["to"], rec, k)
    uo = g(u.variables["uo"], rec, k); uo0 = g(u.variables["uo"], rec, 0)
    def to_T(f):  # U point i = east face of cell i -> T point i = 0.5*(u[i-1] + u[i])
        out = np.full_like(f, np.nan); out[:, 1:] = 0.5 * (f[:, :-1] + f[:, 1:]); return out
    wet = np.isfinite(Tk) & np.isfinite(tos)
    sel = (lon >= lon_lo) & (lon <= lon_hi); wetw = wet & sel
    # nav_lat is 0 on NEMO land cells: test latitude on WET, in-window cells only
    rows = [j for j in range(lat.shape[0]) if wetw[j].sum() >= 16 and np.any(np.abs(lat[j][wetw[j]]) <= halfwidth)]
    print(f"[NEMO] {t_path.split('/')[-2]} rec {rec} level {k}  rows {rows[0]}..{rows[-1]}")
    band_report("SST", np.nan_to_num(tos), wetw, rows)
    band_report(f"T level {k}", np.nan_to_num(Tk), wetw, rows)
    uT = to_T(uo); uT0 = to_T(uo0)
    band_report(f"u level {k}", np.nan_to_num(uT), wetw & np.isfinite(uT), rows)
    band_report("u surface", np.nan_to_num(uT0), wetw & np.isfinite(uT0), rows)


def selftest() -> None:
    ny, nx = 5, 120
    wet = np.ones((ny, nx), bool)
    checker = np.tile(((-1.0) ** np.arange(nx))[None, :], (ny, 1))
    smooth = np.tile(np.cos(2 * np.pi * np.arange(nx) / 20.0)[None, :], (ny, 1))
    hb_c = np.mean([high_band_share(checker[j]) for j in range(ny)]); hb_s = np.mean([high_band_share(smooth[j]) for j in range(ny)])
    td_c = np.mean([two_dx_share(checker, wet, j) for j in range(ny)]); td_s = np.mean([two_dx_share(smooth, wet, j) for j in range(ny)])
    print(f"[selftest] checkerboard: share(<=4dx)={hb_c:.3f} share(2dx)={td_c:.3f}; 20-dx cosine: {hb_s:.3f} {td_s:.3f}")
    assert hb_c > 0.99 and td_c > 0.99 and hb_s < 0.01 and td_s < 0.01, "metric self-check FAILED"
    print("[selftest] OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", nargs="*", default=[])
    ap.add_argument("--nemo-t"); ap.add_argument("--nemo-u"); ap.add_argument("--rec", type=int, default=5)
    ap.add_argument("--level", type=int, default=25)
    ap.add_argument("--lon-lo", type=float, default=140.0); ap.add_argument("--lon-hi", type=float, default=280.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest and not a.snapshots and not a.nemo_t:
        return 0
    for s in a.snapshots:
        ours(s, a.level, a.lon_lo, a.lon_hi, a.lat_halfwidth)
    if a.nemo_t:
        nemo(a.nemo_t, a.nemo_u, a.rec, a.level, a.lon_lo, a.lon_hi, a.lat_halfwidth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
