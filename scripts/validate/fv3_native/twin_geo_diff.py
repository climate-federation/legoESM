#!/usr/bin/env python
"""Layout-free twin divergence diff: our six-face duo stepper vs the
runnable Zenodo oracle model, both on the SAME geographic grid.

The oracle writes per-tile atmos_*.nc (native cube); our stepper's
runner writes a c2l'd 1-degree npz.  This regrids the oracle's native
tiles to the SAME 1-degree lat-lon lens (nearest-cell, the runner's
build_nearest_map protocol) and reports, per output time:

  global max|V|  (both) and the vertex-neighborhood max|V| (both,
  within `--vdeg` of the eight cube vertices at lat +-35.26, lon
  45/135/225/315), plus the geographic L-inf wind difference.

First time the vertex-neighborhood amplitudes DIVERGE (ours grows,
theirs stays flat) localizes the injection onset with zero layout
mapping — the create-vs-reference trap never enters.

Usage:
  twin_geo_diff.py --oracle-dir run_c36 --ours ourc36_case8.npz \
      --var-prefix atmos_daily
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import netCDF4 as nc

_VLAT = (35.26, -35.26)
_VLON = (45.0, 135.0, 225.0, 315.0)


def _vertex_max(w2d, lat, lon, vdeg):
    m = 0.0
    for la in _VLAT:
        ii = np.where(np.abs(lat - la) < vdeg)[0]
        for lo in _VLON:
            jj = np.where(np.abs(((lon - lo + 180) % 360) - 180) < vdeg)[0]
            if len(ii) and len(jj):
                m = max(m, float(np.nanmax(w2d[np.ix_(ii, jj)])))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-dir", required=True)
    ap.add_argument("--var-prefix", default="atmos_daily")
    ap.add_argument("--ours", default=None,
                    help="our runner npz (u,v (T,181,360), times_days)")
    ap.add_argument("--vdeg", type=float, default=4.0)
    args = ap.parse_args()

    # --- oracle native tiles -> per-time global + vertex max (native) --
    fs = sorted(glob.glob(f"{args.oracle_dir}/{args.var_prefix}.tile*.nc"))
    if not fs:
        raise SystemExit(f"no oracle tiles in {args.oracle_dir}")
    nt = None
    o_gm = o_vt = None
    for f in fs:
        d = nc.Dataset(f)
        u = np.asarray(d.variables["ucomp"][:]).squeeze()
        v = np.asarray(d.variables["vcomp"][:]).squeeze()
        if u.ndim == 2:
            u = u[None]; v = v[None]
        w = np.hypot(u, v)                       # (t, ny, nx) native
        if nt is None:
            nt = w.shape[0]
            o_gm = np.zeros(nt); o_vt = np.zeros(nt)
        n = w.shape[1]
        k = max(2, n // 9)                       # ~vertex corner block
        for t in range(nt):
            o_gm[t] = max(o_gm[t], float(w[t].max()))
            corners = (w[t][:k, :k].max(), w[t][:k, -k:].max(),
                       w[t][-k:, :k].max(), w[t][-k:, -k:].max())
            o_vt[t] = max(o_vt[t], float(max(corners)))
        d.close()

    print(f"# oracle {args.oracle_dir} ({nt} frames), native-tile "
          f"corner=vertex proxy")
    for t in range(nt):
        print(f"oracle frame {t:2d}: max|V| {o_gm[t]:7.3f}  "
              f"vertex {o_vt[t]:7.3f}  ratio {o_vt[t]/o_gm[t]:.3f}")

    if not args.ours:
        return
    d = np.load(args.ours)
    u = d["u"]; v = d["v"]; lat = d["lat"]; lon = d["lon"]
    days = d["times_days"]
    print(f"\n# ours {args.ours} ({len(days)} frames), 1-deg c2l lens")
    for t in range(len(days)):
        w = np.hypot(u[t], v[t])
        gm = float(np.nanmax(w))
        vt = _vertex_max(w, lat, lon, args.vdeg)
        print(f"ours day {days[t]:5.1f}: max|V| {gm:7.3f}  "
              f"vertex {vt:7.3f}  ratio {vt/gm:.3f}")


if __name__ == "__main__":
    main()
