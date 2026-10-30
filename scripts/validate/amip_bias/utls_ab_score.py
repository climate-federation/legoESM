#!/usr/bin/env python3
"""Pre-registered score of the vertical-filter A/B (control vs flux-form q_v filter).

From each arm's OWN checkpoints (no published output), at the requested days:

  water  : 45-70N, mass-weighted mean q_v over every layer whose full-level
           pressure lies in [70, 150] hPa, in ppmv (molar).
  temp   : 45-70N, mass-weighted mean T over layers in [100, 200] hPa; the
           score is (flux arm) - (control) in K.

Area weights and latitudes come from the res-6 Voronoi mesh of the code on
PYTHONPATH (cell order checked against the run in the caller's environment);
layer pressure is the hybrid A*p0 + B*p_s from each checkpoint's own
meta_vgrid.  Pre-registered (2026-10-07, user-approved), window days 20-30:
water confirm >= 1.5 ppmv, refute <= 0.8 (control ~0.5); temp confirm <= -1.5 K,
refute |dT| <= 0.5 K.  Prints the numbers only.

Usage: utls_ab_score.py CTL_DIR FLUX_DIR DAY [DAY ...]
"""
from __future__ import annotations

import sys

import numpy as np

P0 = 1.0e5                      # meta_vgrid A is stored normalised by this [Pa]
PPMV = 28.9644 / 18.01528 * 1e6  # const-ok: molar-mass ratio, ppmv printout only


def layer_pressure(meta_vgrid, p_s):
    """(full-level pressure, layer mass) [Pa] from a checkpoint's hybrid A/B."""
    ph = meta_vgrid[0][None, :] * P0 + meta_vgrid[1][None, :] * p_s[:, None]
    return 0.5 * (ph[:, 1:] + ph[:, :-1]), np.diff(ph, axis=1)


def band_layer_mean(field, p_full, dp, lat, area, lat_band, p_band_hpa):
    """Mass-weighted mean of ``field`` over cells in ``lat_band`` and layers
    whose full-level pressure is inside ``p_band_hpa`` (inclusive)."""
    cell = (lat >= lat_band[0]) & (lat <= lat_band[1])
    lay = (p_full >= p_band_hpa[0] * 100.0) & (p_full <= p_band_hpa[1] * 100.0)
    w = np.where(cell[:, None] & lay, dp * area[:, None], 0.0)
    if not (w > 0).any():
        raise SystemExit("FATAL: empty band/layer selection")
    if not np.isfinite(field[w > 0]).all():
        raise SystemExit("FATAL: non-finite values in the scored region")
    sel = w > 0                      # NaN outside the region must not leak in as NaN*0
    return float((field[sel] * w[sel]).sum() / w[sel].sum())


def score_day(run_dir, day, lat, area):
    d = np.load(f"{run_dir}/checkpoint_day_{day:04d}.npz")
    pf, dp = layer_pressure(d["meta_vgrid"], d["p_s"])
    return (band_layer_mean(d["trc_q_v"], pf, dp, lat, area, (45, 70), (70, 150)) * PPMV,
            band_layer_mean(d["T"], pf, dp, lat, area, (45, 70), (100, 200)))


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    ctl, flux, days = argv[0], argv[1], [int(x) for x in argv[2:]]
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(6)
    lat = np.rad2deg(np.asarray(mesh.latCell, dtype=np.float64))
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    rows = []
    print("day   q_v 70-150 hPa [ppmv] ctl | flux    T 100-200 hPa [K] ctl | flux | flux-ctl")
    for day in days:
        qc, tc = score_day(ctl, day, lat, area)
        qf, tf = score_day(flux, day, lat, area)
        rows.append((qc, qf, tc, tf))
        print(f"{day:4d}   {qc:8.3f} | {qf:8.3f}          {tc:7.2f} | {tf:7.2f} | {tf - tc:+6.2f}")
    a = np.mean(rows, axis=0)
    print(f"mean   {a[0]:8.3f} | {a[1]:8.3f}          {a[2]:7.2f} | {a[3]:7.2f} | {a[3] - a[2]:+6.2f}"
          f"   (days {days[0]}-{days[-1]}, {len(days)} checkpoints)")


if __name__ == "__main__":
    main(sys.argv[1:])
