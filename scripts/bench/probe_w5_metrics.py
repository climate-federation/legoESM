#!/usr/bin/env python
"""Quantify Williamson-5 downstream wave propagation from regridded snapshots.

Reads ``snapshots_latlon.npz`` (181x360) from a matrix-runner output tree and
reports, at the final snapshot day, metrics that distinguish a *propagating*
mountain Rossby-wave train (good: latlon / MPAS) from an *over-damped*,
amplitude-trapped solution (cube symptom):

  eddy_rms        RMS of the zonal-asymmetry h' = h - zonalmean(h) over the
                  whole sphere (area-weighted by cos lat).
  eddy_rms_down   same, restricted to the *downstream* sector (east of the
                  mountain at 90 E .. 270 E i.e. lon in [-90, +90] here after
                  the W5 mountain is centred at lon = -90 / 270E) and the
                  20 S .. 60 N band where the wave train lives.
  east_reach_deg  furthest eastward longitude (deg from the mountain) at which
                  |h'| zonal profile exceeds 20% of its global-max — a proxy
                  for how far the train has propagated.
  speed_max       max wind speed (m/s).

Usage:
  JAX_PLATFORMS=cpu .venv/bin/python scripts/probe_w5_metrics.py \
      LABEL1=path/to/dir1 LABEL2=path/to/dir2 ...
where each dir is the case dir holding snapshots_latlon.npz, OR a parent that
contains shallow_water/williamson5/<grid>/<res>/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

MOUNTAIN_LON = -90.0  # W5 mountain centre (270 E) on the [-180,180] canvas


def _find_npz(p: Path) -> Path | None:
    if (p / "snapshots_latlon.npz").exists():
        return p / "snapshots_latlon.npz"
    hits = list(p.glob("**/williamson5/**/snapshots_latlon.npz"))
    return hits[0] if hits else None


def metrics(npz: Path) -> dict:
    d = np.load(npz)
    h = d["height"][-1].astype(float)     # final day (nlat,nlon)
    nlat, nlon = h.shape
    # Derive coords from the field's own shape: the stored ``lat``/``lon``
    # arrays are the (181,360) target canvas even when a grid (latlon)
    # writes its fields at native resolution, so trust the data shape.
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False) if nlon != 360 \
        else d["lon"].astype(float)
    lat = (np.linspace(-90.0, 90.0, nlat) if nlat != 181
           else d["lat"].astype(float))
    coslat = np.cos(np.deg2rad(lat))[:, None]
    hbar = (h * 0 + (h.mean(axis=1, keepdims=True)))  # zonal mean per lat
    hp = h - hbar                                     # eddy / asymmetry
    w = np.broadcast_to(coslat, hp.shape)
    eddy_rms = float(np.sqrt((w * hp**2).sum() / w.sum()))
    # downstream sector: east of mountain, wave-train band
    dlon = ((lon - MOUNTAIN_LON + 180.0) % 360.0) - 180.0   # signed deg E of mtn
    east = (dlon > 10.0) & (dlon < 170.0)
    band = (lat > -20.0) & (lat < 60.0)
    sel = band[:, None] & east[None, :]
    wsel = w[sel]
    eddy_rms_down = float(np.sqrt((wsel * hp[sel]**2).sum() / wsel.sum()))
    # east reach: zonal profile of |h'| max over band, vs global max
    prof = np.abs(hp[band, :]).max(axis=0)            # (360,)
    gmax = prof.max()
    thr = 0.20 * gmax
    eastmask = (dlon > 0) & (prof > thr)
    east_reach = float(dlon[eastmask].max()) if eastmask.any() else 0.0
    spd = None
    if "wind_speed" in d:
        spd = float(np.nanmax(d["wind_speed"][-1]))
    elif {"u", "v"} <= set(d.files):
        spd = float(np.nanmax(np.sqrt(d["u"][-1]**2 + d["v"][-1]**2)))
    day = float(d["times_days"][-1]) if "times_days" in d else float("nan")
    return dict(day=day, eddy_rms=eddy_rms, eddy_rms_down=eddy_rms_down,
                east_reach_deg=east_reach, speed_max=spd)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    rows = []
    for a in argv:
        label, _, path = a.partition("=")
        if not path:
            label, path = path, label  # bare path
        npz = _find_npz(Path(path))
        if npz is None:
            print(f"  {label or path}: NO snapshots_latlon.npz under {path}")
            continue
        m = metrics(npz)
        rows.append((label or path, m))
    if not rows:
        return 2
    print(f"{'label':22s} {'day':>5s} {'eddy_rms':>9s} "
          f"{'eddy_down':>9s} {'reach_E':>8s} {'spd_max':>8s}")
    for label, m in rows:
        print(f"{label:22s} {m['day']:5.1f} {m['eddy_rms']:9.2f} "
              f"{m['eddy_rms_down']:9.3f} {m['east_reach_deg']:8.1f} "
              f"{(m['speed_max'] or float('nan')):8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
