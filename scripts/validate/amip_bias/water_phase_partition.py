#!/usr/bin/env python3
"""Where the model's water sits: by species per band, and by phase globally.

Two questions the published scorecard cannot answer.

First, the published ``clivi`` is CLOUD ICE ONLY -- snow and graupel are
radiatively inert in this model and the diagnostic deliberately excludes them
-- and the reference it is scored against, ERA5 total column cloud ice water
(GRIB parameter 79), excludes falling snow too.  So a low ``clivi`` can mean
either little frozen water or frozen water parked in the precipitating
species, and only the species breakdown separates those.

Second, a band water budget's residual is uninterpretable until the global
store is reconciled against the model's OWN budget tracker, which works in
vapour and on the native mesh while the published fluxes are monthly means
interpolated to a regular grid.

Usage: water_phase_partition.py <run> [<run> ...]
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from cloud_layers import mesh_coords  # noqa: E402

from legoesm import constants  # noqa: E402

BANDS = {"SO stormtrack": (-60.0, -30.0), "NH midlat": (30.0, 60.0),
         "ITCZ": (-10.0, 10.0)}
SPECIES = ("trc_q_c", "trc_q_i", "trc_q_s", "trc_q_g", "trc_q_r")
FROZEN = ("trc_q_i", "trc_q_s", "trc_q_g")


def column_paths(ck):
    """Per-column mass path [kg/m2] of every water species in a checkpoint."""
    z = np.load(ck, allow_pickle=True)
    vg = np.asarray(z["meta_vgrid"])
    ps = np.asarray(z["p_s"], dtype=np.float64)
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    dp = p_half[:, 1:] - p_half[:, :-1]
    out = {s: (np.asarray(z[s], dtype=np.float64) * dp).sum(1) / constants.g
           for s in SPECIES + ("trc_q_v",)}
    if not all(np.all(np.isfinite(v)) for v in out.values()):
        raise SystemExit(f"FATAL: non-finite water path in {ck}")
    return float(np.asarray(z["day"])), out


def _report(run):
    rd = f"{rb.ROOT}/{run}"
    lat, _lon, area = mesh_coords(json.load(open(f"{rd}/experiment_config.json")))
    cks = sorted(glob.glob(f"{rd}/checkpoint_day_*.npz"))
    if not cks:
        raise SystemExit(f"FATAL: {run} has no checkpoints")
    series = [column_paths(c) for c in cks]

    day, p = series[-1]
    print(f"\n=== {run}: column water by species, day {day:.0f} [g/m2] ===")
    print(f"{'band':<16}" + "".join(f"{s[4:]:>9}" for s in SPECIES)
          + f"{'frozen':>9}")
    for name, (lo, hi) in BANDS.items():
        w = area * ((lat >= lo) & (lat <= hi))
        v = [float((p[s] * w).sum() / w.sum()) * 1e3 for s in SPECIES]
        fr = sum(float((p[s] * w).sum() / w.sum()) * 1e3 for s in FROZEN)
        print(f"{name:<16}" + "".join(f"{x:9.3f}" for x in v) + f"{fr:9.3f}")
    print("Only the cloud-ice column is published as clivi and only it is seen "
          "by radiation; snow and graupel are radiatively inert here, so they "
          "belong in neither the clivi comparison nor the cloud feedback.")

    print(f"\n=== {run}: global store by phase [kg/m2, native mesh area] ===")
    print(f"{'day':>6}{'vapour':>10}{'condensate':>12}{'total':>10}{'d(tot)/dt':>11}")
    prev = None
    for d, pp in series:
        v = float((pp["trc_q_v"] * area).sum() / area.sum())
        c = sum(float((pp[s] * area).sum() / area.sum()) for s in SPECIES)
        rate = "" if prev is None else f"{(v + c - prev) / (d - pd_):+11.4f}"
        print(f"{d:6.0f}{v:10.3f}{c:12.4f}{v + c:10.3f}{rate:>11}")
        prev, pd_ = v + c, d

    ts = np.load(f"{rd}/timeseries.npz", allow_pickle=True)
    if "moisture_residual" in ts.files:
        r = np.asarray(ts["moisture_residual"])
        r = r[np.isfinite(r)]
        if r.size:
            print(f"model's own moisture-budget residual E - P - dW/dt: "
                  f"{r.mean():+.4f} mm/day (last {r.size} days).  Its own "
                  f"docstring calls anything above 0.01 a non-closure.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for r in sys.argv[1:]:
        _report(r)
