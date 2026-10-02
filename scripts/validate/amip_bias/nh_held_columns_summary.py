#!/usr/bin/env python3
"""Summarise <replay>/held_log.npz from nh_held_columns_replay.py by region.

Per region (packed land columns, counted, not area-weighted): share of columns
held per land step, share of columns ever held, longest consecutive held run (land
steps), and the share of holds that are non-converged canopy solves.  Cell
coordinates come from a mesh npz with lat/lon in degrees (same cell order as the
checkpoint); the sanity line checks that snow-covered cells sit poleward of 25
degrees or are high terrain.

Usage: nh_held_columns_summary.py <replay_dir> <mesh.npz> <checkpoint.npz>
"""
import sys

import numpy as np

REGIONS = {"45-70N": (45, 70, 0, 360), "Siberia": (50, 70, 60, 140),
           "Canada": (50, 70, 220, 300), "Alaska": (60, 70, 190, 220),
           "Australia": (-40, -10, 110, 155), "global": (-90, 90, 0, 360)}


def longest_run(col):
    best = cur = 0
    for v in col:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main(replay, mesh, ckpt):
    log = np.load(f"{replay}/held_log.npz")
    held, conv, cells = log["held"].astype(bool), log["converged"].astype(bool), log["cells"]
    m = np.load(mesh)
    lat, lon = m["lat"][cells], m["lon"][cells] % 360
    ck = np.load(ckpt)
    snow = ck["land_ml_snow_depth"][cells] > 1.0
    orog = ck["phis"][cells] / 9.81
    print(f"land steps {held.shape[0]}, packed columns {held.shape[1]}; sanity: snow>1 kg/m2 "
          f"columns equatorward of 25 deg below 1500 m: "
          f"{int((snow & (np.abs(lat) < 25) & (orog < 1500)).sum())} of {int(snow.sum())}")
    for k, (a, b, c, d) in REGIONS.items():
        r = (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)
        if not r.any():
            continue
        h = held[:, r]
        nc = (h & ~conv[:, r]).sum()
        runs = [longest_run(h[:, j]) for j in np.flatnonzero(h.any(0))]
        print(f"[{k}] columns {int(r.sum())}: held per step {h.mean():.4f} "
              f"(mean {h.sum(1).mean():.1f} cols), ever held {h.any(0).mean():.3f}, "
              f"longest run {max(runs) if runs else 0} of {held.shape[0]} steps, "
              f"non-converged share {nc / max(h.sum(), 1):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
