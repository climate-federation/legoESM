"""How far the LES boundary-layer top moves before day 5, and how strong the day-5
radiative cooling is that the SCM applies on every day, over all CloudBench
artifacts.

The SCM gets the window-weighted mean of the LES day-5 radiation frames for t < 96 h
(``bridge.rad_tendency_fn``). Per artifact this reports: the height of the sharpest
theta_l step (0.3-5 km) at frame 0 and in the day-5 mean; the strongest cooling of
the day-5 mean radiation above 300 m [K/day] and its height; the mean cooling within
300 m of the day-5 step and within 300 m of the frame-0 step. Numbers only.

usage: cloudbench_rad_shift.py OUT_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ART = Path("/glade/derecho/scratch/pg2328/les_suite/artifacts/cloudbench")


def step_height(z, x, lo=300.0, hi=5000.0):
    g = np.diff(x) / np.diff(z)
    zc = 0.5 * (z[1:] + z[:-1])
    m = (zc >= lo) & (zc <= hi)
    return float(zc[m][np.argmax(g[m])])


def rad_mean(t, r):
    t, r = t[1:], r[1:]
    edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    w = np.diff(edges)
    return (r * w[:, None]).sum(0) / w.sum()


def main() -> int:
    rows = {}
    for p in sorted(ART.glob("cloudbench_*_amip.npz")):
        d = np.load(p, allow_pickle=True)
        z, th, t = d["heights_m"], d["theta"], d["times_s"]
        rad = 86400.0 * rad_mean(t, d["rad_T_tendency"])
        z0, z5 = step_height(z, th[0]), step_height(z, th[1:].mean(0))
        up = z > 300.0
        k = np.argmin(np.where(up, rad, np.inf))

        def near(zc):
            m = np.abs(z - zc) <= 300.0
            return float(rad[m].mean())
        rows[p.name] = {"z_step_frame0_m": z0, "z_step_day5_m": z5,
                        "peak_cooling_K_day": float(rad[k]), "z_peak_cooling_m": float(z[k]),
                        "cooling_near_day5_step_K_day": near(z5),
                        "cooling_near_frame0_step_K_day": near(z0)}
    a = {k: np.array([r[k] for r in rows.values()]) for k in next(iter(rows.values()))}
    rise = a["z_step_day5_m"] - a["z_step_frame0_m"]
    pct = lambda x: {q: float(np.percentile(x, q)) for q in (10, 50, 90)}
    summary = {"n": len(rows), "step_rise_m": pct(rise),
               "n_rise_over_500m": int((rise > 500).sum()),
               "n_rise_over_1000m": int((rise > 1000).sum()),
               "peak_cooling_K_day": pct(a["peak_cooling_K_day"]),
               "n_peak_below_minus5": int((a["peak_cooling_K_day"] < -5).sum()),
               "cooling_near_day5_step_K_day": pct(a["cooling_near_day5_step_K_day"]),
               "cooling_near_frame0_step_K_day": pct(a["cooling_near_frame0_step_K_day"])}
    Path(sys.argv[1]).write_text(json.dumps({"summary": summary, "cases": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
