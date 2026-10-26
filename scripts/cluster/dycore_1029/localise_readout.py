"""#1029 readout for localise_at_interval_1.sbatch (loc4 job dir).

Per seed: matrix interval-1 trip location (results.txt notes), its distance
from the DCMIP 2-0-0 centre (0N, 270E) and p/p_s from the probe's last clean
state; then the probe's per-step series over the last steps before the trip.
Prints numbers only; the reading rule lives in the sbatch header.
Usage: python localise_readout.py <loc4_job_dir>
"""
from __future__ import annotations

import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.atmosphere.idealized.topography import great_circle_distance
from legoesm.grids.latlon import create_latlon_grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from burst_anatomy_probe import runner  # noqa: E402  (matrix runner module)


def notes(d: Path) -> dict:
    txt = open(glob.glob(str(d / "**/results.txt"), recursive=True)[0]).read()
    line = next(l for l in txt.splitlines() if l.startswith("notes:"))
    kv = dict(re.findall(r"(\w+)=([-+0-9.eE]+|nan|None)", line))
    kv["_line"] = line
    return kv


def main(job: Path) -> None:
    sigma = runner._create_vertical(40, "hybrid")
    for s in (65537, 42):
        n = notes(job / f"s{s}")
        print(f"seed {s}: {n['_line'][:400]}")
        lat, lon, k = (float(n.get(x, "nan")) for x in
                       ("max_wind_lat_deg", "max_wind_lon_deg", "max_wind_k"))
        r = float(great_circle_distance(math.radians(lon), math.radians(lat),
                                        3 * math.pi / 2, 0.0, 1.0))
        print(f"  argmax lat {lat} lon {lon} k {k}; r from (0N,270E) = "
              f"{math.degrees(r):.1f} deg (R_m = 135)")
        pd = job / f"probe_s{s}"
        meta = json.loads((pd / "meta.json").read_text())
        clean = sorted(pd.glob("clean_*.npz"), key=lambda p: int(p.stem[6:]))
        if clean and math.isfinite(lat):
            z = np.load(clean[-1])
            g = create_latlon_grid(72, 144)
            i = int(np.argmin(abs(np.degrees(np.asarray(g.lat)) - lat)))
            j = int(np.argmin(abs(np.degrees(np.asarray(g.lon)) - lon)))
            ps = float(z["p_s"][i, j])
            p = np.asarray(sigma.pressure_at_full(np.asarray([ps])))[0]
            print(f"  last clean state {clean[-1].stem}: p_s {ps/100:.1f} hPa, "
                  f"p(k={int(k)}) {p[int(k)]/100:.1f} hPa, p/p_s {p[int(k)]/ps:.3f}, "
                  f"z_s {float(z['phis'][i, j]) / constants.g:.0f} m")
            for c in clean:     # hottest cell in each saved clean state
                zc = np.load(c)
                a, b, kk = np.unravel_index(int(np.argmax(zc["T"])), zc["T"].shape)
                pc = np.asarray(sigma.pressure_at_full(np.asarray([float(zc["p_s"][a, b])])))[0]
                gl, go = np.degrees(np.asarray(g.lat)), np.degrees(np.asarray(g.lon))
                print(f"  {c.stem}: T max {float(zc['T'][a, b, kk]):.1f} K at "
                      f"{gl[a]:.2f}N {go[b]:.2f}E k={kk} p={pc[kk]/100:.1f} hPa "
                      f"z_s={float(zc['phis'][a, b]) / constants.g:.0f} m "
                      f"r={math.degrees(float(great_circle_distance(math.radians(go[b]), math.radians(gl[a]), 3 * math.pi / 2, 0.0, 1.0))):.1f} deg; "
                      f"column T(k={max(kk-3,0)}..{min(kk+3,39)}) "
                      f"{np.round(zc['T'][a, b, max(kk-3,0):kk+4], 1).tolist()}; "
                      f"T<=50.01 K cells {int(np.sum(zc['T'] <= 50.01))}")
        print(f"  probe trip {meta['trip_step']}")
        rows = [json.loads(l) for l in open(pd / "diag.jsonl")]
        fine = [d for d in rows if d["step"] >= meta["trip_step"] - 500] if meta["trip_step"] else rows
        for d in fine[:-10:50] + fine[-10:]:
            print(f"   step {d['step']} umax {d['umax']:.3g} spd@{d['spd_at']} "
                  f"n_unst_low {d['n_unst_low']:.0f} n_unst {d['n_unst']:.0f} "
                  f"dthmin {d['dth_min']:.2f}@{d['dth_at']} dpmin {d['dp_min']:.0f} "
                  f"ps [{d['ps_min']:.0f},{d['ps_max']:.0f}] T [{d['T_min']:.1f},{d['T_max']:.1f}]")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
