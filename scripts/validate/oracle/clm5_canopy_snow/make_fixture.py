"""Fixture for CLM5.0 canopy snow, snowveg_flag ON_RAD
(tests/land/unit/oracle_fixtures/clm5_canopy_snow.npz).

Oracle: CTSM release-clm5.0.34 CanopyHydrologyMod.F90 / CanopyFluxesMod.F90 line
ranges compiled verbatim (build.sh, canopy_snow_driver.F90).  Cases: a grid
over canopy snow, canopy liquid, snowfall, air temperature (cold, between the
270.15 K unloading threshold and freezing, warm), wind, exposed leaf / stem
area, leaf temperature and wet-canopy evaporation (a fraction of the stores).
Rain is zero (the liquid interception is a separate, pre-existing port).  A
second block of dry steps (no precipitation) pins the recorded fall-off
difference.

usage: python make_fixture.py --clm <CTSM checkout> --build <dir with driver> --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

IN = ("snocan", "liqcan", "forc_snow", "forc_rain", "forc_t", "forc_wind", "elai",
      "esai", "dewmx", "dtime", "t_veg", "evap")
OUT = ("snocan_hyd", "grnd_snow", "fwet", "fcansno", "snocan_evap", "liqcan_evap",
       "snocanfall", "snounload")
FILES = {"CanopyHydrologyMod.F90": "src/biogeophys/CanopyHydrologyMod.F90",
         "CanopyFluxesMod.F90": "src/biogeophys/CanopyFluxesMod.F90"}
DT = 1800.0


def build_points():
    rows = []
    for (sn, lq, snow, T, U, (el, es), tv, fe) in itertools.product(
            (0.0, 0.4, 6.0, 30.0), (0.0, 0.05), (2.0e-5, 4.0e-4),
            (255.0, 271.0, 276.0), (0.5, 8.0), ((2.0, 0.6), (0.0, 0.5), (4.5, 1.0)),
            (262.0, 281.0), (0.0, 0.3, 0.9)):
        e = fe * (sn + lq) / DT
        rows.append((sn, lq, snow, 0.0, T, U, el, es, 0.1, DT, tv, e))
    # Dry steps (no precipitation): CLM5 skips its interception block, so a
    # store above capacity is NOT shed (the port sheds it below freezing --
    # recorded difference); unloading and FracWet still run.
    for (sn, T, U, (el, es)) in itertools.product(
            (2.0, 30.0), (255.0, 271.0, 276.0), (0.5, 8.0), ((0.5, 0.1), (2.0, 0.6))):
        rows.append((sn, 0.0, 0.0, 0.0, T, U, el, es, 0.1, DT, 262.0, 0.0))
    return np.asarray(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clm", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pts = build_points()
    pin, pout = a.build / "cs_in.txt", a.build / "cs_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "canopy_snow_driver"), str(pin), str(pout)],
                   check=True, stdout=subprocess.DEVNULL)
    po = np.loadtxt(pout)
    fx = {f"in_{k}": pts[:, i] for i, k in enumerate(IN)}
    fx.update({f"out_{k}": po[:, i] for i, k in enumerate(OUT)})
    sha = subprocess.run(["git", "-C", str(a.clm), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    meta = dict(source="ESCOMP/CTSM release-clm5.0.34", sha=sha,
                md5={k: hashlib.md5((a.clm / v).read_bytes()).hexdigest()
                     for k, v in FILES.items()},
                generator="scripts/validate/oracle/clm5_canopy_snow")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **fx)
    print(f"wrote {a.out}: {len(pts)} cases")


if __name__ == "__main__":
    main()
