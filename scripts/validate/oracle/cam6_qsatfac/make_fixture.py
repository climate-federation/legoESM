"""Fixture for CAM6 do_subgrid_growth (tests/atmosphere/microphysics/oracle_fixtures/cam6_qsatfac.npz).

Oracle: ESCOMP/CAM cam_cesm2_1_rel_60 wv_sat_methods.F90, micro_mg_utils.F90 and micro_mg2_0.F90 compiled
UNMODIFIED (build.sh) with qsatfac_driver.F90, which holds the verbatim cldfrc2m.F90 aist_vector lines that
set qsatfac (iceopt 5, :1010-1024, :1086-1117) and the micro_mg2_0.F90 lines that apply it as qsfm (:880,
:905-928, :933, :1118), then calls the unmodified MG2 routines that read qvl/qvi.

Points: the MG2 ice points of morrison_mg2_warm_rain/make_fixture.py (build_ice_points: -38 to +3 C, vapour
from ice-subsaturated to water-supersaturated, cloud ice and snow over their slope bounds), each once with
the tropospheric ramp (rhmini 0.8, rhmaxi 1.0, the CAM6 CLUBB values) and once with the stratospheric step
(rhminis = rhmaxis = 1.0).

usage: python make_fixture.py --cam <CAM checkout> --build <dir with qsatfac_driver> --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IN = ("T", "q", "qi", "ni_kg", "qcic", "qs", "ns_kg", "precip_frac", "icldm", "lcldm", "p",
      "rhmini", "rhmaxi")
OUT = ("qsatfac", "aist_raw", "qvl", "qvi", "gate", "vap_dep", "ice_sublim", "berg", "bergs", "prds")
RAMPS = ((0.8, 1.0), (1.0, 1.0))
FILES = ("cldfrc2m.F90", "micro_mg2_0.F90", "micro_mg_utils.F90", "wv_sat_methods.F90")


def build_points():
    spec = importlib.util.spec_from_file_location(
        "mg2wr_make_fixture", HERE.parent / "morrison_mg2_warm_rain/make_fixture.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    base = gen.build_ice_points()
    return np.concatenate([np.column_stack([base, np.full(len(base), lo), np.full(len(base), hi)])
                           for lo, hi in RAMPS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pts = build_points()
    pin, pout = a.build / "qsatfac_in.txt", a.build / "qsatfac_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "qsatfac_driver"), str(pin), str(pout)], check=True)
    po = np.loadtxt(pout)
    fx = {f"in_{k}": pts[:, i] for i, k in enumerate(IN)}
    fx.update({f"out_{k}": po[:, i] for i, k in enumerate(OUT)})
    fx["dcs"] = np.array(500.0e-6)
    fx["rhosn"] = np.array(250.0)
    src = a.cam / "src/physics/cam"
    meta = dict(
        oracle="ESCOMP/CAM", tag="cam_cesm2_1_rel_60",
        sha256={f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in FILES},
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        generator="python scripts/validate/oracle/cam6_qsatfac/make_fixture.py "
                  f"--cam <CAM> --build <dir> --out {a.out.name}")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, len(pts), "points")


if __name__ == "__main__":
    main()
