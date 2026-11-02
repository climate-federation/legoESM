"""Thin-air (rho < 0.1 kg/m3) MG2 point oracle fixture
(tests/atmosphere/microphysics/oracle_fixtures/mg2_thin_air.npz).

MG2's density factors carry no floor: rhof = (rhosu/rho)^0.54, arn = ar rhof,
asn = as rhof, ain = ai (rhosu/rho)^0.35 (micro_mg2_0.F90:895-901).  The point
grid spans rho from about 0.03 to 0.4 kg/m3 so it crosses 0.1 kg/m3, and runs
  * pt_*: ../morrison_mg2_ice_snow/mg2_snow_procs.F90 (unmodified
    micro_mg_utils.F90 kernels; rhof/arn/asn lines verbatim): rain
    evaporation (pre, am), snow fall speeds, self-aggregation, riming,
    Bergeron onto snow, snow sublimation;
  * ice_*: mg2_ice_fall_procs.F90: the cloud-ice fall speeds of the
    post-process state (:2088-2089, :2119-2147).
The points are physical states only as far as the kernels need (warm rain at
low pressure isolates the density factor from freezing).

usage: python make_fixture.py --cam <CAM checkout> --build <dir> --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DCS = 500.0e-6           # CAM6 micro_mg_dcs
RHOSN = 250.0            # MG2 rhosn
RHOI = 500.0             # MG2 rhoi
RHOW = 1000.0            # MG2 rhow
PT_IN = ("T", "p", "rho", "qfrac", "qcic", "ncic", "qs", "ns", "lcldm",
         "precip_frac", "qiic", "qr", "nr", "qi", "ni")
PT_OUT = ("qsic", "nsic", "lams", "n0s", "ums", "uns", "nsagg", "pgam", "lamc",
          "ncic_adj", "psacws", "npsacws", "bergs", "prds", "qvl", "qvi",
          "qric", "nric", "lamr", "mnuccr", "nnuccr", "lami", "n0i", "niic",
          "prci", "nprci", "pre", "am")
ICE_IN = ("T", "p", "rho", "dumi", "dumni")
ICE_OUT = ("dumni_adj", "lami", "vtrmi", "vni", "rhof", "ain")
PRESSURES = (2500.0, 5000.0, 8000.0, 12000.0, 30000.0)


def build_points(R_d):
    pts = []
    # rain in precipitating clear air (no cloud water, no snow, no ice);
    # warm so that MG2's Bigg freezing is off (T >= 269.15 K)
    for p in PRESSURES:
        for T in (276.0, 285.0):
            rho = p / (R_d * T)
            for qfrac in (0.3, 0.8):
                for lcldm, pf in ((1.0e-4, 1.0), (0.3, 0.8)):
                    for qr_ip, lam_r in ((1.0e-5, 8000.0), (3.0e-4, 3000.0)):
                        nr_ip = lam_r ** 3 * qr_ip / (math.pi * RHOW)
                        pts.append((T, p, rho, qfrac, 0.0, 0.0, 0.0, 0.0, lcldm, pf, 0.0,
                                    qr_ip * pf, nr_ip * pf, 0.0, 0.0))
    # snow, with and without supercooled cloud water; no rain, no ice
    for p in PRESSURES:
        for T in (230.0, 255.0, 268.0):
            rho = p / (R_d * T)
            for qfrac in (0.6, 0.97):
                for qcic, ncic in ((0.0, 0.0), (2.0e-4, 50.0e6 / rho)):
                    if T < 240.0 and qcic > 0:
                        continue
                    for qs_ip, lam_s in ((1.0e-5, 2000.0), (3.0e-4, 2000.0), (5.0e-3, 600.0)):
                        for lcldm, pf in ((1.0, 1.0), (0.3, 0.8)):
                            ns_ip = lam_s ** 3 * qs_ip / (math.pi * RHOSN)
                            pts.append((T, p, rho, qfrac, qcic, ncic, qs_ip * pf, ns_ip * pf,
                                        lcldm, pf, 0.0, 0.0, 0.0, 0.0, 0.0))
    return np.array(pts)


def build_ice_points(R_d):
    pts = []
    for p in PRESSURES:
        for T in (215.0, 240.0):
            rho = p / (R_d * T)
            for qi in (1.0e-7, 1.0e-5, 2.0e-4):
                # mean radius 1.5/lam [m]: 8, 19, 40, 150 um and the slope bounds
                for r in (8.0e-6, 19.0e-6, 40.0e-6, 150.0e-6, 0.5e-6, 2.0e-3):
                    lam = 1.5 / r
                    pts.append((T, p, rho, qi, lam ** 3 * qi / (math.pi * RHOI)))
    return np.array(pts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    from legoesm import constants as C
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.cam), str(a.build)], check=True)
    fx = {"dcs": np.array(DCS)}
    pts = build_points(C.R_d)
    pin, pout = a.build / "pt_in.txt", a.build / "pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_snow_procs"), str(pin), str(pout), repr(DCS)], check=True)
    po = np.loadtxt(pout)
    for i, k in enumerate(PT_IN):
        fx[f"pt_in_{k}"] = pts[:, i]
    for i, k in enumerate(PT_OUT):
        fx[f"pt_out_{k}"] = po[:, i]
    ice = build_ice_points(C.R_d)
    iin, iout = a.build / "ice_in.txt", a.build / "ice_out.txt"
    np.savetxt(iin, ice, fmt="%.17e", header=str(len(ice)), comments="")
    subprocess.run([str(a.build / "mg2_ice_fall_procs"), str(iin), str(iout), repr(DCS)],
                   check=True)
    io = np.loadtxt(iout)
    for i, k in enumerate(ICE_IN):
        fx[f"ice_in_{k}"] = ice[:, i]
    for i, k in enumerate(ICE_OUT):
        fx[f"ice_out_{k}"] = io[:, i]
    files = ["src/physics/cam/micro_mg2_0.F90", "src/physics/cam/micro_mg_utils.F90",
             "src/physics/cam/wv_sat_methods.F90"]
    meta = dict(oracle="ESCOMP/CAM", tag="cam_cesm2_1_rel_60",
                sha256={f: hashlib.sha256((a.cam / f).read_bytes()).hexdigest() for f in files},
                gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                        text=True).stdout.split("\n")[0],
                generator="python scripts/validate/oracle/morrison_mg2_thin_air/make_fixture.py "
                          f"--cam <CAM> --build <dir> --out {a.out.name}")
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", len(pts), "points", len(ice), "ice")


if __name__ == "__main__":
    main()
