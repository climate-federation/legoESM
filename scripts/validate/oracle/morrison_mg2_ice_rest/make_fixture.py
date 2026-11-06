"""Generate the CAM6 MG2 ice follow-up oracle fixture by RUNNING the oracle.

Oracle: ESCOMP/CAM ``cam_cesm2_1_rel_60`` (commit
a03b84b7c4e34f965b115686f22a043b85739e56), ``src/physics/cam/micro_mg_utils.F90``
(``size_dist_param_basic`` on ``mg_ice_props`` / ``mg_snow_props``,
``size_dist_param_liq``, ``ice_autoconversion``, ``accrete_cloud_water_snow``)
compiled UNMODIFIED, called by ``mg2_ice_rest_procs.F90``; micro_mg_dcs = 500 um.

Points (in-cloud states):
  * ice: T x in-cloud q_i (through MG2's ``qiic >= qsmall`` gate, oracle row 30)
    x slope (inside and outside the ice bounds ``1/[2 dcs, 1 um]``, row 26);
  * riming: in-cloud q_c at, just above and well above qsmall with snow present
    (the ``qcic == qsmall`` boundary of ``accrete_cloud_water_snow``, row 31).

Usage (CPU node; needs gfortran and the CAM checkout)::

    python scripts/validate/oracle/morrison_mg2_ice_rest/make_fixture.py \\
        --cam /burg/glab/users/pg2328/oracles/CAM --build /tmp/mg2ir_bld \\
        --out tests/atmosphere/microphysics/oracle_fixtures/mg2_ice_rest.npz
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DCS = 500.0e-6      # CAM6 MG2 micro_mg_dcs (namelist_defaults_cam.xml:1244)
RHOI = 500.0        # MG2 rhoi [kg/m^3]
RHOSN = 250.0       # MG2 rhosn [kg/m^3]
QSMALL = 1.0e-18    # MG2 qsmall (micro_mg_utils.F90)
PT_IN = ("T", "p", "rho", "qiic", "niic", "qcic", "ncic", "qsic", "nsic")
PT_OUT = ("niic", "lami", "n0i", "prci", "nprci", "pgam", "lamc", "psacws",
          "npsacws")


def build_points():
    from legoesm import constants as C
    p = 50000.0
    pts = []
    for T in (250.0, C.T_freeze, 275.0):
        rho = p / (C.R_d * T)
        for qi in (5.0e-19, QSMALL, 3.0e-18, 1.0e-16, 5.0e-15, 1.0e-14, 2.0e-14,
                   1.0e-9, 1.0e-6):
            # slopes [1/m]: below 1/(2 dcs); between it and the SAM 1/600 um;
            # inside both; between the E3SM 1/10 um and 1/1 um; above 1/1 um
            for lam in (300.0, 1200.0, 2.0e4, 3.0e5, 3.0e6):
                ni = lam ** 3 * qi / (math.pi * RHOI)
                pts.append((T, p, rho, qi, ni, 0.0, 0.0, 0.0, 0.0))
    T = 260.0
    rho = p / (C.R_d * T)
    qs, lam_s = 1.0e-4, 2000.0
    ns = lam_s ** 3 * qs / (math.pi * RHOSN)
    for qc in (QSMALL, QSMALL * (1.0 + 1.0e-6), 1.0e-9, 1.0e-6, 2.0e-4):
        pts.append((T, p, rho, 0.0, 0.0, qc, 100.0e6 / rho, qs, ns))
    return np.array(pts)


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.cam), str(a.build)],
                   check=True)
    pts = build_points()
    pin = a.build / "pt_in.txt"
    pout = a.build / "pt_out.txt"
    np.savetxt(pin, pts, fmt="%.17e", header=str(len(pts)), comments="")
    subprocess.run([str(a.build / "mg2_ice_rest_procs"), str(pin), str(pout),
                    repr(DCS)], check=True)
    po = np.loadtxt(pout)
    fx = {"dcs": np.array(DCS)}
    for i, k in enumerate(PT_IN):
        fx[f"pt_in_{k}"] = pts[:, i]
    for i, k in enumerate(PT_OUT):
        fx[f"pt_out_{k}"] = po[:, i]
    meta = dict(
        oracle="ESCOMP/CAM", tag="cam_cesm2_1_rel_60", sha=_git(a.cam, "rev-parse", "HEAD"),
        files=["src/physics/cam/micro_mg_utils.F90", "src/physics/cam/micro_mg2_0.F90",
               "src/physics/cam/wv_sat_methods.F90"],
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        legoesm_sha=_git(HERE, "rev-parse", "HEAD"),
        generator=("python scripts/validate/oracle/morrison_mg2_ice_rest/"
                   f"make_fixture.py --cam <CAM> --build <dir> --out {a.out.name}"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


if __name__ == "__main__":
    main()
