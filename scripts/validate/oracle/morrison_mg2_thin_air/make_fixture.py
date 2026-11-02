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
    post-process state (:2088-2089, :2119-2147);
  * isc_*: mg2_ice_sed_cols.F90: cloud-ice sedimentation of whole columns in
    MG2's order, the post-process ice qi + qitend dt sized and transported
    (:2063-2064, :2206-2214, :2232-2314), with sub-steps, clear-air
    sublimation and the surface ice flux;
  * ism_*: the same driver on two-level columns near 0 C where the ice
    falling into clear air cools the air before MG2's post-sedimentation
    snow melt tests it (:2536-2566);
  * rfz_*: mg2_rain_frz_pts.F90: the post-sedimentation rain freezing at
    -40 C (:2571-2610) on points around rainfrze, and the start-of-step one
    (:1182-1212) on the same rain and the bare temperature.
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


ISC_IN = ("qi", "ni", "qitend", "nitend", "icldm", "rho", "pdel", "t", "tlat", "dums", "dumns")
ISC_OUT = ("qitend", "nitend", "qisedten", "qvlat", "tlat", "smelt", "nsmelt")
ISC_DT = 1800.0          # CAM6 physics step [s]
ISC_P = (5000.0, 15000.0, 25000.0, 35000.0, 45000.0, 55000.0)   # level pressures, top first [Pa]
ISC_T = (210.0, 215.0, 222.0, 230.0, 238.0, 246.0)              # [K]
ISC_DP = (6000.0, 10000.0, 10000.0, 5000.0, 4000.0, 3000.0)     # layer thickness [Pa]
ISC_CLD = ((1.0, 1.0, 1.0, 1.0, 1.0, 1.0),          # uniform
           (1.0, 0.8, 0.5, 0.5, 0.2, 0.05),         # shrinking downward: ice falls into clear air
           (0.1, 0.3, 0.6, 1.0, 1.0, 1.0))          # growing downward
ISC_QIC = (2.0e-5, 1.0e-4, 0.0, 3.0e-5, 1.0e-6, 2.0e-4)          # in-cloud ice [kg/kg]; level 3 empty
ISC_R = ((150.0e-6, 60.0e-6, 40.0e-6, 25.0e-6, 19.0e-6, 8.0e-6),  # mean radius 1.5/lam [m]
         (1.0e-3, 600.0e-6, 400.0e-6, 1.5e-3, 1.0e-3, 1.5e-3))     # fast: several sub-steps


def build_ice_columns(R_d):
    """Ice columns for the MG2 sedimentation order: grid ice and number, and
    process tendencies before sedimentation that grow, deplete or leave the
    ice (the empty level 3 gains ice only through its tendency)."""
    rows = []
    for cld in ISC_CLD:
        for radii in ISC_R:
            for kind in ("none", "grow", "deplete"):
                for k in range(len(ISC_P)):
                    rho = ISC_P[k] / (R_d * ISC_T[k])
                    lam = 1.5 / radii[k]
                    qi = ISC_QIC[k] * cld[k]
                    ni = lam ** 3 * qi / (math.pi * RHOI)
                    if kind == "grow":
                        new = 5.0e-5 * cld[k]
                        qt, nt = (qi + new) / ISC_DT, (ni + lam ** 3 * new / (math.pi * RHOI)) / ISC_DT
                    elif kind == "deplete":
                        qt, nt = -0.6 * qi / ISC_DT, -0.3 * ni / ISC_DT
                    else:
                        qt = nt = 0.0
                    rows.append((qi, ni, qt, nt, cld[k], rho, ISC_DP[k],
                                 ISC_T[k], 0.0, 0.0, 0.0))
    return np.array(rows), len(ISC_CLD) * len(ISC_R) * 3, len(ISC_P)


# Snow-melt order columns: two levels near 0 C.  Ice falls from a full cloud
# (top) into a 5 % cloud (bottom), where 95 % of the inflow sublimates and
# cools the air that holds the snow; MG2 tests its +2 C snow melt on that
# cooled temperature.  The bottom air sits 0.2, 0.6 or 3 K above snowmelt
# after the processes' heating (0.2 K over the step).
ISM_P = (60000.0, 65000.0)       # [Pa]
ISM_DP = (3000.0, 3000.0)        # [Pa]
ISM_CLD = (1.0, 0.05)
ISM_T_TOP = 272.0                # [K]
ISM_SNOWMELT_OFFSET = 2.0        # MG2 snowmelt = tmelt + 2 K (:329)
ISM_HEAT_K = 0.2                 # processes' warming over the step [K]
ISM_OFF = (0.2, 0.6, 3.0)        # bottom T + heating - snowmelt [K]
ISM_QS = (1.0e-8, 5.0e-4, 2.0e-3)   # bottom snow [kg/kg]; the larger two melt heat-limited
ISM_QI = (0.0, 5.0e-5, 2.0e-4)   # top in-cloud ice [kg/kg]
ISM_R = 150.0e-6                 # top ice mean radius 1.5/lam [m]
ISM_LAMS = 2000.0                # snow slope [1/m] for its number


def build_melt_columns(R_d, cp, T_freeze):
    rows = []
    lam = 1.5 / ISM_R
    for qi in ISM_QI:
        for off in ISM_OFF:
            for qs in ISM_QS:
                T_bot = T_freeze + ISM_SNOWMELT_OFFSET + off - ISM_HEAT_K
                for k, T in enumerate((ISM_T_TOP, T_bot)):
                    rho = ISM_P[k] / (R_d * T)
                    q = qi * ISM_CLD[k] if k == 0 else 0.0
                    n = lam ** 3 * q / (math.pi * RHOI)
                    tl = cp * ISM_HEAT_K / ISC_DT if k == 1 else 0.0
                    s = qs if k == 1 else 0.0
                    rows.append((q, n, 0.0, 0.0, ISM_CLD[k], rho, ISM_DP[k], T, tl,
                                 s, ISM_LAMS ** 3 * s / (math.pi * RHOSN)))
    return np.array(rows), len(ISM_QI) * len(ISM_OFF) * len(ISM_QS), len(ISM_P)


# Post-sedimentation rain freezing points (rfz_*, mg2_rain_frz_pts.F90,
# micro_mg2_0.F90:2571-2610): t + tlat/cpp dt from 5 K below rainfrze
# (= tmelt - 40 K, :331) to 0.5 K above it, tlat the processes' and the
# sedimentation's heating (a 0.3 K warming over the step), rain from 1e-8 to
# 1e-3 kg/kg (the larger amounts freeze heat-limited near rainfrze) with mean
# drops 1/lamr of 100 um (to cloud ice) and 1 mm (bounded to 500 um by MG2's
# rain slope limits), each point at two values of Dcs.
RFZ_IN = ("t", "tlat", "dumr", "dumnr")
RFZ_OUT = ("qrfrz", "nrfrz", "qsfrz", "nsfrz", "qifrz", "nifrz", "lamr", "tlat",
           "minstrf", "ninstrf", "tlat0")   # the last three: start-of-step freezing
RFZ_RAINFRZE_OFFSET = 40.0      # MG2 rainfrze = tmelt - 40 K (:331)
RFZ_HEAT_K = 0.3                 # heating before the freezing over the step [K]
RFZ_OFF = (-5.0, -0.5, -0.05, 0.5)       # t + tlat/cpp dt - rainfrze [K]
RFZ_QR = (1.0e-8, 1.0e-5, 1.0e-3)        # grid rain [kg/kg]
RFZ_DROP = (100.0e-6, 1.0e-3)            # mean drop 1/lamr [m]
# Dcs [m]: CAM6's 500 um, where MG2's rain slope bound (1/500 um) puts every
# large drop exactly on 1/Dcs (cloud ice in MG2), and 200 um, where 1 mm
# drops go to snow
RFZ_DCS = (DCS, 200.0e-6)


# Start-of-step rain freezing points (rfp_*, the same driver's minstrf columns,
# micro_mg2_0.F90:1182-1212, on the bare t): 0.01, 0.1 and 1 K below rainfrze
# and 0.1 K above, rain 1e-4 and 3e-3 kg/kg (heat-limited at the larger
# amount up to 1 K below).
RFP_OFF = (-1.0, -0.1, -0.01, 0.1)      # t - rainfrze [K]
RFP_QR = (1.0e-4, 3.0e-3)               # rain [kg/kg]
RFP_LAM = 3000.0                        # rain slope for its number [1/m]


def build_rain_prefreeze_points(T_freeze):
    return np.array([(T_freeze - RFZ_RAINFRZE_OFFSET + off, 0.0, qr,
                      RFP_LAM ** 3 * qr / (math.pi * RHOW))
                     for off in RFP_OFF for qr in RFP_QR])


def build_rain_freeze_points(cp, T_freeze):
    pts = []
    for off in RFZ_OFF:
        for qr in RFZ_QR:
            for drop in RFZ_DROP:
                lam = 1.0 / drop
                pts.append((T_freeze - RFZ_RAINFRZE_OFFSET + off - RFZ_HEAT_K,
                            cp * RFZ_HEAT_K / ISC_DT, qr, lam ** 3 * qr / (math.pi * RHOW)))
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
    fx["isc_dt"] = np.array(ISC_DT)
    for tag, (cols, ncol, nlev) in (("isc", build_ice_columns(C.R_d)),
                                     ("ism", build_melt_columns(C.R_d, C.c_pd, C.T_freeze))):
        cin, cout = a.build / f"{tag}_in.txt", a.build / f"{tag}_out.txt"
        np.savetxt(cin, cols, fmt="%.17e", header=f"{ncol} {nlev} {ISC_DT!r}", comments="")
        subprocess.run([str(a.build / "mg2_ice_sed_cols"), str(cin), str(cout), repr(DCS)],
                       check=True)
        lines = cout.read_text().split("\n")
        body = np.array([[float(x) for x in ln.split()] for ln in lines[:ncol * nlev]])
        tail = np.array([[float(x) for x in ln.split()] for ln in lines[ncol * nlev:]
                         if ln.strip()])
        for i, k in enumerate(ISC_IN):
            fx[f"{tag}_in_{k}"] = cols[:, i].reshape(ncol, nlev)
        for i, k in enumerate(ISC_OUT):
            fx[f"{tag}_out_{k}"] = body[:, i].reshape(ncol, nlev)
        fx[f"{tag}_out_preci"], fx[f"{tag}_out_nstep"] = tail[:, 0], tail[:, 1].astype(int)
    rfz = build_rain_freeze_points(C.c_pd, C.T_freeze)
    rin, rout = a.build / "rfz_in.txt", a.build / "rfz_out.txt"
    np.savetxt(rin, rfz, fmt="%.17e", header=f"{len(rfz)} {ISC_DT!r}", comments="")
    ro, dcs_col = [], []
    for dcs in RFZ_DCS:
        subprocess.run([str(a.build / "mg2_rain_frz_pts"), str(rin), str(rout), repr(dcs)],
                       check=True)
        ro.append(np.loadtxt(rout))
        dcs_col.append(np.full(len(rfz), dcs))
    ro = np.concatenate(ro)
    rfz = np.concatenate([rfz] * len(RFZ_DCS))
    fx["rfz_in_dcs"] = np.concatenate(dcs_col)
    for i, k in enumerate(RFZ_IN):
        fx[f"rfz_in_{k}"] = rfz[:, i]
    for i, k in enumerate(RFZ_OUT):
        fx[f"rfz_out_{k}"] = ro[:, i]
    rfp = build_rain_prefreeze_points(C.T_freeze)
    pin, pout = a.build / "rfp_in.txt", a.build / "rfp_out.txt"
    np.savetxt(pin, rfp, fmt="%.17e", header=f"{len(rfp)} {ISC_DT!r}", comments="")
    subprocess.run([str(a.build / "mg2_rain_frz_pts"), str(pin), str(pout), repr(DCS)], check=True)
    po = np.loadtxt(pout)
    for i, k in enumerate(RFZ_IN):
        fx[f"rfp_in_{k}"] = rfp[:, i]
    for k in ("minstrf", "ninstrf", "tlat0"):
        fx[f"rfp_out_{k}"] = po[:, RFZ_OUT.index(k)]
    files =["src/physics/cam/micro_mg2_0.F90", "src/physics/cam/micro_mg_utils.F90",
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
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", len(pts), "points", len(ice), "ice",
          ncol, "ice columns")


if __name__ == "__main__":
    main()
