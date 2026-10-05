#!/usr/bin/env python3
"""Round 212 / VORTEX_SMT round 2 -- the gdept statement, proved to the bit.

Round 1 left ONE finding open and PLAUSIBLE: the seamount run's INITIAL
temperature differs from the flat run's by up to 2.79e-05 K on 3023 cells
that are wet in both, including columns 532 km from the summit whose
``k_bot`` is still 10.  Round 1's guess was "the depths handed to
``usr_def_istate`` are rebuilt from the 3-D scale factors under
``key_vco_1d3d``".

**That guess is REFUTED by NEMO's own macros, and the real statement is
cited and reconstructed here.**

  src/OCE/DOM/domzgr_substitute.h90:71   #if defined key_vco_1d || defined key_vco_1d3d
                                   :75     #define DEPt_0(i,j,k)   gdept_1d(k)
                                   :78     #define gdept_0(i,j,k)  gdept_1d(k)
                                   :82-100 ONLY the E3*_0 macros split on the key

  So ``gdept_0`` is the 1-D ladder under BOTH keys.  What differs is the
  key_qco stretching:

  src/OCE/DOM/domzgr_substitute.h90:139  #define gdept(i,j,k,t) ((DEPt_0(i,j,k) Tisf(r3t,risfdep,i,j,t))
                                   :51     #define Tisf(r3,isf,i,j,t)  ) Time(r3,i,j,t)     [no key_isf]
                                   :50     #define Time(r3,i,j,t)      *(1._wp+r3(i,j,t))
  i.e.                 gdept(i,j,k,t) = gdept_1d(k) * ( 1 + r3t(i,j,t) )

  src/OCE/DOM/domqco.F90  dom_qco_r3c :    pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)
  src/OCE/DOM/domain.F90:139-144      :    ht_0 = SUM_k e3t_0(:,:,jk) * tmask(:,:,jk)
  src/OCE/DOM/domain.F90:158          :    r1_ht_0 = ssmask / ( ht_0 + 1 - ssmask )
  src/OCE/DOM/istate.F90:127-130      :    zgdept(:,:,jk) = gdept(:,:,jk,Kbb)
                                           CALL usr_def_istate( zgdept, tmask, ... )

  Under ``key_vco_1d``  e3t_0 = e3t_1d        -> ht_0 = 5000 m in every wet column.
  Under ``key_vco_1d3d`` e3t_0 = e3t_3d       -> ht_0 IS the seamount bathymetry.
  The vortex's initial ssh is non-zero, so the two runs hand ``usr_def_istate``
  different depths.  Nothing is "rebuilt"; the 1-D ladder is simply stretched
  by a different column depth.

This probe reconstructs NEMO's initial-T statement
(tests/VORTEX/MY_SRC/usrdef_istate.F90:69-88) in fp64 from each run's OWN
``mesh_mask.nc`` and its OWN recorded kt=1 ssh, and compares against the
recorded kt=1 T.  Reconstructing BOTH runs is the control: a reconstruction
that only matched the seamount run would prove nothing.

Pre-impl search (RULE 4): grepped scripts/validate/ocean_fidelity for an
existing step-entry reader -- ``nemo_testcase_phase3_trajectory_gate.read_entry``
(self-describing, note BD) is REUSED, as round 1's sanity probe reuses it.
No second reader is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import netCDF4

sys.path.insert(0, str(Path(__file__).parent))
from nemo_testcase_phase3_trajectory_gate import read_entry, require  # noqa: E402

# --- src/OCE/DOM/phycst.F90 ---------------------------------------------
RPI = 3.141592653589793                      # :25
RAD = 3.141592653589793 / 180.0              # :26
RDAY = 24.0 * 60.0 * 60.0                    # :29
GRAV = 9.80665                               # :38
RSIYEA = 365.25 * RDAY * 2.0 * RPI / 6.283076   # phy_cst
RSIDAY = RDAY / (1.0 + RDAY / RSIYEA)           # phy_cst
OMEGA = 2.0 * RPI / RSIDAY                      # phy_cst, no key_cice


def istate_T(pdept, tmask, glamt, gphit, rho0, rn_a0, gphi0, umax):
    """tests/VORTEX/MY_SRC/usrdef_istate.F90:69-88, statement by statement."""
    zf0 = 2.0 * OMEGA * np.sin(RAD * gphi0)                       # :69
    zumax = umax * np.sign(zf0)                                   # :70
    zlambda = np.sqrt(2.0) * 60.0e3                               # :71
    zn2 = 3.0e-3 ** 2                                             # :72
    zH = 0.5 * 5000.0                                             # :73
    zP0 = rho0 * zf0 * zumax * zlambda * np.sqrt(np.exp(1.0) / 2.0)  # :75
    zx = glamt * 1.0e3                                            # :79
    zy = gphit * 1.0e3                                            # :80
    zdt = pdept                                                   # :82
    zrho1 = rho0 * (1.0 + zn2 * zdt / GRAV)                       # :83
    shallow = zdt < zH                                            # :84
    # the Fortran NEVER evaluates EXP(zdt-zH) outside the branch, and at
    # zdt ~ 4750 m it would overflow; so neither does this.
    zsafe = np.where(shallow, zdt, zH)
    corr = (zP0 * (1.0 - np.exp(zsafe - zH))
            * np.exp(-(zx[..., None] ** 2 + zy[..., None] ** 2) / zlambda ** 2)
            / (GRAV * (zH - 1.0 + np.exp(-zH))))                  # :85-86
    zrho1 = np.where(shallow, zrho1 - corr, zrho1)                # :84-87
    return (20.0 + (rho0 - zrho1) / rn_a0) * tmask                # :88


def load(run_dir: Path, case: str):
    with netCDF4.Dataset(run_dir / "mesh_mask.nc") as h:
        tmask = np.asarray(h.variables["tmask"][0], dtype=np.float64)   # (z,y,x)
        glamt = np.asarray(h.variables["glamt"][0], dtype=np.float64)
        gphit = np.asarray(h.variables["gphit"][0], dtype=np.float64)
        gdept_1d = np.asarray(h.variables["gdept_1d"][0], dtype=np.float64)
        if "e3t_0" in h.variables:            # key_vco_1d3d: the 3-D field
            e3t_0 = np.asarray(h.variables["e3t_0"][0], dtype=np.float64)
            vco = "1d3d"
        else:                                 # key_vco_1d: the 1-D ladder only
            e3t_1d = np.asarray(h.variables["e3t_1d"][0], dtype=np.float64)
            e3t_0 = np.broadcast_to(e3t_1d[:, None, None], tmask.shape).copy()
            vco = "1d"
    # domain.F90:139-144 then :158
    ht_0 = np.einsum("kyx,kyx->yx", e3t_0, tmask)
    ssmask = (tmask.sum(axis=0) > 0.0).astype(np.float64)
    r1_ht_0 = ssmask / (ht_0 + 1.0 - ssmask)
    rec = read_entry(run_dir / "oracle_step_entry_kt00000001.bin", case)
    # record arrays are (y, x, z); mesh_mask is (z, y, x)
    return {
        "vco": vco,
        "tmask": np.transpose(tmask, (1, 2, 0)),
        "glamt": glamt, "gphit": gphit, "gdept_1d": gdept_1d,
        "ht_0": ht_0, "r1_ht_0": r1_ht_0,
        "T": rec["T"], "ssh": rec["ssh"],
    }


def reconstruct(d, rho0, rn_a0, gphi0, umax):
    # domzgr_substitute.h90:139/:51/:50 + domqco dom_qco_r3c
    r3t = d["ssh"] * d["r1_ht_0"]
    pdept = d["gdept_1d"][None, None, :] * (1.0 + r3t[..., None])
    return istate_T(pdept, d["tmask"], d["glamt"], d["gphit"],
                    rho0, rn_a0, gphi0, umax), pdept


def ulp_of(a):
    return np.nextafter(np.abs(a), np.inf) - np.abs(a)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smt", required=True, type=Path)
    p.add_argument("--flat", required=True, type=Path)
    p.add_argument("--case", default="vortex")
    p.add_argument("--rho0", type=float, required=True)
    p.add_argument("--rn-a0", type=float, required=True)
    p.add_argument("--gphi0", type=float, required=True)
    p.add_argument("--umax", type=float, required=True)
    p.add_argument("--json", type=Path)
    a = p.parse_args()

    out = {"omega": OMEGA, "rho0": a.rho0, "rn_a0": a.rn_a0,
           "rn_ppgphi0": a.gphi0, "rn_ppumax": a.umax}
    runs = {"smt": load(a.smt, a.case), "flat": load(a.flat, a.case)}
    require(runs["smt"]["vco"] == "1d3d", "the smt run is not a key_vco_1d3d run")
    require(runs["flat"]["vco"] == "1d", "the flat run is not a key_vco_1d run")
    require(np.array_equal(runs["smt"]["ssh"], runs["flat"]["ssh"]),
            "kt=1 ssh is not bit-identical between the runs; the statement "
            "below assumes it (round 1 measured that it is)")
    require(np.array_equal(runs["smt"]["gdept_1d"], runs["flat"]["gdept_1d"]),
            "gdept_1d differs between the runs -- P1 would be refuted")
    out["gdept_1d_bit_identical"] = True

    for k, d in runs.items():
        rec, pdept = reconstruct(d, a.rho0, a.rn_a0, a.gphi0, a.umax)
        wet = d["tmask"] > 0.0
        err = np.abs(rec - d["T"])[wet]
        out[k] = {
            "ht_0_min": float(d["ht_0"][d["ht_0"] > 0].min()),
            "ht_0_max": float(d["ht_0"].max()),
            "n_wet": int(wet.sum()),
            "recon_max_abs_err_K": float(err.max()),
            "recon_max_ulp": float((err / ulp_of(d["T"][wet])).max()),
            "n_exact": int((err == 0.0).sum()),
        }
        d["recon"] = rec
        d["pdept"] = pdept

    both = (runs["smt"]["tmask"] > 0) & (runs["flat"]["tmask"] > 0)
    dT_rec = runs["smt"]["T"] - runs["flat"]["T"]
    dT_pred = runs["smt"]["recon"] - runs["flat"]["recon"]
    ddep = runs["smt"]["pdept"] - runs["flat"]["pdept"]
    j, i, k = np.unravel_index(np.argmax(np.abs(np.where(both, dT_rec, 0.0))),
                               dT_rec.shape)
    jp, ip, kp = np.unravel_index(
        np.argmax(np.abs(np.where(both, dT_pred, 0.0))), dT_pred.shape)
    resid = np.abs(dT_pred - dT_rec)[both]
    out["difference"] = {
        "n_cells_wet_in_both": int(both.sum()),
        "recorded_max_abs_dT_K": float(np.abs(dT_rec[both]).max()),
        "predicted_max_abs_dT_K": float(np.abs(dT_pred[both]).max()),
        "argmax_recorded_jik": [int(j), int(i), int(k)],
        "argmax_predicted_jik": [int(jp), int(ip), int(kp)],
        "argmax_same_cell": [int(j), int(i), int(k)] == [int(jp), int(ip), int(kp)],
        "max_abs_residual_K": float(resid.max()),
        "max_depth_difference_m": float(np.abs(ddep[both]).max()),
        "n_cells_dT_nonzero": int((dT_rec[both] != 0.0).sum()),
    }
    # the control: the SAME statement with the flat ht_0 in BOTH runs must
    # erase the difference.  If it does not, the mechanism is not the only one.
    ctrl = runs["smt"].copy()
    ctrl["r1_ht_0"] = runs["flat"]["r1_ht_0"]
    ctrl_T, _ = reconstruct(ctrl, a.rho0, a.rn_a0, a.gphi0, a.umax)
    out["control_flat_ht0_in_smt"] = {
        "max_abs_dT_vs_flat_recon_K":
            float(np.abs((ctrl_T - runs["flat"]["recon"])[both]).max()),
    }
    txt = json.dumps(out, indent=2, sort_keys=True)
    print(txt)
    if a.json:
        a.json.write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
