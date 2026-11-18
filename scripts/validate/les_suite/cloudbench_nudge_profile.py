"""Effective free-troposphere relaxation rate the CloudBench LES applied, per height.

For artifacts whose LES microphysics source is ~0 above 2.5 km (cases listed on
the command line, chosen from cloudbench_micro_budget.py output), the day-5 budget
residual without nudging is the nudging the LES actually applied:
N_eff(z) = observed - (turbulent flux convergence + subsidence + advection +
radiation/Pi), and r_eff(z) = N_eff / <x0 - x>_t (theta_l and q_t separately),
against the recorded rate (swirl-lm defaults, ramp 3.0-3.5 km, tau 1 day).
Numbers only. usage: cloudbench_nudge_profile.py OUT_JSON CASE [CASE ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloudbench_jump_audit import ART, FORCING, _upwind_dxdz  # noqa: E402
from legoesm.atmosphere.les_suite.cloudbench import read_sounding  # noqa: E402

from legoesm import constants  # noqa: E402

ZBINS = np.arange(2500.0, 6001.0, 250.0)


def case_rates(case):
    d = np.load(ART / f"cloudbench_{case}_amip.npz", allow_pickle=True)
    site, month = int(case[1:4]), int(case[6:8])
    snd = read_sounding(FORCING / str(site) / str(month) / "amip" / "sounding.csv")
    z, t = d["heights_m"], d["times_s"][1:]
    rho = np.interp(z, snd["z"], snd["rho"])
    exner = (np.interp(z, snd["z"], snd["p"]) / constants.p_ref) ** constants.kappa
    w, rate = d["subsidence_w"], d["nudge_rate"]

    def tmean(a):
        return np.trapezoid(a, t, axis=0) / (t[-1] - t[0])
    out = {}
    for key, xk, fr, fs, adv, extra in (
            ("theta_l", "theta", "wtheta_resolved", "wtheta_sgs", d["theta_adv"],
             d["rad_T_tendency"][1:] / exner),
            ("q_t", "qt", "wqt_resolved", "wqt_sgs", d["qv_adv"], 0.0 * d["rad_T_tendency"][1:])):
        x = d[xk]
        xs = x[1:]
        obs = (xs[-1] - xs[0]) / (t[-1] - t[0])
        flux = (d[fr] + d[fs])[1:]
        rhs = (tmean(-np.gradient(rho * flux, z, axis=1) / rho)
               + tmean(-w * _upwind_dxdz(xs, z, w)) + adv + tmean(extra))
        n_eff = obs - rhs
        dev = tmean(x[0] - xs)
        rows = []
        for lo, hi in zip(ZBINS[:-1], ZBINS[1:]):
            m = (z >= lo) & (z < hi)
            num, den = (n_eff[m] * dev[m]).sum(), (dev[m] ** 2).sum()
            rows.append({"z_lo_m": float(lo), "r_eff_per_day": float(86400 * num / den),
                         "r_recorded_per_day": float(86400 * rate[m].mean()),
                         "rms_dev": float(np.sqrt((dev[m] ** 2).mean()))})
        out[key] = rows
    return out


def main() -> int:
    res = {c: case_rates(c) for c in sys.argv[2:]}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for key in ("theta_l", "q_t"):
        print(key, " z_lo  recorded  r_eff(median over cases with rms_dev > thr)")
        for i, lo in enumerate(ZBINS[:-1]):
            thr = 0.5 if key == "theta_l" else 2e-4
            v = [res[c][key][i]["r_eff_per_day"] for c in res if res[c][key][i]["rms_dev"] > thr]
            rec = res[next(iter(res))][key][i]["r_recorded_per_day"]
            print(f"  {lo:6.0f} {rec:6.3f}  {np.median(v) if v else float('nan'):7.3f}  n={len(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
