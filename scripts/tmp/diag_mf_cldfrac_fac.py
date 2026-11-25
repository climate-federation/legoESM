"""CLUBB-MF plume-area factor sweep from saved SCM window means (offline, no rerun).

Inputs: npz from scripts/tmp diag_shallow_clubb_mf_bsort.py (bsort off arms),
run at factor 1. In the SCM radiation is prescribed, so the factor only scales
the plume diagnostics: grid plume water = F * <a*qc>, plume fraction =
min(F*<a>, 0.8). Window means taken before the factor (linear while the cap
does not bind, F <= 2 here). Cover = CLUBB cloud + plume fraction (probe's
additive combination), interpolated to LES heights up to LES cloud top + 200 m.

Usage: python scripts/tmp/diag_mf_cldfrac_fac.py bsm2_b_none.npz [...]
"""
import sys

import numpy as np

FACTORS = (1.0, 1.25, 1.5, 1.75, 2.0, 10.0)

for path in sys.argv[1:]:
    d = np.load(path)
    rho, dz, z = d["rho"], d["dz"], d["z"]
    lz, lcf = d["les_z"], d["les_cf"]
    order = np.argsort(z)  # model levels are stored top-first
    keep = lz <= lz[lcf > 1e-4].max() + 200.0
    print(f"{path} git={d['git']} LES LWP {float(d['lwp_les']):.2f} cfmax {lcf.max():.4f}"
          f" CLUBB LWP {float(d['lwp_scm']):.3f}")
    for f in FACTORS:
        lwp = float(np.sum(rho * f * d["mfaq"] * dz)) * 1e3 + float(d["lwp_scm"])
        cf = np.minimum(d["ccf"] + np.minimum(f * d["mfcf"], 0.8), 1.0)
        cfi = np.interp(lz[keep], z[order], cf[order])
        rmse = float(np.sqrt(np.mean((cfi - lcf[keep]) ** 2)))
        print(f"  F={f:5.2f} LWP={lwp:7.2f} cfmax={cf.max():.4f} cfRMSE={rmse:.4f}"
              f" bias={float(np.mean(cfi - lcf[keep])):+.4f}")
