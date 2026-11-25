"""CLUBB-MF plume-area factor sweep from saved SCM steps (offline, no rerun per factor).

Inputs (from scripts/tmp diag_mf_fac_steps.py, run at factor 1, bsort off):
  STEPS npz: per-step window arrays mfcf, mfqc, ccf (step, nlev), top-first.
  OUT npz:   window means + rho, dz, z, LES profiles.
In the SCM radiation is prescribed and convection is off (ZM deep cloud = 0),
so the factor only changes the plume cloud the model WOULD hand radiation.
Per step that cloud is built with the model's own functions
(clubb_mf.mf_deep_cloud: factor, frac/ic gates, bottom layer zeroed;
mf_concld: 0.8 cap), then time-averaged:
  grid plume water = <concld * icwmr>, cover = min(<ccf + concld>, 1).
The "ungated" column repeats the earlier (pre-gate) instrument for comparison.
Cover RMSE vs LES cloud fraction on LES heights up to LES cloud top + 200 m.

Usage (compute node): python diag_mf_cldfrac_fac.py STEPS.npz OUT.npz [...]
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.turbulence.clubb_mf import mf_concld, mf_deep_cloud

jax.config.update("jax_enable_x64", True)
FACTORS = (1.0, 1.25, 1.5, 1.75, 2.0, 10.0)


def score(steps, out):
    s, d = np.load(steps), np.load(out)
    rho, dz, z = d["rho"], d["dz"], d["z"]
    lz, lcf = d["les_z"], d["les_cf"]
    order = np.argsort(z)  # model levels are stored top-first
    keep = lz <= lz[lcf > 1e-4].max() + 200.0
    a, qc, ccf = (jnp.asarray(s[k]) for k in ("mfcf", "mfqc", "ccf"))
    zero = jnp.zeros_like(a)
    lwp_clubb = float(d["lwp_scm"])
    print(f"{out} git={d['git']} steps={a.shape[0]} LES LWP {float(d['lwp_les']):.2f}"
          f" cfmax {lcf.max():.4f} CLUBB LWP {lwp_clubb:.3f}")
    for f in FACTORS:
        cfg = CloudConfig()._replace(mf_cldfrac_fac=f)
        deepcu, icwmr = mf_deep_cloud(zero, zero, a, qc, cfg)
        concld = mf_concld(deepcu, cfg)
        water = np.asarray(jnp.mean(concld * icwmr, 0))
        cf = np.minimum(np.asarray(jnp.mean(ccf + concld, 0)), 1.0)
        lwp = float(np.sum(rho * water * dz)) * 1e3 + lwp_clubb
        lwp_ungated = float(np.sum(rho * f * np.asarray(jnp.mean(a * qc, 0)) * dz)) * 1e3 + lwp_clubb
        cfi = np.interp(lz[keep], z[order], cf[order])
        rmse = float(np.sqrt(np.mean((cfi - lcf[keep]) ** 2)))
        print(f"  F={f:5.2f} LWP={lwp:7.2f} (ungated {lwp_ungated:7.2f}) cfmax={cf.max():.4f}"
              f" cfRMSE={rmse:.4f} bias={float(np.mean(cfi - lcf[keep])):+.4f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        score(args[i], args[i + 1])
