"""Audit a saved MPAS AMIP checkpoint for the orphan rain-number pathology (#1515).

The #1515 detonations ran on a tree whose rain number N_r could survive with no
rain mass ("orphan" number) and grow far above the size-distribution ceiling.
This reports, from the saved state alone, the quantities the issue used:

- orphan cell-levels: N_r > 1e12 with q_r < 1e-8 (the issue's definition);
- cell-levels with N_r > 1e15, max N_r, and N_r with q_r below the model's
  clearing threshold;
- cell-levels above the post-step PSD ceiling
  N_hi = lamr_max**3 * q_r / (pi * rho_w)   [per kg; rho cancels per mass].

Checkpoints stamped ``number_convention='per_mass'`` hold N_r in [1/kg] and
every threshold here is applied per kg. The issue quoted [1/m^3]; per-volume
number is per-mass times air density (< 1.3 kg/m^3), so the two are not
interchangeable near a threshold -- convert per level before comparing.

Usage: rain_number_audit.py CHECKPOINT.npz [...]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig

ORPHAN_N = 1.0e12
ORPHAN_Q = 1.0e-8
HUGE_N = 1.0e15
CLEAR_Q = 1.0e-14  # morrison_microphysics clears N_r below this q_r


def audit(n_r: np.ndarray, q_r: np.ndarray, lamr_max: float) -> dict:
    if not (np.isfinite(n_r).all() and np.isfinite(q_r).all()):
        raise ValueError("non-finite rain state")
    n_hi = lamr_max ** 3 * np.clip(q_r, 0.0, None) / (np.pi * constants.rho_water)
    return {
        "max_N_r_per_kg": float(n_r.max()),
        "orphan_levels": int(((n_r > ORPHAN_N) & (q_r < ORPHAN_Q)).sum()),
        "levels_N_r_gt_1e15": int((n_r > HUGE_N).sum()),
        "levels_N_r_without_mass": int(((n_r > 0.0) & (q_r < CLEAR_Q)).sum()),
        "max_N_r_without_mass": float(np.where(q_r < CLEAR_Q, n_r, 0.0).max()),
        "levels_above_psd_ceiling_x10": int((n_r > 10.0 * n_hi + 1.0).sum()),
        "levels_with_mass_above_psd_ceiling_x10": int(
            ((q_r >= CLEAR_Q) & (n_r > 10.0 * n_hi)).sum()),
        "n_levels": int(n_r.size),
    }


def audit_checkpoint(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    conv = str(z["number_convention"]) if "number_convention" in z else "missing"
    if conv != "per_mass":
        raise ValueError(f"{path}: number_convention={conv!r}, expected 'per_mass'")
    out = audit(np.asarray(z["trc_N_r"], float), np.asarray(z["trc_q_r"], float),
                MorrisonConfig().lamr_max)
    out["checkpoint"] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="+")
    for p in ap.parse_args(argv).checkpoints:
        print(json.dumps(audit_checkpoint(p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
