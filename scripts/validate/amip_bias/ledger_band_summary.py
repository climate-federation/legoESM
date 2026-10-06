#!/usr/bin/env python3
"""Band temperature budget [K/day] by process from ledger_band_replay.py outputs.

Each <dir>/budget_ledger_columns.npz holds per-column mean energy rates [W/m2]
(c_p * band-weighted column integral of each process's dT/dt).  Converted to
K/day with the SAME band mass the ledger integrates over (p_s * dsigma with
dsigma = d(A+B), weight = fractional overlap of the A+B band), so the
ledger's known hybrid-mass approximation (#1400) changes the relative layer
weights inside the band, not the K/day scale.  p_s is the mean of the run's
checkpoints at the start and end of the ledger window (the ledger's own
step-mean p_s is not stored; a ~1 % p_s change gives a ~1 % scale error).

Regions use the land fraction and areas of a same-mesh surface capture.

Usage: ledger_band_summary.py <capture.npz> <run_dir> <start_day> LABEL=DIR LO HI [LABEL=DIR LO HI ...]
"""
import sys

import numpy as np

from legoesm import constants

DT_S = 112.5   # dynamics step of the production MPAS res6 deck [s]; checked against the window

REGIONS = {"45-70N land": lambda la, fl: (la >= 45) & (la <= 70) & (fl > 0.5),
           "45-70N ocean": lambda la, fl: (la >= 45) & (la <= 70) & (fl < 0.05),
           "tropics": lambda la, fl: np.abs(la) <= 30}


def band_weight(eta_half, lo, hi):
    top, bot = eta_half[:-1], eta_half[1:]
    ov = np.clip(np.minimum(bot, hi) - np.maximum(top, lo), 0.0, None)
    return ov / np.maximum(bot - top, 1e-30)


def main(a):
    cap, run_dir, start = a[0], a[1], int(a[2])
    c = np.load(cap, allow_pickle=True)
    lat, fl, area = c["lat"], c["f_land"], c["area"]
    rest = a[3:]
    if len(rest) % 3:
        raise SystemExit(__doc__)
    for i in range(0, len(rest), 3):
        label, d = rest[i].split("=", 1)
        lo, hi = float(rest[i + 1]), float(rest[i + 2])
        z = np.load(f"{d}/budget_ledger_columns.npz", allow_pickle=True)
        rates, procs = z["ledger_rates"], [str(p) for p in z["processes"]]
        n_steps, end_day = int(z["n_steps"]), float(z["day"])
        if rates.shape[0] != lat.size or not np.isfinite(rates).all():
            raise SystemExit(f"{d}: {rates.shape} columns vs {lat.size}, or non-finite rates")
        if "area_cell" in z.files and not np.allclose(
                np.asarray(z["area_cell"]) / np.asarray(z["area_cell"]).sum(), area / area.sum(), rtol=1e-6):
            raise SystemExit(f"{d}: ledger column order/areas differ from the capture's")
        if abs(n_steps * DT_S - (end_day - start) * 86400.0) > DT_S:
            raise SystemExit(f"{d}: {n_steps} steps of {DT_S} s do not span days {start}-{end_day:g}")
        ck0 = np.load(f"{run_dir}/checkpoint_day_{start:04d}.npz", allow_pickle=True)
        ck1 = np.load(f"{run_dir}/checkpoint_day_{int(round(end_day)):04d}.npz", allow_pickle=True)
        vg = np.asarray(ck0["meta_vgrid"], dtype=np.float64)          # (2, nlev+1): A, B
        eta = vg[0] + vg[1]
        w = band_weight(eta, lo, hi)
        ps = 0.5 * (np.asarray(ck0["p_s"], dtype=np.float64) + np.asarray(ck1["p_s"], dtype=np.float64))
        mass = ps * (w * np.diff(eta)).sum() / constants.g             # kg/m2 in the band
        kday = rates[:, :, 1] / (constants.c_pd * mass[:, None]) * 86400.0
        print(f"\n{label} band A+B {lo:.2f}-{hi:.2f}: ledger window ends day {end_day:g}, "
              f"{n_steps} steps ({n_steps * DT_S / 86400:.2f} days)")
        print("  region         " + " ".join(f"{p[:9]:>9s}" for p in procs) + "      sum")
        for name, sel in REGIONS.items():
            m = sel(lat, fl)
            ww = area * m
            row = (kday * ww[:, None]).sum(0) / ww.sum()
            print(f"  {name:14s} " + " ".join(f"{x:+9.3f}" for x in row) + f" {row.sum():+8.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 7:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
