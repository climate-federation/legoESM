"""Global per-process budget table from ``budget_ledger_columns.npz`` (#1354).

The artifact holds per-column mean rates (ncol, 7, 2) over the LAST diagnostic
window: rows = LEDGER_PROCESSES, columns = (water [kg/m2/s], dry-enthalpy
energy [W/m2]).  This prints the unweighted global mean per row -- unweighted
is the ledger's own documented convention (MPAS SCVT cells are near-equal-area;
a few % on the mean), and the same reduction for every row so closure is exact
under it.

Reference values printed alongside are the PRE-REGISTERED healthy expectations
from the run card (scripts/cluster/mpas_energy_1354/ledger_3d_ginsburg.sbatch),
so the deltas are readable in place.  Numbers only, no verdicts.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=pathlib.Path,
                   help="budget_ledger_columns.npz from the run")
    p.add_argument("--hfss", type=float, default=20.5,
                   help="reference sensible heat flux [W/m2] (CMOR mean)")
    p.add_argument("--hfls", type=float, default=58.1,
                   help="reference latent heat flux [W/m2] (CMOR mean)")
    args = p.parse_args(argv)

    d = np.load(args.artifact, allow_pickle=True)
    rates = np.asarray(d["ledger_rates"], dtype=np.float64)   # (ncol, 7, 2)
    procs = [str(x) for x in d["processes"]]
    nstep = int(d["n_steps"]) if "n_steps" in d else -1
    day = float(d["day"]) if "day" in d else float("nan")

    L_v = 2.501e6  # display conversion only; ledger stores raw kg/m2/s
    g_mean = rates.mean(axis=0)                                # (7, 2)

    print(f"{args.artifact}")
    print(f"last diagnostic window ending day {day:g}, {nstep} steps, "
          f"{rates.shape[0]} columns, unweighted global mean")
    print()
    hdr = (f"{'process':>14s} {'water [mm/day]':>15s} {'L_v*water':>10s} "
           f"{'energy [W/m2]':>14s}")
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(procs):
        w, e = g_mean[i]
        print(f"{name:>14s} {w * 86400.0:15.4f} {L_v * w:10.3f} {e:14.3f}")
    tw, te = g_mean.sum(axis=0)
    print("-" * len(hdr))
    print(f"{'TOTAL':>14s} {tw * 86400.0:15.4f} {L_v * tw:10.3f} {te:14.3f}")
    print()
    print("pre-registered healthy references (run card):")
    print(f"  turbulence energy ~ +hfss = {args.hfss:+.1f}   "
          f"(~{2 * args.hfss:.0f} = Louis double-count hypothesis)")
    print(f"  turbulence L_v*water ~ +hfls = {args.hfls:+.1f}")
    print("  radiation energy ~ toa_net - sfc_net ~ -97")
    print("  convection/microphysics: energy ~ -L_v*water (heating paid by "
          "drying); excess = unpaid heating")
    print("  dynamics ~ 0 (1-3 ok); other_physics/clips: residual buckets, "
          "an unledgered creation lands THERE unnamed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
