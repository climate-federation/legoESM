#!/usr/bin/env python
"""Run the UNMODIFIED 90-day acceptance gate at each snapshot day the twin saved.

``acceptance_gate_90d.py`` scores day 90 only, but ``kamm_twin_90d.py
--save-3d`` already snapshots days 0/30/60/90 and NEMO's ``RUN_90D_TWIN``
already dumps restarts on a 10-day cadence -- so the day-30 and day-60 tables
cost NO extra model time. This driver composes the gate's OWN loaders,
metrics, thresholds and classifier (nothing is re-derived here, and no gate
constant is touched) and prints one table per day plus the SIGNED gap next to
each metric's noise floor.

Why signed: the gate reports |diff| against a threshold, which is the right
thing for PASS/FAIL but throws away the direction. A campaign baseline needs
the sign -- "legoESM's ACC is 0.7 Sv ABOVE NEMO" and "0.7 Sv below" call for
opposite fixes.

Usage
-----
    JAX_ENABLE_X64=1 python gate90_time_series.py TWIN.npz [--level 5] \
        [--days 30 60 90]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_DIR, os.path.dirname(_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("candidate", help="kamm_twin_90d.py --save-3d npz")
    p.add_argument("--level", type=int, default=5, choices=(5, 2, 1))
    p.add_argument("--days", type=int, nargs="+", default=[30, 60, 90])
    args = p.parse_args(argv)

    import acc_thermal_wind as A
    import acceptance_gate_90d as G

    wet = A.tmask
    G.instrument_self_checks(wet)

    z = np.load(args.candidate)
    # #1455: refuse to score an artifact whose seasonal clock is unknown or
    # is the legacy relative one -- that is the confound this baseline exists
    # to remove, and a silently-scored antiphase npz would re-create it.
    if "seasonal_t0_seconds" not in z:
        raise SystemExit(
            f"{args.candidate} predates the seasonal_t0_seconds stamp; re-run "
            "the twin with the current kamm_twin_90d.py")
    t0 = float(z["seasonal_t0_seconds"])
    print(f"candidate seasonal clock: t0 = {t0:.0f} s "
          f"(= day {t0 / 86400.0:.2f} of the 360-day year)")
    if t0 == 0.0:
        raise SystemExit(
            "candidate ran on the LEGACY RELATIVE clock (seasonal_t0_seconds=0) "
            "-- it is antiphase to every NEMO baseline below and must not be "
            "scored here (#1455)")

    n_fail_total = 0
    for day in args.days:
        kt = G.KT_RESTART + day * G.STEPS_PER_DAY
        cand = G.load_candidate(args.candidate, day=day)
        nemo = G.load_nemo_day90(kt=kt)
        cm, nm = G.metrics(cand, wet), G.metrics(nemo, wet)
        rows = G.classify(cm, nm, args.level)
        print(f"\n### DAY {day}  (NEMO kt={kt})")
        if day != 90:
            # The gate's table header is hardcoded "NEMO d90"; at day 30/60
            # the column really holds NEMO's day-30/60 state (kt above).
            print(f'    [the "NEMO d90" column below is NEMO at DAY {day}, '
                  f"kt={kt}]")
        n_fail_total += G.print_gate(rows, args.level, tag=f"[day {day}] ")
        print(f"\n{'metric':<34}{'signed gap':>14}{'floor':>12}"
              f"{'gap/floor':>12}")
        for k, c, n, _d, _t, _ok in rows:
            signed = c - n
            floor = G.FLOORS[k]
            print(f"{G.LABELS[k]:<34}{signed:>+14.6e}{floor:>12.1e}"
                  f"{signed / floor:>+12.1f}")
    return 1 if n_fail_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
