#!/usr/bin/env python
"""#1455 PRE-REGISTERED A/B: does the twin harness's seasonal-clock offset own
the southern-band surface warm bias?

ONE VARIABLE.  Both arms are the same recipe, the same bridged NEMO restart,
the same 30 days, the same metric.  They differ only in the ``t_seconds``
handed to the DINO analytic surface forcing:

  armA  ``DINO_TWIN_SEASONAL_KT0=0``        t = (k+1)*dt        -- the harness
        as every recorded result was produced (seasonal day 0.03 .. 30)
  armB  ``DINO_TWIN_SEASONAL_KT0=restart``  t = (5760+k+1)*dt   -- NEMO's own
        clock (usrdef_sbc.F90:536, ztime = REAL(kt)*rn_Dt; seasonal day
        180.03 .. 210), matching the NEMO run the metric compares against

METRIC: unchanged, imported -- ``acceptance_gate_90d.surface_sigma_south``
MEAN, read through ``sigma_mean_gap_decompose.time_series`` so the arms are
scored by exactly the instrument that produced the published day-30 value.

PRE-REGISTRATION (written and committed BEFORE either arm ran)
--------------------------------------------------------------
Published reference, commit a5183778c: the day-30 gap is -0.0132 kg/m3 and the
day-90 gap -0.0230, against a 9.5e-05 noise floor.

  REPRODUCTION CHECK   armA day-30 gap must land within +-20% of -0.0132.
                       Outside that, the two arms are not the recorded
                       configuration and NEITHER number below is readable.

  CONFIRM   |gap_B| <= 0.30 * |gap_A|     (>=70% of the violation removed)
  REFUTE    |gap_B| >= 0.85 * |gap_A|     (<=15% removed)
  PARTIAL   anything between

The 0.30 bound is set by the offline arithmetic in
``twin_forcing_clock_alignment.py``: the T* mis-phasing alone predicts a
day-30 warm anomaly 2.3x LARGER than the one measured, so if it is the owner,
removing it must collapse the gap rather than trim it.

Usage
-----
    python twin_seasonal_clock_ab.py armA=A.npz armB=B.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS_DIR, os.path.dirname(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DAY = 30
PUBLISHED_DAY30_GAP = -0.0132      # kg/m3, commit a5183778c
REPRO_TOL = 0.20                   # armA must reproduce it to +-20%
CONFIRM_FRAC = 0.30
REFUTE_FRAC = 0.85


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit(__doc__.strip().splitlines()[-1])

    import acc_metric_reconciliation as R
    import acc_thermal_wind as A
    import acceptance_gate_90d as G
    import sigma_mean_gap_decompose as D

    print(f"repo HEAD = {R.head_sha()}   fp64 = {np.zeros(1).dtype}")
    G.instrument_self_checks(A.tmask)

    gaps = {}
    for spec in argv:
        label, path = spec.split("=", 1)
        print(f"\n=== PROVENANCE ===\n  {label:<10}{R.stamp(path)}")
        series = D.time_series(path, label)
        if DAY not in series:
            raise SystemExit(f"{label}: no day-{DAY} pair in {path}")
        lmean, nmean, _, _ = series[DAY]
        gaps[label] = lmean - nmean

    if set(gaps) != {"armA", "armB"}:
        raise SystemExit(f"expected labels armA and armB, got {sorted(gaps)}")
    if not all(np.isfinite(v) for v in gaps.values()):
        raise SystemExit(f"non-finite gap: {gaps}")

    ga, gb = gaps["armA"], gaps["armB"]
    repro = abs(ga - PUBLISHED_DAY30_GAP) / abs(PUBLISHED_DAY30_GAP)
    frac = abs(gb) / abs(ga) if ga != 0.0 else np.inf

    print(f"\n=== PRE-REGISTERED READOUT (day {DAY}) ===")
    print(f"  armA  (harness clock, seasonal day 0.03..30)   gap {ga:+.6f} kg/m3")
    print(f"  armB  (NEMO clock,    seasonal day 180.03..210) gap {gb:+.6f} kg/m3")
    print(f"  gate floor {G.FLOORS['smean']:g}   armA {abs(ga)/G.FLOORS['smean']:.0f}x  "
          f"armB {abs(gb)/G.FLOORS['smean']:.0f}x")
    print(f"  reproduction check: armA vs published {PUBLISHED_DAY30_GAP:+.4f} "
          f"-> {100*repro:.1f}% off (pre-registered tolerance {100*REPRO_TOL:.0f}%)"
          f"  [{'ok' if repro <= REPRO_TOL else 'OUT OF TOLERANCE'}]")
    band = ("CONFIRM" if frac <= CONFIRM_FRAC
            else "REFUTE" if frac >= REFUTE_FRAC else "PARTIAL")
    print(f"  |gap_B|/|gap_A| = {frac:.3f}   pre-registered bands: "
          f"CONFIRM <= {CONFIRM_FRAC}, REFUTE >= {REFUTE_FRAC}  -> {band}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
