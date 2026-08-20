#!/usr/bin/env python
"""#1455 PRE-REGISTERED A/B: does the twin harness's seasonal-clock offset own
the southern-band surface warm bias?

ONE VARIABLE.  Both arms are the same recipe, the same bridged NEMO restart,
the same 30 days, the same metric.  They differ only in the ``t_seconds``
handed to the DINO analytic surface forcing:

  armA  ``DINO_TWIN_SEASONAL_KT0=0``        t = (k+1)*dt        -- the harness
        as every result recorded BEFORE 2026-08-20 was produced (seasonal day
        0.03 .. 30). This is no longer the harness default: armB's clock is,
        so armA now REQUIRES the env var to be set explicitly.
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
# The one variable under test, read from the ARTIFACT (kamm_twin_90d.py stamps
# ``seasonal_t0_seconds``) rather than trusted from the filename label.
EXPECTED_T0_SEC = {"armA": 0.0, "armB": 180.0 * 86400.0}


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

    gaps, day0 = {}, {}
    for spec in argv:
        label, path = spec.split("=", 1)
        print(f"\n=== PROVENANCE ===\n  {label:<10}{R.stamp(path)}")
        z = np.load(path)
        # The arm is identified by what it RAN, not by what it is called.
        if "stable" not in z or not bool(z["stable"]):
            raise SystemExit(
                f"{label}: twin is not marked stable (blew_up_at_step="
                f"{z['blew_up_at_step'] if 'blew_up_at_step' in z else '?'}) "
                "-- a blown-up arm must not be scored")
        if "seasonal_t0_seconds" not in z:
            raise SystemExit(
                f"{label}: {path} predates the seasonal_t0_seconds stamp, so the "
                "variable under test cannot be read from the artifact; re-run the "
                "arm with the current kamm_twin_90d.py")
        t0 = float(z["seasonal_t0_seconds"])
        want = EXPECTED_T0_SEC.get(label)
        if want is None:
            raise SystemExit(f"unknown arm label {label!r}; expected armA or armB")
        if abs(t0 - want) > 0.5:
            raise SystemExit(
                f"{label}: seasonal_t0_seconds={t0:.0f} but this label requires "
                f"{want:.0f} -- the arms are mislabelled or the wrong npz was passed")
        series = D.time_series(path, label)
        if DAY not in series:
            raise SystemExit(f"{label}: no day-{DAY} pair in {path}")
        lmean, nmean, _, _ = series[DAY]
        gaps[label] = lmean - nmean
        day0[label] = (series[0][0] - series[0][1]) if 0 in series else np.nan

    if set(gaps) != {"armA", "armB"}:
        raise SystemExit(f"expected labels armA and armB, got {sorted(gaps)}")
    if not all(np.isfinite(v) for v in gaps.values()):
        raise SystemExit(f"non-finite gap: {gaps}")
    # Both arms must start from the SAME bridged state, or they differ in more
    # than the one variable.
    if not np.isfinite(list(day0.values())).all():
        raise SystemExit(f"missing day-0 pair: {day0}")
    if abs(day0["armA"] - day0["armB"]) > 1e-12:
        raise SystemExit(
            f"day-0 gaps differ ({day0['armA']:.3e} vs {day0['armB']:.3e}) -- the "
            "arms did not start from the same bridged state")

    ga, gb = gaps["armA"], gaps["armB"]
    repro = abs(ga - PUBLISHED_DAY30_GAP) / abs(PUBLISHED_DAY30_GAP)
    frac = abs(gb) / abs(ga) if ga != 0.0 else np.inf

    print(f"\n=== PRE-REGISTERED READOUT (day {DAY}) ===")
    print(f"  armA  (harness clock, seasonal day 0.03..30)   gap {ga:+.6f} kg/m3")
    print(f"  armB  (NEMO clock,    seasonal day 180.03..210) gap {gb:+.6f} kg/m3")
    print(f"  gate floor {G.FLOORS['smean']:g}   armA {abs(ga)/G.FLOORS['smean']:.0f}x  "
          f"armB {abs(gb)/G.FLOORS['smean']:.0f}x")
    print(f"  day-0 gap (both arms, must be identical) {day0['armA']:+.3e} kg/m3")
    print(f"  reproduction check: armA vs published {PUBLISHED_DAY30_GAP:+.4f} "
          f"-> {100*repro:.1f}% off (pre-registered tolerance {100*REPRO_TOL:.0f}%)"
          f"  [{'ok' if repro <= REPRO_TOL else 'OUT OF TOLERANCE'}]")
    if repro > REPRO_TOL:
        # The pre-registration says NEITHER number is readable in this case, so
        # the band is not computed at all -- a failure branch with no effect is
        # a gate that cannot fail.
        raise SystemExit(
            f"armA is {100*repro:.1f}% off the published day-{DAY} gap "
            f"({PUBLISHED_DAY30_GAP:+.4f}), outside the pre-registered "
            f"{100*REPRO_TOL:.0f}% tolerance: these arms are not the recorded "
            "configuration and the A/B readout is not interpretable")
    band = ("CONFIRM" if frac <= CONFIRM_FRAC
            else "REFUTE" if frac >= REFUTE_FRAC else "PARTIAL")
    print(f"  |gap_B|/|gap_A| = {frac:.3f}   pre-registered bands: "
          f"CONFIRM <= {CONFIRM_FRAC}, REFUTE >= {REFUTE_FRAC}  -> {band}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
