"""#1455: score the START-MODE A/B -- bridged leap-frog vs legacy forward-Euler.

Two 90-day twins at ONE commit differing in ONE variable (the twin's start
mode), plus one 1e-14-perturbed member of each arm so the within-arm noise
floor is MEASURED AT THE SAME COMMIT rather than transferred from an older
ensemble.  Pre-registration: ``PREREG_startmode_ab90.md`` (written and
committed before either arm ran).

Everything numeric is imported, not re-derived: the five acceptance-gate
metrics come from ``acceptance_gate_90d`` (which itself composes
``acc_thermal_wind``), and the sixth -- the circumpolar CHANNEL-BAND transport
-- is ``acc_thermal_wind.acc_band`` under the recorded ACC reducer (median over
longitudes 2..-2), so the band number is the recorded protocol restricted to
the band and not a second definition of the same thing.

Usage
-----
  startmode_ab_score.py LABEL=path.npz [LABEL=path.npz ...] [--level 5]

Prints, for every arm: its stamps (start mode, ladder, dtype, ladder hash),
its six metrics, and |metric - NEMO day 90|.  Then the pairwise |arm - arm|
table, which is where the registered decision is read: the measured within-arm
floor (a member against its own 1e-14 twin) sits next to the across-arm
difference, so "bigger than chance" is a comparison between two measured
numbers rather than a transfer.

This script issues NO verdict.  It prints numbers and the registered bars next
to them; the interpretation belongs in the report, per the campaign rule that a
probe must never print its own conclusion.
"""
import argparse
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_thermal_wind as A          # noqa: E402
import acceptance_gate_90d as G       # noqa: E402
import kamm_twin_90d as _twin         # noqa: E402

KEYS = G.KEYS + ("accband",)
LABELS = dict(G.LABELS, accband="channel-band transport [Sv]")
FLOORS = dict(G.FLOORS)
# The band transport has no recorded 1e-14 floor of its own.  It is the SAME
# integral as ACC restricted to the channel rows, so the ACC floor is the only
# defensible transfer -- and it is flagged as a transfer everywhere it is used
# rather than quietly reused as if it had been measured for this metric.
FLOORS["accband"] = G.FLOORS["acc"]
ACCBAND_FLOOR_IS_A_TRANSFER = True


def all_metrics(st, wet):
    """The gate's five, plus the channel-band transport under the SAME reducer
    the recorded ACC metric uses (median over longitudes 2..-2)."""
    m = G.metrics(st, wet)
    band = A.acc_band(st["u"], A.umask)
    m["accband"] = float(np.median(band[2:-2]))
    return m


def stamps_of(path):
    with np.load(path) as d:
        return {
            "start": _twin.start_mode_of(d) or "UNSTAMPED",
            "ladder": (str(d["nemo_ladder_mode"])
                       if "nemo_ladder_mode" in d.files else "UNSTAMPED"),
            "dtype": (str(d["control_dtype"])
                      if "control_dtype" in d.files else "UNSTAMPED"),
            "hash": (str(d["vertical_ladder_sha256"])[:16]
                     if "vertical_ladder_sha256" in d.files else "UNSTAMPED"),
        }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("arms", nargs="+", help="LABEL=path.npz")
    p.add_argument("--level", type=int, default=5,
                   help="gate level whose threshold is printed (default 5)")
    args = p.parse_args(argv)

    arms = {}
    for spec in args.arms:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=path.npz, got {spec!r}")
        label, path = spec.split("=", 1)
        arms[label] = path

    # every arm's grid/precision/start is printed BEFORE any number, so a
    # mislabelled file cannot be read as a result
    print("ARMS")
    for label, path in arms.items():
        s = stamps_of(path)
        print(f"  {label:<10} start={s['start']:<10} ladder={s['ladder']:<6} "
              f"dtype={s['dtype']:<8} ladder_sha={s['hash']}  {path}")
    hashes = {stamps_of(p)["hash"] for p in arms.values()}
    if len(hashes) != 1:
        raise SystemExit(f"arms stand on DIFFERENT vertical ladders {hashes} -- "
                         "this is not a one-variable comparison")
    print()

    loaded = {label: G.load_candidate(path) for label, path in arms.items()}
    wet = A.tmask & (next(iter(loaded.values()))["land_mask"] > 0.5)[:, :, None]
    for label, st in loaded.items():
        w = A.tmask & (st["land_mask"] > 0.5)[:, :, None]
        if not np.array_equal(w, wet):
            raise SystemExit(f"{label}: different wet mask -- not comparable")
    G.instrument_self_checks(wet)
    nemo_m = all_metrics(G.load_nemo_day90(), wet)
    mets = {label: all_metrics(st, wet) for label, st in loaded.items()}

    print("METRICS, and |arm - NEMO day 90|")
    hdr = f"{'metric':<34}{'NEMO d90':>13}"
    for label in arms:
        hdr += f"{label:>13}{label + ' err':>14}"
    print(hdr)
    for k in KEYS:
        row = f"{LABELS[k]:<34}{nemo_m[k]:>13.6f}"
        for label in arms:
            row += f"{mets[label][k]:>13.6f}{abs(mets[label][k] - nemo_m[k]):>14.3e}"
        print(row)

    print("\nPAIRWISE |arm - arm|, against the registered bars")
    names = list(arms)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    hdr = f"{'metric':<34}" + "".join(f"{a + '-' + b:>16}" for a, b in pairs)
    print(hdr + f"{'1x floor':>12}{f'{args.level}x gate':>12}")
    for k in KEYS:
        row = f"{LABELS[k]:<34}"
        for a, b in pairs:
            row += f"{abs(mets[a][k] - mets[b][k]):>16.3e}"
        note = " (floor TRANSFERRED from ACC)" if k == "accband" else ""
        print(row + f"{FLOORS[k]:>12.3e}{args.level * FLOORS[k]:>12.3e}{note}")

    print("\nDIRECTION, per metric: which arm sits CLOSER to NEMO at day 90")
    for k in KEYS:
        errs = {label: abs(mets[label][k] - nemo_m[k]) for label in arms}
        best = min(errs, key=errs.get)
        print(f"  {LABELS[k]:<34} closest={best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
