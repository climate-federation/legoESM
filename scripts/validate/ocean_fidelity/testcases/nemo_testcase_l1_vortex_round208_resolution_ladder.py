#!/usr/bin/env python3
"""Round 208 / VORTEX round 21: score decision 74's resolution ladder.

Every number in the round-208 receipt that is not read straight out of a gate
report or an ``ocean.output`` is produced here, so it can be re-run against a
changed model instead of being an uncommitted probe's claim.

Four sections, in the order the receipt uses them:

``table``       the three-resolution registry table, per card, with the ratio
                of each row to its 30 km counterpart.
``scaling``     the fitted exponent ``p`` in ``e ~ r**p`` with
                ``r = 30 km / dx``, for every row well above the rounding
                floor at 30 km.  A per-step injection proportional to the
                timestep gives ``p = -1`` exactly, because ``dt ~ 1/r``.
``crossings``   every row whose AT-BAR / DEBT status differs between rungs.
``inertness``   the "0/50 moved" control, run against the certified reference
                AND -- as its own non-vacuity check -- against a registry that
                is KNOWN to differ.  A comparison that can only say "0 moved"
                proves nothing.

The gate reports are the input; this script runs no model and reads no NEMO
record.  Paths default to the round's evidence directories and are flags so a
later round can point it at its own.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LADDERS = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round208/ladders")
CERTIFIED = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round207")
# The rungs, and the refinement ratio r = 30 km / dx that names each one.
RUNGS = (("30km", 1.0), ("15km", 2.0), ("10km", 3.0))
CARDS = ("VORTEX", "VORTEX_VEC")
# Rows at or under this are rounding floor; a power law fitted through them
# would be fitting quantisation, not a trend.
FLOOR_FOR_FITTING = 1.0e-14


class LadderError(RuntimeError):
    pass


def registry(path: Path) -> dict[str, dict]:
    """``{kt<N>.before.<field> -> row}``.

    The case prefix is dropped from the key because it differs between rungs
    by construction; everything after it is the row's identity.
    """
    if not path.is_file():
        raise LadderError(f"missing gate report {path}")
    report = json.loads(path.read_text())
    rows = {}
    for step in report["steps"]:
        for row in step["rows"]:
            rows[row["name"].split(".", 1)[1]] = row
    if len(rows) != 50:
        raise LadderError(f"{path}: {len(rows)} rows, expected the 50-row "
                          "registry (--max-step 10 --continue-after-first)")
    return rows


def report_path(ladders: Path, card: str, rung: str) -> Path:
    return ladders / (f"{card}-zco.json" if rung == "30km"
                      else f"{card}-{rung}-zco.json")


def kt_of(key: str) -> int:
    return int(key.split(".")[0][2:])


def moved(a: dict[str, dict], b: dict[str, dict]) -> list[str]:
    """Rows differing in value OR status.  Raises on a key mismatch.

    The raise matters: a silent key mismatch would return an empty list and
    read as "nothing moved".
    """
    if set(a) != set(b):
        raise LadderError(f"registry keys differ: {sorted(set(a) ^ set(b))}")
    return [k for k in sorted(a, key=lambda k: (kt_of(k), k))
            if a[k]["normalized_max_abs"] != b[k]["normalized_max_abs"]
            or a[k]["status"] != b[k]["status"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", type=Path, default=LADDERS)
    ap.add_argument("--certified", type=Path, default=CERTIFIED)
    ap.add_argument("--section", default="all",
                    choices=("all", "table", "scaling", "crossings",
                             "inertness"))
    args = ap.parse_args()
    want = args.section

    for card in CARDS:
        reg = {r: registry(report_path(args.ladders, card, r))
               for r, _ in RUNGS}
        print(f"\n################ {card}")

        if want in ("all", "inertness"):
            cert = registry(args.certified / f"card_{card}-zco.json")
            claim = moved(reg["30km"], cert)
            control = moved(reg["30km"], reg["15km"])
            print(f"  inertness  30 km vs certified : {len(claim)}/50 moved")
            for k in claim:
                print(f"      MOVED {k}: {cert[k]['normalized_max_abs']!r} -> "
                      f"{reg['30km'][k]['normalized_max_abs']!r}")
            print(f"  NON-VACUITY 30 km vs 15 km    : {len(control)}/50 moved"
                  "  <- the same comparison CAN fire")
            if not control:
                raise LadderError(
                    "the inertness comparison reports no move even against a "
                    "different rung; it cannot detect one and proves nothing")

        if want in ("all", "table"):
            print(f"\n  {'row':<20}" + "".join(f"{r:>16}" for r, _ in RUNGS)
                  + f"{'15/30':>9}{'10/30':>9}")
            for k in sorted(reg["30km"], key=lambda k: (kt_of(k), k)):
                v = {r: reg[r][k]["normalized_max_abs"] for r, _ in RUNGS}
                def ratio(x, base):
                    if base:
                        return f"{x / base:>9.2f}"
                    return "    exact" if not x else "      inf"
                print(f"  {k.replace('.before.', ' '):<20}"
                      + "".join(f"{v[r]:>16.7e}" for r, _ in RUNGS)
                      + ratio(v["15km"], v["30km"])
                      + ratio(v["10km"], v["30km"]))

        if want in ("all", "scaling"):
            fitted = [k for k in reg["30km"]
                      if reg["30km"][k]["normalized_max_abs"]
                      > FLOOR_FOR_FITTING]
            print(f"\n  exponent p in e ~ r**p, r = 30km/dx "
                  f"(dt alone gives p = -1); rows > {FLOOR_FOR_FITTING:g} "
                  "at 30 km only")
            if not fitted:
                print("  (every row is at the rounding floor on this card)")
            spread = []
            for k in sorted(fitted, key=lambda k: (kt_of(k), k)):
                e0 = reg["30km"][k]["normalized_max_abs"]
                p = {r: math.log(reg[r][k]["normalized_max_abs"] / e0)
                        / math.log(ratio)
                     for r, ratio in RUNGS if ratio != 1.0}
                spread.append(abs(p["15km"] - p["10km"]))
                print(f"  {k.replace('.before.', ' '):<20}"
                      f"{p['15km']:>9.2f}{p['10km']:>9.2f}")
            if spread:
                # Quoted in the receipt instead of an eyeballed "+/- 0.1".
                print(f"  max |p(15km) - p(10km)| over {len(spread)} rows = "
                      f"{max(spread):.2f}")

        if want in ("all", "crossings"):
            print("\n  status crossings between rungs")
            found = False
            for k in sorted(reg["30km"], key=lambda k: (kt_of(k), k)):
                st = [reg[r][k]["status"] for r, _ in RUNGS]
                if len(set(st)) > 1:
                    found = True
                    print(f"  {k:<20} " + " -> ".join(f"{x:>7}" for x in st)
                          + "   " + "  ".join(
                              f"{reg[r][k]['normalized_max_abs']:.6e}"
                              for r, _ in RUNGS))
            if not found:
                print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
