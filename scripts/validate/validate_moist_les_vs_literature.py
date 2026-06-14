"""Compare the moist true-LES runs (BOMEX / DYCOMS-II RF01 / RICO) to the
published GCSS/GEWEX intercomparison targets.

Reads the per-case ``<out>/<case>_les_final.npz`` (+ the recorded
``profiles/prof_*.npz`` time series) written by ``run_{bomex,dycoms,rico}_les``
and prints a PASS/CHECK table of the headline metrics against the literature
reference RANGES (an LES that lands in the published ensemble spread is
"faithful"; a point value is not expected to match exactly — the cases
themselves have 10–20 % model spread).

References
----------
* BOMEX   — Siebesma et al. 2003, JAS 60, 1201 (table 2 / fig. 5).
* DYCOMS  — Stevens et al. 2005, MWR 133, 1443 (RF01; table 1).
* RICO    — van Zanten et al. 2011, JAMES 3, M06001 (fig. 4 / table 3).

Usage
-----
    python scripts/validate/validate_moist_les_vs_literature.py \\
        results/les_bomex results/les_dycoms results/les_rico
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# case → {metric: (low, high, units, source)}; ranges span the published spread.
_TARGETS = {
    "bomex": {
        "cloud_cover": (0.07, 0.18, "frac", "Siebesma'03 ~0.10-0.15"),
        "lwp": (3.0, 12.0, "g/m^2", "Siebesma'03 ~5-10"),
    },
    "dycoms": {
        "lwp": (40.0, 90.0, "g/m^2", "Stevens'05 ~50-80"),
        "cloud_cover": (0.95, 1.0, "frac", "Stevens'05 ~1.0 overcast"),
        "zi": (820.0, 880.0, "m", "Stevens'05 840-870"),
    },
    "rico": {
        "cloud_cover": (0.08, 0.25, "frac", "vanZanten'11 ~0.10-0.20"),
        "lwp": (5.0, 30.0, "g/m^2", "vanZanten'11 ~10-20"),
        # accumulated precip over the 24 h run (~0.3 mm/day mean ± spread)
        "precip_accum_mm": (0.05, 1.5, "mm/24h", "vanZanten'11 ~0.3 mm/day"),
    },
}


def _case_of(out_dir: Path) -> str:
    for c in _TARGETS:
        if c in out_dir.name.lower():
            return c
    return out_dir.name.lower()


def _load_final(out_dir: Path):
    hits = list(out_dir.glob("*_les_final.npz"))
    if not hits:
        return None
    return np.load(hits[0])


def report(out_dir: Path) -> bool:
    case = _case_of(out_dir)
    tgt = _TARGETS.get(case)
    d = _load_final(out_dir)
    print(f"\n=== {case.upper()}  ({out_dir}) ===")
    if d is None:
        print("  [no *_les_final.npz — run not finished?]")
        return False
    if tgt is None:
        print("  [no reference targets registered]")
        return False
    ok_all = True
    for metric, (lo, hi, units, src) in tgt.items():
        if metric not in d.files:
            print(f"  {metric:16s}  MISSING in npz")
            ok_all = False
            continue
        val = float(np.asarray(d[metric]))
        ok = lo <= val <= hi
        ok_all &= ok
        flag = "PASS" if ok else "CHECK"
        print(f"  {metric:16s} = {val:9.3f} {units:8s} "
              f"[{lo:g}, {hi:g}]  {flag}   ({src})")
    return ok_all


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("usage: validate_moist_les_vs_literature.py <out_dir> [...]")
        return 2
    all_ok = True
    for a in args:
        all_ok &= report(Path(a))
    print("\n" + ("ALL METRICS IN RANGE" if all_ok
                  else "SOME METRICS OUT OF RANGE — inspect profiles/figures"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
