#!/usr/bin/env python
"""How much of the SCM-RCE convection ranking is physics, and how much is search.

A derivative-free search over 20-26 parameters with a few hundred evaluations
cannot be called converged: the draws sit ~0.76 of each parameter's range apart,
so the reported optimum is the best of N lottery tickets refined locally.  The
only honest statement about that is a MEASUREMENT, and the cheapest one is to
run the identical protocol under a second seed and report the disagreement:

* rank churn — how far each scheme moves between the two seeds' orderings, and
  whether any pair swaps;
* score disagreement — |objective(seed A) - objective(seed B)| per scheme,
  against that scheme's own window-to-window noise floor, so a difference
  inside the noise is not read as a search failure;
* parameter disagreement — whether the two seeds landed on the SAME parameter
  values or on distant ones with similar scores.  Distant-but-tied is
  non-identifiability, and it is invisible in a score table.

Reads only the merged CSVs and tuned-parameter JSONs the campaign already
writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _ff(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_csv(path: Path) -> dict[str, dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return {r["scheme"]: r for r in rows}


def _arm_dir(root: Path, objective: str, param_set: str, seed: str) -> Path:
    return root / f"arm_{objective}_{param_set}_seed{seed}"


def _find_csv(arm: Path) -> Path:
    candidates = sorted(arm.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no merged CSV in {arm}")
    return candidates[0]


def _ranking(rows: dict[str, dict]) -> list[str]:
    """Schemes ordered by the objective they were tuned under, best first."""
    def key(scheme: str) -> float:
        value = _ff(rows[scheme].get("tuned_objective"))
        return value if math.isfinite(value) else float("inf")
    return sorted(rows, key=key)


def compare_seeds(rows_a: dict, rows_b: dict, *, seed_a: str, seed_b: str) -> dict:
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        raise ValueError("the two seeds share no scheme")
    rank_a = {s: i for i, s in enumerate(_ranking(rows_a)) if s in common}
    rank_b = {s: i for i, s in enumerate(_ranking(rows_b)) if s in common}

    per_scheme = {}
    for scheme in common:
        a = _ff(rows_a[scheme].get("tuned_objective"))
        b = _ff(rows_b[scheme].get("tuned_objective"))
        # The noise floor to compare against is the LARGER of the two runs'
        # own window-to-window spreads: a difference inside it says nothing.
        noise = max(
            _ff(rows_a[scheme].get("tuned_thermo_window_std")),
            _ff(rows_b[scheme].get("tuned_thermo_window_std")),
        )
        delta = abs(a - b) if math.isfinite(a) and math.isfinite(b) else float("nan")
        per_scheme[scheme] = {
            f"objective_seed{seed_a}": a,
            f"objective_seed{seed_b}": b,
            "abs_difference": delta,
            "window_noise": noise,
            "difference_exceeds_noise": (
                bool(delta > noise) if math.isfinite(delta) and math.isfinite(noise)
                else None),
            "rank_shift": rank_b[scheme] - rank_a[scheme],
        }

    swaps = []
    order_a = [s for s in _ranking(rows_a) if s in common]
    order_b = [s for s in _ranking(rows_b) if s in common]
    for i, si in enumerate(order_a):
        for sj in order_a[i + 1:]:
            if order_b.index(si) > order_b.index(sj):
                swaps.append([si, sj])

    n = len(common)
    max_pairs = n * (n - 1) // 2
    return {
        "seeds": [seed_a, seed_b],
        "n_schemes": n,
        "ranking_seed_" + seed_a: order_a,
        "ranking_seed_" + seed_b: order_b,
        "max_rank_shift": max(abs(v["rank_shift"]) for v in per_scheme.values()),
        "n_pair_swaps": len(swaps),
        "pair_swap_fraction": (len(swaps) / max_pairs) if max_pairs else 0.0,
        "pair_swaps": swaps,
        "n_differences_exceeding_noise": sum(
            1 for v in per_scheme.values() if v["difference_exceeds_noise"]),
        "per_scheme": per_scheme,
    }


def compare_parameters(arm_a: Path, arm_b: Path) -> dict:
    """Fraction of each parameter's range separating the two seeds' choices.

    Two distant parameter sets with similar scores is NON-IDENTIFIABILITY: the
    data does not constrain the parameters, and quoting a tuned value as if it
    were a calibration would be wrong even though the score is real.
    """
    def _load(arm: Path) -> dict:
        path = arm / "tuned_parameters.json"
        return json.loads(path.read_text()) if path.exists() else {}

    a, b = _load(arm_a), _load(arm_b)
    out = {}
    for scheme in sorted(set(a) & set(b)):
        per_param = {}
        for name in sorted(set(a[scheme]) & set(b[scheme])):
            pa, pb = a[scheme][name], b[scheme][name]
            lo, hi = float(pa["bounds"][0]), float(pa["bounds"][1])
            span = hi - lo
            if not span > 0:
                continue
            per_param[name] = abs(float(pa["tuned"]) - float(pb["tuned"])) / span
        if per_param:
            out[scheme] = {
                "n_params": len(per_param),
                "max_range_fraction": max(per_param.values()),
                "mean_range_fraction": sum(per_param.values()) / len(per_param),
                "per_parameter_range_fraction": per_param,
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--objective", default="thermo")
    parser.add_argument("--param-set", default="physical")
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if len(args.seeds) < 2:
        raise SystemExit(
            "--seeds needs at least two: the whole point is the disagreement "
            "between independent searches.")
    seed_a, seed_b = args.seeds[0], args.seeds[1]
    arm_a = _arm_dir(args.arm_dir, args.objective, args.param_set, seed_a)
    arm_b = _arm_dir(args.arm_dir, args.objective, args.param_set, seed_b)
    rows_a = _read_csv(_find_csv(arm_a))
    rows_b = _read_csv(_find_csv(arm_b))

    result = compare_seeds(rows_a, rows_b, seed_a=seed_a, seed_b=seed_b)
    result["parameters"] = compare_parameters(arm_a, arm_b)

    print(f"seeds {seed_a} vs {seed_b}: {result['n_schemes']} schemes")
    print(f"  ranking {seed_a}: {result['ranking_seed_' + seed_a]}")
    print(f"  ranking {seed_b}: {result['ranking_seed_' + seed_b]}")
    print(f"  pair swaps: {result['n_pair_swaps']} "
          f"({result['pair_swap_fraction']:.1%} of pairs), "
          f"max rank shift {result['max_rank_shift']}")
    print(f"  score differences exceeding the window noise: "
          f"{result['n_differences_exceeding_noise']}/{result['n_schemes']}")
    for scheme, p in sorted(result["parameters"].items()):
        print(f"  {scheme:<18} tuned values differ by up to "
              f"{p['max_range_fraction']:.0%} of the range "
              f"(mean {p['mean_range_fraction']:.0%}, {p['n_params']} params)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
