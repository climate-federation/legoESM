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


def _ranking(rows: dict[str, dict], subset: set[str] | None = None) -> list[str]:
    """Schemes ordered by the objective they were tuned under, best first.

    Restricted to ``subset`` BEFORE ranking, and to rows with a FINITE score.
    Ranking the full table and then filtering leaves gaps, so a scheme present
    in only one seed silently shifts every rank below it; and a crashed run's
    +inf score would otherwise be ordered by scheme NAME and take part in the
    swap count, manufacturing churn out of two failures.
    """
    names = [s for s in rows if subset is None or s in subset]
    finite = [s for s in names
              if math.isfinite(_ff(rows[s].get("tuned_objective")))]
    return sorted(finite, key=lambda s: _ff(rows[s].get("tuned_objective")))


def compare_seeds(rows_a: dict, rows_b: dict, *, seed_a: str, seed_b: str) -> dict:
    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    shared = set(rows_a) & set(rows_b)
    # A scheme that produced a non-finite score in EITHER seed is not ranked:
    # it is a failure, reported as one, not a rank.
    common = sorted(
        s for s in shared
        if math.isfinite(_ff(rows_a[s].get("tuned_objective")))
        and math.isfinite(_ff(rows_b[s].get("tuned_objective"))))
    non_finite = sorted(shared - set(common))
    if not common:
        raise ValueError(
            "the two seeds share no scheme with a finite score in both "
            f"(shared={sorted(shared)}, non-finite={non_finite})")
    subset = set(common)
    rank_a = {s: i for i, s in enumerate(_ranking(rows_a, subset))}
    rank_b = {s: i for i, s in enumerate(_ranking(rows_b, subset))}

    per_scheme = {}
    for scheme in common:
        a = _ff(rows_a[scheme].get("tuned_objective"))
        b = _ff(rows_b[scheme].get("tuned_objective"))
        # The noise floor of a DIFFERENCE of two independent scores is the
        # quadrature sum of their own spreads, not the larger of them; and a
        # missing (NaN) spread must be ignored rather than poisoning the
        # comparison, because `max(nan, x)` is NaN.
        floors = [f for f in (
            _ff(rows_a[scheme].get("tuned_thermo_window_std")),
            _ff(rows_b[scheme].get("tuned_thermo_window_std")),
        ) if math.isfinite(f)]
        noise = math.sqrt(sum(f * f for f in floors)) if floors else float("nan")
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
    order_a = _ranking(rows_a, subset)
    order_b = _ranking(rows_b, subset)
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
        # Reported, never silently dropped: a scheme that ran under one seed
        # only, or crashed under either, is itself a disagreement.
        "schemes_only_in_seed_" + seed_a: only_a,
        "schemes_only_in_seed_" + seed_b: only_b,
        "schemes_non_finite_in_one_or_both": non_finite,
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
        incomparable: list[str] = []
        for name in sorted(set(a[scheme]) & set(b[scheme])):
            pa, pb = a[scheme][name], b[scheme][name]
            lo, hi = float(pa["bounds"][0]), float(pa["bounds"][1])
            # Differing bounds between the two runs means the two tuned values
            # are not on a common scale, so a range fraction would be a number
            # with no meaning.  Skip it rather than compute it.
            if (float(pb["bounds"][0]), float(pb["bounds"][1])) != (lo, hi):
                incomparable.append(name)
                continue
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
                "incomparable_bounds": incomparable,
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
