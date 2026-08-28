#!/usr/bin/env python
"""How much of the tuned ranking is physics, and how much is search luck?

The pairwise reporter (``scm_rce_tq_seed_agreement``) answers that for TWO
independent searches.  With five or more it becomes a distribution rather than
a disagreement: this reports, per scheme, the SPREAD of the tuned score across
seeds next to the improvement the tuning bought, so a gain smaller than the
seed spread is visibly not a result.

It also reports how each scheme's RANK moves across seeds, because a stable
mean score with a churning rank means the schemes are tied, not ordered.

Loaders come from the pairwise module rather than being re-implemented, so the
two reporters can never disagree about how an arm directory is read.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.validate.scm_rce_tq_seed_agreement import (  # noqa: E402
    arm_dir, find_arm_csv, ranking_by_score, read_arm_csv,
)

#: Below this many seeds the word "spread" is not meaningful — two points give
#: a difference, not a distribution.  The caller is told rather than silently
#: given a degenerate answer.
MIN_SEEDS_FOR_SPREAD = 3


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def collect(root: Path, objective: str, param_set: str, seeds: list[str],
            suffix: str) -> tuple[dict, list[str]]:
    """Per-scheme prior/tuned scores for every seed that has a merged table."""
    per_seed, present = {}, []
    for seed in seeds:
        arm = arm_dir(root, objective, param_set, seed, suffix)
        try:
            rows = read_arm_csv(find_arm_csv(arm))
        except (FileNotFoundError, OSError):
            continue
        per_seed[seed] = rows
        present.append(seed)
    if not present:
        raise SystemExit(f"no merged arm tables under {root} for seeds {seeds}")
    return per_seed, present


def summarise(per_seed: dict, seeds: list[str]) -> dict:
    schemes = sorted(set.intersection(*(set(per_seed[s]) for s in seeds)))
    rankings = {s: ranking_by_score(per_seed[s], set(schemes)) for s in seeds}
    out = {}
    for name in schemes:
        tuned = [_f(per_seed[s][name].get("tuned_score")) for s in seeds]
        prior = [_f(per_seed[s][name].get("prior_score")) for s in seeds]
        ranks = [rankings[s].index(name) + 1 for s in seeds]
        best, worst = min(tuned), max(tuned)
        mean = statistics.fmean(tuned)
        sd = statistics.stdev(tuned) if len(tuned) > 1 else 0.0
        prior_mean = statistics.fmean(prior)
        out[name] = {
            "n_seeds": len(seeds),
            "prior_mean": prior_mean,
            "tuned_mean": mean, "tuned_sd": sd,
            "tuned_min": best, "tuned_max": worst,
            "tuned_spread": worst - best,
            "improvement": prior_mean - mean,
            # The comparison that decides whether a gain is real: an
            # improvement smaller than the seed-to-seed spread is search noise
            # wearing a result's clothes.
            "improvement_over_spread": (
                (prior_mean - mean) / (worst - best)
                if worst > best else float("inf")),
            "rank_best": min(ranks), "rank_worst": max(ranks),
            "rank_churn": max(ranks) - min(ranks),
            "ranks": ranks,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--objective", default="thermo")
    ap.add_argument("--param-set", default="physical")
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--arm-suffix", default="",
                    help="protocol tag (e.g. 'f0'); arms with different tags "
                         "are not comparable and must not be mixed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    per_seed, present = collect(args.arm_dir, args.objective, args.param_set,
                                args.seeds, args.arm_suffix)
    if len(present) < len(args.seeds):
        missing = [s for s in args.seeds if s not in present]
        print(f"NOTE: {len(present)} of {len(args.seeds)} seeds have merged "
              f"tables; missing {missing}")
    if len(present) < MIN_SEEDS_FOR_SPREAD:
        raise SystemExit(
            f"{len(present)} seed(s) is not a spread — need at least "
            f"{MIN_SEEDS_FOR_SPREAD}. Use the pairwise agreement reporter for "
            "two.")

    stats = summarise(per_seed, present)
    print(f"seeds: {' '.join(present)}  ({len(present)} independent searches)\n")
    print("%-16s %8s %8s %7s %8s %8s | %5s %s" % (
        "scheme", "prior", "tuned", "sd", "spread", "gain/sprd",
        "churn", "ranks"))
    for name in sorted(stats, key=lambda n: stats[n]["tuned_mean"]):
        r = stats[name]
        ratio = ("inf" if r["improvement_over_spread"] == float("inf")
                 else "%.1f" % r["improvement_over_spread"])
        print("%-16s %8.2f %8.2f %7.2f %8.2f %8s | %5d %s" % (
            name, r["prior_mean"], r["tuned_mean"], r["tuned_sd"],
            r["tuned_spread"], ratio, r["rank_churn"],
            "".join(str(x) for x in r["ranks"])))

    print("\nRead: 'spread' is max-min of the tuned score across seeds; "
          "'gain/sprd' is the tuning improvement divided by that spread.\n"
          "A scheme with gain/sprd below ~1 did not measurably improve — the "
          "search noise is as large as the gain.\n'churn' is how many places "
          "the scheme moves between seeds; a churn comparable to the field "
          "size means it has no defensible rank.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"seeds": present, "objective": args.objective,
             "param_set": args.param_set, "arm_suffix": args.arm_suffix,
             "per_scheme": stats}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
