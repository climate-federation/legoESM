"""Run a land-surface evaluation recipe and emit a scorecard.

This is the single command-line entry point for the config-driven land
evaluation pipeline (``docs/land/evaluation_pipeline.md``).  It loads a
recipe YAML (``config/land_eval/*.yaml``), scores each model run against its
references, prints per-variable metric tables, and writes a machine-readable
``scorecard.json``.

Usage
-----
    JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/eval_land.py \
        --recipe config/land_eval/chats7_parity.yaml \
        [--output-dir validation_output] [--list-metrics]

The scoring core is pure numpy (no JAX needed); ``JAX_ENABLE_X64`` is only
relevant if the recipe first invokes a model run (not done here — model runs
go through ``scripts/run/run_chats7_offline.py`` /
``scripts/run/run_fluxnet_offline.py``, which the recipe references by
output directory).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_pkg_root = REPO_ROOT / "packages"
_paths = [REPO_ROOT / "src"]
_paths += [
    p for p in sorted(_pkg_root.iterdir())
    if p.is_dir() and (p / "legoesm").exists()
]
for _p in _paths:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from legoesm.land.evaluation.metrics import METRIC_REGISTRY, SCORE_METRICS  # noqa: E402
from legoesm.land.evaluation.recipe import load_recipe, run_recipe  # noqa: E402
from legoesm.land.evaluation.scorecard import Scorecard, write_scorecard_json  # noqa: E402


def _print_scorecard_table(card: Scorecard) -> None:
    """Print one scorecard as a per-variable metric table."""
    print(f"\n{'=' * 92}")
    print(f"  {card.case}  |  {card.model}  vs  {card.reference}")
    print(f"{'=' * 92}")
    header = f"  {'Variable':<26} {'N':>6} {'score':>7}"
    metric_cols: list[str] = []
    for v in card.variables:
        for mn in v.metrics:
            if mn not in metric_cols:
                metric_cols.append(mn)
    for mn in metric_cols:
        header += f" {mn:>11}"
    print(header)
    print(f"  {'-' * 90}")
    for v in card.variables:
        if v.n == 0:
            print(f"  {v.label:<26}  (no valid data)")
            continue
        line = f"  {v.label:<26} {v.n:>6} {v.score:>7.3f}"
        for mn in metric_cols:
            val = v.metrics.get(mn)
            line += f" {val:>11.4f}" if val is not None else f" {'-':>11}"
        print(line)
    gs = card.group_scores()
    gs_str = ", ".join(f"{g}={s:.3f}" for g, s in gs.items())
    print(f"  {'-' * 90}")
    print(f"  overall_score = {card.overall_score():.3f}   groups: {gs_str}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a land evaluation recipe -> scorecard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--recipe", type=pathlib.Path,
        help="Path to a recipe YAML (e.g. config/land_eval/chats7_parity.yaml).",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Override the recipe's output_dir for scorecard.json.",
    )
    p.add_argument(
        "--list-metrics", action="store_true",
        help="Print the available metric names and exit.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_metrics:
        print("Available metrics (score-type marked *):")
        for name in sorted(METRIC_REGISTRY):
            star = " *" if name in SCORE_METRICS else ""
            print(f"  {name}{star}")
        return 0

    if not args.recipe:
        print("error: --recipe is required (or use --list-metrics)",
              file=sys.stderr)
        return 2

    recipe = load_recipe(args.recipe)
    print(f"[eval-land] recipe '{recipe.name}': {len(recipe.cases)} case(s)")
    cards = run_recipe(recipe, REPO_ROOT)

    for card in cards:
        _print_scorecard_table(card)

    out_dir = args.output_dir or (REPO_ROOT / recipe.output_dir)
    out_path = pathlib.Path(out_dir) / f"scorecard_{recipe.name}.json"
    write_scorecard_json(cards, out_path)
    print(f"\n[eval-land] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
