#!/usr/bin/env python
"""Plot AIMIP intercomparison scorecard.

Reads ``aimip_scorecard.json`` produced by ``scripts/run_aimip.py``
and emits three PNGs alongside it:

* ``aimip_loss_curves.png`` — per-variant training-loss curves.
* ``aimip_rmse_per_variable.png`` — per-variable area-weighted RMSE at
  mid-level (T, u, v, p_s) for each variant.
* ``aimip_eval_loss_bar.png`` — bar chart of the held-out
  ``spectral_state_vs_carry_loss`` mean per variant.

Usage::

    python scripts/plot_aimip_results.py \\
        results/aimip_001/aimip_scorecard.json

Or omit the path to default to ``results/aimip_001/aimip_scorecard.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_VARIABLES = ("T", "u", "v", "p_s")
_VAR_LABELS = {
    "T": "T [K]",
    "u": "u [m/s]",
    "v": "v [m/s]",
    "p_s": "p_s [Pa]",
}


def _load_scorecard(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def plot_loss_curves(scorecard: dict, out_path: Path) -> None:
    """Per-variant training-loss curves on one axes."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, payload in scorecard["variants"].items():
        history = payload.get("train_loss_history", [])
        if not history:
            continue
        ax.plot(
            range(1, len(history) + 1), history,
            marker="o", linewidth=1.5, label=variant,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (combined)")
    ax.set_title("AIMIP — training-loss curves per variant")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rmse_per_variable(scorecard: dict, out_path: Path) -> None:
    """Bar chart: per-variable RMSE for each variant."""
    variants = list(scorecard["variants"].keys())
    if not variants:
        return

    n_vars = len(_VARIABLES)
    n_variants = len(variants)
    bar_width = 0.8 / n_variants
    x = list(range(n_vars))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, variant in enumerate(variants):
        rmse_payload = scorecard["variants"][variant]["eval_metrics"]["rmse"]
        values = [rmse_payload[v]["mean"] for v in _VARIABLES]
        positions = [xi + (i - (n_variants - 1) / 2) * bar_width for xi in x]
        ax.bar(positions, values, width=bar_width, label=variant)

    ax.set_xticks(x)
    ax.set_xticklabels([_VAR_LABELS[v] for v in _VARIABLES])
    ax.set_ylabel("area-weighted RMSE @ mid-level")
    ax.set_title("AIMIP — eval RMSE per variable")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eval_loss_bar(scorecard: dict, out_path: Path) -> None:
    """Bar chart: held-out spectral loss per variant."""
    variants = list(scorecard["variants"].keys())
    values = [
        scorecard["variants"][v]["eval_metrics"]["loss"]["mean"]
        for v in variants
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(variants, values, color=["tab:blue", "tab:orange", "tab:green"][:len(variants)])
    ax.set_ylabel("held-out spectral loss")
    ax.set_title("AIMIP — eval loss per variant")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scorecard",
        nargs="?",
        default="results/aimip_001/aimip_scorecard.json",
        type=Path,
        help="Path to aimip_scorecard.json (default: results/aimip_001/...)",
    )
    args = parser.parse_args()

    if not args.scorecard.exists():
        print(f"ERROR: scorecard not found: {args.scorecard}", file=sys.stderr)
        sys.exit(1)

    scorecard = _load_scorecard(args.scorecard)
    out_dir = args.scorecard.parent

    plot_loss_curves(scorecard, out_dir / "aimip_loss_curves.png")
    plot_rmse_per_variable(scorecard, out_dir / "aimip_rmse_per_variable.png")
    plot_eval_loss_bar(scorecard, out_dir / "aimip_eval_loss_bar.png")

    print(f"Wrote plots to {out_dir}")
    for name in (
        "aimip_loss_curves.png",
        "aimip_rmse_per_variable.png",
        "aimip_eval_loss_bar.png",
    ):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
