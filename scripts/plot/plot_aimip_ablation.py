#!/usr/bin/env python
"""Plot AIMIP greedy-ablation summary.

Reads per-variant JSONs from ``results/aimip_001/ablation/*.json``
and emits a multi-panel summary chart:

* T RMSE per variant (grouped by stage)
* T bias per variant (grouped by stage)
* Combined objective (T RMSE + |T bias|) per variant with the
  cumulative winner trajectory overlaid as a dashed line.

Output: ``results/aimip_001/aimip_ablation_summary.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Stage definitions match scripts/run_aimip_ablation_loop.sh.
_STAGES: list[tuple[str, list[str]]] = [
    ("baseline",     ["baseline"]),
    ("gwd",          ["gwd_lindzen", "gwd_hines", "gwd_rayleigh"]),
    ("convection",   ["conv_sbm", "conv_emanuel"]),
    ("turbulence",   ["turb_tke", "turb_smagorinsky"]),
    ("microphysics", ["micro_sundqvist"]),
    ("cloud",        ["cloud_sundqvist"]),
]

_STAGE_COLORS = {
    "baseline":     "tab:gray",
    "gwd":          "tab:blue",
    "convection":   "tab:orange",
    "turbulence":   "tab:green",
    "microphysics": "tab:purple",
    "cloud":        "tab:red",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/aimip_001/ablation"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/aimip_001/aimip_ablation_summary.png"),
    )
    args = parser.parse_args()

    rows = []  # (label, stage, T_rmse, T_bias, obj)
    for stage, labels in _STAGES:
        for label in labels:
            p = args.results / f"{label}.json"
            if not p.exists():
                continue
            d = json.load(p.open())
            if d.get("status") != "ok":
                continue
            T = d["metrics"]["T"]
            rows.append((label, stage, T["rmse"], T["bias"],
                         T["rmse"] + abs(T["bias"])))

    if not rows:
        raise SystemExit(f"No ok results under {args.results}")

    labels = [r[0] for r in rows]
    stages = [r[1] for r in rows]
    rmses  = [r[2] for r in rows]
    biases = [r[3] for r in rows]
    objs   = [r[4] for r in rows]
    colors = [_STAGE_COLORS[s] for s in stages]

    # Cumulative best-so-far (winners chain).
    cum_best = []
    cur_best = float("inf")
    for o in objs:
        cur_best = min(cur_best, o)
        cum_best.append(cur_best)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    x = list(range(len(labels)))

    axes[0].bar(x, rmses, color=colors)
    axes[0].set_ylabel("T RMSE @ mid-level [K]")
    axes[0].set_title("AIMIP greedy ablation — T RMSE per variant")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)
    for i, v in enumerate(rmses):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, biases, color=colors)
    axes[1].set_ylabel("T bias @ mid-level [K]")
    axes[1].set_title("AIMIP greedy ablation — T bias per variant")
    axes[1].axhline(0.0, color="black", lw=0.5)
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)
    for i, v in enumerate(biases):
        axes[1].text(i, v, f"{v:+.2f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8)

    axes[2].bar(x, objs, color=colors, label="objective")
    axes[2].plot(
        x, cum_best, color="black", marker="o", ls="--",
        lw=1.2, label="cumulative best",
    )
    axes[2].set_ylabel("T RMSE + |T bias|")
    axes[2].set_title("AIMIP greedy ablation — combined objective + winner trajectory")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.5)
    axes[2].legend(loc="upper right")
    for i, v in enumerate(objs):
        axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)

    # Stage legend swatches.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=s)
        for s, c in _STAGE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncols=len(_STAGE_COLORS),
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
