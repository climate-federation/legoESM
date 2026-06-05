#!/usr/bin/env python
"""Plot the final T106 head-to-head scorecard (classical / column_nn / sfno).

Reads ``results/aimip_t106_headtohead/aimip_scorecard.json`` (written
by ``run_aimip.py`` for the head-to-head suite) and emits a 4-panel
figure with the per-variable comparison on the held-out 2017 windows:

  Panel 1: RMSE per variable (T, T_sfc, u, v, q, p_s) — bars per variant
  Panel 2: Bias per variable — bars per variant
  Panel 3: Training-loss curves — per-variant line plot
  Panel 4: Eval loss summary table embedded as text annotations

Also writes ``headtohead_scorecard.txt`` with a plain-text table for
the LaTeX / paper drop-in.

Usage
-----
::

    python scripts/plot_aimip_headtohead_scorecard.py \
        [results/aimip_t106_headtohead/aimip_scorecard.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_VARIANTS = ("classical", "column_nn", "sfno_physics")
_VARIABLES = ("T", "T_sfc", "u", "v", "q", "p_s")
_VAR_LABELS = {
    "T": "T [K]",
    "T_sfc": "T_sfc [K]",
    "u": "u [m/s]",
    "v": "v [m/s]",
    "q": "q [g/kg]",
    "p_s": "p_s [Pa]",
}
_VAR_SCALE = {  # display rescale (e.g. q from kg/kg to g/kg)
    "T": 1.0,
    "T_sfc": 1.0,
    "u": 1.0,
    "v": 1.0,
    "q": 1000.0,    # kg/kg -> g/kg for readability
    "p_s": 1.0,
}
_COLORS = {
    "classical": "#1f77b4",
    "column_nn": "#2ca02c",
    "sfno_physics": "#d62728",
}


def _safe_get(metrics: dict, *path, default=None):
    cur = metrics
    for p in path:
        if cur is None or p not in cur:
            return default
        cur = cur[p]
    return cur


def _gather(scorecard: dict) -> dict:
    """Pull per-variant arrays of RMSE / bias / train_loss for plotting."""
    out: dict = {}
    for v in _VARIANTS:
        payload = scorecard.get("variants", {}).get(v)
        if not payload:
            out[v] = None
            continue
        em = payload.get("eval_metrics", {})
        rmse = {var: _safe_get(em, "rmse", var, "mean") for var in _VARIABLES}
        bias = {var: _safe_get(em, "bias", var, "mean") for var in _VARIABLES}
        # Rescale for display.
        rmse = {var: (rmse[var] * _VAR_SCALE[var]) if rmse[var] is not None else None
                for var in _VARIABLES}
        bias = {var: (bias[var] * _VAR_SCALE[var]) if bias[var] is not None else None
                for var in _VARIABLES}
        out[v] = {
            "rmse": rmse,
            "bias": bias,
            "loss_mean": _safe_get(em, "loss", "mean"),
            "train_history": payload.get("train_loss_history", []),
        }
    return out


def _bar_per_var(ax, data: dict, key: str, title: str) -> None:
    n_var = len(_VARIABLES)
    variants = [v for v in _VARIANTS if data.get(v) is not None]
    n_var_present = len(variants)
    width = 0.8 / max(n_var_present, 1)
    x = np.arange(n_var)
    for i, v in enumerate(variants):
        vals = [data[v][key][var] if data[v][key][var] is not None else np.nan
                for var in _VARIABLES]
        ax.bar(
            x + (i - (n_var_present - 1) / 2.0) * width,
            vals, width, label=v, color=_COLORS.get(v, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_VAR_LABELS[var] for var in _VARIABLES], rotation=20, ha="right")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def _loss_curves(ax, data: dict) -> None:
    for v in _VARIANTS:
        if data.get(v) is None:
            continue
        hist = data[v]["train_history"]
        if not hist:
            continue
        ax.plot(
            range(1, len(hist) + 1), hist,
            marker="o", linewidth=1.6, color=_COLORS.get(v, None), label=v,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (composite)")
    ax.set_title("Training-loss curves")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)


def _eval_loss_panel(ax, data: dict) -> None:
    ax.axis("off")
    lines = ["Held-out eval loss (lower = better)"]
    for v in _VARIANTS:
        if data.get(v) is None:
            lines.append(f"  {v:14s}: -")
            continue
        loss = data[v]["loss_mean"]
        lines.append(f"  {v:14s}: {loss:.4f}" if loss is not None else f"  {v:14s}: -")
    lines.append("")
    lines.append("Per-variable RMSE / Bias")
    header = f"  {'var':>5s} | " + " | ".join(f"{v[:6]:>7s}" for v in _VARIANTS)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for var in _VARIABLES:
        rmse_row = f"  {var:>5s} | " + " | ".join(
            f"{data[v]['rmse'][var]:>7.3f}"
            if data.get(v) and data[v]['rmse'][var] is not None else f"{'-':>7s}"
            for v in _VARIANTS
        )
        bias_row = f"  {'    ' + var + ' bias':>5s} | " + " | ".join(
            f"{data[v]['bias'][var]:>+7.3f}"
            if data.get(v) and data[v]['bias'][var] is not None else f"{'-':>7s}"
            for v in _VARIANTS
        )
        lines.append(rmse_row)
        lines.append(bias_row)
    ax.text(
        0.0, 1.0, "\n".join(lines),
        family="monospace", fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        transform=ax.transAxes,
    )


def _write_text_table(data: dict, out_path: Path) -> None:
    rows: list[str] = []
    rows.append("# AIMIP T106 head-to-head scorecard\n")
    rows.append(f"#  variants: {', '.join(_VARIANTS)}\n")
    rows.append("\n## Eval loss (held-out)\n")
    rows.append(f"  {'variant':<14s}: loss_mean\n")
    for v in _VARIANTS:
        d = data.get(v)
        loss = d["loss_mean"] if d else None
        rows.append(f"  {v:<14s}: {loss:.6f}\n" if loss is not None else f"  {v:<14s}: -\n")
    rows.append("\n## Per-variable RMSE\n")
    rows.append(f"  {'var':>6s}  " + "  ".join(f"{v:>14s}" for v in _VARIANTS) + "\n")
    for var in _VARIABLES:
        rows.append(
            f"  {var:>6s}  "
            + "  ".join(
                f"{data[v]['rmse'][var]:>14.4f}"
                if data.get(v) and data[v]['rmse'][var] is not None
                else f"{'-':>14s}"
                for v in _VARIANTS
            ) + "\n"
        )
    rows.append("\n## Per-variable bias\n")
    rows.append(f"  {'var':>6s}  " + "  ".join(f"{v:>14s}" for v in _VARIANTS) + "\n")
    for var in _VARIABLES:
        rows.append(
            f"  {var:>6s}  "
            + "  ".join(
                f"{data[v]['bias'][var]:>+14.4f}"
                if data.get(v) and data[v]['bias'][var] is not None
                else f"{'-':>14s}"
                for v in _VARIANTS
            ) + "\n"
        )
    out_path.write_text("".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scorecard", type=Path, nargs="?",
        default=Path("results/aimip_t106_headtohead/aimip_scorecard.json"),
        help="Path to aimip_scorecard.json.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for output PNG/TXT (default: scorecard's parent).",
    )
    args = parser.parse_args()

    scorecard_path = args.scorecard.resolve()
    if not scorecard_path.exists():
        print(f"ERROR: scorecard not found: {scorecard_path}", file=sys.stderr)
        sys.exit(2)
    scorecard = json.loads(scorecard_path.read_text())

    out_dir = args.out_dir or scorecard_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _gather(scorecard)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _bar_per_var(axes[0, 0], data, "rmse", "Per-variable RMSE (held-out)")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    _bar_per_var(axes[0, 1], data, "bias", "Per-variable bias (held-out)")
    axes[0, 1].set_ylabel("bias")
    _loss_curves(axes[1, 0], data)
    _eval_loss_panel(axes[1, 1], data)

    fig.suptitle(
        "AIMIP T106 head-to-head: classical vs column_nn vs sfno_physics\n"
        "(multi-day [24,48,72]h supervision; bias + grid-CRPS + spectral-CRPS loss; RRTMGP radiation)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = out_dir / "headtohead_scorecard.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    txt_path = out_dir / "headtohead_scorecard.txt"
    _write_text_table(data, txt_path)

    print(f"Wrote {png_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
