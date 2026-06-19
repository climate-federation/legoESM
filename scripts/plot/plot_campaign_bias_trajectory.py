"""Plot the LES-informed correction campaign's bias trajectory from its output JSON.

After the (multi-day) campaign writes its corrected-config JSON
(``run_correction_campaign.py --out <file>``), this renders the done-criterion's
VISUAL CONFIRMATION — did the LES-informed coefficient updates LOWER the bias? — as
a two-panel PNG:

  1. the global combined bias per round (baseline → updated, with ACCEPTED rounds
     marked; the monotonic gate keeps only bias-lowering rounds), plus the
     campaign-START → final markers;
  2. the per-VARIABLE bias as a FINAL/BASELINE fraction (T / q_v / wind) — a bar
     below 1 means that variable's RMSE fell.

It only READS the recorded RMSE values from the output dict (no physics is
recomputed, so no model helpers are needed); a ``health.status`` annotation
surfaces a ``non_finite_bias`` / ``stalled`` run.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def extract_campaign_trajectory(output: dict) -> dict:
    """Pull the plottable arrays from a campaign output dict (no matplotlib).

    Tolerant of a partial / mock output: missing ``biases`` / ``summary`` /
    ``per_variable_bias`` yield empty lists or ``None`` rather than raising.
    """
    biases = output.get("biases", []) or []
    accepted = output.get("accepted", []) or []
    summary = output.get("summary", {}) or {}
    pv = summary.get("per_variable_bias") or {}

    def _f(x):
        # A DIVERGED round serializes its bias as JSON null (iter 245); map it to NaN so
        # matplotlib renders a GAP in the trajectory rather than crashing on float(None).
        return float("nan") if x is None else float(x)

    return {
        "round_baseline": [_f(b[0]) for b in biases],
        "round_updated": [_f(b[1]) for b in biases],
        "accepted": [bool(a) for a in accepted],
        "initial_bias": summary.get("initial_bias"),
        "final_bias": summary.get("final_bias"),
        "pv_baseline": pv.get("baseline"),
        "pv_final": pv.get("final"),
        "health": output.get("health", {}) or {},
    }


_PV_KEYS = (("T_rmse_K", "T"), ("qv_rmse_kg_kg", "q_v"), ("wind_rmse_m_s", "wind"))


def plot_campaign_bias_trajectory(output: dict, out_png: str) -> str:
    """Render the two-panel trajectory PNG; returns ``out_png``."""
    tr = extract_campaign_trajectory(output)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    rounds = list(range(len(tr["round_updated"])))
    if rounds:
        ax0.plot(rounds, tr["round_baseline"], "o--", color="0.6", label="baseline (pre-round)")
        ax0.plot(rounds, tr["round_updated"], "o-", color="C0", label="updated (post-round)")
        acc = [r for r, a in zip(rounds, tr["accepted"]) if a]
        if acc:
            ax0.plot(acc, [tr["round_updated"][r] for r in acc], "P", color="C2",
                     markersize=11, label="accepted (kept)")
    for val, lab, c in ((tr["initial_bias"], "start", "k"), (tr["final_bias"], "final", "C3")):
        if val is not None:
            ax0.axhline(float(val), ls=":", color=c, lw=1, label=f"{lab}={float(val):.4g}")
    ax0.set_xlabel("correction round")
    ax0.set_ylabel("global combined bias")
    ax0.set_title("Global bias per round")
    if ax0.get_legend_handles_labels()[0]:          # avoid an empty-legend warning on 0 rounds
        ax0.legend(fontsize=8)

    base, fin = tr["pv_baseline"], tr["pv_final"]
    if isinstance(base, dict) and isinstance(fin, dict):
        labels, fracs = [], []
        for key, lab in _PV_KEYS:
            b, f = base.get(key), fin.get(key)
            if b not in (None, 0) and f is not None:
                labels.append(lab)
                fracs.append(float(f) / float(b))
        if fracs:
            colors = ["C2" if x < 1.0 else "C3" for x in fracs]
            ax1.bar(labels, fracs, color=colors)
            ax1.axhline(1.0, color="0.4", lw=1)
            ax1.set_ylabel("final / baseline RMSE")
            ax1.set_title("Per-variable change (<1 = improved)")
    else:
        ax1.text(0.5, 0.5, "no per-variable bias\nin the output", ha="center", va="center")
        ax1.set_axis_off()

    status = (tr["health"] or {}).get("status")
    if status:
        fig.suptitle(f"campaign health: {status}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_json", help="campaign --out JSON file")
    p.add_argument("--png", default=None, help="output PNG (default: <output_json>.bias.png)")
    args = p.parse_args(argv)
    with open(args.output_json) as f:
        output = json.load(f)
    png = args.png or f"{args.output_json}.bias.png"
    plot_campaign_bias_trajectory(output, png)
    print(f"[plot] wrote campaign bias trajectory to {png}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
