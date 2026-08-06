"""Paired before/after summary of the SCM turbulence-tuning campaign.

Reads ``tuned_parameters.json`` from
``scripts/run/run_scm_les_turbulence_tuning.py`` and draws, per scored
variable plus the combined score, a box of the across-scheme distribution
before and after tuning with every scheme's own pair drawn on top and joined.

The pairing is the point. The nine schemes are measured TWICE, so the two
samples are not independent: a bare pair of boxes would hide both which scheme
is which and whether tuning moved each one individually or just shifted a few.
The connecting segments carry that, the boxes carry the population shift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots 1 and 2 (see the dataviz palette reference);
# the pair passes the lightness, chroma, CVD-separation, normal-vision and
# contrast checks against the light surface. The tritan separation sits in the
# floor band, so it is carried with secondary encoding: the two conditions also
# differ by x position and are directly labelled.
BEFORE = "#2a78d6"
AFTER = "#008300"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a9994"


def _panel(ax, names, before, after, title, ylabel, annotate_worst=True):
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)

    bp = ax.boxplot(
        [before, after], positions=[0, 1], widths=0.45, patch_artist=True,
        medianprops=dict(color=INK, lw=2.0),
        whiskerprops=dict(color=MUTED, lw=1.2),
        capprops=dict(color=MUTED, lw=1.2),
        flierprops=dict(marker="", ls="none"),   # points drawn individually
        zorder=1,
    )
    for patch, colour in zip(bp["boxes"], (BEFORE, AFTER)):
        patch.set_facecolor(colour)
        patch.set_alpha(0.16)
        patch.set_edgecolor(colour)
        patch.set_linewidth(1.6)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.055, 0.055, size=before.size)
    for i, name in enumerate(names):
        x0, x1 = 0.0 + jitter[i], 1.0 + jitter[i]
        improved = after[i] < before[i]
        ax.plot([x0, x1], [before[i], after[i]], "-",
                color=(AFTER if improved else "#e34948"),
                alpha=0.55, lw=1.4, zorder=2)
        ax.plot(x0, before[i], "o", ms=8, color=BEFORE,
                mec=SURFACE, mew=2.0, zorder=3)
        ax.plot(x1, after[i], "o", ms=8, color=AFTER,
                mec=SURFACE, mew=2.0, zorder=3)

    # Label only the extremes: a name on every point is noise.
    if annotate_worst:
        for idx, ha, dx in ((int(np.argmin(after)), "left", 0.09),
                            (int(np.argmax(after)), "left", 0.09)):
            ax.annotate(names[idx], (1.0 + jitter[idx] + dx, after[idx]),
                        fontsize=8.5, color=INK_2, va="center", ha=ha)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["default", "tuned"], fontsize=10, color=INK)
    ax.set_xlim(-0.45, 1.75)
    ax.set_title(title, fontsize=11, color=INK, pad=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
    ax.grid(axis="y", alpha=0.22, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)

    med_b, med_a = float(np.median(before)), float(np.median(after))
    return med_b, med_a


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("indir", type=Path,
                   help="results/scm_les_turbulence/<case>")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    payload = json.loads((args.indir / "tuned_parameters.json").read_text())
    arms = [s for s in payload["schemes"]
            if s.get("score_default") is not None
            and s.get("score_tuned") is not None]
    if not arms:
        raise SystemExit("no arm has both a default and a tuned score")
    arms.sort(key=lambda s: s["score_tuned"])
    names = [s["scheme"] for s in arms]

    scored = list(payload["protocol"]["scored_variables"])
    panels = [("combined", None, None)] + [
        (v, "components_default", "components_tuned") for v in scored
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 5.6))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    medians = {}
    for ax, (label, kd, kt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        if kd is None:
            before = [s["score_default"] for s in arms]
            after = [s["score_tuned"] for s in arms]
            title = "combined score"
            ylabel = "normalized profile RMSE  (lower = better)"
        else:
            before = [(s.get(kd) or {}).get(label, np.nan) for s in arms]
            after = [(s.get(kt) or {}).get(label, np.nan) for s in arms]
            title = {"theta": r"$\theta$", "qv": r"$q_v$"}.get(label, label)
            ylabel = None
        medians[label] = _panel(ax, names, before, after, title, ylabel)

    mb, ma = medians["combined"]
    win = payload["les_reference"]["window_hours"]
    fig.suptitle(
        f"{args.indir.name}: SCM turbulence closures before and after tuning "
        f"against LES  —  median {mb:.4f} → {ma:.4f} "
        f"({100.0 * (ma - mb) / mb:+.1f}%)",
        fontsize=12.5, color=INK, y=0.975,
    )
    fig.text(0.5, 0.925,
             f"{len(arms)} schemes, each measured twice; segments join a "
             f"scheme's own pair.  LES time-mean {win[0]:.2f}-{win[1]:.2f} h, "
             "same window both sides.",
             ha="center", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = args.out or (args.indir / "tuning_before_after.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    for k, (b, a) in medians.items():
        print(f"  {k:9s} median {b:.5f} -> {a:.5f}  ({100*(a-b)/b:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
