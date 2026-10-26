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
import math
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

# Above this the panels wrap onto a second row. One row of nine at 4 inches
# each is a 36-inch strip nobody can read side by side; the five-case campaign
# that set the format was six panels wide.
_MAX_PANELS_PER_ROW = 6


def _grid(n_panels: int) -> tuple[int, int]:
    """(nrows, ncols) for ``n_panels``, wrapping onto two rows when wide."""
    if n_panels <= _MAX_PANELS_PER_ROW:
        return 1, n_panels
    return 2, math.ceil(n_panels / 2)


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

    # Panels are the JOINT score plus one per CASE, read from
    # per_case_default / per_case_tuned. Not one per scored variable: the
    # multi-case driver scores different variables in different cases
    # (ekman scores u,v; cbl scores theta only), so a per-variable panel is
    # ragged and half empty. The case is also the unit the joint averages
    # over, which is what "one parameter set across all cases" is about.
    cases = list(payload["protocol"]["cases"])
    panels = [("combined", None, None)] + [
        (c, "per_case_default", "per_case_tuned") for c in cases
    ]
    nrows, ncols = _grid(len(panels))
    fig, axgrid = plt.subplots(nrows, ncols,
                               figsize=(4.0 * ncols, 5.6 * nrows))
    axes = np.atleast_1d(axgrid).ravel()
    # A wrapped grid can have spare cells; an empty box with axes drawn on it
    # reads as a panel whose data went missing.
    for spare in axes[len(panels):]:
        spare.set_visible(False)
    fig.patch.set_facecolor(SURFACE)

    medians = {}
    for i, (ax, (label, kd, kt)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(SURFACE)
        if kd is None:
            before = [s["score_default"] for s in arms]
            after = [s["score_tuned"] for s in arms]
            title = "combined score"
        else:
            before = [(s.get(kd) or {}).get(label, np.nan) for s in arms]
            after = [(s.get(kt) or {}).get(label, np.nan) for s in arms]
            title = label
        # The y axis is the same quantity in every panel, so it is labelled
        # once per ROW rather than once per figure -- on two rows, labelling
        # only the first panel leaves the whole second row unlabelled.
        ylabel = ("normalized profile RMSE  (lower = better)"
                  if i % ncols == 0 else None)
        medians[label] = _panel(ax, names, before, after, title, ylabel)

    mb, ma = medians["combined"]
    # Each case carries its OWN LES averaging window; there is no single
    # top-level window to quote.
    wins = "  ".join(f"{c['case']} {c['window_hours'][0]:.1f}-"
                     f"{c['window_hours'][1]:.1f} h"
                     for c in payload["cases"])
    # The header band is a FRACTION of the figure, so it has to shrink as the
    # figure grows a second row or the title floats a long way above the plots.
    y_title = 0.975 if nrows == 1 else 0.987
    y_sub = 0.925 if nrows == 1 else 0.962
    top = 0.90 if nrows == 1 else 0.945
    fig.suptitle(
        f"SCM turbulence closures before and after joint tuning against LES "
        f"—  median {mb:.4f} → {ma:.4f} "
        f"({100.0 * (ma - mb) / mb:+.1f}%)",
        fontsize=12.5, color=INK, y=y_title,
    )
    fig.text(0.5, y_sub,
             f"{len(arms)} schemes x {len(cases)} cases, ONE parameter set per "
             f"scheme across all cases; segments join a scheme's own pair.  "
             f"LES time-mean windows: {wins} (same window both sides).",
             ha="center", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, top))
    out = args.out or (args.indir / "tuning_before_after.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    for k, (b, a) in medians.items():
        print(f"  {k:9s} median {b:.5f} -> {a:.5f}  ({100*(a-b)/b:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
