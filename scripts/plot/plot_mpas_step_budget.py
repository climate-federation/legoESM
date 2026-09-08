"""Where the MPAS step goes at 4 and at 64 GPUs — measured, not modelled.

Every bar segment is a difference between two arms that differ in ONE
switch, run interleaved with >=40 timed steps. The two device counts
carry the SAME per-device load (40,961 cells x 26 levels), so the bars
are directly comparable and the growth is the strong-scaling gap.

Arms (all committed knobs, all bit-identical or timing-only):
  full        production step
  no-comm     every halo ppermute replaced by the identity
              (LEGOESM_MPAS_HALO_NOCOMM) -> full - nocomm = communication
  no-staging  the per-round gather/concat/scatter skipped entirely
              (LEGOESM_MPAS_HALO_NOSTAGE) -> nocomm - nostage = staging
  ballast     each message repeated N times on the wire
              (LEGOESM_MPAS_HALO_BALLAST) -> the payload term alone
  no-fix-mass the one global reduction removed

Sources: 26942819 (64-GPU three-arm + ballast), 26972228 (per-rank
spread), 26972229 (mass-fix pricing at both scales), barrier_s7_4
(4-GPU three-arm).

Usage:
    python scripts/plot/plot_mpas_step_budget.py --out fig_budget.pdf
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- measured (ms) --------------------------------------------------------
# name: (4 GPUs, 64 GPUs). Same per-device load at both.
TERMS = [
    ("compute (no halo at all)",          2.715, 3.460, "#4C72B0"),
    ("halo payload on the wire",          0.080, 1.815, "#DD8452"),
    ("latency inside the exchanges",      0.055, 1.230, "#C44E52"),
    ("on-device halo packing",            0.080, 0.315, "#8172B3"),
]
FULL = (2.930, 6.820)
IDEAL = 2.000          # serial single GPU at the same per-device load
MASSFIX = (0.040, 0.345)   # inside the compute bar; the one global reduction
SPREAD = (None, 0.024)     # per-rank max-min at 64: skew refuted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fig_mpas_budget.pdf")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5,
        "axes.linewidth": 0.6, "figure.dpi": 300, "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(150 / 25.4, 88 / 25.4))

    xs = [0, 1]
    labels = ["4 GPUs\n(subdiv-7)", "64 GPUs\n(subdiv-9)"]
    bottom = [0.0, 0.0]
    for name, v4, v64, col in TERMS:
        vals = [v4, v64]
        ax.bar(xs, vals, 0.52, bottom=bottom, color=col, label=name,
               edgecolor="white", linewidth=0.7)
        for i, v in enumerate(vals):
            if v >= 0.18:
                ax.text(xs[i], bottom[i] + v / 2, f"{v:.2f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
        bottom = [bottom[i] + vals[i] for i in range(2)]

    ax.axhline(IDEAL, ls="--", lw=1.0, color="#333333")
    ax.text(1.60, IDEAL + 0.30, f"ideal {IDEAL:.2f} ms\n(same work on one "
                                f"GPU,\nno parallel overhead)",
            fontsize=7, va="bottom", ha="left", color="#333333")

    for i, tot in enumerate(FULL):
        ax.text(xs[i], tot + 0.12, f"{tot:.2f} ms", ha="center",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("time per step (ms)")
    ax.set_ylim(0, 7.9)
    ax.set_xlim(-0.55, 2.65)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=7.5, handlelength=1.2)

    ax.set_title("Same work per GPU, 16x more GPUs: the overhead is "
                 "communication latency", fontsize=9, loc="left", pad=10)
    fig.text(0.02, 0.015,
             f"All 64 GPUs finish each step within {SPREAD[1]:.3f} ms of "
             f"one another, so none of this is GPUs waiting on stragglers; "
             f"the {TERMS[2][2]:.2f} ms is latency inside the 11 "
             f"back-to-back exchanges.\nThe one global sum costs "
             f"{MASSFIX[1]:.3f} ms at 64 GPUs and {MASSFIX[0]:.3f} at 4 "
             f"(inside the compute bar). Every segment is a one-switch A/B, "
             f">=40 timed steps, arms interleaved.",
             fontsize=6.4, color="#555555", va="bottom")

    fig.subplots_adjust(left=0.10, right=0.63, top=0.88, bottom=0.22)
    fig.savefig(args.out, bbox_inches="tight")
    if args.png:
        fig.savefig(args.png, bbox_inches="tight")
    print(args.out if not args.png else f"{args.out} {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
