"""Publication figure: strong scaling per grid, per precision, vs ideal.

One panel per (component, grid). Each panel plots measured ms/step against
device count on log-log axes, one line per precision, with a DASHED IDEAL
line anchored at each series' own base point (t_base * n_base / n).

Every number is a measured Levante receipt; the SOURCES table below carries
the SLURM job id for each series so a reader can trace any point. Panels
with only one precision measured say so in-panel rather than leaving the
reader to guess — no interpolation, no fabricated series.

Anchoring note: ideal lines are anchored at each series' FIRST measured
point. Where that base leg is cache-disadvantaged (a single device holding
the whole problem), the measured curve can sit BELOW ideal; that is a
property of the base leg, not superlinear parallelism, and is flagged in
the caption rather than hidden by re-anchoring.

Usage
-----
    python scripts/plot/plot_scaling_paper_figure.py --out fig_scaling.pdf
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Measured data -------------------------------------------------------
# (devices, ms/step). Job ids are the provenance for each series.
SOURCES = {
    "atm_latlon": "26450848/26453240/26449147",
    "atm_cube": "26452894/26453782",
    "atm_mpas": "26454476/26454618/26486288/26493638/26493734",
    "atm_ico_cpu": "26452579",
    "oc_latlon": "26460444-501/26460365/26493592",
    "oc_tripole": "26493837/26493648",
    "oc_mpas": "26494036",
}

PANELS = [
    dict(
        key="atm_latlon", title="lat–lon", sub="720×1440 L26 · A100 NCCL",
        series=[("float32", [(4, 7.72), (8, 5.40), (16, 3.54)])],
        note="f64 pending",
    ),
    dict(
        key="atm_cube", title="cubed-sphere", sub="C384/C768 L60 · A100 NCCL",
        series=[("float32 (C768)", [(6, 58.35), (24, 14.09)]),
                ("float32 (C384)", [(6, 15.44), (24, 8.81)])],
        note="f64 pending",
    ),
    dict(
        key="atm_mpas", title="MPAS icosahedral", sub="L8 28 km L26 · A100 NCCL",
        series=[("float32", [(2, 19.90), (4, 14.12), (8, 6.92), (16, 7.10)]),
                ("float64", [(2, 38.34), (4, 20.09), (8, 18.98)])],
        note="incl. fusion fix",
    ),
    dict(
        key="atm_ico_cpu", title="icosahedral", sub="subdiv-7 L26 · Milan CPU–MPI",
        series=[("float64", [(1, 10150.35), (2, 4929.97), (4, 2760.63),
                             (8, 1245.45), (16, 1248.49), (32, 614.30),
                             (64, 305.02)])],
        note="",
    ),
    dict(
        key="oc_latlon", title="lat–lon", sub="576×1152 L20 · A100 NCCL",
        series=[("float32", [(1, 36.45), (2, 22.48), (4, 12.84), (8, 11.04), (16, 8.71)]),
                ("float64", [(1, 64.81), (2, 41.63), (4, 22.09)]),
                ("mixed (f64 store)", [(1, 52.80), (4, 19.13)])],
        note="best arm shown",
    ),
    dict(
        key="oc_tripole", title="tripole (ORCA fold)", sub="576×1152 L20 · A100 NCCL",
        series=[("float32", [(1, 35.52), (2, 25.03), (4, 15.91)]),
                ("float64", [(1, 63.84), (2, 43.36), (4, 24.12)])],
        note="fold +1.2–3.7 %",
    ),
    dict(
        key="oc_mpas", title="MPAS Voronoi", sub="subdiv-7 164k cells · Milan CPU–MPI",
        series=[("float64", [(1, 3944.43), (2, 1615.85), (4, 720.87),
                             (8, 362.42), (16, 311.53)])],
        note="f32 pending",
    ),
]

COLORS = {"float32": "#0072B2", "float64": "#D55E00",
          "mixed (f64 store)": "#009E73",
          "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9"}
MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
           "float32 (C768)": "o", "float32 (C384)": "^"}


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # editable text in the PDF (journal requirement)
        "ps.fonttype": 42,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fig_scaling.pdf")
    ap.add_argument("--png", default=None, help="also write a PNG here")
    args = ap.parse_args()

    _style()
    ncol, nrow = 4, 2
    # Nature double-column ~180 mm
    fig, axs = plt.subplots(nrow, ncol, figsize=(180 / 25.4, 100 / 25.4))
    axs = axs.ravel()

    for i, spec in enumerate(PANELS):
        ax = axs[i]
        for label, pts in spec["series"]:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            c = COLORS.get(label, "#444444")
            ax.plot(xs, ys, MARKERS.get(label, "o") + "-", color=c,
                    label=label, markerfacecolor="white",
                    markeredgewidth=0.9, clip_on=False, zorder=3)
            # ideal anchored at this series' own base point
            n0, t0 = xs[0], ys[0]
            ideal = [t0 * n0 / n for n in xs]
            ax.plot(xs, ideal, "--", color=c, lw=0.7, alpha=0.55, zorder=2)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        allx = sorted({p[0] for _, pts in spec["series"] for p in pts})
        ax.set_xticks(allx)
        ax.set_xticklabels([str(x) for x in allx])
        ax.minorticks_off()
        ax.set_title(spec["title"], pad=9, loc="left", fontweight="bold")
        ax.text(0, 1.015, spec["sub"], transform=ax.transAxes, fontsize=5.5,
                color="#555555", va="bottom")
        if spec["note"]:
            ax.text(0.98, 0.98, spec["note"], transform=ax.transAxes,
                    fontsize=5.2, color="#888888", ha="right", va="top",
                    style="italic")
        ax.tick_params(direction="out", length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="lower left", handlelength=1.6,
                  borderpad=0.2, labelspacing=0.25)
        if i % ncol == 0:
            ax.set_ylabel("time per step (ms)")
        if i >= ncol:
            ax.set_xlabel("devices (GPUs or MPI ranks)")
        # panel letter
        ax.text(-0.30, 1.22, chr(ord("a") + i), transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

    # last cell: legend/provenance instead of an empty frame
    ax = axs[len(PANELS)]
    ax.axis("off")
    handles = [
        Line2D([], [], color="#0072B2", marker="o", markerfacecolor="white",
               markeredgewidth=0.9, label="float32"),
        Line2D([], [], color="#D55E00", marker="s", markerfacecolor="white",
               markeredgewidth=0.9, label="float64"),
        Line2D([], [], color="#009E73", marker="D", markerfacecolor="white",
               markeredgewidth=0.9, label="mixed (f64 storage,\nf32 internals)"),
        Line2D([], [], color="#666666", ls="--", lw=0.7,
               label="ideal (anchored at\neach series' base)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              handlelength=1.8, labelspacing=0.8, borderpad=0)
    ax.text(0, 0.02, "Levante: 4×A100-80 SXM/node (NVLink, IB HDR200);\n"
                     "2×AMD Milan 7763 CPU nodes. Every point is a\n"
                     "measured receipt; job ids in the source table.",
            transform=ax.transAxes, fontsize=5.2, color="#777777", va="bottom")

    # row band labels
    for row, name in ((0, "ATMOSPHERE"), (1, "OCEAN")):
        y = 0.945 if row == 0 else 0.475
        fig.text(0.008, y, name, fontsize=7.5, fontweight="bold",
                 rotation=90, va="top", ha="left", color="#222222")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.885, bottom=0.105,
                        wspace=0.50, hspace=0.78)
    fig.savefig(args.out, bbox_inches="tight")
    if args.png:
        fig.savefig(args.png, bbox_inches="tight")
    print(args.out if not args.png else f"{args.out} {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
