#!/usr/bin/env python
"""Paper figures for the route-B GPU scaling campaign (docs/performance/scaling/
routeb_campaign_report_2026-07.md). Dry-dynamics dycore, A100-40GB, Derecho.

Two figures, both from the campaign `all_tidy.csv` (GPU rows only):

  FIG 1 (money, 2 panels):
    (a) SYPD vs #A100 (log-log) — absolute capability (Demo 1) + throughput
        scaling (Demo 2), with a recessive ideal-linear guide.
    (b) strong-scaling efficiency vs #A100, BASELINED TO N=2 (Demo 3) — the
        1->2 halo-comm onset is absorbed into the baseline, so this measures
        multi-GPU scaling quality, not the one-time comm cost.

  FIG 2 (optional): SYPD vs resolution (km) at 1 and 16 GPUs — the coarse->fine
        envelope.

Palette: Okabe-Ito subset (CVD-safe, validated ΔE~37). Curves also carry
distinct markers + linestyles (secondary encoding for print / colorblind).

Run:  python scripts/plot/plot_routeb_campaign_paper.py \
          [--csv $SCRATCH/legoesm_scaling/routeb_campaign_202607/all_tidy.csv] \
          [--out results/paper_figs]
Reads the CSV when present; otherwise uses the embedded 2026-07 campaign
snapshot (so the figure regenerates on Derecho and renders anywhere).
"""
from __future__ import annotations

import argparse
import os

# --- curves to plot: (grid, resolution) -> label; a deliberate, uncluttered 4
# spanning all three grids and coarse->fine. Fixed order = fixed hue slot.
CURVES = [
    ("latlon", 1024, "lat-lon LL1024 (19.5 km)"),
    ("latlon", 512, "lat-lon LL512 (39 km)"),
    ("cubed-sphere", 192, "cube C192 (52 km)"),
    ("icosahedral", 8, "ico L8 (28 km)"),
]
# Okabe-Ito (CVD-safe), fixed order matched to CURVES; + markers + linestyles.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]
LINES = ["-", "--", "-.", ":"]

# Embedded 2026-07 campaign snapshot (GPU): (grid,res) -> {N: (sypd, mcells_s)}
SNAPSHOT = {
    ("latlon", 512): {1: (12.179, 1010.67), 2: (21.635, 1795.29),
                      4: (37.160, 3083.63), 8: (45.729, 3794.64),
                      16: (52.722, 4374.96)},
    ("latlon", 1024): {1: (2.948, 978.45), 2: (5.169, 1715.57),
                       4: (10.552, 3502.53), 8: (19.258, 6392.26),
                       16: (29.786, 9886.67)},
    ("cubed-sphere", 192): {1: (14.739, 1031.95), 2: (12.941, 906.04),
                            3: (14.590, 1021.54), 6: (20.252, 1417.93)},
    ("cubed-sphere", 96): {1: (130.160, 759.44), 2: (73.055, 426.25),
                           3: (82.797, 483.09), 6: (91.322, 532.83)},
    ("icosahedral", 8): {1: (1.831, 379.93), 2: (3.811, 790.71),
                         4: (4.442, 921.62), 8: (8.800, 1825.56),
                         16: (9.955, 2065.35)},
    ("icosahedral", 7): {1: (5.116, 265.38), 2: (14.861, 770.83),
                         4: (23.270, 1206.97), 8: (18.815, 975.93),
                         16: (19.655, 1019.47)},
}

# fig2: per-GRID resolution sweeps (km) — NEVER connect across grids.
GRID_RES = {
    "lat-lon": ("#0072B2", "o", [("latlon", 512, 39), ("latlon", 1024, 19.5)]),
    "cube": ("#009E73", "^", [("cubed-sphere", 96, 104), ("cubed-sphere", 192, 52)]),
    "ico": ("#D55E00", "D", [("icosahedral", 7, 56), ("icosahedral", 8, 28)]),
}


def load(csv_path):
    """Return {(grid,res): {N: (sypd, mcells_s)}} from the CSV, or the snapshot."""
    if not csv_path or not os.path.exists(csv_path):
        return SNAPSHOT
    import csv
    out: dict = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["backend"] != "GPU":
                continue
            key = (r["grid"], int(r["resolution"]))
            out.setdefault(key, {})[int(r["n_devices"])] = (
                float(r["sypd"]), float(r["mcells_per_s"]))
    return out


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="both", color="#e6e6e3", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, color="#999999")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.environ.get("WB_CAMPAIGN_CSV", ""))
    p.add_argument("--out", default="results/paper_figs")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(a.csv)
    os.makedirs(a.out, exist_ok=True)
    INK, MUTED = "#1a1a1a", "#666666"

    # ---- FIG 1: (a) SYPD vs N (log-log)  |  (b) eff vs N (baseline N=2) ----
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    for (grid, res, label), c, mk, ls in zip(CURVES, COLORS, MARKERS, LINES):
        series = data.get((grid, res))
        if not series:
            continue
        Ns = sorted(series)
        sypd = [series[n][0] for n in Ns]
        axa.plot(Ns, sypd, ls, color=c, marker=mk, markersize=6, linewidth=1.9,
                 label=label, zorder=5, markeredgecolor="white",
                 markeredgewidth=0.6)
        # efficiency baselined to N=2 (only for N>=2)
        if 2 in series:
            base = series[2][0]
            Ne = [n for n in Ns if n >= 2]
            eff = [(series[n][0] / base) / (n / 2.0) for n in Ne]
            axb.plot(Ne, eff, ls, color=c, marker=mk, markersize=6,
                     linewidth=1.9, label=label, zorder=5,
                     markeredgecolor="white", markeredgewidth=0.6)

    # (a) ideal-linear guide anchored at the LL1024 N=2 point (recessive)
    if ("latlon", 1024) in data and 2 in data[("latlon", 1024)]:
        b = data[("latlon", 1024)][2][0]
        xg = [2, 16]
        axa.plot(xg, [b, b * 8], color="#bdbdbd", linewidth=1.2, linestyle=(0, (4, 3)),
                 zorder=2)
        axa.annotate("ideal (linear)", xy=(16, b * 8), xytext=(0, 3),
                     textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    axa.set_xscale("log", base=2); axa.set_yscale("log")
    axa.set_xticks([1, 2, 4, 8, 16]); axa.set_xticklabels([1, 2, 4, 8, 16])
    axa.set_xlabel("A100 GPUs", fontsize=10, color=INK)
    axa.set_ylabel("SYPD (simulated yr / day)", fontsize=10, color=INK)
    axa.set_title("(a) throughput: SYPD vs GPUs", fontsize=11, color=INK, loc="left")
    axa.axvline(4, color="#d9d9d9", linewidth=1.0, zorder=1)
    axa.annotate("1 node | multi-node", xy=(4, axa.get_ylim()[0]), xytext=(2, 2),
                 textcoords="offset points", fontsize=7.5, color=MUTED, rotation=90,
                 va="bottom")
    _style_axes(axa)
    axa.legend(fontsize=8, frameon=False, loc="lower right")

    # (b) efficiency panel
    axb.axhline(1.0, color="#bdbdbd", linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    axb.annotate("ideal (100%)", xy=(16, 1.0), xytext=(0, 3),
                 textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    axb.set_xscale("log", base=2)
    axb.set_xticks([2, 4, 8, 16]); axb.set_xticklabels([2, 4, 8, 16])
    axb.set_ylim(0, 1.15)
    axb.set_xlabel("A100 GPUs", fontsize=10, color=INK)
    axb.set_ylabel("strong-scaling efficiency (baseline N=2)", fontsize=10, color=INK)
    axb.set_title("(b) scaling quality vs GPUs", fontsize=11, color=INK, loc="left")
    axb.axvline(4, color="#d9d9d9", linewidth=1.0, zorder=1)
    _style_axes(axb)

    fig.suptitle("Differentiable JAX dycore — dry-dynamics scaling (A100, Derecho)",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}/fig1_scaling.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- FIG 2: SYPD vs resolution (km) at 1 A100 — ONE LINE PER GRID ----
    # (never connect across grids: each line is a single grid's own resolution
    #  sweep, so the coarse->fine slope is a real per-grid property.)
    fig2, ax2 = plt.subplots(figsize=(5.4, 4.0))
    for label, (col, mk, entries) in GRID_RES.items():
        pts = sorted((km, data[(g, r)][1][0]) for g, r, km in entries
                     if (g, r) in data and 1 in data[(g, r)])
        if len(pts) >= 1:
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, "-", color=col, marker=mk, markersize=6.5,
                     linewidth=1.9, label=label, markeredgecolor="white",
                     markeredgewidth=0.6)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("resolution (km)", fontsize=10, color=INK)
    ax2.set_ylabel("SYPD on 1 A100", fontsize=10, color=INK)
    ax2.set_title("SYPD vs resolution, per grid (1 A100)", fontsize=11,
                  color=INK, loc="left")
    ax2.invert_xaxis()  # coarse (left) -> fine (right)
    _style_axes(ax2)
    ax2.legend(fontsize=9, frameon=False, loc="upper right", title="grid",
               title_fontsize=9)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(f"{a.out}/fig2_resolution.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"wrote {a.out}/fig1_scaling.{{png,pdf}} and fig2_resolution.{{png,pdf}}")


if __name__ == "__main__":
    main()
