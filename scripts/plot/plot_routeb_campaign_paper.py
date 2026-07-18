#!/usr/bin/env python
"""Paper figure for the route-B GPU scaling campaign
(docs/performance/scaling/routeb_campaign_report_2026-07.md).
Dry-dynamics dycore, A100-40GB, Derecho.

THE MESSAGE: GPU strong scaling is **resolution-dependent**. At coarse
resolution (~1deg) there is too little work per device, so the halo comms
dominate and it does NOT scale (throughput flattens/drops as GPUs are added).
At fine resolution (~25 km) each device has enough compute to hide the comms,
so it DOES scale. Shown on icosahedral (the grid with a full 1->16 ladder
spanning ~112 km down to ~28 km).

Two panels, per figure:
  (a) SYPD vs #A100 (log-log) — coarse peaks then drops; fine climbs.
  (b) strong-scaling efficiency vs #A100, BASELINED TO N=2 (the ratio to
      2 A100s, so the one-time 1->2 halo-comm onset is not charged against
      scaling) — coarse collapses toward zero; fine holds.

FIG 1 = icosahedral (~112 -> ~28 km). FIG 2 (confirmation) = lat-lon
(~39 -> ~19.5 km), same trend on a second grid.

Resolution encoded as a sequential blue ramp (coarse = light, fine = dark) +
distinct markers/linestyles (grayscale/print safe).

Run:  python scripts/plot/plot_routeb_campaign_paper.py \
          [--csv $SCRATCH/legoesm_scaling/routeb_campaign_202607/all_tidy.csv] \
          [--out results/paper_figs]
Reads the CSV when present; else the embedded 2026-07 snapshot.
"""
from __future__ import annotations

import argparse
import os

# Sequential blue ramp: coarse (light) -> fine (dark). ColorBrewer Blues.
RAMP3 = ["#9ecae1", "#4292c6", "#084594"]
RAMP2 = ["#6baed6", "#084594"]
MARKERS = ["o", "^", "D"]
LINES = ["--", "-.", "-"]

# Figures: (title, out-name, grid, [(resolution, km, label)] coarse->fine).
FIGS = [
    ("Icosahedral: GPU scaling improves with resolution", "fig1_ico_scaling",
     "icosahedral", RAMP3, [
         (6, 112, "L6  ~112 km  (≈1°)"),
         (7, 56, "L7  ~56 km"),
         (8, 28, "L8  ~28 km"),
     ]),
    ("Lat-lon: same trend (confirmation)", "fig2_latlon_scaling",
     "latlon", RAMP2, [
         (512, 39, "LL512  ~39 km"),
         (1024, 19.5, "LL1024  ~19.5 km"),
     ]),
]

# Embedded 2026-07 campaign snapshot (GPU): (grid,res) -> {N: (sypd, mcells_s)}
SNAPSHOT = {
    ("icosahedral", 6): {1: (46.312, 480.57), 2: (57.866, 600.46),
                         4: (61.201, 635.07), 8: (37.310, 387.15),
                         16: (30.827, 319.89)},
    ("icosahedral", 7): {1: (5.116, 265.38), 2: (14.861, 770.83),
                         4: (23.270, 1206.97), 8: (18.815, 975.93),
                         16: (19.655, 1019.47)},
    ("icosahedral", 8): {1: (1.831, 379.93), 2: (3.811, 790.71),
                         4: (4.442, 921.62), 8: (8.800, 1825.56),
                         16: (9.955, 2065.35)},
    ("latlon", 512): {1: (12.179, 1010.67), 2: (21.635, 1795.29),
                      4: (37.160, 3083.63), 8: (45.729, 3794.64),
                      16: (52.722, 4374.96)},
    ("latlon", 1024): {1: (2.948, 978.45), 2: (5.169, 1715.57),
                       4: (10.552, 3502.53), 8: (19.258, 6392.26),
                       16: (29.786, 9886.67)},
}


def load(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return SNAPSHOT
    import csv
    out: dict = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["backend"] != "GPU":
                continue
            out.setdefault((r["grid"], int(r["resolution"])), {})[int(r["n_devices"])] = (
                float(r["sypd"]), float(r["mcells_per_s"]))
    return out


def _style(ax, INK="#1a1a1a"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="both", color="#ececec", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, color="#999999")


def make_figure(data, title, outname, grid, ramp, resolutions, out):
    import matplotlib.pyplot as plt
    INK, MUTED = "#1a1a1a", "#666666"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    for (res, km, label), col, mk, ls in zip(resolutions, ramp, MARKERS, LINES):
        s = data.get((grid, res))
        if not s:
            continue
        Ns = sorted(s)
        axa.plot(Ns, [s[n][0] for n in Ns], ls, color=col, marker=mk,
                 markersize=6.5, linewidth=2.0, label=label, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.7)
        if 2 in s:
            base = s[2][0]
            Ne = [n for n in Ns if n >= 2]
            axb.plot(Ne, [(s[n][0] / base) / (n / 2.0) for n in Ne], ls,
                     color=col, marker=mk, markersize=6.5, linewidth=2.0,
                     label=label, zorder=5, markeredgecolor="white",
                     markeredgewidth=0.7)

    axa.set_xscale("log", base=2); axa.set_yscale("log")
    axa.set_xticks([1, 2, 4, 8, 16]); axa.set_xticklabels([1, 2, 4, 8, 16])
    axa.set_xlabel("A100 GPUs", fontsize=10, color=INK)
    axa.set_ylabel("SYPD (simulated yr / day)", fontsize=10, color=INK)
    axa.set_title("(a) throughput vs GPUs", fontsize=11, color=INK, loc="left")
    axa.axvline(4, color="#dcdcdc", linewidth=1.0, zorder=1)
    axa.annotate("1 node | multi-node", xy=(4, axa.get_ylim()[0]), xytext=(2, 3),
                 textcoords="offset points", fontsize=7.5, color=MUTED,
                 rotation=90, va="bottom")
    _style(axa); axa.legend(fontsize=8.5, frameon=False, loc="best")

    axb.axhline(1.0, color="#bdbdbd", linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    axb.annotate("ideal (100%)", xy=(16, 1.0), xytext=(0, 3),
                 textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    axb.set_xscale("log", base=2)
    axb.set_xticks([2, 4, 8, 16]); axb.set_xticklabels([2, 4, 8, 16])
    axb.set_ylim(0, 1.15)
    axb.set_xlabel("A100 GPUs", fontsize=10, color=INK)
    axb.set_ylabel("strong-scaling efficiency (ratio to 2 A100)", fontsize=10, color=INK)
    axb.set_title("(b) scaling quality vs GPUs", fontsize=11, color=INK, loc="left")
    axb.axvline(4, color="#dcdcdc", linewidth=1.0, zorder=1)
    _style(axb); axb.legend(fontsize=8.5, frameon=False, loc="best")

    fig.suptitle(title + "  —  dry dynamics, A100 (Derecho)",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/{outname}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.environ.get("WB_CAMPAIGN_CSV", ""))
    p.add_argument("--out", default="results/paper_figs")
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    data = load(a.csv)
    os.makedirs(a.out, exist_ok=True)
    for title, outname, grid, ramp, res in FIGS:
        make_figure(data, title, outname, grid, ramp, res, a.out)
    print(f"wrote {a.out}/{{{','.join(f[1] for f in FIGS)}}}.{{png,pdf}}")


if __name__ == "__main__":
    main()
