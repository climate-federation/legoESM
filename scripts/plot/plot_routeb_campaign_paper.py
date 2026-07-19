#!/usr/bin/env python
"""Paper figure for the route-B GPU scaling campaign
(docs/performance/scaling/routeb_campaign_report_2026-07.md).
Dry-dynamics dycore, A100-40GB, Derecho.

THE MESSAGE: GPU strong scaling is **resolution-dependent**. At coarse
resolution (~1deg) there is too little work per device, so the halo comms
dominate and it does NOT scale (throughput flattens as GPUs are added).
At fine resolution (~20 km) each device has enough compute to hide the comms,
so it DOES scale.

Throughput metric is **Mcells/s**, NOT SYPD. Mcells/s is timestep-free, so it
compares resolutions (and grids) without the dt confound: at these sizes one
A100 is saturated regardless of resolution, so all curves START at the same
per-device throughput (~1000 Mcells/s for lat-lon) and then FAN OUT purely
according to how well halo exchange is hidden. SYPD would fold dt and problem
size back in and obscure exactly the variable under test.

Two panels, per figure:
  (a) Mcells/s vs #A100 (log-log) — coarse flattens; fine climbs near-ideal.
      Faint dotted ray = ideal scaling from the N=1 anchor.
  (b) strong-scaling efficiency vs #A100, BASELINED TO N=2 (the ratio to
      2 A100s, so the one-time 1->2 halo-comm onset is not charged against
      scaling) — coarse collapses toward zero; fine holds.
  Panel (b) is identical whether computed from SYPD or Mcells/s (same problem
  size per curve => the ratio is the same), so only (a) changes with metric.

FIG 1 (main) = lat-lon (~104 -> ~19.5 km), the full three-resolution ramp.
FIG 2 (SI confirmation) = icosahedral (~112 -> ~28 km), same trend, 2nd grid.

Resolution encoded as a sequential blue ramp (coarse = light, fine = dark) +
distinct markers/linestyles (grayscale/print safe).

Run:  python scripts/plot/plot_routeb_campaign_paper.py \
          [--csv $SCRATCH/legoesm_scaling/routeb_campaign_202607/all_tidy.csv] \
          [--out results/paper_figs]
Reads the aggregated campaign CSV; --csv is REQUIRED (there is deliberately no
embedded fallback snapshot — a stale hardcoded copy silently masks divergence
between the figures and the aggregated results).
"""
from __future__ import annotations

import argparse
import os

# Sequential blue ramp: coarse (light) -> fine (dark). ColorBrewer Blues.
RAMP3 = ["#9ecae1", "#4292c6", "#084594"]
MARKERS = ["o", "^", "D"]
LINES = ["--", "-.", "-"]

# Figures: (title, out-name, grid, ramp, [(resolution, km, label)] coarse->fine).
FIGS = [
    ("Lat-lon: GPU scaling improves with resolution", "fig1_latlon_scaling",
     "latlon", RAMP3, [
         (192, 104, "LL192  ~104 km  (≈1°)"),
         (512, 39, "LL512  ~39 km"),
         (1024, 19.5, "LL1024  ~19.5 km"),
     ]),
    ("Icosahedral: same trend (confirmation)", "fig2_ico_scaling",
     "icosahedral", RAMP3, [
         (6, 112, "L6  ~112 km  (≈1°)"),
         (7, 56, "L7  ~56 km"),
         (8, 28, "L8  ~28 km"),
     ]),
]

def load(csv_path):
    """Load GPU rows from the aggregated campaign CSV.

    Deliberately no embedded-snapshot fallback: a hardcoded copy silently
    diverges from the aggregated results (it did — the 2026-07-17 snapshot
    carried pre-rerun icosahedral values), so a missing CSV must fail loudly.
    """
    if not csv_path:
        raise SystemExit(
            "--csv is required (or set WB_CAMPAIGN_CSV): path to all_tidy.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"--csv not found: {csv_path}")
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


def make_figure(data, title, outname, grid, ramp, resolutions, out,
                panel_b="efficiency"):
    """Two-panel scaling figure.

    ``panel_b`` selects how panel (b) normalises, both baselined at N=2:
      "efficiency" -- speedup / ideal speedup; ideal is a flat line at 1.0.
      "speedup"    -- raw speedup S(N)=thr(N)/thr(2); ideal is the diagonal.

    They carry the same information (efficiency = speedup / (N/2)); the choice
    is about legibility.  Speedup makes ANTI-scaling visceral -- ico L6 reads
    0.52x, i.e. 16 GPUs runs at HALF the speed of 2, where "7% efficiency"
    needs a mental step.  Efficiency is more compact for tables.
    """
    import matplotlib.pyplot as plt
    if panel_b not in ("efficiency", "speedup"):
        raise ValueError(
            f"panel_b must be 'efficiency' or 'speedup', got {panel_b!r}")
    INK = "#1a1a1a"  # MUTED dropped with the in-axes annotations
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    for (res, km, label), col, mk, ls in zip(resolutions, ramp, MARKERS, LINES):
        s = data.get((grid, res))
        if not s:
            continue
        Ns = sorted(s)
        # index 1 = Mcells/s (index 0 = SYPD). Mcells/s is timestep-free.
        axa.plot(Ns, [s[n][1] for n in Ns], ls, color=col, marker=mk,
                 markersize=6.5, linewidth=2.0, label=label, zorder=5,
                 markeredgecolor="white", markeredgewidth=0.7)
        # Faint ideal-scaling ray from this curve's own N=1 anchor.
        if 1 in s:
            axa.plot(Ns, [s[1][1] * n for n in Ns], ":", color=col,
                     linewidth=1.1, alpha=0.45, zorder=3)
        if 2 in s:
            base = s[2][1]
            Ne = [n for n in Ns if n >= 2]
            if panel_b == "speedup":
                yb = [s[n][1] / base for n in Ne]
            else:
                yb = [(s[n][1] / base) / (n / 2.0) for n in Ne]
            axb.plot(Ne, yb, ls,
                     color=col, marker=mk, markersize=6.5, linewidth=2.0,
                     label=label, zorder=5, markeredgecolor="white",
                     markeredgewidth=0.7)

    axa.set_xscale("log", base=2); axa.set_yscale("log")
    axa.set_xticks([1, 2, 4, 8, 16]); axa.set_xticklabels([1, 2, 4, 8, 16])
    axa.set_xlabel("A100 GPUs", fontsize=10, color=INK)
    axa.set_ylabel("throughput (Mcells / s)", fontsize=10, color=INK)
    axa.set_title("(a) throughput vs GPUs", fontsize=11, color=INK, loc="left")
    # In-axes annotations deliberately removed (the 1-node/multi-node marker,
    # the "dotted = ideal scaling" note, and the "ideal (100%)" label): they
    # belong in the caption, not on the plot.  The dotted rays and the y=1
    # reference line are kept -- they read as ideal scaling without a label.
    _style(axa); axa.legend(fontsize=8.5, frameon=False, loc="best")

    axb.set_xscale("log", base=2)
    axb.set_xticks([2, 4, 8, 16]); axb.set_xticklabels([2, 4, 8, 16])
    if panel_b == "speedup":
        # Ideal = the N/2 diagonal. Curves BELOW y=1 are anti-scaling: more
        # GPUs than 2, less throughput than 2. That reading is the reason to
        # prefer this panel over efficiency.
        ns_all = sorted({n for res, _km, _lab in resolutions
                         for n in (data.get((grid, res)) or {}) if n >= 2})
        if ns_all:
            axb.plot(ns_all, [n / 2.0 for n in ns_all], ":", color="#bdbdbd",
                     linewidth=1.4, zorder=2)
        axb.axhline(1.0, color="#dcdcdc", linewidth=1.0, zorder=1)
        axb.set_yscale("log", base=2)
        axb.set_yticks([0.5, 1, 2, 4, 8])
        axb.set_yticklabels(["0.5×", "1×", "2×", "4×", "8×"])
    else:
        axb.axhline(1.0, color="#bdbdbd", linewidth=1.2,
                    linestyle=(0, (4, 3)), zorder=2)
        axb.set_ylim(0, 1.15)
    axb.set_xlabel("A100 GPUs", fontsize=10, color=INK)
    if panel_b == "speedup":
        axb.set_ylabel("speed-up (relative to 2 A100)", fontsize=10, color=INK)
    else:
        axb.set_ylabel("strong-scaling efficiency (ratio to 2 A100)",
                       fontsize=10, color=INK)
    axb.set_title("(b) scaling quality vs GPUs", fontsize=11, color=INK, loc="left")
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
    p.add_argument("--panel-b", default="efficiency",
                   choices=["efficiency", "speedup"],
                   help="panel (b) normalisation, both baselined at N=2: "
                        "'efficiency' = speedup/ideal (flat line at 1.0); "
                        "'speedup' = raw ratio to 2 GPUs (ideal = diagonal, "
                        "and anti-scaling reads directly as <1x)")
    p.add_argument("--suffix", default="",
                   help="appended to output filenames (compare variants)")
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    data = load(a.csv)
    os.makedirs(a.out, exist_ok=True)
    for title, outname, grid, ramp, res in FIGS:
        make_figure(data, title, outname + a.suffix, grid, ramp, res, a.out,
                    panel_b=a.panel_b)
    names = ",".join(f[1] + a.suffix for f in FIGS)
    print(f"wrote {a.out}/{{{names}}}.{{png,pdf}} "
          f"(panel b = {a.panel_b})")


if __name__ == "__main__":
    main()
