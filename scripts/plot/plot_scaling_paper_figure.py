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
    "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64), "
                  "LL2048@64 26502539, @128 26534060, LL2304@96/144 26628072/26657279, fused 26681636/26681858, "
                  "packed-exchange A/B 26891278 (@128 4.527 vs 5.131, ratio 0.882)",
    "atm_cube": "26452894/26453782",
    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
                "ragged A/B 26824483 (s8@16) + 26825520 (s9@32), "
                "s8 np32-128 26549646/26538474, s9 26600095, "
                "s8-lloyd0 26628076, s10@128 26677812, "
                "size-colouring A/B 26857404 (s9@64 8.11 ms), ladder 26859802, "
                "wide-halo A/B 26880593 (s9@64 8.13→6.98), "
                "METIS A/B 26885812 (s9@32 10.44/10.62→9.26/9.59), "
                "s10@128 baseline 26892202/26896489 (12.78/12.54/12.83)",
    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic; "
                   "lat-lon 2-D r512 26628073",
    "oc_latlon": "26460444-501/26460365/26493592, LL2304@96/128 26646038/26646039, fused A/B 26692291",
    "oc_tripole": "26493837/26493648",
    "oc_mpas": "26494036 (f64), 26494908 (f32), "
               "FESOM2 native ref 26851736 (T4d-T2d)/72",
}

PANELS = [
    dict(
        key="atm_latlon", title="lat–lon", sub="720×1440 L26 · A100 NCCL",
        series=[("float32", [(4, 7.72), (8, 5.40), (16, 3.54)]),
                ("float64", [(4, 16.11), (8, 11.34), (16, 5.62)]),
                ("float32 (LL2048)", [(64, 6.73), (128, 5.58)]),
                ("f32 (LL2304)", [(96, 7.87), (144, 5.78)]),
                ("f32 LL2048 fused+ovl", [(64, 5.278), (128, 4.745)]),
                ("f32 LL2048 packed exch", [(128, 4.527)]),
                # 2^n ladder 2026-08-18/19 (jobs 27051xxx), best env
                # (packed exch + overlap + mcp2p), 60-step blocks:
                ("f32 LL2048 ladder (best env)",
                 [(1, 201.48), (2, 110.92), (4, 55.40), (8, 27.72),
                  (16, 14.26), (32, 6.98), (64, 4.74), (128, 3.96)]),
                ("f32 LL4096 ladder (best env)",
                 [(4, 222.76), (8, 111.85), (16, 55.97), (32, 29.27),
                  (64, 16.24), (128, 9.33)]),
                ],
        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 f64 @128", 128, 9.60),
                 ("LL2880 @144", 144, 7.40),
                 # 27036060: first atm 192-GPU point, 63.98 GC/s (lane best).
                 # LL2304@192 (12-row bands) deadlocks NCCL init — thin-band
                 # trigger, 7 env candidates refuted; >=15-row bands run.
                 ("LL2880 @192", 192, 6.74)],
        note="packed exchange @128: −11.8 %\n(25→13 collectives/step)\nLL4096 ladder: 87 % eff @128, 93.5 GC/s",
    ),
    dict(
        key="atm_cube", title="cubed-sphere", sub="C384/C768 L60 · A100 NCCL",
        series=[("float32 (C768)", [(6, 58.35), (24, 14.09)]),
                ("float32 (C384)", [(6, 15.44), (24, 8.81)])],
        note="f64 pending",
    ),
    dict(
        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9/10 L26 · A100 NCCL",
        series=[("f32 (s8 · lloyd-50)", [(2, 19.90), (4, 14.12), (8, 6.92),
                                         (16, 7.10), (32, 8.13), (64, 5.27),
                                         (128, 6.47)]),
                ("f32 s9 · step 0 (baseline)",
                 [(32, 12.47), (64, 9.60), (128, 11.48)]),
                ("f32 s9 · step 1 (size-colouring)",
                 [(16, 21.22), (32, 10.52), (64, 8.40)]),
                ("f32 s9 · step 2 (METIS + wide halo)",
                 [(32, 9.26), (64, 6.98)]),
                # step 3: NCCL multi-channel p2p env pair
                # (MIN/MAX_NCHANNELS=8 + P2P_NET_CHUNKSIZE=131072), wide
                # halo, sfc — CONFIRMED jobs 26999539 + 27002564
                # (5.87/5.88 replicated vs same-job base 7.18/7.26).
                ("f32 s9 · step 3 (NCCL mcp2p)", [(64, 5.87)]),
                # step 4: ragged_all_to_all halo (2 collectives/fill,
                # 27040575 ratio 0.933 vs coloured) + mcp2p pair —
                # 27041921: 5.70/5.75 vs same-job ragged base 6.26/6.17.
                ("f32 s9 · step 4 (ragged + mcp2p)", [(64, 5.73)]),
                # 2^n ladder 2026-08-18/19, best env (wide + ragged +
                # mcp2p), 60-step blocks; s9@64 5.74 reproduces step 4:
                ("f32 s9 ladder (best env)",
                 [(1, 173.84), (2, 81.76), (4, 42.03), (8, 23.06),
                  (16, 19.16), (32, 7.20), (64, 5.74), (128, 5.09)]),
                ("f32 s10 ladder (best env)",
                 [(2, 264.56), (4, 117.91), (8, 87.60), (16, 37.51),
                  (32, 19.95), (64, 15.91), (128, 8.57)]),
                ("f32 s10 · now (was 18.20)", [(128, 12.54), (192, 10.57)]),
                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
        scatter=[("s10 2026-06", 128, 18.20)],
        note="each step is an A/B receipt:\nsize-colouring −12…−20 %, METIS −10.5 %,\nwide halo −14.1 %, NCCL mcp2p −13…−18 %,\nragged 2-collective halo −8 %",
    ),
    dict(
        key="atm_ico_cpu", title="ico + lat-lon 2-D", sub="subdiv-7 / r512 L26 · Milan CPU–MPI",
        series=[("float32", [(1, 7399.72), (2, 3250.27), (4, 1673.07),
                             (8, 827.06), (16, 444.58), (32, 230.35),
                             (64, 131.58)]),
                ("float64", [(1, 9987.81), (2, 4727.46), (4, 2349.06),
                             (8, 1161.98), (16, 624.06), (32, 350.89),
                             (64, 220.57), (128, 92.4), (256, 56.2),
                             (512, 66.7)]),
                ("f64 lat-lon 2-D (r512)", [(64, 297.57), (128, 161.03),
                                            (256, 72.06), (512, 44.78)])],
        note="ico to 1024; lat-lon 2-D\neff 0.83 @512",
    ),
    dict(
        key="oc_latlon", title="lat–lon", sub="LL576/LL2304 L20 · A100 NCCL",
        series=[("float32", [(1, 36.45), (2, 22.48), (4, 12.84), (8, 11.04), (16, 8.71)]),
                ("float64", [(1, 64.81), (2, 41.63), (4, 22.09)]),
                ("mixed (f64 store)", [(1, 52.80), (4, 19.13)]),
                ("f32 (LL2304)", [(96, 18.25), (128, 16.33)]),
                ("f32 LL2304 fused", [(128, 15.807)])],
        note="fused halo @128 = 15.81 ms\n= 13.4 GC/s (\u22124.9 %; overlap null)",
    ),
    dict(
        key="oc_tripole", title="tripole (ORCA fold)", sub="576×1152 L20 · A100 NCCL",
        series=[("float32", [(1, 35.52), (2, 25.03), (4, 15.91)]),
                ("float64", [(1, 63.84), (2, 43.36), (4, 24.12)])],
        note="fold +1.2–3.7 %",
    ),
    dict(
        key="oc_mpas", title="MPAS Voronoi", sub="subdiv-7/8 · Milan CPU–MPI, 32 rpn",
        series=[("float64 (subdiv-7)", [(32, 190.22), (64, 147.65),
                                        (128, 102.93), (256, 65.71),
                                        (512, 63.83)]),
                ("float64 (subdiv-8)", [(32, 861.25), (64, 494.89),
                                        (128, 309.05), (256, 254.41),
                                        (512, 194.80)]),
                ("FESOM2 native (CORE2 ref)", [(32, 393.0), (64, 380.1),
                                               (128, 170.7), (256, 84.1),
                                               (512, 34.9)])],
        note="32 ranks/node fixed;\nFESOM2 = Fortran ref, CORE2 127k tri "
             "L47\n(np\u226432\u2013128 packed on one node)",
    ),
]

COLORS = {"float32": "#0072B2", "float64": "#D55E00",
          "f32 · LL1536/2048 @64": "#009E73",
          "float64 (packed)": "#E69F00",
          "mixed (f64 store)": "#009E73",
          "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
          "float32 (LL2048)": "#009E73",
          "f64 lat-lon 2-D (r512)": "#CC79A7",
          "f32 (s8 lloyd-0)": "#009E73",
          "f32 (s10 lloyd-0)": "#000000",
          "f32 s9 · step 3 (NCCL mcp2p)": "#009E73",
          "f32 (LL2304)": "#CC79A7",
          "f32 LL2048 fused+ovl": "#000000",
          "f32 LL2048 packed exch": "#E31A1C",
          "f32 LL2304 fused": "#000000",
          "f32 s8@16+s9@32 ragged": "#000000",
          "f32 s9 · step 0 (baseline)": "#56B4E9",
          "f32 s9 · step 1 (size-colouring)": "#D62728",
          "f32 s9 · step 2 (METIS + wide halo)": "#E31A1C",
          "f32 s10 · now (was 18.20)": "#7B3294",
          "f32 s8+s9 size-colouring": "#2CA02C",
          "f32 (s8 · lloyd-50)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
          "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00",
          "FESOM2 native (CORE2 ref)": "#555555"}
MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
           "f32 · LL1536/2048 @64": "*",
           "float64 (packed)": "s",
           "float32 (C768)": "o", "float32 (C384)": "^",
           "float32 (LL2048)": "^",
           "f64 lat-lon 2-D (r512)": "D",
           "f32 (s8 lloyd-0)": "v",
           "f32 (s10 lloyd-0)": "*",
           "f32 s9 · step 3 (NCCL mcp2p)": "P",
           "f32 (LL2304)": "^",
           "f32 LL2048 fused+ovl": "*",
           "f32 LL2048 packed exch": "P",
           "f32 LL2304 fused": "*",
           "f32 s8@16+s9@32 ragged": "*",
           "f32 s9 · step 0 (baseline)": "^",
           "f32 s9 · step 1 (size-colouring)": "X",
           "f32 s9 · step 2 (METIS + wide halo)": "P",
           "f32 s10 · now (was 18.20)": "P",
           "f32 s8+s9 size-colouring": "X",
           "f32 (s8 · lloyd-50)": "o", "float32 (subdiv-9)": "^",
           "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v",
           "FESOM2 native (CORE2 ref)": "P"}


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

        for lab, x, y in spec.get("scatter", []):
            ax.plot([x], [y], "*", color="#009E73", markersize=7,
                    markeredgewidth=0.8, clip_on=False, zorder=4)
            ax.annotate(lab, (x, y), fontsize=5.2, color="#009E73",
                        textcoords="offset points", xytext=(-4, 5), ha="right")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        allx = sorted({p[0] for _, pts in spec["series"] for p in pts}
                      | {x for _, x, _ in spec.get("scatter", [])})
        # thin crowded tick sets to powers spanning the range
        if len(allx) > 6:
            allx = [x for i, x in enumerate(allx) if i % 3 == 0 or x == allx[-1]]
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
        Line2D([], [], color="#009E73", marker="*", ls="none", markersize=7,
               label="high-resolution point\n(not part of a ladder)"),
        Line2D([], [], color="#666666", ls="--", lw=0.7,
               label="ideal (anchored at\neach series' base)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              handlelength=1.8, labelspacing=0.55, borderpad=0)
    ax.text(0, -0.06, "Levante: 4×A100-80 SXM/node (NVLink, IB HDR200);\n"
                      "2×AMD Milan 7763 CPU nodes. Every point is a\n"
                      "measured receipt; job ids in the source table.",
            transform=ax.transAxes, fontsize=5.0, color="#777777", va="top")

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
