"""Where does a section's transport difference actually sit?

A single integral says ours is 10 Sv stronger through Drake; it cannot say
whether that is a broad excess spread over the whole passage or a jet that is
too fast in three rows. Those have different causes and different fixes, so
the per-row profile is the diagnostic and the integral is only its summary.

Reads the JSON the section probe writes -- the per-row arrays are already in
it, so no rerun is needed to change how this is drawn.

Transport per row, not per metre: each bar is one grid row's contribution in
Sv, so the bars SUM to the section total printed in the title. That makes the
visual area directly comparable to the headline number, which a per-unit-
latitude normalisation would quietly break on a grid whose rows are not
equally tall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS = "#2e8b57"
NEMO = "#1f5fa9"
DIFF = "#b03030"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cell-out", default=None,
                   help="also draw the single-cell depth profile, when "
                        "the JSON carries one")
    a = p.parse_args()

    d = json.loads(Path(a.json).read_text())

    # A single cell, level by level, when the probe was asked for one. Drawn
    # here rather than in a new file because it reads the same JSON: the
    # per-row panels answer "which cell", this answers "which depth", and the
    # two belong to the same instrument.
    if a.cell_out and "cell" in d:
        c = d["cell"]
        z, o, n = c["depth_m"], c["ours_Sv"], c["nemo_Sv"]
        f2, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 5.6), sharey=True)
        axl.plot(o, z, color=OURS, lw=1.8, marker="o", ms=3, label="ours")
        axl.plot(n, z, color=NEMO, lw=1.8, marker="s", ms=3, label="NEMO")
        axl.set_xlabel("transport in this level [Sv]", fontsize=9)
        axl.set_ylabel("depth [m]", fontsize=9)
        axl.legend(fontsize=8)
        axr.plot([x - y for x, y in zip(o, n)], z, color=DIFF, lw=1.8,
                 marker="o", ms=3)
        axr.axvline(0.0, color="0.5", lw=0.8)
        axr.set_xlabel("ours - NEMO [Sv]", fontsize=9)
        for ax in (axl, axr):
            ax.invert_yaxis()
            ax.grid(ls=":", lw=0.5, alpha=0.6)
            ax.set_axisbelow(True)
        f2.suptitle(
            f"The Florida strait, one cell at {c['lat']:.2f}N "
            f"{c['lon']:.2f}E, {c['wet_levels']} wet levels.\n"
            f"Column total ours {sum(o):+.1f} Sv, NEMO {sum(n):+.1f} Sv. "
            "An excess spread evenly over depth is barotropic; a top-heavy "
            "one would be wind-driven and a sheared one density-driven.",
            fontsize=8.5, y=0.99)
        f2.tight_layout(rect=(0, 0, 1, 0.88))
        Path(a.cell_out).parent.mkdir(parents=True, exist_ok=True)
        f2.savefig(a.cell_out, dpi=150)
        print(f"[plot] {a.cell_out}")

    secs = d["sections"]
    fig, axes = plt.subplots(1, len(secs), figsize=(6.0 * len(secs), 5.2))
    axes = [axes] if len(secs) == 1 else list(axes)

    for ax, (nm, s) in zip(axes, secs.items()):
        # A zonal section stores LONGITUDE in the per-row coordinate, so the
        # axis has to be named from the section's own orientation rather than
        # from the field name -- otherwise the y axis of half the panels reads
        # "latitude" while showing longitude, which is the kind of mislabel
        # that gets quoted back later as a fact.
        zonal = s.get("orientation") == "zonal"
        lat = s["lat_per_row"]
        o = s["ours_per_row_Sv"]
        n = s["nemo_per_row_Sv"]
        ax.plot(o, lat, color=OURS, lw=1.7, label="ours")
        ax.plot(n, lat, color=NEMO, lw=1.7, label="NEMO")
        ax.plot([x - y for x, y in zip(o, n)], lat, color=DIFF, lw=1.2,
                ls="--", label="ours - NEMO")
        ax.axvline(0.0, color="0.5", lw=0.8)
        tot_o, tot_n = s["transport_ours_Sv"], s["transport_nemo_Sv"]
        where = (f"{s['lat']:.1f}N, northward" if zonal
                 else f"{s['lon_mean']:.1f}E, eastward")
        ax.set_title(f"{nm} at {where}\n"
                     f"ours {tot_o:+.2f} Sv, NEMO {tot_n:+.2f} Sv, "
                     f"difference {tot_o - tot_n:+.2f} Sv",
                     fontsize=9.5, loc="left")
        ax.set_xlabel("transport per grid cell [Sv]", fontsize=9)
        ax.grid(ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel("longitude degE" if zonal else "latitude", fontsize=9)
        nw = s["wet_cells"] if zonal else s["wet_rows"]
        na = s["cells"] if zonal else s["rows"]
        ax.text(0.98, 0.02, f"{nw} wet of {na}",
                transform=ax.transAxes, fontsize=7, color="0.35",
                ha="right", va="bottom")
    axes[0].legend(fontsize=8, loc="best")

    fig.suptitle(
        "Volume transport through sections at day 30, cell by cell. "
        "Bars sum to the section totals in the titles. Land faces verified "
        "aligned against NEMO's mask before these were computed.",
        fontsize=8.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")

    # Is the difference BROAD or CONCENTRATED? Say it with a number rather
    # than leaving the reader to eyeball the curve: how many rows carry half
    # of the total difference, out of how many wet rows.
    for nm, s in secs.items():
        nwet = s.get("wet_cells", s.get("wet_rows"))
        dif = sorted(((abs(x - y), la) for x, y, la
                      in zip(s["ours_per_row_Sv"], s["nemo_per_row_Sv"],
                             s["lat_per_row"])), reverse=True)
        tot = sum(v for v, _ in dif)
        run, k = 0.0, 0
        while k < len(dif) and run < 0.5 * tot:
            run += dif[k][0]
            k += 1
        top = ", ".join(f"{la:+.1f}" for _, la in dif[:3])
        print(f"[plot] {nm}: half the row-by-row difference sits in {k} of "
              f"{nwet} wet; largest at {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
