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
    a = p.parse_args()

    d = json.loads(Path(a.json).read_text())
    secs = d["sections"]
    fig, axes = plt.subplots(1, len(secs), figsize=(6.0 * len(secs), 5.2))
    axes = [axes] if len(secs) == 1 else list(axes)

    for ax, (nm, s) in zip(axes, secs.items()):
        lat = s["lat_per_row"]
        o = s["ours_per_row_Sv"]
        n = s["nemo_per_row_Sv"]
        ax.plot(o, lat, color=OURS, lw=1.7, label="ours")
        ax.plot(n, lat, color=NEMO, lw=1.7, label="NEMO")
        ax.plot([x - y for x, y in zip(o, n)], lat, color=DIFF, lw=1.2,
                ls="--", label="ours - NEMO")
        ax.axvline(0.0, color="0.5", lw=0.8)
        tot_o, tot_n = s["transport_ours_Sv"], s["transport_nemo_Sv"]
        ax.set_title(f"{nm} at {s['lon_mean']:.1f}E\n"
                     f"ours {tot_o:+.2f} Sv, NEMO {tot_n:+.2f} Sv, "
                     f"difference {tot_o - tot_n:+.2f} Sv",
                     fontsize=9.5, loc="left")
        ax.set_xlabel("transport per grid row [Sv]", fontsize=9)
        ax.grid(ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.text(0.98, 0.02, f"{s['wet_rows']} wet of {s['rows']} rows",
                transform=ax.transAxes, fontsize=7, color="0.35",
                ha="right", va="bottom")
    axes[0].set_ylabel("latitude", fontsize=9)
    axes[0].legend(fontsize=8, loc="best")

    fig.suptitle(
        "Volume transport through meridional sections at day 30, row by row. "
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
              f"{s['wet_rows']} wet rows; largest at latitudes {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
