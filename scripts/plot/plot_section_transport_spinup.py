"""WHEN does a section transport diverge from NEMO's?

A single day-30 number says ours is stronger; it cannot say whether that was
true from the start, grew steadily, or appeared at one moment. Those have
different causes, and the third one is testable against things that happen at
a known time -- a viscosity-schedule segment boundary, for instance.

Reads the per-day JSONs the section probe already writes, so no rerun is
needed to change how this is drawn.

SAMPLING, stated on the figure because it bounds every reading of it: our
points are SNAPSHOTS at the named day, NEMO's are FIVE-DAY MEANS over the
interval ending there. A five-day mean lags and smooths, so NEMO's curve is
biased low wherever the quantity is still rising -- most severely at day 5,
where a ramp from rest averages to about half its endpoint. Compare SHAPE and
TIMING, never level, and never read the day-5 gap as a spin-up rate.
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="directory of transports_d*.json")
    p.add_argument("--tag", default="_halfvisc", help="filename tag to read")
    p.add_argument("--days", default="5,10,15,20,25,30")
    p.add_argument("--mark-days", default="",
                   help="comma-separated days to mark with a dashed rule "
                        "(e.g. viscosity-schedule segment boundaries)")
    p.add_argument("--mark-label", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    days = [int(d) for d in a.days.split(",")]
    d = {}
    for day in days:
        f = Path(a.dir) / f"transports_d{day}{a.tag}.json"
        if not f.is_file():
            raise SystemExit(f"FATAL: missing {f} -- refusing to draw a curve "
                             "with a hole in it")
        d[day] = json.loads(f.read_text())["sections"]

    secs = list(d[days[0]])
    marks = [float(x) for x in a.mark_days.split(",") if x.strip()]
    fig, axes = plt.subplots(1, len(secs), figsize=(4.1 * len(secs), 4.4))
    axes = [axes] if len(secs) == 1 else list(axes)

    for ax, nm in zip(axes, secs):
        o = [d[day][nm]["transport_ours_Sv"] for day in days]
        n = [d[day][nm]["transport_nemo_Sv"] for day in days]
        for x in marks:
            ax.axvline(x, color="0.55", lw=1.0, ls="--", zorder=0)
        ax.plot(days, o, color=OURS, lw=1.8, marker="o", ms=4, label="ours")
        ax.plot(days, n, color=NEMO, lw=1.8, marker="s", ms=4, label="NEMO")
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_title(f"{nm}\nday 30: ours {o[-1]:+.1f}, NEMO {n[-1]:+.1f} Sv",
                     fontsize=9.5, loc="left")
        ax.set_xlabel("day", fontsize=9)
        ax.grid(ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("volume transport [Sv]", fontsize=9)
    axes[0].legend(fontsize=8, loc="best")

    sub = ("Ours are SNAPSHOTS at the day; NEMO's are FIVE-DAY MEANS ending "
           "there, so NEMO lags wherever the transport is still rising -- "
           "read timing and shape, not level.")
    if marks and a.mark_label:
        sub += f"  Dashed: {a.mark_label}."
    fig.suptitle("When does each section diverge from NEMO?\n" + sub,
                 fontsize=8.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")

    # Say WHEN in numbers, not only in ink: the interval over which the gap to
    # NEMO grows the most, and what the gap is at each end of it.
    for nm in secs:
        gap = [d[day][nm]["transport_ours_Sv"] - d[day][nm]["transport_nemo_Sv"]
               for day in days]
        jumps = [(abs(gap[i + 1]) - abs(gap[i]), days[i], days[i + 1])
                 for i in range(len(days) - 1)]
        big, d0, d1 = max(jumps)
        print(f"[when] {nm}: |gap| grows most over days {d0}-{d1} "
              f"(+{big:.2f} Sv); gap {gap[0]:+.2f} at day {days[0]}, "
              f"{gap[-1]:+.2f} at day {days[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
