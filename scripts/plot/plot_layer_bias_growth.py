"""Layer-mean temperature bias against NEMO, by day, from the tracer report.

Reads the JSON ``global_tracer_content.py`` writes when it is given BOTH
``--snapshot`` and ``--nemo-gridt``, so the two sides on the plot came from one
code path, one mesh and one set of depth bins.

It recomputes nothing: the layer means are heat / volume per bin, straight out
of the report, which is why a panel here can never disagree with the table the
probe printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_H = "heat_by_depth_bin_C_m3"
_V = "volume_by_depth_bin_m3"


def layer_means(rec):
    return {k: rec[_H][k] / rec[_V][k] for k in rec[_H] if rec[_V].get(k, 0) > 0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    a = p.parse_args()

    r = json.loads(Path(a.report).read_text())
    ours = [r["snapshots"][k] for k in sorted(r["snapshots"])]
    nemo = [r["nemo"][k] for k in sorted(r["nemo"])]
    if len(ours) != len(nemo):
        raise SystemExit(f"FATAL: {len(ours)} snapshots vs {len(nemo)} NEMO "
                         "records -- the two sides must be paired by day")
    days = [int(k.replace("day", "")) for k in sorted(r["nemo"])]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    layers = list(layer_means(ours[0]))
    for L in layers:
        bias = [layer_means(o)[L] - layer_means(n)[L] for o, n in zip(ours, nemo)]
        ax.plot(days, bias, marker="o", label=L)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("day"); ax.set_ylabel("layer-mean T bias vs NEMO [$^\\circ$C]")
    ax.set_title("bias by layer")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    col = [o["mean_T_C"] - n["mean_T_C"] for o, n in zip(ours, nemo)]
    ax2.plot(days, col, marker="s", color="k")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("day"); ax2.set_ylabel("column-mean T bias [$^\\circ$C]")
    # Fixed to the top panel's scale: a column bias that merely LOOKS large on
    # its own axis is how a redistribution gets reported as a heat source.
    ax2.set_ylim(ax.get_ylim())
    ax2.set_title("whole column, same scale")
    ax2.grid(alpha=0.3)

    if a.title:
        fig.suptitle(a.title)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"[plot] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
