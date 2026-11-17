"""Stratification at the deep-convection sites: ours vs NEMO, by depth.

One panel per site, N2 against depth, both models on the same axes. Reads the
JSON the probe writes -- never a regex over the job log, because a parsed
table is not data.

WHY LOG-X AND WHY SIGN IS DRAWN SEPARATELY. N2 spans 1e-7 to 5e-4 across
these columns, so a linear axis shows only the surface. It also changes SIGN:
a negative value means the column is statically unstable, which is the single
most interesting thing a profile can say here and must not be swallowed by
taking an absolute value. So magnitude goes on the log axis and any
statically-unstable level is marked with an open circle at the axis edge, per
model.

WHAT THE PANELS MAY AND MAY NOT BE READ FOR. Our state is an INSTANTANEOUS
day-30 snapshot; NEMO's is a monthly mean over the same days. During boreal
winter that mismatch manufactures exactly the difference the northern panels
show -- NEMO's mean averages away intermittent overturning that our snapshot
can catch -- so those panels are labelled SAMPLING-CONFOUNDED on their face
and carry no claim. The austral-summer panel is the opposite case: a mean
SMOOTHS a stratification peak rather than sharpening one, and summer
stratification grows through January, so the confound works against the
difference there and the panel is quotable.
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
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--confounded", default="",
                   help="comma-separated site names to mark as "
                        "sampling-confounded on the figure face")
    a = p.parse_args()

    d = json.loads(Path(a.json).read_text())
    sites = d["sites"]
    flagged = {s for s in a.confounded.split(",") if s}
    unknown = flagged - set(sites)
    if unknown:
        raise SystemExit(f"FATAL: --confounded names no such site: {unknown}")

    fig, axes = plt.subplots(1, len(sites), figsize=(4.1 * len(sites), 5.6),
                             sharey=True)
    axes = [axes] if len(sites) == 1 else list(axes)

    for ax, (nm, s) in zip(axes, sites.items()):
        z = [r["depth_m"] for r in s["levels"]]
        for key, col, lab in (("n2_ours", OURS, "ours"),
                              ("n2_nemo", NEMO, "NEMO")):
            v = [r[key] for r in s["levels"]]
            ax.plot([abs(x) for x in v], z, color=col, lw=1.6, label=lab)
            # Statically unstable levels: magnitude is on the axis, so the
            # sign is drawn instead of being silently discarded.
            unst = [(abs(x), zz) for x, zz in zip(v, z) if x < 0]
            if unst:
                ax.scatter([u[0] for u in unst], [u[1] for u in unst],
                           facecolors="none", edgecolors=col, s=34, lw=1.2,
                           zorder=5)
        ax.set_xscale("log")
        ax.set_title(nm + ("\nSAMPLING-CONFOUNDED" if nm in flagged else ""),
                     fontsize=10, loc="left",
                     color="#b03030" if nm in flagged else "black")
        ax.set_xlabel("|N$^2$|  [1/s$^2$]", fontsize=9)
        ax.grid(ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.text(0.03, 0.02,
                f"{s['columns']} columns\n{s['seafloor_excluded']} sea-floor "
                "interfaces excluded", transform=ax.transAxes, fontsize=6.5,
                color="0.35", va="bottom")
    axes[0].set_ylabel("depth [m]", fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Stratification at the deep-convection sites, day 30. Open circles "
        "mark statically UNSTABLE levels. Ours is an instantaneous snapshot; "
        "NEMO is a monthly mean over the same days.", fontsize=8.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")
    for nm, s in sites.items():
        deep = [r for r in s["levels"] if r["depth_m"] > 350.0]
        if deep:
            rr = [r["n2_ours"] / r["n2_nemo"] for r in deep
                  if r["n2_nemo"] != 0]
            print(f"[plot] {nm}: deep(>350 m) ratio "
                  f"{min(rr):.3f}-{max(rr):.3f} over {len(rr)} levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
