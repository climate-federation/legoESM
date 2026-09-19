"""The tracer-variance budget as a figure: who destroys variance, and do we match.

Two questions, three panels.

WHO DESTROYS IT (panels a, b) -- NEMO's own archived per-operator trends,
turned into chi_op = 2*integral(tracer' * (dtracer/dt)_op) dV by
tracer_variance_budget_nemo.py. Terms span four decades and carry BOTH signs,
so they are drawn as |chi| on a LOG axis with the sign carried by colour and
printed in the label. A signed linear axis would render every term except
solar heating as a zero-length bar; a symlog axis would draw bars whose
visual length is not proportional to anything. Log-of-magnitude plus an
explicit sign is the honest rendering, and it is why the sign is never left
to the colour alone.

DO WE MATCH IT (panel c) -- our vertical-mixing closure against NEMO's, both
evaluated on NEMO's OWN restart state, so the comparison carries no
state-difference confound. The shaded band is the range PRE-REGISTERED before
the number was computed: our diffusivity had already been measured at
0.89-1.49x NEMO's band by band, so a faithful closure had to land there.

The panel-c bars are chi_zdf(theta) only. Advection, GM and isoneutral
diffusion are NOT in the Mode-A comparison and our side has no per-operator
tracer tendency archive at all, so panels a and b show NEMO alone for those
operators. The figure says so on its face rather than leaving the reader to
assume the pairing extends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Plain-language names: the figure is read by people who do not know NEMO's
# four-letter trend codes, and the code is kept alongside so it stays greppable.
PRETTY = {
    "ttrd_zdf": "vertical mixing\n(ttrd_zdf)",
    "ttrd_totad": "advection\n(ttrd_totad)",
    "ttrd_ldf": "lateral diffusion\n(ttrd_ldf, isoneutral inside)",
    "ttrd_bbl": "bottom boundary layer\n(ttrd_bbl)",
    "ttrd_qsr": "solar heating\n(ttrd_qsr)",
    "strd_zdf": "vertical mixing\n(strd_zdf)",
    "strd_totad": "advection\n(strd_totad)",
    "strd_ldf": "lateral diffusion\n(strd_ldf, isoneutral inside)",
    "strd_bbl": "bottom boundary layer\n(strd_bbl)",
}
DESTROY = "#1f5fa9"
CREATE = "#c0392b"


def _ops_panel(ax, terms, title, units):
    """Horizontal |chi| bars, log axis, sign in the colour AND the label."""
    items = sorted(terms.items(), key=lambda kv: abs(kv[1]))
    ys = range(len(items))
    mags = [abs(v) for _, v in items]
    cols = [DESTROY if v < 0 else CREATE for _, v in items]
    ax.barh(list(ys), mags, color=cols, height=0.62)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([PRETTY.get(k, k) for k, _ in items], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel(f"|chi|  [{units}]", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    for y, (_, v) in zip(ys, items):
        ax.text(abs(v) * 1.35, y, f"{'-' if v < 0 else '+'}{abs(v):.3g}",
                va="center", fontsize=7,
                color=DESTROY if v < 0 else CREATE)
    ax.set_xlim(min(mags) * 0.25, max(mags) * 12.0)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nemo-budget", required=True,
                   help="variance_budget.json from tracer_variance_budget_nemo.py")
    p.add_argument("--ours-json", required=True,
                   help="tendency_match_rec*.json carrying variance_budget_zdf_theta")
    p.add_argument("--band", default="0.89,1.49",
                   help="pre-registered ratio band, lo,hi (use the = form: "
                        "argparse eats values that start with a minus)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    nb = json.loads(Path(a.nemo_budget).read_text())
    ours = json.loads(Path(a.ours_json).read_text())["variance_budget_zdf_theta"]
    lo, hi = (float(x) for x in a.band.split(","))
    eb = nb["daily_mean_over_exact"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    _ops_panel(axes[0], nb["operators"]["theta"]["terms"],
               "(a) NEMO: who changes TEMPERATURE variance", "K^2 m^3 / s")
    _ops_panel(axes[1], nb["operators"]["salt"]["terms"],
               "(b) NEMO: who changes SALINITY variance", "psu^2 m^3 / s")

    ax = axes[2]
    co, cn, ratio = ours["chi_ours"], ours["chi_nemo"], ours["ratio"]
    ax.bar([0, 1], [abs(co), abs(cn)], color=["#2e8b57", DESTROY], width=0.55)
    ax.axhspan(abs(cn) * lo, abs(cn) * hi, color="0.82", zorder=0)
    ax.text(0.5, abs(cn) * hi * 1.02,
            f"pre-registered band {lo:g}-{hi:g}x", ha="center", fontsize=7.5,
            color="0.35")
    for x, v in ((0, co), (1, cn)):
        ax.text(x, abs(v) * 1.02, f"{v:.4g}", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ours", "NEMO"], fontsize=9)
    ax.set_ylabel("|chi| vertical mixing, temperature  [K^2 m^3 / s]", fontsize=8)
    ax.set_title(f"(c) same state, same volumes: ratio {ratio:.4f}",
                 fontsize=9, loc="left")
    ax.set_ylim(0, abs(cn) * hi * 1.22)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Tracer-variance decay by operator. Blue destroys variance, red creates it. "
        "Panel (c) is VERTICAL MIXING ONLY -- advection, GM and isoneutral "
        "diffusion are not in the state-matched comparison.",
        fontsize=8.5, y=0.985)
    fig.text(0.005, 0.012,
             f"NEMO terms are daily means; the approximation's measured error bar "
             f"(daily-mean / exact, on vertical mixing) is {eb['theta']:.4f} theta, "
             f"{eb['salt']:.4f} salt. Advection is NOT covered by that bar -- it "
             f"moved 14x between two runs of the same instrument, so read it as "
             f"indicative. Cell count in (c): {ours['n_cells']}.",
             fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")
    print(f"[plot] theta terms {len(nb['operators']['theta']['terms'])}, "
          f"salt {len(nb['operators']['salt']['terms'])}, ratio {ratio:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
