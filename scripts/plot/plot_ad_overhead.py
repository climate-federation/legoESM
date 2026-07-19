#!/usr/bin/env python
"""Paper figure: what reverse-mode AD costs (single A100).

Companion to ``plot_routeb_campaign_paper.py``.  Those figures show FORWARD
throughput scaling across GPUs; this one shows the cost of the gradient on ONE
device -- the number that turns "gradients are available" into "gradients are
affordable".

Reads the JSON written by ``scripts/bench/bench_ad_overhead.py`` (one file per
grid/resolution case, e.g. from ``ad_overhead_gpu.pbs``).

THE MESSAGE, in two panels:
  (a) wall time relative to a forward step -- a gradient costs ~5x, and
      CHECKPOINTING IS CHEAPER THAN STORING THE TAPE (the counter-intuitive
      result: remat is usually sold as trading time FOR memory).
  (b) peak reverse-mode tape -- checkpointing cuts it ~11x, which is what
      makes gradients at the finest resolutions fit on one GPU at all.

Panels are SEPARATE rather than one dual-axis chart: time and memory are
different scales and a twin y-axis would invite false visual correlation.

Forward is drawn as a REFERENCE LINE, not a third bar -- it is the baseline
both other modes are expressed against, so only two series are encoded by
colour.  Palette is a validated categorical pair (all six checks pass:
lightness band, chroma floor, CVD separation dE 21.5 protan, normal-vision
floor 26.4, contrast >= 3:1) and is DELIBERATELY not the Blues resolution ramp
used by the scaling figures -- there, colour means resolution; reusing it for
"AD mode" would make one colour mean two things across a figure set.

Run:  python scripts/plot/plot_ad_overhead.py \
          --json-dir $SCRATCH/legoesm_scaling/ad_overhead_<stamp> \
          [--out results/paper_figs]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Validated categorical pair (see module docstring).  Not the Blues ramp.
COL_GRAD = "#2c6fbb"
COL_CKPT = "#b5562a"
INK, MUTED, RULE = "#1a1a1a", "#666666", "#bdbdbd"

# Grid label -> short display name, in a fixed order (never cycled).
GRID_LABEL = {
    "latlon": "lat-lon",
    "cubed-sphere": "cubed-sphere",
    "icosahedral": "icosahedral",
    "spectral": "spectral",
}
GRID_ORDER = ["latlon", "cubed-sphere", "icosahedral", "spectral"]


def load_cases(json_dir: str):
    """Load bench_ad_overhead JSONs into a list of per-case dicts.

    Skips any case whose forward baseline or grad mode failed -- a ratio
    cannot be plotted without both, and silently drawing a partial bar would
    misrepresent an OOM as a measurement.
    """
    paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not paths:
        raise SystemExit(f"no JSON files found in {json_dir}")

    cases = []
    skipped = []
    for path in paths:
        with open(path) as fh:
            d = json.load(fh)
        modes = d.get("modes", {})
        fwd = modes.get("forward", {})
        if not fwd.get("ok"):
            skipped.append((os.path.basename(path), "forward failed"))
            continue
        entry = {
            "grid": d["grid"], "resolution": d["resolution"],
            "n_levels": d.get("n_levels"), "steps": d.get("steps"),
            "precision": d.get("precision"), "physics": d.get("physics_level"),
            "forward_tape": fwd.get("temp_bytes"),
            "modes": {},
        }
        for mode in ("grad", "grad_ckpt"):
            m = modes.get(mode, {})
            if m.get("ok"):
                entry["modes"][mode] = {
                    "ratio": m.get("ratio_vs_forward"),
                    "tape": m.get("temp_bytes"),
                }
            else:
                # An OOM here is a RESULT (plain BPTT does not fit) -- keep it
                # so the figure can mark it rather than silently omit the bar.
                entry["modes"][mode] = {"ratio": None, "tape": None,
                                        "failed": m.get("error", "failed")}
        if not any(v.get("ratio") for v in entry["modes"].values()):
            skipped.append((os.path.basename(path), "no gradient mode ran"))
            continue
        cases.append(entry)

    cases.sort(key=lambda c: (GRID_ORDER.index(c["grid"])
                              if c["grid"] in GRID_ORDER else 99,
                              c["resolution"]))
    return cases, skipped


def _style(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="y", color="#ececec", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, color="#999999")


def make_figure(cases, out, filename="fig3_ad_cost"):
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"{GRID_LABEL.get(c['grid'], c['grid'])}\n"
              f"{c['resolution']} · L{c['n_levels']}" for c in cases]
    x = np.arange(len(cases), dtype=float)
    # 2px-equivalent gap between adjacent bars in a pair.
    width = 0.34
    gap = 0.02

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # ---- (a) wall time relative to forward -------------------------------
    axa.axhline(1.0, color=RULE, linewidth=1.2, linestyle=(0, (4, 3)),
                zorder=2)
    axa.annotate("forward = 1×", xy=(len(cases) - 0.5, 1.0), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=MUTED,
                 ha="right", va="bottom")

    for offset, key, col, name in (
            (-(width / 2 + gap), "grad", COL_GRAD, "full BPTT"),
            (+(width / 2 + gap), "grad_ckpt", COL_CKPT, "checkpointed")):
        vals = [c["modes"][key]["ratio"] for c in cases]
        pos = x + offset
        drawn = [(p, v) for p, v in zip(pos, vals) if v is not None]
        if drawn:
            axa.bar([p for p, _ in drawn], [v for _, v in drawn], width,
                    color=col, label=name, zorder=5)
        for p, v in zip(pos, vals):
            if v is None:
                axa.annotate("OOM", xy=(p, 0.15), fontsize=8, color=MUTED,
                             ha="center", rotation=90, va="bottom")
            else:
                axa.annotate(f"{v:.1f}×", xy=(p, v), xytext=(0, 3),
                             textcoords="offset points", fontsize=8.5,
                             color=INK, ha="center", va="bottom")

    axa.set_xticks(x)
    axa.set_xticklabels(labels, fontsize=8.5, color=INK)
    axa.set_ylabel("gradient cost (× a forward step)", fontsize=10, color=INK)
    axa.set_title("(a) wall time", fontsize=11, color=INK, loc="left")
    axa.set_ylim(0, max(
        [v for c in cases for v in
         (c["modes"]["grad"]["ratio"], c["modes"]["grad_ckpt"]["ratio"])
         if v is not None] + [1.0]) * 1.22)
    _style(axa)
    axa.legend(fontsize=8.5, frameon=False, loc="upper left")

    # ---- (b) peak reverse-mode tape --------------------------------------
    for offset, key, col, name in (
            (-(width / 2 + gap), "grad", COL_GRAD, "full BPTT"),
            (+(width / 2 + gap), "grad_ckpt", COL_CKPT, "checkpointed")):
        vals = [(c["modes"][key]["tape"] or 0) / 1e9 for c in cases]
        pos = x + offset
        drawn = [(p, v) for p, v in zip(pos, vals) if v > 0]
        if drawn:
            axb.bar([p for p, _ in drawn], [v for _, v in drawn], width,
                    color=col, label=name, zorder=5)
        for p, v in zip(pos, vals):
            if v > 0:
                txt = f"{v:.2f}" if v < 1 else f"{v:.1f}"
                axb.annotate(f"{txt} GB", xy=(p, v), xytext=(0, 3),
                             textcoords="offset points", fontsize=8,
                             color=INK, ha="center", va="bottom")

    fwd_tape = [c["forward_tape"] for c in cases if c["forward_tape"]]
    if fwd_tape:
        fwd_gb = sum(fwd_tape) / len(fwd_tape) / 1e9
        axb.axhline(fwd_gb, color=RULE, linewidth=1.2, linestyle=(0, (4, 3)),
                    zorder=2)
        axb.annotate(f"forward ({fwd_gb:.2f} GB)",
                     xy=(len(cases) - 0.5, fwd_gb), xytext=(0, 4),
                     textcoords="offset points", fontsize=8, color=MUTED,
                     ha="right", va="bottom")

    axb.set_yscale("log")
    axb.set_xticks(x)
    axb.set_xticklabels(labels, fontsize=8.5, color=INK)
    axb.set_ylabel("peak reverse-mode tape (GB, log)", fontsize=10, color=INK)
    axb.set_title("(b) memory", fontsize=11, color=INK, loc="left")
    _style(axb)
    axb.legend(fontsize=8.5, frameon=False, loc="upper left")

    meta = cases[0]
    fig.suptitle(
        f"Reverse-mode AD cost — 1× A100, {meta['steps']} steps, "
        f"{meta['precision']}, physics={meta['physics']}",
        fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = os.path.join(out, f"{filename}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", required=True,
                   help="directory of bench_ad_overhead JSON files")
    p.add_argument("--out", default="results/paper_figs")
    p.add_argument("--name", default="fig3_ad_cost")
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    cases, skipped = load_cases(a.json_dir)
    for name, why in skipped:
        print(f"SKIPPED {name}: {why}")
    written = make_figure(cases, a.out, a.name)
    print(f"plotted {len(cases)} case(s): "
          f"{', '.join(c['grid'] + str(c['resolution']) for c in cases)}")
    for path in written:
        print(f"wrote {path}")
    return written


if __name__ == "__main__":
    main()
