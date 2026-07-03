#!/usr/bin/env python
"""Cross-site FREE-RUNNING baseline-vs-fixed skill for the two land fixes
(canopy S_top**exp bare-soil evaporation resistance + Niu-2005 K(z) retention
decay), on top of Pierre's #671 merge.  Reuses the collect/skill machinery of
:mod:`plot_ec_site_xsite`.

    baseline = canopy Kelvin-h_r only, uniform K   (soil_evap_resistance_exp=0, k_decay=0)
    fixed    = S_top**3 evap resistance + K(z)=exp(-z/0.25 m)

Usage
-----
    JAX_ENABLE_X64=1 python scripts/plot/plot_ec_site_fix_compare.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_ec_site_xsite import (  # noqa: E402
    collect, _FLUXES, _METRICS, _UNIT, _set_pub_style, _ax_metric)

BASE = "diagnostics/ec_site_fix_xsite/baseline"
FIX = "diagnostics/ec_site_fix_xsite/fixed"
OUT = "diagnostics/ec_site_fix_xsite"
_LBL_B = "baseline (Kelvin-$h_r$, uniform K)"
_LBL_F = "fixed ($S_{top}^{3}$ evap + K(z) decay)"
cB, cF = "#0072B2", "#D55E00"


def main() -> int:
    _set_pub_style()
    b = {r["site"]: r for r in collect(BASE)}
    f = {r["site"]: r for r in collect(FIX)}
    common = sorted(set(b) & set(f), key=lambda s: (b[s]["pft"], s))
    if not common:
        print("no common sites"); return 1
    pft = {s: b[s]["pft"] for s in common}
    pfts = sorted(set(pft.values()))

    for metric in ("r", "rmse"):
        # ---- per-site, one panel per flux ----
        fig, axes = plt.subplots(len(_FLUXES), 1,
                                 figsize=(max(9, 0.6 * len(common)), 12), sharex=True)
        x = np.arange(len(common)); w = 0.4
        for ax, (key, label) in zip(axes, _FLUXES):
            bv = [b[s].get(f"{label}_{metric}", np.nan) for s in common]
            fv = [f[s].get(f"{label}_{metric}", np.nan) for s in common]
            ax.bar(x - w/2, bv, w, label=_LBL_B, color=cB, edgecolor="0.3", lw=0.3)
            ax.bar(x + w/2, fv, w, label=_LBL_F, color=cF, edgecolor="0.3", lw=0.3)
            _ax_metric(ax, metric, label)
            if key == "gpp":
                ax.legend(loc="lower left", ncol=2)
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([f"{s}  ({pft[s]})" for s in common], rotation=90)
        axes[0].set_title(f"Cross-site free-running: baseline vs fixed — "
                          f"{_METRICS[metric]} ({len(common)} sites)")
        p = os.path.join(OUT, f"fix_xsite_{metric}_by_site.png")
        fig.savefig(p); plt.close(fig); print(f"  plot -> {p}")

        # ---- PFT-mean ----
        fig, axes = plt.subplots(1, len(_FLUXES), figsize=(15, 3.6))
        for ax, (key, label) in zip(axes, _FLUXES):
            bm = [np.nanmean([b[s].get(f"{label}_{metric}", np.nan)
                              for s in common if pft[s] == q]) for q in pfts]
            fm = [np.nanmean([f[s].get(f"{label}_{metric}", np.nan)
                              for s in common if pft[s] == q]) for q in pfts]
            xp = np.arange(len(pfts))
            ax.bar(xp - 0.2, bm, 0.4, color=cB, edgecolor="0.3", lw=0.3, label="baseline")
            ax.bar(xp + 0.2, fm, 0.4, color=cF, edgecolor="0.3", lw=0.3, label="fixed")
            ax.set_xticks(xp); ax.set_xticklabels(pfts, rotation=45, ha="right")
            ax.set_title(label)
            ax.set_ylabel(_METRICS[metric] + (f" ({_UNIT[label]})" if metric == "rmse" else ""))
            if metric == "r":
                ax.axhline(0.8, color="0.3", ls="--", lw=0.8); ax.set_ylim(-0.45, 1.0)
            if key == "gpp":
                ax.legend(fontsize=7)
        fig.suptitle(f"Mean EC {_METRICS[metric]} by PFT — baseline vs fixed "
                     f"(free-running, best-year)", y=1.04)
        p = os.path.join(OUT, f"fix_xsite_{metric}_by_pft.png")
        fig.savefig(p); plt.close(fig); print(f"  plot -> {p}")

    # ---- summary table (Pearson r) ----
    print(f"\n=== free-running baseline -> fixed, Pearson r ({len(common)} sites) ===")
    print(f"  {'site':9s} {'pft':4s} | " + "  ".join(f"{l:^11s}" for _, l in _FLUXES))
    dsum = {l: [] for _, l in _FLUXES}
    for s in common:
        cells = []
        for _, l in _FLUXES:
            rb, rf = b[s].get(l+"_r", np.nan), f[s].get(l+"_r", np.nan)
            cells.append(f"{rb:+.2f}->{rf:+.2f}")
            if np.isfinite(rb) and np.isfinite(rf):
                dsum[l].append(rf - rb)
        print(f"  {s:9s} {pft[s]:4s} | " + "  ".join(cells))
    print("  " + "-" * 60)
    print(f"  {'MEAN dr':14s} | " + "  ".join(
        f"{np.mean(dsum[l]):+.2f}      " if dsum[l] else "  n/a      " for _, l in _FLUXES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
