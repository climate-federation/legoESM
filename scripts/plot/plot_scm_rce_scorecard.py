#!/usr/bin/env python
"""Journal figures for the SCM-RCE convection tuning, in the PR #1697 style.

Three figures, each vector PDF + 300-DPI PNG, Okabe-Ito colourblind-safe:

* fig1_scores   -- default -> tuned JOINT score per scheme, ranked, with
                   2-seed error bars (dumbbell).
* fig2_percase  -- per-RCE (295/300/305 K) tuned thermo score, schemes x cases
                   heatmap.  The RCE analog of the LES dry/moist split: it shows
                   where a scheme's skill is SST-dependent (or fails a case).
* fig3_profiles -- tuned SCM vs SAM CRM T and q_v at each of the three RCE cases.

matplotlib + numpy only, no model import.  fig1 reads the f0joint arm
checkpoints (prior/tuned joint scores per seed); fig2/fig3 read the per-SST
JSONs (scm_rce_per_sst_profiles.py).  Type-42 fonts so PDFs stay editable.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.titlesize": 11,
})
# Okabe-Ito
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
      "#0072B2", "#D55E00", "#CC79A7", "#999999", "#44AA99"]
BLOWUP = np.inf
PLOT_TOP_KM = 15.0


def _save(fig, out_prefix, name):
    for ext in ("pdf", "png"):
        p = Path(f"{out_prefix}_{name}.{ext}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def _joint_scores(arm_root, seeds):
    """{scheme: {'prior':[..], 'tuned':[..]}} across seeds (joint 3-SST mean)."""
    out = {}
    for s in seeds:
        for p in glob.glob(str(arm_root / f"arm_thermo_physical_f0joint_seed{s}"
                                / "scheme_*.json")):
            d = json.loads(Path(p).read_text())
            r = out.setdefault(d["scheme"], {"prior": [], "tuned": []})
            r["prior"].append(d["prior"]["thermo_score"])
            r["tuned"].append(d["tuned"]["thermo_score"])
    return out


def fig1_scores(arm_root, seeds, out_prefix):
    d = _joint_scores(arm_root, seeds)
    names = sorted(d, key=lambda n: np.mean(d[n]["tuned"]))
    pri = np.array([np.mean(d[n]["prior"]) for n in names])
    tun = np.array([np.mean(d[n]["tuned"]) for n in names])
    tun_e = np.array([np.std(d[n]["tuned"]) for n in names])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(len(names)):
        ax.plot([pri[i], tun[i]], [y[i], y[i]], "-", color="#BBBBBB", lw=1.5,
                zorder=1)
    ax.scatter(pri, y, s=42, color=OI[8], label="default", zorder=2)
    ax.errorbar(tun, y, xerr=tun_e, fmt="o", ms=7, color=OI[5],
                label="tuned (2-seed mean +/- sd)", zorder=3, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([f"{i+1}. {n}" for i, n in
                                          enumerate(names)])
    ax.invert_yaxis()
    ax.set_xlabel("joint thermo score over 295/300/305 K (lower = closer to CRM)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("SCM convection tuned jointly across three RCE cases")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out_prefix, "scores")
    return names


def _per_sst(in_dir, glob_pat):
    data = {}
    for p in sorted(glob.glob(str(in_dir / glob_pat))):
        d = json.loads(Path(p).read_text())
        data[d["scheme"]] = d["per_sst"]
    return data


def fig2_percase(in_dir, ssts, order, out_prefix):
    data = _per_sst(in_dir, "per_sst_*.json")
    schemes = [n for n in order if n in data] or sorted(data)
    M = np.full((len(schemes), len(ssts)), np.nan)
    for i, n in enumerate(schemes):
        for j, s in enumerate(ssts):
            v = data[n].get(str(s), {}).get("thermo", np.nan)
            M[i, j] = np.nan if v == BLOWUP else v
    fig, ax = plt.subplots(figsize=(4.6, 6))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r",
                   vmin=0, vmax=np.nanpercentile(M, 95))
    ax.set_xticks(range(len(ssts)))
    ax.set_xticklabels([f"{s} K" for s in ssts])
    ax.set_yticks(range(len(schemes))); ax.set_yticklabels(schemes)
    for i in range(len(schemes)):
        for j in range(len(ssts)):
            txt = "fail" if np.isnan(M[i, j]) else f"{M[i, j]:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if (np.isnan(M[i, j]) or
                                      M[i, j] > np.nanmean(M)) else "black")
    fig.colorbar(im, ax=ax, label="tuned thermo score", fraction=0.06)
    ax.set_title("Per-RCE tuned score\n(each scheme at each SST)")
    _save(fig, out_prefix, "percase")


def fig3_profiles(in_dir, ssts, order, out_prefix):
    data = _per_sst(in_dir, "per_sst_*.json")
    fig, axes = plt.subplots(2, len(ssts), figsize=(4 * len(ssts), 9),
                             sharey=True)
    schemes = [n for n in order if n in data] or sorted(data)
    colors = {n: OI[i % len(OI)] for i, n in enumerate(schemes)}
    for j, sst in enumerate(ssts):
        axT, axQ = axes[0, j], axes[1, j]
        ref = None
        for n in schemes:
            ps = data[n].get(str(sst))
            if not ps or not ps.get("T_profile"):
                continue
            z = np.asarray(ps["z_m"]) / 1000.0
            m = z <= PLOT_TOP_KM
            ref = ps
            axT.plot(np.asarray(ps["T_profile"])[m], z[m], color=colors[n],
                     lw=1.3, label=n)
            axQ.plot(np.asarray(ps["qv_profile"])[m] * 1000.0, z[m],
                     color=colors[n], lw=1.3)
        if ref is not None:
            z = np.asarray(ref["z_m"]) / 1000.0; m = z <= PLOT_TOP_KM
            axT.plot(np.asarray(ref["T_ref"])[m], z[m], "k", lw=2.4,
                     label="SAM CRM", zorder=10)
            axQ.plot(np.asarray(ref["qv_ref"])[m] * 1000.0, z[m], "k", lw=2.4,
                     zorder=10)
        axT.set_title(f"RCE {sst} K")
        axQ.set_xlabel("q$_v$ [g/kg]")
        if j == 0:
            axT.set_ylabel("height [km]"); axQ.set_ylabel("height [km]")
        axT.set_ylim(0, PLOT_TOP_KM)
        for ax in (axT, axQ):
            ax.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=6, loc="upper right", ncol=1)
    axes[0, 0].set_xlabel("")
    axes[0, len(ssts) // 2].set_title(
        f"RCE {ssts[len(ssts)//2]} K   (top: temperature [K])")
    fig.suptitle("Joint-tuned SCM convection vs SAM CRM at each RCE case",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, out_prefix, "profiles")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-root", type=Path, required=True)
    ap.add_argument("--per-sst-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", default=["20260815", "20260816"])
    ap.add_argument("--ssts", nargs="+", type=int, default=[295, 300, 305])
    ap.add_argument("--out-prefix", type=Path, required=True)
    a = ap.parse_args(argv)
    order = fig1_scores(a.arm_root, a.seeds, a.out_prefix)
    fig2_percase(a.per_sst_dir, a.ssts, order, a.out_prefix)
    fig3_profiles(a.per_sst_dir, a.ssts, order, a.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
