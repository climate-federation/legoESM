"""Publication-ready figures for the LES->SCM turbulence tuning campaign.

Three figures, each written as vector PDF + 300-DPI PNG:
  fig1_scores        default vs tuned joint score, ranked, seed error bars
  fig2_percase       per-case tuned normalized error heatmap (9 x 8)
  fig3_profiles      SCM(tuned) vs LES profiles for representative cases

Reads the aggregated scores (all_results.json) and the per-seed profile npz.
No model import -- numpy + matplotlib only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- publication style ------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,          # embed as editable TrueType, not Type-3
    "ps.fonttype": 42,
})

# Okabe-Ito colourblind-safe, extended to 9
CB = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
      "#D55E00", "#F0E442", "#000000", "#999999"]
CASES = ["ekman", "gabls1", "cbl", "wangara", "bomex", "rico", "astex", "dycoms"]
CASE_LABEL = {"ekman": "Ekman", "gabls1": "GABLS1", "cbl": "CBL",
              "wangara": "Wangara", "bomex": "BOMEX", "rico": "RICO",
              "astex": "ASTEX", "dycoms": "DYCOMS"}


def load_scores(p):
    rows = json.load(open(p))
    rows.sort(key=lambda r: r["t"] if "t" in r else r["tuned"])
    return rows


def _g(r, *ks):
    for k in ks:
        if k in r:
            return r[k]
    raise KeyError(ks)


# ---- fig 1: default -> tuned dumbbell ---------------------------------------
def fig_scores(rows, out):
    names = [_g(r, "s", "scheme") for r in rows]
    dflt = [_g(r, "d", "default") for r in rows]
    tun = [_g(r, "t", "tuned") for r in rows]
    std = [_g(r, "sd", "tuned_std") for r in rows]
    y = np.arange(len(rows))[::-1]                      # best at top

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for yi, d, t in zip(y, dflt, tun):
        ax.plot([t, d], [yi, yi], color="#c7cdd2", lw=1.4, zorder=1,
                solid_capstyle="round")
    ax.scatter(dflt, y, s=22, facecolor="white", edgecolor="#98a2ac",
               linewidth=1.0, zorder=2, label="default")
    ax.errorbar(tun, y, xerr=std, fmt="o", ms=5.0, color="#0072B2",
                ecolor="#0072B2", elinewidth=1.0, capsize=2.0, zorder=3,
                label="tuned")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Joint normalized error  (mean of per-case)")
    ax.set_xlim(0, max(dflt) * 1.06)
    ax.axvline(0, color="#dfe3e7", lw=0.7, zorder=0)
    ax.grid(axis="x", color="#eef1f3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.4)
    ax.margins(y=0.04)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig1_scores.{ext}")
    plt.close(fig)


# ---- fig 2: per-case heatmap ------------------------------------------------
def fig_percase(rows, out):
    names = [_g(r, "s", "scheme") for r in rows]
    M = np.array([[_g(r, "pct", "pc_tuned")[c] for c in CASES] for r in rows])
    cmap = LinearSegmentedColormap.from_list(
        "err", ["#0f766e", "#4bab7e", "#b9c24a", "#e6ad3a", "#dd7a34", "#c0392b"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=3.0, aspect="auto")
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels([CASE_LABEL[c] for c in CASES], rotation=40, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.6, color="white" if v >= 1.0 else "#12201a")
    # dry / moist divider
    ax.axvline(3.5, color="white", lw=1.6)
    ax.set_xticks(np.arange(-.5, len(CASES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, extend="max")
    cb.set_label("tuned normalized error", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.text(1.5, -0.9, "dry", ha="center", fontsize=7, color="#5c6c79")
    ax.text(5.5, -0.9, "moist", ha="center", fontsize=7, color="#5c6c79")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig2_percase.{ext}")
    plt.close(fig)


# ---- fig 3: profiles vs LES -------------------------------------------------
def _load_profiles(indir, clubbdir, case):
    d = dict(np.load(f"{indir}/profiles_{case}.npz", allow_pickle=True))
    if clubbdir and Path(f"{clubbdir}/profiles_{case}.npz").exists():
        dc = np.load(f"{clubbdir}/profiles_{case}.npz", allow_pickle=True)
        for k in dc.keys():
            if k.startswith("scm_clubb_"):
                d[k] = dc[k]
    return d


def fig_profiles(indir, clubbdir, out):
    # (case, variable, x-label) panels
    panels = [("gabls1", "theta", r"$\theta$ (K)"),
              ("wangara", "theta", r"$\theta$ (K)"),
              ("bomex", "theta", r"$\theta$ (K)"),
              ("bomex", "qv", r"$q_v$ (g kg$^{-1}$)")]
    schemes = ["ysu", "clubb", "smagorinsky", "louis", "mynn25", "edmf"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=False)
    for ax, (case, var, xlab) in zip(axes, panels):
        d = _load_profiles(indir, clubbdir, case)
        mask = d["mask"].astype(bool)
        z = d["z_scm"][mask]
        les = d[f"les_scmlev_{var}"][mask]
        sc = 1e3 if var == "qv" else 1.0
        ax.plot(les * sc, z, color="black", lw=1.6, label="LES", zorder=5)
        for s, c in zip(schemes, CB):
            key = f"scm_{s}_tuned_{var}"
            if key in d:
                ax.plot(d[key][mask] * sc, z, color=c, lw=1.0, label=s)
        ax.set_xlabel(xlab)
        ax.set_title(CASE_LABEL[case], fontsize=8.5)
        ax.grid(color="#eef1f3", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("height (m)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=7, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig3_profiles.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--profiles", required=True, help="8-scheme seed dir")
    ap.add_argument("--clubb-profiles", default=None, help="clubb seed dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = load_scores(a.scores)
    fig_scores(rows, a.out)
    fig_percase(rows, a.out)
    fig_profiles(a.profiles, a.clubb_profiles, a.out)
    print(f"wrote fig1_scores, fig2_percase, fig3_profiles (pdf+png) to {a.out}")


if __name__ == "__main__":
    main()
