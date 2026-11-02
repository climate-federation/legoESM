"""Figures 1-4 for the differentiable-MARBL (legoESM BGC) paper.

  Fig. 1  model structure: inputs, legoESM core and ocean physics, the BGC
          complexity ladder, and the differentiable (JAX) layer
  Fig. 2  verification in the BATS column: total-P and total-N budgets,
          upper-ocean Chl section, seasonal NPP by phytoplankton type
  Fig. 3  gradients: reverse/forward mode vs finite differences, and the
          parameter-sensitivity (elasticity) matrix
  Fig. 4  identical-twin calibration: loss, parameter trajectories, recovery

Inputs (defaults):
  results/marbl_column_experiments/{gradcheck.json, sensitivity.npz, twin_seed*.json}
  results/marbl_column_bats/bats_woa_5yr.nc        (Fig. 2; --column-run to change)
Output: results/paper_figs/fig{1..4}.png and .pdf

Usage:  python scripts/plot_marbl_paper_figs.py [--figs 1 2 3 4]
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

EXP = Path("results/marbl_column_experiments")
OUT = Path("results/paper_figs")

# reference palette (validated categorical order; blue<->red diverging, gray midpoint)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
DIVERGING = LinearSegmentedColormap.from_list("bl_rd", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a8f", "#c0302f"])
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 110,
})


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="right")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png", "and .pdf")


# ----------------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

    def box(x, y, w, h, title, lines=(), fc="#f6f5f2", ec=MUTED, tc=INK, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=8.3,
                fontweight="bold" if bold else "normal", color=tc)
        for i, s in enumerate(lines):
            ax.text(x + w / 2, y + h - 5.0 - 3.0 * i, s, ha="center", va="top", fontsize=7.0, color=INK2)

    def arrow(p, q, color=MUTED, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=color, ls=ls))

    # inputs
    box(2, 50, 30, 12, "Forcing", ["JRA55-do v1.6 repeat year 1990/91", "xCO2, dust/Fe, N deposition"])
    box(35, 50, 30, 12, "Initial state", ["WOA18 NO3, PO4, SiO3, O2", "GLODAPv2 DIC, ALK; WOA T, S"])
    box(68, 50, 30, 12, "Evaluation / calibration data", ["ESA-CCI Chl, NPP products", "WOA, GLODAP, pCO2 products"])

    # physics
    box(2, 21, 30, 25, "legoESM ocean physics",
        ["1-D column: KPP, Jerlov SW,", "inertial currents, T/S relaxation", "",
         "3-D lat-lon C-grid (1°, 0.5°):", "TVD advection, K_h diffusion,", "implicit KPP mixing, (GM/Redi)"])
    # ladder (bottom = simplest)
    ax.text(51.5, 46.9, "BGC complexity ladder  ↑", ha="center", va="center",
            fontsize=8.3, fontweight="bold")
    rungs = [("NPZD", "6 tracers"), ("NPZDv2 / MARBL-lite", "18 tracers, 2 PFTs"),
             ("MARBL-CESM simplified", "18 tracers"), ("MARBL (CESM2 base)", "32 tracers, 3 PFTs, C/N/P/Fe/Si, O2")]
    for i, (t, s) in enumerate(rungs):
        y, h = 21 + 6.2 * i, 5.4
        full = i == 3
        ax.add_patch(FancyBboxPatch((36, y), 31, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc="#e8f1fc" if full else "#f6f5f2", ec=C[0] if full else MUTED,
                                    lw=1.1 if full else 0.9))
        ax.text(51.5, y + h * 0.66, t, ha="center", va="center", fontsize=7.8, fontweight="bold",
                color=C[0] if full else INK)
        ax.text(51.5, y + h * 0.25, s, ha="center", va="center", fontsize=6.6, color=INK2)

    # outputs
    box(72, 21, 26, 25, "Diagnostics",
        ["NPP by PFT, export at 100 m", "air-sea CO2 and O2 fluxes", "pCO2, pH, surface Chl", "",
         "budgets: total P, N, C, Fe"])

    # JAX layer
    ax.add_patch(FancyBboxPatch((2, 2), 96, 14, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#eef5fd", ec=C[0], lw=1.0))
    ax.text(50, 14.0, "Differentiable layer (JAX)", ha="center", va="top", fontsize=8.3,
            fontweight="bold", color=C[0])
    items = ["jit / lax.map over\nwater columns", "jax.grad, jax.jvp:\nexact derivatives",
             "jax.jacrev: sensitivity of\nall 133 parameters", "Adam on bounded log-\nscale parameters"]
    for i, s in enumerate(items):
        ax.text(14 + 24 * i, 8.5, s, ha="center", va="center", fontsize=7.0, color=INK2)

    arrow((17, 50), (17, 46.6)); arrow((51.5, 50), (51.5, 48.4))
    arrow((32.4, 33.5), (35.6, 33.5)); arrow((67.4, 33.5), (71.6, 33.5))
    arrow((83, 50), (83, 46.6), ls="--")
    for x in (17, 51.5, 85):
        arrow((x, 20.6), (x, 16.6), color=C[0])
    save(fig, "fig1_model_structure")


# ----------------------------------------------------------------------- Fig 2
def load_column(path):
    import xarray as xr
    ds = xr.open_dataset(path)
    z = ds.z.values
    zi = np.r_[0, 0.5 * (z[1:] + z[:-1]), z[-1] + 0.5 * (z[-1] - z[-2])]
    dz = np.diff(zi)
    P = (ds.PO4 + ds.DOP + ds.DOPr + ds.spP + ds.diatP + ds.diazP + ds.zooC / 117).values
    N = (ds.NO3 + ds.NH4 + ds.DON + ds.DONr + (ds.spC + ds.diatC + ds.diazC + ds.zooC) * 16 / 117).values
    chl = (ds.spChl + ds.diatChl + ds.diazChl).values
    return dict(day=ds.day.values, z=z, P_tot=(P * dz).sum(1), N_tot=(N * dz).sum(1), chl=chl,
                npp=ds.NPP.values * 86400 * 365 * 12e-3, h_bl=ds.h_bl.values)


def fig2(d, n_dep=70.0):
    yr = d["day"] / 365.0
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    ax = axs[0, 0]
    dP = d["P_tot"] - d["P_tot"][0]
    ax.plot(yr, dP, color=C[0])
    rel = np.max(np.abs(dP)) / d["P_tot"][0]
    ax.set_title(f"Total phosphorus: max |drift| / inventory = {rel:.1e}", loc="left")
    ax.set_ylabel("change in column P [mmol m$^{-2}$]"); ax.set_xlabel("model year")
    lim = max(1.0, 1.2 * np.max(np.abs(dP)))
    ax.set_ylim(-lim, lim)
    panel_label(ax, "a")

    ax = axs[0, 1]
    dN = d["N_tot"] - d["N_tot"][0]
    ax.plot(yr, dN, color=C[1], label="model total N")
    ax.plot(yr, n_dep * yr, color=MUTED, ls="--", lw=1.2, label=f"deposition only ({n_dep:.0f} mmol m$^{{-2}}$ yr$^{{-1}}$)")
    ax.fill_between(yr, n_dep * yr, dN, where=dN >= n_dep * yr, color=C[1], alpha=0.12, lw=0,
                    label="N$_2$ fixation (difference)")
    ax.set_title("Total nitrogen: gain = deposition + fixation", loc="left")
    ax.set_ylabel("change in column N [mmol m$^{-2}$]"); ax.set_xlabel("model year")
    ax.legend(loc="upper left", fontsize=7)
    panel_label(ax, "b")

    ax = axs[1, 0]
    last = slice(-2 * 365, None)
    sel = d["z"] <= 250
    pc = ax.pcolormesh(yr[last], d["z"][sel], d["chl"][last][:, sel].T, cmap=SEQ_BLUE, shading="auto")
    ax.plot(yr[last], d["h_bl"][last], color=INK, lw=0.9, label="KPP boundary layer")
    ax.set_ylim(250, 0); ax.grid(False)
    ax.set_ylabel("depth [m]"); ax.set_xlabel("model year")
    ax.set_title("Total chlorophyll, last two years", loc="left")
    fig.colorbar(pc, ax=ax, label="Chl [mg m$^{-3}$]", pad=0.01)
    ax.legend(loc="lower right", fontsize=7)
    panel_label(ax, "c")

    ax = axs[1, 1]
    npp = d["npp"][-365:]
    doy = np.arange(365)
    for i, (lab, col) in enumerate(zip(("small phytoplankton", "diatoms", "diazotrophs"), C)):
        ax.plot(doy, npp[:, i], color=col, label=lab)
    ax.set_xlabel("day of year (last model year)"); ax.set_ylabel("NPP [g C m$^{-2}$ yr$^{-1}$]")
    ax.set_title(f"NPP by phytoplankton type (annual total {npp.sum(1).mean():.0f} g C m$^{{-2}}$ yr$^{{-1}}$)",
                 loc="left")
    ax.legend(fontsize=7)
    panel_label(ax, "d")
    fig.suptitle("MARBL in the legoESM column at BATS: conservation and behaviour", x=0.01, ha="left",
                 fontsize=10)
    save(fig, "fig2_verification_column")


# ----------------------------------------------------------------------- Fig 3
def fig3(g, s, top=20):
    fig = plt.figure(figsize=(7.2, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.55], width_ratios=[1.1, 1])

    res = g["results"]
    ax = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(res):
        hs = sorted(float(h) for h in r["fd"])
        fd = {float(h): v for h, v in r["fd"].items()}
        err = [abs(fd[h] - r["reverse"]) / max(abs(r["reverse"]), 1e-30) for h in hs]
        ax.loglog(hs, np.maximum(err, 1e-16), marker="o", ms=4, color=C[i % len(C)], label=r["param"])
    hh = np.array([1e-5, 1e-2])
    ax.loglog(hh, 0.3 * hh ** 2, color=MUTED, ls="--", lw=1)
    ax.text(3e-3, 0.3 * 3e-3 ** 2 * 3, "h$^2$", color=MUTED, fontsize=7.5)
    ax.set_xlabel("finite-difference step h (in ln p)")
    ax.set_ylabel("|FD − reverse| / |reverse|")
    ax.set_title(f"Gradient check: d NPP / d ln p ({g['days']} days)", loc="left")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    names = [r["param"] for r in res]
    y = np.arange(len(names))
    rev = np.array([r["reverse"] for r in res]); fwd = np.array([r["forward"] for r in res])
    ax.barh(y, np.abs(rev), height=0.6, color=C[0], label="reverse mode (jax.grad)")
    ax.scatter(np.abs(fwd), y, color=C[1], s=18, zorder=3, label="forward mode (jax.jvp)", marker="D")
    ax.set_xscale("log"); ax.set_yticks(y, names); ax.invert_yaxis()
    for yi, r in zip(y, res):
        ax.text(abs(r["reverse"]) * 4.0, yi, f"err {r['rel_err_best_fd']:.0e}", va="center", fontsize=6.5, color=INK2)
    ax.set_xlabel("|d NPP / d ln p| [g C m$^{-2}$ yr$^{-1}$]")
    ax.set_title("Derivative magnitude (label: rel. error vs FD)", loc="left")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    E, names_all, metrics = s["elasticity"], [str(n) for n in s["names"]], [str(m) for m in s["metrics"]]
    idx = list(s["rank"][:top])
    M = E[:, idx].T
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    labels = {"NPP": "NPP", "POC100": "export\n100 m", "Chl_sfc": "surface\nChl", "NO3_sfc": "surface\nNO$_3$",
              "fCO2": "air-sea\nCO$_2$ flux"}
    ax.set_xticks(range(len(metrics)), [labels.get(m, m) for m in metrics])
    ax.set_yticks(range(len(idx)), [names_all[i] for i in idx], fontsize=7)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.grid(False)
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=6.3,
                color="white" if abs(v) > 0.6 * vmax else INK)
    fig.colorbar(im, ax=ax, label="elasticity  d ln(metric) / d ln(parameter)", pad=0.01, shrink=0.9)
    ax.set_title(f"Sensitivity of 1-year means to all {len(names_all)} MARBL parameters "
                 f"(top {top} shown; one reverse pass per metric)", loc="left", pad=28)
    panel_label(ax, "c")
    save(fig, "fig3_gradients_sensitivity")


# ----------------------------------------------------------------------- Fig 4
def fig4(twins):
    t0 = twins[0]
    names = t0["params"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1, 1.15, 1.1]))

    ax = axs[0]
    for k, t in enumerate(twins):
        it = [h["iter"] for h in t["history"]]
        ax.semilogy(it, [h["loss"] for h in t["history"]], color=C[0], alpha=1.0 if k == 0 else 0.45,
                    label=f"seed {t.get('seed', k)}" if len(twins) > 1 else None)
    ax.set_xlabel("Adam iteration"); ax.set_ylabel("normalised misfit")
    ax.set_title("Loss", loc="left")
    if len(twins) > 1:
        ax.legend(fontsize=7)
    panel_label(ax, "a")

    ax = axs[1]
    th = np.array([h["theta"] for h in t0["history"]])
    it = [h["iter"] for h in t0["history"]]
    for i, n in enumerate(names):
        ax.plot(it, np.exp(th[:, i]), color=C[i % len(C)], label=n)
    ax.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_xlabel("Adam iteration"); ax.set_ylabel("scale factor (true = 1)")
    ax.set_title("Parameter trajectories", loc="left")
    ax.legend(fontsize=6, loc="upper right", ncol=1)
    panel_label(ax, "b")

    ax = axs[2]
    y = np.arange(len(names))
    a0, a1 = np.exp(t0["theta_init"]), np.exp(t0["theta_final"])
    for yi, p, q in zip(y, a0, a1):
        ax.annotate("", xy=(q, yi), xytext=(p, yi), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
    ax.scatter(a0, y, color=MUTED, s=16, label="initial", zorder=3)
    ax.scatter(a1, y, color=C[0], s=22, label="recovered", zorder=3)
    ax.axvline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_yticks(y, names, fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("scale factor"); ax.set_title("Recovery", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "c")
    fig.suptitle("Identical-twin calibration in the BATS column (6 most influential parameters)",
                 x=0.01, ha="left", fontsize=10)
    save(fig, "fig4_twin_calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--column-run", default="results/marbl_column_bats/bats_woa_5yr.nc")
    a = ap.parse_args()
    if 1 in a.figs:
        fig1()
    if 2 in a.figs:
        fig2(load_column(a.column_run))
    if 3 in a.figs:
        fig3(json.loads((EXP / "gradcheck.json").read_text()), np.load(EXP / "sensitivity.npz", allow_pickle=True))
    if 4 in a.figs:
        twins = []
        for f in sorted(glob.glob(str(EXP / "twin_seed*.json"))):
            t = json.loads(Path(f).read_text())
            t["seed"] = Path(f).stem.replace("twin_seed", "")
            twins.append(t)
        fig4(twins)


if __name__ == "__main__":
    main()
