"""Silvestri et al. 2024 reproduction — comparison plotter (Figs 3-5, 7-10).

Reads the npz metric files saved by the run drivers and produces the paper's
comparison figures:

  2D turbulence (§4, ``--case turb2d``, from ``run_silvestri_turbulence_2d``):
    * Fig 4: integrated KE(t) and enstrophy(t), one curve per scheme.
    * Fig 5: isotropic energy + enstrophy spectra at t=3.6.
    * Fig 3: vorticity snapshots at t=3.6 (one panel per scheme).

  Baroclinic jet (§5, ``--case jet``, from ``run_silvestri_baroclinic_jet``):
    * Fig 8: TKE / EKE / eddy-APE time series.
    * Fig 9: zonal eddy-energy + enstrophy spectra.
    * Fig 10: zonal-mean buoyancy sections.
    * Fig 7: surface relative-vorticity snapshots.

PNGs are written to ``--outdir`` (gitignored). Operates on whatever npz files are
present, so it works incrementally as matrix runs complete.

Usage::

    .venv/bin/python scripts/plot/plot_silvestri_comparison.py \\
        --case turb2d --indir results/silvestri_turb2d --outdir results/silvestri_plots
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _load_all(indir, prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(indir, f"{prefix}*.npz"))):
        d = np.load(f, allow_pickle=True)
        out[os.path.basename(f)[:-4]] = d
    return out


def plot_turb2d(indir, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = _load_all(indir, "turb2d_")
    if not runs:
        print(f"no turb2d_*.npz in {indir}"); return
    os.makedirs(outdir, exist_ok=True)

    # Fig 4: KE(t) and enstrophy(t).
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in runs.items():
        lab = f"{d['scheme']} N{int(d['N'])}"
        ax[0].plot(d["t"], d["ke"], label=lab)
        ax[1].plot(d["t"], d["enstrophy"], label=lab)
    ax[0].set(xlabel="t", ylabel="kinetic energy", title="Fig 4a: KE(t)")
    ax[1].set(xlabel="t", ylabel="enstrophy", title="Fig 4b: enstrophy(t)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig4_turb2d_timeseries.png", dpi=130)
    plt.close(fig)

    # Fig 5: spectra at t=3.6 (log-log).
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in runs.items():
        lab = f"{d['scheme']} N{int(d['N'])}"
        ke = d["k_energy"]; Pe = d["P_energy"]
        kz = d["k_enstrophy"]; Pz = d["P_enstrophy"]
        m = ke > 0
        ax[0].loglog(ke[m], np.maximum(Pe[m], 1e-30), label=lab)
        ax[1].loglog(kz[m], np.maximum(Pz[m], 1e-30), label=lab)
    ax[0].set(xlabel="k", ylabel="E(k)", title="Fig 5a: energy spectrum @ t=3.6")
    ax[1].set(xlabel="k", ylabel="Z(k)", title="Fig 5b: enstrophy spectrum @ t=3.6")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig5_turb2d_spectra.png", dpi=130)
    plt.close(fig)

    # Fig 3: vorticity snapshots at t=3.6.
    n = len(runs)
    cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows),
                             squeeze=False)
    for axx in axes.ravel():
        axx.axis("off")
    for (name, d), axx in zip(runs.items(), axes.ravel()):
        z = d["zeta_36"]
        lim = np.percentile(np.abs(z), 99)
        axx.imshow(z, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axx.set_title(f"{d['scheme']} N{int(d['N'])}", fontsize=8)
    fig.suptitle("Fig 3: vorticity @ t=3.6")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_turb2d_vorticity.png", dpi=130)
    plt.close(fig)
    print(f"wrote turb2d figures (fig3/4/5) to {outdir} ({n} runs)")


def plot_jet(indir, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = _load_all(indir, "silvestri_jet_")
    if not runs:
        print(f"no silvestri_jet_*.npz in {indir}"); return
    os.makedirs(outdir, exist_ok=True)
    _SPD = 86400.0

    # Fig 8: TKE / EKE / eddy-APE time series.
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, d in runs.items():
        if bool(d["blew"]):
            continue
        lab = str(d["scheme"])
        td = d["t"] / _SPD
        ax[0].plot(td, d["tke"], label=lab)
        ax[1].plot(td, d["eke"], label=lab)
        ax[2].plot(td, d["ape"], label=lab)
    for a, t in zip(ax, ("total KE", "eddy KE", "eddy APE")):
        a.set(xlabel="day", ylabel=t, title=f"Fig 8: {t}(t)")
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig8_jet_energy.png", dpi=130)
    plt.close(fig)

    # Fig 9: zonal eddy-energy + enstrophy spectra.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in runs.items():
        if bool(d["blew"]) or d["k_energy"] is None or d["k_energy"].ndim == 0:
            continue
        lab = str(d["scheme"])
        ke = d["k_energy"]; Pe = d["P_energy"]; kz = d["k_enstrophy"]; Pz = d["P_enstrophy"]
        m = ke > 0
        ax[0].loglog(ke[m], np.maximum(Pe[m], 1e-30), label=lab)
        ax[1].loglog(kz[m], np.maximum(Pz[m], 1e-30), label=lab)
    ax[0].set(xlabel="k [1/m]", ylabel="E(k)", title="Fig 9a: zonal energy spectrum")
    ax[1].set(xlabel="k [1/m]", ylabel="Z(k)", title="Fig 9b: zonal enstrophy spectrum")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig9_jet_spectra.png", dpi=130)
    plt.close(fig)

    # Fig 10: zonal-mean buoyancy (surface row) + Fig 7: surface vorticity.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in runs.items():
        if bool(d["blew"]) or d["zonal_mean_buoyancy"] is None \
                or d["zonal_mean_buoyancy"].ndim == 0:
            continue
        bzm = d["zonal_mean_buoyancy"]            # (n_lat, nlev)
        ax[0].plot(bzm[:, 0], label=str(d["scheme"]))   # surface
    ax[0].set(xlabel="lat index", ylabel="surface zonal-mean b",
              title="Fig 10: zonal-mean buoyancy (surface)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    # one vorticity snapshot
    for name, d in runs.items():
        if bool(d["blew"]) or d["zeta_surface"] is None \
                or d["zeta_surface"].ndim == 0:
            continue
        z = d["zeta_surface"]
        lim = np.percentile(np.abs(z), 99)
        im = ax[1].imshow(z, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax[1].set_title(f"Fig 7: surface ζ — {d['scheme']}")
        break
    ax[1].axis("off")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig10_jet_buoyancy_vorticity.png", dpi=130)
    plt.close(fig)
    print(f"wrote jet figures (fig7/8/9/10) to {outdir} ({len(runs)} runs)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", choices=("turb2d", "jet"), required=True)
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", default="results/silvestri_plots")
    args = ap.parse_args()
    if args.case == "turb2d":
        plot_turb2d(args.indir, args.outdir)
    else:
        plot_jet(args.indir, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
