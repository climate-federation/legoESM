#!/usr/bin/env python
"""DINO model-vs-model: our lat-lon + MPAS runs vs the paper's R1 equilibrium.

Loads the Kamm et al. 2025 R1 production restart (the 3000-yr equilibrium NEMO
state from the Zenodo deposit), extracts the equilibrium SST/SSS, regrids it to
our common box, and overlays it on our runs together with the restoring targets
T*/S*. This is the honest comparison the "SST is below T*" question needs: the
paper's *equilibrium* SST is itself below T* (equatorial upwelling), and our
<=1-yr spin-up sits below the paper's equilibrium (the spin-up gap).

The paper grid is the same 50deg basin but at longitude 0..50 (ours is -50..0),
so the paper longitude is shifted by -50 to align. Regrid + zonal helpers are
the shared ``compare_omip_nemo`` + ``plot_dino_cross_grid`` ones (no re-derived
interpolation).

Usage::

    JAX_ENABLE_X64=1 python scripts/plot/plot_dino_vs_paper.py \
        data/dino_paper/Reference_experiments/EXP_R1/restart_prod_1deg.nc \
        results/dino_latlon_kpp results/dino_mpas --day 90 --out results/dino_vs_paper.png
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("plot_dino_cross_grid",
                                               _HERE / "plot_dino_cross_grid.py")
XG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(XG)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paper_restart", type=Path)
    p.add_argument("latlon_dir", type=Path)
    p.add_argument("mpas_dir", type=Path)
    p.add_argument("--day", type=float, default=90.0)
    p.add_argument("--out", type=Path, default=Path("results/dino_vs_paper.png"))
    return p.parse_args()


def _paper_surface(restart):
    """Regridded paper R1 equilibrium SST, SSS [common box]. NEMO land = 0."""
    from scripts.validate.compare_omip_nemo import regrid_curv_to_latlon
    import xarray as xr
    d = xr.open_dataset(restart, decode_times=False)
    tn = np.asarray(d["tn"])[0, 0]                 # (y, x) surface temperature
    sn = np.asarray(d["sn"])[0, 0]
    lat = np.asarray(d["nav_lat"]); lon = np.asarray(d["nav_lon"]) - 50.0  # 0..50 -> -50..0
    wet = tn != 0.0                                # NEMO masks land with 0
    T, ocn = regrid_curv_to_latlon(np.where(wet, tn, 0.0).ravel(), lat.ravel(),
                                   lon.ravel(), wet.ravel(), XG.TGT_LAT, XG.TGT_LON)
    S, _ = regrid_curv_to_latlon(np.where(wet, sn, 0.0).ravel(), lat.ravel(),
                                 lon.ravel(), wet.ravel(), XG.TGT_LAT, XG.TGT_LON)
    return (np.where(ocn > 0.5, T, np.nan), np.where(ocn > 0.5, S, np.nan))


def main():
    args = _parse_args()
    from legoesm.ocean.experiments.dino import dino_T_star_annual_mean, dino_S_star
    Tpap, Spap = _paper_surface(args.paper_restart)
    ll_snap, ll_day = XG._snap_for_day(args.latlon_dir / "snapshots", args.day)
    mp_snap, mp_day = XG._snap_for_day(args.mpas_dir / "snapshots", args.day)
    Tll, Sll = XG._regrid_TS(XG._latlon_coords(args.latlon_dir), ll_snap)
    Tmp, Smp = XG._regrid_TS(XG._mpas_coords(args.mpas_dir), mp_snap)
    lat, lon = XG.TGT_LAT, XG.TGT_LON
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    Tstar = np.array([float(dino_T_star_annual_mean(la)) for la in lat])
    Sstar = np.array([float(dino_S_star(la)) for la in lat])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 11, "savefig.dpi": 300, "figure.dpi": 120})
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    rows = [("SST", "[°C]", Tpap, Tll, Tmp, Tstar),
            ("SSS", "[g kg$^{-1}$]", Spap, Sll, Smp, Sstar)]
    letters = iter("abcdef")
    for r, (name, unit, pap, ll, mp, star) in enumerate(rows):
        vmin = np.nanmin([np.nanmin(pap), np.nanmin(ll)])
        vmax = np.nanmax([np.nanmax(pap), np.nanmax(ll)])
        for c, (fld, ttl) in enumerate([(pap, "paper R1 (3000-yr equil.)"),
                                        (ll, f"ours lat-lon (day {ll_day:.0f})")]):
            im = ax[r, c].imshow(fld, origin="lower", extent=ext, aspect="auto",
                                 cmap="viridis", vmin=vmin, vmax=vmax)
            ax[r, c].set_title(f"{name} {ttl}")
            ax[r, c].set_xlabel("lon [°]"); ax[r, c].set_ylabel("lat [°]")
            ax[r, c].text(-0.08, 1.04, f"({next(letters)})", transform=ax[r, c].transAxes,
                          fontweight="bold")
            fig.colorbar(im, ax=ax[r, c], label=f"{name} {unit}", shrink=0.85)
        ax[r, 2].plot(np.nanmean(pap, axis=1), lat, "-", color="C3", lw=2.2, label="paper R1 equil.")
        ax[r, 2].plot(np.nanmean(ll, axis=1), lat, "-", color="C0", lw=2, label="ours lat-lon")
        ax[r, 2].plot(np.nanmean(mp, axis=1), lat, "--", color="C1", lw=2, label="ours MPAS")
        ax[r, 2].plot(star, lat, ":", color="k", lw=1.5, label="restoring target")
        ax[r, 2].set_title(f"{name} zonal mean")
        ax[r, 2].set_xlabel(f"{name} {unit}"); ax[r, 2].set_ylabel("lat [°]")
        ax[r, 2].legend(frameon=False, fontsize=8); ax[r, 2].grid(alpha=0.3)
        ax[r, 2].text(-0.12, 1.04, f"({next(letters)})", transform=ax[r, 2].transAxes,
                      fontweight="bold")
    # quantify the spin-up gap (ours vs paper equilibrium) where both wet
    both = np.isfinite(Tpap) & np.isfinite(Tll)
    dT = float(np.sqrt(np.mean((Tll[both] - Tpap[both]) ** 2)))
    fig.suptitle("DINO model-vs-model: ours (spin-up) vs paper R1 equilibrium vs "
                 f"restoring target  —  SST RMS(ours−paper)={dT:.2f} °C", fontsize=13)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"SST equator: target={Tstar[np.argmin(np.abs(lat))]:.1f}  "
          f"paper-equil={np.nanmean(Tpap[np.argmin(np.abs(lat))]):.1f}  "
          f"ours-ll={np.nanmean(Tll[np.argmin(np.abs(lat))]):.1f}  "
          f"(RMS ours-vs-paper {dT:.2f} °C)")


if __name__ == "__main__":
    main()
