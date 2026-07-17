#!/usr/bin/env python
"""Publication-grade DINO figures: lat-lon Mercator vs MPAS Voronoi.

Produces two paper-quality figures from a matched lat-lon + MPAS DINO run pair:

  Figure 1 -- surface state at a matched time: SST + SSS on each grid (regridded
    to a common box) with the zonal means overlaid against the paper restoring
    targets T*/S* (Kamm et al. 2025).
  Figure 2 -- circulation + spin-up: the barotropic streamfunction (lat-lon),
    the ACC channel-transport spin-up trajectory vs the paper's 206 Sv R1
    equilibrium, and the cross-grid SST/SSS pattern correlation through time.

All numerics REUSE the canonical diagnostics + the sibling plotters' helpers
(no re-derived transport / regrid / correlation code): ``compare_omip_nemo``
regrid, ``diagnostics_streamfunction`` + ``diagnostics_climate`` for Psi/ACC,
and ``plot_dino_cross_grid`` / ``plot_dino_acc`` for snapshot selection,
coords-cached regrid, correlation-vs-time, and partial-cell thickness.

Usage::

    JAX_ENABLE_X64=1 python scripts/plot/plot_dino_publication.py \
        results/dino_latlon_kpp results/dino_mpas --day 90 --outdir results
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PAPER_ACC_R1_SV = 206.0
_SV = 1.0e6


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


XG = _load_sibling("plot_dino_cross_grid")   # _latlon_coords/_mpas_coords/_regrid_TS/_stats/_corr_vs_time/_snap_for_day/TGT_*
ACC = _load_sibling("plot_dino_acc")          # _grid_and_z/_snaps


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("latlon_dir", type=Path)
    p.add_argument("mpas_dir", type=Path)
    p.add_argument("--day", type=float, default=90.0,
                   help="Matched time for the surface-state figure (default 90).")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    return p.parse_args()


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "savefig.dpi": 300, "figure.dpi": 120,
        "font.family": "DejaVu Sans",
    })


def _panel(ax, letter):
    ax.text(-0.08, 1.04, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=12, va="bottom", ha="left")


def _acc_series_and_psi(run_dir):
    """ACC(t) [Sv] + the last barotropic streamfunction [Sv] for a lat-lon run,
    via the canonical partial-cell diagnostics (partial_cell_thickness)."""
    from legoesm.ocean.diagnostics_streamfunction import barotropic_streamfunction
    from legoesm.ocean.diagnostics_climate import acc_transport
    cfg, grid, dz_ref = ACC._grid_and_z(run_dir)
    lat_deg = np.degrees(np.asarray(grid.lat)); lon_deg = np.degrees(np.asarray(grid.lon))
    days, acc, last_psi, last_day = [], [], None, None
    for s in ACC._snaps(run_dir):
        with np.load(s) as d:
            u = np.asarray(d["u"]); mask = np.asarray(d["land_mask"])
            H = np.asarray(d["H_bathy"]); day = float(d["time_days"])
        if not np.isfinite(u).all():
            continue
        from legoesm.ocean.diagnostics_streamfunction import (
            partial_cell_thickness,
        )
        psi = np.asarray(barotropic_streamfunction(
            u, partial_cell_thickness(H, dz_ref), mask, grid))
        a = acc_transport(psi * _SV, lat_deg,
                          drake_lat_south=cfg.channel_lat_south_deg,
                          drake_lat_north=cfg.channel_lat_north_deg).transport_Sv
        if not np.isfinite(a):
            continue
        days.append(day); acc.append(a); last_psi = psi; last_day = day
    if not days:
        raise SystemExit(f"{run_dir}: no finite lat-lon snapshots for Psi/ACC")
    return cfg, lat_deg, lon_deg, np.array(days), np.array(acc), last_psi, last_day


def _figure_surface_state(args, out):
    """Fig 1: SST + SSS, both grids regridded, vs paper T*/S* targets."""
    from legoesm.ocean.experiments.dino import dino_T_star_annual_mean, dino_S_star
    import matplotlib.pyplot as plt
    ll_snap, ll_day = XG._snap_for_day(args.latlon_dir / "snapshots", args.day)
    mp_snap, mp_day = XG._snap_for_day(args.mpas_dir / "snapshots", args.day)
    Tll, Sll = XG._regrid_TS(XG._latlon_coords(args.latlon_dir), ll_snap)
    Tmp, Smp = XG._regrid_TS(XG._mpas_coords(args.mpas_dir), mp_snap)
    if not all(np.isfinite(f).any() for f in (Tll, Tmp, Sll, Smp)):
        raise SystemExit(
            f"surface fields all-NaN at day {args.day:.0f} (blown-up snapshot?) "
            "— choose a --day before the run goes unstable")
    lat, lon = XG.TGT_LAT, XG.TGT_LON
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    Tstar = np.array([float(dino_T_star_annual_mean(la)) for la in lat])
    Sstar = np.array([float(dino_S_star(la)) for la in lat])

    fig, ax = plt.subplots(2, 3, figsize=(13.5, 9), constrained_layout=True)
    rows = [("SST", "[°C]", Tll, Tmp, Tstar, "thermal", "RdBu_r"),
            ("SSS", "[g kg$^{-1}$]", Sll, Smp, Sstar, "haline", "RdBu_r")]
    letters = iter("abcdef")
    for r, (name, unit, ll, mp, star, _cmo, _d) in enumerate(rows):
        cmap = "viridis"
        vmin = np.nanmin([np.nanmin(ll), np.nanmin(mp)])
        vmax = np.nanmax([np.nanmax(ll), np.nanmax(mp)])
        for c, (fld, gname) in enumerate([(ll, "lat-lon Mercator"), (mp, "MPAS Voronoi")]):
            im = ax[r, c].imshow(fld, origin="lower", extent=ext, aspect="auto",
                                 cmap=cmap, vmin=vmin, vmax=vmax)
            ax[r, c].set_title(f"{name} {gname}"); _panel(ax[r, c], next(letters))
            ax[r, c].set_xlabel("longitude [°]"); ax[r, c].set_ylabel("latitude [°]")
            fig.colorbar(im, ax=ax[r, c], label=f"{name} {unit}", shrink=0.85)
        ax[r, 2].plot(np.nanmean(ll, axis=1), lat, "-", color="C0", lw=2, label="lat-lon")
        ax[r, 2].plot(np.nanmean(mp, axis=1), lat, "--", color="C1", lw=2, label="MPAS")
        ax[r, 2].plot(star, lat, ":", color="k", lw=1.5, label="paper target")
        ax[r, 2].set_title(f"{name} zonal mean"); _panel(ax[r, 2], next(letters))
        ax[r, 2].set_xlabel(f"{name} {unit}"); ax[r, 2].set_ylabel("latitude [°]")
        ax[r, 2].legend(frameon=False); ax[r, 2].grid(alpha=0.3)
    cT, rT = XG._stats(Tll, Tmp); cS, rS = XG._stats(Sll, Smp)
    fig.suptitle(f"DINO surface state, day {0.5*(ll_day+mp_day):.0f} — "
                 f"SST corr {cT:.3f} (RMS {rT:.2f} °C), SSS corr {cS:.3f} "
                 f"(RMS {rS:.2f} g kg$^{{-1}}$)", fontsize=13)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  (SST corr {cT:.3f}, SSS corr {cS:.3f})")


def _figure_circulation(args, out):
    """Fig 2: barotropic Psi + ACC spin-up + cross-grid correlation vs time."""
    import matplotlib.pyplot as plt
    cfg, lat_deg, lon_deg, days, acc, psi, last_day = _acc_series_and_psi(args.latlon_dir)
    if not list((args.mpas_dir / "snapshots").glob("snapshot_*.npz")):
        raise SystemExit(f"{args.mpas_dir}: no MPAS snapshots for the cross-grid panel")
    cdays, cT, cS, _rT, _rS = XG._corr_vs_time(args.latlon_dir, args.mpas_dir)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    ext = [lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()]
    pmax = np.nanpercentile(np.abs(psi), 99) or 1.0
    im = ax[0].imshow(psi, origin="lower", extent=ext, aspect="auto",
                      cmap="RdBu_r", vmin=-pmax, vmax=pmax)
    ax[0].axhline(cfg.channel_lat_south_deg, color="k", lw=0.7, ls=":")
    ax[0].axhline(cfg.channel_lat_north_deg, color="k", lw=0.7, ls=":")
    ax[0].set_title(f"Barotropic streamfunction, day {last_day:.0f}")
    ax[0].set_xlabel("longitude [°]"); ax[0].set_ylabel("latitude [°]")
    fig.colorbar(im, ax=ax[0], label="$\\psi$ [Sv]", shrink=0.85); _panel(ax[0], "a")

    ax[1].plot(days, acc, "o-", color="C0", lw=2, ms=4)
    ax[1].axhline(PAPER_ACC_R1_SV, color="C3", ls="--",
                  label=f"Kamm 2025 R1 equilibrium ({PAPER_ACC_R1_SV:.0f} Sv)")
    ax[1].set_xlabel("model day"); ax[1].set_ylabel("ACC channel transport [Sv]")
    ax[1].set_title("ACC transport spin-up (lat-lon)")
    ax[1].legend(frameon=False); ax[1].grid(alpha=0.3); _panel(ax[1], "b")

    ax[2].plot(cdays, cT, "o-", color="C3", lw=2, ms=4, label="SST")
    ax[2].plot(cdays, cS, "s-", color="C0", lw=2, ms=4, label="SSS")
    ax[2].axhline(1.0, color="k", lw=0.6, ls=":")
    finite = [c for c in (list(cT) + list(cS)) if np.isfinite(c)]
    ax[2].set_ylim(min(0.9, min(finite) - 0.02) if finite else 0.9, 1.002)
    ax[2].set_xlabel("model day"); ax[2].set_ylabel("pattern correlation")
    ax[2].set_title("Cross-grid consistency (lat-lon vs MPAS)")
    ax[2].legend(frameon=False); ax[2].grid(alpha=0.3); _panel(ax[2], "c")

    fig.suptitle("DINO circulation + spin-up — lat-lon Mercator vs MPAS Voronoi",
                 fontsize=13)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  (ACC {acc[0]:.1f}→{acc[-1]:.1f} Sv over {days[-1]:.0f} d)")


def main():
    args = _parse_args()
    _style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    _figure_surface_state(args, args.outdir / "dino_fig1_surface_state.png")
    _figure_circulation(args, args.outdir / "dino_fig2_circulation.png")


if __name__ == "__main__":
    main()
