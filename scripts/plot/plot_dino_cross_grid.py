#!/usr/bin/env python
"""DINO lat-lon vs MPAS cross-grid comparison at a matched time.

The two DINO discretizations (structured Mercator C-grid and unstructured
MPAS Voronoi/TRiSK) integrate the SAME idealized basin, so at the same model
time they should produce the same ocean. This regrids both runs' surface
fields to a common 1deg lat-lon box and plots them side by side with the
difference map and zonal-mean profiles, and prints the pattern correlation +
RMS difference. It exists because the per-grid ``plot_dino.py`` panels are NOT
directly comparable (different end-times, structured pcolormesh vs Voronoi
scatter, and the monitor KE uses different formulas per grid) — which makes
two physically near-identical runs look "vastly different" at a glance.

Usage::

    JAX_ENABLE_X64=1 python scripts/plot/plot_dino_cross_grid.py \
        results/dino_latlon results/dino_mpas            # last common day
    ... --day 90 --out results/dino_cross_grid.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Common comparison box (the DINO basin: lon [-50,0], lat [-70,70]).
TGT_LAT = np.arange(-69.5, 70.0, 1.0)
TGT_LON = np.arange(-49.5, 0.0, 1.0)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("latlon_dir", type=Path, help="results/dino_latlon")
    p.add_argument("mpas_dir", type=Path, help="results/dino_mpas")
    p.add_argument("--day", type=float, default=None,
                   help="Model day to compare (default: last day common to both).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG (default: <latlon_dir>/../dino_cross_grid.png).")
    return p.parse_args()


def _snap_for_day(snap_dir: Path, day):
    """Return the snapshot npz whose time_days is closest to ``day`` (or the
    last finite-field snapshot if ``day`` is None)."""
    snaps = sorted(snap_dir.glob("snapshot_*.npz"))
    if not snaps:
        raise SystemExit(f"no snapshots in {snap_dir}")
    days = []
    for s in snaps:
        with np.load(s) as d:
            days.append(float(d["time_days"]))
    days = np.array(days)
    if day is None:
        # last snapshot whose OCEAN surface T is finite (skip NaN-contaminated
        # tails). Mask to wet cells: land/buffer cells can stay finite while the
        # ocean has blown to NaN, which would otherwise select a bad snapshot.
        for i in range(len(snaps) - 1, -1, -1):
            with np.load(snaps[i]) as d:
                t0 = np.asarray(d["T"])[..., 0]
                wet = np.asarray(d["land_mask"]) > 0.5
                if np.isfinite(t0[wet]).any():
                    return snaps[i], days[i]
        return snaps[-1], days[-1]
    i = int(np.argmin(np.abs(days - day)))
    return snaps[i], days[i]


def _regrid(fld, lat, lon, mask):
    from scripts.validate.compare_omip_nemo import regrid_curv_to_latlon
    L, ocn = regrid_curv_to_latlon(fld, lat, lon, mask, TGT_LAT, TGT_LON)
    return np.where(ocn > 0.5, L, np.nan)


def _read_meta(run_dir):
    """Load the run's grid resolution from its self-describing metadata, so the
    rebuilt grid/mesh matches the snapshot shapes for ANY --n-lon /
    --mpas-resolution-km the run used (not the 50/97 km defaults)."""
    import json
    with open(Path(run_dir) / "run_metadata.json") as f:
        return json.load(f)


def _load_latlon(run_dir, snap):
    from legoesm.ocean.experiments.dino import DINOConfig, dino_lat_lon_grid
    n_lon = int(_read_meta(run_dir)["args"].get("n_lon", 50))
    g = dino_lat_lon_grid(DINOConfig(), n_lon=n_lon)
    lat = np.degrees(np.asarray(g.lat)); lon = np.degrees(np.asarray(g.lon))
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    with np.load(snap) as d:
        T = np.asarray(d["T"])[..., 0]; S = np.asarray(d["S"])[..., 0]
        m = np.asarray(d["land_mask"]) > 0.5
    return (_regrid(T.ravel(), LA.ravel(), LO.ravel(), m.ravel()),
            _regrid(S.ravel(), LA.ravel(), LO.ravel(), m.ravel()))


def _load_mpas(run_dir, snap):
    from legoesm.ocean.experiments.dino import DINOConfig
    from legoesm.grids.voronoi import create_regional_voronoi_mesh
    res_km = float(_read_meta(run_dir)["args"].get("mpas_resolution_km", 97.0))
    cfg = DINOConfig()
    mesh = create_regional_voronoi_mesh(
        lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
        lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
        resolution_km=res_km, periodic_x=True)
    lat = np.degrees(np.asarray(mesh.latCell))
    lon = np.degrees(np.asarray(mesh.lonCell)); lon = np.where(lon > 180, lon - 360, lon)
    with np.load(snap) as d:
        T = np.asarray(d["T"])[..., 0]; S = np.asarray(d["S"])[..., 0]
        m = np.asarray(d["land_mask"]) > 0.5
    return (_regrid(T.ravel(), lat, lon, m.ravel()),
            _regrid(S.ravel(), lat, lon, m.ravel()))


def _stats(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    da, db = a[both], b[both]
    corr = float(np.corrcoef(da, db)[0, 1]) if both.sum() > 10 else float("nan")
    rms = float(np.sqrt(np.mean((da - db) ** 2))) if both.sum() else float("nan")
    return corr, rms


def main():
    args = _parse_args()
    ll_snap, ll_day = _snap_for_day(args.latlon_dir / "snapshots", args.day)
    mp_snap, mp_day = _snap_for_day(args.mpas_dir / "snapshots", args.day)
    print(f"lat-lon: {ll_snap.name} (day {ll_day:.0f})")
    print(f"MPAS:    {mp_snap.name} (day {mp_day:.0f})")
    if abs(ll_day - mp_day) > 5.0:
        print(f"WARNING: comparing different times ({ll_day:.0f} vs {mp_day:.0f} d)")

    Tll, Sll = _load_latlon(args.latlon_dir, ll_snap)
    Tmp, Smp = _load_mpas(args.mpas_dir, mp_snap)
    cT, rT = _stats(Tll, Tmp); cS, rS = _stats(Sll, Smp)
    print(f"SST: corr={cT:.3f} RMS={rT:.3f} C   SSS: corr={cS:.3f} RMS={rS:.3f} PSU")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [TGT_LON[0], TGT_LON[-1], TGT_LAT[0], TGT_LAT[-1]]
    fig, ax = plt.subplots(2, 4, figsize=(18, 11))
    rows = [("SST [C]", Tll, Tmp, cT, rT, "viridis", "RdBu_r"),
            ("SSS [PSU]", Sll, Smp, cS, rS, "viridis", "RdBu_r")]
    for r, (name, ll, mp, corr, rms, cmap, dcmap) in enumerate(rows):
        vmin = np.nanmin([np.nanmin(ll), np.nanmin(mp)])
        vmax = np.nanmax([np.nanmax(ll), np.nanmax(mp)])
        for c, (fld, title) in enumerate([
                (ll, f"lat-lon (day {ll_day:.0f})"),
                (mp, f"MPAS (day {mp_day:.0f})")]):
            im = ax[r, c].imshow(fld, origin="lower", extent=ext, aspect="auto",
                                 cmap=cmap, vmin=vmin, vmax=vmax)
            ax[r, c].set_title(f"{name} {title}"); fig.colorbar(im, ax=ax[r, c])
        dmax = np.nanpercentile(np.abs(ll - mp), 99)
        im = ax[r, 2].imshow(ll - mp, origin="lower", extent=ext, aspect="auto",
                             cmap=dcmap, vmin=-dmax, vmax=dmax)
        ax[r, 2].set_title(f"{name} diff (latlon-MPAS)\ncorr={corr:.3f} RMS={rms:.3f}")
        fig.colorbar(im, ax=ax[r, 2])
        ax[r, 3].plot(np.nanmean(ll, axis=1), TGT_LAT, label="lat-lon", lw=2)
        ax[r, 3].plot(np.nanmean(mp, axis=1), TGT_LAT, label="MPAS", lw=2, ls="--")
        ax[r, 3].set_title(f"{name} zonal mean"); ax[r, 3].set_ylabel("lat")
        ax[r, 3].legend(); ax[r, 3].grid(alpha=0.3)
    fig.suptitle(
        f"DINO cross-grid: lat-lon Mercator vs MPAS Voronoi (day "
        f"{0.5*(ll_day+mp_day):.0f})  —  SST corr {cT:.3f}, SSS corr {cS:.3f}",
        fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (args.latlon_dir.parent / "dino_cross_grid.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
