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


def _latlon_coords(run_dir):
    """Raveled (lat_deg, lon_deg) of the lat-lon Mercator cell centres. Built
    once and reused across snapshots (so corr-vs-time does not rebuild the
    grid per day)."""
    from legoesm.ocean.experiments.dino import DINOConfig, dino_lat_lon_grid
    n_lon = int(_read_meta(run_dir)["args"].get("n_lon", 50))
    g = dino_lat_lon_grid(DINOConfig(), n_lon=n_lon)
    lat = np.degrees(np.asarray(g.lat)); lon = np.degrees(np.asarray(g.lon))
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    return LA.ravel(), LO.ravel()


def _mpas_coords(run_dir):
    """Raveled (lat_deg, lon_deg) of the MPAS cell centres (built once)."""
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
    return lat, lon


def _surface_TS(snap):
    """Raveled surface T, S and wet-mask from a snapshot npz."""
    with np.load(snap) as d:
        T = np.asarray(d["T"])[..., 0].ravel()
        S = np.asarray(d["S"])[..., 0].ravel()
        m = (np.asarray(d["land_mask"]) > 0.5).ravel()
    return T, S, m


def _regrid_TS(coords, snap):
    """Regrid a snapshot's surface T, S onto the common box given cached coords."""
    lat, lon = coords
    T, S, m = _surface_TS(snap)
    return _regrid(T, lat, lon, m), _regrid(S, lat, lon, m)


def _load_latlon(run_dir, snap):
    return _regrid_TS(_latlon_coords(run_dir), snap)


def _load_mpas(run_dir, snap):
    return _regrid_TS(_mpas_coords(run_dir), snap)


def _stats(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    da, db = a[both], b[both]
    corr = float(np.corrcoef(da, db)[0, 1]) if both.sum() > 10 else float("nan")
    rms = float(np.sqrt(np.mean((da - db) ** 2))) if both.sum() else float("nan")
    return corr, rms


def _day_of(snap):
    with np.load(snap) as d:
        return float(d["time_days"])


def _corr_vs_time(ll_dir, mp_dir):
    """SST/SSS pattern correlation + RMS between the grids at EVERY common
    snapshot day (matched within 0.5 d). Coords are built once per grid and
    reused; NaN-blown snapshots are skipped. Returns parallel lists."""
    ll_coords = _latlon_coords(ll_dir)
    mp_coords = _mpas_coords(mp_dir)
    ll_snaps = sorted((ll_dir / "snapshots").glob("snapshot_*.npz"))
    mp_items = [(_day_of(s), s) for s in
                sorted((mp_dir / "snapshots").glob("snapshot_*.npz"))]
    days, cT, cS, rT, rS = [], [], [], [], []
    for ls in ll_snaps:
        lday = _day_of(ls)
        # nearest MPAS snapshot to this lat-lon day; require a real match
        # (<= 0.5 d) — robust to differing cadences, no rounding collisions.
        mday, ms = min(mp_items, key=lambda it: abs(it[0] - lday))
        if abs(mday - lday) > 0.5:
            continue
        Tll, Sll = _regrid_TS(ll_coords, ls)
        Tmp, Smp = _regrid_TS(mp_coords, ms)
        # skip if ANY field is blown (NaN) on either grid — else corr/RMS NaN.
        if not all(np.isfinite(f).any() for f in (Tll, Tmp, Sll, Smp)):
            continue
        ct, rt = _stats(Tll, Tmp); cs, rs = _stats(Sll, Smp)
        days.append(round(lday)); cT.append(ct); cS.append(cs)
        rT.append(rt); rS.append(rs)
    return days, cT, cS, rT, rS


def _plot_corr_vs_time(ll_dir, mp_dir, out):
    """The consistency-through-spin-up highlight: pattern correlation between
    the two grids stays near 1 at every matched day."""
    days, cT, cS, rT, rS = _corr_vs_time(ll_dir, mp_dir)
    if not days:
        print("corr-vs-time: no common finite snapshot days — skipped")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))
    a0.plot(days, cT, "o-", label="SST", color="C3")
    a0.plot(days, cS, "s-", label="SSS", color="C0")
    a0.axhline(1.0, color="k", lw=0.6, ls=":")
    finite_c = [c for c in (cT + cS) if np.isfinite(c)]
    lo = min(0.9, min(finite_c) - 0.02) if finite_c else 0.9
    a0.set_ylim(lo, 1.002)
    a0.set_xlabel("day"); a0.set_ylabel("pattern correlation (lat-lon vs MPAS)")
    a0.set_title("Cross-grid correlation through spin-up"); a0.legend(); a0.grid(alpha=0.3)
    a1.plot(days, rT, "o-", label="SST [C]", color="C3")
    a1.plot(days, rS, "s-", label="SSS [PSU]", color="C0")
    a1.set_xlabel("day"); a1.set_ylabel("RMS difference")
    a1.set_title("Cross-grid RMS difference"); a1.legend(); a1.grid(alpha=0.3)
    fig.suptitle("DINO lat-lon vs MPAS: consistency maintained at every matched day",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}  (SST corr {min(cT):.3f}-{max(cT):.3f} over "
          f"days {days[0]}-{days[-1]})")


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

    # Consistency-through-spin-up: correlation at every matched snapshot day.
    _plot_corr_vs_time(args.latlon_dir, args.mpas_dir,
                       Path(str(out).replace(".png", "_corr_vs_time.png")))


if __name__ == "__main__":
    main()
