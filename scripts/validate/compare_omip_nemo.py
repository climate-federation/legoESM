#!/usr/bin/env python
"""Score a legoESM tripole snapshot against the NEMO ORCA1 reference grid_T.

Both fields live on (different) curvilinear eORCA1 grids, so each is regridded
to a common regular lat-lon grid by cKDTree inverse-distance weighting over its
ocean cells (same pattern as ``legoesm.grids.regridding``, but for a *curvilinear*
2-D source rather than a regular 1-D-axis one). On the common grid we report
area-weighted (cos-lat) global means, bias, RMSE and pattern correlation for SST
and SSS, plus zonal means, and score against provisional tolerances.

Faithfulness caveats (see OMIP_faithful.md): SSS is GATED (informational, not
pass/fail) by default — a legacy of the original runoff=0 forcing. Dai-Trenberth
runoff is now wired (latlon/tripole/mpas), so pass ``--sss-faithful`` for a run
that actually applied ``--runoff`` to score SSS as a real verdict. Runs are short
spinups (a few years), not 40-yr equilibrium.

Usage:
    python scripts/compare_omip_nemo.py \
        --legoesm-snapshot results/omip_nemo/legoesm_tripole/snapshot_year004.npz \
        --nemo-gridt /…/RUN_REF/ORCA1_1m_..._grid_T.nc --nemo-time-idx -1 \
        --output-dir results/omip_nemo/compare_year4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Provisional "match" tolerances for a few-year spinup, model-vs-model
# (NEMO-vs-observation SST RMSE is ~0.5-1 K; we allow more for coarse legoESM).
_TOL = {
    "sst_rmse_excellent_C": 1.5, "sst_rmse_good_C": 2.5,
    # ENSO-box |bias| caps on the SST verdict (regional failure must not hide
    # under the global L2; nino3 +2.8 C with global rmse 0.94 is NOT excellent).
    "sst_box_bias_excellent_C": 1.0, "sst_box_bias_good_C": 2.0,
    "sss_rmse_excellent": 0.5, "sss_rmse_good": 1.0,   # gated (runoff=0)
}


def capped_sst_verdict(global_rmse, sst_boxes, tol=None):
    """SST verdict = global-rmse tier CAPPED at the worst ENSO-box |bias| tier.

    A global L2 hides an area-small regional failure (user callout 2026-08-18:
    nino3 +2.8 C scored "excellent" under global rmse 0.94).  Returns
    ``(verdict, capped_by)`` where ``capped_by`` is ``(box_name, bias)`` when a
    box demoted the verdict, else ``None``.
    """
    t = tol or _TOL
    rank = ("excellent", "good", "poor")

    def tier(x, exc, good):
        return "excellent" if x < exc else "good" if x < good else "poor"

    v = tier(global_rmse, t["sst_rmse_excellent_C"], t["sst_rmse_good_C"])
    capped_by = None
    for bn, bs in (sst_boxes or {}).items():
        if bs is None:
            continue
        bv = tier(abs(bs["bias"]),
                  t["sst_box_bias_excellent_C"], t["sst_box_bias_good_C"])
        if rank.index(bv) > rank.index(v):
            v, capped_by = bv, (bn, bs["bias"])
    return v, capped_by


def _xyz(lat_rad, lon_rad):
    """Unit-sphere Cartesian (inline; trivial, avoids a private cross-module import)."""
    cl = np.cos(lat_rad)
    return np.stack([cl * np.cos(lon_rad), cl * np.sin(lon_rad),
                     np.sin(lat_rad)], axis=-1)


def regrid_curv_to_latlon(field2d, src_lat_deg, src_lon_deg, ocean_mask,
                          tgt_lat_deg, tgt_lon_deg, k=4, max_deg=2.5):
    """IDW-regrid a curvilinear 2-D field (ocean cells only) onto a regular
    lat-lon grid. Returns (regridded 2-D, ocean_flag 2-D) where ocean_flag=0
    for target cells farther than ``max_deg`` from any source ocean cell."""
    from scipy.spatial import cKDTree
    m = np.asarray(ocean_mask).ravel() > 0.5
    if not m.any():
        raise ValueError("no ocean source cells")
    src_xyz = _xyz(np.deg2rad(np.asarray(src_lat_deg).ravel()[m]),
                   np.deg2rad(np.asarray(src_lon_deg).ravel()[m]))
    vals = np.asarray(field2d, dtype=np.float64).ravel()[m]
    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon_deg, tgt_lat_deg)
    tgt_xyz = _xyz(np.deg2rad(tgt_lat2d.ravel()), np.deg2rad(tgt_lon2d.ravel()))
    tree = cKDTree(src_xyz)
    d, idx = tree.query(tgt_xyz, k=k)
    d = np.maximum(d, 1e-12)
    w = (1.0 / d) / (1.0 / d).sum(axis=1, keepdims=True)
    out = (vals[idx] * w).sum(axis=1).reshape(tgt_lat2d.shape)
    # chord distance for max_deg separation on the unit sphere
    chord = 2.0 * np.sin(np.deg2rad(max_deg) / 2.0)
    ocean = (d[:, 0].reshape(tgt_lat2d.shape) < chord).astype(np.float64)
    return out, ocean


def _wstats(a, b, area):
    """Area-weighted bias, RMSE, pattern correlation of a vs b over area>0."""
    w = np.asarray(area, dtype=np.float64)
    sel = w > 0
    w = w[sel]; x = np.asarray(a)[sel]; y = np.asarray(b)[sel]
    W = w.sum()
    bias = float((w * (x - y)).sum() / W)
    rmse = float(np.sqrt((w * (x - y) ** 2).sum() / W))
    xm = (w * x).sum() / W; ym = (w * y).sum() / W
    cov = (w * (x - xm) * (y - ym)).sum() / W
    sx = np.sqrt((w * (x - xm) ** 2).sum() / W)
    sy = np.sqrt((w * (y - ym) ** 2).sum() / W)
    corr = float(cov / (sx * sy)) if sx > 0 and sy > 0 else float("nan")
    return {"lego_mean": float(xm), "nemo_mean": float(ym),
            "bias": bias, "rmse": rmse, "corr": corr}


def _band_breakdown(fieldL, fieldN, area, tgt_lat):
    """Per-latitude-band area-weighted stats, to localise where a global RMSE
    comes from (Southern Ocean vs tropics vs Arctic etc.). ``area`` already
    carries the ocean mask (0 on land/outside). Returns {band: stats}."""
    bands = {
        "antarctic_S_of_45S": (-90.0, -45.0),
        "SH_midlat_45S_23S": (-45.0, -23.0),
        "tropics_23S_23N": (-23.0, 23.0),
        "NH_midlat_23N_45N": (23.0, 45.0),
        "arctic_N_of_45N": (45.0, 90.0),
    }
    lat2d = tgt_lat[:, None] * np.ones((1, area.shape[1]))
    out = {}
    for name, (lo, hi) in bands.items():
        m = (lat2d >= lo) & (lat2d < hi)
        a = area * m
        out[name] = _wstats(fieldL, fieldN, a) if a.sum() > 0 else None
    return out


# --- Named lat-lon boxes (lon in 0-360, the target grid's convention) -------
# ENSO indices are the standard CLIVAR definitions. Niño 3.4 is the one used
# for ENSO state; Niño 3 and 4 separate the eastern cold-tongue bias from the
# western warm-pool bias, which fail for different reasons in an ocean-only
# run -- the cold tongue is upwelling/mixing, the warm pool is surface flux.
_BOXES = {
    "nino34_5S5N_170W120W": (-5.0, 5.0, 190.0, 240.0),
    "nino3_5S5N_150W90W":   (-5.0, 5.0, 210.0, 270.0),
    "nino4_5S5N_160E150W":  (-5.0, 5.0, 160.0, 210.0),
    "eq_pacific_2S2N":      (-2.0, 2.0, 140.0, 280.0),
}


def _box_breakdown(fieldL, fieldN, area, tgt_lat, tgt_lon):
    """Area-weighted stats over named lat-lon boxes (ENSO regions).

    The latitude bands alone cannot answer "is the Pacific right": tropics
    23S-23N averages the equatorial cold tongue together with both subtropical
    gyres, so a cold-tongue error and an off-equatorial error of opposite sign
    cancel in that number.
    """
    lat2d = tgt_lat[:, None] * np.ones((1, area.shape[1]))
    lon2d = np.ones((area.shape[0], 1)) * tgt_lon[None, :]
    out = {}
    for name, (la, lb, lo, hi) in _BOXES.items():
        m = (lat2d >= la) & (lat2d <= lb) & (lon2d >= lo) & (lon2d <= hi)
        a = area * m
        out[name] = _wstats(fieldL, fieldN, a) if a.sum() > 0 else None
    return out


def _load_legoesm(path, use_mean=False):
    """Fields of one legoESM snapshot.

    ``use_mean=True`` reads the driver's ``--state-accumulate`` window means
    instead of the instantaneous state: NEMO's 5-day files are window means
    for EVERY field (tos/sos plain hourly means; to/so thickness-weighted
    @toce_e3t/@e3t), and an instantaneous 00 UTC snapshot against them
    carries the diurnal phase (nino3 SST +0.49 vs +0.12 window-matched,
    2026-09-25). Surface fields come from the plain means, the 3-D columns
    from the thickness-weighted ones, and the MLD from ``mld_mean``; a
    snapshot lacking any of them is refused, never silently downgraded.
    """
    s = np.load(path)
    # lat_T/lon_T are written by run_omip_core2._save_snapshot ALREADY IN DEGREES
    # (via _grid_lat2d_deg, which applies np.rad2deg at save time). Applying
    # rad2deg AGAIN here corrupted the coordinates (45 deg -> 2578) -> the regrid
    # mapped every cell to nonsense lat-lon, scrambling the SST/SSS pattern
    # (corr ~0.1) and inflating the bias. Use the stored degrees as-is.
    keys = set(getattr(s, "files", []))
    if use_mean:
        need = ("T_mean", "S_mean", "T_mean_hw", "S_mean_hw", "mld_mean")
        missing = [k for k in need if k not in keys]
        if missing:
            raise SystemExit(f"{path}: use_mean requested but the snapshot lacks "
                             f"{missing}; run the driver with --state-accumulate "
                             "--mld-accumulate (no silent fallback to the instantaneous state)")
    kT, kS = ("T_mean", "S_mean") if use_mean else ("T", "S")
    kT3, kS3 = ("T_mean_hw", "S_mean_hw") if use_mean else ("T", "S")
    return {
        "sst": np.asarray(s[kT])[..., 0], "sss": np.asarray(s[kS])[..., 0],
        # Full T/S columns + geometry for the mixed-layer-depth diagnostic
        # (present only in snapshots written after _save_snapshot grew the MLD
        # geometry; None for older snapshots -> MLD comparison is skipped).
        "T3d": np.asarray(s[kT3]), "S3d": np.asarray(s[kS3]),
        "mld_mean": np.asarray(s["mld_mean"]) if use_mean else None,
        "H_bathy": np.asarray(s["H_bathy"]) if "H_bathy" in keys else None,
        "z_center_ref": np.asarray(s["z_center_ref"]) if "z_center_ref" in keys else None,
        "lat": np.asarray(s["lat_T"]),
        "lon": np.asarray(s["lon_T"]),
        "mask": np.asarray(s["land_mask"]),
    }


def _nemo_record_months(ds, tdim, nt):
    """Calendar month (1-12) of each of the ``nt`` time records, decoded from CF
    metadata (``units`` + ``calendar``).  Returns a length-``nt`` list, or ``None``
    if the time axis cannot be decoded (caller then falls back to a positional
    Jan-first assumption, with a warning).  Used so ``--nemo-month`` selects by the
    TRUE calendar month rather than blindly trusting record order."""
    try:
        import cftime
    except Exception:
        return None
    tv = ds[tdim] if tdim in ds.variables else None
    if tv is None:
        return None
    units = tv.attrs.get("units")
    if not units:
        return None
    calendar = tv.attrs.get("calendar", "standard")
    try:
        dates = cftime.num2date(np.asarray(tv.values), units, calendar)
        months = [int(np.atleast_1d(dates)[i].month) for i in range(nt)]
    except Exception:
        return None
    return months


def _load_nemo(path, tidx, month=None):
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    tdim = "time_counter" if "time_counter" in ds["tos"].dims else None
    if month is not None:
        # Climatological calendar-month mean: average every record whose CALENDAR
        # month is M, giving the NYF climatological month -- the SEASONALLY-MATCHED
        # reference for an instantaneous legoESM snapshot (whose perpetual-year date
        # is ~day-of-run).  Without this the scorer compares a spring snapshot to the
        # ANNUAL mean, manufacturing a hemispheric seasonal dipole that masquerades
        # as model bias.  The record months are decoded from CF time metadata (NOT
        # assumed Jan-first/positional ``(M-1)::12``) so a file with spin-up records,
        # a non-January start, or dropped months still selects the correct month or
        # fails loudly.
        if not (1 <= month <= 12):
            raise ValueError(f"--nemo-month must be 1..12, got {month}")
        if tdim is None:
            raise ValueError(f"--nemo-month set but {path} has no time dimension")
        nt = int(ds.sizes[tdim])
        rec_months = _nemo_record_months(ds, tdim, nt)
        if rec_months is not None:
            midx = [i for i in range(nt) if rec_months[i] == month]
        else:
            # CF time undecodable: fall back to positional stride, but ONLY if the
            # file is whole monthly years (Jan-first contract); else refuse.
            if nt % 12 != 0:
                raise ValueError(
                    f"--nemo-month: cannot decode time metadata of {path} and "
                    f"n_time={nt} is not a multiple of 12 (not whole monthly years) "
                    "-- refusing to guess the calendar month.")
            print(f"[nemo-month] WARN: time metadata undecodable; assuming "
                  f"Jan-first monthly contract (positional (M-1)::12).")
            midx = list(range(month - 1, nt, 12))
        if not midx:
            raise ValueError(f"--nemo-month {month}: no matching records "
                             f"(n_time={nt})")
        print(f"[nemo-month] month={month}: averaging {len(midx)} records "
              f"at indices {midx}")
        sel = lambda v: np.asarray(ds[v].isel({tdim: midx}).mean(dim=tdim))
    else:
        sel = (lambda v: np.asarray(ds[v].isel({tdim: tidx})) if tdim
               else np.asarray(ds[v]))
    sst = sel("tos"); sss = sel("sos")
    # NEMO density-threshold MLD (dsigma=0.01 wrt 10m); None if not archived.
    mld = sel("mldr10_1") if "mldr10_1" in ds.variables else None
    lat = np.asarray(ds["nav_lat"]); lon = np.asarray(ds["nav_lon"])
    # NEMO land/fill is already NaN here (xarray CF-decodes _FillValue=1e20), so
    # finiteness alone is the correct ocean mask. The old ``& (|sst| > 1e-6)``
    # magnitude clause was a redundant land test that would silently drop genuine
    # near-0 C ocean cells (upwelling / near-freezing) in float32.
    mask = np.isfinite(sst).astype(np.float64)
    return {"sst": np.nan_to_num(sst), "sss": np.nan_to_num(sss),
            "mld": mld,  # NEMO mldr10_1 [m] (finite over ocean, NaN land) or None
            "lat": lat, "lon": lon % 360.0, "mask": mask,
            "n_time": int(ds.sizes.get("time_counter", 1))}


def _plot(out_dir, tgt_lat, tgt_lon, fields, ocean, label="legoESM"):
    """Model | NEMO | Δ maps + zonal means.  ``label`` (e.g. the grid name) is
    shown in the model-panel titles, the difference title, the legend, and a
    figure suptitle so the grid is unambiguous in the saved PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, (L, N) in fields.items():
        Lm = np.where(ocean, L, np.nan); Nm = np.where(ocean, N, np.nan)
        vmin = np.nanmin([Lm, Nm]); vmax = np.nanmax([Lm, Nm])
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        for a, dat, ttl in [(ax[0], Lm, f"{label} {name}"),
                            (ax[1], Nm, f"NEMO {name}")]:
            im = a.pcolormesh(tgt_lon, tgt_lat, dat, vmin=vmin, vmax=vmax,
                              cmap="RdYlBu_r", shading="auto")
            a.set_title(ttl); plt.colorbar(im, ax=a, shrink=0.8)
        dmax = np.nanmax(np.abs(Lm - Nm))
        im = ax[2].pcolormesh(tgt_lon, tgt_lat, Lm - Nm, vmin=-dmax, vmax=dmax,
                              cmap="RdBu_r", shading="auto")
        ax[2].set_title(f"{name} diff ({label}-NEMO)")
        plt.colorbar(im, ax=ax[2], shrink=0.8)
        fig.suptitle(f"{label} vs NEMO — {name}", fontsize=13)
        fig.tight_layout(); fig.savefig(out_dir / f"{name}_maps.png", dpi=90)
        plt.close(fig)
    # zonal means
    fig, ax = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4))
    if len(fields) == 1:
        ax = [ax]
    for a, (name, (L, N)) in zip(ax, fields.items()):
        Lz = np.nanmean(np.where(ocean, L, np.nan), axis=1)
        Nz = np.nanmean(np.where(ocean, N, np.nan), axis=1)
        a.plot(Lz, tgt_lat, label=label); a.plot(Nz, tgt_lat, label="NEMO")
        a.set_title(f"zonal-mean {name}"); a.set_ylabel("lat"); a.legend()
    fig.suptitle(f"{label} vs NEMO — zonal means", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "zonal_means.png", dpi=90)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legoesm-snapshot", type=Path, required=True)
    p.add_argument("--use-mean-fields", action="store_true",
                   help="score the driver's --state-accumulate/--mld-accumulate "
                        "window means (the statistic NEMO's 5-day files hold) "
                        "instead of the instantaneous snapshot; refuses "
                        "snapshots that lack them")
    p.add_argument("--nemo-gridt", type=Path, required=True)
    p.add_argument("--nemo-time-idx", type=int, default=-1,
                   help="NEMO grid_T time record (default last).")
    p.add_argument("--nemo-month", type=int, default=None,
                   help="Calendar month 1-12: average all records of that month "
                        "across the file (climatological-month mean) instead of a "
                        "single --nemo-time-idx. Use a MONTHLY grid_T file. This is "
                        "the seasonally-matched reference for an instantaneous "
                        "legoESM snapshot; comparing a spring snapshot to the annual "
                        "mean fabricates a hemispheric seasonal dipole.")
    p.add_argument("--res-deg", type=float, default=1.0)
    p.add_argument("--output-dir", type=Path, default=Path("results/omip_nemo/compare"))
    p.add_argument("--grid-label", type=str, default="legoESM",
                   help="Grid/run name shown in the map+zonal plot titles "
                        "(e.g. 'MPAS ico6 5yr+SSS').")
    p.add_argument("--freeze-clamp-C", type=float, default=None,
                   help="Floor legoESM SST at this temperature [deg C] before "
                        "scoring, to mimic NEMO's sea-ice-capped surface "
                        "(~-1.9 C). legoESM has no sea ice so high-lat cells "
                        "cool below freezing; this tests how much that inflates "
                        "the SST RMSE/bias vs NEMO.")
    p.add_argument("--sss-faithful", action="store_true",
                   help="Un-gate SSS: report it as a real pass/fail metric with a "
                        "verdict (same excellent/good/poor logic as SST), NOT the "
                        "legacy 'GATED: runoff=0' caveat. Set this when the run "
                        "actually applied the freshwater closure (--runoff is now "
                        "wired for latlon/tripole/mpas). Default off keeps the old "
                        "informational-only behaviour for legacy no-runoff runs.")
    args = p.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    L = _load_legoesm(args.legoesm_snapshot, use_mean=args.use_mean_fields)
    N = _load_nemo(args.nemo_gridt, args.nemo_time_idx, month=args.nemo_month)
    print(f"[load] legoESM {L['sst'].shape}, NEMO {N['sst'].shape} "
          f"({N['n_time']} time records)")

    r = args.res_deg
    tgt_lat = np.arange(-89.5, 90.0, r)
    tgt_lon = np.arange(0.5, 360.0, r)

    # WET CELLS ONLY: dry cells carry fill values (measured -1.83 to 31.81 on
    # the MPAS snapshot), so an unmasked count reported 1810 supercooled cells
    # where the ocean has 109 — the same land-contamination trap that bit the
    # IC regridder and three probes this session.
    _wet = np.asarray(L["mask"]) > 0.5
    _sst_wet = np.asarray(L["sst"])[_wet]
    n_below, sst_min = 0, float(np.nanmin(_sst_wet))
    if args.freeze_clamp_C is not None:
        n_below = int((_sst_wet < args.freeze_clamp_C).sum())
        L["sst"] = np.maximum(L["sst"], args.freeze_clamp_C)
        print(f"[freeze-clamp] floored legoESM SST at {args.freeze_clamp_C} C "
              f"({n_below} cells were below)")
        # A liquid ocean below its freezing point is a MODEL DEFECT (NEMO's
        # SI3 never lets it happen); the clamp exists so the SST score is not
        # dominated by it, and it hid a -8 C supercooling for weeks. Shout.
        if n_below > 0:
            print(f"[freeze-clamp] WARNING: {n_below} wet cells supercooled "
                  f"(min {sst_min:.2f} C) — the score below is on the FLOORED "
                  "field; the raw state has an ocean below freezing.",
                  flush=True)
    sstL, ocL = regrid_curv_to_latlon(L["sst"], L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)
    sstN, ocN = regrid_curv_to_latlon(N["sst"], N["lat"], N["lon"], N["mask"], tgt_lat, tgt_lon)
    sssL, _ = regrid_curv_to_latlon(L["sss"], L["lat"], L["lon"], L["mask"], tgt_lat, tgt_lon)
    sssN, _ = regrid_curv_to_latlon(N["sss"], N["lat"], N["lon"], N["mask"], tgt_lat, tgt_lon)
    ocean = (ocL > 0.5) & (ocN > 0.5)
    area = (np.cos(np.deg2rad(tgt_lat))[:, None]
            * np.ones_like(tgt_lon)[None, :]) * ocean

    sst = _wstats(sstL, sstN, area)
    sss = _wstats(sssL, sssN, area)
    print(f"[SST] {sst}")
    if args.sss_faithful:
        print(f"[SSS] {sss}  (faithful: --runoff applied; scored)")
    else:
        print(f"[SSS] {sss}  (GATED: runoff=0, informational)")
    sst_bands = _band_breakdown(sstL, sstN, area, tgt_lat)
    print("[SST bands]")
    for bn, bs in sst_bands.items():
        if bs is not None:
            print(f"   {bn:22s} rmse={bs['rmse']:.2f} bias={bs['bias']:+.2f} "
                  f"corr={bs['corr']:.3f}")
    # ENSO boxes. Reported for SST only: the equatorial Pacific is where an
    # ocean-only run's cold-tongue error lives, and the 23S-23N band average
    # hides it by mixing the cold tongue with both subtropical gyres.
    sst_boxes = _box_breakdown(sstL, sstN, area, tgt_lat, tgt_lon)
    print("[SST Pacific/ENSO boxes]")
    for bn, bs in sst_boxes.items():
        if bs is not None:
            print(f"   {bn:24s} rmse={bs['rmse']:.2f} bias={bs['bias']:+.2f} "
                  f"corr={bs['corr']:.3f}")

    # Per-band SSS too (only when the freshwater closure is applied, --sss-faithful):
    # localises WHERE a global SSS bias comes from -- the river-mouth / Arctic /
    # basin breakdown that pinned the MPAS runoff over-concentration (PR #560).
    sss_bands = (_band_breakdown(sssL, sssN, area, tgt_lat)
                 if args.sss_faithful else None)
    if sss_bands is not None:
        print("[SSS bands]")
        for bn, bs in sss_bands.items():
            if bs is not None:
                print(f"   {bn:22s} rmse={bs['rmse']:.2f} bias={bs['bias']:+.2f} "
                      f"corr={bs['corr']:.3f}")

    def _verdict(rmse, exc, good):
        return ("excellent" if rmse < exc else "good" if rmse < good else "poor")
    sst_v, capped_by = capped_sst_verdict(sst["rmse"], sst_boxes)
    if capped_by is not None:
        print(f"[verdict] SST capped to {sst_v} by {capped_by[0]} "
              f"bias {capped_by[1]:+.2f} C")
    sss_v = (_verdict(sss["rmse"], _TOL["sss_rmse_excellent"], _TOL["sss_rmse_good"])
             if args.sss_faithful else None)

    # Mixed-layer depth (de Boyer Montegut / Treguier 2023, GMD 16:3849).
    # Needs legoESM full T/S + geometry (z_center_ref, H_bathy from a snapshot
    # written after _save_snapshot grew the MLD geometry) AND NEMO mldr10_1.
    # Uses delta_sigma=0.01 to MATCH NEMO mldr10_1 (a larger threshold -> deeper
    # MLD, so 0.03-vs-0.01 would bias the model deep).  Snapshot/annual-state
    # MLD: MLD(mean T,S) is not seasonal-mean MLD -- label accordingly.
    plot_fields = {"SST": (sstL, sstN), "SSS": (sssL, sssN)}
    mld_report = None
    if (N.get("mld") is not None and L.get("z_center_ref") is not None
            and L.get("H_bathy") is not None):
        from legoesm.ocean.diagnostics import mixed_layer_depth
        z_c = np.asarray(L["z_center_ref"], dtype=np.float64)         # (nlev,)
        Hb = np.asarray(L["H_bathy"], dtype=np.float64)
        # Per-level wet mask: level centre above the sea floor AND in the ocean.
        wet = ((z_c[(None,) * Hb.ndim + (slice(None),)] < Hb[..., None])
               & (L["mask"][..., None] > 0.5)).astype(np.float64)
        bottom = Hb
        if L.get("mld_mean") is not None:
            # --use-mean-fields: the window-mean MLD (mean of MLDs, as NEMO's
            # mldr10_1 is), never the MLD of the mean state.
            mldL = np.asarray(L["mld_mean"], dtype=np.float64)
        else:
            mldL = np.asarray(mixed_layer_depth(
                L["T3d"], L["S3d"], z_c, delta_sigma=0.01,
                wet_mask=wet, bottom_depth=bottom))
        mldL_g, ocLm = regrid_curv_to_latlon(
            np.nan_to_num(mldL, nan=0.0), L["lat"], L["lon"],
            np.isfinite(mldL).astype(np.float64), tgt_lat, tgt_lon)
        mldN_g, ocNm = regrid_curv_to_latlon(
            np.nan_to_num(N["mld"], nan=0.0), N["lat"], N["lon"],
            np.isfinite(N["mld"]).astype(np.float64), tgt_lat, tgt_lon)
        mld_ocean = ocean & (ocLm > 0.5) & (ocNm > 0.5)
        mld_area = (np.cos(np.deg2rad(tgt_lat))[:, None]
                    * np.ones_like(tgt_lon)[None, :]) * mld_ocean
        mld_raw = _wstats(mldL_g, mldN_g, mld_area)
        mld_log = _wstats(np.log1p(np.maximum(mldL_g, 0.0)),
                          np.log1p(np.maximum(mldN_g, 0.0)), mld_area)
        # Median bias is robust to the deep-convection tail that dominates RMSE.
        finite = mld_ocean & np.isfinite(mldL_g) & np.isfinite(mldN_g)
        med_bias = float(np.median((mldL_g - mldN_g)[finite])) if finite.any() else float("nan")
        # Per-band MLD (same latitude bands as SST/SSS) so a global MLD bias can
        # be attributed to a band rather than blamed on a few deep-convection
        # cells that inflate the global mean -- e.g. is the mixed layer too deep
        # exactly in the NH-midlat band where SST is coldest (entrainment link)?
        mld_bands = _band_breakdown(mldL_g, mldN_g, mld_area, tgt_lat)
        mld_report = {**mld_raw, "rmse_log1p_m": mld_log["rmse"],
                      "median_bias_m": med_bias, "delta_sigma": 0.01,
                      "bands": mld_bands,
                      "method": "de Boyer Montegut / Treguier 2023; dsigma=0.01 wrt 10m "
                                "to match NEMO mldr10_1; snapshot/annual-state (not seasonal)"}
        print(f"[MLD] rmse {mld_raw['rmse']:.1f} m  bias {mld_raw['bias']:+.1f} m  "
              f"median-bias {med_bias:+.1f} m  corr {mld_raw['corr']:.3f}  "
              f"rmse(log1p) {mld_log['rmse']:.3f}")
        print("[MLD bands]")
        for bn, bs in mld_bands.items():
            if bs is not None:
                print(f"  {bn:22s} bias {bs['bias']:+7.1f} m  lego {bs['lego_mean']:6.1f}  "
                      f"nemo {bs['nemo_mean']:6.1f}  corr {bs['corr']:.3f}")
        # Cap at the 99th percentile for display so deep-convection cells don't
        # wash out the colour scale (scoring above uses raw metres).
        cap = float(np.nanpercentile(np.where(mld_ocean, mldN_g, np.nan), 99))
        plot_fields["MLD"] = (np.minimum(mldL_g, cap), np.minimum(mldN_g, cap))
    elif N.get("mld") is not None:
        print("[mld] SKIPPED: NEMO mldr10_1 present but the legoESM snapshot lacks "
              "the z_center_ref and/or H_bathy geometry (written by _save_snapshot "
              "only for runs after the MLD-geometry change) -- re-run to enable the "
              "MLD comparison.")

    report = {
        "legoesm_snapshot": str(args.legoesm_snapshot),
        "nemo_gridt": str(args.nemo_gridt), "nemo_time_idx": args.nemo_time_idx,
        "sst_supercooled_cells": n_below, "sst_raw_min_C": sst_min,
        "n_ocean_cells": int(ocean.sum()),
        "SST": sst, "SST_verdict": sst_v,
        "SST_bands": sst_bands,
        "SST_pacific_boxes": sst_boxes,
        # SSS is a real scored metric when the run applied the freshwater closure
        # (--sss-faithful); otherwise the legacy informational-only gated key.
        **({"SSS": sss, "SSS_verdict": sss_v, "SSS_bands": sss_bands}
           if args.sss_faithful else {"SSS_gated_runoff0": sss}),
        "MLD_dsigma0p01": mld_report,
        "tolerances": _TOL,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    _plot(out, tgt_lat, tgt_lon, plot_fields, ocean, label=args.grid_label)
    print(f"[verdict] SST match = {sst_v} (RMSE {sst['rmse']:.3f} C, "
          f"bias {sst['bias']:.3f} C, corr {sst['corr']:.3f})")
    if sss_v is not None:
        print(f"[verdict] SSS match = {sss_v} (RMSE {sss['rmse']:.3f}, "
              f"bias {sss['bias']:.3f}, corr {sss['corr']:.3f})")
    print(f"[done] report + plots -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
