#!/usr/bin/env python
"""AMIP pattern evaluation + zonal-mean physicality panels from CMOR output.

Complements the global-MEAN scorecards (``plot_amip_cmor_diagnostics.py`` /
``scripts/validate/amip_skill_score.py``) with the SPATIAL dimension the
2026-07-24 ladder analysis needs:

1. PATTERN evaluation of tas / pr / rlut / rsut: model map, reference map,
   model−reference bias map, plus area-weighted bias, centered RMSE and
   centered pattern correlation per field.  References: ERA5 (tas), GPCP
   (pr), CERES-EBAF (rlut/rsut) from the ClimateEval reference store.
2. ZONAL-MEAN pressure–latitude sections of ta and ua vs ERA5 (jet position/
   strength, tropopause, polar inversions) + 2-D zonal-mean line profiles.
3. Physicality table: TOA net budget, E−P closure, land Bowen ratio.

Calendar-month matching: the model mean uses whatever months the run wrote;
the reference climatology is built from the SAME calendar months (a 30-day
January run is compared against reference Januaries, never an annual mean).
Reference epoch: ERA5/GPCP are sliced to the model years when they cover
them (epoch-matched); CERES-EBAF starts 2000-03, so rlut/rsut compare
against its full-period per-month climatology — a stated protocol caveat,
not a silent one (printed in the metrics table and figure titles).

Pure numpy/xarray/matplotlib — no legoesm import; runs on any CMOR dir.

Usage::

    python scripts/plot/plot_amip_pattern_eval.py <run_dir> \
        [--ref-root /scratch/b/b309178/climateeval_data] [--out prefix]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Metric helpers (dependency-light, unit-tested in
# tests/unit/test_amip_pattern_eval_metrics.py)
# ---------------------------------------------------------------------------

def area_weights(lat_deg: np.ndarray) -> np.ndarray:
    """cos(lat) weights normalised to mean 1 (shape (nlat,))."""
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=float)))
    return w / w.mean()


def weighted_global_mean(field: np.ndarray, lat_deg: np.ndarray) -> float:
    """Area-weighted mean of ``field`` (..., nlat, nlon); NaN-aware."""
    w = area_weights(lat_deg)[..., :, None] * np.ones_like(field)
    m = np.isfinite(field)
    return float((field * w)[m].sum() / w[m].sum())


def pattern_stats(model: np.ndarray, ref: np.ndarray,
                  lat_deg: np.ndarray) -> dict:
    """Area-weighted bias, centered RMSE and centered pattern correlation.

    Both fields (nlat, nlon) on the SAME grid.  "Centered" = the respective
    area-weighted means are removed first (Taylor-diagram convention), so the
    correlation measures pattern agreement independent of the mean bias.
    NaNs (missing reference cells) are excluded pairwise.
    """
    model = np.asarray(model, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ok = np.isfinite(model) & np.isfinite(ref)
    w2 = area_weights(lat_deg)[:, None] * np.ones_like(model)
    w = w2[ok]
    m = model[ok]
    r = ref[ok]
    mbar = (m * w).sum() / w.sum()
    rbar = (r * w).sum() / w.sum()
    mc, rc = m - mbar, r - rbar
    bias = mbar - rbar
    rmse_c = float(np.sqrt((w * (mc - rc) ** 2).sum() / w.sum()))
    denom = np.sqrt((w * mc**2).sum() * (w * rc**2).sum())
    corr = float((w * mc * rc).sum() / denom) if denom > 0 else np.nan
    return {"bias": float(bias), "rmse_centered": rmse_c, "pattern_corr": corr}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# (cmor var, ref dataset dir, ref var, unit scale to display, display unit,
#  reference-epoch policy: "match" = slice ref to the model years when
#  covered, "climatology" = full-period per-month climatology)
PATTERN_FIELDS = (
    ("tas", "reanalysis_ERA5", "tas", 1.0, "K", "match"),
    ("pr", "observation_GPCP", "pr", 86400.0, "mm/day", "match"),
    ("rlut", "observation_CERES-EBAF", "rlut", 1.0, "W/m2", "climatology"),
    ("rsut", "observation_CERES-EBAF", "rsut", 1.0, "W/m2", "climatology"),
)
ZONAL3D_FIELDS = (("ta", "K", 1.0), ("ua", "m/s", 1.0))


def parse_months(spec):
    """Parse a ``--months`` spec ("1-8", "1,2,12", "3") into a sorted list.

    Returns None for an empty spec (= use every month the run wrote).  Raises
    on anything outside 1-12 rather than silently dropping it, so a typo can
    never quietly change the comparison window.
    """
    if not spec:
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise ValueError(f"--months range {part!r} is reversed")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out or any(m < 1 or m > 12 for m in out):
        raise ValueError(f"--months {spec!r} must select calendar months 1-12")
    return sorted(out)


def _open_cmor(run_dir: Path, var: str, months=None):
    """Open a CMOR variable, optionally restricted to calendar ``months``.

    The month filter is applied HERE so every consumer (pattern maps, zonal
    sections, the TOA/Bowen table) shares one window — comparing two runs
    whose CMOR archives have different lengths requires pinning the window on
    both, else the difference is a sampling confound rather than a result.
    """
    import xarray as xr
    hits = sorted(glob.glob(str(run_dir / "cmor" / "Amon" / f"{var}_*.nc")))
    if not hits:
        return None
    da = xr.open_mfdataset(hits, combine="by_coords")[var]
    if months is not None:
        da = da.sel(time=da["time"].dt.month.isin(months))
        if da.sizes.get("time", 0) == 0:
            return None
    return da


def _open_ref(ref_root: Path, dataset: str, var: str):
    """Open a reference variable time-CHUNKED and horizontally DECIMATED.

    The ERA5 native6 3-D monthly files are ~90 GB (567 x 37 x 721 x 1440);
    un-chunked opens + full-resolution month means OOM a login node.  Chunk
    by single months and stride the horizontal axes down to ~180 points
    (~1 deg) BEFORE any arithmetic — the comparison grid is the model's
    5 deg CMOR grid, so ~1 deg reference sampling is accuracy-neutral.
    """
    import xarray as xr
    hits = sorted(glob.glob(str(ref_root / dataset / "mon" / var / "*.nc")))
    if not hits:
        return None
    ds = xr.open_mfdataset(hits, combine="by_coords", chunks={"time": 1})
    # obs4MIPs/native6 files name the variable canonically.
    da = ds[var] if var in ds else ds[list(ds.data_vars)[0]]
    for dim in ("lat", "lon"):
        n = da.sizes.get(dim, 0)
        if n > 360:
            da = da.isel({dim: slice(None, None, max(1, n // 180))})
    return da


def _monthly_matched_ref(ref, model_times, epoch: str):
    """Reference field averaged over the model's calendar months.

    ``epoch='match'``: restrict the reference to the model's YEARS first when
    the reference covers them (epoch-matched); else fall back to climatology.
    ``epoch='climatology'``: per-month climatology over the full reference
    period, averaged over the model's month set (weighted by how often each
    calendar month appears in the model mean).
    Returns (ref_mean(lat[,plev],lon), note_string).
    """
    months = model_times.dt.month.values
    years = np.unique(model_times.dt.year.values)
    note = ""
    r = ref
    ry = np.unique(ref["time"].dt.year.values)
    if epoch == "match" and set(years).issubset(set(ry)):
        r = ref.sel(time=ref["time"].dt.year.isin(years))
        note = f"epoch-matched {years.min()}-{years.max()}"
    else:
        note = f"climatology {ry.min()}-{ry.max()}"
    # weight reference calendar months by the model's month multiplicity
    uniq, counts = np.unique(months, return_counts=True)
    parts = []
    for mth, cnt in zip(uniq, counts):
        sel = r.sel(time=r["time"].dt.month == mth)
        if sel.sizes.get("time", 0) == 0:
            continue
        parts.append(sel.mean("time") * float(cnt))
    out = sum(parts) / float(counts.sum())
    return out, note + f", months={[int(m) for m in uniq]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ref-root", type=Path,
                    default=Path("/scratch/b/b309178/climateeval_data"))
    ap.add_argument("--out", default="", help="output file prefix "
                    "(default <run_dir>/pattern_eval)")
    ap.add_argument("--label", default="")
    ap.add_argument("--months", default="",
                    help="restrict BOTH model and reference to these "
                         "calendar months (e.g. '1-8' or '1,2,12'); "
                         "default = every month the run wrote. Pin this "
                         "when comparing runs of different length.")
    args = ap.parse_args()
    months = parse_months(args.months)
    run = args.run_dir
    label = args.label or run.name
    out_prefix = args.out or str(run / "pattern_eval")

    metrics = {}

    # ---- 1. Pattern maps -------------------------------------------------
    nf = len(PATTERN_FIELDS)
    fig, axes = plt.subplots(nf, 3, figsize=(16, 3.1 * nf))
    for i, (var, refset, refvar, scale, unit, epoch) in enumerate(PATTERN_FIELDS):
        da = _open_cmor(run, var, months)
        if da is None:
            for ax in axes[i]:
                ax.set_axis_off()
            axes[i, 0].set_title(f"{var}: no CMOR output")
            continue
        model = da.mean("time") * scale
        ref_da = _open_ref(args.ref_root, refset, refvar)
        if ref_da is None:
            for ax in axes[i]:
                ax.set_axis_off()
            axes[i, 0].set_title(f"{var}: no reference in {refset}")
            continue
        ref_m, note = _monthly_matched_ref(ref_da, da["time"], epoch)
        ref_m = ref_m * scale
        # regrid reference (finer) onto the model 5° grid
        ref_i = ref_m.interp(lat=model["lat"], lon=model["lon"],
                             method="linear")
        mv = np.asarray(model.values)
        rv = np.asarray(ref_i.values)
        st = pattern_stats(mv, rv, model["lat"].values)
        st["ref"] = f"{refset} ({note})"
        metrics[var] = st

        vmin = np.nanpercentile(np.concatenate([mv.ravel(), rv.ravel()]), 2)
        vmax = np.nanpercentile(np.concatenate([mv.ravel(), rv.ravel()]), 98)
        for j, (fld, ttl) in enumerate(
                ((mv, f"{label} {var} [{unit}]"),
                 (rv, f"{refset.split('_')[-1]} {var}"),
                 (mv - rv, f"bias  r={st['pattern_corr']:.2f} "
                           f"rmse={st['rmse_centered']:.3g}"))):
            ax = axes[i, j]
            if j < 2:
                pm = ax.pcolormesh(model["lon"], model["lat"], fld,
                                   vmin=vmin, vmax=vmax, cmap="viridis")
            else:
                a = np.nanpercentile(np.abs(fld), 98)
                pm = ax.pcolormesh(model["lon"], model["lat"], fld,
                                   vmin=-a, vmax=a, cmap="RdBu_r")
            plt.colorbar(pm, ax=ax, shrink=0.85)
            ax.set_title(ttl, fontsize=9)
    fig.suptitle(f"AMIP pattern evaluation — {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_maps.png", dpi=130)
    plt.close(fig)

    # ---- 2. Zonal-mean pressure-latitude sections (ta, ua vs ERA5) ------
    fig2, axes2 = plt.subplots(len(ZONAL3D_FIELDS), 3,
                               figsize=(15, 4.2 * len(ZONAL3D_FIELDS)))
    for i, (var, unit, scale) in enumerate(ZONAL3D_FIELDS):
        da = _open_cmor(run, var, months)
        ref_da = _open_ref(args.ref_root, "reanalysis_ERA5", var)
        if da is None or ref_da is None:
            for ax in axes2[i]:
                ax.set_axis_off()
            continue
        model_z = (da.mean("time").mean("lon") * scale)
        # (_open_ref already time-chunks + decimates the fine reanalysis.)
        ref_m, note = _monthly_matched_ref(ref_da, da["time"], "match")
        # ERA5 plev may be ascending/descending + finer: interp to model levels
        ref_z = (ref_m.mean("lon")
                 .interp(plev=model_z["plev"], lat=model_z["lat"],
                         method="linear") * scale)
        mz = np.asarray(model_z.values)   # (plev, lat)
        rz = np.asarray(ref_z.values)
        p_hpa = model_z["plev"].values / 100.0
        lat = model_z["lat"].values
        vmin, vmax = np.nanpercentile(np.concatenate(
            [mz.ravel(), rz.ravel()]), [2, 98])
        for j, (fld, ttl, diverging) in enumerate(
                ((mz, f"{label} zonal {var} [{unit}]", var == "ua"),
                 (rz, f"ERA5 zonal {var} ({note.split(',')[0]})", var == "ua"),
                 (mz - rz, f"bias zonal {var}", True))):
            ax = axes2[i, j]
            if j == 2:
                a = np.nanpercentile(np.abs(fld), 98)
                kw = dict(vmin=-a, vmax=a, cmap="RdBu_r")
            elif diverging:
                a = max(abs(vmin), abs(vmax))
                kw = dict(vmin=-a, vmax=a, cmap="RdBu_r")
            else:
                kw = dict(vmin=vmin, vmax=vmax, cmap="viridis")
            pm = ax.pcolormesh(lat, p_hpa, fld, **kw)
            ax.invert_yaxis()
            ax.set_yscale("log")
            ax.set_ylabel("p [hPa]" if j == 0 else "")
            ax.set_title(ttl, fontsize=9)
            plt.colorbar(pm, ax=ax, shrink=0.85)
        st = pattern_stats(mz.T, rz.T, lat)   # weight rows by lat
        metrics[f"zonal_{var}"] = {**st, "ref": f"ERA5 ({note})"}
    fig2.suptitle(f"Zonal-mean sections — {label}", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_zonal.png", dpi=130)
    plt.close(fig2)

    # ---- 3. Physicality table -------------------------------------------
    lines = [f"# AMIP pattern evaluation — {label}", ""]
    for k, v in metrics.items():
        lines.append(
            f"{k:10s} bias={v['bias']:+9.3f}  rmse_c={v['rmse_centered']:8.3f}"
            f"  r_pattern={v['pattern_corr']:6.3f}   vs {v['ref']}")
    # TOA budget + E-P closure from the model's own CMOR output
    extras = {}
    for v in ("rsdt", "rsut", "rlut", "pr", "evspsbl", "hfls", "hfss"):
        da = _open_cmor(run, v, months)
        if da is not None:
            extras[v] = weighted_global_mean(
                np.asarray(da.mean("time").values), da["lat"].values)
    if {"rsdt", "rsut", "rlut"} <= extras.keys():
        toa_net = extras["rsdt"] - extras["rsut"] - extras["rlut"]
        lines.append(f"\nTOA net (rsdt-rsut-rlut) = {toa_net:+.2f} W/m2 "
                     "(obs ~ +0.9; AMIP-equilibrated |net| < ~5 healthy)")
    if {"pr", "evspsbl"} <= extras.keys():
        emp = (extras["evspsbl"] - extras["pr"]) * 86400.0
        lines.append(f"E - P (global) = {emp:+.3f} mm/day "
                     "(atmospheric water closure: -> 0 over the mean)")
    if {"hfls", "hfss"} <= extras.keys():
        lines.append(f"global Bowen (hfss/hfls) = "
                     f"{extras['hfss'] / max(extras['hfls'], 1e-9):.2f} "
                     "(obs ~ 0.2-0.3)")
    report = "\n".join(lines)
    print(report)
    with open(f"{out_prefix}_metrics.txt", "w") as fh:
        fh.write(report + "\n")
    print(f"\nwrote {out_prefix}_maps.png / _zonal.png / _metrics.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
