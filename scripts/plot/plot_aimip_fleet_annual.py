#!/usr/bin/env python
"""Download AIMIP-1 fleet monthly tas + plot annual global-mean spread.

For each model in :data:`MODELS`, downloads monthly near-surface air
temperature (`tas`) for ensemble members r1..r5 from the public AIMIP
S3 bucket (``s3://ai-mip``, anon access via DKRZ endpoint).  Computes
the annual global-mean tas time series per ensemble member, then
plots each model as an envelope between min and max across members
with the ensemble median as a centre line.  ERA5 is overlaid as a
reference.

This is the canonical AIMIP figure (Phase-1 paper-style time-mean and
trend comparison).  Our local legoESM v12 classical / column_nn /
sfno_physics models were trained on 6-hour forecast pairs, NOT on
46-year AMIP integrations, so we cannot honestly overlay our model
on this figure.  Drift over 46 years from a 6h-trained model is
unbounded; the AIMIP submissions are full inference simulations.
If you want our model in this comparison, that is a separate
multi-week effort (run our model in 46-year inference mode + stable
SST forcing).

Usage::

    .venv/bin/python scripts/plot_aimip_fleet_annual.py \\
        --out results/aimip_001/aimip_fleet_tas_annual.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import fsspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


logger = logging.getLogger("aimip-fleet")

S3_ENDPOINT = "https://s3.eu-dkrz-1.dkrz.cloud"
S3_BUCKET = "ai-mip"

# Models with their submission path templates.  Each entry contains:
#   - org folder
#   - model folder
#   - file pattern relative to model folder (with {i_r} = ensemble member,
#     {table}, {varname}, {grid}, {label}, {time_period})
# Versions / labels are baked into the path per ai2cm/AIMIP/SUBMISSIONS.md.
#
# 2026-05-25: MD-1.5 and cBottle entries removed -- 2026-05-24 fetch
# hung indefinitely on the S3 endpoint (sleeping >15 hours with no
# bytes transferred).  Restore those once the upstream bucket
# response times stabilise.  DLESyM also drops because the NC files
# raise ``StopIteration`` (xarray fails to parse them via h5py --
# likely a non-standard NC4 layout).
MODELS = {
    "ACE2.1-ERA5": dict(
        path="Ai2/ACE2-1-ERA5",
        # tas is published on the NATIVE grid (gn), NOT gr — the gr path 404s
        # despite the submissions manifest listing gr (2026-06-30 bucket probe:
        # only .../Amon/tas/gn/v20251130/...gn_197810-202412.nc exists).
        template="aimip/r{i_r}i1p1f1/Amon/tas/gn/v20251130/"
                 "tas_Amon_ACE2-ERA5_aimip_r{i_r}i1p1f1_gn_197810-202412.nc",
    ),
    "ArchesWeather": dict(
        path="ArchesWeather/ArchesWeather-V2",
        template="aimip/r{i_r}i1p1f1/Amon/tas/gn/"
                 "tas_Amon_ArchesWeather_aimip_r{i_r}i1p1f1_gn_197810-202501.nc",
    ),
    "NeuralGCM": dict(
        path="Google/NeuralGCM",
        template="aimip/r{i_r}i1p1f1/Amon/tas/gn/v20260304/"
                 "tas_Amon_NeuralGCM_aimip_r{i_r}i1p1f1_gn_197810-202412.nc",
    ),
    "NeuralGCM-HRD": dict(
        path="Google/NeuralGCM-HRD",
        template="aimip/r{i_r}i1p1f1/Amon/tas/gn/v20260304/"
                 "tas_Amon_NeuralGCM-HRD_aimip_r{i_r}i1p1f1_gn_197810-202412.nc",
    ),
}

# cBottle (and DLESyM, MD-1.5) intentionally skipped pending upstream
# bucket fix; see comment above.
CBOTTLE = None

ERA5_PATH = "ERA5/mon/ERA5_2m_temperature_mon_full_sfc_1940-2024.nc"
ERA5_VARNAME = "T2M"

# v13: restrict to full years 1979-2024 to avoid spikes at series
# boundaries.  ArchesWeather goes 1978-10 -> 2025-01 (only 3 months
# for "1978" and 1 month for "2025"); resampling to annual means on
# those partial windows gave seasonal-biased outliers that visually
# dominated the plot.
YEAR_MIN = 1979
YEAR_MAX = 2024

N_ENS = 5

COLORS = {
    "ACE2.1-ERA5": "#E69F00",
    "ArchesWeather": "#56B4E9",
    "NeuralGCM": "#009E73",
    "NeuralGCM-HRD": "#0072B2",
    "DLESyM": "#D55E00",
    "MD-1.5": "#CC79A7",
    "cBottle1.3": "#882255",
    "ERA5": "k",
}


def _open_remote(fs, key: str) -> xr.Dataset:
    """Open a netCDF over the public AIMIP S3 endpoint via fsspec."""
    full = f"{S3_BUCKET}/{key}"
    with fs.open(full, "rb") as fh:
        return xr.open_dataset(fh, engine="h5netcdf").load()


def _global_annual(ds: xr.Dataset, varname: str = "tas") -> xr.DataArray:
    """Lat-weighted global mean -> annual mean over full years YEAR_MIN..YEAR_MAX."""
    da = ds[varname]
    # weight by cos(lat)
    if "lat" in da.dims:
        lat_name = "lat"
    else:
        lat_name = next(d for d in da.dims if "lat" in d.lower())
    lon_name = "lon" if "lon" in da.dims else next(
        d for d in da.dims if "lon" in d.lower()
    )
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights.name = "wgt"
    monthly = da.weighted(weights).mean(dim=(lat_name, lon_name))
    # Trim to full calendar years before computing annual means.
    monthly = monthly.sel(
        time=slice(f"{YEAR_MIN}-01-01", f"{YEAR_MAX}-12-31"),
    )
    # Annual mean over Jan-Dec calendar years.
    annual = monthly.groupby("time.year").mean()
    return annual


# Our variants' overlay colors on the fleet figure.
LEGOESM_VARIANT_COLORS = {
    "classical": "#AA3377", "column_nn": "#EE7733", "sfno_physics": "#009988",
    "column_nn_dense": "#CC3311", "sfno_physics_dense": "#33BBEE",
}


def _legoesm_variant_from_stem(stem: str) -> str:
    """Parse the variant out of a legoesm_<variant>_amip*.csv stem.

    Prefers the LONGEST matching key so a specific variant
    (``column_nn_dense``) wins over its prefix (``column_nn``). Falls
    back to the raw stem for a non-conforming name (still plotted,
    labelled by filename).
    """
    matches = [v for v in LEGOESM_VARIANT_COLORS if f"_{v}_" in f"_{stem}_"]
    return max(matches, key=len) if matches else stem


def _subtract_baseline(arr: xr.DataArray, b0: int, b1: int) -> xr.DataArray:
    """Return ``arr`` as an anomaly from its mean over calendar years [b0, b1].

    ``arr`` carries a ``year`` coordinate (fleet arrays are (member, year);
    ERA5 is (year,)).  The baseline mean is reduced over ALL dims, so a single
    scalar offset is removed and the inter-member envelope is preserved -- the
    paper Fig-3 convention (anomalies from the 1979-2014 training-period mean).
    Falls back to the full-series mean if the baseline window is empty.
    """
    sub = arr.sel(year=slice(b0, b1))
    base = float(sub.mean()) if sub.size else float(arr.mean())
    return arr - base


def _load_model(fs, name: str, info: dict) -> xr.DataArray | None:
    """Return (n_ens, n_years) DataArray of annual global-mean tas."""
    series = []
    for i_r in range(1, N_ENS + 1):
        key = f"{info['path']}/{info['template'].format(i_r=i_r)}"
        try:
            ds = _open_remote(fs, key)
            annual = _global_annual(ds)
            series.append(annual)
            logger.info(f"  [{name}] r{i_r}: {len(annual)} years")
        except Exception as exc:
            logger.warning(f"  [{name}] r{i_r}: skip ({exc!r})")
    if not series:
        return None
    # Align on a common time axis.
    return xr.concat(
        series, dim=xr.DataArray(np.arange(1, len(series) + 1), dims="member"),
    )


def _load_era5(fs) -> xr.DataArray | None:
    try:
        ds = _open_remote(fs, ERA5_PATH)
        return _global_annual(ds, varname=ERA5_VARNAME)
    except Exception as exc:
        logger.warning(f"ERA5: skip ({exc!r})")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/aimip_001/aimip_fleet_tas_annual.png"),
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("results/aimip_001/aimip_fleet_tas_annual.csv"),
    )
    parser.add_argument(
        "--anomaly", action="store_true",
        help="Plot anomalies from the baseline-period mean (paper Fig-3 "
             "style) rather than absolute tas.",
    )
    parser.add_argument(
        "--anomaly-base-start", type=int, default=1979,
        help="First calendar year of the anomaly baseline (paper: 1979).",
    )
    parser.add_argument(
        "--anomaly-base-end", type=int, default=2014,
        help="Last calendar year of the anomaly baseline (paper training "
             "period ends 2014).",
    )
    parser.add_argument(
        "--legoesm-csv", type=Path, default=None, action="append",
        dest="legoesm_csvs",
        help="Annual CSV (year,annual_global_mean_surfT_K) from "
             "scripts/run/run_aimip_amip_inference.py -> overlays a trained "
             "legoESM variant's prescribed-SST AMIP run on the fleet figure. "
             "REPEATABLE (one per variant: classical / column_nn / "
             "sfno_physics — the variant is parsed from the filename "
             "legoesm_<variant>_amip*.csv). Plotted as an anomaly from its "
             "OWN baseline window (so the near-surface-vs-2m absolute offset "
             "drops out, leaving a like-for-like trend/variability "
             "comparison); bias/RMSE vs ERA5 join the metrics CSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fs = fsspec.filesystem(
        "s3",
        anon=True,
        client_kwargs={"endpoint_url": S3_ENDPOINT},
    )

    # ERA5 reference.
    logger.info("Loading ERA5 reference...")
    era5_annual = _load_era5(fs)

    # AIMIP fleet.
    fleet: dict[str, xr.DataArray] = {}
    for name, info in MODELS.items():
        logger.info(f"Loading {name}...")
        arr = _load_model(fs, name, info)
        if arr is not None:
            fleet[name] = arr

    # cBottle (gated -- see CBOTTLE = None above).
    if CBOTTLE is not None:
        logger.info(f"Loading {CBOTTLE['name']}...")
        cb = _load_model(fs, CBOTTLE["name"], CBOTTLE)
        if cb is not None:
            fleet[CBOTTLE["name"]] = cb

    # Paper Fig 3 plots ANOMALIES from the 1979-2014 (training-period) mean:
    # subtract each series' own baseline-window mean so absolute offsets drop
    # out and the test-period trend/variability divergence is what shows.
    # Done before metrics so bias/RMSE are computed on the plotted quantity.
    if args.anomaly:
        b0, b1 = args.anomaly_base_start, args.anomaly_base_end
        if era5_annual is not None:
            era5_annual = _subtract_baseline(era5_annual, b0, b1)
        for _name in list(fleet):
            fleet[_name] = _subtract_baseline(fleet[_name], b0, b1)
        logger.info(f"Anomaly mode: removed {b0}-{b1} baseline mean per series")

    # Compute per-model mean bias + RMSE vs ERA5 reference.
    # ``bias`` = time-mean(ensmedian - era5).
    # ``rmse`` = sqrt(time-mean((ensmedian - era5)^2)).
    # Computed on the year-aligned intersection.
    metrics: dict[str, dict[str, float]] = {}
    era5_year_map = None
    if era5_annual is not None:
        era5_year_map = {
            int(y): float(v)
            for y, v in zip(np.asarray(era5_annual["year"]),
                            np.asarray(era5_annual.values))
        }
    for name, arr in fleet.items():
        if era5_year_map is None:
            continue
        years = np.asarray(arr["year"])
        members = np.asarray(arr.values)  # (n_member, n_year)
        ens_med = np.median(members, axis=0)
        diffs = []
        for y, m in zip(years, ens_med):
            if int(y) in era5_year_map:
                diffs.append(m - era5_year_map[int(y)])
        diffs = np.asarray(diffs)
        if diffs.size:
            metrics[name] = {
                "bias": float(diffs.mean()),
                "rmse": float(np.sqrt(np.mean(diffs ** 2))),
            }

    # Plot.
    fig, ax = plt.subplots(figsize=(11, 6.0), constrained_layout=True)
    rows = []
    for name, arr in fleet.items():
        years = np.asarray(arr["year"])
        members = np.asarray(arr.values)  # (n_member, n_year)
        median = np.median(members, axis=0)
        lo = members.min(axis=0)
        hi = members.max(axis=0)
        color = COLORS.get(name, "tab:gray")
        if name in metrics:
            label = (
                f"{name}  "
                f"(bias={metrics[name]['bias']:+.2f} K, "
                f"RMSE={metrics[name]['rmse']:.2f} K)"
            )
        else:
            label = name
        ax.fill_between(years, lo, hi, alpha=0.18, color=color, linewidth=0)
        ax.plot(years, median, color=color, linewidth=1.7, label=label)
        for y, m, l, h in zip(years, median, lo, hi):
            rows.append((name, int(y), float(l), float(m), float(h)))

    if era5_annual is not None:
        years = np.asarray(era5_annual["year"])
        vals = np.asarray(era5_annual.values)
        ax.plot(
            years, vals, color="k", linewidth=2.5, label="ERA5 (reference)",
            linestyle="--",
        )
        for y, v in zip(years, vals):
            rows.append(("ERA5", int(y), float("nan"), float(v), float("nan")))

    # legoESM (our trained variants, prescribed-SST AMIP inference) overlays.
    # Plotted as an anomaly from each model's OWN baseline-window mean: the
    # near-surface sigma-level T carries a constant offset vs ERA5 2-m tas,
    # so removing each series' own baseline leaves a like-for-like trend +
    # variability comparison (the honest way to place a free-of-absolute-bias
    # model on the fleet figure).  bias/RMSE vs ERA5 (on the plotted anomaly)
    # join the metrics CSV so our variants rank against the fleet.
    for lcsv in (args.legoesm_csvs or []):
        if not lcsv.exists():
            logger.warning(f"--legoesm-csv not found: {lcsv}")
            continue
        _variant = _legoesm_variant_from_stem(lcsv.stem)
        _d = np.genfromtxt(lcsv, delimiter=",", names=True)
        ly = np.atleast_1d(_d["year"]).astype(int)
        lt = np.atleast_1d(_d["annual_global_mean_surfT_K"]).astype(float)
        if not ly.size:
            continue
        if args.anomaly:
            _m = (ly >= args.anomaly_base_start) & (ly <= args.anomaly_base_end)
            _base = float(lt[_m].mean()) if _m.any() else float(lt.mean())
            lt = lt - _base
        _mname = f"legoESM-{_variant}"
        if era5_year_map is not None:
            # finite-only: one NaN year (e.g. a mid-chain partial CSV) must
            # not turn the whole bias/RMSE into NaN (codex LOW).
            _diffs = np.asarray([
                v - era5_year_map[int(y)]
                for y, v in zip(ly, lt)
                if int(y) in era5_year_map and np.isfinite(v)
            ])
            if _diffs.size:
                metrics[_mname] = {
                    "bias": float(_diffs.mean()),
                    "rmse": float(np.sqrt(np.mean(_diffs ** 2))),
                }
        _label = f"{_mname} (AMIP, prescribed ERA5 SST)"
        if _mname in metrics:
            _label += (
                f"  (bias={metrics[_mname]['bias']:+.2f} K, "
                f"RMSE={metrics[_mname]['rmse']:.2f} K)"
            )
        ax.plot(
            ly, lt, color=LEGOESM_VARIANT_COLORS.get(_variant, "#AA3377"),
            linewidth=2.5, marker="o", markersize=3, label=_label,
        )
        for y, v in zip(ly, lt):
            rows.append((_mname, int(y), float("nan"), float(v), float("nan")))
        logger.info(f"Overlaid {_mname}: {ly.size} years from {lcsv}")

    ax.set_xlabel("year")
    ax.set_ylabel(
        "global-mean tas anomaly [K]" if args.anomaly
        else "global-mean tas [K]"
    )
    _kind = (
        f"annual global-mean 2-m air temperature anomaly "
        f"(rel. {args.anomaly_base_start}-{args.anomaly_base_end})"
        if args.anomaly else
        "annual global-mean surface air temperature"
    )
    ax.set_title(
        f"AIMIP-1 fleet — {_kind}\n"
        f"(shaded = ensemble envelope, {YEAR_MIN}-{YEAR_MAX})"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {args.out}")

    # Dump CSV.
    with args.csv.open("w") as fh:
        fh.write("model,year,ens_lo,ens_med,ens_hi\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    logger.info(f"Wrote {args.csv}")

    # Dump metrics table.
    metrics_path = args.csv.with_name("aimip_fleet_tas_metrics.csv")
    with metrics_path.open("w") as fh:
        fh.write("model,bias_K,rmse_K\n")
        for name, m in metrics.items():
            fh.write(f"{name},{m['bias']:.4f},{m['rmse']:.4f}\n")
    logger.info(f"Wrote {metrics_path}")
    for name, m in metrics.items():
        logger.info(
            f"  {name:20}  bias = {m['bias']:+.3f} K   RMSE = {m['rmse']:.3f} K"
        )


if __name__ == "__main__":
    main()
