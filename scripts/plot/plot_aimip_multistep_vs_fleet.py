#!/usr/bin/env python
"""Plot annual-mean RMSE and bias for legoESM multi-step variants vs the AIMIP fleet.

Reads our :file:`aimip_scorecard.json` produced by
``scripts/run/run_aimip.py`` for the three legoESM variants (classical,
column_nn, sfno_physics) and overlays them against the AIMIP-1 fleet
(ACE2.1, ArchesWeather, NeuralGCM, NeuralGCM-HRD) restricted to the
training (2015-2016) and test (2017) calendar periods.

Two quantities differ in their physical content and that is called
out in the figure title:

- AIMIP fleet: ``tas`` (near-surface 2-m air temperature) computed
  from the official 46-year AMIP submissions and time-averaged over
  each subperiod, with RMSE/bias against ERA5 2 m temperature.
- legoESM variants: mid-level ``T`` (≈ 500 hPa) from the AIMIP
  scorecard's per-IC 24-h forecast skill, area-weighted RMSE and bias
  averaged over ICs in the corresponding window.

These are not directly comparable in absolute K, but the rank ordering
across our variants is meaningful within each subperiod and the AIMIP
fleet provides a reference scale.  We plot them side-by-side with
explicit grouping in the title.

Usage::

    .venv/bin/python scripts/plot_aimip_multistep_vs_fleet.py \\
        --scorecard results/aimip_multistep_001/aimip_scorecard.json \\
        --out-prefix results/aimip_multistep_001/aimip_multistep_vs_fleet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import fsspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


logger = logging.getLogger("aimip-multistep-plot")

S3_ENDPOINT = "https://s3.eu-dkrz-1.dkrz.cloud"
S3_BUCKET = "ai-mip"

MODELS = {
    "ACE2.1-ERA5": dict(
        path="Ai2/ACE2-1-ERA5",
        template="aimip/r{i_r}i1p1f1/Amon/tas/gr/v20251130/"
                 "tas_Amon_ACE2-ERA5_aimip_r{i_r}i1p1f1_gr_197810-202412.nc",
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

ERA5_PATH = "ERA5/mon/ERA5_2m_temperature_mon_full_sfc_1940-2024.nc"
ERA5_VARNAME = "T2M"

# Subperiods.
TRAIN_YEARS = (2015, 2016)
TEST_YEARS = (2017,)

N_ENS = 5

COLORS = {
    "ACE2.1-ERA5": "#E69F00",
    "ArchesWeather": "#56B4E9",
    "NeuralGCM": "#009E73",
    "NeuralGCM-HRD": "#0072B2",
    "legoESM/classical": "#882255",
    "legoESM/column_nn": "#CC79A7",
    "legoESM/sfno_physics": "#332288",
}


def _open_remote(fs, key: str) -> xr.Dataset:
    full = f"{S3_BUCKET}/{key}"
    with fs.open(full, "rb") as fh:
        return xr.open_dataset(fh, engine="h5netcdf").load()


def _global_annual(ds: xr.Dataset, varname: str = "tas") -> xr.DataArray:
    """Latitude-weighted global mean per calendar year."""
    da = ds[varname]
    lat_name = "lat" if "lat" in da.dims else next(d for d in da.dims if "lat" in d.lower())
    lon_name = "lon" if "lon" in da.dims else next(d for d in da.dims if "lon" in d.lower())
    weights = np.cos(np.deg2rad(da[lat_name]))
    weights.name = "wgt"
    monthly = da.weighted(weights).mean(dim=(lat_name, lon_name))
    annual = monthly.groupby("time.year").mean()
    return annual


def _slice_years(da: xr.DataArray, years: tuple[int, ...]) -> np.ndarray:
    y = np.asarray(da["year"])
    mask = np.isin(y, np.asarray(years))
    return np.asarray(da.values)[..., mask]


def _load_model(fs, name: str, info: dict) -> xr.DataArray | None:
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


def _fleet_metric(
    fleet: dict[str, xr.DataArray],
    era5_annual: xr.DataArray,
    years: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """Per-model (ensemble-median minus ERA5) annual-mean RMSE + bias on ``years``."""
    out: dict[str, dict[str, float]] = {}
    era5_y = np.asarray(era5_annual["year"])
    era5_v = np.asarray(era5_annual.values)
    era5_map = {int(y): float(v) for y, v in zip(era5_y, era5_v)}
    target = np.asarray(
        [era5_map.get(int(y), np.nan) for y in years], dtype=np.float64,
    )
    for name, arr in fleet.items():
        ys = np.asarray(arr["year"])
        # ensemble-median over members axis 0
        med = np.median(np.asarray(arr.values), axis=0)
        # align to target years
        y_to_val = {int(y): float(v) for y, v in zip(ys, med)}
        pred = np.asarray(
            [y_to_val.get(int(y), np.nan) for y in years], dtype=np.float64,
        )
        diff = pred - target
        diff = diff[~np.isnan(diff)]
        if diff.size == 0:
            out[name] = {"bias": float("nan"), "rmse": float("nan")}
            continue
        out[name] = {
            "bias": float(np.mean(diff)),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
        }
    return out


def _legoesm_metrics(scorecard_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Extract per-variant near-surface T RMSE + bias for both periods.

    Returns ``{variant_name: {period: {"rmse": ..., "bias": ...}}}``
    where ``period`` is ``"train"`` (in-sample 2015-2016) or
    ``"test"`` (held-out 2017).  Both reports come from
    :func:`run_aimip._evaluate_variant`, which writes
    ``eval_metrics`` for the test window and
    ``eval_metrics_train_period`` for the train window.
    """
    if not scorecard_path.exists():
        logger.warning(f"No scorecard at {scorecard_path}")
        return {}
    with scorecard_path.open() as fh:
        data = json.load(fh)
    variants = data.get("variants", {})
    out: dict[str, dict[str, dict[str, float]]] = {}

    def _pull(em_block: dict, key: str) -> dict[str, float]:
        # ``T_sfc`` is the lowest-sigma level T (closest to AIMIP-fleet
        # ``tas``).  Returns NaN when the block is missing or T_sfc is
        # absent (older scorecards).
        rmse = float(em_block.get("rmse", {}).get(key, {}).get("mean", float("nan")))
        bias = float(em_block.get("bias", {}).get(key, {}).get("mean", float("nan")))
        return {"rmse": rmse, "bias": bias}

    for vname, vdata in variants.items():
        em_test = vdata.get("eval_metrics", {}) or {}
        em_train = vdata.get("eval_metrics_train_period", {}) or {}
        out[f"legoESM/{vname}"] = {
            "test": _pull(em_test, "T_sfc"),
            "train": _pull(em_train, "T_sfc"),
        }
    return out


def _make_bar_panels(
    fleet_metrics: dict[str, dict[str, float]],
    lego_metrics: dict[str, dict[str, dict[str, float]]],
    lego_period: str,
    years_label: str,
    out_path: Path,
    *,
    is_train: bool,
):
    """Two panels: RMSE (left) and bias (right) bars.

    Parameters
    ----------
    fleet_metrics : per-model {"rmse": K, "bias": K} for this period.
    lego_metrics  : per-variant {period: {"rmse": K, "bias": K}}; we
                    select the ``lego_period`` block.
    lego_period   : "train" or "test" — which legoESM eval block to
                    plot alongside the fleet bars.
    """
    fleet_names = [n for n in fleet_metrics if np.isfinite(fleet_metrics[n]["rmse"])]
    lego_names = [
        n for n in lego_metrics
        if np.isfinite(lego_metrics[n].get(lego_period, {}).get("rmse", float("nan")))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    for ax, key, ylabel, title in zip(
        axes,
        ["rmse", "bias"],
        ["RMSE vs ERA5 [K]", "Bias vs ERA5 [K]"],
        [
            f"Annual-mean RMSE ({years_label})",
            f"Annual-mean bias ({years_label})",
        ],
    ):
        all_names = fleet_names + lego_names
        x = np.arange(len(all_names))
        for i, name in enumerate(fleet_names):
            ax.bar(
                x[i], fleet_metrics[name][key],
                color=COLORS.get(name, "tab:gray"), edgecolor="black",
            )
        for j, name in enumerate(lego_names):
            val = lego_metrics[name][lego_period][key]
            ax.bar(
                x[len(fleet_names) + j], val,
                color=COLORS.get(name, "tab:gray"),
                edgecolor="black", hatch="//",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if key == "bias":
            ax.axhline(0, color="k", linewidth=0.5)
    if is_train:
        suptitle = (
            "AIMIP-fleet (46-yr AMIP annual-mean tas)  vs  legoESM "
            "multi-step CRPS (24-h forecast T_sfc skill on training ICs)\n"
            f"Training period {years_label}"
        )
    else:
        suptitle = (
            "AIMIP-fleet (46-yr AMIP annual-mean tas)  vs  legoESM "
            "multi-step CRPS (24-h forecast T_sfc skill on held-out ICs)\n"
            f"Held-out test period {years_label}"
        )
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scorecard", type=Path,
        default=Path("results/aimip_multistep_001/aimip_scorecard.json"),
    )
    parser.add_argument(
        "--out-prefix", type=Path,
        default=Path("results/aimip_multistep_001/aimip_multistep_vs_fleet"),
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("results/aimip_multistep_001/aimip_multistep_vs_fleet_metrics.csv"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load legoESM scorecard first (cheap, no network).
    lego_metrics = _legoesm_metrics(args.scorecard)
    logger.info(f"legoESM scorecard: {len(lego_metrics)} variants")
    for k, v in lego_metrics.items():
        logger.info(f"  {k}: {v}")

    # AIMIP fleet from S3.
    fs = fsspec.filesystem(
        "s3", anon=True, client_kwargs={"endpoint_url": S3_ENDPOINT},
    )
    logger.info("Loading ERA5 reference...")
    era5_annual = _load_era5(fs)
    if era5_annual is None:
        logger.error("ERA5 reference failed to load; cannot compute fleet metrics.")
        return

    fleet: dict[str, xr.DataArray] = {}
    for name, info in MODELS.items():
        logger.info(f"Loading {name}...")
        arr = _load_model(fs, name, info)
        if arr is not None:
            fleet[name] = arr

    # Per-period metrics.
    train_metrics = _fleet_metric(fleet, era5_annual, TRAIN_YEARS)
    test_metrics = _fleet_metric(fleet, era5_annual, TEST_YEARS)

    _make_bar_panels(
        train_metrics, lego_metrics, "train",
        f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        args.out_prefix.with_name(args.out_prefix.name + "_train.png"),
        is_train=True,
    )
    _make_bar_panels(
        test_metrics, lego_metrics, "test",
        f"{TEST_YEARS[0]}",
        args.out_prefix.with_name(args.out_prefix.name + "_test.png"),
        is_train=False,
    )

    # Dump CSV.
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w") as fh:
        fh.write("source,model,period,metric,value_K\n")
        for name, m in train_metrics.items():
            for k, v in m.items():
                fh.write(f"aimip-fleet,{name},train,{k},{v}\n")
        for name, m in test_metrics.items():
            for k, v in m.items():
                fh.write(f"aimip-fleet,{name},test,{k},{v}\n")
        for name, periods in lego_metrics.items():
            for period_name, m in periods.items():
                for k, v in m.items():
                    fh.write(f"legoESM,{name},{period_name},{k},{v}\n")
    logger.info(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
