#!/usr/bin/env python
"""Run a semimonthly SFNO inference campaign and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legoesm.ml.s2s.paths import SFNO_SLAB_RESULTS_ROOT
from legoesm.ml.s2s.sfno_slab import (
    CHAOSBENCH_ATMOS_VARS,
    CHAOSBENCH_DATA_DIR,
    CHAOSBENCH_PRESSURE_LEVELS,
    ChaosBenchS2SConfig,
    DEFAULT_ARCO_SST_CACHE_PATH,
    DEFAULT_ARCO_SST_STATS_PATH,
    aggregate_long_records,
    available_s2s_dates,
    load_training_metadata,
    metadata_path_for_checkpoint,
)


WINDOWS = ("wk3_4", "wk5_6")
EXPERIMENTS = ("coupled", "uncoupled")
FIELD_STEMS = {
    "t-850": "t850",
    "z-500": "z500",
    "q-700": "q700",
    "sosstsst": "sst",
}
FIELD_TITLES = {
    "t-850": "T850",
    "z-500": "Z500",
    "q-700": "Q700",
    "sosstsst": "SST",
}
COLORS = {"coupled": "#1f77b4", "uncoupled": "#ff7f0e"}


def _metadata_get(metadata: dict[str, object], section: str, key: str, default=None):
    section_value = metadata.get(section, {})
    if isinstance(section_value, dict) and key in section_value:
        return section_value[key]
    return default


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _generate_semimonthly_start_times(
    year: int,
    *,
    months: tuple[int, ...],
    days: tuple[int, ...],
    hour: int = 0,
) -> list[str]:
    """Return ISO timestamps for the requested monthly initialization schedule."""
    start_times: list[str] = []
    for month in months:
        for day in days:
            start = datetime(int(year), int(month), int(day), int(hour), 0, 0)
            start_times.append(start.strftime("%Y-%m-%dT%H:%M:%S"))
    return start_times


def _write_rows(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _longify_case_metrics(case_dir: Path, init_time: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics_dir = case_dir / "metrics"
    daily_rows = _read_csv(metrics_dir / "daily_ensemble_metrics.csv")
    window_rows = _read_csv(metrics_dir / "window_ensemble_metrics.csv")

    daily_records: list[dict[str, object]] = []
    for row in daily_rows:
        for metric in ("rmse", "mae", "crps"):
            if metric not in row or row[metric] in ("", None):
                continue
            daily_records.append(
                {
                    "init_time": init_time,
                    "experiment": row["experiment"],
                    "field": row["field"],
                    "metric": metric,
                    "lead_day": int(row["lead_day"]),
                    "value": float(row[metric]),
                }
            )

    window_records: list[dict[str, object]] = []
    for row in window_rows:
        for metric in ("rmse", "mae", "crps"):
            if metric not in row or row[metric] in ("", None):
                continue
            window_records.append(
                {
                    "init_time": init_time,
                    "experiment": row["experiment"],
                    "field": row["field"],
                    "metric": metric,
                    "window": row["window"],
                    "value": float(row[metric]),
                }
            )

    return daily_records, window_records


def _save_campaign_metric_figures(case_window_records: list[dict[str, object]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for field, stem in FIELD_STEMS.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        for ax, metric in zip(axes, ("rmse", "crps"), strict=True):
            positions: list[float] = []
            data: list[np.ndarray] = []
            facecolors: list[str] = []
            for window_index, window in enumerate(WINDOWS):
                center = float(window_index)
                for experiment, offset in (("coupled", -0.18), ("uncoupled", 0.18)):
                    values = [
                        float(record["value"])
                        for record in case_window_records
                        if record["field"] == field
                        and record["metric"] == metric
                        and record["window"] == window
                        and record["experiment"] == experiment
                    ]
                    if not values:
                        continue
                    positions.append(center + offset)
                    data.append(np.asarray(values, dtype=float))
                    facecolors.append(COLORS[experiment])
            artists = ax.boxplot(
                data,
                positions=positions,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
            )
            for patch, color in zip(artists["boxes"], facecolors, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            for median in artists["medians"]:
                median.set_color("black")
            ax.set_xticks(range(len(WINDOWS)), labels=WINDOWS)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{FIELD_TITLES[field]} {metric.upper()}")
            ax.grid(True, alpha=0.3)
            handles = [
                plt.Line2D([0], [0], color=COLORS["coupled"], lw=8, label="coupled"),
                plt.Line2D([0], [0], color=COLORS["uncoupled"], lw=8, label="uncoupled"),
            ]
            ax.legend(handles=handles)
        fig.savefig(plots_dir / f"{stem}_campaign_metric_summary.png", dpi=150)
        plt.close(fig)


def _case_output_dir(base_output_dir: Path, date_tag: str) -> Path:
    return base_output_dir / f"inference_{date_tag}"


def _valid_date_tags(
    *,
    checkpoint: Path,
    year: int,
    requested_date_tags: list[str],
    n_steps: int,
    arco_sst_cache_path: str | None,
    arco_sst_stats_path: str | None,
) -> list[str]:
    """Filter requested init dates to dates with a full forecast horizon on disk."""
    metadata_path = metadata_path_for_checkpoint(checkpoint)
    metadata = load_training_metadata(metadata_path) if metadata_path.exists() else {}
    data_dir = str(_metadata_get(metadata, "data", "data_dir", CHAOSBENCH_DATA_DIR))
    atmosphere_vars = tuple(_metadata_get(metadata, "data", "atmosphere_vars", CHAOSBENCH_ATMOS_VARS))
    pressure_levels = tuple(_metadata_get(metadata, "data", "pressure_levels", CHAOSBENCH_PRESSURE_LEVELS))
    land_vars = tuple(_metadata_get(metadata, "data", "land_vars", ()))
    ocean_vars = tuple(_metadata_get(metadata, "data", "ocean_vars", ("sosstsst",)))
    ocean_source = str(_metadata_get(metadata, "data", "ocean_source", "oras5"))
    lead_time = int(_metadata_get(metadata, "data", "lead_time", 1))
    gaussian_n_max = int(_metadata_get(metadata, "data", "gaussian_n_max", 79))

    data_config = ChaosBenchS2SConfig(
        years=(int(year),),
        data_dir=data_dir,
        atmosphere_vars=tuple(str(value) for value in atmosphere_vars),
        pressure_levels=tuple(int(value) for value in pressure_levels),
        land_vars=tuple(str(value) for value in land_vars),
        ocean_vars=tuple(str(value) for value in ocean_vars),
        ocean_source=ocean_source,
        arco_sst_cache_path=arco_sst_cache_path or str(_metadata_get(metadata, "data", "arco_sst_cache_path", DEFAULT_ARCO_SST_CACHE_PATH)),
        arco_sst_stats_path=arco_sst_stats_path or str(_metadata_get(metadata, "data", "arco_sst_stats_path", DEFAULT_ARCO_SST_STATS_PATH)),
        n_steps=int(n_steps),
        lead_time=lead_time,
        gaussian_n_max=gaussian_n_max,
    )
    dates = available_s2s_dates(data_config)
    n_valid = len(dates) - data_config.lead_time - data_config.n_steps + 1
    valid_input_dates = set(dates[: max(n_valid, 0)])

    skipped = [date_tag for date_tag in requested_date_tags if date_tag not in valid_input_dates]
    if skipped:
        print(
            "skip invalid init dates:"
            f" {', '.join(skipped)}"
            f" for lead_time={data_config.lead_time}, n_steps={data_config.n_steps}",
            flush=True,
        )
    return [date_tag for date_tag in requested_date_tags if date_tag in valid_input_dates]


def _run_case(
    *,
    checkpoint: Path,
    date_tag: str,
    n_steps: int,
    n_members: int,
    base_output_dir: Path,
    fields: str,
    windows: str,
    plot_members: str,
    surface_source: str,
    era5_store: str,
    arco_sst_cache_path: str | None,
    arco_sst_stats_path: str | None,
    python_executable: str,
    skip_existing: bool,
) -> None:
    """Run one init date through the canonical ensemble-inference entrypoint."""
    case_dir = _case_output_dir(base_output_dir, date_tag)
    metrics_path = case_dir / "metrics" / "window_ensemble_metrics.csv"
    if skip_existing and metrics_path.exists():
        print(f"skip existing case: {date_tag}", flush=True)
        return

    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable,
        str(Path(__file__).resolve().parent / "sfno_slab.py"),
        "ensemble-inference",
        "--checkpoint",
        str(checkpoint),
        "--sample-date",
        date_tag,
        "--n-steps",
        str(n_steps),
        "--n-members",
        str(n_members),
        "--surface-source",
        surface_source,
        "--output-dir",
        str(case_dir),
        "--fields",
        fields,
        "--windows",
        windows,
        "--plot-members",
        plot_members,
    ]
    if surface_source == "arco":
        # Reuse cached surface forcing when present so reruns can skip the ARCO fetch.
        surface_path = case_dir / f"surface_forcing_{date_tag}_{n_steps}d.nc"
        if surface_path.exists():
            cmd.extend(["--surface-forcing-path", str(surface_path)])
        else:
            cmd.extend(["--surface-output-path", str(surface_path)])
        cmd.extend(["--era5-store", era5_store])
    if arco_sst_cache_path:
        cmd.extend(["--arco-sst-cache-path", arco_sst_cache_path])
    if arco_sst_stats_path:
        cmd.extend(["--arco-sst-stats-path", arco_sst_stats_path])

    print("run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a semimonthly SFNO inference campaign")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--months", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--days", default="1,15")
    parser.add_argument("--n-steps", type=int, default=42)
    parser.add_argument("--n-members", type=int, default=5)
    parser.add_argument("--fields", default="t-850,z-500,q-700,sosstsst")
    parser.add_argument("--windows", default="wk3_4=15:28,wk5_6=29:42")
    parser.add_argument("--plot-members", default="0,1")
    parser.add_argument("--surface-source", choices=("arco", "chaosbench"), default="arco")
    parser.add_argument("--era5-store", default="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")
    parser.add_argument("--arco-sst-cache-path", default=None)
    parser.add_argument("--arco-sst-stats-path", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args()

    start_times = _generate_semimonthly_start_times(
        args.year,
        months=_parse_csv_ints(args.months),
        days=_parse_csv_ints(args.days),
    )
    date_tags = [start_time[:10].replace("-", "") for start_time in start_times]
    date_tags = _valid_date_tags(
        checkpoint=args.checkpoint,
        year=int(args.year),
        requested_date_tags=date_tags,
        n_steps=int(args.n_steps),
        arco_sst_cache_path=args.arco_sst_cache_path,
        arco_sst_stats_path=args.arco_sst_stats_path,
    )

    if not args.aggregate_only:
        for date_tag in date_tags:
            _run_case(
                checkpoint=args.checkpoint,
                date_tag=date_tag,
                n_steps=int(args.n_steps),
                n_members=int(args.n_members),
                base_output_dir=args.output_dir,
                fields=args.fields,
                windows=args.windows,
                plot_members=args.plot_members,
                surface_source=args.surface_source,
                era5_store=args.era5_store,
                arco_sst_cache_path=args.arco_sst_cache_path,
                arco_sst_stats_path=args.arco_sst_stats_path,
                python_executable=args.python_executable,
                skip_existing=bool(args.skip_existing),
            )

    case_daily_records: list[dict[str, object]] = []
    case_window_records: list[dict[str, object]] = []
    for date_tag in date_tags:
        case_dir = _case_output_dir(args.output_dir, date_tag)
        daily_records, window_records = _longify_case_metrics(case_dir, init_time=date_tag)
        case_daily_records.extend(daily_records)
        case_window_records.extend(window_records)

    metrics_dir = args.output_dir / "metrics"
    plots_dir = args.output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _write_rows(
        metrics_dir / "case_ensemble_daily_metrics.csv",
        case_daily_records,
        ("init_time", "experiment", "field", "metric", "lead_day", "value"),
    )
    _write_rows(
        metrics_dir / "case_ensemble_window_metrics.csv",
        case_window_records,
        ("init_time", "experiment", "field", "metric", "window", "value"),
    )
    _write_rows(
        metrics_dir / "init_mean_ensemble_daily_metrics.csv",
        aggregate_long_records(
            case_daily_records,
            group_keys=("experiment", "field", "metric", "lead_day"),
        ),
        ("experiment", "field", "metric", "lead_day", "value"),
    )
    _write_rows(
        metrics_dir / "init_mean_ensemble_window_metrics.csv",
        aggregate_long_records(
            case_window_records,
            group_keys=("experiment", "field", "metric", "window"),
        ),
        ("experiment", "field", "metric", "window", "value"),
    )

    _save_campaign_metric_figures(case_window_records, plots_dir)


if __name__ == "__main__":
    main()
