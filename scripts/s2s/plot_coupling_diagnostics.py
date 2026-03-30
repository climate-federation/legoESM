#!/usr/bin/env python
"""Create proof-of-coupling plots from forecast datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from legoesm import constants
from legoesm.ml.s2s.neuralgcm_slab.metrics import DEFAULT_FIELD_SPECS, build_daily_metric_table


def _open_dataset(path: Path) -> xr.Dataset:
    if path.suffix == ".zarr":
        return xr.open_zarr(path)
    return xr.open_dataset(path)


def _global_mean(field: xr.DataArray) -> np.ndarray:
    dims = [dim for dim in field.dims if dim != "lead_day"]
    return np.asarray(field.mean(dim=dims))


def _last_truth_field(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    field = dataset[variable]
    if "time" in field.dims:
        return field.isel(time=-1)
    return field.isel(lead_day=-1)


def _plot_map_panel(
    ax: plt.Axes,
    field: xr.DataArray,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plot_field = field.transpose("latitude", "longitude")
    plot_field.plot(
        ax=ax,
        x="longitude",
        y="latitude",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _save_sst_actual_plot(
    coupled: xr.Dataset,
    uncoupled: xr.Dataset,
    *,
    out: Path,
    last_lead: int,
) -> None:
    coupled_sst = coupled["sea_surface_temperature"].isel(lead_day=-1)
    uncoupled_sst = uncoupled["sea_surface_temperature"].isel(lead_day=-1)
    vmin = float(np.nanmin([float(coupled_sst.min()), float(uncoupled_sst.min())]))
    vmax = float(np.nanmax([float(coupled_sst.max()), float(uncoupled_sst.max())]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
    _plot_map_panel(axes[0], coupled_sst, title=f"Coupled SST, lead day {last_lead}", vmin=vmin, vmax=vmax)
    _plot_map_panel(axes[1], uncoupled_sst, title=f"Uncoupled SST, lead day {last_lead}", vmin=vmin, vmax=vmax)
    mappable = axes[1].collections[0]
    fig.colorbar(mappable, ax=axes, shrink=0.9, label="K")
    fig.savefig(out / "sst_actual_last_lead.png", dpi=150)
    plt.close(fig)


def _save_three_panel_plot(
    truth_field: xr.DataArray,
    coupled_field: xr.DataArray,
    uncoupled_field: xr.DataArray,
    *,
    out: Path,
    stem: str,
    title_prefix: str,
    colorbar_label: str,
    cmap: str = "viridis",
) -> None:
    vmin = float(
        np.nanmin([float(truth_field.min()), float(coupled_field.min()), float(uncoupled_field.min())])
    )
    vmax = float(
        np.nanmax([float(truth_field.max()), float(coupled_field.max()), float(uncoupled_field.max())])
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
    _plot_map_panel(axes[0], truth_field, title=f"Truth {title_prefix}", cmap=cmap, vmin=vmin, vmax=vmax)
    _plot_map_panel(axes[1], coupled_field, title=f"Coupled {title_prefix}", cmap=cmap, vmin=vmin, vmax=vmax)
    _plot_map_panel(axes[2], uncoupled_field, title=f"Uncoupled {title_prefix}", cmap=cmap, vmin=vmin, vmax=vmax)
    mappable = axes[2].collections[0]
    fig.colorbar(mappable, ax=axes, shrink=0.9, label=colorbar_label)
    fig.savefig(out / f"{stem}_actual_last_lead.png", dpi=150)
    plt.close(fig)


def _save_rmse_plots(
    coupled: xr.Dataset,
    uncoupled: xr.Dataset | None,
    truth: xr.Dataset,
    *,
    out: Path,
) -> None:
    coupled_rmse = build_daily_metric_table(coupled, truth, field_specs=DEFAULT_FIELD_SPECS)
    uncoupled_rmse = (
        build_daily_metric_table(uncoupled, truth, field_specs=DEFAULT_FIELD_SPECS)
        if uncoupled is not None
        else None
    )
    lead = np.asarray(coupled["lead_day"], dtype=float)

    label_map = {
        "rmse.temperature_850": "T850",
        "rmse.geopotential_500": "Z500",
        "rmse.specific_humidity_700": "Q700",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes_flat, label_map, strict=True):
        ax.plot(lead, coupled_rmse[key], marker="o", label="coupled")
        if uncoupled_rmse is not None:
            ax.plot(lead, uncoupled_rmse[key], marker="o", label="uncoupled")
        ax.set_title(f"{label_map[key]} RMSE")
        ax.set_xlabel("Lead day")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(out / "rmse_timeseries.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot coupling diagnostics from forecast datasets")
    parser.add_argument("--coupled", type=Path, required=True, help="Coupled forecast dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots")
    parser.add_argument("--uncoupled", type=Path, default=None, help="Optional uncoupled forecast dataset")
    parser.add_argument("--truth", type=Path, default=None, help="Optional truth dataset on the same grid")
    args = parser.parse_args()

    coupled = _open_dataset(args.coupled)
    uncoupled = _open_dataset(args.uncoupled) if args.uncoupled is not None else None
    truth = _open_dataset(args.truth) if args.truth is not None else None
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    lead = np.asarray(coupled["lead_day"])
    last_lead = int(lead[-1]) if len(lead) else 0

    plt.figure(figsize=(8, 4))
    plt.plot(lead, _global_mean(coupled["sea_surface_temperature"]), label="coupled")
    if uncoupled is not None:
        plt.plot(lead, _global_mean(uncoupled["sea_surface_temperature"]), label="uncoupled")
    if truth is not None and "sea_surface_temperature" in truth:
        truth_sst = truth["sea_surface_temperature"]
        if "time" in truth_sst.dims:
            truth_sst = truth_sst.isel(time=slice(0, len(lead))).rename({"time": "lead_day"})
            truth_sst = truth_sst.assign_coords({"lead_day": lead})
        plt.plot(lead, _global_mean(truth_sst), label="truth")
    plt.xlabel("Lead day")
    plt.ylabel("Global mean SST [K]")
    plt.title("Global mean SST evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "global_mean_sst.png", dpi=150)
    plt.close()

    if uncoupled is not None:
        _save_sst_actual_plot(coupled, uncoupled, out=out, last_lead=last_lead)

    if truth is not None:
        _save_rmse_plots(coupled, uncoupled, truth, out=out)

        if uncoupled is not None:
            t850_true = _last_truth_field(truth, "temperature").sel(level=850, method="nearest")
            t850_coupled = coupled["temperature"].sel(level=850, method="nearest").isel(lead_day=-1)
            t850_uncoupled = uncoupled["temperature"].sel(level=850, method="nearest").isel(lead_day=-1)
            _save_three_panel_plot(
                t850_true,
                t850_coupled,
                t850_uncoupled,
                out=out,
                stem="t850",
                title_prefix=f"T850, lead day {last_lead}",
                colorbar_label="K",
            )

            z500_true = _last_truth_field(truth, "geopotential").sel(level=500, method="nearest") / constants.g
            z500_coupled = coupled["geopotential"].sel(level=500, method="nearest").isel(lead_day=-1) / constants.g
            z500_uncoupled = (
                uncoupled["geopotential"].sel(level=500, method="nearest").isel(lead_day=-1) / constants.g
            )
            _save_three_panel_plot(
                z500_true,
                z500_coupled,
                z500_uncoupled,
                out=out,
                stem="z500",
                title_prefix=f"Z500, lead day {last_lead}",
                colorbar_label="m",
            )

            q700_true = _last_truth_field(truth, "specific_humidity").sel(level=700, method="nearest")
            q700_coupled = coupled["specific_humidity"].sel(level=700, method="nearest").isel(lead_day=-1)
            q700_uncoupled = uncoupled["specific_humidity"].sel(level=700, method="nearest").isel(lead_day=-1)
            _save_three_panel_plot(
                q700_true,
                q700_coupled,
                q700_uncoupled,
                out=out,
                stem="q700",
                title_prefix=f"Q700, lead day {last_lead}",
                colorbar_label="kg kg-1",
            )


if __name__ == "__main__":
    main()
