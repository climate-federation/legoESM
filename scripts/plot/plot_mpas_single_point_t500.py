#!/usr/bin/env python
"""Plot the MPAS single-point T500 3DVar, 4DVar, and TLM responses."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/4DVar_single/experiment.yaml"),
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--tlm-input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output_directory = Path(config["output"]["directory"])
    input_path = args.input or output_directory / "t500_single_point_timeseries.npz"
    tlm_path = args.tlm_input or output_directory / "t500_tlm.npz"
    output_path = args.output or output_directory / "single_point_t500_response.png"
    with np.load(input_path, allow_pickle=False) as data:
        hours = data["hours"]
        latitude = data["latitude_cell_degrees"]
        longitude = data["longitude_cell_degrees"]
        background_z500 = data["z500_background"]
        three_dvar = data["t500_3dvar_increment"]
        four_dvar = data["t500_4dvar_increment"]
        observation_latitude = float(data["observation_latitude_degrees"])
        observation_longitude = float(data["observation_longitude_degrees"])
    with np.load(tlm_path, allow_pickle=False) as data:
        tangent_linear = data["t500_4dvar_tangent_increment"]

    plot_config = config["plot"]
    longitude = np.where(longitude > 180.0, longitude - 360.0, longitude)
    observation_longitude = (
        observation_longitude if observation_longitude <= 180.0 else observation_longitude - 360.0
    )
    extent = [
        *plot_config["longitude_extent_degrees"],
        *plot_config["latitude_extent_degrees"],
    ]
    minimum, maximum = plot_config["temperature_increment_limits_k"]
    interval = float(plot_config["temperature_increment_interval_k"])
    levels = np.arange(float(minimum), float(maximum) + 0.5 * interval, interval)
    z_interval = float(plot_config["geopotential_height_interval_m"])
    z_minimum = z_interval * np.floor(np.nanmin(background_z500) / z_interval)
    z_maximum = z_interval * np.ceil(np.nanmax(background_z500) / z_interval)
    z_levels = np.arange(z_minimum, z_maximum + 0.5 * z_interval, z_interval)
    triangulation = mtri.Triangulation(longitude, latitude)
    projection = ccrs.PlateCarree()
    rows = (
        ("3DVar", three_dvar),
        ("4DVar", four_dvar),
        ("TLM", tangent_linear),
    )
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "mathtext.default": "bf",
        }
    )
    figure = plt.figure(figsize=(16.0, 8.2), constrained_layout=False)
    grid = figure.add_gridspec(
        5,
        5,
        width_ratios=(0.18, 1, 1, 1, 1),
        height_ratios=(1, 1, 1, 0.10, 0.104),
        wspace=0.04,
        hspace=0.02,
        left=0.02,
        right=0.99,
        bottom=0.07,
        top=0.92,
    )
    axes = np.empty((3, 4), dtype=object)
    filled = None
    for row, (name, field) in enumerate(rows):
        label_axis = figure.add_subplot(grid[row, 0])
        label_axis.axis("off")
        label_axis.text(
            0.12,
            0.5,
            name,
            rotation=90,
            va="center",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
        for column, _ in enumerate(hours):
            axis = figure.add_subplot(grid[row, column + 1], projection=projection)
            axes[row, column] = axis
            filled = axis.tricontourf(
                triangulation,
                field[column],
                levels=levels,
                cmap="RdBu_r",
                extend="both",
                transform=projection,
            )
            contours = axis.tricontour(
                triangulation,
                background_z500[column],
                levels=z_levels,
                colors="black",
                linewidths=0.5,
                transform=projection,
            )
            axis.clabel(contours, z_levels[::2], fontsize=7, fmt="%d", inline=True)
            axis.coastlines(resolution="50m", linewidth=0.7)
            axis.add_feature(
                cfeature.BORDERS.with_scale("50m"),
                linewidth=0.45,
                edgecolor="black",
            )
            axis.plot(
                observation_longitude,
                observation_latitude,
                "+",
                color="black",
                ms=13,
                mew=2,
                transform=projection,
                zorder=10,
            )
            axis.set_extent(extent, crs=projection)
            labels = axis.gridlines(
                draw_labels=True,
                linewidth=0.3,
                color="0.4",
                alpha=0.4,
                linestyle=":",
            )
            labels.top_labels = False
            labels.right_labels = False
            labels.bottom_labels = row == 2
            labels.left_labels = column == 0
            labels.xlabel_style = labels.ylabel_style = {"size": 11}

    time_labels = (
        r"$t_0$",
        r"$t_0+02\mathbf{h}$",
        r"$t_0+04\mathbf{h}$",
        r"$t_0+06\mathbf{h}$",
    )
    for column, label in enumerate(time_labels):
        bounds = axes[0, column].get_position()
        figure.text(
            0.5 * (bounds.x0 + bounds.x1),
            0.94,
            label,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
    colorbar_grid = grid[4, 1:].subgridspec(
        1,
        3,
        width_ratios=(1, 8, 1),
        wspace=0.0,
    )
    colorbar_axis = figure.add_subplot(colorbar_grid[0, 1])
    colorbar = figure.colorbar(
        filled,
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=levels,
        format="%.1f",
    )
    colorbar.set_label("500 hPa temperature increment (K)", fontsize=12, labelpad=7)
    colorbar.ax.tick_params(labelsize=11, length=4, width=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=int(plot_config["dpi"]),
        facecolor="white",
    )
    plt.close(figure)
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
