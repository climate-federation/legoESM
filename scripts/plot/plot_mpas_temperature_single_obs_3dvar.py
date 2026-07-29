#!/usr/bin/env python
"""Plot and validate the NMC-B Jung temperature single-observation response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import griddata

EARTH_RADIUS_M = 6_371_000.0


def _increment_colormap():
    """Discrete blue-white-yellow-red palette matching Jung et al. Fig. 8."""
    from matplotlib.colors import ListedColormap

    colors = (
        "#17365d",
        "#225a9a",
        "#2f78b7",
        "#4a9acb",
        "#72b9dc",
        "#a0d5e9",
        "#c9e9f2",
        "#eaf6f8",
        "#ffffff",
        "#ffffff",
        "#fff3c4",
        "#fed976",
        "#feb24c",
        "#fd8d3c",
        "#f03b20",
        "#d7191c",
        "#b10f1a",
        "#7f0000",
    )
    return ListedColormap(colors, name="jung_increment")


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/3DVar_single/experiment.yaml"),
    )
    known, _ = config_parser.parse_known_args()
    with known.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    output_directory = Path(config["output"]["directory"])
    plot = config["plot"]

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument(
        "--input",
        type=Path,
        default=output_directory / "jung_nmc_temperature_single_obs_3dvar.npz",
    )
    parser.add_argument("--summary", type=Path, default=output_directory / "summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=output_directory / "temperature_single_obs_3dvar_increments_3x3.png",
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=output_directory / "validation.json",
    )
    parser.add_argument("--lon-min", type=float, default=float(plot["longitude_extent_degrees"][0]))
    parser.add_argument("--lon-max", type=float, default=float(plot["longitude_extent_degrees"][1]))
    parser.add_argument("--lat-min", type=float, default=float(plot["latitude_extent_degrees"][0]))
    parser.add_argument("--lat-max", type=float, default=float(plot["latitude_extent_degrees"][1]))
    parser.add_argument("--dpi", type=int, default=int(plot["dpi"]))
    return parser.parse_args()


def _interpolate(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = 6.0
    selected = (
        (lon >= args.lon_min - margin)
        & (lon <= args.lon_max + margin)
        & (lat >= args.lat_min - margin)
        & (lat <= args.lat_max + margin)
        & np.isfinite(values)
    )
    x = np.linspace(args.lon_min, args.lon_max, 361)
    y = np.linspace(args.lat_min, args.lat_max, 301)
    xx, yy = np.meshgrid(x, y)
    zz = griddata((lon[selected], lat[selected]), values[selected], (xx, yy), method="linear")
    return xx, yy, zz


def _regional_limit(
    values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    args: argparse.Namespace,
) -> float:
    selected = (
        (lon >= args.lon_min)
        & (lon <= args.lon_max)
        & (lat >= args.lat_min)
        & (lat <= args.lat_max)
        & np.isfinite(values)
    )
    maximum = float(np.max(np.abs(values[selected])))
    if maximum <= 0.0:
        return 1.0
    exponent = np.floor(np.log10(maximum))
    scale = 10.0**exponent
    return float(np.ceil(maximum / scale * 2.0) / 2.0 * scale)


def _local_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    obs_lon: float,
    obs_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0 = np.deg2rad(obs_lon)
    lat0 = np.deg2rad(obs_lat)
    dlon = (lon_rad - lon0 + np.pi) % (2.0 * np.pi) - np.pi
    return EARTH_RADIUS_M * np.cos(lat0) * dlon, EARTH_RADIUS_M * (lat_rad - lat0)


def _clockwise_tangential_mean(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    radius = np.sqrt(x * x + y * y)
    selected = (radius >= 300_000.0) & (radius <= 700_000.0)
    clockwise_u = y[selected] / radius[selected]
    clockwise_v = -x[selected] / radius[selected]
    return float(np.mean(u[selected] * clockwise_u + v[selected] * clockwise_v))


def main() -> None:
    import cartopy.crs as ccrs
    from matplotlib.ticker import FuncFormatter

    args = _parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        lon = np.asarray(data["longitude_cell_degrees"], dtype=np.float64)
        lat = np.asarray(data["latitude_cell_degrees"], dtype=np.float64)
        fields = {
            "u": np.asarray(data["zonal_wind_increment_m_s"], dtype=np.float64),
            "v": np.asarray(data["meridional_wind_increment_m_s"], dtype=np.float64),
            "T": np.asarray(data["temperature_increment_k"], dtype=np.float64),
        }
        pressures = np.asarray(data["diagnostic_model_level_pressures_hpa"], dtype=np.float64)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    obs_lon = float(summary["observation"]["cell_longitude_degrees"])
    obs_lat = float(summary["observation"]["cell_latitude_degrees"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "mathtext.fontset": "dejavusans",
        }
    )
    projection = ccrs.PlateCarree()
    increment_cmap = _increment_colormap()
    fig = plt.figure(figsize=(10.8, 10.8), facecolor="white")
    grid = fig.add_gridspec(
        6,
        3,
        left=0.075,
        right=0.985,
        top=0.975,
        bottom=0.055,
        wspace=0.15,
        hspace=0.34,
        height_ratios=(1.0, 0.065, 1.0, 0.065, 1.0, 0.065),
    )
    columns = (
        ("u", r"$\delta u$", r"m s$^{-1}$"),
        ("v", r"$\delta v$", r"m s$^{-1}$"),
        ("T", r"$\delta T$", "K"),
    )
    diagnostics: dict[str, object] = {
        "source": str(args.input),
        "observation_cell_lon_lat_deg": [obs_lon, obs_lat],
        "pressures_hpa": pressures.tolist(),
        "panels": {},
    }

    for row, pressure in enumerate(pressures):
        for column, (name, symbol, unit) in enumerate(columns):
            ax = fig.add_subplot(grid[2 * row, column], projection=projection)
            cax = fig.add_subplot(grid[2 * row + 1, column])
            values = fields[name][row]
            limit = _regional_limit(values, lon, lat, args)
            xx, yy, zz = _interpolate(lon, lat, values, args)
            contour = ax.contourf(
                xx,
                yy,
                zz,
                levels=np.linspace(-limit, limit, increment_cmap.N + 1),
                cmap=increment_cmap,
                extend="both",
                transform=projection,
            )
            ax.set_extent([args.lon_min, args.lon_max, args.lat_min, args.lat_max], crs=projection)
            ax.coastlines(resolution="110m", linewidth=0.6, color="0.15")
            gridlines = ax.gridlines(
                crs=projection,
                draw_labels=True,
                xlocs=np.arange(-60, 1, 20),
                ylocs=np.arange(20, 61, 20),
                linewidth=0.3,
                color="0.55",
                alpha=0.55,
                linestyle=":",
                x_inline=False,
                y_inline=False,
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.left_labels = column == 0
            gridlines.bottom_labels = row == 2
            gridlines.xlabel_style = {"size": 10}
            gridlines.ylabel_style = {"size": 10}
            regional = (
                (lon >= args.lon_min)
                & (lon <= args.lon_max)
                & (lat >= args.lat_min)
                & (lat <= args.lat_max)
            )
            minimum = float(np.min(values[regional]))
            maximum = float(np.max(values[regional]))
            ax.text(
                0.015,
                0.975,
                f"{symbol} at {pressure:.0f} hPa\nmin/max = {minimum:.3f} / {maximum:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
                zorder=12,
            )
            if row == 1:
                ax.plot(
                    obs_lon,
                    obs_lat,
                    marker="x",
                    markersize=8,
                    markeredgewidth=2.0,
                    color="black",
                    transform=projection,
                    zorder=15,
                )
            colorbar = fig.colorbar(
                contour,
                cax=cax,
                orientation="horizontal",
                ticks=np.linspace(-limit, limit, 5),
            )
            colorbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            colorbar.ax.tick_params(labelsize=9, length=2.5, pad=1.5)
            colorbar.set_label(f"[{unit}]", fontsize=10, labelpad=1.5)
            diagnostics["panels"][f"{name}_{pressure:.1f}hPa"] = {
                "min": minimum,
                "max": maximum,
                "color_limit": limit,
            }

    x, y = _local_xy(lon, lat, obs_lon, obs_lat)
    clockwise = [
        _clockwise_tangential_mean(fields["u"][row], fields["v"][row], x, y) for row in range(3)
    ]
    diagnostics["thermal_wind_validation"] = {
        "clockwise_tangential_mean_ms_300_700km": clockwise,
        "sign_convention": "positive is clockwise and anticyclonic in the Northern Hemisphere",
        "upper_anticyclonic": bool(clockwise[0] > 0.0),
        "lower_cyclonic": bool(clockwise[2] < 0.0),
        "observation_level_weaker_than_upper_and_lower": bool(
            abs(clockwise[1]) < abs(clockwise[0]) and abs(clockwise[1]) < abs(clockwise[2])
        ),
    }
    diagnostics["temperature_max_by_row_K"] = [float(np.max(row)) for row in fields["T"]]
    diagnostics["observation_level_temperature_is_largest"] = bool(
        diagnostics["temperature_max_by_row_K"][1] == max(diagnostics["temperature_max_by_row_K"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    args.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
