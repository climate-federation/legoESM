#!/usr/bin/env python
"""Run a Jung et al. single-temperature-observation 3DVar test with NMC B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import griddata

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import HydrostaticState  # noqa: E402
from legoesm.da.control_vector import (  # noqa: E402
    build_control_spec,
    control_to_increment,
)
from legoesm.da.gen_be import GenBETransform, load_gen_be_params  # noqa: E402
from legoesm.grids.factory import create_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402


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
    inputs = config["inputs"]
    grid = config["grid"]
    observation = config["observation"]
    background_error = config["background_error"]
    diagnostics = config["diagnostics"]
    plot = config["plot"]

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--sample", type=Path, default=Path(inputs["nmc_sample"]))
    parser.add_argument("--gen-be", type=Path, default=Path(inputs["gen_be_path"]))
    parser.add_argument("--output-dir", type=Path, default=Path(config["output"]["directory"]))
    parser.add_argument("--resolution", type=int, default=int(grid["resolution"]))
    parser.add_argument("--nlev", type=int, default=int(grid["levels"]))
    parser.add_argument("--lloyd-iterations", type=int, default=int(grid["lloyd_iterations"]))
    parser.add_argument(
        "--background",
        choices=("f24", "f48"),
        default=str(config["background"]["sample_state"]),
    )
    parser.add_argument(
        "--obs-lat", type=float, default=float(observation["latitude_degrees_north"])
    )
    parser.add_argument(
        "--obs-lon", type=float, default=float(observation["longitude_degrees_east"])
    )
    parser.add_argument(
        "--obs-pressure-hpa", type=float, default=float(observation["pressure_hpa"])
    )
    parser.add_argument("--innovation-k", type=float, default=float(observation["increment_k"]))
    parser.add_argument(
        "--obs-error-k", type=float, default=float(observation["error_standard_deviation_k"])
    )
    parser.add_argument(
        "--plot-pressure-levels-hpa",
        default=",".join(str(value) for value in diagnostics["pressure_levels_hpa"]),
    )
    parser.add_argument(
        "--n-diffusion-iter", type=int, default=int(background_error["diffusion_iterations"])
    )
    parser.add_argument(
        "--len-scale-multiplier",
        type=float,
        default=float(background_error["horizontal_length_scale_multiplier"]),
    )
    parser.add_argument(
        "--background-error-std-multiplier",
        type=float,
        default=float(background_error["standard_deviation_multiplier"]),
    )
    parser.add_argument("--lon-min", type=float, default=float(plot["longitude_extent_degrees"][0]))
    parser.add_argument("--lon-max", type=float, default=float(plot["longitude_extent_degrees"][1]))
    parser.add_argument("--lat-min", type=float, default=float(plot["latitude_extent_degrees"][0]))
    parser.add_argument("--lat-max", type=float, default=float(plot["latitude_extent_degrees"][1]))
    parser.add_argument("--dpi", type=int, default=int(plot["dpi"]))
    return parser.parse_args()


def _state_from_sample(path: Path, background: str) -> HydrostaticState:
    prefix = "f24" if background == "f24" else "f48"
    with np.load(path, allow_pickle=False) as data:
        return HydrostaticState(
            u=Field(
                jnp.asarray(data[f"{prefix}_u"]), name="u", dims=("cell", "level"), units="m/s"
            ),
            v=Field(
                jnp.asarray(data[f"{prefix}_v"]), name="v", dims=("cell", "level"), units="m/s"
            ),
            T=Field(jnp.asarray(data[f"{prefix}_T"]), name="T", dims=("cell", "level"), units="K"),
            p_s=Field(jnp.asarray(data[f"{prefix}_p_s"]), name="p_s", dims=("cell",), units="Pa"),
            phis=Field(
                jnp.asarray(data[f"{prefix}_phis"]), name="phis", dims=("cell",), units="m2/s2"
            ),
            tracers={
                "q_v": Field(
                    jnp.asarray(data[f"{prefix}_tracer_q_v"]),
                    name="q_v",
                    dims=("cell", "level"),
                    units="kg/kg",
                )
            },
        )


def _nearest_cell(mesh, lat_deg: float, lon_deg: float) -> int:
    lat0 = np.deg2rad(lat_deg)
    lon0 = np.deg2rad(lon_deg % 360.0)
    lat = np.asarray(mesh.latCell)
    lon = np.asarray(mesh.lonCell)
    cos_distance = np.sin(lat) * np.sin(lat0) + np.cos(lat) * np.cos(lat0) * np.cos(lon - lon0)
    return int(np.argmax(cos_distance))


def _nearest_model_levels(
    background: HydrostaticState,
    sigma,
    observation_cell: int,
    target_pressures_hpa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pressure_profile_hpa = (
        np.asarray(jax.device_get(sigma.pressure_at_full(background.p_s.data)[observation_cell]))
        / 100.0
    )
    levels = np.argmin(
        np.abs(pressure_profile_hpa[None, :] - target_pressures_hpa[:, None]),
        axis=1,
    ).astype(np.int32)
    return levels, pressure_profile_hpa[levels]


def _native_level_differences(
    increment: HydrostaticState,
    model_levels: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.stack(
            [
                np.asarray(jax.device_get(getattr(increment, name).data[:, level]))
                for level in model_levels
            ],
            axis=0,
        )
        for name in ("u", "v", "T")
    }


def _regular_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = 8.0
    selected = (
        (lon >= args.lon_min - margin)
        & (lon <= args.lon_max + margin)
        & (lat >= args.lat_min - margin)
        & (lat <= args.lat_max + margin)
        & np.isfinite(field)
    )
    xi = np.linspace(args.lon_min, args.lon_max, 241)
    yi = np.linspace(args.lat_min, args.lat_max, 181)
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata(
        (lon[selected], lat[selected]),
        np.asarray(field)[selected],
        (xx, yy),
        method="linear",
    )
    return xx, yy, zz


def _map_axis(ax, args: argparse.Namespace, show_left: bool, show_bottom: bool):
    import cartopy.crs as ccrs

    ax.set_extent(
        [args.lon_min, args.lon_max, args.lat_min, args.lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="110m", linewidth=0.65, color="0.2")
    gridlines = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="0.55",
        alpha=0.55,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.left_labels = show_left
    gridlines.bottom_labels = show_bottom
    gridlines.xlabel_style = {"size": 11}
    gridlines.ylabel_style = {"size": 11}


def _symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, 99.5))
    return max(limit, float(np.finfo(np.float64).eps))


def _plot_three_level_response(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    differences: dict[str, np.ndarray],
    pressure_levels_hpa: np.ndarray,
    obs_lon: float,
    obs_lat: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    import cartopy.crs as ccrs

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(11.2, 8.8),
        subplot_kw={"projection": projection},
    )
    columns = (
        ("u", r"$\delta u$", r"m s$^{-1}$"),
        ("v", r"$\delta v$", r"m s$^{-1}$"),
        ("T", r"$\delta T$", "K"),
    )
    limits = {name: _symmetric_limit(differences[name]) for name, _, _ in columns}
    contour_sets = []
    for row, pressure_hpa in enumerate(pressure_levels_hpa):
        for column, (name, title, _) in enumerate(columns):
            ax = axes[row, column]
            xx, yy, zz = _regular_grid(lon, lat, differences[name][row], args)
            limit = limits[name]
            levels = np.linspace(-limit, limit, 19)
            contours = ax.contourf(
                xx,
                yy,
                zz,
                levels=levels,
                cmap="RdBu_r",
                extend="both",
                transform=projection,
            )
            _map_axis(ax, args, show_left=column == 0, show_bottom=row == 2)
            if row == 0:
                ax.set_title(title, fontweight="bold", pad=7)
            if column == 0:
                ax.text(
                    -0.16,
                    0.5,
                    f"{pressure_hpa:g} hPa",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
            if np.isclose(pressure_hpa, args.obs_pressure_hpa):
                ax.plot(
                    obs_lon,
                    obs_lat,
                    marker="x",
                    markersize=8,
                    markeredgewidth=1.8,
                    color="black",
                    transform=projection,
                    zorder=10,
                )
            if row == 0:
                contour_sets.append(contours)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.95, bottom=0.17, wspace=0.045, hspace=0.075)
    colorbar_left = (0.125, 0.425, 0.725)
    for column, (_, _, unit) in enumerate(columns):
        colorbar_axis = fig.add_axes([colorbar_left[column], 0.075, 0.235, 0.022])
        limit = limits[columns[column][0]]
        ticks = np.linspace(-limit, limit, 5)
        colorbar = fig.colorbar(
            contour_sets[column],
            cax=colorbar_axis,
            orientation="horizontal",
            ticks=ticks,
        )
        tick_labels = (
            [f"{tick:.3f}" for tick in ticks] if unit == "K" else [f"{tick:.1e}" for tick in ticks]
        )
        colorbar.set_ticklabels(tick_labels)
        colorbar.set_label(unit, fontsize=12)
        colorbar.ax.tick_params(labelsize=9)
    fig.savefig(path.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return limits


def _plot_surface_pressure(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    pressure_increment: np.ndarray,
    obs_lon: float,
    obs_lat: float,
    args: argparse.Namespace,
) -> float:
    import cartopy.crs as ccrs

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(7.4, 5.2), subplot_kw={"projection": projection})
    xx, yy, zz = _regular_grid(lon, lat, pressure_increment, args)
    limit = _symmetric_limit(pressure_increment)
    is_zero = float(np.nanmax(np.abs(pressure_increment))) <= 1.0e-12
    plot_limit = 1.0e-3 if is_zero else limit
    contours = ax.contourf(
        xx,
        yy,
        zz,
        levels=np.linspace(-plot_limit, plot_limit, 19),
        cmap="RdBu_r",
        extend="both",
        transform=projection,
    )
    _map_axis(ax, args, show_left=True, show_bottom=True)
    ax.plot(
        obs_lon,
        obs_lat,
        marker="x",
        markersize=8,
        markeredgewidth=1.8,
        color="black",
        transform=projection,
        zorder=10,
    )
    title = r"$\delta p_s$ (zero increment)" if is_zero else r"$\delta p_s$"
    ax.set_title(title, fontweight="bold", pad=8)
    pressure_ticks = np.linspace(-plot_limit, plot_limit, 5)
    colorbar = fig.colorbar(
        contours,
        ax=ax,
        orientation="horizontal",
        fraction=0.055,
        pad=0.11,
        aspect=30,
        ticks=pressure_ticks,
    )
    colorbar.set_ticklabels([f"{tick:.2f}" for tick in pressure_ticks])
    colorbar.set_label("Pa", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.16)
    fig.savefig(path.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return limit


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"jax={jax.__version__} backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )

    mesh = create_grid("mpas", args.resolution, lloyd_iterations=args.lloyd_iterations)
    sigma = create_sigma_coordinate(args.nlev)
    background = _state_from_sample(args.sample, args.background)
    params = load_gen_be_params(args.gen_be)
    params = params._replace(len_scale=params.len_scale * float(args.len_scale_multiplier))
    spec = build_control_spec(
        background,
        grid=mesh,
        fields=("u", "v", "T", "p_s", "tracers"),
    )
    transform = GenBETransform(
        params,
        spec,
        mesh,
        n_diffusion_iter=args.n_diffusion_iter,
    )
    obs_cell = _nearest_cell(mesh, args.obs_lat, args.obs_lon)
    lat = np.rad2deg(np.asarray(mesh.latCell))
    lon = ((np.rad2deg(np.asarray(mesh.lonCell)) + 180.0) % 360.0) - 180.0
    obs_lat = float(lat[obs_cell])
    obs_lon = float(lon[obs_cell])
    pressure_levels_hpa = np.asarray(
        [float(value) for value in args.plot_pressure_levels_hpa.split(",")],
        dtype=np.float64,
    )
    if pressure_levels_hpa.size != 3:
        raise ValueError("Exactly three plot pressure levels are required")
    diagnostic_levels, diagnostic_pressures_hpa = _nearest_model_levels(
        background, sigma, obs_cell, pressure_levels_hpa
    )
    observation_levels, observation_pressures_hpa = _nearest_model_levels(
        background, sigma, obs_cell, np.asarray([args.obs_pressure_hpa])
    )
    observation_level = int(observation_levels[0])
    observation_model_pressure_hpa = float(observation_pressures_hpa[0])
    zero = jnp.zeros(spec.total_size, dtype=jnp.float64)
    be_scale = float(args.background_error_std_multiplier)

    def increment_from_white(control: jax.Array) -> HydrostaticState:
        model_increment = be_scale * transform.sqrt_multiply(control)
        return control_to_increment(model_increment, spec, background)

    def observation_from_white(control: jax.Array) -> jax.Array:
        increment_local = increment_from_white(control)
        return (
            background.T.data[obs_cell, observation_level]
            + increment_local.T.data[obs_cell, observation_level]
        )

    h_background, gradient = jax.value_and_grad(observation_from_white)(zero)
    jax.block_until_ready(gradient)
    innovation = jnp.asarray(args.innovation_k, dtype=jnp.float64)
    obs_error = jnp.asarray(args.obs_error_k, dtype=jnp.float64)
    hbht = jnp.vdot(gradient, gradient)
    analysis_control = gradient * innovation / (hbht + obs_error * obs_error)
    increment = increment_from_white(analysis_control)
    h_analysis_linear = h_background + jnp.vdot(gradient, analysis_control)
    h_analysis_nonlinear = observation_from_white(analysis_control)
    differences = _native_level_differences(increment, diagnostic_levels)
    ps_increment = np.asarray(jax.device_get(increment.p_s.data))
    control_np = np.asarray(jax.device_get(analysis_control))
    jb = 0.5 * float(np.vdot(control_np, control_np))
    linear_residual = float(h_analysis_linear - (h_background + innovation))
    jo = 0.5 * (linear_residual / float(obs_error)) ** 2

    limits = _plot_three_level_response(
        args.output_dir / "figure01_jung_nmc_temperature_observation_response_3x3",
        lon,
        lat,
        differences,
        pressure_levels_hpa,
        obs_lon,
        obs_lat,
        args,
    )
    ps_limit = _plot_surface_pressure(
        args.output_dir / "figure02_jung_surface_pressure_response",
        lon,
        lat,
        ps_increment,
        obs_lon,
        obs_lat,
        args,
    )

    np.savez_compressed(
        args.output_dir / "jung_nmc_temperature_single_obs_3dvar.npz",
        sample=str(args.sample),
        gen_be=str(args.gen_be),
        latitude_cell_degrees=lat,
        longitude_cell_degrees=lon,
        observation_cell=np.asarray(obs_cell),
        observation_latitude_cell_degrees=np.asarray(obs_lat),
        observation_longitude_cell_degrees=np.asarray(obs_lon),
        observation_pressure_hpa=np.asarray(args.obs_pressure_hpa),
        observation_model_level_zero_based=np.asarray(observation_level),
        observation_model_level_one_based=np.asarray(observation_level + 1),
        observation_model_level_pressure_hpa=np.asarray(observation_model_pressure_hpa),
        innovation_k=np.asarray(args.innovation_k),
        observation_error_standard_deviation_k=np.asarray(args.obs_error_k),
        pressure_levels_hpa=pressure_levels_hpa,
        diagnostic_model_levels_zero_based=diagnostic_levels,
        diagnostic_model_levels_one_based=diagnostic_levels + 1,
        diagnostic_model_level_pressures_hpa=diagnostic_pressures_hpa,
        temperature_increment_k=differences["T"],
        zonal_wind_increment_m_s=differences["u"],
        meridional_wind_increment_m_s=differences["v"],
        surface_pressure_increment_pa=ps_increment,
        h_background_k=np.asarray(jax.device_get(h_background)),
        h_analysis_linear_k=np.asarray(jax.device_get(h_analysis_linear)),
        h_analysis_nonlinear_k=np.asarray(jax.device_get(h_analysis_nonlinear)),
        hbht_k2=np.asarray(jax.device_get(hbht)),
        cost_jb=np.asarray(jb),
        cost_jo=np.asarray(jo),
    )
    summary = {
        "experiment": (
            "Jung et al. single-temperature-observation 3DVar response "
            "using NMC-trained static B"
        ),
        "reference": "https://doi.org/10.5194/gmd-17-3879-2024",
        "sample": str(args.sample),
        "background": args.background,
        "gen_be": str(args.gen_be),
        "grid": {
            "type": "MPAS quasi-uniform",
            "resolution": args.resolution,
            "levels": args.nlev,
            "lloyd_iterations": args.lloyd_iterations,
        },
        "observation": {
            "variable": "temperature",
            "target_pressure_hpa": args.obs_pressure_hpa,
            "model_level_zero_based": observation_level,
            "model_level_one_based": observation_level + 1,
            "model_level_pressure_hpa": observation_model_pressure_hpa,
            "requested_latitude_degrees": args.obs_lat,
            "requested_longitude_degrees_east": args.obs_lon,
            "cell": obs_cell,
            "cell_latitude_degrees": obs_lat,
            "cell_longitude_degrees": obs_lon,
            "innovation_k": args.innovation_k,
            "error_standard_deviation_k": args.obs_error_k,
        },
        "background_covariance_runtime": {
            "length_scale_multiplier": args.len_scale_multiplier,
            "standard_deviation_multiplier": args.background_error_std_multiplier,
            "diffusion_iterations": args.n_diffusion_iter,
            "additional_localization": False,
        },
        "diagnostic_target_pressures_hpa": pressure_levels_hpa.tolist(),
        "diagnostic_model_levels_zero_based": diagnostic_levels.tolist(),
        "diagnostic_model_levels_one_based": (diagnostic_levels + 1).tolist(),
        "diagnostic_model_level_pressures_hpa": diagnostic_pressures_hpa.tolist(),
        "fit": {
            "h_background_k": float(h_background),
            "observation_k": float(h_background + innovation),
            "h_analysis_linear_k": float(h_analysis_linear),
            "h_analysis_nonlinear_k": float(h_analysis_nonlinear),
            "linear_o_minus_a_k": linear_residual,
            "hbht_k2": float(hbht),
            "cost_jb": jb,
            "cost_jo": jo,
            "cost_total": jb + jo,
        },
        "plot_limits": {**limits, "p_s": ps_limit},
        "regional_extrema_by_row": {
            name: [
                {"min": float(np.nanmin(values)), "max": float(np.nanmax(values))}
                for values in differences[name]
            ]
            for name in ("u", "v", "T")
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved outputs in {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
