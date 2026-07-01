#!/usr/bin/env python
"""Offline single-point multilayer land spin-up driver (LMIP).

Runs the multi-layer soil land model at a single latitude/longitude point
with synthetic diurnal+seasonal atmospheric forcing.  Designed for 10-year
soil spin-up studies before coupling to ERA5.

Usage::

    # Quick smoke test (10 days):
    JAX_ENABLE_X64=1 python scripts/run_lmip.py \\
      --lat 45.5 --lon -93.1 --days 10 \\
      --soil-texture loam --veg-type c3_grass --output /tmp/lmip_test

    # Seasonal cycle (1 year):
    JAX_ENABLE_X64=1 python scripts/run_lmip.py \\
      --lat 45.5 --lon -93.1 --days 365 \\
      --soil-texture loam --veg-type c3_grass --output /tmp/lmip_1yr

    # Full 10-year spin-up:
    JAX_ENABLE_X64=1 python scripts/run_lmip.py \\
      --lat 45.5 --lon -93.1 --days 3650 \\
      --soil-texture loam --veg-type c3_grass --output lmip_output/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple

sys.stdout.reconfigure(line_buffering=True)

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Insert src/ directory so legoesm can be found regardless of install state.
_SRC_ROOT = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# ===========================================================================
# legoESM imports
# ===========================================================================

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.richards import RichardsConfig
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.carbon.carbon_cycle import init_carbon_state
from legoesm.land.multilayer_land import (
    step_multilayer_land,
    init_multilayer_land_state,
)
from legoesm.land.surface_params import (
    CLM5_PFT_NAMES,
    _CLM5_PFT_TABLE_RAW,
    PARAM_NAMES,
)

# ===========================================================================
# Constants
# ===========================================================================

U_MIN = 1.0  # m/s — wind speed floor
_SECS_PER_DAY = 86400.0
_DEFAULT_LAND_CONFIG = MultiLayerLandConfig()
_VALID_CARBON_SCHEMES = ("none", "differland", "seasonal")


class LMIPRunConfig(NamedTuple):
    """CLI-resolved LMIP config and run controls."""

    land: MultiLayerLandConfig
    max_wallclock_seconds: float
    restart_buffer_seconds: float
    seed: int


def _wallclock_exhausted(elapsed_s: float, max_s: float, buffer_s: float) -> bool:
    """True when the loop should checkpoint and exit before wallclock expiry."""
    return max_s > 0.0 and elapsed_s >= (max_s - buffer_s)


# ===========================================================================
# Soil texture presets (Carsel & Parrish 1988, Table 1)
# Van Genuchten parameters for 11 USDA texture classes.
# Keys: theta_r, theta_sat, alpha_vg [1/m], n_vg, K_sat [m/s]
# ===========================================================================

# USDA texture van-Genuchten presets (Carsel & Parrish 1988) — shared with the
# global CLM reference-soil-map provider (no duplicated soil constants).
from legoesm.land.soil_texture import SOIL_TEXTURE_VG as _SOIL_TEXTURE_PRESETS


# ===========================================================================
# PFT parameter extraction
# ===========================================================================

def _get_pft_row(veg_type: str) -> dict:
    """Look up CLM5 PFT parameters by name.

    Returns a dict with albedo_veg, emissivity, z0, root_depth,
    theta_wp, theta_fc from the CLM5 PFT table.
    """
    if veg_type not in CLM5_PFT_NAMES:
        valid = ", ".join(CLM5_PFT_NAMES)
        raise ValueError(
            f"Unknown veg_type {veg_type!r}. Valid choices: {valid}"
        )
    idx = CLM5_PFT_NAMES.index(veg_type)
    row = _CLM5_PFT_TABLE_RAW[idx]
    return {name: val for name, val in zip(PARAM_NAMES, row)}


def build_config_from_args(args: argparse.Namespace) -> LMIPRunConfig:
    """Resolve LMIP CLI arguments into the land config and run controls."""
    texture_kwargs = _SOIL_TEXTURE_PRESETS[args.soil_texture]
    pft_row = _get_pft_row(args.veg_type)
    land = MultiLayerLandConfig(
        albedo_land=pft_row["albedo_veg"],
        emissivity_land=pft_row["emissivity"],
        z0_land=(
            args.z0_land
            if args.z0_land is not None
            else pft_row["z0"]
        ),
        Cd_land=(
            args.cd_land
            if args.cd_land is not None
            else _DEFAULT_LAND_CONFIG.Cd_land
        ),
        Ch_land=(
            args.ch_land
            if args.ch_land is not None
            else _DEFAULT_LAND_CONFIG.Ch_land
        ),
        beta_min=(
            args.beta_min
            if args.beta_min is not None
            else _DEFAULT_LAND_CONFIG.beta_min
        ),
        bulk_scheme=args.bulk_scheme,
        root_depth=pft_row["root_depth"],
        theta_wp=pft_row["theta_wp"],
        theta_fc=pft_row["theta_fc"],
        snow_albedo_feedback=args.snow_albedo_feedback,
        soil_grid=SoilGridConfig(
            n_layers=args.n_layers,
            total_depth=args.soil_depth,
            # Keep the historical LMIP top-layer thickness.
            growth_factor=1.5,
        ),
        hydraulics=SoilHydraulicsConfig(**texture_kwargs),
        thermal=SoilThermalConfig(),
        richards=RichardsConfig(),
        carbon=CarbonConfig(scheme=args.carbon_scheme),
    )
    return LMIPRunConfig(
        land=land,
        max_wallclock_seconds=args.max_wallclock_seconds,
        restart_buffer_seconds=args.restart_buffer_seconds,
        seed=args.seed,
    )


# ===========================================================================
# Synthetic atmospheric forcing
# ===========================================================================

def _make_forcing(
    lat_rad: float,
    lon_rad: float,
    day: float,
    hour: float,
    *,
    dtype=jnp.float64,
    precip_rate: float = 2e-5,
) -> AtmToSurface:
    """Construct synthetic single-column atmospheric forcing.

    Parameters
    ----------
    lat_rad : float
        Latitude in radians.
    lon_rad : float
        Longitude in radians (used to compute local solar hour angle).
    day : float
        Day of year [0, 365).
    hour : float
        UTC hour of day [0, 24).
    dtype :
        JAX dtype (default float64).
    precip_rate : float
        Constant precipitation rate [kg/m2/s].

    Returns
    -------
    AtmToSurface
        Forcing with shape (1,) for all fields.

    Notes
    -----
    Atmospheric temperature includes three components:
    1. Latitudinal mean: T_base = 288 - 30·|φ|/(π/2)
    2. Seasonal: amplitude ~15 K × |φ|/(π/2), NH peak at doy≈200 (July)
    3. Diurnal: ±3 K, peak at local hour 14
    This produces physically realistic annual mean and seasonal cycle
    across latitudes. At 45.5°N: T_atm ≈ 265 K (Jan) to 280 K (Jul).
    """
    # --- Solar geometry ---
    # Solar declination (degrees → radians)
    decl_rad = 23.45 * jnp.pi / 180.0 * jnp.sin(
        2.0 * jnp.pi * (day - 80.0) / 365.0
    )
    # Local hour angle: UTC hour shifted by longitude (15 deg/hour)
    local_hour = hour + lon_rad * (180.0 / jnp.pi) / 15.0
    ha = (local_hour - 12.0) * 15.0 * jnp.pi / 180.0
    cos_sza = (
        jnp.sin(lat_rad) * jnp.sin(decl_rad)
        + jnp.cos(lat_rad) * jnp.cos(decl_rad) * jnp.cos(ha)
    )
    cos_sza = jnp.maximum(cos_sza, 0.0)

    sw_down = jnp.asarray([constants.S_0 * cos_sza], dtype=dtype)

    # --- Atmospheric temperature: latitudinal baseline + seasonal + diurnal ---
    # Latitudinal mean (from test forcing, adapted from CLM convention)
    T_base = 288.0 - 30.0 * abs(lat_rad) / (jnp.pi / 2.0)
    # Seasonal: amplitude proportional to |latitude|, NH peak at doy≈200 (July).
    # cos(2π*(day-200)/365)=1 at doy=200 (summer peak),
    # ≈ -1 at doy=15 (winter minimum).
    T_seasonal_amp = 15.0 * abs(lat_rad) / (jnp.pi / 2.0)
    T_season = T_seasonal_amp * jnp.cos(2.0 * jnp.pi * (day - 200.0) / 365.0)
    # Diurnal: ±3 K, peak at local solar noon + 2 h
    diurnal_amp = 3.0
    T_atm = jnp.asarray(
        [T_base + T_season
         + diurnal_amp * jnp.cos(2.0 * jnp.pi * (local_hour - 14.0) / 24.0)],
        dtype=dtype,
    )

    # --- LW down: effective emissivity ~0.75 of blackbody ---
    lw_down = jnp.asarray([0.75 * constants.sigma_sb * T_atm[0] ** 4], dtype=dtype)

    # --- Humidity: ~60% RH using model's saturation_mixing_ratio ---
    # saturation_mixing_ratio requires p [Pa]; use standard surface pressure
    p_sfc = jnp.asarray([1.0e5], dtype=dtype)
    q_sat = saturation_mixing_ratio(T_atm, p_sfc)
    q_atm = (0.6 * q_sat).astype(dtype)

    # --- Precipitation: rain below 275 K threshold becomes snow ---
    precip_total = jnp.asarray([precip_rate], dtype=dtype)
    precip_snow = jnp.where(
        T_atm < 275.0,
        precip_total,
        jnp.zeros(1, dtype=dtype),
    )

    rho = jnp.asarray([1.2], dtype=dtype)

    return AtmToSurface(
        sw_down=sw_down,
        lw_down=lw_down,
        precip_total=precip_total,
        precip_snow=precip_snow,
        T_lowest=T_atm,
        q_lowest=q_atm,
        u_lowest=jnp.asarray([3.0], dtype=dtype),
        v_lowest=jnp.asarray([2.0], dtype=dtype),
        p_lowest=jnp.asarray([9.5e4], dtype=dtype),
        p_surface=p_sfc,
        rho_lowest=rho,
        cos_zenith=jnp.asarray([cos_sza], dtype=dtype),
        co2_ppmv=jnp.asarray([412.0], dtype=dtype),
        has_radiation=jnp.ones(1, dtype=dtype),
        has_precipitation=jnp.ones(1, dtype=dtype),
    )


# ===========================================================================
# Output helpers
# ===========================================================================

def _append_netcdf(
    out_path: Path,
    days: np.ndarray,
    T_soil: np.ndarray,
    theta_soil: np.ndarray,
    psi_soil: np.ndarray,
    snow_depth: np.ndarray,
    shflx: np.ndarray,
    lhflx: np.ndarray,
    runoff_surface: np.ndarray,
    runoff_subsurface: np.ndarray,
    attrs: dict,
) -> None:
    """Append daily diagnostics to a NetCDF file.

    Creates the file with correct dimensions on first call; appends
    along the time axis on subsequent calls.

    Parameters
    ----------
    out_path : Path
        Path to the NetCDF output file.
    days : np.ndarray, shape (n_days,)
        Day index of each record.
    T_soil : np.ndarray, shape (n_days, n_layers)
    theta_soil : np.ndarray, shape (n_days, n_layers)
    psi_soil : np.ndarray, shape (n_days, n_layers)
    snow_depth : np.ndarray, shape (n_days,)
    shflx : np.ndarray, shape (n_days,)
    lhflx : np.ndarray, shape (n_days,)
    runoff_surface : np.ndarray, shape (n_days,)
    runoff_subsurface : np.ndarray, shape (n_days,)
    attrs : dict
        Global attributes (lat, lon, soil_texture, etc.).
    """
    try:
        import netCDF4 as nc  # type: ignore
    except ImportError:
        # Fall back to scipy.io.netcdf if netCDF4 not available
        _append_netcdf_scipy(
            out_path, days, T_soil, theta_soil, psi_soil, snow_depth,
            shflx, lhflx, runoff_surface, runoff_subsurface, attrs,
        )
        return

    n_days, n_layers = T_soil.shape
    file_exists = out_path.exists()

    with nc.Dataset(str(out_path), "a" if file_exists else "w") as ds:
        if not file_exists:
            ds.createDimension("time", None)   # unlimited
            ds.createDimension("layer", n_layers)

            t_var = ds.createVariable("time", "f8", ("time",))
            t_var.units = "days since model start"
            t_var.long_name = "simulation day"

            for name, shape, units, long_name in [
                ("T_soil",            ("time", "layer"), "K",       "Soil temperature"),
                ("theta_soil",        ("time", "layer"), "m3 m-3",  "Volumetric water content"),
                ("psi_soil",          ("time", "layer"), "m",       "Soil matric potential"),
                ("snow_depth",        ("time",),         "kg m-2",  "Snow water equivalent"),
                ("shflx",             ("time",),         "W m-2",   "Sensible heat flux (up+)"),
                ("lhflx",             ("time",),         "W m-2",   "Latent heat flux (up+)"),
                ("runoff_surface",    ("time",),         "kg m-2 s-1", "Surface runoff"),
                ("runoff_subsurface", ("time",),         "kg m-2 s-1", "Subsurface runoff"),
            ]:
                v = ds.createVariable(name, "f4", shape, zlib=True, complevel=4)
                v.units = units
                v.long_name = long_name

            for k, v in attrs.items():
                setattr(ds, k, str(v))

        # Append
        t0 = len(ds.variables["time"])
        ds.variables["time"][t0:t0 + n_days] = days
        ds.variables["T_soil"][t0:t0 + n_days, :] = T_soil
        ds.variables["theta_soil"][t0:t0 + n_days, :] = theta_soil
        ds.variables["psi_soil"][t0:t0 + n_days, :] = psi_soil
        ds.variables["snow_depth"][t0:t0 + n_days] = snow_depth
        ds.variables["shflx"][t0:t0 + n_days] = shflx
        ds.variables["lhflx"][t0:t0 + n_days] = lhflx
        ds.variables["runoff_surface"][t0:t0 + n_days] = runoff_surface
        ds.variables["runoff_subsurface"][t0:t0 + n_days] = runoff_subsurface


def _append_netcdf_scipy(
    out_path: Path,
    days: np.ndarray,
    T_soil: np.ndarray,
    theta_soil: np.ndarray,
    psi_soil: np.ndarray,
    snow_depth: np.ndarray,
    shflx: np.ndarray,
    lhflx: np.ndarray,
    runoff_surface: np.ndarray,
    runoff_subsurface: np.ndarray,
    attrs: dict,
) -> None:
    """Fallback NetCDF writer using numpy .npz when no NetCDF library available.

    Writes one .npz file per checkpoint segment (overwritten at each call).
    A lightweight index file ``land_spinup_index.json`` tracks all chunks.
    """
    import json as _json

    # Write npz chunk
    chunk_id = int(days[0])
    chunk_path = out_path.parent / f"land_spinup_chunk_{chunk_id:07d}.npz"
    np.savez_compressed(
        str(chunk_path),
        time=days,
        T_soil=T_soil,
        theta_soil=theta_soil,
        psi_soil=psi_soil,
        snow_depth=snow_depth,
        shflx=shflx,
        lhflx=lhflx,
        runoff_surface=runoff_surface,
        runoff_subsurface=runoff_subsurface,
    )

    # Update index
    index_path = out_path.parent / "land_spinup_index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = _json.load(f)
    else:
        index = {"attrs": {k: str(v) for k, v in attrs.items()}, "chunks": []}
    index["chunks"].append(str(chunk_path.name))
    with open(index_path, "w") as f:
        _json.dump(index, f, indent=2)


def _save_restart(
    restart_path: Path,
    step: int,
    day: float,
    state,
    carbon_state=None,
) -> None:
    """Save a restart checkpoint as .npz."""
    payload = {
        "step": np.array(step, dtype=np.int64),
        "day": np.array(day, dtype=np.float64),
        "T_soil": np.asarray(state.T_soil),
        "theta_soil": np.asarray(state.theta_soil),
        "psi_soil": np.asarray(state.psi_soil),
        "snow_depth": np.asarray(state.snow_depth),
        "snow_age": np.asarray(state.snow_age),
        "surface_water": np.asarray(
            state.surface_water if state.surface_water is not None
            else np.zeros_like(np.asarray(state.snow_depth))),
    }
    if carbon_state is not None:
        for field in carbon_state._fields:
            payload[f"carbon_{field}"] = np.asarray(
                getattr(carbon_state, field)
            )
    np.savez_compressed(str(restart_path), **payload)


def _load_restart(restart_path: Path, config: MultiLayerLandConfig):
    """Load a restart checkpoint and reconstruct MultiLayerLandState."""
    from legoesm.land.state import MultiLayerLandState
    data = np.load(str(restart_path))
    state = MultiLayerLandState(
        T_soil=jnp.asarray(data["T_soil"]),
        psi_soil=jnp.asarray(data["psi_soil"]),
        theta_soil=jnp.asarray(data["theta_soil"]),
        runoff_surface=jnp.zeros(1),
        runoff_subsurface=jnp.zeros(1),
        snow_depth=jnp.asarray(data["snow_depth"]),
        snow_age=jnp.asarray(data["snow_age"]),
        # Backward-compat: old checkpoints predate surface ponding -> start dry.
        surface_water=jnp.asarray(data["surface_water"]) if "surface_water" in data
        else jnp.zeros_like(jnp.asarray(data["snow_depth"])),
    )
    start_step = int(data["step"])
    start_day = float(data["day"])
    carbon_state = None
    if config.carbon.scheme == "differland":
        carbon_fields = [
            "C_lab", "C_fol", "C_root", "C_wood", "C_lit", "C_som",
        ]
        if all(f"carbon_{field}" in data for field in carbon_fields):
            from legoesm.land.carbon.config import CarbonState
            carbon_state = CarbonState(**{
                field: jnp.asarray(data[f"carbon_{field}"])
                for field in carbon_fields
            })
        else:
            carbon_state = init_carbon_state(
                tuple(state.T_soil.shape[:-1]), config.carbon
            )
    return state, carbon_state, start_step, start_day


# ===========================================================================
# Main driver
# ===========================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline single-point multilayer land spin-up (LMIP)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lat", type=float, required=True,
                   help="Latitude [deg]")
    p.add_argument("--lon", type=float, default=0.0,
                   help="Longitude [deg] (used for local solar hour angle)")
    p.add_argument("--days", type=int, default=3650,
                   help="Total simulation length [days]")
    p.add_argument("--dt", type=float, default=1800.0,
                   help="Timestep [s]")
    p.add_argument("--soil-texture", default="loam",
                   choices=sorted(_SOIL_TEXTURE_PRESETS),
                   help="USDA soil texture class")
    p.add_argument("--veg-type", default="c3_grass",
                   choices=list(CLM5_PFT_NAMES),
                   help="CLM5 plant functional type")
    p.add_argument("--t-init", type=float, default=278.0,
                   help="Initial uniform soil temperature [K]. "
                        "Should be close to the local annual-mean atmospheric "
                        "temperature to avoid large initial flux transients. "
                        "For 45.5N use ~278 K; for tropics use ~290 K.")
    p.add_argument("--n-layers", type=int, default=10,
                   help="Number of soil layers")
    p.add_argument("--soil-depth", type=float, default=3.0,
                   help="Total soil depth [m]")
    p.add_argument("--bulk-scheme", default="most",
                   choices=["constant", "most"],
                   help="Bulk flux scheme")
    p.add_argument("--output", default="lmip_output",
                   help="Output directory")
    p.add_argument("--checkpoint-days", type=int, default=100,
                   help="Save restart every N days")
    p.add_argument("--max-wallclock-seconds", type=float, default=0.0,
                   help="Wallclock budget [s] for clean checkpoint+exit")
    p.add_argument("--restart-buffer-seconds", type=float, default=600.0,
                   help="Wallclock buffer [s] reserved for restart writes")
    p.add_argument("--seed", type=int, default=0,
                   help="Master RNG seed for reproducibility metadata")
    p.add_argument("--diag-interval", type=int, default=1,
                   help="Save diagnostics every N days")
    p.add_argument("--start-day", type=float, default=0.0,
                   help="Starting day of year for seasonal forcing [0, 365)")
    p.add_argument("--restart-from", default=None,
                   help="Path to .npz restart file to resume from")
    p.add_argument("--precip-rate", type=float, default=2e-5,
                   help="Constant precipitation rate [kg/m2/s]")
    p.add_argument("--cd-land", type=float, default=None,
                   help="Land drag coefficient for the constant bulk scheme")
    p.add_argument("--ch-land", type=float, default=None,
                   help="Land heat transfer coefficient")
    p.add_argument("--z0-land", type=float, default=None,
                   help="Override PFT roughness length [m]")
    p.add_argument("--beta-min", type=float, default=None,
                   help="Minimum soil moisture availability")
    p.add_argument("--carbon-scheme", default="none",
                   choices=_VALID_CARBON_SCHEMES,
                   help="Land carbon cycle scheme")
    p.add_argument("--snow-albedo-feedback",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Enable/disable snow albedo feedback")
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Latitude/longitude in radians ---
    lat_rad = float(args.lat * jnp.pi / 180.0)
    lon_rad = float(args.lon * jnp.pi / 180.0)
    lat_jnp = jnp.asarray([lat_rad])   # shape (1,) for snow_albedo_feedback

    run_config = build_config_from_args(args)
    config = run_config.land

    dt = args.dt
    n_total_steps = int(args.days * _SECS_PER_DAY / dt)
    steps_per_day = int(_SECS_PER_DAY / dt)
    # steps_per_day must divide evenly for clean daily accumulation
    if steps_per_day * args.days < n_total_steps:
        n_total_steps = steps_per_day * args.days

    # --- Initialise state ---
    if args.restart_from is not None:
        print(f"Loading restart from {args.restart_from}")
        state, carbon_state, start_step, start_day_abs = _load_restart(
            Path(args.restart_from), config
        )
        print(f"  Resumed at step {start_step}, day {start_day_abs:.2f}")
    else:
        state = init_multilayer_land_state(1, config, T_init=args.t_init)
        carbon_state = (
            init_carbon_state((1,), config.carbon)
            if config.carbon.scheme == "differland"
            else None
        )
        start_step = 0
        start_day_abs = 0.0

    # --- JIT-compile step function ---
    # Config, lat, dt captured in closure (compile-time constants per CLAUDE.md).
    # doy is a traced argument (changes every step → prevents recompile).
    @jax.jit
    def _step(state, carbon_state, forcing, doy):
        return step_multilayer_land(
            state, forcing, config, U_MIN, dt,
            lat=lat_jnp, carbon_state=carbon_state, doy=doy,
        )

    # --- Diagnostic accumulators (daily means) ---
    T_soil_acc = np.zeros(args.n_layers, dtype=np.float64)
    theta_soil_acc = np.zeros(args.n_layers, dtype=np.float64)
    psi_soil_acc = np.zeros(args.n_layers, dtype=np.float64)
    snow_depth_acc = 0.0
    shflx_acc = 0.0
    lhflx_acc = 0.0
    runoff_sfc_acc = 0.0
    runoff_sub_acc = 0.0
    acc_count = 0

    # Buffered daily diagnostics between checkpoint flushes
    diag_days: list[float] = []
    diag_T_soil: list[np.ndarray] = []
    diag_theta_soil: list[np.ndarray] = []
    diag_psi_soil: list[np.ndarray] = []
    diag_snow_depth: list[float] = []
    diag_shflx: list[float] = []
    diag_lhflx: list[float] = []
    diag_runoff_sfc: list[float] = []
    diag_runoff_sub: list[float] = []

    nc_path = out_dir / "land_spinup.nc"
    nc_attrs = {
        "lat": args.lat,
        "lon": args.lon,
        "soil_texture": args.soil_texture,
        "veg_type": args.veg_type,
        "dt_s": dt,
        "start_day": args.start_day,
        "n_layers": args.n_layers,
        "soil_depth_m": args.soil_depth,
    }

    t_wall_start = time.time()
    status = "PASS"
    error_msg = ""

    def _flush_buffered_diagnostics() -> None:
        if len(diag_days) == 0:
            return
        _append_netcdf(
            nc_path,
            days=np.array(diag_days),
            T_soil=np.stack(diag_T_soil),
            theta_soil=np.stack(diag_theta_soil),
            psi_soil=np.stack(diag_psi_soil),
            snow_depth=np.array(diag_snow_depth),
            shflx=np.array(diag_shflx),
            lhflx=np.array(diag_lhflx),
            runoff_surface=np.array(diag_runoff_sfc),
            runoff_subsurface=np.array(diag_runoff_sub),
            attrs=nc_attrs,
        )
        diag_days.clear()
        diag_T_soil.clear()
        diag_theta_soil.clear()
        diag_psi_soil.clear()
        diag_snow_depth.clear()
        diag_shflx.clear()
        diag_lhflx.clear()
        diag_runoff_sfc.clear()
        diag_runoff_sub.clear()

    try:
        for global_step in range(start_step, start_step + n_total_steps):
            step_in_run = global_step - start_step
            # Absolute simulation time from t=0 of this run segment
            t_sim = start_day_abs * _SECS_PER_DAY + step_in_run * dt

            # Day of year (seasonal cycle wraps every 365 days)
            doy = (args.start_day + t_sim / _SECS_PER_DAY) % 365.0
            hour = (t_sim / 3600.0) % 24.0

            forcing = _make_forcing(
                lat_rad, lon_rad, doy, hour,
                precip_rate=args.precip_rate,
            )

            state, response, carbon_state = _step(
                state, carbon_state, forcing, jnp.asarray(doy)
            )

            # Accumulate
            T_soil_acc += np.asarray(state.T_soil[0])
            theta_soil_acc += np.asarray(state.theta_soil[0])
            psi_soil_acc += np.asarray(state.psi_soil[0])
            snow_depth_acc += float(np.asarray(state.snow_depth[0]))
            shflx_acc += float(np.asarray(response.shflx[0]))
            lhflx_acc += float(np.asarray(response.lhflx[0]))
            runoff_sfc_acc += float(np.asarray(state.runoff_surface[0]))
            runoff_sub_acc += float(np.asarray(state.runoff_subsurface[0]))
            acc_count += 1

            # End of day: store daily mean if within diag_interval cadence
            is_end_of_day = (step_in_run + 1) % steps_per_day == 0
            if is_end_of_day:
                day_idx = (step_in_run + 1) // steps_per_day  # 1-based day number
                abs_day = start_day_abs + day_idx

                if day_idx % args.diag_interval == 0:
                    n = acc_count
                    diag_days.append(abs_day)
                    diag_T_soil.append(T_soil_acc / n)
                    diag_theta_soil.append(theta_soil_acc / n)
                    diag_psi_soil.append(psi_soil_acc / n)
                    diag_snow_depth.append(snow_depth_acc / n)
                    diag_shflx.append(shflx_acc / n)
                    diag_lhflx.append(lhflx_acc / n)
                    diag_runoff_sfc.append(runoff_sfc_acc / n)
                    diag_runoff_sub.append(runoff_sub_acc / n)

                # Reset accumulators
                T_soil_acc[:] = 0.0
                theta_soil_acc[:] = 0.0
                psi_soil_acc[:] = 0.0
                snow_depth_acc = 0.0
                shflx_acc = 0.0
                lhflx_acc = 0.0
                runoff_sfc_acc = 0.0
                runoff_sub_acc = 0.0
                acc_count = 0

                # Checkpoint: flush diagnostics + save restart
                if day_idx % args.checkpoint_days == 0:
                    _flush_buffered_diagnostics()

                    restart_path = out_dir / f"restart_day{int(abs_day):06d}.npz"
                    _save_restart(
                        restart_path, global_step + 1, abs_day, state,
                        carbon_state,
                    )
                    elapsed = time.time() - t_wall_start
                    print(
                        f"  day {abs_day:7.1f} / {args.days:d}  "
                        f"T_sfc={float(np.asarray(state.T_soil[0, 0])):.2f} K  "
                        f"theta={float(np.asarray(state.theta_soil[0, 0])):.3f}  "
                        f"snow={float(np.asarray(state.snow_depth[0])):.3f} kg/m2  "
                        f"wall={elapsed:.0f}s",
                        flush=True,
                    )

                if _wallclock_exhausted(
                    time.time() - t_wall_start,
                    run_config.max_wallclock_seconds,
                    run_config.restart_buffer_seconds,
                ):
                    _flush_buffered_diagnostics()
                    restart_path = out_dir / f"restart_day{int(abs_day):06d}.npz"
                    _save_restart(
                        restart_path, global_step + 1, abs_day, state,
                        carbon_state,
                    )
                    print(
                        f"  Wallclock budget "
                        f"{run_config.max_wallclock_seconds:.0f}s nearly "
                        f"reached at day {abs_day:.2f}; restart saved: "
                        f"{restart_path.name}.",
                        flush=True,
                    )
                    sys.exit(0)

        # --- Flush remaining diagnostics ---
        _flush_buffered_diagnostics()

        # --- Validation: check for NaN in final state ---
        final_T = np.asarray(state.T_soil)
        final_theta = np.asarray(state.theta_soil)
        if np.any(np.isnan(final_T)):
            status = "FAIL"
            error_msg = "NaN in final T_soil"
        elif np.any(np.isnan(final_theta)):
            status = "FAIL"
            error_msg = "NaN in final theta_soil"
        elif np.any(final_T < 150.0) or np.any(final_T > 380.0):
            status = "FAIL"
            error_msg = f"T_soil out of physical range: [{final_T.min():.1f}, {final_T.max():.1f}] K"
        elif np.any(final_theta < 0.0):
            status = "FAIL"
            error_msg = f"theta_soil negative: min={final_theta.min():.4f}"

    except Exception:  # noqa: BLE001
        status = "FAIL"
        error_msg = traceback.format_exc()
        print(f"\nERROR:\n{error_msg}", file=sys.stderr)

    # --- Save final restart ---
    final_restart = out_dir / "restart_final.npz"
    _save_restart(final_restart, start_step + n_total_steps,
                  start_day_abs + args.days, state, carbon_state)

    # --- Summary JSON ---
    wall_time = time.time() - t_wall_start
    final_T_arr = np.asarray(state.T_soil)
    final_theta_arr = np.asarray(state.theta_soil)
    summary = {
        "status": status,
        "error": error_msg,
        "wall_time_s": round(wall_time, 1),
        "lat": args.lat,
        "lon": args.lon,
        "days": args.days,
        "dt_s": dt,
        "soil_texture": args.soil_texture,
        "veg_type": args.veg_type,
        "n_layers": args.n_layers,
        "soil_depth_m": args.soil_depth,
        "bulk_scheme": args.bulk_scheme,
        "seed": run_config.seed,
        "Cd_land": config.Cd_land,
        "Ch_land": config.Ch_land,
        "z0_land": config.z0_land,
        "beta_min": config.beta_min,
        "snow_albedo_feedback": config.snow_albedo_feedback,
        "carbon_scheme": config.carbon.scheme,
        "t_init_K": args.t_init,
        "start_day": args.start_day,
        "final_T_soil_K": final_T_arr[0].tolist(),
        "final_theta_soil": final_theta_arr[0].tolist(),
        "final_snow_depth_kg_m2": float(np.asarray(state.snow_depth[0])),
        "output_nc": str(nc_path),
        "final_restart": str(final_restart),
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStatus : {status}")
    if error_msg:
        print(f"Error  : {error_msg}")
    print(f"Runtime: {wall_time:.1f} s")
    print(f"Output : {out_dir}/")
    print(f"  land_spinup.nc  — daily diagnostics")
    print(f"  results.json    — run summary")
    print(f"  restart_final.npz")

    if status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
