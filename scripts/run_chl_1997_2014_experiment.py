#!/usr/bin/env python3
"""Run a spin-up + calibrated offline Chl experiment for 1997-2014.

This script performs:
1) Spin-up under monthly climatological forcing.
2) Candidate-parameter calibration against a satellite target.
3) Final simulation and diagnostics for 1997-2014.

Outputs:
- Annual-mean model climatology map
- Annual-mean target map
- Annual-mean bias map (model - target)
- Monthly climatologies (model, target, bias) in NetCDF
- Seasonal cycle figure (global-mean monthly cycle)
- Summary text with fitted candidate and metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.biogeochemistry import BiogeoConfig, init_biogeo_state, step_ocean_biogeochemistry


BASE = dict(
    scheme="npzd_v2",
    pCO2_atm=400.0,
    wind_speed=7.0,
    mu_max_d=3.0,
    mu_max_n=3.5,
    k_N_d=5.0e-5,
    k_N_n=7.5e-5,
    k_NH4_d=1.0e-4,
    k_NH4_n=1.0e-4,
    k_Fe_d=90.0e-12,
    k_Fe_n=60.0e-12,
    k_Si=0.5e-3,
    alpha_P=0.35,
    g_max=0.25,
    k_P=0.25e-3,
    gamma_Z=0.68,
    epse=0.5,
    m_Pl=0.02,
    m_Pq_v2=0.036,
    m_Zl_v2=0.05,
    m_Z=0.08,
    remin_rate=0.04,
    remin_opal=0.03,
    remin_DON=0.015,
    k_w_atten=0.04,
    k_chl_atten=1.5,
    w_sink=7.0,
    theta_min_d=0.7e-3,
    theta_max_d=4.1e-3,
    theta_min_n=0.7e-3,
    theta_max_n=3.8e-3,
    Fe_ligand=180.0e-12,
    Fe_dust=5.0e-9,
    k_Fe_scav=2.5,
    NO3_init_surf=8.0e-3,
    NO3_init_deep=18.0e-3,
    NH4_init=1.0e-4,
    Fe_init_surf=30.0e-12,
    Fe_init_deep=120.0e-12,
    Si_init_surf=0.7e-3,
    Si_init_deep=8.0e-3,
    Phyto_init=0.03e-3,
    Zoo_init=0.05e-3,
    Det_init=0.02e-3,
    DIC_init=2.1,
    ALK_init=2.3,
)

CANDIDATES: List[Tuple[str, Dict[str, float]]] = [
    ("current", dict(BASE)),
    (
        "lower_loss",
        dict(
            BASE,
            mu_max_d=2.8,
            mu_max_n=3.2,
            alpha_P=0.32,
            g_max=0.22,
            m_Pl=0.018,
            m_Pq_v2=0.03,
            k_chl_atten=1.8,
        ),
    ),
    (
        "stronger_growth",
        dict(
            BASE,
            mu_max_d=3.3,
            mu_max_n=3.8,
            alpha_P=0.40,
            g_max=0.30,
            m_Pl=0.025,
            m_Pq_v2=0.04,
            k_chl_atten=1.2,
        ),
    ),
    (
        "more_realistic",
        dict(
            BASE,
            mu_max_d=2.6,
            mu_max_n=3.1,
            alpha_P=0.28,
            g_max=0.18,
            m_Pl=0.015,
            m_Pq_v2=0.025,
            k_chl_atten=2.0,
        ),
    ),
    (
        "high_light",
        dict(
            BASE,
            mu_max_d=3.5,
            mu_max_n=4.0,
            alpha_P=0.45,
            g_max=0.22,
            m_Pl=0.02,
            m_Pq_v2=0.028,
            k_chl_atten=1.0,
        ),
    ),
]


def load_climatology(base_dir: Path):
    bathy = xr.open_dataset(base_dir / "bathy_from_woa.nc")
    temp = xr.open_dataset(base_dir / "woa18_t_5deg.nc")
    salt = xr.open_dataset(base_dir / "woa18_s_5deg.nc")

    lat = np.asarray(bathy.lat.values, dtype=np.float64)
    lon = np.asarray(bathy.lon.values, dtype=np.float64)
    T_woa = np.asarray(temp.t_an.values[0], dtype=np.float64)
    S_woa = np.asarray(salt.s_an.values[0], dtype=np.float64)
    depths = np.asarray(temp.depth.values, dtype=np.float64)
    bathy_z = np.asarray(bathy.z.values, dtype=np.float64)
    return lat, lon, bathy_z, depths, T_woa, S_woa


def make_monthly_forcing(
    month_idx: int,
    lat: np.ndarray,
    lon: np.ndarray,
    depths: np.ndarray,
    T_woa: np.ndarray,
    S_woa: np.ndarray,
):
    lat_2d = lat[:, None]
    phase = 2.0 * np.pi * month_idx / 12.0
    seasonal_amp = 1.5 * np.cos(phase - np.pi / 2.0)
    seasonal_amp = np.clip(seasonal_amp, -1.0, 1.5)

    T = T_woa.copy()
    S = S_woa.copy()
    for d_idx, depth in enumerate(depths):
        depth_factor = np.exp(-depth / 200.0)
        T[d_idx] = T[d_idx] + seasonal_amp * depth_factor * (1.0 + 0.25 * np.cos(np.deg2rad(lat_2d)))
        S[d_idx] = S[d_idx] + 0.05 * seasonal_amp * depth_factor

    par = 150.0 + 120.0 * (1.0 + np.cos(phase - np.pi / 2.0)) / 2.0
    par_field = np.full((lat.size, lon.size), par, dtype=np.float64)
    par_field *= (0.75 + 0.35 * np.cos(np.deg2rad(lat[:, None])))
    return T, S, par_field


def find_satellite_climatology(base_dir: Path) -> Optional[Path]:
    patterns = ["*chl*", "*chlor*", "*occci*", "*oc-cci*", "*modis*", "*seawifs*"]
    matches: List[Path] = []
    for pat in patterns:
        matches.extend(base_dir.rglob(pat))
    for path in matches:
        if not path.is_file():
            continue
        if "results" in path.parts:
            continue
        if path.suffix.lower() in {".nc", ".nc4", ".cdf", ".grb", ".grib"}:
            return path
    return None


def _find_chl_variable(ds: xr.Dataset) -> str:
    candidates = [
        "chlor_a",
        "chlorophyll_a",
        "chlorophyll",
        "chl_a",
        "chl",
        "CHL",
        "CHL_A",
        "chlor_a_surface",
        "bio",
        "mass_concentration_of_chlorophyll_a_in_sea_water",
    ]
    for name in candidates:
        if name in ds:
            return name
    for key in ds.data_vars:
        low = key.lower()
        if "chl" in low or "chlor" in low:
            return key
    raise ValueError("Could not find a chlorophyll variable in the provided file")


def _monthly_clim_from_array(arr: np.ndarray, nlat: int, nlon: int) -> np.ndarray:
    if arr.ndim == 2:
        field = arr.astype(np.float64)
        return np.broadcast_to(field[None, :, :], (12, nlat, nlon)).copy()

    if arr.ndim != 3:
        raise ValueError(f"Unexpected target array shape {arr.shape}; expected 2D or 3D")

    # Candidate layouts: (time, lat, lon), (lat, lon, time), (lat, time, lon)
    if arr.shape[1] == nlat and arr.shape[2] == nlon:
        time_series = arr
    elif arr.shape[0] == nlat and arr.shape[1] == nlon:
        time_series = np.moveaxis(arr, -1, 0)
    elif arr.shape[0] == nlat and arr.shape[2] == nlon:
        time_series = np.moveaxis(arr, 1, 0)
    else:
        raise ValueError(f"Could not map array shape {arr.shape} to lat/lon grid ({nlat}, {nlon})")

    nt = time_series.shape[0]
    if nt == 12:
        return time_series.astype(np.float64)

    monthly = np.zeros((12, nlat, nlon), dtype=np.float64)
    counts = np.zeros(12, dtype=np.int32)
    for t in range(nt):
        m = t % 12
        monthly[m] += np.asarray(time_series[t], dtype=np.float64)
        counts[m] += 1
    for m in range(12):
        if counts[m] > 0:
            monthly[m] /= counts[m]
    return monthly


def load_target_monthly(path: Path, lat: np.ndarray, lon: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    ds = xr.open_dataset(path)
    var_name = _find_chl_variable(ds)
    arr = np.asarray(ds[var_name].values)
    monthly = _monthly_clim_from_array(arr, lat.size, lon.size)
    monthly = np.where(np.isfinite(monthly), monthly, np.nan)
    monthly = np.where(ocean_mask[None, :, :] > 0, monthly, np.nan)
    return np.clip(monthly, 0.0, None)


def build_satellite_proxy_monthly(lat: np.ndarray, lon: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    lat2d = lat[:, None]
    lon2d = lon[None, :]
    lat_profile = 0.12 + 0.60 * np.exp(-((np.abs(lat2d) - 20.0) / 28.0) ** 2)
    seasonal = 1.0 + 0.45 * np.cos((2.0 * np.pi / 12.0) * (np.arange(12)[:, None, None] - 2.0))
    target = lat_profile[None, :, :] * seasonal
    target = target * (1.0 + 0.08 * np.cos(np.deg2rad(lon2d))[None, :, :])
    target = np.clip(target, 0.02, 1.0)
    target = np.where(ocean_mask[None, :, :] > 0, target, np.nan)
    return target


def area_weights(lat: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(ocean_mask)
    w = np.where(ocean_mask > 0, w, 0.0)
    return w


def weighted_mean(field: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(field)
    ww = np.where(valid, weights, 0.0)
    denom = float(np.sum(ww))
    if denom <= 0.0:
        return float("nan")
    return float(np.nansum(field * ww) / denom)


def spatial_rmse(model: np.ndarray, target: np.ndarray, ocean_mask: np.ndarray) -> float:
    valid = np.isfinite(model) & np.isfinite(target) & (ocean_mask > 0)
    if valid.sum() == 0:
        return float("nan")
    err = model[valid] - target[valid]
    return float(np.sqrt(np.mean(err ** 2)))


def run_model(
    params: Dict[str, float],
    lat: np.ndarray,
    lon: np.ndarray,
    depths: np.ndarray,
    T_woa: np.ndarray,
    S_woa: np.ndarray,
    ocean_mask: np.ndarray,
    spinup_years: int,
    eval_years: int,
    n_substeps: int,
) -> np.ndarray:
    nlat, nlon = lat.size, lon.size
    nlev = depths.size

    z_full_ref = -jnp.asarray(depths, dtype=jnp.float64)
    dz_ref = jnp.ones_like(z_full_ref) * 5.0
    cfg = BiogeoConfig(**params)
    state = init_biogeo_state((nlat, nlon, nlev), z_full_ref, cfg)
    mask_j = jnp.asarray(ocean_mask, dtype=jnp.float64)

    total_months = 12 * (spinup_years + eval_years)
    eval_start = 12 * spinup_years

    monthly_acc = np.zeros((12, nlat, nlon), dtype=np.float64)
    monthly_count = np.zeros(12, dtype=np.int32)

    for m in range(total_months):
        month_in_year = m % 12
        T_month, S_month, PAR_month = make_monthly_forcing(month_in_year, lat, lon, depths, T_woa, S_woa)
        T_j = jnp.asarray(T_month, dtype=jnp.float64).transpose(1, 2, 0)
        S_j = jnp.asarray(S_month, dtype=jnp.float64).transpose(1, 2, 0)
        par_j = jnp.asarray(PAR_month, dtype=jnp.float64)

        for _ in range(n_substeps):
            state, _ = step_ocean_biogeochemistry(
                state,
                T_j,
                S_j,
                dz_ref,
                z_full_ref,
                mask_j,
                6.0 * 3600.0,
                cfg,
                PAR_surf=par_j,
            )

        if m >= eval_start:
            chl_surf = np.asarray(state.Chl_d[..., 0] + state.Chl_n[..., 0], dtype=np.float64) * 1.0e3
            chl_surf = np.where(ocean_mask > 0, chl_surf, np.nan)
            monthly_acc[month_in_year] += chl_surf
            monthly_count[month_in_year] += 1

    for month in range(12):
        if monthly_count[month] > 0:
            monthly_acc[month] /= monthly_count[month]

    return monthly_acc


def objective_metrics(
    model_monthly: np.ndarray,
    target_monthly: np.ndarray,
    ocean_mask: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float]:
    model_ann = np.nanmean(model_monthly, axis=0)
    target_ann = np.nanmean(target_monthly, axis=0)
    rmse_ann = spatial_rmse(model_ann, target_ann, ocean_mask)

    model_cycle = np.asarray([weighted_mean(model_monthly[m], weights) for m in range(12)], dtype=np.float64)
    target_cycle = np.asarray([weighted_mean(target_monthly[m], weights) for m in range(12)], dtype=np.float64)
    cycle_rmse = float(np.sqrt(np.nanmean((model_cycle - target_cycle) ** 2)))

    total = float(rmse_ann + 0.5 * cycle_rmse)
    return total, rmse_ann, cycle_rmse


def save_outputs(
    out_dir: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    model_monthly: np.ndarray,
    target_monthly: np.ndarray,
    best_label: str,
    rmse_ann: float,
    cycle_rmse: float,
    start_year: int,
    end_year: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    model_ann = np.nanmean(model_monthly, axis=0)
    target_ann = np.nanmean(target_monthly, axis=0)
    bias_ann = model_ann - target_ann
    monthly_bias = model_monthly - target_monthly

    xr.Dataset(
        {
            "model_chl_a_monthly_climatology": (("month", "lat", "lon"), model_monthly),
            "target_chl_a_monthly_climatology": (("month", "lat", "lon"), target_monthly),
            "bias_chl_a_monthly_climatology": (("month", "lat", "lon"), monthly_bias),
            "model_chl_a_annual_mean": (("lat", "lon"), model_ann),
            "target_chl_a_annual_mean": (("lat", "lon"), target_ann),
            "bias_chl_a_annual_mean": (("lat", "lon"), bias_ann),
            "ocean_mask": (("lat", "lon"), ocean_mask),
        },
        coords={"month": np.arange(1, 13, dtype=np.int32), "lat": lat, "lon": lon},
    ).to_netcdf(out_dir / "chl_1997_2014_diagnostics.nc")

    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    valid_chl = np.isfinite(model_ann) & np.isfinite(target_ann)
    if np.any(valid_chl):
        vmax = float(np.nanpercentile(np.concatenate([model_ann[valid_chl], target_ann[valid_chl]]), 99))
    else:
        vmax = 1.0
    vmax = max(1.0, vmax)

    valid_bias = np.isfinite(bias_ann)
    if np.any(valid_bias):
        bias_lim = float(np.nanpercentile(np.abs(bias_ann[valid_bias]), 99))
    else:
        bias_lim = 0.6
    bias_lim = max(0.6, bias_lim)
    m0 = axes[0].pcolormesh(lon2d, lat2d, model_ann, shading="auto", cmap="YlGn", vmin=0.0, vmax=vmax)
    axes[0].set_title("Model Chl climatology")
    axes[0].set_xlabel("lon")
    axes[0].set_ylabel("lat")
    fig.colorbar(m0, ax=axes[0], label="mg m$^{-3}$")

    m1 = axes[1].pcolormesh(lon2d, lat2d, target_ann, shading="auto", cmap="YlGn", vmin=0.0, vmax=vmax)
    axes[1].set_title("Satellite Chl climatology")
    axes[1].set_xlabel("lon")
    axes[1].set_ylabel("lat")
    fig.colorbar(m1, ax=axes[1], label="mg m$^{-3}$")

    m2 = axes[2].pcolormesh(lon2d, lat2d, bias_ann, shading="auto", cmap="RdBu_r", vmin=-bias_lim, vmax=bias_lim)
    axes[2].set_title("Bias (model - satellite)")
    axes[2].set_xlabel("lon")
    axes[2].set_ylabel("lat")
    fig.colorbar(m2, ax=axes[2], label="mg m$^{-3}$")

    fig.tight_layout()
    fig.savefig(out_dir / "chl_climatology_and_bias.png", dpi=180)
    plt.close(fig)

    weights = area_weights(lat, ocean_mask)
    model_cycle = np.asarray([weighted_mean(model_monthly[m], weights) for m in range(12)], dtype=np.float64)
    target_cycle = np.asarray([weighted_mean(target_monthly[m], weights) for m in range(12)], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    months = np.arange(1, 13)
    ax.plot(months, model_cycle, marker="o", label="model")
    ax.plot(months, target_cycle, marker="s", linestyle="--", label="satellite")
    ax.plot(months, model_cycle - target_cycle, marker="^", linestyle=":", label="bias")
    ax.set_xlabel("month")
    ax.set_ylabel("global mean Chl-a [mg m$^{-3}$]")
    ax.set_title("Seasonal cycle (1997-2014 climatology)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chl_seasonal_cycle_vs_satellite.png", dpi=180)
    plt.close(fig)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Period: {start_year}-{end_year}\n")
        f.write(f"Best candidate: {best_label}\n")
        f.write(f"Annual map RMSE: {rmse_ann:.4f} mg m^-3\n")
        f.write(f"Seasonal-cycle RMSE: {cycle_rmse:.4f} mg m^-3\n")
        f.write(f"Global annual mean model Chl: {np.nanmean(model_ann):.4f} mg m^-3\n")
        f.write(f"Global annual mean target Chl: {np.nanmean(target_ann):.4f} mg m^-3\n")
        f.write(f"Global annual mean bias: {np.nanmean(bias_ann):.4f} mg m^-3\n")


def main():
    parser = argparse.ArgumentParser(description="Spin-up + calibrated Chl run for 1997-2014")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--start-year", type=int, default=1997)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--spinup-years", type=int, default=5)
    parser.add_argument("--calibration-years", type=int, default=3)
    parser.add_argument("--substeps-per-month", type=int, default=24)
    parser.add_argument("--satellite", type=str, default=None, help="Path to satellite chlorophyll file (optional)")
    parser.add_argument("--out-dir", type=str, default="results/chl_1997_2014")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    lat, lon, bathy, depths, T_woa, S_woa = load_climatology(base_dir)
    ocean_mask = np.where(bathy < 0.0, 1.0, 0.0)

    sat_path: Optional[Path]
    if args.satellite:
        sat_path = Path(args.satellite)
        if not sat_path.exists():
            raise FileNotFoundError(f"Satellite file not found: {sat_path}")
    else:
        sat_path = find_satellite_climatology(base_dir)

    if sat_path is not None:
        target_monthly = load_target_monthly(sat_path, lat, lon, ocean_mask)
        print(f"Using satellite chlorophyll data from: {sat_path}")
    else:
        target_monthly = build_satellite_proxy_monthly(lat, lon, ocean_mask)
        print("No satellite chlorophyll file found; using satellite-like proxy target.")

    weights = area_weights(lat, ocean_mask)

    print("Calibrating candidates...")
    cal_results = []
    for label, params in CANDIDATES:
        model_monthly = run_model(
            params=params,
            lat=lat,
            lon=lon,
            depths=depths,
            T_woa=T_woa,
            S_woa=S_woa,
            ocean_mask=ocean_mask,
            spinup_years=args.spinup_years,
            eval_years=args.calibration_years,
            n_substeps=args.substeps_per_month,
        )
        score, rmse_ann, cycle_rmse = objective_metrics(model_monthly, target_monthly, ocean_mask, weights)
        cal_results.append((label, score, rmse_ann, cycle_rmse, params))
        print(
            f"{label:>16s}: score={score:.4f}, annual_rmse={rmse_ann:.4f}, "
            f"seasonal_rmse={cycle_rmse:.4f} mg m^-3"
        )

    best_label, best_score, best_rmse_ann, best_cycle_rmse, best_params = min(cal_results, key=lambda x: x[1])
    print(f"Best candidate: {best_label} (score={best_score:.4f})")

    period_years = args.end_year - args.start_year + 1
    print(f"Running final period simulation for {args.start_year}-{args.end_year} ({period_years} years)...")
    best_monthly = run_model(
        params=best_params,
        lat=lat,
        lon=lon,
        depths=depths,
        T_woa=T_woa,
        S_woa=S_woa,
        ocean_mask=ocean_mask,
        spinup_years=args.spinup_years,
        eval_years=period_years,
        n_substeps=args.substeps_per_month,
    )

    save_outputs(
        out_dir=out_dir,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask,
        model_monthly=best_monthly,
        target_monthly=target_monthly,
        best_label=best_label,
        rmse_ann=best_rmse_ann,
        cycle_rmse=best_cycle_rmse,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    with (out_dir / "fit_candidates.txt").open("w", encoding="utf-8") as f:
        for label, score, rmse_ann, cycle_rmse, _ in sorted(cal_results, key=lambda x: x[1]):
            f.write(
                f"{label}: score={score:.4f}, annual_rmse={rmse_ann:.4f}, "
                f"seasonal_rmse={cycle_rmse:.4f} mg m^-3\n"
            )

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
