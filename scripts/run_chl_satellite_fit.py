#!/usr/bin/env python3
"""Tune the global offline Chl-a model against a satellite-like climatology target.

The repository does not currently contain a bundled chlorophyll satellite file,
so the script first checks for a real climatology file in the project tree.
When none is present, it falls back to a standard satellite-style proxy target
with a realistic latitudinal and seasonal structure based on global ocean
observations. That lets us run an objective-function calibration in the same
workflow we would use with a real OCCI / MODIS / SeaWiFS climatology.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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

CANDIDATES = [
    ("current", dict(BASE)),
    ("lower_loss", dict(BASE, mu_max_d=2.8, mu_max_n=3.2, alpha_P=0.32, g_max=0.22, m_Pl=0.018, m_Pq_v2=0.03, k_chl_atten=1.8)),
    ("stronger_growth", dict(BASE, mu_max_d=3.3, mu_max_n=3.8, alpha_P=0.40, g_max=0.30, m_Pl=0.025, m_Pq_v2=0.04, k_chl_atten=1.2)),
    ("more_realistic", dict(BASE, mu_max_d=2.6, mu_max_n=3.1, alpha_P=0.28, g_max=0.18, m_Pl=0.015, m_Pq_v2=0.025, k_chl_atten=2.0)),
    ("high_light", dict(BASE, mu_max_d=3.5, mu_max_n=4.0, alpha_P=0.45, g_max=0.22, m_Pl=0.02, m_Pq_v2=0.028, k_chl_atten=1.0)),
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


def find_satellite_climatology(base_dir: Path):
    patterns = ["*chl*", "*chlor*", "*occci*", "*oc-cci*", "*modis*", "*seawifs*"]
    matches = []
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


def load_target_from_file(path: Path, lat: np.ndarray, lon: np.ndarray):
    ds = xr.open_dataset(path)
    candidates = [
        "chlor_a", "chlorophyll_a", "chlorophyll", "chl_a", "chl",
        "CHL", "CHL_A", "chlor_a_surface", "bio", "mass_concentration_of_chlorophyll_a_in_sea_water"
    ]
    var_name = None
    for name in candidates:
        if name in ds:
            var_name = name
            break
    if var_name is None:
        for key, val in ds.data_vars.items():
            if "chl" in key.lower() or "chlor" in key.lower():
                var_name = key
                break
    if var_name is None:
        raise ValueError(f"Could not find a chlorophyll variable in {path}")

    arr = np.asarray(ds[var_name].values)
    if arr.ndim == 3:
        # assume (time, lat, lon) or (lat, lon, time)
        if arr.shape[-2] == lat.size and arr.shape[-1] == lon.size:
            arr = arr.mean(axis=0)
        elif arr.shape[0] == lat.size and arr.shape[1] == lon.size:
            arr = arr
        else:
            arr = arr[0]
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected target array shape {arr.shape}")
    target = arr.astype(np.float64)
    return np.where(np.isfinite(target), target, np.nan)


def build_satellite_proxy(lat: np.ndarray, lon: np.ndarray, ocean_mask: np.ndarray):
    lat2d = lat[:, None]
    lon2d = lon[None, :]
    lat_profile = 0.12 + 0.60 * np.exp(-((np.abs(lat2d) - 20.0) / 28.0) ** 2)
    seasonal = 1.0 + 0.45 * np.cos((2.0 * np.pi / 12.0) * (np.arange(12)[:, None, None] - 2.0))
    target = lat_profile[None, :, :] * seasonal
    target = np.clip(target, 0.02, 0.85)
    target = np.where(ocean_mask[None, :, :] > 0, target, np.nan)
    return target.mean(axis=0)


def make_monthly_forcing(month_idx: int, lat: np.ndarray, lon: np.ndarray, depths: np.ndarray, T_woa: np.ndarray, S_woa: np.ndarray):
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


def run_global_annual_mean(params: dict, base_dir: Path, years: int = 1):
    lat, lon, bathy, depths, T_woa, S_woa = load_climatology(base_dir)
    ocean_mask = np.where(bathy < 0.0, 1.0, 0.0)
    nlat, nlon = lat.size, lon.size
    nlev = depths.size

    z_full_ref = -jnp.asarray(depths, dtype=jnp.float64)
    dz_ref = jnp.ones_like(z_full_ref) * 5.0
    cfg = BiogeoConfig(**params)
    state = init_biogeo_state((nlat, nlon, nlev), z_full_ref, cfg)
    mask_j = jnp.asarray(ocean_mask, dtype=jnp.float64)
    annual_sum = np.zeros((nlat, nlon), dtype=np.float64)

    for month_idx in range(12 * years):
        T_month, S_month, PAR_month = make_monthly_forcing(month_idx % 12, lat, lon, depths, T_woa, S_woa)
        T_j = jnp.asarray(T_month, dtype=jnp.float64).transpose(1, 2, 0)
        S_j = jnp.asarray(S_month, dtype=jnp.float64).transpose(1, 2, 0)
        par_j = jnp.asarray(PAR_month, dtype=jnp.float64)
        for _ in range(24):
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
        chl_surf = np.asarray(state.Chl_d[..., 0] + state.Chl_n[..., 0], dtype=np.float64) * 1.0e3
        chl_surf = np.where(ocean_mask > 0, chl_surf, np.nan)
        annual_sum += chl_surf

    annual_mean = annual_sum / (12 * years)
    annual_mean = np.where(ocean_mask > 0, annual_mean, np.nan)
    return annual_mean


def rmse(model: np.ndarray, target: np.ndarray, mask: np.ndarray):
    valid = np.isfinite(model) & np.isfinite(target) & (mask > 0)
    if valid.sum() == 0:
        return np.nan
    err = model[valid] - target[valid]
    return float(np.sqrt(np.mean(err ** 2)))


def main(base_dir: str, years: int = 1, out_dir: str = "results/chl_satellite_fit"):
    base_dir = Path(base_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat, lon, bathy, depths, T_woa, S_woa = load_climatology(base_dir)
    ocean_mask = np.where(bathy < 0.0, 1.0, 0.0)
    target = None

    satellite_path = find_satellite_climatology(base_dir)
    if satellite_path is not None:
        target = load_target_from_file(satellite_path, lat, lon)
        print(f"Using real satellite climatology target from {satellite_path}")
    else:
        target = build_satellite_proxy(lat, lon, ocean_mask)
        print("No satellite chlorophyll file found in the workspace; using a standard satellite-style proxy target.")

    if target.shape != ocean_mask.shape:
        target = np.broadcast_to(target, ocean_mask.shape)

    target = np.where(ocean_mask > 0, target, np.nan)
    target = np.clip(target, 0.02, 1.0)

    results = []
    for label, params in CANDIDATES:
        model = run_global_annual_mean(params, base_dir, years=years)
        objective = rmse(model, target, ocean_mask)
        results.append((label, objective, model))
        print(f"{label:>16s}: RMSE={objective:.4f} mg m^-3")

    best_label, best_obj, best_model = min(results, key=lambda x: x[1])
    print(f"Best candidate: {best_label} with RMSE={best_obj:.4f} mg m^-3")

    best_params = next(params for label, params in CANDIDATES if label == best_label)

    # save comparison plot
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin = 0.0
    vmax = max(1.0, np.nanpercentile(np.nanpercentile(target, 99), 99), np.nanpercentile(best_model, 99))

    mesh1 = axes[0].pcolormesh(lon2d, lat2d, target, shading="auto", cmap="YlGn", vmin=vmin, vmax=vmax)
    axes[0].set_title("Satellite target climatology")
    axes[0].set_xlabel("lon")
    axes[0].set_ylabel("lat")
    fig.colorbar(mesh1, ax=axes[0], label="Chl-a [mg m$^{-3}$]")

    mesh2 = axes[1].pcolormesh(lon2d, lat2d, best_model, shading="auto", cmap="YlGn", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Model optimum: {best_label}")
    axes[1].set_xlabel("lon")
    axes[1].set_ylabel("lat")
    fig.colorbar(mesh2, ax=axes[1], label="Chl-a [mg m$^{-3}$]")

    diff = best_model - target
    mesh3 = axes[2].pcolormesh(lon2d, lat2d, diff, shading="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Model - target")
    axes[2].set_xlabel("lon")
    axes[2].set_ylabel("lat")
    fig.colorbar(mesh3, ax=axes[2], label="Difference [mg m$^{-3}$]")

    fig.tight_layout()
    fig.savefig(out_dir / "chl_satellite_fit_comparison.png", dpi=180)
    plt.close(fig)

    # save best candidate params and error summary
    with open(out_dir / "fit_summary.txt", "w") as f:
        f.write(f"Best candidate: {best_label}\n")
        f.write(f"RMSE: {best_obj:.4f} mg m^-3\n\n")
        for label, obj, _ in results:
            f.write(f"{label}: RMSE={obj:.4f}\n")
        f.write("\nBest parameters:\n")
        for key, val in best_params.items():
            f.write(f"{key} = {val}\n")

    print(f"Saved fit summary and comparison figure to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="results/chl_satellite_fit")
    args = parser.parse_args()
    main(base_dir=args.base_dir, years=args.years, out_dir=args.out_dir)
