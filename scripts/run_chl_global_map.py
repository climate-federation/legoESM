#!/usr/bin/env python3
"""Global offline 2D Chl-a map using real WOA climatology and bathymetry.

This is a pragmatic global surface-ocean diagnostic for the MARBL-lite
npzd_v2 BGC scheme. It uses the existing project bathymetry and WOA
temperature/salinity climatology, then applies a simple seasonal monthly
forcing to produce a zonal surface Chl-a map over one year.

It is intentionally lightweight and CPU-safe; it is meant to generate a
qualitative global map rather than a full coupled ocean simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.biogeochemistry import BiogeoConfig, init_biogeo_state, step_ocean_biogeochemistry


def load_climatology(base_dir: str):
    bathy = xr.open_dataset(Path(base_dir) / "bathy_from_woa.nc")
    temp = xr.open_dataset(Path(base_dir) / "woa18_t_5deg.nc")
    salt = xr.open_dataset(Path(base_dir) / "woa18_s_5deg.nc")

    z = np.asarray(bathy.z.values, dtype=np.float64)
    lat = np.asarray(bathy.lat.values, dtype=np.float64)
    lon = np.asarray(bathy.lon.values, dtype=np.float64)
    T_woa = np.asarray(temp.t_an.values[0], dtype=np.float64)   # (depth, lat, lon)
    S_woa = np.asarray(salt.s_an.values[0], dtype=np.float64)
    depths = np.asarray(temp.depth.values, dtype=np.float64)

    return lat, lon, z, depths, T_woa, S_woa


def make_monthly_forcing(month_idx: int, lat: np.ndarray, lon: np.ndarray, depths: np.ndarray, T_woa: np.ndarray, S_woa: np.ndarray):
    """Add a simple seasonal cycle on top of WOA climatology."""
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

    # surface PAR [W/m²]
    par = 150.0 + 120.0 * (1.0 + np.cos(phase - np.pi / 2.0)) / 2.0
    par_field = np.full((lat.size, lon.size), par, dtype=np.float64)
    par_field *= (0.75 + 0.35 * np.cos(np.deg2rad(lat[:, None])))

    return T, S, par_field


def build_cfg():
    # Updated to the best stable satellite-like candidate from the seasonal scan.
    return BiogeoConfig(
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


def run_global_map(base_dir: str = ".", years: int = 1, out_dir: str = "results/chl_global_map"):
    lat, lon, bathy, depths, T_woa, S_woa = load_climatology(base_dir)
    lat_j = jnp.asarray(lat, dtype=jnp.float64)
    lon_j = jnp.asarray(lon, dtype=jnp.float64)
    z_full_ref = -jnp.asarray(depths, dtype=jnp.float64)
    dz_ref = jnp.ones_like(z_full_ref) * 5.0
    dz_ref = jnp.asarray(dz_ref, dtype=jnp.float64)

    cfg = build_cfg()
    nlat, nlon = lat.size, lon.size
    nlev = depths.size
    ocean_mask = (bathy < 0.0).astype(np.float64)
    ocean_mask_3d = np.broadcast_to(ocean_mask[:, :, None], (nlat, nlon, nlev))

    state = init_biogeo_state((nlat, nlon, nlev), z_full_ref, cfg)
    # Zero out land cells to avoid spurious values in masked areas.
    state = state._replace(**{k: jnp.where(jnp.asarray(ocean_mask_3d, dtype=bool), v, 0.0) for k, v in state._asdict().items() if v is not None})

    month_series = []
    annual_mean = np.zeros((nlat, nlon), dtype=np.float64)
    seasonal_means = np.zeros((12, nlat, nlon), dtype=np.float64)
    step_dt = 6.0 * 3600.0

    # The tuned biology needs meaningful spin-up time before it settles into a
    # seasonal cycle. A handful of substeps per month is too short to build a
    # realistic Chl-a field, so we integrate multiple monthly cycles and average
    # only the final year of the transient.
    spinup_years = max(3, years)
    total_months = 12 * spinup_years
    for month_idx in range(total_months):
        T_month, S_month, PAR_month = make_monthly_forcing(month_idx % 12, lat, lon, depths, T_woa, S_woa)
        T_j = jnp.asarray(T_month, dtype=jnp.float64).transpose(1, 2, 0)
        S_j = jnp.asarray(S_month, dtype=jnp.float64).transpose(1, 2, 0)
        par_j = jnp.asarray(PAR_month, dtype=jnp.float64)
        mask_j = jnp.asarray(ocean_mask, dtype=jnp.float64)

        n_substeps = 30
        for _ in range(n_substeps):
            state, _ = step_ocean_biogeochemistry(
                state,
                T_j,
                S_j,
                dz_ref,
                z_full_ref,
                mask_j,
                step_dt,
                cfg,
                PAR_surf=par_j,
            )

        chl_surf = np.asarray(state.Chl_d[..., 0] + state.Chl_n[..., 0], dtype=np.float64) * 1.0e3
        chl_surf = np.where(ocean_mask > 0, chl_surf, np.nan)
        month_series.append(chl_surf)
        seasonal_means[month_idx % 12] = chl_surf

        if month_idx >= (total_months - 12):
            annual_mean += chl_surf

    annual_mean = annual_mean / 12.0
    annual_mean = np.where(ocean_mask > 0, annual_mean, np.nan)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # save annual mean map
    ds = xr.Dataset(
        {
            "chl_a_surface": (("lat", "lon"), annual_mean),
            "ocean_mask": (("lat", "lon"), ocean_mask),
        },
        coords={"lat": lat, "lon": lon},
    )
    ds.to_netcdf(out_dir_p / "chl_surface_annual_mean.nc")

    # save monthly fields
    xr.Dataset(
        {
            "chl_a_surface": (("month", "lat", "lon"), np.asarray(month_series, dtype=np.float64)),
        },
        coords={"month": np.arange(len(month_series)), "lat": lat, "lon": lon},
    ).to_netcdf(out_dir_p / "chl_surface_monthly.nc")

    # seasonal mean map: DJF, MAM, JJA, SON using available months
    seasonal_months = [(0, 1, 11), (2, 3, 4), (5, 6, 7), (8, 9, 10)]
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={"projection": None})
    for ax, months in zip(axes.ravel(), seasonal_months):
        data = np.nanmean(np.asarray([seasonal_means[m] for m in months]), axis=0)
        data = np.where(ocean_mask > 0, data, np.nan)
        vmax = max(1.0, np.nanpercentile(data, 99))
        cm = ax.pcolormesh(lon2d, lat2d, data, shading="auto", cmap="YlGn", vmin=0.0, vmax=vmax)
        ax.set_title(f"Seasonal mean ({months[0]+1}/{months[1]+1}/{months[2]+1})")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        fig.colorbar(cm, ax=ax, label="Chl-a [mg m$^{-3}$]")
    fig.tight_layout()
    fig.savefig(out_dir_p / "chl_surface_seasonal_means.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    annual_clean = np.where(ocean_mask > 0, annual_mean, np.nan)
    vmax = max(1.0, np.nanpercentile(annual_clean, 99))
    m = ax.pcolormesh(lon2d, lat2d, annual_clean, shading="auto", cmap="YlGn", vmin=0.0, vmax=vmax)
    ax.set_title("Annual mean surface Chl-a (global offline 2D map)")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    fig.colorbar(m, ax=ax, label="Chl-a [mg m$^{-3}$]")
    fig.tight_layout()
    fig.savefig(out_dir_p / "chl_surface_annual_mean.png", dpi=180)
    plt.close(fig)

    print(f"Saved global Chl-a map outputs to {out_dir_p}")
    print(f"Annual mean surface Chl-a: {np.nanmean(annual_mean):.3f} mg m^-3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="results/chl_global_map")
    args = parser.parse_args()
    run_global_map(base_dir=args.base_dir, years=args.years, out_dir=args.out_dir)
