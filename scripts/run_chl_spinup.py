#!/usr/bin/env python3
"""Offline 1-degree global Chl-a spin-up prototype for legoESM.

This script is intentionally compact and CPU-safe. It uses the existing
MARBL-like NPZD-v2 BGC core from the project, but it does not run a full
coupled ocean simulation; instead it advances the biogeochemistry in an
offline, global 1-degree climatological setup for a short spin-up period.

Output:
- monthly means of surface Chl-a (C1_d + Chl_n)
- global annual mean plot
- seasonal cycle plot
- optional comparison plot against a satellite reference file, if provided

Usage:
  JAX_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python scripts/run_chl_spinup.py --months 12 --dt-hours 6
  JAX_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python scripts/run_chl_spinup.py --months 12 --satellite path/to/chl.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.biogeochemistry import BiogeoConfig, init_biogeo_state, step_ocean_biogeochemistry
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid


def analytical_woa_from_latlon(lat_deg: np.ndarray, z_full_ref: np.ndarray):
    """Return a simple temperature/salinity climatology from latitude and depth."""
    lat = np.asarray(lat_deg)
    z = np.asarray(z_full_ref)
    z_abs = np.abs(z)

    T = 28.0 * np.cos(np.deg2rad(lat))[:, None] ** 2
    T = T[..., None] * np.exp(-z_abs[None, None, :] / 500.0) + 1.5
    T = T - 1.0 * np.exp(-z_abs[None, None, :] / 120.0)
    T = np.clip(T, -2.0, 30.0)

    S = 34.5 + 1.0 * np.cos(np.deg2rad(2.0 * lat))[:, None] ** 2
    S = S[..., None] + (34.7 - 34.5) * np.exp(-z_abs[None, None, :] / 1000.0)
    S = np.clip(S, 33.0, 37.0)
    return T, S


def build_monthly_forcing(month_index: int, n_lat: int, n_lon: int, z_full_ref: np.ndarray):
    """Return monthly T,S,PAR fields for a simple climatological cycle."""
    lat = np.linspace(-89.5, 89.5, n_lat)
    T, S = analytical_woa_from_latlon(lat, z_full_ref)

    # Broad seasonal cycle in surface insolation; weak variations in T/S.
    phase = 2.0 * np.pi * month_index / 12.0
    seasonal_amp = 0.25 + 0.25 * np.cos(phase)
    T = T + seasonal_amp * np.exp(-np.abs(z_full_ref)[None, None, :] / 200.0)
    T = np.clip(T, -1.5, 30.0)

    par = 150.0 + 110.0 * (1.0 + np.cos(phase - np.pi / 2.0)) / 2.0
    par_field = np.full((n_lat, n_lon), par, dtype=np.float64)
    return T, S, par_field


def global_mean_1d(x):
    if x.size == 0:
        return np.nan
    return float(np.nanmean(x))


def try_load_satellite(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        ds = xr.open_dataset(p)
        return ds
    except Exception:
        return None


def make_reference_climatology(months: int):
    """Build a satellite-like reference climatology if no real product is supplied."""
    seasonal = np.array([
        0.55, 0.62, 0.74, 0.88, 1.02, 1.12,
        1.15, 1.05, 0.94, 0.78, 0.63, 0.56,
    ], dtype=np.float64)
    ref = np.interp(np.arange(months), np.arange(12), seasonal)
    ref = ref / ref.mean()
    return ref


def build_tuned_bgc_config() -> BiogeoConfig:
    """Return a calmer MARBL-lite configuration suitable for a longer Chl-a spin-up."""
    return BiogeoConfig(
        scheme="npzd_v2",
        pCO2_atm=400.0,
        wind_speed=7.0,
        mu_max=0.9,
        k_N=0.7e-3,
        alpha_P=0.08,
        g_max=0.35,
        k_P=0.15e-3,
        gamma_Z=0.65,
        m_P=0.04,
        m_Pq=0.12,
        m_Zl=0.04,
        m_Z=0.10,
        remin_rate=0.035,
        w_sink=8.0,
        k_w_atten=0.04,
        k_chl_atten=10.0,
        NO3_init_surf=5.0e-3,
        NO3_init_deep=20.0e-3,
        Phyto_init=0.03e-3,
        Zoo_init=0.05e-3,
        Det_init=0.02e-3,
        DIC_init=2.1,
        ALK_init=2.3,
        mu_max_d=0.65,
        k_N_d=2.0e-4,
        k_NH4_d=1.0e-4,
        k_Fe_d=80.0e-12,
        k_Si=0.5e-3,
        R_SiN=1.0,
        R_FeN_d=7.5e-6,
        theta_min_d=0.5e-3,
        theta_max_d=4.0e-3,
        mu_max_n=0.52,
        k_N_n=2.5e-4,
        k_NH4_n=1.5e-4,
        k_Fe_n=60.0e-12,
        R_FeN_n=5.0e-6,
        theta_min_n=0.4e-3,
        theta_max_n=3.5e-3,
        m_Pl=0.05,
        m_Pq_v2=0.10,
        psi_coeff=1.5e3,
        k_nitrif=0.12,
        kI_nitrif=0.0025,
        m_Zl_v2=0.08,
        remin_opal=0.03,
        remin_DON=0.015,
        k_Fe_scav=2.5,
        Fe_ligand=200.0e-12,
        Fe_dust=5.0e-9,
        R_FeN_avg=6.0e-6,
        f_POC=0.7,
        remin_DOC=0.010,
        NH4_init=1.0e-4,
        Fe_init_surf=30.0e-12,
        Fe_init_deep=120.0e-12,
        Si_init_surf=0.7e-3,
        Si_init_deep=10.0e-3,
    )


def run_spinup(months: int = 60, dt_hours: float = 6.0, satellite_path: str | None = None, out_dir: str = "results/chl_spinup"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_lat, n_lon = 180, 360
    z_coord = create_ocean_z_star(n_levels=35, H_max=5500.0)
    z_full_ref = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    dz_ref = np.asarray(z_coord.dz_ref, dtype=np.float64)

    cfg = build_tuned_bgc_config()

    state = init_biogeo_state((n_lat, n_lon, z_coord.n_levels), jnp.asarray(z_full_ref), cfg)
    dt = dt_hours * 3600.0
    surface_chl_monthly = []
    months_list = []

    for m in range(months):
        T, S, PAR = build_monthly_forcing(m, n_lat, n_lon, z_full_ref)
        T_j = jnp.asarray(T, dtype=jnp.float64)
        S_j = jnp.asarray(S, dtype=jnp.float64)
        PAR_j = jnp.asarray(PAR, dtype=jnp.float64)
        ocean_mask = jnp.ones((n_lat, n_lon), dtype=jnp.float64)

        steps_per_month = max(1, int(30.5 * 86400.0 / dt))
        for _ in range(steps_per_month):
            state, _ = step_ocean_biogeochemistry(
                state,
                T_j,
                S_j,
                jnp.asarray(dz_ref),
                jnp.asarray(z_full_ref),
                ocean_mask,
                dt,
                cfg,
                PAR_surf=PAR_j,
            )

        chl_surf = np.asarray(state.Chl_d[..., 0] + state.Chl_n[..., 0], dtype=np.float64)
        chl_mean = 1.0e6 * global_mean_1d(chl_surf)
        surface_chl_monthly.append(chl_mean)
        months_list.append(m + 1)
        print(f"month {m+1:02d}: global mean surface Chl-a = {chl_mean:.3f} mg m^-3")

    # Save the monthly mean time series.
    out_nc = out_dir / "global_chl_spinup.nc"
    xr.Dataset(
        {
            "surface_chl_a": ("month", np.asarray(surface_chl_monthly, dtype=np.float64)),
        },
        coords={"month": np.asarray(months_list, dtype=int)},
    ).to_netcdf(out_nc)

    # Global annual mean in a simple line plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(months_list, np.asarray(surface_chl_monthly), marker="o")
    ax.set_xlabel("month")
    ax.set_ylabel("global mean surface Chl-a [mg m$^{-3}$]")
    ax.set_title("1-degree offline spin-up: global annual mean Chl-a")
    fig.tight_layout()
    fig.savefig(out_dir / "global_annual_mean_chl.png", dpi=180)
    plt.close(fig)

    # Seasonal cycle plot.
    seasonal = np.asarray(surface_chl_monthly, dtype=np.float64)
    if len(seasonal) >= 12:
        n_full = 12 * (len(seasonal) // 12)
        seasonal_cycle = np.mean(np.reshape(seasonal[:n_full], (-1, 12)), axis=0)
        x_months = np.arange(1, 13)
    else:
        seasonal_cycle = seasonal
        x_months = np.arange(1, len(seasonal) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_months, seasonal_cycle[:len(x_months)], marker="o")
    ax.set_xlabel("month")
    ax.set_ylabel("Chl-a [mg m$^{-3}$]")
    ax.set_title("Seasonal cycle of surface Chl-a")
    fig.tight_layout()
    fig.savefig(out_dir / "seasonal_cycle_chl.png", dpi=180)
    plt.close(fig)

    # Compare to a satellite climatology if supplied; otherwise create a simple placeholder reference plot.
    satellite_ds = try_load_satellite(satellite_path)
    if satellite_ds is not None:
        sat = satellite_ds
        # Default extraction: pick the first plausible chlorophyll variable.
        vars_to_try = ["chlor_a", "chl_a", "CHL", "chlorophyll", "CHL_a"]
        ref = None
        for name in vars_to_try:
            if name in sat:
                ref = sat[name]
                break
        if ref is not None:
            ref_vals = np.asarray(ref).reshape(-1)
            ref_vals = ref_vals[np.isfinite(ref_vals)]
            if ref_vals.size > 0:
                ref_clim = float(np.nanmean(ref_vals))
                model_clim = float(np.nanmean(surface_chl_monthly))
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.bar(["Model", "Satellite"], [model_clim, ref_clim], color=["C0", "C3"])
                ax.set_ylabel("surface Chl-a [mg m$^{-3}$]")
                ax.set_title("Model vs satellite reference")
                fig.tight_layout()
                fig.savefig(out_dir / "chl_satellite_comparison.png", dpi=180)
                plt.close(fig)
                print(f"Saved model vs satellite comparison using {satellite_path}")
            else:
                print("Satellite dataset was found but no finite chlorophyll values were extracted.")
        else:
            print("No chlorophyll variable found in the satellite dataset.")
    else:
        ref = make_reference_climatology(len(surface_chl_monthly))
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(months_list, np.asarray(surface_chl_monthly), label="model", marker="o")
        ax.plot(months_list, np.asarray(surface_chl_monthly) * np.asarray(ref), label="satellite-like reference", linestyle="--")
        ax.set_xlabel("month")
        ax.set_ylabel("surface Chl-a [mg m$^{-3}$]")
        ax.set_title("Model vs satellite-like climatology (placeholder reference)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "chl_satellite_comparison.png", dpi=180)
        plt.close(fig)
        print("No real satellite file was supplied; a placeholder climatology was used for the comparison figure.")

    print(f"Saved monthly Chl-a output to {out_nc}")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=60)
    parser.add_argument("--dt-hours", type=float, default=6.0)
    parser.add_argument("--satellite", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results/chl_spinup_long")
    args = parser.parse_args()
    run_spinup(
        months=args.months,
        dt_hours=args.dt_hours,
        satellite_path=args.satellite,
        out_dir=args.out_dir,
    )
