#!/usr/bin/env python3
"""Tuned 1D seasonal Chl-a test for the MARBL-lite npzd_v2 scheme.

The model receives realistic forcing from a simple seasonal ocean column:
- surface PAR from solar geometry
- mixed-layer depth from a seasonal cycle
- temperature and salinity profiles from SST and MLD

The diagnostic target is surface chlorophyll, i.e. Chl_d + Chl_n.
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


def make_column_grid(H=200.0, nlev=40):
    dz = np.full(nlev, H / nlev, dtype=np.float64)
    z_interfaces = np.concatenate(([0.0], -np.cumsum(dz)))
    z_full = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])
    return z_full.astype(np.float64), dz.astype(np.float64)


def par_surface(t_day: float, latitude_deg: float = 40.0) -> float:
    doy = t_day % 365.25
    angle = 2.0 * np.pi * (doy - 355.0) / 365.25
    par_max = 350.0 * np.cos(np.deg2rad(latitude_deg)) + 50.0
    par_min = 0.05 * par_max
    return float(par_min + (par_max - par_min) * 0.5 * (1.0 - np.cos(angle)))


def mixed_layer_depth(t_day: float) -> float:
    doy = t_day % 365.25
    angle = 2.0 * np.pi * (doy - 40.0) / 365.25
    return float(12.0 + 110.0 * 0.5 * (1.0 + np.cos(angle)))


def sst_profile(t_day: float) -> float:
    doy = t_day % 365.25
    angle = 2.0 * np.pi * (doy - 40.0) / 365.25
    return float(7.0 + 11.0 * 0.5 * (1.0 - np.cos(angle)))


def make_T_S(z_full: np.ndarray, sst_val: float, mld: float):
    T = np.full_like(z_full, sst_val, dtype=np.float64)
    S = np.full_like(z_full, 35.0, dtype=np.float64)
    for k, z in enumerate(z_full):
        depth = -float(z)
        if depth > mld:
            frac = min((depth - mld) / 100.0, 1.0)
            T[k] = sst_val - (sst_val - 4.0) * frac
            S[k] = 35.0 + 0.3 * frac
    return T, S


def build_tuned_cfg() -> BiogeoConfig:
    # Best-performing stable candidate from the recent seasonal scan.
    # This sits in the satellite-like range while remaining numerically bounded.
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


def run_column_chl(years: int = 3, dt_minutes: float = 30.0, out_dir: str = "results/chl_column_tuned"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z_full, dz = make_column_grid(H=200.0, nlev=40)
    z_jax = jnp.asarray(z_full)
    dz_jax = jnp.asarray(dz)

    cfg = build_tuned_cfg()
    state = init_biogeo_state((z_full.size,), z_jax, cfg)
    mask = jnp.ones((), dtype=jnp.float64)

    dt = dt_minutes * 60.0
    days_total = years * 365.25
    n_steps = int(days_total * 86400.0 / dt)
    save_every = max(1, int(24.0 * 3600.0 / dt))

    surface_chl = []
    time_days = []
    times = []
    chl_profile_history = []

    for step in range(n_steps):
        t_day = step * dt / 86400.0
        sst = sst_profile(t_day)
        mld = mixed_layer_depth(t_day)
        par = par_surface(t_day)
        T_col, S_col = make_T_S(z_full, sst, mld)

        state, _ = step_ocean_biogeochemistry(
            state,
            jnp.asarray(T_col),
            jnp.asarray(S_col),
            dz_jax,
            z_jax,
            mask,
            dt,
            cfg,
            PAR_surf=jnp.asarray(par),
        )

        if step % save_every == 0:
            chl_profile = np.asarray(state.Chl_d + state.Chl_n, dtype=np.float64) * 1.0e3
            chl_surf = float(chl_profile[0])
            time_days.append(t_day)
            surface_chl.append(chl_surf)
            times.append(f"{t_day/365.25:.2f} y")
            chl_profile_history.append(chl_profile)

    ds = xr.Dataset(
        {
            "surface_chl_a": ("time", np.asarray(surface_chl, dtype=np.float64)),
            "time_days": ("time", np.asarray(time_days, dtype=np.float64)),
            "chl_a_profile": (("time", "depth"), np.asarray(chl_profile_history, dtype=np.float64)),
        },
        coords={
            "time": np.arange(len(surface_chl), dtype=int),
            "depth": z_full.astype(np.float64),
        },
    )
    ds.to_netcdf(out_dir / "chl_column_tuned.nc")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_days, surface_chl, color="tab:green", linewidth=2)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("surface Chl-a [mg m$^{-3}$]")
    ax.set_title("1D seasonal Chl-a cycle, tuned MARBL-lite column")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "chl_column_tuned_timeseries.png", dpi=180)
    plt.close(fig)

    if len(chl_profile_history) > 0:
        profile_array = np.asarray(chl_profile_history, dtype=np.float64)
        profile_times = np.asarray(time_days, dtype=np.float64)
        depth_m = -z_full
        fig, ax = plt.subplots(figsize=(9, 6))
        mesh = ax.pcolormesh(
            profile_times,
            depth_m,
            profile_array.T,
            shading="auto",
            cmap="YlGn",
            vmin=0.0,
            vmax=max(1.0, np.nanpercentile(profile_array, 99)),
        )
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Chl-a [mg m$^{-3}$]")
        ax.set_xlabel("time [days]")
        ax.set_ylabel("depth [m]")
        ax.set_title("Vertical Chl-a profile in the tuned 1D seasonal column")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_dir / "chl_column_tuned_vertical_profile.png", dpi=180)
        plt.close(fig)

        idx = np.linspace(0, len(profile_times) - 1, 4, dtype=int)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
        for ax, i in zip(axes.ravel(), idx):
            prof = profile_array[i]
            ax.plot(prof, depth_m, color="tab:green", linewidth=2)
            ax.set_ylim(depth_m[0], depth_m[-1])
            ax.set_title(f"day {profile_times[i]:.0f}")
            ax.set_xlabel("Chl-a [mg m$^{-3}$]")
            ax.set_ylabel("depth [m]")
            ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_dir / "chl_column_tuned_profile_snapshots.png", dpi=180)
        plt.close(fig)

    seasonal = np.asarray(surface_chl, dtype=np.float64)
    if len(seasonal) >= 365:
        monthly = np.reshape(seasonal[: 12 * (len(seasonal) // 12)], (-1, 12))
        seasonal_cycle = np.nanmean(monthly, axis=0)
        x = np.arange(1, 13)
    else:
        seasonal_cycle = seasonal
        x = np.arange(1, len(seasonal) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, seasonal_cycle, marker="o", color="tab:green")
    ax.set_xlabel("month")
    ax.set_ylabel("surface Chl-a [mg m$^{-3}$]")
    ax.set_title("Seasonal Chl-a cycle from 1D tuned run")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "chl_column_tuned_seasonal.png", dpi=180)
    plt.close(fig)

    print(f"Saved surface Chl-a diagnostics to {out_dir}")
    print(f"Final surface Chl-a: {surface_chl[-1]:.3f} mg m^-3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--dt-minutes", type=float, default=30.0)
    parser.add_argument("--out-dir", type=str, default="results/chl_column_tuned")
    args = parser.parse_args()
    run_column_chl(years=args.years, dt_minutes=args.dt_minutes, out_dir=args.out_dir)
