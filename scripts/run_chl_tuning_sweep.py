#!/usr/bin/env python3
"""1D parameter sweep for the npzd_v2 Chl-a scheme.

The script runs a small set of biologically motivated growth/loss perturbations
in the seasonal forced column and saves a comparison figure.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.biogeochemistry import BiogeoConfig, init_biogeo_state, step_ocean_biogeochemistry


BASE = dict(
    scheme="npzd_v2",
    pCO2_atm=400.0,
    wind_speed=7.0,
    k_N_d=1.5e-4,
    k_N_n=2.0e-4,
    k_NH4_d=1.0e-4,
    k_NH4_n=1.0e-4,
    k_Fe_d=90.0e-12,
    k_Fe_n=70.0e-12,
    k_Si=0.5e-3,
    alpha_P=0.12,
    g_max=0.45,
    k_P=0.20e-3,
    gamma_Z=0.68,
    epse=0.5,
    m_Pl=0.05,
    m_Pq_v2=0.08,
    m_Zl_v2=0.08,
    m_Z=0.10,
    remin_rate=0.04,
    remin_opal=0.03,
    remin_DON=0.015,
    k_w_atten=0.04,
    k_chl_atten=6.0,
    w_sink=9.0,
    theta_min_d=0.6e-3,
    theta_max_d=4.5e-3,
    theta_min_n=0.5e-3,
    theta_max_n=4.0e-3,
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
    ("base", dict(BASE)),
    ("sweep1", dict(BASE, mu_max_d=1.8, mu_max_n=1.2, alpha_P=0.18, k_N_d=8e-5, k_N_n=1.2e-4, g_max=0.30, m_Pl=0.03, m_Pq_v2=0.04, k_chl_atten=4.0)),
    ("sweep2", dict(BASE, mu_max_d=2.4, mu_max_n=1.6, alpha_P=0.25, k_N_d=5e-5, k_N_n=8e-5, g_max=0.22, m_Pl=0.02, m_Pq_v2=0.03, k_chl_atten=3.0)),
    ("sweep3", dict(BASE, mu_max_d=2.8, mu_max_n=1.8, alpha_P=0.30, k_N_d=4e-5, k_N_n=6e-5, g_max=0.18, m_Pl=0.015, m_Pq_v2=0.025, k_chl_atten=2.5)),
    ("sweep4", dict(BASE, mu_max_d=3.0, mu_max_n=2.0, alpha_P=0.35, k_N_d=5e-5, k_N_n=8e-5, g_max=0.15, m_Pl=0.010, m_Pq_v2=0.020, k_chl_atten=1.5)),
    ("sweep5", dict(BASE, mu_max_d=4.0, mu_max_n=2.5, alpha_P=0.45, k_N_d=3e-5, k_N_n=6e-5, g_max=0.12, m_Pl=0.008, m_Pq_v2=0.015, k_chl_atten=1.0)),
]


def make_column_grid(H=200.0, nlev=40):
    dz = np.full(nlev, H / nlev, dtype=np.float64)
    z_interfaces = np.concatenate(([0.0], -np.cumsum(dz)))
    z_full = 0.5 * (z_interfaces[:-1] + z_interfaces[1:])
    return z_full.astype(np.float64), dz.astype(np.float64)


def par_surface(t_day, latitude_deg=40.0):
    doy = t_day % 365.25
    angle = 2.0 * np.pi * (doy - 355.0) / 365.25
    par_max = 350.0 * np.cos(np.deg2rad(latitude_deg)) + 50.0
    par_min = 0.05 * par_max
    return float(par_min + (par_max - par_min) * 0.5 * (1.0 - np.cos(angle)))


def mixed_layer_depth(t_day):
    doy = t_day % 365.25
    angle = 2.0 * np.pi * (doy - 40.0) / 365.25
    return float(12.0 + 110.0 * 0.5 * (1.0 + np.cos(angle)))


def sst_profile(t_day):
    doy = t_day % 365.25
    angle = 2.0 * np.pi * (doy - 40.0) / 365.25
    return float(7.0 + 11.0 * 0.5 * (1.0 - np.cos(angle)))


def make_T_S(z_full, sst_val, mld):
    T = np.full_like(z_full, sst_val, dtype=np.float64)
    S = np.full_like(z_full, 35.0, dtype=np.float64)
    for k, z in enumerate(z_full):
        depth = -float(z)
        if depth > mld:
            frac = min((depth - mld) / 100.0, 1.0)
            T[k] = sst_val - (sst_val - 4.0) * frac
            S[k] = 35.0 + 0.3 * frac
    return T, S


def run_candidate(label, params, out_dir, n_steps=180):
    z_full, dz = make_column_grid()
    z_jax = jnp.asarray(z_full)
    dz_jax = jnp.asarray(dz)
    cfg = BiogeoConfig(**params)
    state = init_biogeo_state((z_full.size,), z_jax, cfg)
    mask = jnp.ones((), dtype=jnp.float64)
    dt = 6.0 * 3600.0
    values = []
    times = []
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
        if step % 6 == 0:
            chl = float(np.asarray(state.Chl_d + state.Chl_n, dtype=np.float64)[0] * 1.0e3)
            values.append(chl)
            times.append(t_day)
    return np.asarray(times), np.asarray(values)


def main():
    out_dir = Path("results/chl_tuning_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = []
    for label, params in CANDIDATES:
        t, v = run_candidate(label, params, out_dir)
        ax.plot(t, v, label=label, linewidth=2)
        summary.append((label, float(v.max()), float(v[-1]), float(v.mean())))
    ax.set_xlabel("time [days]")
    ax.set_ylabel("surface Chl-a [mg m$^{-3}$]")
    ax.set_title("1D Chl-a tuning sweep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chl_tuning_sweep.png", dpi=180)
    plt.close(fig)

    print("candidate summary:")
    for label, peak, final, avg in summary:
        print(f"{label:<7s} peak={peak:.3f} final={final:.3f} mean={avg:.3f} mg/m^3")
    print(f"Saved sweep figure to {out_dir / 'chl_tuning_sweep.png'}")


if __name__ == "__main__":
    main()
