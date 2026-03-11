#!/usr/bin/env python
"""RCE experiments at 2.5-degree resolution (C36).

Runs two experiments sequentially:
  1. RCE with slab ocean (aquaplanet)
  2. RCE with slab land + bucket hydrology

Both use gray radiation (Frierson 2006) + SBM convection, operator-split.
Results saved to results/ocean/rce_25deg/ and results/land/rce_25deg/.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_rce_25deg.py
    JAX_ENABLE_X64=1 python scripts/run_rce_25deg.py --days 300
"""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="RCE experiments at 2.5 degree")
parser.add_argument("--days", type=int, default=200, help="Integration length [days]")
parser.add_argument("--nlev", type=int, default=20, help="Number of vertical levels")
parser.add_argument("--diag-days", type=int, default=5, help="Diagnostic interval [days]")
args = parser.parse_args()

N = 36           # C36 ~ 2.5 degrees
NLEV = args.nlev
DT = 300.0       # 300s for CFL stability at C36
N_DAYS = args.days
DIAG_DAYS = args.diag_days

# ---------------------------------------------------------------------------
# 1. Grid and vertical coordinate (shared)
# ---------------------------------------------------------------------------
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate

grid = create_cubed_sphere(N)
sigma = create_sigma_coordinate(NLEV)

shape_2d = (6, N, N)
shape_3d = (6, N, N, NLEV)

# ---------------------------------------------------------------------------
# 2. Atmospheric dycore (shared, with edge blending for C36)
# ---------------------------------------------------------------------------
from legoesm.atmosphere.dynamics.primitive_eq import (
    PrimitiveEquationModel,
    PrimitiveEquationConfig,
)
from legoesm.atmosphere.physics.held_suarez import held_suarez_init

HYPERDIFF = 5e16 * (48 / N) ** 4
dycore_config = PrimitiveEquationConfig(
    hyperdiff_coeff=HYPERDIFF,
    hyperdiff_ps_coeff=HYPERDIFF,
    use_conservation_fixer=True,
    fix_mass=True,
    edge_blend_uv=0.15,
    edge_blend_T=0.10,
    edge_blend_p_s=0.20,
    edge_blend_width=2,
)
model = PrimitiveEquationModel(grid, sigma, dycore_config)

# ---------------------------------------------------------------------------
# 3. Physics configuration (shared)
# ---------------------------------------------------------------------------
from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.core.operators_3d import hyperdiffusion_3d
from legoesm.core.operators import hyperdiffusion
from legoesm.core.field import Field
from legoesm.atmosphere.dynamics.edge_blending import blend_scalar_cube_edges

sbm_config = SBMConfig(tau_c=7200.0, RH_ref=0.7)

_sigma_full = sigma.sigma_full
_sigma_half = sigma.sigma_half
_dsigma = sigma.dsigma
_RH_init = 0.6

# Rayleigh friction (shared)
_sigma_b = 0.7
_k_f_max = 1.0 / 86400.0
_k_free = 0.1 / 86400.0
_k_f = _k_free + _k_f_max * jnp.maximum(0.0, (_sigma_full - _sigma_b) / (1.0 - _sigma_b))
_fric_decay = jnp.exp(-_k_f * DT)

# Snapshot days
snapshot_days = {10, 30, 60, 100, 200, 300}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


# ============================= PLOTTING HELPERS =============================

def save_timeseries(diag, output_dir, title, sfc_label, sfc_data):
    """Save time-series plot."""
    fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

    axes[0].plot(diag["times"], sfc_data, "b-o", ms=3, label=sfc_label)
    axes[0].plot(diag["times"], diag["T_low"], "r--s", ms=3, label="T_low (atm)")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(diag["times"], diag["T_atm"], "r-o", ms=3)
    axes[1].set_ylabel("Global-mean T_atm [K]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(diag["times"], diag["max_wind"], "g-o", ms=3)
    axes[2].set_ylabel("Max wind speed [m/s]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(diag["times"], diag["precip"], "c-o", ms=3)
    axes[3].set_ylabel("Precip [mm/day]")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(diag["times"], diag["CWV"], "blue", marker="s", ms=3)
    axes[4].set_ylabel("CWV [kg/m2]")
    axes[4].set_xlabel("Time [days]")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "timeseries.png", dpi=150)
    plt.close()
    print(f"  Saved {output_dir / 'timeseries.png'}")


def save_final_state(arrays, output_dir, n_days, sfc_label):
    """Save final-state 5-panel snapshot."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))

    im0 = axes[0].imshow(arrays["sfc"][0], origin="lower", cmap="coolwarm")
    axes[0].set_title(f"{sfc_label} [K] (face 0, day {n_days})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(arrays["T_low"][0], origin="lower", cmap="coolwarm")
    axes[1].set_title("Lowest-level T [K] (face 0)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(arrays["wind"][0], origin="lower", cmap="magma")
    axes[2].set_title("Surface wind [m/s] (face 0)")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(arrays["q_v"][0], origin="lower", cmap="YlGnBu")
    axes[3].set_title("q_v lowest level [g/kg] (face 0)")
    plt.colorbar(im3, ax=axes[3])

    im4 = axes[4].imshow(arrays["precip"][0], origin="lower", cmap="Blues")
    axes[4].set_title("Precip [mm/day] (face 0)")
    plt.colorbar(im4, ax=axes[4])

    plt.tight_layout()
    plt.savefig(output_dir / "final_state.png", dpi=150)
    plt.close()
    print(f"  Saved {output_dir / 'final_state.png'}")


def save_profiles(diag, output_dir):
    """Save vertical profile evolution."""
    if not diag["profiles_T"]:
        return
    n_diag = len(diag["profiles_T"])
    n_show = min(8, n_diag)
    indices = np.linspace(0, n_diag - 1, n_show, dtype=int)
    colors = cm.viridis(np.linspace(0, 1, n_show))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for i, idx in enumerate(indices):
        lbl = f"day {diag['times'][idx]:.0f}"
        ax1.plot(diag["profiles_T"][idx], diag["sigma"], color=colors[i], label=lbl)
        ax2.plot(diag["profiles_qv"][idx], diag["sigma"], color=colors[i], label=lbl)

    ax1.set_xlabel("Temperature [K]"); ax1.set_ylabel("Sigma")
    ax1.set_title("Global-mean temperature profiles")
    ax1.invert_yaxis(); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Specific humidity [g/kg]"); ax2.set_ylabel("Sigma")
    ax2.set_title("Global-mean moisture profiles")
    ax2.invert_yaxis(); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "profiles.png", dpi=150)
    plt.close()
    print(f"  Saved {output_dir / 'profiles.png'}")


def save_snapshots(snapshots, output_dir, sfc_key, sfc_label):
    """Save 2D snapshot evolution."""
    snap_days = sorted(snapshots.keys())
    if not snap_days:
        return
    fields = [
        (sfc_key, sfc_label, "coolwarm"),
        ("q_v_low", "q_v lowest [g/kg]", "YlGnBu"),
        ("precip", "Precip [mm/day]", "Blues"),
        ("wind", "Wind speed [m/s]", "magma"),
    ]
    n_f, n_s = len(fields), len(snap_days)
    fig, axes = plt.subplots(n_f, n_s, figsize=(4 * n_s, 3.5 * n_f))
    if n_s == 1:
        axes = axes[:, None]

    for j, day in enumerate(snap_days):
        snap = snapshots[day]
        for i, (key, label, cmap) in enumerate(fields):
            ax = axes[i, j]
            im = ax.imshow(snap[key], origin="lower", cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=0.8)
            if i == 0:
                ax.set_title(f"Day {day}", fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.tick_params(labelsize=7)

    plt.suptitle("2D snapshots (face 0)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "snapshots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'snapshots.png'}")


def save_results_txt(diag, output_dir, header, sfc_label, sfc_data, extra_cols=None):
    """Save text diagnostics."""
    with open(output_dir / "results.txt", "w") as f:
        f.write(header + "\n\n")
        hdr = f"{'Day':>6s}  {sfc_label:>10s}  {'<T_atm>':>10s}  {'<T_low>':>10s}  {'<Precip>':>10s}  {'<CWV>':>10s}  {'max|v|':>10s}"
        if extra_cols:
            for name in extra_cols:
                hdr += f"  {name:>10s}"
        f.write(hdr + "\n")
        for i in range(len(diag["times"])):
            line = (f"{diag['times'][i]:6.0f}  {sfc_data[i]:10.3f}  {diag['T_atm'][i]:10.3f}"
                    f"  {diag['T_low'][i]:10.3f}  {diag['precip'][i]:10.3f}"
                    f"  {diag['CWV'][i]:10.1f}  {diag['max_wind'][i]:10.3f}")
            if extra_cols:
                for name in extra_cols:
                    line += f"  {diag[name][i]:10.3f}"
            f.write(line + "\n")


# =============================== EXPERIMENT 1 ===============================
# RCE with slab ocean
# ============================================================================

OUTPUT_OCEAN = Path("results/ocean/rce_25deg")
OUTPUT_OCEAN.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  [1/2] Moist RCE: Slab Ocean (C36/L20, 2.5 deg)")
print("=" * 70)
print(f"  Grid: C{N}/L{NLEV}, dt={DT:.0f}s, {N_DAYS} days")

gray_config_ocean = GrayRadiationConfig(
    tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
    sfc_albedo=0.31, perpetual_equinox=True,
)

from legoesm.ocean.simple_ocean import SimpleOceanConfig
ocean_config = SimpleOceanConfig(
    mode="slab", h_mix=50.0, rho_ocean=1025.0, c_ocean=3994.0,
    Q_flux=0.0, albedo_ocean=0.06, emissivity_ocean=0.97,
    Cd_ocean=1.5e-3, Ch_ocean=1.5e-3, U_min=1.0, T_freeze=271.35,
)
_C_mix = ocean_config.rho_ocean * ocean_config.c_ocean * ocean_config.h_mix
_C_H_ocean = ocean_config.Ch_ocean

# Initial state
state = held_suarez_init(grid, sigma, T_init=280.0)
ocean_sst = jnp.full(shape_2d, 300.0)

# Moisture initialization
p_full_init = state.p_s.data[..., None] * _sigma_full
q_sat_init = saturation_mixing_ratio(state.T.data, p_full_init)
q_v = _RH_init * q_sat_init * _sigma_full ** 2
q_v = jnp.minimum(q_v, q_sat_init)
print(f"  SST_init=300 K, Moisture: RH_init={_RH_init}")


@jax.jit
def physics_ocean(T, p_s, q_v, u, v, ocean_sst, lat, dt):
    """Operator-split physics for ocean RCE."""
    nlev = _sigma_full.shape[0]
    sh3 = T.shape
    sh2 = p_s.shape
    ncol = sh2[0] * sh2[1] * sh2[2]

    p_full = p_s[..., None] * _sigma_full
    p_half = p_s[..., None] * _sigma_half

    T_col = T.reshape(ncol, nlev)
    pf_col = p_full.reshape(ncol, nlev)
    ph_col = p_half.reshape(ncol, nlev + 1)
    qv_col = q_v.reshape(ncol, nlev)
    sst_col = ocean_sst.reshape(ncol)
    lat_col = lat.reshape(ncol)

    insol = perpetual_equinox_insolation(lat_col, gray_config_ocean.S_0)

    rad = gray_radiation(T=T_col, p_full=pf_col, p_half=ph_col,
                         sfc_temperature=sst_col, lat=lat_col, q_v=qv_col,
                         insolation=insol, config=gray_config_ocean)
    dT_rad = rad.heating_rate.reshape(sh3)

    conv = sbm_convection(T=T_col, q_v=qv_col, p_full=pf_col, p_half=ph_col,
                          dt=dt, config=sbm_config)
    dT_conv = conv.dT_dt.reshape(sh3)
    dqv_conv = conv.dq_v_dt.reshape(sh3)
    precip = conv.precipitation.reshape(sh2)

    # BL coupling
    rho_low = (p_s * _sigma_full[-1]) / (constants.R_d * T[..., -1])
    wspd = jnp.sqrt(u[..., -1]**2 + v[..., -1]**2 + 1.0)
    dp_low = p_s * (_sigma_half[-1] - _sigma_half[-2])
    shflx = rho_low * constants.c_pd * _C_H_ocean * wspd * (ocean_sst - T[..., -1])
    q_sat_sfc = saturation_mixing_ratio(ocean_sst, p_s)
    lhflx = rho_low * constants.L_v * _C_H_ocean * wspd * (q_sat_sfc - q_v[..., -1])
    evap = lhflx / constants.L_v
    dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
    dq_BL = constants.g * evap / dp_low

    # Ocean energy balance
    sw_sfc = (rad.sw_flux_down[:, -1] - rad.sw_flux_up[:, -1]).reshape(sh2)
    lw_sfc = (rad.lw_flux_down[:, -1] - rad.lw_flux_up[:, -1]).reshape(sh2)
    sst_new = ocean_sst + dt * (sw_sfc + lw_sfc - shflx - lhflx) / _C_mix
    sst_new = jnp.maximum(sst_new, ocean_config.T_freeze)

    dT = dT_rad + dT_conv
    dT = dT.at[..., -1].add(dT_BL)
    dqv = dqv_conv
    dqv = dqv.at[..., -1].add(dq_BL)

    # Hyperdiffusion on q_v
    dqv = dqv + hyperdiffusion_3d(q_v, grid, HYPERDIFF)

    return dT, dqv, sst_new, precip, sw_sfc, lw_sfc


# --- Integration ---
n_steps = int(N_DAYS * 86400 / DT)
diag_interval = int(DIAG_DAYS * 86400 / DT)

diag_ocean = {k: [] for k in ["times", "sst", "T_atm", "T_low", "max_wind",
                                "precip", "CWV", "profiles_T", "profiles_qv"]}
diag_ocean["sigma"] = np.asarray(_sigma_full)
snapshots_ocean = {}

print(f"\n  Starting ocean RCE: {n_steps} steps")
print(f"  {'Day':>6s}  {'<SST>':>8s}  {'<T_atm>':>8s}  {'<T_low>':>8s}"
      f"  {'<Precip>':>8s}  {'<CWV>':>6s}  {'max|v|':>8s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")

t0 = time.time()

# JIT warmup
state = model.step_with_physics(state, DT)
dT, dqv, ocean_sst, _pr, _sw, _lw = physics_ocean(
    state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
    ocean_sst, grid.lat, DT)
new_T = state.T.data + DT * dT
q_v = jnp.maximum(q_v + DT * dqv, 0.0)
q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
excess = jnp.maximum(q_v - q_sat, 0.0)
q_v = q_v - excess
new_T = new_T + constants.L_v * excess / constants.c_pd
state = state._replace(T=state.T.replace(data=new_T))
_precip_ls = jnp.sum(excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)
q_v = blend_scalar_cube_edges(q_v, 0.10, width=2)
q_v = jnp.maximum(q_v, 0.0)
state = state._replace(
    u=state.u.replace(data=state.u.data * _fric_decay),
    v=state.v.replace(data=state.v.data * _fric_decay))

jax.block_until_ready(state.u.data)
t_jit = time.time() - t0
print(f"  JIT compiled in {t_jit:.1f}s")
t0 = time.time()

blowup = False
for step in range(1, n_steps):
    state = model.step_with_physics(state, DT)
    dT, dqv, ocean_sst, _pr, _sw, _lw = physics_ocean(
        state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
        ocean_sst, grid.lat, DT)
    new_T = state.T.data + DT * dT
    q_v = jnp.maximum(q_v + DT * dqv, 0.0)
    q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
    excess = jnp.maximum(q_v - q_sat, 0.0)
    q_v = q_v - excess
    new_T = new_T + constants.L_v * excess / constants.c_pd
    state = state._replace(T=state.T.replace(data=new_T))
    _precip_ls = jnp.sum(excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)
    q_v = blend_scalar_cube_edges(q_v, 0.10, width=2)
    q_v = jnp.maximum(q_v, 0.0)
    state = state._replace(
        u=state.u.replace(data=state.u.data * _fric_decay),
        v=state.v.replace(data=state.v.data * _fric_decay))

    if (step + 1) % diag_interval == 0:
        jax.block_until_ready(state.u.data)
        day = (step + 1) * DT / 86400.0
        m_sst = float(jnp.mean(ocean_sst))
        m_T = float(jnp.mean(state.T.data))
        m_Tl = float(jnp.mean(state.T.data[..., -1]))
        mx_v = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
        m_pr = float(jnp.mean(_pr + _precip_ls)) * 86400.0
        cwv = jnp.sum(q_v * state.p_s.data[..., None] * _dsigma, axis=-1) / constants.g
        m_cwv = float(jnp.mean(cwv))

        diag_ocean["times"].append(day)
        diag_ocean["sst"].append(m_sst)
        diag_ocean["T_atm"].append(m_T)
        diag_ocean["T_low"].append(m_Tl)
        diag_ocean["max_wind"].append(mx_v)
        diag_ocean["precip"].append(m_pr)
        diag_ocean["CWV"].append(m_cwv)
        diag_ocean["profiles_T"].append(np.asarray(jnp.mean(state.T.data, axis=(0, 1, 2))))
        diag_ocean["profiles_qv"].append(np.asarray(jnp.mean(q_v, axis=(0, 1, 2))) * 1000.0)

        iday = int(round(day))
        if iday in snapshot_days:
            snapshots_ocean[iday] = {
                "SST": np.asarray(ocean_sst[0]),
                "T_low": np.asarray(state.T.data[0, :, :, -1]),
                "q_v_low": np.asarray(q_v[0, :, :, -1]) * 1000.0,
                "precip": np.asarray((_pr + _precip_ls)[0]) * 86400.0,
                "wind": np.asarray(jnp.sqrt(state.u.data[0, :, :, -1]**2 + state.v.data[0, :, :, -1]**2)),
            }

        print(f"  {day:6.0f}  {m_sst:8.2f}  {m_T:8.2f}  {m_Tl:8.2f}"
              f"  {m_pr:8.2f}  {m_cwv:6.1f}  {mx_v:8.2f}")

        if not jnp.all(jnp.isfinite(state.u.data)) or mx_v > 500:
            print(f"  BLOWUP at day {day:.0f}")
            blowup = True
            break

jax.block_until_ready(state.u.data)
wall_ocean = time.time() - t0
print(f"\n  Ocean RCE complete: {wall_ocean:.1f}s wall time")
if blowup:
    print("  WARNING: simulation did not complete cleanly")

# Save ocean results
save_results_txt(diag_ocean, OUTPUT_OCEAN,
                 f"RCE Slab Ocean C{N}/L{NLEV}, dt={DT}s, {N_DAYS}d",
                 "<SST>", diag_ocean["sst"])
save_timeseries(diag_ocean, OUTPUT_OCEAN,
                "RCE Ocean (C36, 2.5 deg): Slab Ocean + Gray Rad + SBM",
                "SST", diag_ocean["sst"])
save_final_state({
    "sfc": np.asarray(ocean_sst),
    "T_low": np.asarray(state.T.data[..., -1]),
    "wind": np.sqrt(np.asarray(state.u.data[..., -1])**2 + np.asarray(state.v.data[..., -1])**2),
    "q_v": np.asarray(q_v[..., -1]) * 1000.0,
    "precip": np.asarray((_pr + _precip_ls)) * 86400.0,
}, OUTPUT_OCEAN, N_DAYS, "SST")
save_profiles(diag_ocean, OUTPUT_OCEAN)
save_snapshots(snapshots_ocean, OUTPUT_OCEAN, "SST", "SST [K]")


# =============================== EXPERIMENT 2 ===============================
# RCE with slab land + bucket hydrology
# ============================================================================

OUTPUT_LAND = Path("results/land/rce_25deg")
OUTPUT_LAND.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("  [2/2] Moist RCE: Slab Land + Bucket Hydrology (C36/L20, 2.5 deg)")
print("=" * 70)
print(f"  Grid: C{N}/L{NLEV}, dt={DT:.0f}s, {N_DAYS} days")

gray_config_land = GrayRadiationConfig(
    tau_equator=7.2, tau_pole=1.8, S_0=1360.0,
    sfc_albedo=0.25, perpetual_equinox=True,
)

from legoesm.land.config import LandConfig
land_config = LandConfig(
    C_soil=2.0e6, d_soil=1.0,
    albedo_land=0.25, emissivity_land=0.96,
    W_max=150.0, beta_min=0.1,
)
_C_land = land_config.C_soil * land_config.d_soil
_W_max = land_config.W_max
_beta_min = land_config.beta_min
_C_H_land = 1.5e-3
_T_min = 200.0

# Fresh initial state
state = held_suarez_init(grid, sigma, T_init=280.0)
T_land = jnp.full(shape_2d, 280.0)
W_bucket = jnp.full(shape_2d, 0.6 * _W_max)

p_full_init = state.p_s.data[..., None] * _sigma_full
q_sat_init = saturation_mixing_ratio(state.T.data, p_full_init)
q_v = _RH_init * q_sat_init * _sigma_full ** 2
q_v = jnp.minimum(q_v, q_sat_init)
print(f"  T_land_init=280 K, W_init={0.6*_W_max:.0f} kg/m2")


@jax.jit
def physics_land(T, p_s, q_v, u, v, T_land, W_bucket, lat, dt):
    """Operator-split physics for land RCE."""
    nlev = _sigma_full.shape[0]
    sh3 = T.shape
    sh2 = p_s.shape
    ncol = sh2[0] * sh2[1] * sh2[2]

    p_full = p_s[..., None] * _sigma_full
    p_half = p_s[..., None] * _sigma_half

    T_col = T.reshape(ncol, nlev)
    pf_col = p_full.reshape(ncol, nlev)
    ph_col = p_half.reshape(ncol, nlev + 1)
    qv_col = q_v.reshape(ncol, nlev)
    tl_col = T_land.reshape(ncol)
    lat_col = lat.reshape(ncol)

    insol = perpetual_equinox_insolation(lat_col, gray_config_land.S_0)

    rad = gray_radiation(T=T_col, p_full=pf_col, p_half=ph_col,
                         sfc_temperature=tl_col, lat=lat_col, q_v=qv_col,
                         insolation=insol, config=gray_config_land)
    dT_rad = rad.heating_rate.reshape(sh3)

    conv = sbm_convection(T=T_col, q_v=qv_col, p_full=pf_col, p_half=ph_col,
                          dt=dt, config=sbm_config)
    dT_conv = conv.dT_dt.reshape(sh3)
    dqv_conv = conv.dq_v_dt.reshape(sh3)
    precip = conv.precipitation.reshape(sh2)

    # BL coupling
    rho_low = (p_s * _sigma_full[-1]) / (constants.R_d * T[..., -1])
    wspd = jnp.sqrt(u[..., -1]**2 + v[..., -1]**2 + 1.0)
    dp_low = p_s * (_sigma_half[-1] - _sigma_half[-2])

    shflx = rho_low * constants.c_pd * _C_H_land * wspd * (T_land - T[..., -1])
    q_sat_sfc = saturation_mixing_ratio(T_land, p_s)
    beta = _beta_min + (1.0 - _beta_min) * jnp.clip(W_bucket / _W_max, 0.0, 1.0)
    lhflx = rho_low * constants.L_v * _C_H_land * wspd * (beta * q_sat_sfc - q_v[..., -1])
    lhflx = jnp.maximum(lhflx, 0.0)
    evap = lhflx / constants.L_v
    dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
    dq_BL = constants.g * evap / dp_low

    # Land energy balance
    sw_sfc = (rad.sw_flux_down[:, -1] - rad.sw_flux_up[:, -1]).reshape(sh2)
    lw_sfc = (rad.lw_flux_down[:, -1] - rad.lw_flux_up[:, -1]).reshape(sh2)
    T_land_new = T_land + dt * (sw_sfc + lw_sfc - shflx - lhflx) / _C_land

    # Bucket hydrology
    W_new = jnp.clip(W_bucket + dt * (precip - evap), 0.0, _W_max)

    dT = dT_rad + dT_conv
    dT = dT.at[..., -1].add(dT_BL)
    dqv = dqv_conv
    dqv = dqv.at[..., -1].add(dq_BL)

    # Hyperdiffusion on q_v, T_land, and W_bucket
    dqv = dqv + hyperdiffusion_3d(q_v, grid, HYPERDIFF)
    _tl = Field(data=T_land, name="tl", dims=("face", "x", "y"), units="K")
    T_land_new = T_land_new + dt * hyperdiffusion(_tl, grid, HYPERDIFF).data
    _wb = Field(data=W_bucket, name="W", dims=("face", "x", "y"), units="kg/m2")
    W_new = W_new + dt * hyperdiffusion(_wb, grid, HYPERDIFF).data

    # Edge blending on T_land and W_bucket (q_v blended in integration loop)
    T_land_new = blend_scalar_cube_edges(T_land_new, 0.10, width=2)
    T_land_new = jnp.maximum(T_land_new, _T_min)
    W_new = blend_scalar_cube_edges(W_new, 0.10, width=2)
    W_new = jnp.clip(W_new, 0.0, _W_max)

    return dT, dqv, T_land_new, W_new, precip, sw_sfc, lw_sfc


# --- Integration ---
diag_land = {k: [] for k in ["times", "T_land", "T_atm", "T_low", "max_wind",
                               "precip", "CWV", "W_bucket", "profiles_T", "profiles_qv"]}
diag_land["sigma"] = np.asarray(_sigma_full)
snapshots_land = {}

print(f"\n  Starting land RCE: {n_steps} steps")
print(f"  {'Day':>6s}  {'<T_land>':>8s}  {'<T_atm>':>8s}  {'<T_low>':>8s}"
      f"  {'<Precip>':>8s}  {'<W>':>6s}  {'<CWV>':>6s}  {'max|v|':>8s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}")

t0 = time.time()

# JIT warmup
state = model.step_with_physics(state, DT)
dT, dqv, T_land, W_bucket, _pr, _sw, _lw = physics_land(
    state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
    T_land, W_bucket, grid.lat, DT)
new_T = state.T.data + DT * dT
q_v = jnp.maximum(q_v + DT * dqv, 0.0)
q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
excess = jnp.maximum(q_v - q_sat, 0.0)
q_v = q_v - excess
new_T = new_T + constants.L_v * excess / constants.c_pd
state = state._replace(T=state.T.replace(data=new_T))
_precip_ls = jnp.sum(excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)
q_v = blend_scalar_cube_edges(q_v, 0.10, width=2)
q_v = jnp.maximum(q_v, 0.0)
state = state._replace(
    u=state.u.replace(data=state.u.data * _fric_decay),
    v=state.v.replace(data=state.v.data * _fric_decay))

jax.block_until_ready(state.u.data)
t_jit = time.time() - t0
print(f"  JIT compiled in {t_jit:.1f}s")
t0 = time.time()

blowup = False
for step in range(1, n_steps):
    state = model.step_with_physics(state, DT)
    dT, dqv, T_land, W_bucket, _pr, _sw, _lw = physics_land(
        state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
        T_land, W_bucket, grid.lat, DT)
    new_T = state.T.data + DT * dT
    q_v = jnp.maximum(q_v + DT * dqv, 0.0)
    q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
    excess = jnp.maximum(q_v - q_sat, 0.0)
    q_v = q_v - excess
    new_T = new_T + constants.L_v * excess / constants.c_pd
    state = state._replace(T=state.T.replace(data=new_T))
    _precip_ls = jnp.sum(excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)
    q_v = blend_scalar_cube_edges(q_v, 0.10, width=2)
    q_v = jnp.maximum(q_v, 0.0)
    state = state._replace(
        u=state.u.replace(data=state.u.data * _fric_decay),
        v=state.v.replace(data=state.v.data * _fric_decay))

    if (step + 1) % diag_interval == 0:
        jax.block_until_ready(state.u.data)
        day = (step + 1) * DT / 86400.0
        m_tl = float(jnp.mean(T_land))
        m_T = float(jnp.mean(state.T.data))
        m_Tl = float(jnp.mean(state.T.data[..., -1]))
        mx_v = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))
        m_pr = float(jnp.mean(_pr + _precip_ls)) * 86400.0
        m_W = float(jnp.mean(W_bucket))
        cwv = jnp.sum(q_v * state.p_s.data[..., None] * _dsigma, axis=-1) / constants.g
        m_cwv = float(jnp.mean(cwv))

        diag_land["times"].append(day)
        diag_land["T_land"].append(m_tl)
        diag_land["T_atm"].append(m_T)
        diag_land["T_low"].append(m_Tl)
        diag_land["max_wind"].append(mx_v)
        diag_land["precip"].append(m_pr)
        diag_land["CWV"].append(m_cwv)
        diag_land["W_bucket"].append(m_W)
        diag_land["profiles_T"].append(np.asarray(jnp.mean(state.T.data, axis=(0, 1, 2))))
        diag_land["profiles_qv"].append(np.asarray(jnp.mean(q_v, axis=(0, 1, 2))) * 1000.0)

        iday = int(round(day))
        if iday in snapshot_days:
            snapshots_land[iday] = {
                "T_land": np.asarray(T_land[0]),
                "T_low": np.asarray(state.T.data[0, :, :, -1]),
                "q_v_low": np.asarray(q_v[0, :, :, -1]) * 1000.0,
                "precip": np.asarray((_pr + _precip_ls)[0]) * 86400.0,
                "wind": np.asarray(jnp.sqrt(state.u.data[0, :, :, -1]**2 + state.v.data[0, :, :, -1]**2)),
            }

        print(f"  {day:6.0f}  {m_tl:8.2f}  {m_T:8.2f}  {m_Tl:8.2f}"
              f"  {m_pr:8.2f}  {m_W:6.1f}  {m_cwv:6.1f}  {mx_v:8.2f}")

        if not jnp.all(jnp.isfinite(state.u.data)) or mx_v > 500:
            print(f"  BLOWUP at day {day:.0f}")
            blowup = True
            break

jax.block_until_ready(state.u.data)
wall_land = time.time() - t0
print(f"\n  Land RCE complete: {wall_land:.1f}s wall time")
if blowup:
    print("  WARNING: simulation did not complete cleanly")

# Save land results
save_results_txt(diag_land, OUTPUT_LAND,
                 f"RCE Slab Land C{N}/L{NLEV}, dt={DT}s, {N_DAYS}d",
                 "<T_land>", diag_land["T_land"],
                 extra_cols={"W_bucket": "W_bucket"})
save_timeseries(diag_land, OUTPUT_LAND,
                "RCE Land (C36, 2.5 deg): Slab Land + Gray Rad + SBM",
                "T_land", diag_land["T_land"])
save_final_state({
    "sfc": np.asarray(T_land),
    "T_low": np.asarray(state.T.data[..., -1]),
    "wind": np.sqrt(np.asarray(state.u.data[..., -1])**2 + np.asarray(state.v.data[..., -1])**2),
    "q_v": np.asarray(q_v[..., -1]) * 1000.0,
    "precip": np.asarray((_pr + _precip_ls)) * 86400.0,
}, OUTPUT_LAND, N_DAYS, "T_land")
save_profiles(diag_land, OUTPUT_LAND)
save_snapshots(snapshots_land, OUTPUT_LAND, "T_land", "T_land [K]")

print("\n" + "=" * 70)
print("  Both experiments complete")
print(f"  Ocean results: {OUTPUT_OCEAN}/")
print(f"  Land results:  {OUTPUT_LAND}/")
print("=" * 70)
