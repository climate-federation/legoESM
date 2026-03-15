#!/usr/bin/env python
"""Moist Radiative-Convective Equilibrium (RCE) with slab land and gray radiation.

Land-planet experiment: gray radiation (Frierson 2006) with moisture-coupled
optical depth + Simplified Betts-Miller convection + bucket hydrology.

Coupling strategy: operator splitting.
  1. Dynamics only: step_with_physics (JIT-compiled, RK3, no physics_fn)
  2. All physics operator-split in a single JIT function:
     - Gray radiation (moisture-dependent LW optical depth)
     - SBM convection (produces precipitation)
     - Bulk aerodynamic BL coupling (heat + moisture, wind-dependent)
     - Land energy balance (SW + LW - SH - LH)
     - Bucket hydrology (precip - evaporation)
  3. Rayleigh friction with weak free-atmosphere drag (k_free = 0.1/day)

Moisture is carried as a separate array outside the dycore state
(HydrostaticState has no tracer support). Initialized at 60% RH.

Physics: gray two-stream radiation + SBM convection (operator-split)
Land:    1 m slab soil + bucket hydrology (W_max = 150 kg/m2)
Grid:    C16, 20 levels (low-res for quick verification)
Duration: 100 days

Usage:
    JAX_ENABLE_X64=1 python scripts/run_rce_slab_land.py
    JAX_ENABLE_X64=1 python scripts/run_rce_slab_land.py --days 200
    JAX_ENABLE_X64=1 python scripts/run_rce_slab_land.py --resolution 24
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
parser = argparse.ArgumentParser(description="Moist RCE with slab land")
parser.add_argument("--days", type=int, default=100, help="Integration length [days]")
parser.add_argument("--resolution", type=int, default=16, help="Cubed-sphere N")
parser.add_argument("--nlev", type=int, default=20, help="Number of vertical levels")
parser.add_argument("--dt", type=float, default=None, help="Time step [s] (auto: 600 for C≤24, 300 for C>24)")
parser.add_argument("--diag-days", type=int, default=5, help="Diagnostic interval [days]")
parser.add_argument("--output", type=str, default="results/land/rce_slab_land", help="Output directory")
args = parser.parse_args()

N = args.resolution
NLEV = args.nlev
DT = args.dt if args.dt is not None else (300.0 if N > 24 else 600.0)
N_DAYS = args.days
DIAG_DAYS = args.diag_days

OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  Moist RCE: Slab Land + Gray Atmosphere + Bucket Hydrology")
print("=" * 70)
print(f"  Grid:       C{N} / L{NLEV}")
print(f"  dt:         {DT:.0f} s")
print(f"  Duration:   {N_DAYS} days")
print(f"  Diagnostics every {DIAG_DAYS} days")
print()

# ---------------------------------------------------------------------------
# 1. Grid and vertical coordinate
# ---------------------------------------------------------------------------
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate

grid = create_cubed_sphere(N)
sigma = create_sigma_coordinate(NLEV)

shape_2d = (6, N, N)
shape_3d = (6, N, N, NLEV)

print(f"  Grid created: {6*N*N} columns, {NLEV} levels")

# ---------------------------------------------------------------------------
# 2. Atmospheric model (hydrostatic primitive equations)
# ---------------------------------------------------------------------------
from legoesm.atmosphere.dynamics.primitive_eq import (
    PrimitiveEquationModel,
    PrimitiveEquationConfig,
)
from legoesm.atmosphere.physics.held_suarez import held_suarez_init

HYPERDIFF = 5e16 * (48 / N) ** 4
# Edge blending prevents checkerboard noise at cubed-sphere edges (needed for N>16)
_eb = N > 16
dycore_config = PrimitiveEquationConfig(
    hyperdiff_coeff=HYPERDIFF,
    hyperdiff_ps_coeff=HYPERDIFF,
    use_conservation_fixer=True,
    fix_mass=True,
    edge_blend_uv=0.15 if _eb else 0.0,
    edge_blend_T=0.10 if _eb else 0.0,
    edge_blend_p_s=0.20 if _eb else 0.0,
    edge_blend_width=2 if _eb else 1,
)
model = PrimitiveEquationModel(grid, sigma, dycore_config)

# Initial state: isothermal at rest
state = held_suarez_init(grid, sigma, T_init=280.0)
print(f"  Atmosphere initialized: T={280.0} K isothermal")

# ---------------------------------------------------------------------------
# 3. Physics configuration
# ---------------------------------------------------------------------------
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.convection.config import SBMConfig

# Gray radiation — same optical properties, land surface albedo
gray_config = GrayRadiationConfig(
    tau_equator=7.2,
    tau_pole=1.8,
    S_0=1360.0,
    sfc_albedo=0.25,
    perpetual_equinox=True,
)

# SBM convection — called directly (not through integration bridge)
sbm_config = SBMConfig(tau_c=7200.0, RH_ref=0.7)

print(f"  Physics: gray radiation (moist) + SBM convection (operator-split)")

# ---------------------------------------------------------------------------
# 4. Slab land surface + bucket hydrology
# ---------------------------------------------------------------------------
from legoesm.land.config import LandConfig

land_config = LandConfig(
    C_soil=2.0e6,          # Soil heat capacity [J/m3/K]
    d_soil=1.0,            # Slab depth [m]
    albedo_land=0.25,      # Land surface albedo
    emissivity_land=0.96,  # LW emissivity
    W_max=150.0,           # Bucket capacity [kg/m2]
    beta_min=0.1,          # Minimum moisture availability
)
_C_land = land_config.C_soil * land_config.d_soil  # 2e6 J/m2/K

# Initial land surface temperature
T_land = jnp.full(shape_2d, 280.0)

# Initial bucket moisture: 60% of capacity
W_bucket = jnp.full(shape_2d, 0.6 * land_config.W_max)

print(f"  Slab land: d_soil={land_config.d_soil} m, C={_C_land:.0e} J/m2/K, T_init=280.0 K")
print(f"  Bucket: W_max={land_config.W_max} kg/m2, W_init={0.6*land_config.W_max:.0f} kg/m2")

# ---------------------------------------------------------------------------
# 5. Moisture initialization (60% RH)
# ---------------------------------------------------------------------------
from legoesm import constants
from legoesm.diagnostics.column_integrals import column_water_vapor
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import perpetual_equinox_insolation
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.core.field import Field
from legoesm.core.operators import hyperdiffusion
from legoesm.core.operators_3d import hyperdiffusion_3d
from legoesm.atmosphere.dynamics.edge_blending import blend_scalar_cube_edges

_RH_init = 0.6
p_full_init = state.p_s.data[..., None] * sigma.sigma_full
# Moisture profile: 60% RH at surface, decreasing as sigma^2 with height.
# In an isothermal atmosphere, q_sat increases aloft (e_sat constant, p drops),
# so uniform RH gives unrealistically high upper-level moisture and CWV.
# sigma^2 weighting mimics the real atmosphere where q drops rapidly with height.
q_sat_init = saturation_mixing_ratio(state.T.data, p_full_init)
q_v = _RH_init * q_sat_init * sigma.sigma_full ** 2
q_v = jnp.minimum(q_v, q_sat_init)  # ensure sub-saturation
mean_qv = float(jnp.mean(q_v)) * 1000.0
cwv_init = float(jnp.mean(
    column_water_vapor(q_v, state.p_s.data, sigma.dsigma)
))
print(f"  Moisture: RH_init={_RH_init}, mean q_v={mean_qv:.2f} g/kg, CWV={cwv_init:.1f} kg/m2")

# ---------------------------------------------------------------------------
# 6. Operator-split physics step
# ---------------------------------------------------------------------------
_S_0 = gray_config.S_0
_sigma_full = sigma.sigma_full
_sigma_half = sigma.sigma_half
_dsigma = sigma.dsigma
_C_H = 1.5e-3              # Bulk transfer coefficient for heat (neutral)
_C_E = 1.5e-3              # Bulk transfer coefficient for moisture (neutral)
_T_min = 200.0             # Minimum land temperature [K]
_W_max = land_config.W_max
_beta_min = land_config.beta_min

# Rayleigh friction: BL drag + weak free-atmosphere drag
# k_total(sigma) = k_free + k_BL_max * max(0, (sigma - sigma_b) / (1 - sigma_b))
# k_free represents unresolved eddy momentum transport at C16
_sigma_b = 0.7
_k_f_max = 1.0 / 86400.0  # BL drag: 1/day [1/s]
_k_free = 0.1 / 86400.0   # Free-atmosphere drag: 0.1/day [1/s]
_k_f = _k_free + _k_f_max * jnp.maximum(0.0, (_sigma_full - _sigma_b) / (1.0 - _sigma_b))
_fric_decay = jnp.exp(-_k_f * DT)  # decay factor per timestep, shape (NLEV,)


@jax.jit
def physics_step(T, p_s, q_v, u, v, T_land, W_bucket, lat, dt):
    """Operator-split physics: radiation + convection + BL + land + bucket.

    Returns
    -------
    dT_dt       : (6,N,N,nlev) atmospheric temperature tendency [K/s]
    dq_v_dt     : (6,N,N,nlev) moisture tendency [kg/kg/s]
    T_land_new  : (6,N,N)      updated land temperature [K]
    W_new       : (6,N,N)      updated bucket moisture [kg/m2]
    precip      : (6,N,N)      precipitation rate [kg/m2/s]
    sw_net_sfc  : (6,N,N)      net SW at surface [W/m2]
    lw_net_sfc  : (6,N,N)      net LW at surface [W/m2]
    """
    nlev = _sigma_full.shape[0]
    shape_3d = T.shape
    shape_2d = p_s.shape
    ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

    # Pressure on full and half levels
    p_full = p_s[..., None] * _sigma_full
    p_half = p_s[..., None] * _sigma_half

    # Reshape to columns
    T_col = T.reshape(ncol, nlev)
    p_full_col = p_full.reshape(ncol, nlev)
    p_half_col = p_half.reshape(ncol, nlev + 1)
    q_v_col = q_v.reshape(ncol, nlev)
    T_land_col = T_land.reshape(ncol)
    lat_col = lat.reshape(ncol)

    # TOA insolation
    insol = perpetual_equinox_insolation(lat_col, _S_0)

    # --- (a) Gray radiation with moisture feedback ---
    rad_out = gray_radiation(
        T=T_col,
        p_full=p_full_col,
        p_half=p_half_col,
        sfc_temperature=T_land_col,
        lat=lat_col,
        q_v=q_v_col,
        insolation=insol,
        config=gray_config,
    )
    dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

    # --- (b) SBM convection ---
    conv_out = sbm_convection(
        T=T_col,
        q_v=q_v_col,
        p_full=p_full_col,
        p_half=p_half_col,
        dt=dt,
        config=sbm_config,
    )
    dT_dt_conv = conv_out.dT_dt.reshape(shape_3d)
    dq_v_dt_conv = conv_out.dq_v_dt.reshape(shape_3d)
    precip = conv_out.precipitation.reshape(shape_2d)  # [kg/m2/s]

    # --- (c) Bulk aerodynamic BL coupling ---
    # Lowest-level quantities
    rho_low = (p_s * _sigma_full[-1]) / (constants.R_d * T[..., -1])
    wind_speed = jnp.sqrt(u[..., -1]**2 + v[..., -1]**2 + 1.0)  # 1 m/s gustiness
    dp_low = p_s * (_sigma_half[-1] - _sigma_half[-2])

    # Sensible heat flux [W/m2]
    shflx = rho_low * constants.c_pd * _C_H * wind_speed * (T_land - T[..., -1])

    # Latent heat flux [W/m2]
    q_sat_sfc = saturation_mixing_ratio(T_land, p_s)
    beta = _beta_min + (1.0 - _beta_min) * jnp.clip(W_bucket / _W_max, 0.0, 1.0)
    lhflx = rho_low * constants.L_v * _C_E * wind_speed * (beta * q_sat_sfc - q_v[..., -1])
    lhflx = jnp.maximum(lhflx, 0.0)  # no dew
    evap_rate = lhflx / constants.L_v  # [kg/m2/s]

    # Atmospheric tendencies (lowest level only)
    dT_BL = constants.g * shflx / (constants.c_pd * dp_low)  # [K/s]
    dq_BL = constants.g * evap_rate / dp_low  # [kg/kg/s]

    # --- (f) Land energy balance ---
    sw_net_sfc = (rad_out.sw_flux_down[:, -1] - rad_out.sw_flux_up[:, -1]).reshape(shape_2d)
    lw_net_sfc = (rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]).reshape(shape_2d)

    dT_land_dt = (sw_net_sfc + lw_net_sfc - shflx - lhflx) / _C_land
    T_land_new = T_land + dt * dT_land_dt

    # --- (g) Bucket hydrology ---
    dW_dt = precip - evap_rate
    W_new = W_bucket + dt * dW_dt
    W_new = jnp.clip(W_new, 0.0, _W_max)

    # --- (h) Total atmospheric tendencies ---
    dT_dt = dT_dt_rad + dT_dt_conv
    dT_dt = dT_dt.at[..., -1].add(dT_BL)

    dq_v_dt = dq_v_dt_conv
    dq_v_dt = dq_v_dt.at[..., -1].add(dq_BL)

    # --- (i) Hyperdiffusion on q_v, T_land, and W_bucket ---
    dq_v_dt = dq_v_dt + hyperdiffusion_3d(q_v, grid, HYPERDIFF)
    _tl = Field(data=T_land, name="tl", dims=("face", "x", "y"), units="K")
    T_land_new = T_land_new + dt * hyperdiffusion(_tl, grid, HYPERDIFF).data
    _wb = Field(data=W_bucket, name="W", dims=("face", "x", "y"), units="kg/m2")
    W_new = W_new + dt * hyperdiffusion(_wb, grid, HYPERDIFF).data

    # --- (j) Edge blending on T_land and W_bucket (q_v blended in loop) ---
    T_land_new = blend_scalar_cube_edges(T_land_new, 0.10, width=2)
    T_land_new = jnp.maximum(T_land_new, _T_min)
    W_new = blend_scalar_cube_edges(W_new, 0.10, width=2)
    W_new = jnp.clip(W_new, 0.0, _W_max)

    return dT_dt, dq_v_dt, T_land_new, W_new, precip, sw_net_sfc, lw_net_sfc


# ---------------------------------------------------------------------------
# 7. Time integration
# ---------------------------------------------------------------------------
n_steps = int(N_DAYS * 86400 / DT)
diag_interval = int(DIAG_DAYS * 86400 / DT)

# Diagnostics storage
diag_times = []
diag_Tland = []
diag_T_atm = []
diag_T_low = []
diag_max_wind = []
diag_precip = []
diag_W_bucket = []
diag_CWV = []
diag_profiles_T = []    # global-mean T profiles (nlev,) at each diag time
diag_profiles_qv = []   # global-mean q_v profiles (nlev,) in g/kg
diag_sigma = np.asarray(sigma.sigma_full)  # vertical coordinate for profile plots

# 2D snapshots at selected days (face 0 only, for evolution plots)
snapshot_days = {10, 30, 60, 100, 200, 300, int(N_DAYS)}
snapshots = {}  # day -> dict of 2D arrays

print(f"\n  Starting integration: {n_steps} steps ({N_DAYS} days)")
print(f"  {'Day':>6s}  {'<T_land>':>8s}  {'<T_atm>':>8s}  {'<T_low>':>8s}"
      f"  {'<Precip>':>8s}  {'<W>':>6s}  {'<CWV>':>6s}  {'max|v|':>8s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}"
      f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}")

t_wall_start = time.time()

# JIT warmup: dynamics only
state = model.step_with_physics(state, DT)

# JIT warmup: operator-split physics
dT_dt, dq_v_dt, T_land, W_bucket, _precip, _sw, _lw = physics_step(
    state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
    T_land, W_bucket, grid.lat, DT,
)
new_T = state.T.data + DT * dT_dt
q_v = jnp.maximum(q_v + DT * dq_v_dt, 0.0)
# Large-scale condensation: remove supersaturation, release latent heat
q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
_excess = jnp.maximum(q_v - q_sat, 0.0)
q_v = q_v - _excess
new_T = new_T + constants.L_v * _excess / constants.c_pd
state = state._replace(T=state.T.replace(data=new_T))
# Large-scale precipitation [kg/m2/s]
_precip_ls = jnp.sum(_excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)
# Edge blending on q_v
q_v = blend_scalar_cube_edges(q_v, 0.10, width=2)
q_v = jnp.maximum(q_v, 0.0)

# Rayleigh friction (BL + free-atmosphere drag)
state = state._replace(
    u=state.u.replace(data=state.u.data * _fric_decay),
    v=state.v.replace(data=state.v.data * _fric_decay),
)

jax.block_until_ready(state.u.data)
t_jit = time.time() - t_wall_start
print(f"  JIT compiled in {t_jit:.1f}s")

t_wall_start = time.time()

for step in range(1, n_steps):
    # (a) Dynamics only (no physics_fn — all physics are operator-split)
    state = model.step_with_physics(state, DT)

    # (b) Operator-split physics
    dT_dt, dq_v_dt, T_land, W_bucket, _precip, _sw, _lw = physics_step(
        state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
        T_land, W_bucket, grid.lat, DT,
    )
    new_T = state.T.data + DT * dT_dt
    q_v = jnp.maximum(q_v + DT * dq_v_dt, 0.0)
    # Large-scale condensation: remove supersaturation, release latent heat
    q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
    _excess = jnp.maximum(q_v - q_sat, 0.0)
    q_v = q_v - _excess
    new_T = new_T + constants.L_v * _excess / constants.c_pd
    state = state._replace(T=state.T.replace(data=new_T))
    # Large-scale precipitation [kg/m2/s]
    _precip_ls = jnp.sum(_excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)
    # Edge blending on q_v
    q_v = blend_scalar_cube_edges(q_v, 0.10, width=2)
    q_v = jnp.maximum(q_v, 0.0)

    # (c) Rayleigh friction (BL + free-atmosphere drag)
    state = state._replace(
        u=state.u.replace(data=state.u.data * _fric_decay),
        v=state.v.replace(data=state.v.data * _fric_decay),
    )

    # Diagnostics
    if (step + 1) % diag_interval == 0:
        jax.block_until_ready(state.u.data)
        day = (step + 1) * DT / 86400.0
        mean_Tland = float(jnp.mean(T_land))
        mean_T = float(jnp.mean(state.T.data))
        mean_T_low = float(jnp.mean(state.T.data[..., -1]))
        max_v = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))

        # Precipitation [mm/day]
        mean_precip = float(jnp.mean(_precip + _precip_ls)) * 86400.0

        # Bucket moisture [kg/m2]
        mean_W = float(jnp.mean(W_bucket))

        # Column water vapor [kg/m2]
        cwv = column_water_vapor(q_v, state.p_s.data, _dsigma)
        mean_cwv = float(jnp.mean(cwv))

        diag_times.append(day)
        diag_Tland.append(mean_Tland)
        diag_T_atm.append(mean_T)
        diag_T_low.append(mean_T_low)
        diag_max_wind.append(max_v)
        diag_precip.append(mean_precip)
        diag_W_bucket.append(mean_W)
        diag_CWV.append(mean_cwv)

        # Vertical profiles (global mean)
        diag_profiles_T.append(np.asarray(jnp.mean(state.T.data, axis=(0, 1, 2))))
        diag_profiles_qv.append(np.asarray(jnp.mean(q_v, axis=(0, 1, 2))) * 1000.0)

        # 2D snapshots at selected days (face 0)
        iday = int(round(day))
        if iday in snapshot_days:
            snapshots[iday] = {
                'T_land': np.asarray(T_land[0]),
                'T_low': np.asarray(state.T.data[0, :, :, -1]),
                'q_v_low': np.asarray(q_v[0, :, :, -1]) * 1000.0,
                'precip': np.asarray((_precip + _precip_ls)[0]) * 86400.0,
                'wind': np.asarray(
                    jnp.sqrt(state.u.data[0, :, :, -1]**2
                             + state.v.data[0, :, :, -1]**2)
                ),
            }

        print(f"  {day:6.0f}  {mean_Tland:8.2f}  {mean_T:8.2f}  {mean_T_low:8.2f}"
              f"  {mean_precip:8.2f}  {mean_W:6.1f}  {mean_cwv:6.1f}  {max_v:8.2f}")

        # Stability check
        if not jnp.all(jnp.isfinite(state.u.data)) or max_v > 500:
            print(f"  BLOWUP detected at day {day:.0f}")
            break

jax.block_until_ready(state.u.data)
total_wall = time.time() - t_wall_start
print(f"\n  Integration complete: {total_wall:.1f}s wall time")

# ---------------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / "results.txt", "w") as f:
    f.write(f"Moist RCE: Slab Land + Gray Atmosphere + Bucket Hydrology\n")
    f.write(f"Grid: C{N}/L{NLEV}, dt={DT}s, {N_DAYS} days\n")
    f.write(f"Physics: gray radiation (moisture-coupled) + SBM convection (operator-split)\n")
    f.write(f"Land: d_soil={land_config.d_soil}m, C_land={_C_land:.0e} J/m2/K\n")
    f.write(f"Moisture: RH_init={_RH_init}, BL: bulk aero C_H=C_E={_C_H}\n")
    f.write(f"Bucket: W_max={_W_max} kg/m2, beta_min={_beta_min}\n\n")
    f.write(f"{'Day':>6s}  {'<T_land>':>10s}  {'<T_atm>':>10s}  {'<T_low>':>10s}"
            f"  {'<Precip>':>10s}  {'<W>':>10s}  {'<CWV>':>10s}  {'max|v|':>10s}\n")
    for i in range(len(diag_times)):
        f.write(f"{diag_times[i]:6.0f}  {diag_Tland[i]:10.3f}  {diag_T_atm[i]:10.3f}"
                f"  {diag_T_low[i]:10.3f}  {diag_precip[i]:10.3f}  {diag_W_bucket[i]:10.1f}"
                f"  {diag_CWV[i]:10.1f}  {diag_max_wind[i]:10.3f}\n")
    f.write(f"\nTotal wall time: {total_wall:.1f}s\n")
    if diag_Tland:
        f.write(f"Final <T_land>: {diag_Tland[-1]:.3f} K\n")
        f.write(f"Final <T_atm>: {diag_T_atm[-1]:.3f} K\n")
        f.write(f"Final <Precip>: {diag_precip[-1]:.2f} mm/day\n")
        f.write(f"Final <W_bucket>: {diag_W_bucket[-1]:.1f} kg/m2\n")
        f.write(f"Final <CWV>: {diag_CWV[-1]:.1f} kg/m2\n")

# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

    # Panel 1: T_land and T_low
    axes[0].plot(diag_times, diag_Tland, "b-o", markersize=3, label="T_land")
    axes[0].plot(diag_times, diag_T_low, "r--s", markersize=3, label="T_low (atm)")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Moist RCE: Slab Land + Gray Radiation + SBM Convection")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: T_atm
    axes[1].plot(diag_times, diag_T_atm, "r-o", markersize=3)
    axes[1].set_ylabel("Global-mean T_atm [K]")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Max wind
    axes[2].plot(diag_times, diag_max_wind, "g-o", markersize=3)
    axes[2].set_ylabel("Max wind speed [m/s]")
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Precipitation
    axes[3].plot(diag_times, diag_precip, "c-o", markersize=3)
    axes[3].set_ylabel("Precip [mm/day]")
    axes[3].grid(True, alpha=0.3)

    # Panel 5: W_bucket and CWV
    ax5a = axes[4]
    ax5b = ax5a.twinx()
    l1, = ax5a.plot(diag_times, diag_W_bucket, "brown", marker="o", markersize=3, label="W_bucket")
    l2, = ax5b.plot(diag_times, diag_CWV, "blue", marker="s", markersize=3, label="CWV")
    ax5a.set_ylabel("W_bucket [kg/m2]")
    ax5b.set_ylabel("CWV [kg/m2]")
    ax5a.set_xlabel("Time [days]")
    ax5a.legend(handles=[l1, l2], loc="center right")
    ax5a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rce_timeseries.png", dpi=150)
    plt.close()
    print(f"  Saved timeseries plot to {OUTPUT_DIR / 'rce_timeseries.png'}")
except ImportError:
    print("  matplotlib not available, skipping plot")

# Final state snapshot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_Tland = np.asarray(T_land)
    final_T_sfc = np.asarray(state.T.data[..., -1])
    u_sfc = np.asarray(state.u.data[..., -1])
    v_sfc = np.asarray(state.v.data[..., -1])
    wind_sfc = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    q_sfc = np.asarray(q_v[..., -1]) * 1000.0  # g/kg
    precip_map = np.asarray(_precip + _precip_ls) * 86400.0  # mm/day

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))

    im0 = axes[0].imshow(final_Tland[0], origin="lower", cmap="coolwarm")
    axes[0].set_title(f"T_land [K] (face 0, day {N_DAYS})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(final_T_sfc[0], origin="lower", cmap="coolwarm")
    axes[1].set_title(f"Lowest-level T [K] (face 0)")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(wind_sfc[0], origin="lower", cmap="magma")
    axes[2].set_title(f"Surface wind [m/s] (face 0)")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(q_sfc[0], origin="lower", cmap="YlGnBu")
    axes[3].set_title(f"q_v lowest level [g/kg] (face 0)")
    plt.colorbar(im3, ax=axes[3])

    im4 = axes[4].imshow(precip_map[0], origin="lower", cmap="Blues")
    axes[4].set_title(f"Precip [mm/day] (face 0)")
    plt.colorbar(im4, ax=axes[4])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rce_final_state.png", dpi=150)
    plt.close()
    print(f"  Saved final state plot to {OUTPUT_DIR / 'rce_final_state.png'}")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 10. Vertical profile evolution
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if diag_profiles_T:
        n_diag = len(diag_profiles_T)
        # Select ~8 evenly spaced profiles for clarity
        n_show = min(8, n_diag)
        indices = np.linspace(0, n_diag - 1, n_show, dtype=int)
        colors = cm.viridis(np.linspace(0, 1, n_show))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        for i, idx in enumerate(indices):
            day_label = f"day {diag_times[idx]:.0f}"
            ax1.plot(diag_profiles_T[idx], diag_sigma, color=colors[i], label=day_label)
            ax2.plot(diag_profiles_qv[idx], diag_sigma, color=colors[i], label=day_label)

        ax1.set_xlabel("Temperature [K]")
        ax1.set_ylabel("Sigma")
        ax1.set_title("Global-mean temperature profiles")
        ax1.invert_yaxis()
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Specific humidity [g/kg]")
        ax2.set_ylabel("Sigma")
        ax2.set_title("Global-mean moisture profiles")
        ax2.invert_yaxis()
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rce_profiles.png", dpi=150)
        plt.close()
        print(f"  Saved profile evolution to {OUTPUT_DIR / 'rce_profiles.png'}")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 11. 2D snapshot evolution (face 0)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snap_days_sorted = sorted(snapshots.keys())
    n_snap = len(snap_days_sorted)

    if n_snap > 0:
        fields = [
            ('T_land', 'T_land [K]', 'coolwarm'),
            ('q_v_low', 'q_v lowest [g/kg]', 'YlGnBu'),
            ('precip', 'Precip [mm/day]', 'Blues'),
            ('wind', 'Wind speed [m/s]', 'magma'),
        ]
        n_fields = len(fields)

        fig, axes = plt.subplots(n_fields, n_snap, figsize=(4 * n_snap, 3.5 * n_fields))
        if n_snap == 1:
            axes = axes[:, None]

        for j, day in enumerate(snap_days_sorted):
            snap = snapshots[day]
            for i, (key, label, cmap) in enumerate(fields):
                ax = axes[i, j]
                im = ax.imshow(snap[key], origin="lower", cmap=cmap)
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title(f"{label}\nDay {day}", fontsize=11, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(label, fontsize=10)
                ax.tick_params(labelsize=7)

        plt.suptitle("2D snapshots (face 0)", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rce_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved 2D snapshots to {OUTPUT_DIR / 'rce_snapshots.png'}")
except ImportError:
    pass

print(f"\n  Results saved to {OUTPUT_DIR}/")
print("  Done.")
