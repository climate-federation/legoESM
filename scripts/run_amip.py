#!/usr/bin/env python
"""AMIP simulation with prescribed SST and sea-ice forcing.

Aquaplanet/realistic-boundary experiment: gray radiation (Frierson 2006)
with moisture-coupled optical depth + SBM convection + prescribed
SST/sea-ice from PCMDI reference datasets (COBE-SST2 or HadISST).

Coupling strategy: operator splitting (same as slab ocean RCE).
  1. Dynamics only: step_with_physics (JIT-compiled, RK3, no physics_fn)
  2. All physics operator-split in a single JIT function:
     - Gray radiation (moisture-dependent LW optical depth, seasonal solar)
     - SBM convection (produces precipitation)
     - Bulk aerodynamic BL coupling (heat + moisture, wind-dependent)
     - Large-scale condensation (saturation adjustment with latent heating)
  3. Rayleigh friction with weak free-atmosphere drag (k_free = 0.1/day)

Key differences from slab ocean RCE:
  - SST prescribed from NetCDF forcing data (no slab ocean)
  - Sea-ice concentration blends T_sfc and surface albedo
  - Seasonal solar cycle (daily_mean_insolation, not perpetual equinox)

Usage:
    JAX_ENABLE_X64=1 python scripts/run_amip.py \
        --dataset cobe --forcing-path /path/to/MODEL.SST.COBE-SST2.nc \
        --start-day 0 --days 30 --resolution 16 --dt 600
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
parser = argparse.ArgumentParser(description="AMIP simulation with prescribed SST/SIC")
parser.add_argument("--dataset", type=str, default="cobe",
                    choices=["cobe", "hadisst", "custom"],
                    help="Forcing dataset preset")
parser.add_argument("--forcing-path", type=str, required=True,
                    help="Path to NetCDF forcing file")
parser.add_argument("--sst-var", type=str, default=None,
                    help="SST variable name (for custom dataset)")
parser.add_argument("--sic-var", type=str, default=None,
                    help="SIC variable name (for custom dataset)")
parser.add_argument("--sst-offset", type=float, default=None,
                    help="SST offset (e.g., 273.15 if in Celsius)")
parser.add_argument("--sic-scale", type=float, default=None,
                    help="SIC scale (e.g., 0.01 if in percent)")
parser.add_argument("--start-day", type=float, default=0.0,
                    help="Start day within forcing record")
parser.add_argument("--days", type=int, default=200,
                    help="Integration length [days]")
parser.add_argument("--resolution", type=int, default=16,
                    help="Cubed-sphere N")
parser.add_argument("--nlev", type=int, default=20,
                    help="Number of vertical levels")
parser.add_argument("--dt", type=float, default=600.0,
                    help="Time step [s]")
parser.add_argument("--diag-days", type=int, default=5,
                    help="Diagnostic interval [days]")
args = parser.parse_args()

N = args.resolution
NLEV = args.nlev
DT = args.dt
N_DAYS = args.days
DIAG_DAYS = args.diag_days
START_DAY = args.start_day

OUTPUT_DIR = Path("results/amip")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  AMIP: Prescribed SST/SIC + Gray Atmosphere + SBM Convection")
print("=" * 70)
print(f"  Grid:       C{N} / L{NLEV}")
print(f"  dt:         {DT:.0f} s")
print(f"  Duration:   {N_DAYS} days (start day {START_DAY})")
print(f"  Diagnostics every {DIAG_DAYS} days")
print(f"  Dataset:    {args.dataset}")
print(f"  Forcing:    {args.forcing_path}")
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
# 2. Load AMIP forcing
# ---------------------------------------------------------------------------
from legoesm.forcing.amip import (
    AMIPForcingConfig,
    get_amip_preset,
    load_amip_forcing,
    get_forcing_at_time,
)

if args.dataset == "custom":
    forcing_config = AMIPForcingConfig(
        dataset="custom",
        path=args.forcing_path,
        sst_var=args.sst_var or "sst",
        sic_var=args.sic_var or "sic",
        sst_offset=args.sst_offset if args.sst_offset is not None else 0.0,
        sic_scale=args.sic_scale if args.sic_scale is not None else 1.0,
    )
else:
    forcing_config = get_amip_preset(args.dataset)._replace(path=args.forcing_path)

print(f"  Loading forcing data...")
t_load = time.time()
forcing = load_amip_forcing(forcing_config, grid)
t_load = time.time() - t_load
print(f"  Forcing loaded in {t_load:.1f}s: {forcing.times.shape[0]} time records")
print(f"  Time range: day {float(forcing.times[0]):.0f} to {float(forcing.times[-1]):.0f}")

# Check start_day is within range
if START_DAY < float(forcing.times[0]) or START_DAY > float(forcing.times[-1]):
    print(f"  WARNING: start_day={START_DAY} outside forcing range "
          f"[{float(forcing.times[0]):.0f}, {float(forcing.times[-1]):.0f}]")

# Show initial SST/SIC
sst_init, sic_init = get_forcing_at_time(forcing, START_DAY)
print(f"  Initial SST: min={float(jnp.min(sst_init)):.1f} K, "
      f"max={float(jnp.max(sst_init)):.1f} K, "
      f"mean={float(jnp.mean(sst_init)):.1f} K")
print(f"  Initial SIC: mean={float(jnp.mean(sic_init)):.3f}, "
      f"max={float(jnp.max(sic_init)):.3f}")

# Sea-ice parameters
_T_ice = forcing_config.T_ice
_albedo_ice = forcing_config.albedo_ice
_albedo_ocean = forcing_config.albedo_ocean

# ---------------------------------------------------------------------------
# 3. Atmospheric model (hydrostatic primitive equations)
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
)
model = PrimitiveEquationModel(grid, sigma, dycore_config)

# Initial state: isothermal at rest (280 K)
state = held_suarez_init(grid, sigma, T_init=280.0)
print(f"  Atmosphere initialized: T=280.0 K isothermal")

# ---------------------------------------------------------------------------
# 4. Physics configuration
# ---------------------------------------------------------------------------
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.convection.config import SBMConfig

# Gray radiation -- use perpetual_equinox=False for seasonal cycle
# sfc_albedo here is a placeholder; actual albedo is computed per-column
# from sea-ice blending and applied post-radiation.
gray_config = GrayRadiationConfig(
    tau_equator=7.2,
    tau_pole=1.8,
    S_0=1360.0,
    sfc_albedo=_albedo_ocean,   # base ocean albedo (ice effect applied post-rad)
    perpetual_equinox=False,    # seasonal solar cycle
)

# SBM convection
sbm_config = SBMConfig(tau_c=7200.0, RH_ref=0.7)

print(f"  Physics: gray radiation (moist, seasonal) + SBM convection (operator-split)")

# ---------------------------------------------------------------------------
# 5. Moisture initialization (60% RH)
# ---------------------------------------------------------------------------
from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
from legoesm.atmosphere.physics.convection.sbm import sbm_convection

_RH_init = 0.6
p_full_init = state.p_s.data[..., None] * sigma.sigma_full
q_sat_init = saturation_mixing_ratio(state.T.data, p_full_init)
q_v = _RH_init * q_sat_init * sigma.sigma_full ** 2
q_v = jnp.minimum(q_v, q_sat_init)
mean_qv = float(jnp.mean(q_v)) * 1000.0
cwv_init = float(jnp.mean(
    jnp.sum(q_v * state.p_s.data[..., None] * sigma.dsigma, axis=-1) / constants.g
))
print(f"  Moisture: RH_init={_RH_init}, mean q_v={mean_qv:.2f} g/kg, CWV={cwv_init:.1f} kg/m2")

# ---------------------------------------------------------------------------
# 6. Operator-split physics step
# ---------------------------------------------------------------------------
_S_0 = gray_config.S_0
_sigma_full = sigma.sigma_full
_sigma_half = sigma.sigma_half
_dsigma = sigma.dsigma
_C_H = 1.5e-3   # bulk transfer coefficient for heat
_C_E = 1.5e-3   # bulk transfer coefficient for moisture

# Rayleigh friction: BL drag + weak free-atmosphere drag
_sigma_b = 0.7
_k_f_max = 1.0 / 86400.0  # BL drag: 1/day [1/s]
_k_free = 0.1 / 86400.0   # Free-atmosphere drag: 0.1/day [1/s]
_k_f = _k_free + _k_f_max * jnp.maximum(0.0, (_sigma_full - _sigma_b) / (1.0 - _sigma_b))
_fric_decay = jnp.exp(-_k_f * DT)


@jax.jit
def physics_step(T, p_s, q_v, u, v, sst, sic, lat, day_of_year, dt):
    """Operator-split physics with prescribed SST and seasonal solar.

    Returns
    -------
    dT_dt       : (6,N,N,nlev) atmospheric temperature tendency [K/s]
    dq_v_dt     : (6,N,N,nlev) moisture tendency [kg/kg/s]
    precip      : (6,N,N)      precipitation rate [kg/m2/s]
    sw_net_sfc  : (6,N,N)      net SW at surface [W/m2]
    lw_net_sfc  : (6,N,N)      net LW at surface [W/m2]
    """
    nlev = _sigma_full.shape[0]
    shape_3d = T.shape
    shape_2d = p_s.shape
    ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

    # Blended surface temperature and albedo
    T_sfc = sic * _T_ice + (1.0 - sic) * sst
    albedo = sic * _albedo_ice + (1.0 - sic) * _albedo_ocean

    # Pressure on full and half levels
    p_full = p_s[..., None] * _sigma_full
    p_half = p_s[..., None] * _sigma_half

    # Reshape to columns
    T_col = T.reshape(ncol, nlev)
    p_full_col = p_full.reshape(ncol, nlev)
    p_half_col = p_half.reshape(ncol, nlev + 1)
    q_v_col = q_v.reshape(ncol, nlev)
    T_sfc_col = T_sfc.reshape(ncol)
    lat_col = lat.reshape(ncol)

    # Seasonal TOA insolation
    insol = daily_mean_insolation(lat_col, day_of_year, _S_0)

    # --- (a) Gray radiation with moisture feedback ---
    rad_out = gray_radiation(
        T=T_col,
        p_full=p_full_col,
        p_half=p_half_col,
        sfc_temperature=T_sfc_col,
        lat=lat_col,
        q_v=q_v_col,
        insolation=insol,
        config=gray_config,
    )
    dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

    # Surface fluxes — apply spatially varying albedo post-radiation
    # The gray radiation uses ocean albedo internally; correct for ice:
    sw_down_sfc = rad_out.sw_flux_down[:, -1].reshape(shape_2d)
    sw_net_sfc = sw_down_sfc * (1.0 - albedo)
    lw_net_sfc = (rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]).reshape(shape_2d)

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
    precip = conv_out.precipitation.reshape(shape_2d)

    # --- (c) Bulk aerodynamic BL coupling ---
    rho_low = (p_s * _sigma_full[-1]) / (constants.R_d * T[..., -1])
    wind_speed = jnp.sqrt(u[..., -1]**2 + v[..., -1]**2 + 1.0)  # 1 m/s gustiness
    dp_low = p_s * (_sigma_half[-1] - _sigma_half[-2])

    # Sensible heat flux [W/m2] (positive upward)
    shflx = rho_low * constants.c_pd * _C_H * wind_speed * (T_sfc - T[..., -1])

    # Latent heat flux [W/m2] (saturated surface)
    q_sat_sfc = saturation_mixing_ratio(T_sfc, p_s)
    lhflx = rho_low * constants.L_v * _C_E * wind_speed * (q_sat_sfc - q_v[..., -1])
    evap_rate = lhflx / constants.L_v

    # Atmospheric tendencies (lowest level only)
    dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
    dq_BL = constants.g * evap_rate / dp_low

    # --- (d) Total atmospheric tendencies ---
    dT_dt = dT_dt_rad + dT_dt_conv
    dT_dt = dT_dt.at[..., -1].add(dT_BL)

    dq_v_dt = dq_v_dt_conv
    dq_v_dt = dq_v_dt.at[..., -1].add(dq_BL)

    return dT_dt, dq_v_dt, precip, sw_net_sfc, lw_net_sfc


# ---------------------------------------------------------------------------
# 7. Time integration
# ---------------------------------------------------------------------------
n_steps = int(N_DAYS * 86400 / DT)
diag_interval = int(DIAG_DAYS * 86400 / DT)

# Diagnostics storage
diag_times = []
diag_sst = []
diag_sic = []
diag_T_atm = []
diag_T_low = []
diag_max_wind = []
diag_precip = []
diag_CWV = []
diag_profiles_T = []
diag_profiles_qv = []
diag_sigma = np.asarray(sigma.sigma_full)

# 2D snapshots at selected days
snapshot_days = {10, 30, 60, 100, 200, 300}
snapshots = {}

print(f"\n  Starting integration: {n_steps} steps ({N_DAYS} days)")
print(f"  {'Day':>6s}  {'<SST>':>8s}  {'<SIC>':>6s}  {'<T_atm>':>8s}"
      f"  {'<T_low>':>8s}  {'<Precip>':>8s}  {'<CWV>':>6s}  {'max|v|':>8s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}"
      f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")

t_wall_start = time.time()

# JIT warmup
day = START_DAY
day_of_year = day % 365.0 + 1.0  # 1-365 for solar

sst, sic = get_forcing_at_time(forcing, day)

state = model.step_with_physics(state, DT)

dT_dt, dq_v_dt, _precip, _sw, _lw = physics_step(
    state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
    sst, sic, grid.lat, day_of_year, DT,
)
new_T = state.T.data + DT * dT_dt
q_v = jnp.maximum(q_v + DT * dq_v_dt, 0.0)
# Large-scale condensation
q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
_excess = jnp.maximum(q_v - q_sat, 0.0)
q_v = q_v - _excess
new_T = new_T + constants.L_v * _excess / constants.c_pd
state = state._replace(T=state.T.replace(data=new_T))
_precip_ls = jnp.sum(_excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)

# Rayleigh friction
state = state._replace(
    u=state.u.replace(data=state.u.data * _fric_decay),
    v=state.v.replace(data=state.v.data * _fric_decay),
)

jax.block_until_ready(state.u.data)
t_jit = time.time() - t_wall_start
print(f"  JIT compiled in {t_jit:.1f}s")

t_wall_start = time.time()

for step in range(1, n_steps):
    day = START_DAY + (step + 1) * DT / 86400.0
    day_of_year = day % 365.0 + 1.0

    # Get prescribed SST and SIC for current time
    sst, sic = get_forcing_at_time(forcing, day)

    # (a) Dynamics only
    state = model.step_with_physics(state, DT)

    # (b) Operator-split physics
    dT_dt, dq_v_dt, _precip, _sw, _lw = physics_step(
        state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
        sst, sic, grid.lat, day_of_year, DT,
    )
    new_T = state.T.data + DT * dT_dt
    q_v = jnp.maximum(q_v + DT * dq_v_dt, 0.0)
    # Large-scale condensation
    q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
    _excess = jnp.maximum(q_v - q_sat, 0.0)
    q_v = q_v - _excess
    new_T = new_T + constants.L_v * _excess / constants.c_pd
    state = state._replace(T=state.T.replace(data=new_T))
    _precip_ls = jnp.sum(_excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)

    # (c) Rayleigh friction
    state = state._replace(
        u=state.u.replace(data=state.u.data * _fric_decay),
        v=state.v.replace(data=state.v.data * _fric_decay),
    )

    # Diagnostics
    if (step + 1) % diag_interval == 0:
        jax.block_until_ready(state.u.data)
        mean_sst = float(jnp.mean(sst))
        mean_sic = float(jnp.mean(sic))
        mean_T = float(jnp.mean(state.T.data))
        mean_T_low = float(jnp.mean(state.T.data[..., -1]))
        max_v = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))

        # Precipitation [mm/day] (convective + large-scale)
        mean_precip = float(jnp.mean(_precip + _precip_ls)) * 86400.0

        # Column water vapor [kg/m2]
        cwv = jnp.sum(q_v * state.p_s.data[..., None] * _dsigma, axis=-1) / constants.g
        mean_cwv = float(jnp.mean(cwv))

        elapsed_day = day - START_DAY
        diag_times.append(elapsed_day)
        diag_sst.append(mean_sst)
        diag_sic.append(mean_sic)
        diag_T_atm.append(mean_T)
        diag_T_low.append(mean_T_low)
        diag_max_wind.append(max_v)
        diag_precip.append(mean_precip)
        diag_CWV.append(mean_cwv)

        # Vertical profiles
        diag_profiles_T.append(np.asarray(jnp.mean(state.T.data, axis=(0, 1, 2))))
        diag_profiles_qv.append(np.asarray(jnp.mean(q_v, axis=(0, 1, 2))) * 1000.0)

        # 2D snapshots
        iday = int(round(elapsed_day))
        if iday in snapshot_days:
            T_sfc_snap = sic * _T_ice + (1.0 - sic) * sst
            snapshots[iday] = {
                'SST': np.asarray(sst[0]),
                'SIC': np.asarray(sic[0]),
                'T_sfc': np.asarray(T_sfc_snap[0]),
                'T_low': np.asarray(state.T.data[0, :, :, -1]),
                'q_v_low': np.asarray(q_v[0, :, :, -1]) * 1000.0,
                'precip': np.asarray((_precip + _precip_ls)[0]) * 86400.0,
                'wind': np.asarray(
                    jnp.sqrt(state.u.data[0, :, :, -1]**2
                             + state.v.data[0, :, :, -1]**2)
                ),
            }

        print(f"  {elapsed_day:6.0f}  {mean_sst:8.2f}  {mean_sic:6.3f}  {mean_T:8.2f}"
              f"  {mean_T_low:8.2f}  {mean_precip:8.2f}  {mean_cwv:6.1f}  {max_v:8.2f}")

        # Stability check
        if not jnp.all(jnp.isfinite(state.u.data)) or max_v > 500:
            print(f"  BLOWUP detected at day {elapsed_day:.0f}")
            break

jax.block_until_ready(state.u.data)
total_wall = time.time() - t_wall_start
print(f"\n  Integration complete: {total_wall:.1f}s wall time")

# ---------------------------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / "results.txt", "w") as f:
    f.write(f"AMIP: Prescribed SST/SIC + Gray Atmosphere + SBM Convection\n")
    f.write(f"Grid: C{N}/L{NLEV}, dt={DT}s, {N_DAYS} days (start day {START_DAY})\n")
    f.write(f"Dataset: {args.dataset}, file: {args.forcing_path}\n")
    f.write(f"Physics: gray radiation (moist, seasonal) + SBM convection (operator-split)\n")
    f.write(f"BL: bulk aero C_H=C_E={_C_H}\n")
    f.write(f"Friction: k_free={_k_free*86400:.1f}/day, k_BL_max={_k_f_max*86400:.1f}/day\n")
    f.write(f"Sea-ice: T_ice={_T_ice} K, albedo_ice={_albedo_ice}, albedo_ocean={_albedo_ocean}\n\n")
    f.write(f"{'Day':>6s}  {'<SST>':>10s}  {'<SIC>':>8s}  {'<T_atm>':>10s}"
            f"  {'<T_low>':>10s}  {'<Precip>':>10s}  {'<CWV>':>10s}  {'max|v|':>10s}\n")
    for i in range(len(diag_times)):
        f.write(f"{diag_times[i]:6.0f}  {diag_sst[i]:10.3f}  {diag_sic[i]:8.4f}"
                f"  {diag_T_atm[i]:10.3f}  {diag_T_low[i]:10.3f}"
                f"  {diag_precip[i]:10.3f}  {diag_CWV[i]:10.1f}"
                f"  {diag_max_wind[i]:10.3f}\n")
    f.write(f"\nTotal wall time: {total_wall:.1f}s\n")
    if diag_sst:
        f.write(f"Final <SST>: {diag_sst[-1]:.3f} K\n")
        f.write(f"Final <SIC>: {diag_sic[-1]:.4f}\n")
        f.write(f"Final <T_atm>: {diag_T_atm[-1]:.3f} K\n")
        f.write(f"Final <Precip>: {diag_precip[-1]:.2f} mm/day\n")
        f.write(f"Final <CWV>: {diag_CWV[-1]:.1f} kg/m2\n")

# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=True)

    # Panel 1: SST and T_low
    axes[0].plot(diag_times, diag_sst, "b-o", markersize=3, label="SST (prescribed)")
    axes[0].plot(diag_times, diag_T_low, "r--s", markersize=3, label="T_low (atm)")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("AMIP: Prescribed SST/SIC + Gray Radiation + SBM Convection")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Sea-ice concentration
    axes[1].plot(diag_times, diag_sic, "c-o", markersize=3)
    axes[1].set_ylabel("Mean SIC")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: T_atm
    axes[2].plot(diag_times, diag_T_atm, "r-o", markersize=3)
    axes[2].set_ylabel("Global-mean T_atm [K]")
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Max wind
    axes[3].plot(diag_times, diag_max_wind, "g-o", markersize=3)
    axes[3].set_ylabel("Max wind speed [m/s]")
    axes[3].grid(True, alpha=0.3)

    # Panel 5: Precipitation
    axes[4].plot(diag_times, diag_precip, "c-o", markersize=3)
    axes[4].set_ylabel("Precip [mm/day]")
    axes[4].grid(True, alpha=0.3)

    # Panel 6: CWV
    axes[5].plot(diag_times, diag_CWV, "blue", marker="s", markersize=3)
    axes[5].set_ylabel("CWV [kg/m2]")
    axes[5].set_xlabel("Time [days]")
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amip_timeseries.png", dpi=150)
    plt.close()
    print(f"  Saved timeseries plot to {OUTPUT_DIR / 'amip_timeseries.png'}")
except ImportError:
    print("  matplotlib not available, skipping plot")

# Final state snapshot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final_T_sfc = np.asarray(sic * _T_ice + (1.0 - sic) * sst)
    u_sfc = np.asarray(state.u.data[..., -1])
    v_sfc = np.asarray(state.v.data[..., -1])
    wind_sfc = np.sqrt(u_sfc ** 2 + v_sfc ** 2)
    q_sfc = np.asarray(q_v[..., -1]) * 1000.0
    precip_map = np.asarray(_precip + _precip_ls) * 86400.0
    sic_map = np.asarray(sic)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    im0 = axes[0, 0].imshow(np.asarray(sst[0]), origin="lower", cmap="coolwarm")
    axes[0, 0].set_title(f"SST [K] (face 0)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(sic_map[0], origin="lower", cmap="Blues_r", vmin=0, vmax=1)
    axes[0, 1].set_title(f"SIC (face 0)")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(final_T_sfc[0], origin="lower", cmap="coolwarm")
    axes[0, 2].set_title(f"Blended T_sfc [K] (face 0)")
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(wind_sfc[0], origin="lower", cmap="magma")
    axes[1, 0].set_title(f"Surface wind [m/s] (face 0)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(q_sfc[0], origin="lower", cmap="YlGnBu")
    axes[1, 1].set_title(f"q_v lowest level [g/kg] (face 0)")
    plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(precip_map[0], origin="lower", cmap="Blues")
    axes[1, 2].set_title(f"Precip [mm/day] (face 0)")
    plt.colorbar(im5, ax=axes[1, 2])

    plt.suptitle(f"AMIP final state (day {N_DAYS})", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amip_final_state.png", dpi=150)
    plt.close()
    print(f"  Saved final state plot to {OUTPUT_DIR / 'amip_final_state.png'}")
except ImportError:
    pass

# Vertical profile evolution
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    if diag_profiles_T:
        n_diag = len(diag_profiles_T)
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
        plt.savefig(OUTPUT_DIR / "amip_profiles.png", dpi=150)
        plt.close()
        print(f"  Saved profile evolution to {OUTPUT_DIR / 'amip_profiles.png'}")
except ImportError:
    pass

# 2D snapshot evolution
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snap_days_sorted = sorted(snapshots.keys())
    n_snap = len(snap_days_sorted)

    if n_snap > 0:
        fields = [
            ('SST', 'SST [K]', 'coolwarm'),
            ('SIC', 'SIC', 'Blues_r'),
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
                if i == 0:
                    ax.set_title(f"Day {day}", fontsize=11, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(label, fontsize=10)
                ax.tick_params(labelsize=7)

        plt.suptitle("AMIP 2D snapshots (face 0)", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "amip_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved 2D snapshots to {OUTPUT_DIR / 'amip_snapshots.png'}")
except ImportError:
    pass

print(f"\n  Results saved to {OUTPUT_DIR}/")
print("  Done.")
