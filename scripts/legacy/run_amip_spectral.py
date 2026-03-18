#!/usr/bin/env python
"""AMIP simulation on the spectral (Gaussian grid) primitive equations.

Atmosphere-only integration with prescribed SST and sea-ice concentration.
Uses the spectral PE dycore (Bourke 1972, vorticity-divergence formulation)
with configurable radiation (gray or RRTMGP), SBM convection, bulk aerodynamic BL
exchange, large-scale condensation, and Rayleigh friction.

Physics is operator-split:
  1. Dynamics: spectral PE step (SSP-RK54 explicit or semi-implicit)
  2. Physics (on Gaussian grid):
     - Radiation (gray moisture-dependent LW OD, or RRTMGP correlated-k)
     - SBM convection
     - Bulk aerodynamic BL exchange (heat + moisture)
     - Large-scale condensation (saturation adjustment)
  3. Rayleigh friction (applied spectrally to vor/div)
  4. Temperature updated in spectral space after physics

Supports:
  - External NetCDF forcing (--forcing-path) with COBE-SST2 or HadISST
  - Analytical forcing (--forcing analytical) for idealized tests without data

Usage:
    JAX_ENABLE_X64=1 python scripts/run_amip_spectral.py \\
        --forcing analytical --days 365 --truncation 42 --dt 600

    JAX_ENABLE_X64=1 python scripts/run_amip_spectral.py \\
        --dataset hadisst --forcing-path /path/to/HadISST.nc \\
        --days 365 --truncation 42 --dt 600
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Spectral PE AMIP simulation with prescribed SST/SIC",
)
parser.add_argument("--dataset", type=str, default="analytical",
                    choices=["cobe", "hadisst", "custom", "analytical"],
                    help="Forcing preset (default: analytical)")
parser.add_argument("--forcing-path", type=str, default=None,
                    help="Path to NetCDF forcing file")
parser.add_argument("--sst-var", type=str, default=None)
parser.add_argument("--sic-var", type=str, default=None)
parser.add_argument("--sst-offset", type=float, default=None)
parser.add_argument("--sic-scale", type=float, default=None)
parser.add_argument("--start-day", type=float, default=0.0)
parser.add_argument("--days", type=int, default=365,
                    help="Integration length [days] (default: 365)")
parser.add_argument("--truncation", type=int, default=21,
                    help="Spectral truncation T (default: 21, ~5.6 deg; T42=~2.8 deg needs dt<=300)")
parser.add_argument("--nlev", type=int, default=40,
                    help="Number of vertical levels (default: 40)")
parser.add_argument("--vertical-coord", type=str, default="hybrid",
                    choices=["sigma", "hybrid"],
                    help="Vertical coordinate type (default: hybrid)")
parser.add_argument("--p-top", type=float, default=None,
                    help="Model top pressure [Pa] for hybrid coord (default: auto)")
parser.add_argument("--stretching", type=float, default=None,
                    help="Sinh stretching for BL resolution (default: auto)")
parser.add_argument("--dt", type=float, default=600.0,
                    help="Time step [s] (default: 600 for T21; use 300 for T42)")
parser.add_argument("--diag-days", type=int, default=5,
                    help="Diagnostic interval [days] (default: 5)")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--checkpoint-days", type=int, default=0)
parser.add_argument("--co2-ppmv", type=float, default=415.0,
                    help="CO2 concentration [ppmv] (stored in config; gray rad ignores it)")
parser.add_argument("--radiation", type=str, default="gray",
                    choices=["gray", "rrtmg", "rrtmgp"],
                    help="Radiation scheme: gray or rrtmgp (default: gray)")
parser.add_argument("--microphysics", type=str, default="none",
                    choices=["none", "kessler", "sundqvist", "seifert_beheng",
                             "morrison", "thompson"],
                    help="Microphysics scheme (default: none = saturation adjustment only)")
# Topography
parser.add_argument("--topography", type=str, default="flat",
                    help="Topography: flat or path to NetCDF file (default: flat)")
parser.add_argument("--topo-smoothing", type=int, default=4,
                    help="Laplacian smoothing passes for topography (default: 4)")
args = parser.parse_args()

if args.dataset != "analytical" and args.forcing_path is None:
    parser.error("--forcing-path is required for non-analytical forcing")

TRUNC = args.truncation
NLEV = args.nlev
DT = args.dt
N_DAYS = args.days
DIAG_DAYS = args.diag_days
START_DAY = args.start_day
CHECKPOINT_DAYS = args.checkpoint_days
MICROPHYSICS = args.microphysics
RADIATION = "rrtmgp" if args.radiation in ("rrtmg", "rrtmgp") else "gray"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.output is not None:
    OUTPUT_DIR = Path(args.output)
else:
    OUTPUT_DIR = Path(f"results/atmosphere/hydrostatic/amip_spectral/T{TRUNC}_L{NLEV}_{N_DAYS}d_{RUN_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"  AMIP Spectral: T{TRUNC}/L{NLEV} — {RADIATION.upper()} + SBM Convection")
print("=" * 70)
print(f"  Truncation: T{TRUNC}")
print(f"  dt:         {DT:.0f} s")
print(f"  Duration:   {N_DAYS} days (start day {START_DAY})")
print(f"  Forcing:    {args.dataset}")
print(f"  CO2:        {args.co2_ppmv} ppmv (config only)")
if MICROPHYSICS != "none":
    print(f"  Microphysics: {MICROPHYSICS}")
print(f"  Diagnostics every {DIAG_DAYS} days")
print(f"  Output:     {OUTPUT_DIR}")
print()

# ---------------------------------------------------------------------------
# 1. Grid and vertical coordinate
# ---------------------------------------------------------------------------
from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis,
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    sh_synthesis,
    sh_synthesis_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels

grid = create_gaussian_grid(TRUNC)
if args.vertical_coord == "hybrid":
    _p_top = args.p_top if args.p_top is not None else 200.0
    _stretch = args.stretching if args.stretching is not None else 2.0
    sigma = make_hybrid_levels(NLEV, p_top_Pa=_p_top, stretching=_stretch)
else:
    sigma = create_sigma_coordinate(NLEV)

n_lat = grid.n_lat
n_lon = grid.n_lon
ncol = n_lat * n_lon
shape_2d = (n_lat, n_lon)
shape_3d = (n_lat, n_lon, NLEV)

print(f"  Grid: {n_lat} x {n_lon} ({ncol} columns), {NLEV} levels")

# Area weights for diagnostics (Gaussian quadrature)
area_weights = np.asarray(grid.weights)  # (n_lat,)
area_2d = area_weights[:, None] * np.ones(n_lon)[None, :]  # (n_lat, n_lon)
area_2d = area_2d / area_2d.sum()  # normalized

# ---------------------------------------------------------------------------
# 1b. Topography
# ---------------------------------------------------------------------------
from legoesm.grids.topography import (
    TopographyConfig, load_real_topography,
)
from legoesm import constants
from legoesm.diagnostics.column_integrals import column_water_vapor

TOPOGRAPHY = args.topography
if TOPOGRAPHY == "flat":
    _phis_data = jnp.zeros(shape_2d)
    _f_land = jnp.zeros(shape_2d)
    _topo_label = "flat"
else:
    # Treat as path to NetCDF file
    _topo_config = TopographyConfig(
        source="file",
        path=TOPOGRAPHY,
        smoothing_passes=args.topo_smoothing,
        edge_blend_strength=0.0,  # no edge blending for Gaussian grid
    )
    _phis_data, _f_land = load_real_topography(grid, config=_topo_config)
    _z_max = float(jnp.max(_phis_data)) / constants.g
    _land_pct = float(jnp.mean(_f_land)) * 100
    _topo_label = f"real ({TOPOGRAPHY}), z_max={_z_max:.0f} m, land={_land_pct:.1f}%"

print(f"  Topography: {_topo_label}")

# ---------------------------------------------------------------------------
# 2. Load or create AMIP forcing
# ---------------------------------------------------------------------------
from legoesm import constants

# Sea-ice parameters (same defaults as cubed-sphere AMIP)
_T_ice = 271.35
_albedo_ice = 0.65
_albedo_ocean = 0.06

if args.dataset == "analytical":
    # Analytical zonally-symmetric SST + SIC with seasonal cycle
    # SST: warm tropics, cold poles (Qobs-like profile)
    # SIC: prescribed from SST < 271.35 K threshold
    print("  Using analytical SST/SIC forcing (zonally symmetric, seasonal)")

    lat_deg = np.degrees(np.asarray(grid.lat))  # (n_lat,)

    def get_sst_sic(day):
        sst_1d, sic_1d = analytical_sst_sic(lat_deg, day, T_ice=_T_ice)
        # Broadcast (n_lat,) → (n_lat, n_lon)
        return (
            jnp.broadcast_to(sst_1d[:, None], shape_2d),
            jnp.broadcast_to(sic_1d[:, None], shape_2d),
        )

    # Initial check
    sst_init, sic_init = get_sst_sic(START_DAY)
    print(f"  Initial SST: min={float(jnp.min(sst_init)):.1f} K, "
          f"max={float(jnp.max(sst_init)):.1f} K, "
          f"mean={float(jnp.mean(sst_init)):.1f} K")
    print(f"  Initial SIC: mean={float(jnp.mean(sic_init)):.3f}")

else:
    from legoesm.forcing.amip import (
        AMIPForcingConfig,
        get_amip_preset,
        load_amip_forcing,
        get_forcing_at_time,
    )

    if args.dataset == "custom":
        forcing_config = AMIPForcingConfig(
            dataset="custom", path=args.forcing_path,
            sst_var=args.sst_var or "sst",
            sic_var=args.sic_var or "sic",
            sst_offset=args.sst_offset if args.sst_offset is not None else 0.0,
            sic_scale=args.sic_scale if args.sic_scale is not None else 1.0,
        )
    else:
        forcing_config = get_amip_preset(args.dataset)._replace(path=args.forcing_path)

    _T_ice = forcing_config.T_ice
    _albedo_ice = forcing_config.albedo_ice
    _albedo_ocean = forcing_config.albedo_ocean

    print("  Loading forcing data...")
    t_load = time.time()
    forcing = load_amip_forcing(forcing_config, grid)
    t_load = time.time() - t_load
    print(f"  Forcing loaded in {t_load:.1f}s: {forcing.times.shape[0]} time records")

    sst_init, sic_init = get_forcing_at_time(forcing, START_DAY)
    print(f"  Initial SST: min={float(jnp.min(sst_init)):.1f} K, "
          f"max={float(jnp.max(sst_init)):.1f} K")
    print(f"  Initial SIC: mean={float(jnp.mean(sic_init)):.3f}")

    def get_sst_sic(day):
        return get_forcing_at_time(forcing, day)

# ---------------------------------------------------------------------------
# 3. Spectral PE model
# ---------------------------------------------------------------------------
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPrimitiveEquationModel,
    SpectralPEConfig,
    SpectralHydrostaticState,
    isothermal_rest_state_spectral,
    spectral_pe_to_grid,
)
from legoesm.core.field import Field

a = grid.radius
eig_max = TRUNC * (TRUNC + 1) / (a * a)
# Hyperdiffusion: 0.5-hour e-folding time at max wavenumber (∇^4 diffusion)
HYPERDIFF = 1.0 / (0.5 * 3600.0 * eig_max ** 2)

pe_config = SpectralPEConfig(
    hyperdiff_coeff=HYPERDIFF,
    hyperdiff_order=2,
    time_integrator="ssp_rk54",
    semi_implicit=False,
)
model = SpectralPrimitiveEquationModel(grid, sigma, pe_config)

# Warn if dt might be too large for the chosen truncation
# CFL limit for SSP-RK54: imaginary stability limit ~3.3 (vs 1.73 for RK3)
c_est = (constants.R_d * 270.0) ** 0.5
dt_max_est = 3.3 * (jnp.pi * a / TRUNC) / (2 * jnp.pi * c_est)
if DT > 0.8 * dt_max_est:
    print(f"  WARNING: dt={DT:.0f}s may be too large for T{TRUNC}")
    print(f"  Estimated max stable dt ~ {float(dt_max_est):.0f}s (recommend dt <= {0.7*float(dt_max_est):.0f}s)")

# ---------------------------------------------------------------------------
# 4. Initial state
# ---------------------------------------------------------------------------
from legoesm.thermo import saturation_mixing_ratio

T_INIT = 270.0
_phis_arg = _phis_data if float(jnp.max(jnp.abs(_phis_data))) > 0 else None
state = isothermal_rest_state_spectral(grid, sigma, T_init=T_INIT, phis=_phis_arg)
_z_max_init = float(jnp.max(_phis_data)) / constants.g
_topo_msg = f", topo z_max={_z_max_init:.0f} m" if _z_max_init > 0 else ""
print(f"  Atmosphere initialized: T={T_INIT} K isothermal{_topo_msg}")

# Moisture initialization (grid space)
p_s_init = 1.0e5
p_full_init = p_s_init * sigma.sigma_full  # (nlev,)
RH_INIT = 0.5
T_init_grid = jnp.full(shape_3d, T_INIT)
p_full_init_3d = jnp.broadcast_to(p_full_init, shape_3d)
q_sat_init = saturation_mixing_ratio(T_init_grid, p_full_init_3d)
q_v = RH_INIT * q_sat_init * sigma.sigma_full ** 2
q_v = jnp.minimum(q_v, q_sat_init)
mean_qv = float(jnp.mean(q_v)) * 1000.0
cwv_init = float(jnp.mean(
    column_water_vapor(q_v, p_s_init, sigma.dsigma)
))
print(f"  Moisture: RH_init={RH_INIT}, mean q_v={mean_qv:.2f} g/kg, CWV={cwv_init:.1f} kg/m2")

# Initialize cloud water and rain (zero initially)
q_c = jnp.zeros(shape_3d)
q_r = jnp.zeros(shape_3d)

# ---------------------------------------------------------------------------
# 4b. Microphysics configuration
# ---------------------------------------------------------------------------
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig, KesslerConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState, make_zero_hydrometeors

_micro_config = MicrophysicsConfig(scheme=MICROPHYSICS)
_micro_fn = None
_micro_backend_config = None
if MICROPHYSICS != "none":
    from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
    from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics

    _micro_backends = {
        "kessler": (kessler_microphysics, _micro_config.kessler),
        "sundqvist": (sundqvist_microphysics, _micro_config.sundqvist),
        "seifert_beheng": (seifert_beheng_microphysics, _micro_config.seifert_beheng),
        "morrison": (morrison_microphysics, _micro_config.morrison),
        "thompson": (thompson_microphysics, _micro_config.thompson),
    }
    _micro_fn, _micro_backend_config = _micro_backends[MICROPHYSICS]

# ---------------------------------------------------------------------------
# 5. Physics configuration
# ---------------------------------------------------------------------------
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.convection.config import SBMConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
    preload_rrtmgp_optics,
    rrtmgp_radiation,
)
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
)
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.forcing.analytical import analytical_sst_sic
from legoesm.forcing.time_utils import day_to_calendar
from legoesm.forcing.surface_utils import blend_surface_temperature, blend_surface_property

S_0 = 1360.0
gray_config = GrayRadiationConfig(
    tau_equator=7.2,
    tau_pole=1.8,
    S_0=S_0,
    sfc_albedo=_albedo_ocean,
    perpetual_equinox=False,
)
rrtmg_config = RRTMGPConfig(
    co2_ppmv=args.co2_ppmv,
    sfc_albedo=_albedo_ocean,
    sfc_emissivity=0.98,
    S_0=S_0,
)
if RADIATION == "rrtmgp":
    preload_rrtmgp_optics(rrtmg_config)
sbm_config = SBMConfig(tau_c=7200.0, RH_ref=0.7)

_sigma_full = sigma.sigma_full
_sigma_half = sigma.sigma_half
_dsigma = sigma.dsigma

# BL exchange coefficients
_C_H = 0.0044
_C_E = 0.0044

# Rayleigh friction: BL drag + weak free-atmosphere
_sigma_b = 0.85
_k_f_max = 1.0 / 86400.0
_k_free = 0.1 / 86400.0
_k_f = _k_free + _k_f_max * jnp.maximum(0.0, (_sigma_full - _sigma_b) / (1.0 - _sigma_b))

print(f"  Physics: {RADIATION} radiation + friction (coupled into dynamics) + SBM + BL (operator-split)")

# ---------------------------------------------------------------------------
# 6. Physics functions
# ---------------------------------------------------------------------------
# The coupled physics_fn goes INSIDE the dynamics RK3 stages.
# It provides: radiation heating + Rayleigh friction (spectral tendencies).
# Convection, BL exchange, and condensation are operator-split AFTER dynamics
# because they modify q_v (which lives in grid space, not spectral).

# Mutable state for forcing (SST, SIC, day) that physics_fn can read.
# Using a list to allow mutation inside closures.
_forcing_state = {
    'sst': jnp.full(shape_2d, 290.0),
    'sic': jnp.zeros(shape_2d),
    'day_of_year': 1.0,
    'seconds_of_day': 43200.0,
    'q_v': q_v,  # will be updated each step
}


def _make_physics_fn():
    """Build the coupled physics function for step_with_physics.

    Returns spectral tendencies: radiation heating + Rayleigh friction.
    """
    def physics_fn(state, grid, sigma_coord):
        fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T_g = fields['T']
        u_g = fields['u']
        v_g = fields['v']
        p_s_g = fields['p_s']

        sst = _forcing_state['sst']
        sic = _forcing_state['sic']
        day_of_year = _forcing_state['day_of_year']
        seconds_of_day = _forcing_state['seconds_of_day']
        q_v_now = _forcing_state['q_v']

        T_sfc = blend_surface_temperature(sst, sic, _T_ice)

        from legoesm.grids.vertical import (
            HybridSigmaPressureCoordinate, pressure_from_hybrid,
        )
        if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
            p_full = pressure_from_hybrid(sigma_coord, p_s_g, full=True)
            p_half = pressure_from_hybrid(sigma_coord, p_s_g, full=False)
        else:
            p_full = p_s_g[..., None] * sigma_coord.sigma_full
            p_half = p_s_g[..., None] * sigma_coord.sigma_half

        T_col = T_g.reshape(-1, NLEV)
        p_full_col = p_full.reshape(-1, NLEV)
        p_half_col = p_half.reshape(-1, NLEV + 1)
        q_v_col = q_v_now.reshape(-1, NLEV)
        T_sfc_col = T_sfc.reshape(-1)
        lat_col = lat_2d_grid.reshape(-1)
        lon_col = lon_2d_grid.reshape(-1)

        # --- Radiation ---
        if RADIATION == "gray":
            insol = daily_mean_insolation(lat_col, day_of_year, S_0)
            rad_out = gray_radiation(
                T=T_col, p_full=p_full_col, p_half=p_half_col,
                sfc_temperature=T_sfc_col, lat=lat_col,
                q_v=q_v_col, insolation=insol, config=gray_config,
            )
        else:
            hour = seconds_of_day / 3600.0
            cosz = jnp.maximum(cos_zenith_angle(lat_col, lon_col, day_of_year, hour), 0.0)
            rad_out = rrtmgp_radiation(
                T=T_col,
                p_full=p_full_col,
                p_half=p_half_col,
                sfc_temperature=T_sfc_col,
                q_v=q_v_col,
                cos_zenith=cosz,
                config=rrtmg_config,
            )
        dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

        # --- Rayleigh friction ---
        du_dt = -_k_f * u_g
        dv_dt = -_k_f * v_g

        # Convert (du, dv) to spectral (dvor, ddiv)
        _im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        _one_over_a = 1.0 / a
        cos_lat_3d = grid.cos_lat[:, None, None]
        du_cos = du_dt * cos_lat_3d
        dv_cos = dv_dt * cos_lat_3d

        dvor_hat = (
            _im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
            + _one_over_a * sh_analysis_dmu_3d(grid, du_cos)
        )
        ddiv_hat = (
            _im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
            - _one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
        )

        dT_hat = sh_analysis_3d(grid, dT_dt_rad)
        dlnps_hat = jnp.zeros_like(state.lnps_hat.data)

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=dvor_hat),
            div_hat=state.div_hat.replace(data=ddiv_hat),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )

    return physics_fn


physics_fn = _make_physics_fn()


@jax.jit
def operator_split_moist_physics(T_g, p_s_g, q_v, q_c, q_r, u_g, v_g, sst, sic, dt):
    """Operator-split moisture physics: convection + microphysics + BL + condensation.

    Applied AFTER the coupled dynamics+radiation step.
    Returns: T_new, q_v_new, dq_c_dt, dq_r_dt, precip, precip_ls.
    """
    T_sfc = blend_surface_temperature(sst, sic, _T_ice)
    p_full = p_s_g[..., None] * _sigma_full
    p_half = p_s_g[..., None] * _sigma_half

    ncol = shape_2d[0] * shape_2d[1]
    T_col = T_g.reshape(-1, NLEV)
    p_full_col = p_full.reshape(-1, NLEV)
    p_half_col = p_half.reshape(-1, NLEV + 1)
    q_v_col = q_v.reshape(-1, NLEV)

    # SBM convection
    conv_out = sbm_convection(
        T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
        dt=dt, config=sbm_config,
    )
    dT_dt_conv = conv_out.dT_dt.reshape(shape_3d)
    dq_v_dt_conv = conv_out.dq_v_dt.reshape(shape_3d)
    precip = conv_out.precipitation.reshape(shape_2d)

    # Microphysics (after convection)
    dT_dt_micro = jnp.zeros(shape_3d)
    dq_v_dt_micro = jnp.zeros(shape_3d)
    dq_c_dt = jnp.zeros(shape_3d)
    dq_r_dt = jnp.zeros(shape_3d)
    precip_micro = jnp.zeros(shape_2d)
    if _micro_fn is not None:
        q_c_col = q_c.reshape(ncol, NLEV)
        q_r_col = q_r.reshape(ncol, NLEV)
        rho_col = p_full_col / (constants.R_d * T_col)
        dp_col = p_half_col[:, 1:] - p_half_col[:, :-1]
        dz_col = dp_col / (rho_col * constants.g)
        hydrometeors = HydrometeorState(
            q_c=q_c_col, q_r=q_r_col,
            q_i=jnp.zeros_like(q_c_col), q_s=jnp.zeros_like(q_c_col),
            q_g=jnp.zeros_like(q_c_col), N_c=jnp.zeros_like(q_c_col),
            N_r=jnp.zeros_like(q_c_col), N_i=jnp.zeros_like(q_c_col),
        )
        micro_out = _micro_fn(
            T=T_col, q_v=q_v_col, hydrometeors=hydrometeors,
            p_full=p_full_col, p_half=p_half_col,
            rho=rho_col, dz=dz_col, dt=dt,
            config=_micro_backend_config,
        )
        dT_dt_micro = micro_out.dT_dt.reshape(shape_3d)
        dq_v_dt_micro = micro_out.dq_v_dt.reshape(shape_3d)
        dq_c_dt = micro_out.dq_c_dt.reshape(shape_3d)
        dq_r_dt = micro_out.dq_r_dt.reshape(shape_3d)
        precip_micro = micro_out.precipitation.reshape(shape_2d)

    # BL exchange
    rho_low = (p_s_g * _sigma_full[-1]) / (constants.R_d * T_g[..., -1])
    wind_speed = jnp.sqrt(u_g[..., -1]**2 + v_g[..., -1]**2 + 1.0)
    dp_low = p_s_g * (_sigma_half[-1] - _sigma_half[-2])

    shflx = rho_low * constants.c_pd * _C_H * wind_speed * (T_sfc - T_g[..., -1])
    q_sat_sfc = saturation_mixing_ratio(T_sfc, p_s_g)
    lhflx = rho_low * constants.L_v * _C_E * wind_speed * (q_sat_sfc - q_v[..., -1])
    evap_rate = lhflx / constants.L_v

    dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
    dq_BL = constants.g * evap_rate / dp_low

    # Temperature tendency from convection + microphysics + BL
    dT_dt_moist = dT_dt_conv + dT_dt_micro
    dT_dt_moist = dT_dt_moist.at[..., -1].add(dT_BL)

    # Moisture tendency
    dq_v_dt = dq_v_dt_conv + dq_v_dt_micro
    dq_v_dt = dq_v_dt.at[..., -1].add(dq_BL)

    # Apply tendencies
    T_new = T_g + dt * dT_dt_moist
    q_v_new = jnp.maximum(q_v + dt * dq_v_dt, 0.0)

    # Large-scale condensation (only if no microphysics)
    if _micro_fn is None:
        q_sat = saturation_mixing_ratio(T_new, p_s_g[..., None] * _sigma_full)
        excess = jnp.maximum(q_v_new - q_sat, 0.0)
        q_v_new = q_v_new - excess
        T_new = T_new + constants.L_v * excess / constants.c_pd
        precip_ls = jnp.sum(excess * p_s_g[..., None] * _dsigma, axis=-1) / (constants.g * dt)
    else:
        precip_ls = jnp.zeros(shape_2d)

    return T_new, q_v_new, dq_c_dt, dq_r_dt, precip + precip_micro, precip_ls


@jax.jit
def compute_rad_diagnostics(T_g, p_s_g, q_v, sst, sic, day_of_year, seconds_of_day):
    """Compute radiation diagnostics (fluxes) for output only."""
    T_sfc = blend_surface_temperature(sst, sic, _T_ice)
    albedo = blend_surface_property(sic, _albedo_ice, _albedo_ocean)
    p_full = p_s_g[..., None] * _sigma_full
    p_half = p_s_g[..., None] * _sigma_half

    T_col = T_g.reshape(-1, NLEV)
    p_full_col = p_full.reshape(-1, NLEV)
    p_half_col = p_half.reshape(-1, NLEV + 1)
    q_v_col = q_v.reshape(-1, NLEV)
    T_sfc_col = T_sfc.reshape(-1)
    lat_col = lat_2d_grid.reshape(-1)
    lon_col = lon_2d_grid.reshape(-1)
    if RADIATION == "gray":
        insol = daily_mean_insolation(lat_col, day_of_year, S_0)
        rad_out = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
    else:
        hour = seconds_of_day / 3600.0
        cosz = jnp.maximum(cos_zenith_angle(lat_col, lon_col, day_of_year, hour), 0.0)
        rad_out = rrtmgp_radiation(
            T=T_col,
            p_full=p_full_col,
            p_half=p_half_col,
            sfc_temperature=T_sfc_col,
            q_v=q_v_col,
            cos_zenith=cosz,
            config=rrtmg_config,
        )

    sw_down_sfc = rad_out.sw_flux_down[:, -1].reshape(shape_2d)
    sw_net_sfc = sw_down_sfc * (1.0 - albedo)
    lw_net_sfc = (rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]).reshape(shape_2d)
    sw_up_toa = rad_out.sw_flux_up[:, 0].reshape(shape_2d)
    lw_up_toa = rad_out.lw_flux_up[:, 0].reshape(shape_2d)

    return sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa


# ---------------------------------------------------------------------------
# 7. Save config
# ---------------------------------------------------------------------------
config_dict = {
    "truncation": TRUNC, "nlev": NLEV, "dt": DT,
    "days": N_DAYS, "start_day": START_DAY,
    "forcing": args.dataset, "radiation": RADIATION,
    "co2_ppmv": args.co2_ppmv, "S_0": S_0,
    "T_init": T_INIT, "RH_init": RH_INIT,
    "hyperdiff_coeff": float(HYPERDIFF),
    "n_lat": n_lat, "n_lon": n_lon, "ncol": ncol,
}
import json
with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)
print(f"  Config saved to {OUTPUT_DIR / 'experiment_config.json'}")

# ---------------------------------------------------------------------------
# 8. Time integration
# ---------------------------------------------------------------------------
n_steps_total = int(N_DAYS * 86400 / DT)
diag_interval = int(DIAG_DAYS * 86400 / DT)
checkpoint_interval = int(CHECKPOINT_DAYS * 86400 / DT) if CHECKPOINT_DAYS > 0 else 0

# Latitude array for radiation (broadcast to 2d) — used by physics_fn
lat_2d_grid = jnp.broadcast_to(grid.lat[:, None], shape_2d)
lon_2d_grid = jnp.broadcast_to(grid.lon[None, :], shape_2d)

# Diagnostics storage
diag_times = []
diag_sst = []
diag_sic = []
diag_T_atm = []
diag_T_low = []
diag_max_wind = []
diag_precip = []
diag_CWV = []
diag_sw_up_toa = []
diag_lw_up_toa = []
diag_sw_net_sfc = []
diag_lw_net_sfc = []
diag_dry_mass = []
diag_profiles_T = []
diag_profiles_qv = []
diag_sigma = np.asarray(sigma.sigma_full)

# 2D snapshots
snapshot_days_set = set()
for d in [5, 10, 15, 20, 25, 30, 60, 100, 200, 300, 365]:
    if d <= N_DAYS:
        snapshot_days_set.add(d)
if N_DAYS not in snapshot_days_set:
    snapshot_days_set.add(N_DAYS)
snapshots = {}

run_status = "COMPLETED"
current_day = START_DAY

print(f"\n  Starting integration: {n_steps_total} steps ({N_DAYS} days)")
print(f"  {'Day':>6s}  {'<SST>':>8s}  {'<SIC>':>6s}  {'<T_atm>':>8s}"
      f"  {'<T_low>':>8s}  {'<Precip>':>8s}  {'<CWV>':>6s}  {'max|v|':>8s}"
      f"  {'SW_TOA':>7s}  {'LW_TOA':>7s}  {'SfcSW':>6s}  {'SfcLW':>6s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}"
      f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}"
      f"  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")

# --- JIT warmup ---
t_wall_start = time.time()

day = current_day
day_of_year, seconds_of_day = day_to_calendar(day)
sst, sic = get_sst_sic(day)

# Set forcing state for physics_fn
_forcing_state['sst'] = sst
_forcing_state['sic'] = sic
_forcing_state['day_of_year'] = day_of_year
_forcing_state['seconds_of_day'] = seconds_of_day
_forcing_state['q_v'] = q_v

# Coupled dynamics + radiation + friction step
state = model.step_with_physics(state, DT, physics_fn)

# Get grid-space fields for moisture physics
fields = spectral_pe_to_grid(state, grid, sigma)
T_grid = fields['T']
u_grid = fields['u']
v_grid = fields['v']
p_s_grid = fields['p_s']

# Operator-split: convection + microphysics + BL + condensation
T_new, q_v, _dq_c_dt, _dq_r_dt, _precip, _precip_ls = operator_split_moist_physics(
    T_grid, p_s_grid, q_v, q_c, q_r, u_grid, v_grid, sst, sic, DT,
)
q_c = jnp.maximum(q_c + DT * _dq_c_dt, 0.0)
q_r = jnp.maximum(q_r + DT * _dq_r_dt, 0.0)

# Transform updated T back to spectral
T_hat_new = sh_analysis_3d(grid, T_new)
state = SpectralHydrostaticState(
    vor_hat=state.vor_hat,
    div_hat=state.div_hat,
    T_hat=state.T_hat.replace(data=T_hat_new),
    lnps_hat=state.lnps_hat,
    phis_hat=state.phis_hat,
)

jax.block_until_ready(state.vor_hat.data)
t_jit = time.time() - t_wall_start
print(f"  JIT compiled in {t_jit:.1f}s")

# --- Main time loop ---
t_wall_start = time.time()

for step in range(1, n_steps_total):
    day = START_DAY + (step + 1) * DT / 86400.0
    day_of_year, seconds_of_day = day_to_calendar(day)

    sst, sic = get_sst_sic(day)

    # Update forcing state for coupled physics_fn
    _forcing_state['sst'] = sst
    _forcing_state['sic'] = sic
    _forcing_state['day_of_year'] = day_of_year
    _forcing_state['seconds_of_day'] = seconds_of_day
    _forcing_state['q_v'] = q_v

    # (a) Coupled dynamics + radiation + friction
    state = model.step_with_physics(state, DT, physics_fn)

    # (b) Get grid-space fields
    fields = spectral_pe_to_grid(state, grid, sigma)
    T_grid = fields['T']
    u_grid = fields['u']
    v_grid = fields['v']
    p_s_grid = fields['p_s']

    # (c) Operator-split moisture physics
    T_new, q_v, _dq_c_dt, _dq_r_dt, _precip, _precip_ls = operator_split_moist_physics(
        T_grid, p_s_grid, q_v, q_c, q_r, u_grid, v_grid, sst, sic, DT,
    )
    q_c = jnp.maximum(q_c + DT * _dq_c_dt, 0.0)
    q_r = jnp.maximum(q_r + DT * _dq_r_dt, 0.0)

    # (d) Transform updated T back to spectral
    T_hat_new = sh_analysis_3d(grid, T_new)
    state = SpectralHydrostaticState(
        vor_hat=state.vor_hat,
        div_hat=state.div_hat,
        T_hat=state.T_hat.replace(data=T_hat_new),
        lnps_hat=state.lnps_hat,
        phis_hat=state.phis_hat,
    )

    # (e) Diagnostics
    if (step + 1) % diag_interval == 0:
        jax.block_until_ready(state.vor_hat.data)

        # Compute radiation diagnostics
        _sw, _lw, _sw_toa, _lw_toa = compute_rad_diagnostics(
            T_new, p_s_grid, q_v, sst, sic, day_of_year, seconds_of_day,
        )

        mean_sst = float(jnp.mean(sst))
        mean_sic = float(jnp.mean(sic))
        mean_T = float(jnp.mean(T_new))
        mean_T_low = float(jnp.mean(T_new[..., -1]))
        max_v = float(jnp.max(jnp.sqrt(u_grid**2 + v_grid**2)))

        mean_precip = float(jnp.mean(_precip + _precip_ls)) * 86400.0

        cwv = column_water_vapor(q_v, p_s_grid, _dsigma)
        mean_cwv = float(jnp.mean(cwv))

        mean_sw_toa = float(jnp.mean(_sw_toa))
        mean_lw_toa = float(jnp.mean(_lw_toa))
        mean_ps = float(jnp.mean(p_s_grid))

        elapsed_day = day - START_DAY
        diag_times.append(elapsed_day)
        diag_sst.append(mean_sst)
        diag_sic.append(mean_sic)
        diag_T_atm.append(mean_T)
        diag_T_low.append(mean_T_low)
        diag_max_wind.append(max_v)
        diag_precip.append(mean_precip)
        diag_CWV.append(mean_cwv)
        diag_sw_up_toa.append(mean_sw_toa)
        diag_lw_up_toa.append(mean_lw_toa)
        diag_dry_mass.append(mean_ps)

        # Area-weighted profiles
        T_w = np.asarray(T_new)
        qv_w = np.asarray(q_v)
        w = area_weights[:, None, None]  # (n_lat, 1, 1) for broadcasting
        diag_profiles_T.append(np.sum(T_w * w, axis=(0, 1)) / np.sum(w * np.ones((1, n_lon, 1))))
        diag_profiles_qv.append(np.sum(qv_w * w, axis=(0, 1)) / np.sum(w * np.ones((1, n_lon, 1))) * 1000.0)

        # Surface radiative fluxes
        mean_sw_sfc = float(jnp.mean(_sw))
        mean_lw_sfc = float(jnp.mean(_lw))
        diag_sw_net_sfc.append(mean_sw_sfc)
        diag_lw_net_sfc.append(mean_lw_sfc)

        # 2D snapshots (already on lat-lon — no regridding needed!)
        iday = int(round(elapsed_day))
        if iday in snapshot_days_set:
            T_sfc_snap = blend_surface_temperature(sst, sic, _T_ice)
            toa_net = -(_sw_toa + _lw_toa)
            sfc_net = _sw + _lw
            snapshots[iday] = {
                'SST': np.asarray(sst),
                'SIC': np.asarray(sic),
                'T_sfc': np.asarray(T_sfc_snap),
                'T_low': np.asarray(T_new[..., -1]),
                'q_v_low': np.asarray(q_v[..., -1]) * 1000.0,
                'precip': np.asarray(_precip + _precip_ls) * 86400.0,
                'wind': np.asarray(jnp.sqrt(u_grid[..., -1]**2 + v_grid[..., -1]**2)),
                'toa_net_rad': np.asarray(toa_net),
                'sfc_net_rad': np.asarray(sfc_net),
            }

        print(f"  {elapsed_day:6.0f}  {mean_sst:8.2f}  {mean_sic:6.3f}  {mean_T:8.2f}"
              f"  {mean_T_low:8.2f}  {mean_precip:8.2f}  {mean_cwv:6.1f}  {max_v:8.2f}"
              f"  {mean_sw_toa:7.1f}  {mean_lw_toa:7.1f}  {mean_sw_sfc:6.1f}  {mean_lw_sfc:6.1f}")

        # Sanity checks
        if not jnp.all(jnp.isfinite(state.vor_hat.data)):
            print(f"  BLOWUP: non-finite vorticity at day {elapsed_day:.0f}")
            run_status = f"BLOWUP at day {elapsed_day:.0f}: non-finite vorticity"
            break
        if max_v > 500:
            print(f"  BLOWUP: runaway wind speed {max_v:.1f} m/s at day {elapsed_day:.0f}")
            run_status = f"BLOWUP at day {elapsed_day:.0f}: max wind {max_v:.1f} m/s"
            break
        if not jnp.all(jnp.isfinite(T_new)):
            print(f"  BLOWUP: non-finite temperature at day {elapsed_day:.0f}")
            run_status = f"BLOWUP at day {elapsed_day:.0f}: non-finite T"
            break
        T_min = float(jnp.min(T_new))
        T_max = float(jnp.max(T_new))
        if T_min < 100 or T_max > 500:
            print(f"  WARNING: T out of bounds [{T_min:.1f}, {T_max:.1f}] K at day {elapsed_day:.0f}")
        if float(jnp.any(q_v < 0)):
            q_v = jnp.maximum(q_v, 0.0)

jax.block_until_ready(state.vor_hat.data)
total_wall = time.time() - t_wall_start
print(f"\n  Integration complete: {total_wall:.1f}s wall time")
print(f"  Status: {run_status}")

# ---------------------------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------------------------
# Save timeseries
np.savez(
    OUTPUT_DIR / "timeseries.npz",
    times=np.array(diag_times),
    sst=np.array(diag_sst),
    sic=np.array(diag_sic),
    T_atm=np.array(diag_T_atm),
    T_low=np.array(diag_T_low),
    max_wind=np.array(diag_max_wind),
    precip=np.array(diag_precip),
    CWV=np.array(diag_CWV),
    sw_up_toa=np.array(diag_sw_up_toa),
    lw_up_toa=np.array(diag_lw_up_toa),
    sw_net_sfc=np.array(diag_sw_net_sfc),
    lw_net_sfc=np.array(diag_lw_net_sfc),
    dry_mass=np.array(diag_dry_mass),
    profiles_T=np.array(diag_profiles_T) if diag_profiles_T else np.array([]),
    profiles_qv=np.array(diag_profiles_qv) if diag_profiles_qv else np.array([]),
    sigma=diag_sigma,
)
print(f"  Saved timeseries to {OUTPUT_DIR / 'timeseries.npz'}")

# Save summary text
with open(OUTPUT_DIR / "results.txt", "w") as f:
    f.write(f"AMIP Spectral PE — T{TRUNC}/L{NLEV}\n")
    f.write(f"Radiation: {RADIATION}\n")
    f.write(f"Forcing: {args.dataset}\n")
    f.write(f"CO2: {args.co2_ppmv} ppmv\n")
    f.write(f"Duration: {N_DAYS} days, dt={DT}s\n")
    f.write(f"Grid: {n_lat}x{n_lon} ({ncol} columns), {NLEV} levels\n")
    f.write(f"Status: {run_status}\n")
    f.write(f"Wall time: {total_wall:.1f}s\n")
    if diag_times:
        f.write(f"\nFinal diagnostics (day {diag_times[-1]:.0f}):\n")
        f.write(f"  <T_atm> = {diag_T_atm[-1]:.2f} K\n")
        f.write(f"  <T_low> = {diag_T_low[-1]:.2f} K\n")
        f.write(f"  max|v|  = {diag_max_wind[-1]:.2f} m/s\n")
        f.write(f"  <Precip> = {diag_precip[-1]:.2f} mm/day\n")
        f.write(f"  <CWV>   = {diag_CWV[-1]:.1f} kg/m2\n")
        f.write(f"  <p_s>   = {diag_dry_mass[-1]:.1f} Pa\n")

# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon_deg = np.degrees(np.asarray(grid.lon))
    lat_deg = np.degrees(np.asarray(grid.lat))

    # --- 8-panel timeseries ---
    if diag_times:
        fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
        t = np.array(diag_times)

        ax = axes[0, 0]; ax.plot(t, diag_sst); ax.set_ylabel("SST [K]"); ax.set_title("Global-mean SST")
        ax = axes[0, 1]; ax.plot(t, diag_sic); ax.set_ylabel("SIC"); ax.set_title("Global-mean SIC")
        ax = axes[1, 0]; ax.plot(t, diag_T_atm, label="<T>"); ax.plot(t, diag_T_low, label="T_low"); ax.set_ylabel("T [K]"); ax.legend(); ax.set_title("Temperature")
        ax = axes[1, 1]; ax.plot(t, diag_max_wind); ax.set_ylabel("m/s"); ax.set_title("Max wind speed")
        ax = axes[2, 0]; ax.plot(t, diag_precip); ax.set_ylabel("mm/day"); ax.set_title("Precipitation")
        ax = axes[2, 1]; ax.plot(t, diag_CWV); ax.set_ylabel("kg/m2"); ax.set_title("Column water vapor")
        ax = axes[3, 0]; ax.plot(t, diag_sw_up_toa, label="SW up"); ax.plot(t, diag_lw_up_toa, label="LW up"); ax.set_ylabel("W/m2"); ax.legend(); ax.set_title("TOA fluxes"); ax.set_xlabel("Day")
        ax = axes[3, 1]; ax.plot(t, diag_dry_mass); ax.set_ylabel("Pa"); ax.set_title("Mean surface pressure"); ax.set_xlabel("Day")

        plt.suptitle(f"AMIP Spectral T{TRUNC}/L{NLEV} — {RADIATION} radiation, {N_DAYS} days", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "amip_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved timeseries plot")

    # --- Final state lat-lon maps ---
    if snapshots:
        last_day = max(snapshots.keys())
        snap = snapshots[last_day]

        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        field_specs = [
            ('SST', 'SST [K]', 'coolwarm'),
            ('SIC', 'Sea-ice conc', 'Blues'),
            ('T_low', 'T lowest level [K]', 'coolwarm'),
            ('q_v_low', 'q_v lowest [g/kg]', 'YlGnBu'),
            ('precip', 'Precip [mm/day]', 'Blues'),
            ('wind', 'Wind speed lowest [m/s]', 'magma'),
            ('toa_net_rad', 'TOA net rad [W/m2]', 'RdBu_r'),
            ('sfc_net_rad', 'Sfc net rad [W/m2]', 'RdBu_r'),
        ]
        for idx, (key, label, cmap) in enumerate(field_specs):
            ax = axes.flat[idx]
            data = snap.get(key)
            if data is not None:
                im = ax.pcolormesh(lon_deg, lat_deg, data, cmap=cmap, shading='auto')
                plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        plt.suptitle(f"AMIP Spectral T{TRUNC}/L{NLEV} — Day {last_day}", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "amip_final_state.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved final state map")

    # --- Snapshot evolution ---
    if len(snapshots) > 0:
        snap_days = sorted(snapshots.keys())
        n_snap = len(snap_days)
        evo_fields = ['T_low', 'q_v_low', 'precip', 'wind']
        evo_labels = ['T low [K]', 'q_v low [g/kg]', 'Precip [mm/day]', 'Wind [m/s]']
        evo_cmaps = ['coolwarm', 'YlGnBu', 'Blues', 'magma']

        fig, axes = plt.subplots(len(evo_fields), n_snap,
                                  figsize=(4.5 * n_snap, 3.2 * len(evo_fields)))
        if len(evo_fields) == 1:
            axes = axes[None, :]
        if n_snap == 1:
            axes = axes[:, None]

        for row, (key, label, cmap) in enumerate(zip(evo_fields, evo_labels, evo_cmaps)):
            vals = [snapshots[d].get(key) for d in snap_days]
            vals_valid = [v for v in vals if v is not None]
            if vals_valid:
                vmin = min(np.nanmin(v) for v in vals_valid)
                vmax = max(np.nanmax(v) for v in vals_valid)
            else:
                vmin, vmax = 0, 1
            for col, d in enumerate(snap_days):
                ax = axes[row, col]
                data = snapshots[d].get(key)
                if data is not None:
                    im = ax.pcolormesh(lon_deg, lat_deg, data,
                                       cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
                ax.set_title(f"{label}\nDay {d}")
                if col == 0:
                    ax.set_ylabel(label)
                if col == n_snap - 1:
                    plt.colorbar(im, ax=ax, shrink=0.8)

        plt.suptitle(f"AMIP Spectral T{TRUNC}/L{NLEV} snapshots", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "amip_snapshots.png", dpi=150, bbox_inches="tight")
        plt.savefig(OUTPUT_DIR / "amip_snapshots_latlon_pixels.png", dpi=150, bbox_inches="tight")
        plt.savefig(OUTPUT_DIR / "amip_snapshots_native.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved snapshot evolution")

    # --- Vertical profiles ---
    if len(diag_profiles_T) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        sigma_plot = diag_sigma

        for i, idx in enumerate([0, len(diag_profiles_T)//2, -1]):
            label = f"Day {diag_times[idx]:.0f}"
            ax1.plot(diag_profiles_T[idx], sigma_plot, label=label)
            ax2.plot(diag_profiles_qv[idx], sigma_plot, label=label)

        ax1.set_xlabel("Temperature [K]"); ax1.set_ylabel("Sigma")
        ax1.invert_yaxis(); ax1.legend(); ax1.set_title("T(sigma)")
        ax2.set_xlabel("q_v [g/kg]"); ax2.set_ylabel("Sigma")
        ax2.invert_yaxis(); ax2.legend(); ax2.set_title("q_v(sigma)")

        plt.suptitle(f"AMIP Spectral T{TRUNC}/L{NLEV} profiles", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "amip_profiles.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved vertical profiles")

except Exception as e:
    print(f"  Could not produce plots: {e}")

print(f"\n  All outputs saved to {OUTPUT_DIR}")
print(f"  Status: {run_status}")
