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

Supports:
  - Checkpoint/restart for long integrations (--checkpoint-days, --restart-from)
  - Serializable experiment config (AMIPExperimentConfig, saved as JSON)
  - External forcing scaffolds (GHG, ozone, aerosol, solar — mostly placeholders)

Usage:
    JAX_ENABLE_X64=1 python scripts/run_amip.py \\
        --dataset cobe --forcing-path /path/to/MODEL.SST.COBE-SST2.nc \\
        --start-day 0 --days 30 --resolution 16 --dt 600

    JAX_ENABLE_X64=1 python scripts/run_amip.py \\
        --dataset hadisst --forcing-path /path/to/HadISST_sst.nc \\
        --days 365 --resolution 24 --dt 450 \\
        --checkpoint-days 30 --output results/amip_hadisst_1yr

    # Restart from checkpoint:
    JAX_ENABLE_X64=1 python scripts/run_amip.py \\
        --restart-from results/amip_hadisst_1yr/checkpoint_day_030.npz \\
        --forcing-path /path/to/HadISST_sst.nc \\
        --days 365
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
    description="AMIP simulation with prescribed SST/SIC",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""\
Presets:
  cobe     COBE-SST2 (SST_cpl [K], ice_cov [%%])
  hadisst  HadISST   (sst [C], sic [fraction])
  custom   User-specified variable names and unit conversions
""",
)
parser.add_argument("--dataset", type=str, default="cobe",
                    choices=["cobe", "hadisst", "custom", "analytical"],
                    help="Forcing dataset preset (default: cobe)")
parser.add_argument("--forcing-path", type=str, default=None,
                    help="Path to NetCDF forcing file")
parser.add_argument("--sst-var", type=str, default=None,
                    help="SST variable name (for custom dataset)")
parser.add_argument("--sic-var", type=str, default=None,
                    help="SIC variable name (for custom dataset)")
parser.add_argument("--sst-offset", type=float, default=None,
                    help="SST offset (e.g., 273.15 if data in Celsius)")
parser.add_argument("--sic-scale", type=float, default=None,
                    help="SIC scale (e.g., 0.01 if data in percent)")
parser.add_argument("--start-day", type=float, default=0.0,
                    help="Start day within forcing record (default: 0)")
parser.add_argument("--days", type=int, default=200,
                    help="Integration length [days] (default: 200)")
parser.add_argument("--resolution", type=int, default=16,
                    help="Cubed-sphere N (default: 16)")
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
                    help="Time step [s] (default: 600)")
parser.add_argument("--diag-days", type=int, default=5,
                    help="Diagnostic interval [days] (default: 5)")
parser.add_argument("--output", type=str, default=None,
                    help="Output directory (default: results/amip/<run_id>)")
# Checkpoint / restart
parser.add_argument("--checkpoint-days", type=int, default=0,
                    help="Checkpoint interval [days]; 0 = no checkpointing (default: 0)")
parser.add_argument("--restart-from", type=str, default=None,
                    help="Path to checkpoint .npz file for restart")
# Radiation
parser.add_argument("--radiation", type=str, default="gray",
                    choices=["gray", "rrtmg"],
                    help="Radiation scheme: gray (Frierson 2006) or rrtmg (RRTMGP correlated-k) (default: gray)")
parser.add_argument("--rad-update-steps", type=int, default=1,
                    help="Recompute radiation every N steps (1 = every step; default: 1)")
parser.add_argument("--diurnal-cycle", action="store_true", default=False,
                    help="Use instantaneous solar zenith angle instead of daily-mean insolation")
# RRTMG gas concentrations
parser.add_argument("--co2-ppmv", type=float, default=415.0,
                    help="CO2 concentration [ppmv] for RRTMG (default: 415)")
parser.add_argument("--ch4-ppbv", type=float, default=1900.0,
                    help="CH4 concentration [ppbv] for RRTMG (default: 1900)")
parser.add_argument("--n2o-ppbv", type=float, default=332.0,
                    help="N2O concentration [ppbv] for RRTMG (default: 332)")
parser.add_argument("--ozone-source", type=str, default="standard",
                    choices=["standard", "analytical", "none"],
                    help="Ozone profile for RRTMG: standard (US Std Atm 1976), "
                         "analytical (lat-dependent Gaussian), none (default: standard)")
parser.add_argument("--clouds", type=str, default="none",
                    choices=["none", "sundqvist", "xu_randall"],
                    help="Cloud fraction scheme for RRTMG: none (clear-sky), "
                         "sundqvist (RH-based), xu_randall (RH+condensate) (default: none)")
parser.add_argument("--discretization", type=str, default="centered",
                    choices=["centered", "finite_volume", "cgrid"],
                    help="Dynamical core discretization (default: centered)")
# Topography
parser.add_argument("--topography", type=str, default="flat",
                    help="Topography: flat, gaussian, or path to NetCDF file (default: flat)")
parser.add_argument("--topo-smoothing", type=int, default=4,
                    help="Laplacian smoothing passes for topography (default: 4)")
parser.add_argument("--topo-edge-blend", type=float, default=0.3,
                    help="Edge blending strength at cube-face boundaries [0-1] (default: 0.3)")
args = parser.parse_args()

# Enforce that forcing-path is required unless restarting or analytical
if args.forcing_path is None and args.restart_from is None and args.dataset != "analytical":
    parser.error("--forcing-path is required (unless using --restart-from or --dataset analytical)")

# ---------------------------------------------------------------------------
# Build AMIPExperimentConfig
# ---------------------------------------------------------------------------
from legoesm.forcing.amip_config import (
    AMIPExperimentConfig,
    config_to_dict,
    save_config,
    save_checkpoint,
    load_checkpoint,
)

_p_top = args.p_top if args.p_top is not None else (200.0 if args.vertical_coord == "hybrid" else 1000.0)
_stretching = args.stretching if args.stretching is not None else (2.0 if args.vertical_coord == "hybrid" else 0.0)
exp_config = AMIPExperimentConfig(
    resolution=args.resolution,
    nlev=args.nlev,
    dt=args.dt,
    vertical_coord=args.vertical_coord,
    p_top_Pa=_p_top,
    stretching=_stretching,
    start_day=args.start_day,
    days=args.days,
    diag_days=args.diag_days,
    checkpoint_days=args.checkpoint_days,
    dataset=args.dataset,
    forcing_path=args.forcing_path or "",
    sst_var=args.sst_var or "",
    sic_var=args.sic_var or "",
    sst_offset=args.sst_offset if args.sst_offset is not None else 0.0,
    sic_scale=args.sic_scale if args.sic_scale is not None else 1.0,
    radiation=args.radiation,
    rad_update_steps=args.rad_update_steps,
    diurnal_cycle=args.diurnal_cycle,
    co2_ppmv=args.co2_ppmv,
    ch4_ppbv=args.ch4_ppbv,
    n2o_ppbv=args.n2o_ppbv,
    ozone_source=args.ozone_source,
    cloud_scheme=args.clouds,
    topography=args.topography,
    topo_smoothing=args.topo_smoothing,
    topo_edge_blend=args.topo_edge_blend,
)

N = exp_config.resolution
NLEV = exp_config.nlev
DT = exp_config.dt
N_DAYS = exp_config.days
DIAG_DAYS = exp_config.diag_days
START_DAY = exp_config.start_day
CHECKPOINT_DAYS = exp_config.checkpoint_days
RADIATION = exp_config.radiation
RAD_UPDATE_STEPS = exp_config.rad_update_steps
DIURNAL_CYCLE = exp_config.diurnal_cycle
OZONE_SOURCE = exp_config.ozone_source
CLOUD_SCHEME = exp_config.cloud_scheme

# Build output directory with run identifier
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.output is not None:
    OUTPUT_DIR = Path(args.output)
else:
    OUTPUT_DIR = Path(f"results/amip/C{N}_L{NLEV}_{N_DAYS}d_{RUN_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISCRETIZATION = args.discretization
_rad_label = "gray (Frierson 2006)" if RADIATION == "gray" else "RRTMG (correlated-k)"
_disc_labels = {"finite_volume": "Finite Volume (PPM)", "cgrid": "C-grid (PPM + div damp)", "centered": "Centered FD"}
_disc_label = _disc_labels.get(DISCRETIZATION, DISCRETIZATION)
print("=" * 70)
print(f"  AMIP: Prescribed SST/SIC + {_rad_label} + SBM Convection")
print("=" * 70)
print(f"  Grid:       C{N} / L{NLEV}")
print(f"  Dycore:     {_disc_label}")
print(f"  dt:         {DT:.0f} s")
print(f"  Duration:   {N_DAYS} days (start day {START_DAY})")
print(f"  Radiation:  {_rad_label}")
if RADIATION == "rrtmg":
    print(f"  CO2: {exp_config.co2_ppmv} ppmv, CH4: {exp_config.ch4_ppbv} ppbv, N2O: {exp_config.n2o_ppbv} ppbv")
if RAD_UPDATE_STEPS > 1:
    print(f"  Radiation update every {RAD_UPDATE_STEPS} steps ({RAD_UPDATE_STEPS * DT:.0f} s)")
print(f"  Diagnostics every {DIAG_DAYS} days")
if CHECKPOINT_DAYS > 0:
    print(f"  Checkpointing every {CHECKPOINT_DAYS} days")
if args.restart_from:
    print(f"  Restarting from: {args.restart_from}")
print(f"  Dataset:    {args.dataset}")
print(f"  Forcing:    {args.forcing_path}")
print(f"  Output:     {OUTPUT_DIR}")
print()

# ---------------------------------------------------------------------------
# 1. Grid and vertical coordinate
# ---------------------------------------------------------------------------
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels

grid = create_cubed_sphere(N)
if exp_config.vertical_coord == "hybrid":
    sigma = make_hybrid_levels(
        NLEV, p_top_Pa=exp_config.p_top_Pa, stretching=exp_config.stretching,
    )
    _vcoord_label = f"hybrid σ-p (p_top={exp_config.p_top_Pa:.0f} Pa, stretch={exp_config.stretching:.1f})"
else:
    sigma = create_sigma_coordinate(NLEV)
    _vcoord_label = "sigma"

shape_2d = (6, N, N)
shape_3d = (6, N, N, NLEV)

print(f"  Grid created: {6*N*N} columns, {NLEV} levels ({_vcoord_label})")

# ---------------------------------------------------------------------------
# 1b. Topography
# ---------------------------------------------------------------------------
from legoesm.grids.topography import (
    TopographyConfig, load_real_topography, gaussian_mountain,
    phis_from_topography, land_mask_from_topography,
)
from legoesm.coupler.config import TileConfig

TOPOGRAPHY = exp_config.topography
if TOPOGRAPHY == "flat":
    _phis_data = jnp.zeros(shape_2d)
    _f_land = jnp.zeros(shape_2d)
    _topo_label = "flat"
elif TOPOGRAPHY == "gaussian":
    _z_s = gaussian_mountain(grid)
    _phis_data = phis_from_topography(_z_s)
    _f_land = land_mask_from_topography(_z_s)
    _topo_label = "Gaussian mountain (h=2500 m)"
else:
    # Treat as path to NetCDF file
    _topo_config = TopographyConfig(
        source="file",
        path=TOPOGRAPHY,
        smoothing_passes=exp_config.topo_smoothing,
        edge_blend_strength=exp_config.topo_edge_blend,
    )
    _phis_data, _f_land = load_real_topography(grid, config=_topo_config)
    _z_max = float(jnp.max(_phis_data)) / constants.g
    _land_pct = float(jnp.mean(_f_land)) * 100
    _topo_label = f"real ({TOPOGRAPHY}), z_max={_z_max:.0f} m, land={_land_pct:.1f}%"

_tile_config = TileConfig(f_land=_f_land, f_lake=jnp.zeros(shape_2d))
print(f"  Topography: {_topo_label}")

# ---------------------------------------------------------------------------
# 2. Load AMIP forcing
# ---------------------------------------------------------------------------
# Sea-ice parameters (defaults for analytical; overridden by forcing_config below)
_T_ice = 271.35
_albedo_ice = 0.65
_albedo_ocean = 0.06

if args.dataset == "analytical":
    # Analytical zonally-symmetric SST + SIC with seasonal cycle
    print("  Using analytical SST/SIC forcing (zonally symmetric, seasonal)")

    lat_deg_cube = np.degrees(np.asarray(grid.lat))  # (6, N, N)

    def _analytical_forcing(day):
        """Compute SST and SIC for a given day (seasonal cycle)."""
        day_of_year = day % 365.0 + 1.0
        # Seasonal shift of SST peak: +/-5 degrees latitude
        lat_shift = -5.0 * np.cos(2.0 * np.pi * day_of_year / 365.0)
        lat_eff = lat_deg_cube - lat_shift

        # Qobs-like SST profile (K): warm equator, cold poles
        sst = 27.0 * (1.0 - np.sin(np.radians(lat_eff))**2) + 273.15
        # Seasonal amplitude (+/-3 K at midlatitudes)
        seasonal_amp = 3.0 * np.cos(np.radians(lat_eff)) * np.cos(2.0 * np.pi * day_of_year / 365.0)
        sst = sst + seasonal_amp
        sst = np.maximum(sst, _T_ice - 1.8)  # freezing floor

        # SIC: ramp from 0 to 1 as SST drops below T_ice
        sic = np.clip(((_T_ice + 0.5) - sst) / 3.0, 0.0, 1.0)

        return jnp.array(sst), jnp.array(sic)

    def get_sst_sic(day):
        return _analytical_forcing(day)

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

    forcing_path = args.forcing_path or exp_config.forcing_path
    if args.dataset == "custom":
        forcing_config = AMIPForcingConfig(
            dataset="custom",
            path=forcing_path,
            sst_var=args.sst_var or "sst",
            sic_var=args.sic_var or "sic",
            sst_offset=args.sst_offset if args.sst_offset is not None else 0.0,
            sic_scale=args.sic_scale if args.sic_scale is not None else 1.0,
        )
    else:
        forcing_config = get_amip_preset(args.dataset)._replace(path=forcing_path)

    print("  Loading forcing data...")
    t_load = time.time()
    forcing = load_amip_forcing(forcing_config, grid)
    t_load = time.time() - t_load
    print(f"  Forcing loaded in {t_load:.1f}s: {forcing.times.shape[0]} time records")
    print(f"  Time range: day {float(forcing.times[0]):.0f} to {float(forcing.times[-1]):.0f}")

    if START_DAY < float(forcing.times[0]) or START_DAY > float(forcing.times[-1]):
        print(f"  WARNING: start_day={START_DAY} outside forcing range "
              f"[{float(forcing.times[0]):.0f}, {float(forcing.times[-1]):.0f}]")
        print(f"  Forcing will be clamped to the nearest available record.")

    sst_init, sic_init = get_forcing_at_time(forcing, START_DAY)
    print(f"  Initial SST: min={float(jnp.min(sst_init)):.1f} K, "
          f"max={float(jnp.max(sst_init)):.1f} K, "
          f"mean={float(jnp.mean(sst_init)):.1f} K")
    print(f"  Initial SIC: mean={float(jnp.mean(sic_init)):.3f}, "
          f"max={float(jnp.max(sic_init)):.3f}")

    _T_ice = forcing_config.T_ice
    _albedo_ice = forcing_config.albedo_ice
    _albedo_ocean = forcing_config.albedo_ocean

    def get_sst_sic(day):
        return get_sst_sic(day)

# ---------------------------------------------------------------------------
# 3. Atmospheric model (hydrostatic primitive equations)
# ---------------------------------------------------------------------------
from legoesm.atmosphere.physics.held_suarez import held_suarez_init

HYPERDIFF = exp_config.hyperdiff_scale * (48 / N) ** 4

_dx_min = float(grid.dx.min()) / 2.0  # single-cell width

if DISCRETIZATION == "finite_volume":
    from legoesm.atmosphere.dynamics.primitive_eq_fv import (
        FVPrimitiveEquationModel,
        FVPrimitiveEquationConfig,
    )
    div_damp_2 = 0.05 * _dx_min ** 2 / DT
    dycore_config = FVPrimitiveEquationConfig(
        div_damp_2=div_damp_2,
        div_damp_4=0.0,
        hyperdiff_coeff=HYPERDIFF,
        hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
        use_limiter=True,
    )
    model = FVPrimitiveEquationModel(grid, sigma, dycore_config)
    print(f"  FV dycore: div_damp_2={div_damp_2:.2e}, hyperdiff={HYPERDIFF:.2e}")
elif DISCRETIZATION == "cgrid":
    from legoesm.atmosphere.dynamics.primitive_eq_cgrid import (
        CGPrimitiveEquationModel,
        CGPrimitiveEquationConfig,
    )
    _div_damp_2 = 0.05 * _dx_min ** 2 / DT
    _div_damp_4 = 0.01 * _dx_min ** 4 / DT
    dycore_config = CGPrimitiveEquationConfig(
        hyperdiff_coeff=HYPERDIFF,
        hyperdiff_ps_coeff=0.0,
        div_damp_2=_div_damp_2,
        div_damp_4=_div_damp_4,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = CGPrimitiveEquationModel(grid, sigma, dycore_config)
    print(f"  C-grid dycore: div_damp_2={_div_damp_2:.2e}, div_damp_4={_div_damp_4:.2e}, hyperdiff={HYPERDIFF:.2e}")
else:
    from legoesm.atmosphere.dynamics.primitive_eq import (
        PrimitiveEquationModel,
        PrimitiveEquationConfig,
    )
    _div_damp = 0.12 * _dx_min ** 2 / DT
    dycore_config = PrimitiveEquationConfig(
        hyperdiff_coeff=HYPERDIFF,
        hyperdiff_ps_coeff=HYPERDIFF,
        div_damp_coeff=_div_damp,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = PrimitiveEquationModel(grid, sigma, dycore_config)
    print(f"  Centered dycore (PPM scalar transport): div_damp={_div_damp:.2e}, hyperdiff={HYPERDIFF:.2e}")

# ---------------------------------------------------------------------------
# 4. Initial state or restart
# ---------------------------------------------------------------------------
from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
)
from legoesm.atmosphere.physics.convection.sbm import sbm_convection

restart_step = 0
restart_day = START_DAY

if args.restart_from:
    print(f"  Loading checkpoint: {args.restart_from}")
    state, q_v, restart_step, restart_day, restored_config, diag_accum = load_checkpoint(
        Path(args.restart_from), grid, sigma,
    )
    print(f"  Restart from step {restart_step}, day {restart_day:.1f}")
    # Use restored config's forcing_path if not overridden
    if not args.forcing_path:
        exp_config = exp_config._replace(forcing_path=restored_config.forcing_path)
else:
    state = held_suarez_init(grid, sigma, T_init=exp_config.T_init, phis=_phis_data)
    _z_max_init = float(jnp.max(_phis_data)) / constants.g
    _topo_msg = f", topo z_max={_z_max_init:.0f} m" if _z_max_init > 0 else ""
    print(f"  Atmosphere initialized: T={exp_config.T_init} K isothermal{_topo_msg}")

    # Moisture initialization
    _RH_init = exp_config.RH_init
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
# 5. Physics configuration
# ---------------------------------------------------------------------------
from legoesm.atmosphere.physics.radiation.config import (
    GrayRadiationConfig, OzoneProfileConfig, RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import _compute_ozone_vmr
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
from legoesm.atmosphere.physics.convection.config import SBMConfig

gray_config = GrayRadiationConfig(
    tau_equator=exp_config.tau_equator,
    tau_pole=exp_config.tau_pole,
    S_0=exp_config.S_0,
    sfc_albedo=_albedo_ocean,
    perpetual_equinox=False,
)

rrtmg_config = RRTMGPConfig(
    co2_ppmv=exp_config.co2_ppmv,
    ch4_ppbv=exp_config.ch4_ppbv,
    n2o_ppbv=exp_config.n2o_ppbv,
    sfc_emissivity=exp_config.sfc_emissivity,
    sfc_albedo=_albedo_ocean,
    S_0=exp_config.S_0,
    use_scan=False,
    include_clouds=False,
)

sbm_config = SBMConfig(tau_c=exp_config.sbm_tau_c, RH_ref=exp_config.sbm_RH_ref)

ozone_config = OzoneProfileConfig(source=OZONE_SOURCE)
cloud_config = CloudConfig(scheme=CLOUD_SCHEME)

_rad_scheme_label = _rad_label
print(f"  Physics: {_rad_scheme_label} + SBM convection (operator-split)")
if RADIATION == "rrtmg":
    _o3_labels = {
        "standard": "US Std Atm 1976 climatology",
        "analytical": "analytical lat-dependent Gaussian",
        "none": "disabled (no ozone absorption)",
    }
    _cloud_labels = {
        "none": "clear-sky only (no cloud-radiation interaction)",
        "sundqvist": "Sundqvist (1988) RH-based cloud fraction",
        "xu_randall": "Xu-Randall (1996) RH+condensate cloud fraction",
    }
    print(f"  RRTMG atmospheric composition:")
    print(f"    CO2 = {exp_config.co2_ppmv} ppmv  (ACTIVE — affects LW/SW absorption)")
    print(f"    CH4 = {exp_config.ch4_ppbv} ppbv  (ACTIVE — affects LW absorption)")
    print(f"    N2O = {exp_config.n2o_ppbv} ppbv  (ACTIVE — affects LW absorption)")
    print(f"    O3  = {_o3_labels.get(OZONE_SOURCE, OZONE_SOURCE)} (ACTIVE — prescribed, not interactive)")
    print(f"    H2O = from prognostic q_v  (ACTIVE — radiatively interactive)")
    print(f"    Clouds: {_cloud_labels.get(CLOUD_SCHEME, CLOUD_SCHEME)}")
    print(f"    Aerosols: none (clear-sky)")

# ---------------------------------------------------------------------------
# 6. Operator-split physics step
# ---------------------------------------------------------------------------
_S_0 = exp_config.S_0
_sigma_full = sigma.sigma_full
_sigma_half = sigma.sigma_half
_dsigma = sigma.dsigma
_C_H = exp_config.C_H
_C_E = exp_config.C_E
_emissivity_ocean = exp_config.sfc_emissivity
_emissivity_ice = exp_config.emissivity_ice

# Rayleigh friction: BL drag + weak free-atmosphere drag
_sigma_b = exp_config.sigma_b
_k_f_max = exp_config.k_BL_max_per_day / 86400.0
_k_free = exp_config.k_free_per_day / 86400.0
_k_f = _k_free + _k_f_max * jnp.maximum(0.0, (_sigma_full - _sigma_b) / (1.0 - _sigma_b))
_fric_decay = jnp.exp(-_k_f * DT)

# Mild moisture smoothing (∇⁴) for grid-scale q_v noise from convection.
# The primary 2Δx checkerboard fix is now the compact ∇² in the dycore.
from legoesm.core.operators_3d import hyperdiffusion_3d as _hyperdiff_3d
_qv_smooth_coeff = HYPERDIFF * 0.5


if RADIATION == "gray":
    @jax.jit
    def radiation_step(T_col, p_full_col, p_half_col, q_v_col, T_sfc_col,
                       lat_col, lon_col, day_of_year, seconds_of_day,
                       albedo_col, emis_col):
        """Gray radiation call (daily-mean or diurnal-cycle insolation)."""
        if DIURNAL_CYCLE:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour)
            insol = _S_0 * jnp.maximum(cos_sza, 0.0)
        else:
            insol = daily_mean_insolation(lat_col, day_of_year, _S_0)
        return gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
else:
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation

    @jax.jit
    def radiation_step(T_col, p_full_col, p_half_col, q_v_col, T_sfc_col,
                       lat_col, lon_col, day_of_year, seconds_of_day,
                       albedo_col, emis_col):
        """RRTMG radiation call with per-column albedo/emissivity."""
        if DIURNAL_CYCLE:
            hour = seconds_of_day / 3600.0
            cos_sza = cos_zenith_angle(lat_col, lon_col, day_of_year, hour)
            cos_zenith = jnp.maximum(cos_sza, 0.0)
        else:
            insol = daily_mean_insolation(lat_col, day_of_year, _S_0)
            cos_zenith = jnp.clip(insol / jnp.clip(_S_0, 1.0e-6, None), 0.0, 1.0)
        o3_vmr = _compute_ozone_vmr(p_full_col, lat_col, ozone_config)
        # Compute cloud properties if cloud scheme is active.
        cloud_kwargs = {}
        if CLOUD_SCHEME != "none":
            dp = p_half_col[:, 1:] - p_half_col[:, :-1]
            cloud_props = compute_cloud_properties(
                T=T_col, p_full=p_full_col, q_v=q_v_col,
                dp=dp, config=cloud_config,
            )
            cloud_kwargs = {
                "cloud_path_liq": cloud_props.lwp,
                "cloud_path_ice": cloud_props.iwp,
                "cloud_r_eff_liq": cloud_props.r_eff_liq,
                "cloud_r_eff_ice": cloud_props.r_eff_ice,
            }
        return rrtmgp_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, q_v=q_v_col,
            cos_zenith=cos_zenith, config=rrtmg_config,
            sfc_albedo_override=albedo_col,
            sfc_emissivity_override=emis_col,
            o3_vmr=o3_vmr,
            **cloud_kwargs,
        )


@jax.jit
def physics_step_no_rad(T, p_s, q_v, u, v, sst, sic, lat, dt,
                        dT_dt_rad_held, sw_net_sfc_held, lw_net_sfc_held,
                        sw_up_toa_held, lw_up_toa_held):
    """Convection + BL exchange with held (pre-computed) radiation tendencies."""
    nlev = _sigma_full.shape[0]
    shape_3d = T.shape
    shape_2d = p_s.shape
    ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

    T_sfc = sic * _T_ice + (1.0 - sic) * sst

    p_full = p_s[..., None] * _sigma_full
    p_half = p_s[..., None] * _sigma_half

    T_col = T.reshape(ncol, nlev)
    p_full_col = p_full.reshape(ncol, nlev)
    p_half_col = p_half.reshape(ncol, nlev + 1)
    q_v_col = q_v.reshape(ncol, nlev)

    conv_out = sbm_convection(
        T=T_col, q_v=q_v_col, p_full=p_full_col, p_half=p_half_col,
        dt=dt, config=sbm_config,
    )
    dT_dt_conv = conv_out.dT_dt.reshape(shape_3d)
    dq_v_dt_conv = conv_out.dq_v_dt.reshape(shape_3d)
    precip = conv_out.precipitation.reshape(shape_2d)

    rho_low = (p_s * _sigma_full[-1]) / (constants.R_d * T[..., -1])
    wind_speed = jnp.sqrt(u[..., -1]**2 + v[..., -1]**2 + 1.0)
    dp_low = p_s * (_sigma_half[-1] - _sigma_half[-2])

    shflx = rho_low * constants.c_pd * _C_H * wind_speed * (T_sfc - T[..., -1])
    q_sat_sfc = saturation_mixing_ratio(T_sfc, p_s)
    lhflx = rho_low * constants.L_v * _C_E * wind_speed * (q_sat_sfc - q_v[..., -1])
    evap_rate = lhflx / constants.L_v

    dT_BL = constants.g * shflx / (constants.c_pd * dp_low)
    dq_BL = constants.g * evap_rate / dp_low

    dT_dt = dT_dt_rad_held + dT_dt_conv
    dT_dt = dT_dt.at[..., -1].add(dT_BL)

    dq_v_dt = dq_v_dt_conv
    dq_v_dt = dq_v_dt.at[..., -1].add(dq_BL)

    return (dT_dt, dq_v_dt, precip, sw_net_sfc_held, lw_net_sfc_held,
            sw_up_toa_held, lw_up_toa_held)


def compute_radiation_and_physics(T, p_s, q_v, u, v, sst, sic, lat, lon,
                                  day_of_year, seconds_of_day, dt):
    """Full physics step: recompute radiation + convection + BL."""
    nlev = _sigma_full.shape[0]
    shape_3d = T.shape
    shape_2d = p_s.shape
    ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

    T_sfc = sic * _T_ice + (1.0 - sic) * sst
    albedo = sic * _albedo_ice + (1.0 - sic) * _albedo_ocean
    emissivity = sic * _emissivity_ice + (1.0 - sic) * _emissivity_ocean

    p_full = p_s[..., None] * _sigma_full
    p_half = p_s[..., None] * _sigma_half

    T_col = T.reshape(ncol, nlev)
    p_full_col = p_full.reshape(ncol, nlev)
    p_half_col = p_half.reshape(ncol, nlev + 1)
    q_v_col = q_v.reshape(ncol, nlev)
    T_sfc_col = T_sfc.reshape(ncol)
    lat_col = lat.reshape(ncol)
    lon_col = lon.reshape(ncol)
    albedo_col = albedo.reshape(ncol)
    emis_col = emissivity.reshape(ncol)

    rad_out = radiation_step(T_col, p_full_col, p_half_col, q_v_col,
                             T_sfc_col, lat_col, lon_col,
                             day_of_year, seconds_of_day,
                             albedo_col, emis_col)
    dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

    sw_down_sfc = rad_out.sw_flux_down[:, -1].reshape(shape_2d)
    sw_net_sfc = sw_down_sfc * (1.0 - albedo)
    lw_net_sfc = (rad_out.lw_flux_down[:, -1] - rad_out.lw_flux_up[:, -1]).reshape(shape_2d)
    sw_up_toa = rad_out.sw_flux_up[:, 0].reshape(shape_2d)
    lw_up_toa = rad_out.lw_flux_up[:, 0].reshape(shape_2d)

    return physics_step_no_rad(
        T, p_s, q_v, u, v, sst, sic, lat, dt,
        dT_dt_rad, sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa,
    ), dT_dt_rad


# ---------------------------------------------------------------------------
# 7. Lat-lon regridding utility
# ---------------------------------------------------------------------------
def regrid_faces_to_latlon(field_faces, n_lon=None, n_lat=None):
    """Interpolate cubed-sphere face data to a regular lat-lon grid.

    Uses inverse-distance weighting of nearest neighbors in 3D Cartesian
    coordinates on the unit sphere.
    """
    if n_lon is None:
        n_lon = max(360, 8 * N)
    if n_lat is None:
        n_lat = n_lon // 2

    cube_lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180.0 / np.pi
    cube_lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180.0 / np.pi

    lon = cube_lon_deg.reshape(-1)
    lat = cube_lat_deg.reshape(-1)
    val = np.asarray(field_faces, dtype=np.float64).reshape(-1)

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon = ((lon[valid] + 180.0) % 360.0) - 180.0
    lat = np.clip(lat[valid], -90.0, 90.0)
    val = val[valid]

    lon_cent = np.linspace(-180.0, 180.0, n_lon, endpoint=False) + 180.0 / n_lon
    lat_cent = np.linspace(-90.0, 90.0, n_lat)
    lon2d, lat2d = np.meshgrid(lon_cent, lat_cent)

    from scipy.spatial import cKDTree

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    src_xyz = np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)],
    )

    lon_t = np.deg2rad(lon2d.reshape(-1))
    lat_t = np.deg2rad(lat2d.reshape(-1))
    cos_lat_t = np.cos(lat_t)
    tgt_xyz = np.column_stack(
        [cos_lat_t * np.cos(lon_t), cos_lat_t * np.sin(lon_t), np.sin(lat_t)],
    )

    k = min(16, src_xyz.shape[0])
    tree = cKDTree(src_xyz)
    dist, idx = tree.query(tgt_xyz, k=k)
    if k == 1:
        field_ll = val[idx].reshape(lon2d.shape)
    else:
        # Gaussian (RBF) weighting: smoother than IDW across face boundaries
        dist = np.maximum(dist, 1.0e-12)
        # Scale length = median distance to nearest neighbor
        sigma = np.median(dist[:, 0]) * 2.0
        w = np.exp(-0.5 * (dist / sigma) ** 2)
        w /= np.sum(w, axis=1, keepdims=True)
        field_ll = np.sum(val[idx] * w, axis=1).reshape(lon2d.shape)

    return lon_cent, lat_cent, field_ll


# ---------------------------------------------------------------------------
# 8. Save experiment config
# ---------------------------------------------------------------------------
save_config(exp_config, OUTPUT_DIR / "experiment_config.json")
print(f"  Config saved to {OUTPUT_DIR / 'experiment_config.json'}")

# ---------------------------------------------------------------------------
# 9. Time integration
# ---------------------------------------------------------------------------
n_steps_total = int(N_DAYS * 86400 / DT)
diag_interval = int(DIAG_DAYS * 86400 / DT)
checkpoint_interval = int(CHECKPOINT_DAYS * 86400 / DT) if CHECKPOINT_DAYS > 0 else 0

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

# 2D snapshots at selected days (store all faces for lat-lon remapping)
snapshot_days_set = set()
for d in [5, 10, 15, 20, 25, 30, 60, 100, 200, 300]:
    if d <= N_DAYS:
        snapshot_days_set.add(d)
if N_DAYS not in snapshot_days_set:
    snapshot_days_set.add(N_DAYS)
snapshots = {}

run_status = "COMPLETED"

# Determine where to start (fresh or restart)
start_step = restart_step
current_day = restart_day

n_steps_remaining = n_steps_total - start_step

print(f"\n  Starting integration: {n_steps_remaining} steps remaining ({N_DAYS} days total)")
print(f"  {'Day':>6s}  {'<SST>':>8s}  {'<SIC>':>6s}  {'<T_atm>':>8s}"
      f"  {'<T_low>':>8s}  {'<Precip>':>8s}  {'<CWV>':>6s}  {'max|v|':>8s}"
      f"  {'SW_TOA':>7s}  {'LW_TOA':>7s}  {'SfcSW':>6s}  {'SfcLW':>6s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}"
      f"  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}"
      f"  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")

# Held radiation tendencies for sub-cycling
_held_dT_rad = jnp.zeros(shape_3d)
_held_sw_net_sfc = jnp.zeros(shape_2d)
_held_lw_net_sfc = jnp.zeros(shape_2d)
_held_sw_up_toa = jnp.zeros(shape_2d)
_held_lw_up_toa = jnp.zeros(shape_2d)

t_wall_start = time.time()

# JIT warmup: run one full step (with radiation)
day = current_day
day_of_year = day % 365.0 + 1.0
seconds_of_day = (day * 86400.0) % 86400.0

sst, sic = get_sst_sic(day)

state = model.step_with_physics(state, DT)

(dT_dt, dq_v_dt, _precip, _sw, _lw, _sw_toa, _lw_toa), _held_dT_rad = \
    compute_radiation_and_physics(
        state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
        sst, sic, grid.lat, grid.lon, day_of_year, seconds_of_day, DT,
    )
_held_sw_net_sfc = _sw
_held_lw_net_sfc = _lw
_held_sw_up_toa = _sw_toa
_held_lw_up_toa = _lw_toa

new_T = state.T.data + DT * dT_dt
q_v = jnp.maximum(q_v + DT * dq_v_dt, 0.0)
q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
_excess = jnp.maximum(q_v - q_sat, 0.0)
q_v = q_v - _excess
new_T = new_T + constants.L_v * _excess / constants.c_pd
state = state._replace(T=state.T.replace(data=new_T))
_precip_ls = jnp.sum(_excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)

# Mild moisture smoothing
q_v = jnp.maximum(q_v + DT * _hyperdiff_3d(q_v, grid, _qv_smooth_coeff), 0.0)

state = state._replace(
    u=state.u.replace(data=state.u.data * _fric_decay),
    v=state.v.replace(data=state.v.data * _fric_decay),
)

jax.block_until_ready(state.u.data)
t_jit = time.time() - t_wall_start
print(f"  JIT compiled in {t_jit:.1f}s")

t_wall_start = time.time()

for step in range(start_step + 1, n_steps_total):
    day = START_DAY + (step + 1) * DT / 86400.0
    day_of_year = day % 365.0 + 1.0
    seconds_of_day = (day * 86400.0) % 86400.0

    sst, sic = get_sst_sic(day)

    # (a) Dynamics (includes compact ∇² damping of 2Δx mode)
    state = model.step_with_physics(state, DT)

    # (b) Physics: recompute radiation on cadence, hold tendencies otherwise
    need_rad = (RAD_UPDATE_STEPS <= 1) or ((step + 1) % RAD_UPDATE_STEPS == 0)

    if need_rad:
        (dT_dt, dq_v_dt, _precip, _sw, _lw, _sw_toa, _lw_toa), _held_dT_rad = \
            compute_radiation_and_physics(
                state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
                sst, sic, grid.lat, grid.lon, day_of_year, seconds_of_day, DT,
            )
        _held_sw_net_sfc = _sw
        _held_lw_net_sfc = _lw
        _held_sw_up_toa = _sw_toa
        _held_lw_up_toa = _lw_toa
    else:
        dT_dt, dq_v_dt, _precip, _sw, _lw, _sw_toa, _lw_toa = \
            physics_step_no_rad(
                state.T.data, state.p_s.data, q_v, state.u.data, state.v.data,
                sst, sic, grid.lat, DT,
                _held_dT_rad, _held_sw_net_sfc, _held_lw_net_sfc,
                _held_sw_up_toa, _held_lw_up_toa,
            )

    new_T = state.T.data + DT * dT_dt
    q_v = jnp.maximum(q_v + DT * dq_v_dt, 0.0)
    q_sat = saturation_mixing_ratio(new_T, state.p_s.data[..., None] * _sigma_full)
    _excess = jnp.maximum(q_v - q_sat, 0.0)
    q_v = q_v - _excess
    new_T = new_T + constants.L_v * _excess / constants.c_pd
    state = state._replace(T=state.T.replace(data=new_T))
    _precip_ls = jnp.sum(_excess * state.p_s.data[..., None] * _dsigma, axis=-1) / (constants.g * DT)

    # (c) Moisture smoothing
    q_v = jnp.maximum(q_v + DT * _hyperdiff_3d(q_v, grid, _qv_smooth_coeff), 0.0)

    # (d) Rayleigh friction
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

        mean_precip = float(jnp.mean(_precip + _precip_ls)) * 86400.0

        cwv = jnp.sum(q_v * state.p_s.data[..., None] * _dsigma, axis=-1) / constants.g
        mean_cwv = float(jnp.mean(cwv))

        # TOA radiative fluxes
        mean_sw_toa = float(jnp.mean(_sw_toa))
        mean_lw_toa = float(jnp.mean(_lw_toa))

        # Dry mass diagnostic: global-mean p_s (proportional to dry air mass)
        mean_ps = float(jnp.mean(state.p_s.data))

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

        diag_profiles_T.append(np.asarray(jnp.mean(state.T.data, axis=(0, 1, 2))))
        diag_profiles_qv.append(np.asarray(jnp.mean(q_v, axis=(0, 1, 2))) * 1000.0)

        # 2D snapshots (all faces for lat-lon remapping)
        iday = int(round(elapsed_day))
        if iday in snapshot_days_set:
            T_sfc_snap = sic * _T_ice + (1.0 - sic) * sst
            # TOA net = SW_down_TOA - SW_up_TOA - LW_up_TOA (positive = net incoming)
            toa_net = -(_sw_toa + _lw_toa)  # approximate: -SW_up - LW_up
            sfc_net = _sw + _lw  # SW_net_sfc + LW_net_sfc
            snapshots[iday] = {
                'SST': np.asarray(sst),
                'SIC': np.asarray(sic),
                'T_sfc': np.asarray(T_sfc_snap),
                'T_low': np.asarray(state.T.data[..., -1]),
                'q_v_low': np.asarray(q_v[..., -1]) * 1000.0,
                'precip': np.asarray(_precip + _precip_ls) * 86400.0,
                'wind': np.asarray(
                    jnp.sqrt(state.u.data[..., -1]**2 + state.v.data[..., -1]**2)
                ),
                'toa_net_rad': np.asarray(toa_net),
                'sfc_net_rad': np.asarray(sfc_net),
            }

        # Surface radiative fluxes
        mean_sw_sfc = float(jnp.mean(_sw))
        mean_lw_sfc = float(jnp.mean(_lw))
        diag_sw_net_sfc.append(mean_sw_sfc)
        diag_lw_net_sfc.append(mean_lw_sfc)

        print(f"  {elapsed_day:6.0f}  {mean_sst:8.2f}  {mean_sic:6.3f}  {mean_T:8.2f}"
              f"  {mean_T_low:8.2f}  {mean_precip:8.2f}  {mean_cwv:6.1f}  {max_v:8.2f}"
              f"  {mean_sw_toa:7.1f}  {mean_lw_toa:7.1f}  {mean_sw_sfc:6.1f}  {mean_lw_sfc:6.1f}")

        # --- Sanity checks ---
        if not jnp.all(jnp.isfinite(state.u.data)):
            print(f"  BLOWUP: non-finite winds at day {elapsed_day:.0f}")
            run_status = f"BLOWUP at day {elapsed_day:.0f}: non-finite winds"
            break
        if max_v > 500:
            print(f"  BLOWUP: runaway wind speed {max_v:.1f} m/s at day {elapsed_day:.0f}")
            run_status = f"BLOWUP at day {elapsed_day:.0f}: max wind {max_v:.1f} m/s"
            break
        if not jnp.all(jnp.isfinite(state.T.data)):
            print(f"  BLOWUP: non-finite temperature at day {elapsed_day:.0f}")
            run_status = f"BLOWUP at day {elapsed_day:.0f}: non-finite T"
            break
        T_min = float(jnp.min(state.T.data))
        T_max = float(jnp.max(state.T.data))
        if T_min < 100 or T_max > 500:
            print(f"  WARNING: temperature out of bounds [{T_min:.1f}, {T_max:.1f}] K at day {elapsed_day:.0f}")
        if float(jnp.any(q_v < 0)):
            q_v = jnp.maximum(q_v, 0.0)

    # --- Checkpoint ---
    if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
        elapsed_day = day - START_DAY
        ckpt_path = OUTPUT_DIR / f"checkpoint_day_{int(elapsed_day):04d}.npz"
        save_checkpoint(
            path=ckpt_path,
            state=state,
            q_v=q_v,
            step=step + 1,
            day=day,
            config=exp_config,
        )
        print(f"  Checkpoint saved: {ckpt_path.name}")

jax.block_until_ready(state.u.data)
total_wall = time.time() - t_wall_start
print(f"\n  Integration complete: {total_wall:.1f}s wall time")
print(f"  Status: {run_status}")

# ---------------------------------------------------------------------------
# 10. Save results
# ---------------------------------------------------------------------------
with open(OUTPUT_DIR / "results.txt", "w") as f:
    f.write(f"AMIP: Prescribed SST/SIC + {_rad_label} + SBM Convection\n")
    f.write(f"Run ID: {RUN_ID}\n")
    f.write(f"Grid: C{N}/L{NLEV}, dt={DT}s, {N_DAYS} days (start day {START_DAY})\n")
    f.write(f"Radiation: {RADIATION} ({_rad_label})\n")
    if RADIATION == "rrtmg":
        f.write(f"  CO2={exp_config.co2_ppmv} ppmv, CH4={exp_config.ch4_ppbv} ppbv, N2O={exp_config.n2o_ppbv} ppbv\n")
        f.write(f"  O3=US Std Atm 1976 climatology (prescribed), Clouds=clear-sky, Aerosols=none\n")
    if RAD_UPDATE_STEPS > 1:
        f.write(f"  Radiation update every {RAD_UPDATE_STEPS} steps\n")
    f.write(f"Dataset: {args.dataset}, file: {args.forcing_path}\n")
    f.write(f"Dycore: {_disc_label}\n")
    if args.dataset != "analytical":
        f.write(f"Forcing time range: day {float(forcing.times[0]):.0f} to {float(forcing.times[-1]):.0f}\n")
    else:
        f.write(f"Forcing: analytical (Qobs-like SST + seasonal cycle)\n")
    f.write(f"Physics: {_rad_label} + SBM convection (operator-split)\n")
    f.write(f"BL: bulk aero C_H=C_E={_C_H}\n")
    f.write(f"Friction: k_free={_k_free*86400:.1f}/day, k_BL_max={_k_f_max*86400:.1f}/day\n")
    f.write(f"Sea-ice: T_ice={_T_ice} K, albedo_ice={_albedo_ice}, albedo_ocean={_albedo_ocean}\n")
    if args.restart_from:
        f.write(f"Restarted from: {args.restart_from} (step {restart_step}, day {restart_day:.1f})\n")
    if CHECKPOINT_DAYS > 0:
        f.write(f"Checkpointing every {CHECKPOINT_DAYS} days\n")
    f.write(f"Status: {run_status}\n\n")
    f.write(f"{'Day':>6s}  {'<SST>':>10s}  {'<SIC>':>8s}  {'<T_atm>':>10s}"
            f"  {'<T_low>':>10s}  {'<Precip>':>10s}  {'<CWV>':>10s}  {'max|v|':>10s}"
            f"  {'SW_TOA':>10s}  {'LW_TOA':>10s}  {'<p_s>':>10s}\n")
    for i in range(len(diag_times)):
        f.write(f"{diag_times[i]:6.0f}  {diag_sst[i]:10.3f}  {diag_sic[i]:8.4f}"
                f"  {diag_T_atm[i]:10.3f}  {diag_T_low[i]:10.3f}"
                f"  {diag_precip[i]:10.3f}  {diag_CWV[i]:10.1f}"
                f"  {diag_max_wind[i]:10.3f}"
                f"  {diag_sw_up_toa[i]:10.1f}  {diag_lw_up_toa[i]:10.1f}"
                f"  {diag_dry_mass[i]:10.1f}\n")
    f.write(f"\nJIT compilation: {t_jit:.1f}s\n")
    f.write(f"Total wall time: {total_wall:.1f}s\n")
    if diag_sst:
        f.write(f"Final <SST>: {diag_sst[-1]:.3f} K\n")
        f.write(f"Final <SIC>: {diag_sic[-1]:.4f}\n")
        f.write(f"Final <T_atm>: {diag_T_atm[-1]:.3f} K\n")
        f.write(f"Final <Precip>: {diag_precip[-1]:.2f} mm/day\n")
        f.write(f"Final <CWV>: {diag_CWV[-1]:.1f} kg/m2\n")
        f.write(f"Final <p_s>: {diag_dry_mass[-1]:.1f} Pa\n")

# Save timeseries as npz for post-processing
np.savez(
    OUTPUT_DIR / "timeseries.npz",
    days=np.array(diag_times),
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
    dry_mass_ps=np.array(diag_dry_mass),
    sigma=diag_sigma,
    profiles_T=np.array(diag_profiles_T) if diag_profiles_T else np.array([]),
    profiles_qv=np.array(diag_profiles_qv) if diag_profiles_qv else np.array([]),
)

# Save final checkpoint
if CHECKPOINT_DAYS > 0:
    final_ckpt = OUTPUT_DIR / "checkpoint_final.npz"
    save_checkpoint(
        path=final_ckpt,
        state=state,
        q_v=q_v,
        step=n_steps_total,
        day=START_DAY + N_DAYS,
        config=exp_config,
    )
    print(f"  Final checkpoint saved: {final_ckpt.name}")

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # ---- Time series ----
    fig, axes = plt.subplots(8, 1, figsize=(10, 24), sharex=True)

    axes[0].plot(diag_times, diag_sst, "b-o", markersize=3, label="SST (prescribed)")
    axes[0].plot(diag_times, diag_T_low, "r--s", markersize=3, label="T_low (atm)")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title(f"AMIP C{N}/L{NLEV}: {args.dataset}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(diag_times, diag_sic, "c-o", markersize=3)
    axes[1].set_ylabel("Mean SIC")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(diag_times, diag_T_atm, "r-o", markersize=3)
    axes[2].set_ylabel("Global-mean T_atm [K]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(diag_times, diag_max_wind, "g-o", markersize=3)
    axes[3].set_ylabel("Max wind speed [m/s]")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(diag_times, diag_precip, "c-o", markersize=3)
    axes[4].set_ylabel("Precip [mm/day]")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(diag_times, diag_CWV, "blue", marker="s", markersize=3)
    axes[5].set_ylabel("CWV [kg/m2]")
    axes[5].grid(True, alpha=0.3)

    axes[6].plot(diag_times, diag_sw_up_toa, "orange", marker="o", markersize=3, label="SW up TOA")
    axes[6].plot(diag_times, diag_lw_up_toa, "red", marker="s", markersize=3, label="LW up TOA")
    axes[6].set_ylabel("TOA flux [W/m2]")
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    axes[7].plot(diag_times, np.array(diag_dry_mass) / 100.0, "k-o", markersize=3)
    axes[7].set_ylabel("Mean p_s [hPa]")
    axes[7].set_xlabel("Time [days]")
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amip_timeseries.png", dpi=150)
    plt.close()
    print(f"  Saved timeseries plot")

    # ---- Vertical profile evolution ----
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
        print(f"  Saved profile evolution")

    # ---- Global lat-lon final state ----
    try:
        T_sfc_final = np.asarray(sic * _T_ice + (1.0 - sic) * sst)
        wind_sfc = np.asarray(jnp.sqrt(state.u.data[..., -1]**2 + state.v.data[..., -1]**2))
        q_sfc = np.asarray(q_v[..., -1]) * 1000.0
        precip_map = np.asarray(_precip + _precip_ls) * 86400.0
        toa_net_map = np.asarray(-(_sw_toa + _lw_toa))
        sfc_net_map = np.asarray(_sw + _lw)

        fields_to_plot = [
            ("SST [K]", np.asarray(sst), "coolwarm", None),
            ("SIC", np.asarray(sic), "Blues_r", (0, 1)),
            ("T_low [K]", np.asarray(state.T.data[..., -1]), "coolwarm", None),
            ("q_v low [g/kg]", q_sfc, "YlGnBu", None),
            ("Precip [mm/day]", precip_map, "Blues", None),
            ("TOA net rad [W/m2]", toa_net_map, "RdBu_r", None),
            ("Sfc net rad [W/m2]", sfc_net_map, "RdBu_r", None),
            ("Surface wind [m/s]", wind_sfc, "magma", None),
        ]

        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        axes = axes.ravel()

        for ax, (title, field, cmap, vlim) in zip(axes, fields_to_plot):
            lon, lat, field_ll = regrid_faces_to_latlon(field)
            kw = dict(cmap=cmap, origin="lower", aspect="auto",
                      extent=[lon[0], lon[-1], lat[0], lat[-1]])
            if vlim is not None:
                kw["vmin"], kw["vmax"] = vlim
            im = ax.imshow(field_ll, **kw)
            ax.set_title(title)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        plt.suptitle(f"AMIP final state (day {N_DAYS}), C{N}/L{NLEV}", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "amip_final_state.png", dpi=150)
        plt.close()
        print(f"  Saved final state lat-lon map")
    except Exception as e:
        print(f"  Could not produce final state lat-lon plot: {e}")

    # ---- Global lat-lon snapshot evolution ----
    snap_days_sorted = sorted(snapshots.keys())
    n_snap = len(snap_days_sorted)

    if n_snap > 0:
        fields = [
            ('SST', 'SST [K]', 'coolwarm', None),
            ('SIC', 'SIC', 'Blues_r', (0, 1)),
            ('T_low', 'T_low [K]', 'coolwarm', None),
            ('q_v_low', 'q_v low [g/kg]', 'YlGnBu', None),
            ('precip', 'Precip [mm/day]', 'Blues', None),
            ('wind', 'Wind [m/s]', 'magma', None),
            ('toa_net_rad', 'TOA net [W/m2]', 'RdBu_r', None),
            ('sfc_net_rad', 'Sfc net [W/m2]', 'RdBu_r', None),
        ]
        n_fields = len(fields)

        fig, axes = plt.subplots(n_fields, n_snap, figsize=(4.5 * n_snap, 2.8 * n_fields))
        if n_snap == 1:
            axes = axes[:, None]
        if n_fields == 1:
            axes = axes[None, :]

        for j, day_snap in enumerate(snap_days_sorted):
            snap = snapshots[day_snap]
            for i, (key, label, cmap, vlim) in enumerate(fields):
                ax = axes[i, j]
                try:
                    lon, lat_arr, field_ll = regrid_faces_to_latlon(snap[key])
                    kw = dict(cmap=cmap, origin="lower", aspect="auto",
                              extent=[lon[0], lon[-1], lat_arr[0], lat_arr[-1]])
                    if vlim is not None:
                        kw["vmin"], kw["vmax"] = vlim
                    im = ax.imshow(field_ll, **kw)
                    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                except Exception:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            transform=ax.transAxes)
                if i == 0:
                    ax.set_title(f"Day {day_snap}", fontsize=11, fontweight="bold")
                if j == 0:
                    ax.set_ylabel(label, fontsize=9)
                ax.tick_params(labelsize=6)

        plt.suptitle(f"AMIP snapshots (lat-lon), C{N}/L{NLEV}", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "amip_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved lat-lon snapshot evolution")

except ImportError:
    print("  matplotlib not available, skipping plots")
except Exception as e:
    print(f"  Plotting error: {e}")

print(f"\n  All outputs saved to {OUTPUT_DIR}/")
print("  Done.")
