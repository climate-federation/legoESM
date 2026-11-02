"""NPZD on a real 3D legoESM ocean — global lat-lon C-grid with full dynamics.

Physics
-------
- LatLonCGridOceanModel: split-explicit baroclinic/barotropic, C-grid
  advection (DST3), linear EOS, KPP-style vertical mixing, GM/Redi,
  convective adjustment, surface T restoring + 3-belt wind stress.
- Biogeochemistry: legoESM NPZD (NO3/Phyto/Zoo/Det/DIC/ALK) from
  legoesm.ocean.biogeochemistry, stepped each ocean dt after dynamics.

Grid
----
- 36 × 72 (5° resolution), 20 z-star levels, 4000 m deep.
- Simplified continent: land strip 20-60°E except south of 55°S
  (Drake Passage open).

Output
------
For each of the 6 BGC tracers one PNG per tracer saved as
  experiments/npzd_3d_<tracer>.png
Each figure shows nz = 20 subplots (4 rows × 5 cols), one panel per
depth level, showing the annual-mean lat-lon map at year 10.

Usage
-----
    JAX_ENABLE_X64=1 .venv/bin/python experiments/npzd_3d_ocean.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean,
    replace_land_mask,
    idealized_bathymetry_latlon_cgrid,
)
from legoesm.ocean.init_woa import init_ocean_from_woa
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.advection import multidim_tracer_advection
from legoesm.ocean.vertical import (
    compute_layer_thickness,
    flux_form_vertical_tracer_advection,
    diagnose_w_from_flux_div,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_uface,
    min_cell_to_vface,
    divergence_cgrid,
)
from legoesm.ocean.biogeochemistry import (
    BiogeoConfig,
    OceanBiogeoState,
    init_biogeo_state,
    step_ocean_biogeochemistry,
)
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_forcings as go_forcings,
    create_eos_config as go_eos,
)
from legoesm.ocean.experiments.global_barotropic_wind import (
    create_simplified_continent_mask,
)

# =============================================================================
# 1. GRID
# =============================================================================
N_LAT, N_LON = 36, 72        # 5-degree global grid
N_LEV = 20
H_MAX = 4000.0
DT    = 1800.0               # 30-min timestep
DAYS  = 365 * 5              # 5-year spin-up

print(f"Grid: {N_LAT}×{N_LON}, {N_LEV} levels, H_max={H_MAX} m")
grid   = create_latlon_grid(N_LAT, N_LON)
z_coord = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX,
                               dz_surface=10.0, dz_deep=400.0)

# =============================================================================
# 2. LAND MASK & BATHYMETRY
# =============================================================================
go_cfg = GlobalOverturningConfig(H_max=H_MAX, n_levels=N_LEV)

# Flat bottom, polar caps closed above 80°
H_bathy, _land_flat = idealized_bathymetry_latlon_cgrid(grid, H_max=H_MAX,
                                                         land_lat_threshold=80.0)

# Simplified continent (Atlantic-like) + Drake Passage open
lon_deg = np.degrees(np.asarray(grid.lon2d))
lat_deg = np.degrees(np.asarray(grid.lat2d))
cont_mask = create_simplified_continent_mask(lon_deg, lat_deg, go_cfg)
land_mask  = jnp.array(cont_mask.astype(np.float64))

# =============================================================================
# 3. INITIAL OCEAN STATE (WOA18 analytical T/S)
# =============================================================================
print("Initialising ocean state (WOA18 analytical T/S)…")
state = rest_state_latlon_cgrid_ocean(
    grid, z_coord,
    T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
)
state = replace_land_mask(state, land_mask)

# Apply WOA18 analytical T/S profiles (analytical fallback, no files needed)
try:
    from legoesm.core.field import Field
    T_woa, S_woa = init_ocean_from_woa(grid, z_coord,
                                        T_path=None, S_path=None)
    state = state._replace(
        T=Field(jnp.array(T_woa), name="T", dims=state.T.dims, units=state.T.units),
        S=Field(jnp.array(S_woa), name="S", dims=state.S.dims, units=state.S.units),
    )
    print("  WOA18 analytical T/S applied.")
except Exception as e:
    print(f"  WOA18 init skipped ({e}); using rest state T/S.")

# =============================================================================
# 4. PHYSICS / DYNAMICS CONFIG
# =============================================================================
import dataclasses as _dc
go_cfg_strong_restore = _dc.replace(go_cfg, tau_T_days=5.0)
physics = go_forcings("latlon", grid, go_cfg_strong_restore)
eos_cfg = go_eos(go_cfg)

# ------------------------------------------------------------------
# 4a. 2-D SST restoring target with east-west asymmetry
# ------------------------------------------------------------------
# The default cos²(lat) target is purely zonal — the ocean can never
# develop realistic east-west asymmetry. We build a simple 2-D pattern
# that captures the key features:
#   - equatorial cold tongue in the east (eastern Pacific/Atlantic)
#   - warm pool in the western Pacific (~145-180°E)
#   - cold eastern boundary currents (California, Peru, Benguela, Canary)
# Grid longitudes are 0→360°.
_lat_r = np.asarray(grid.lat2d)           # (nlat, nlon), radians
_lon_d = np.degrees(np.asarray(grid.lon2d))  # (nlat, nlon), 0→360°
_lat_d = np.degrees(_lat_r)

# Base zonal profile: cos²(lat)  (same equatorial max as default, 25 °C)
_T_star_zonal = 25.0 * np.cos(_lat_r) ** 2

# Equatorial cold tongue: eastern Pacific/Atlantic (lon 200-360°) and
# Indian Ocean east (lon 60-100°).  Cool by up to 4°C near equator.
_cold_tongue_lon = ((_lon_d > 200.0) | (_lon_d < 10.0)) | \
                   ((_lon_d > 60.0) & (_lon_d < 100.0))
_eq_weight = np.exp(-(_lat_d / 10.0) ** 2)          # lat Gaussian (σ=10°)
_T_star_2d = _T_star_zonal - 4.0 * _cold_tongue_lon * _eq_weight

# Warm pool: western Pacific (120-180°E), equatorial/subtropical
_warm_pool = ((_lon_d > 120.0) & (_lon_d < 180.0))
_T_star_2d = _T_star_2d + 3.0 * _warm_pool * np.exp(-(_lat_d / 20.0) ** 2)

# Eastern boundary cooling: California (lon 220-250°, lat 20-60°N),
# Peru (lon 270-290°, lat 5-35°S), Canary (lon 340-360°, lat 10-40°N)
def _eb_cool(lon_w, lon_e, lat_s, lat_n, mag=3.0):
    _in_lon = (_lon_d > lon_w) & (_lon_d < lon_e)
    _in_lat = (_lat_d > lat_s) & (_lat_d < lat_n)
    return -mag * (_in_lon & _in_lat).astype(float)

_T_star_2d += _eb_cool(220, 250, 20, 60)     # California current
_T_star_2d += _eb_cool(270, 290, -35, -5)    # Peru current
_T_star_2d += _eb_cool(340, 360, 10, 40)     # Canary current
_T_star_2d += _eb_cool( 10,  25, -35, -10)   # Benguela current

# Smooth the target (simple 3-point box filter applied twice)
from scipy.ndimage import uniform_filter
_T_star_2d = uniform_filter(_T_star_2d, size=[3, 5], mode="wrap")

# Inject the 2D target into the physics config
_r = physics.surface_forcing.restoring._replace(
    T_star_array=_T_star_2d.astype(np.float64),
    tau_S=60.0 * 86400.0,   # also enable 60-day SSS restoring
)
_sf = physics.surface_forcing._replace(restoring=_r)
physics = physics._replace(surface_forcing=_sf)
print("2-D SST target with east-west asymmetry applied.")

model_cfg = LatLonCGridOceanConfig(
    n_barotropic_substeps=30,
    physics=physics,
    A_h=2e5,
    K_h=2e4,          # raised from 1e3: damps the 2*dx grid-scale tracer mode
    A_v=1e-3,
    K_v=1e-5,
    bottom_drag_r=1.1e-3,
    eos="linear",
    eos_linear=eos_cfg,
    tracer_advection="dst3",
)
model = LatLonCGridOceanModel(grid, z_coord, model_cfg)
print("Ocean model built.")

# =============================================================================
# 5. BGC INITIAL STATE
# =============================================================================
bgc_cfg = BiogeoConfig(
    scheme="npzd",
    pCO2_atm=415.0,
    wind_speed=7.0,
    mu_max=1.5,
    k_N=0.7e-3,
    alpha_P=0.025,
    g_max=0.6,
    k_P=0.2e-3,
    gamma_Z=0.7,
    epse=0.5,
    m_P=0.05,
    m_Pq=0.2,
    m_Zl=0.02,
    m_Z=0.2,
    remin_rate=0.15,   # increased from 0.05: faster mid-water remineralisation
    w_sink=2.0,        # reduced from 10: less material reaches the seafloor
    k_w_atten=0.04,
    k_chl_atten=25.0,
    R_CN=6.625,
    R_ON=10.625,
    R_CaP=0.07,
    # Initial conditions
    NO3_init_surf=5e-3,
    NO3_init_deep=30e-3,
    Phyto_init=0.1e-3,
    Zoo_init=0.05e-3,
    Det_init=0.01e-3,
    DIC_init=2.1,
    ALK_init=2.3,
)

bgc_shape = (N_LAT, N_LON, N_LEV)
bgc_state = init_biogeo_state(
    (N_LAT, N_LON, N_LEV),
    jnp.array(z_coord.z_full_ref),
    bgc_cfg,
)
print("BGC state initialised.")

# =============================================================================
# 6. JIT-COMPILED BGC STEP
# =============================================================================
dz_jax   = jnp.array(z_coord.dz_ref)
z_full_jax = jnp.array(z_coord.z_full_ref)
land_mask_2d = jnp.array(land_mask)   # (nlat, nlon), 1=ocean 0=land
mask_3d = land_mask_2d[..., jnp.newaxis]  # (nlat, nlon, 1)

# Background vertical diffusivity for BGC tracers (matches ocean K_v)
K_V_BGC = 1e-5   # m²/s
K_H_BGC = 2e4    # m²/s — matches ocean K_h; damps 2Δx grid-scale noise

# Precompute cell spacings for horizontal diffusion
_dx_bgc = jnp.array(np.asarray(grid.dx))                          # (nlat, nlon)
_dy_bgc = jnp.array(np.asarray(grid.dy)[:, None] * np.ones((1, N_LON)))  # (nlat, nlon)


@jax.jit
def bgc_step(bgc: OceanBiogeoState, T: jnp.ndarray,
             S: jnp.ndarray, PAR_surf: jnp.ndarray) -> OceanBiogeoState:
    new_bgc, _ = step_ocean_biogeochemistry(
        bgc, T, S, dz_jax, z_full_jax,
        land_mask_2d, DT, bgc_cfg,
        U10=jnp.full((N_LAT, N_LON), bgc_cfg.wind_speed),
        PAR_surf=PAR_surf,
    )
    return new_bgc


@jax.jit
def prepare_transport(
    u: jnp.ndarray, v: jnp.ndarray,
    eta: jnp.ndarray, H_bathy: jnp.ndarray,
    u_mask: jnp.ndarray, v_mask: jnp.ndarray,
):
    """Compute mass-conserving mass fluxes and w_baro (half-level) from the ocean state."""
    h_k  = compute_layer_thickness(eta, H_bathy, z_coord)
    h_u  = min_cell_to_uface(h_k)
    h_v  = min_cell_to_vface(h_k, grid)
    # Apply face masks to zero out land-adjacent fluxes
    mf_u = h_u * u * u_mask[..., jnp.newaxis]
    mf_v = h_v * v * v_mask[..., jnp.newaxis]
    # Diagnose w at half-levels from horizontal flux divergence
    # (mass-conserving, consistent with tracer advection in ocean model)
    flux_div_k = divergence_cgrid(mf_u, mf_v, grid)
    w_baro = diagnose_w_from_flux_div(flux_div_k, z_coord, thickness_weighted=True)
    return mf_u, mf_v, w_baro, h_k, h_u, h_v


@jax.jit
def advect_one_tracer(
    tracer: jnp.ndarray,
    mf_u: jnp.ndarray, mf_v: jnp.ndarray, w_baro: jnp.ndarray,
    h_k: jnp.ndarray,  h_u: jnp.ndarray,  h_v: jnp.ndarray,
) -> jnp.ndarray:
    """DST-3 advection + vertical + horizontal diffusion for a single BGC tracer."""
    # Apply land mask before advection
    tr = tracer * mask_3d
    div_h, vert_div = multidim_tracer_advection(
        tr, mf_u, mf_v, w_baro, h_k, h_u, h_v, grid, DT,
    )
    # Vertical diffusion
    dz_half    = 0.5 * (dz_jax[:-1] + dz_jax[1:])
    F_diff_int = K_V_BGC * jnp.diff(tr, axis=-1) / dz_half
    F_diff = jnp.concatenate(
        [jnp.zeros_like(tr[..., :1]), F_diff_int,
         jnp.zeros_like(tr[..., :1])], axis=-1)
    vdiff_tend = jnp.diff(F_diff, axis=-1) / dz_jax
    # Horizontal Laplacian diffusion (Δ²ψ in lon/lat, land-masked)
    # ∂²tr/∂x² ≈ (tr[i+1] - 2*tr[i] + tr[i-1]) / dx²
    dx2 = _dx_bgc[..., jnp.newaxis] ** 2
    dy2 = _dy_bgc[..., jnp.newaxis] ** 2
    tr_xp = jnp.roll(tr,  1, axis=1) * mask_3d
    tr_xm = jnp.roll(tr, -1, axis=1) * mask_3d
    tr_yp = jnp.concatenate([tr[1:2], tr[:-1]], axis=0) * mask_3d
    tr_ym = jnp.concatenate([tr[1:],  tr[-1:]], axis=0) * mask_3d
    lap = ((tr_xp - 2.0 * tr + tr_xm) / dx2
         + (tr_yp - 2.0 * tr + tr_ym) / dy2)
    hdiff_tend = K_H_BGC * lap * mask_3d
    h_safe = jnp.maximum(h_k, 1.0)
    new_tr = (h_k * tr - DT * (div_h + vert_div)) / h_safe + DT * (vdiff_tend + hdiff_tend)
    return new_tr * mask_3d


def seasonal_par(t_days: float) -> jnp.ndarray:
    """Latitude-dependent seasonal PAR [W/m²] at time t (days)."""
    phase = 2.0 * np.pi * (t_days % 365.25) / 365.25 - np.pi / 2.0
    lat_r = np.asarray(grid.lat2d)           # (nlat, nlon), radians
    lat_d = np.degrees(lat_r)
    lon_d = np.degrees(np.asarray(grid.lon2d))  # 0→360°

    # Base zonal PAR: pole-to-equator mean × seasonal swing (SH phase flip)
    par_mean = 100.0 + 180.0 * (1.0 - np.abs(lat_r) / (np.pi / 2.0))
    phase_2d  = phase - np.where(lat_r < 0, np.pi, 0.0)
    par_val   = par_mean * np.maximum(np.sin(phase_2d), 0.0)

    # ---- 2-D cloud-cover modulation ----
    # Preserves the zonal mean (factor averages to ~1 globally) but breaks
    # the zonal symmetry so phytoplankton patterns develop longitude structure.

    # 1. Subtropical high-pressure clear zones: eastern subtropical ocean
    #    cells receive ~15% more light (low cloud). Centred ~25°N/S, eastern basins.
    cloud_clear = 0.15 * (
        np.exp(-((lat_d - 25.0) / 12.0) ** 2) * np.exp(-((lon_d - 230.0) / 40.0) ** 2)  # NE Pacific
      + np.exp(-((lat_d + 20.0) / 12.0) ** 2) * np.exp(-((lon_d - 270.0) / 30.0) ** 2)  # SE Pacific
      + np.exp(-((lat_d - 25.0) / 12.0) ** 2) * np.exp(-((lon_d - 345.0) / 30.0) ** 2)  # NE Atlantic
      + np.exp(-((lat_d + 20.0) / 12.0) ** 2) * np.exp(-((lon_d -   5.0) / 25.0) ** 2)  # SE Atlantic
    )

    # 2. ITCZ / maritime continent: near-equatorial cloudy band in the
    #    western Pacific and Indian Ocean → ~15% less light.
    cloud_itcz = 0.15 * (
        np.exp(-((lat_d -  5.0) / 8.0) ** 2) * np.exp(-((lon_d - 120.0) / 60.0) ** 2)  # Maritime continent
      + np.exp(-((lat_d -  5.0) / 8.0) ** 2) * np.exp(-((lon_d -  60.0) / 40.0) ** 2)  # Indian ITCZ
    )

    cloud_factor = 1.0 + cloud_clear - cloud_itcz
    par_val = par_val * cloud_factor

    return jnp.array(par_val)

# =============================================================================
# 7. TIME LOOP — 10-year spin-up, accumulate running mean in last year
# =============================================================================
n_steps     = int(DAYS * 86400.0 / DT)
save_start  = int((DAYS - 365) * 86400.0 / DT)   # last-year start step
save_every  = max(1, int(5 * 86400.0 / DT))       # every 5 days

# Running-sum arrays (numpy) — avoids growing a list of JAX arrays
_shape = (N_LAT, N_LON, N_LEV)
sum_NO3   = np.zeros(_shape)
sum_Phyto = np.zeros(_shape)
sum_Zoo   = np.zeros(_shape)
sum_Det   = np.zeros(_shape)
sum_DIC   = np.zeros(_shape)
sum_ALK   = np.zeros(_shape)
sum_T     = np.zeros(_shape)
n_saved   = 0

print(f"\nRunning {DAYS // 365}-year spin-up: {n_steps} steps (dt={DT}s)…")
t0_wall = time.time()

for k in range(n_steps):
    t_days = k * DT / 86400.0

    # ---- Ocean dynamics ----
    state = model.step(state, dt=DT)

    # ---- BGC step ----
    # Sanitize T/S at land cells so carbonate solver can't return NaN
    # (NaN * ocean_mask = NaN in IEEE 754, which would corrupt all DIC/ALK)
    T_3d = jnp.where(land_mask_2d[..., None] > 0.5,
                     jnp.array(state.T.data), 15.0)
    S_3d = jnp.where(land_mask_2d[..., None] > 0.5,
                     jnp.array(state.S.data), 35.0)
    PAR  = seasonal_par(t_days)
    # Clamp DIC/ALK to the range the carbonate solver was designed for,
    # and clean up any NaN that may have accumulated.
    bgc_clean = OceanBiogeoState(
        NO3   = jnp.nan_to_num(bgc_state.NO3,   nan=1e-3),
        Phyto = jnp.nan_to_num(bgc_state.Phyto, nan=1e-3),
        Zoo   = jnp.nan_to_num(bgc_state.Zoo,   nan=1e-3),
        Det   = jnp.nan_to_num(bgc_state.Det,   nan=1e-3),
        DIC   = jnp.clip(jnp.nan_to_num(bgc_state.DIC, nan=2.1), 1.8, 2.5),
        ALK   = jnp.clip(jnp.nan_to_num(bgc_state.ALK, nan=2.3), 2.0, 2.6),
    )
    bgc_state = bgc_step(bgc_clean, T_3d, S_3d, PAR)

    # ---- BGC transport: advection + vertical diffusion by ocean velocity ----
    # Compile once for a single tracer, reuse 6 times — avoids 6× XLA graph OOM
    _u     = jnp.array(state.u.data)
    _v     = jnp.array(state.v.data)
    _eta   = jnp.array(state.eta.data)
    _Hb    = jnp.array(state.H_bathy.data)
    _umask = jnp.array(state.u_mask.data)
    _vmask = jnp.array(state.v_mask.data)
    mf_u, mf_v, w_baro, h_k, h_u, h_v = prepare_transport(
        _u, _v, _eta, _Hb, _umask, _vmask)
    bgc_state = OceanBiogeoState(
        NO3   = advect_one_tracer(bgc_state.NO3,   mf_u, mf_v, w_baro, h_k, h_u, h_v),
        Phyto = advect_one_tracer(bgc_state.Phyto, mf_u, mf_v, w_baro, h_k, h_u, h_v),
        Zoo   = advect_one_tracer(bgc_state.Zoo,   mf_u, mf_v, w_baro, h_k, h_u, h_v),
        Det   = advect_one_tracer(bgc_state.Det,   mf_u, mf_v, w_baro, h_k, h_u, h_v),
        DIC   = advect_one_tracer(bgc_state.DIC,   mf_u, mf_v, w_baro, h_k, h_u, h_v),
        ALK   = advect_one_tracer(bgc_state.ALK,   mf_u, mf_v, w_baro, h_k, h_u, h_v),
    )
    # Non-negativity floor applied ONCE per timestep (after transport + source/sink)
    # nan_to_num guards against any residual NaN before the floor check.
    _eps = 1e-12
    bgc_state = OceanBiogeoState(
        NO3   = jnp.maximum(jnp.nan_to_num(bgc_state.NO3,   nan=_eps),  _eps),
        Phyto = jnp.maximum(jnp.nan_to_num(bgc_state.Phyto, nan=_eps),  _eps),
        Zoo   = jnp.maximum(jnp.nan_to_num(bgc_state.Zoo,   nan=_eps),  _eps),
        Det   = jnp.maximum(jnp.nan_to_num(bgc_state.Det,   nan=_eps),  _eps),
        DIC   = jnp.maximum(jnp.nan_to_num(bgc_state.DIC,   nan=2.1),  1e-3),
        ALK   = jnp.maximum(jnp.nan_to_num(bgc_state.ALK,   nan=2.3),  1e-3),
    )

    # ---- Accumulate last-year running mean (numpy, no JAX history) ----
    if k >= save_start and k % save_every == 0:
        sum_NO3   += np.asarray(bgc_state.NO3)
        sum_Phyto += np.asarray(bgc_state.Phyto)
        sum_Zoo   += np.asarray(bgc_state.Zoo)
        sum_Det   += np.asarray(bgc_state.Det)
        sum_DIC   += np.asarray(bgc_state.DIC)
        sum_ALK   += np.asarray(bgc_state.ALK)
        sum_T     += np.asarray(state.T.data)   # raw ocean T, not the BGC-sanitized copy
        n_saved   += 1

    if (k + 1) % max(1, n_steps // 50) == 0:
        # Block until JAX has materialised all pending computations,
        # then release device buffers and Python objects.
        jax.block_until_ready((
            bgc_state.NO3, bgc_state.Phyto, bgc_state.Zoo,
            bgc_state.Det, bgc_state.DIC,  bgc_state.ALK,
        ))
        jax.clear_caches()
        gc.collect()
    if (k + 1) % max(1, n_steps // 20) == 0:
        pct = 100.0 * (k + 1) / n_steps
        elapsed = time.time() - t0_wall
        _no3 = np.asarray(bgc_state.NO3)
        _dic = np.asarray(bgc_state.DIC)
        _alk = np.asarray(bgc_state.ALK)
        print(f"  {pct:5.1f}%  day {t_days:7.1f}  elapsed {elapsed:.0f}s"
              f"  NO3 mean={np.nanmean(_no3)*1e3:.1f}  "
              f"DIC nan={np.isnan(_dic).sum()}  ALK max={np.nanmax(_alk)*1e3:.0f}")

elapsed = time.time() - t0_wall
print(f"Done in {elapsed:.1f}s  ({elapsed / n_steps * 1000:.1f} ms/step).")
print(f"Accumulated {n_saved} snapshots for annual mean.")

# =============================================================================
# 8. ANNUAL MEAN (last year)
# =============================================================================
ann = {
    "NO3":   (sum_NO3   / n_saved) * 1e3,   # mol/m³ → mmol/m³
    "Phyto": (sum_Phyto / n_saved) * 1e3,
    "Zoo":   (sum_Zoo   / n_saved) * 1e3,
    "Det":   (sum_Det   / n_saved) * 1e3,
    "DIC":   (sum_DIC   / n_saved) * 1e3,
    "ALK":   (sum_ALK   / n_saved) * 1e3,
}
T_ann = sum_T / n_saved   # (nlat, nlon, nlev)

# Save annual means for fast replotting
np.save("experiments/ann_means.npy", {**ann, "T": T_ann})
print("Annual means saved to experiments/ann_means.npy")

# =============================================================================
# 9. PLOTTING — 2×4 panel layout at 8 key depths, PlateCarree projection
# =============================================================================
# Apply land mask to all fields (set ocean=1 cells to NaN where land)
ocean_mask_2d = np.asarray(land_mask)          # 1=ocean, 0=land (nlat, nlon)
ocean_mask_3d = ocean_mask_2d[:, :, np.newaxis] * np.ones((1, 1, N_LEV))

def _mask_land(arr3d):
    out = arr3d.copy().astype(float)
    out[ocean_mask_3d < 0.5] = np.nan
    return out

T_ann   = _mask_land(T_ann)
for k in ann:
    ann[k] = _mask_land(ann[k])

# DIC and ALK: show anomaly from global mean so spatial structure is visible
for k in ("DIC", "ALK"):
    ann[k] = ann[k] - np.nanmean(ann[k])

# 8 representative depth indices (nearest to target depths)
target_depths_m = [5, 25, 50, 100, 200, 500, 1000, 2000]
z_centers = np.abs(np.array(z_coord.z_full_ref))
depth_idx  = [int(np.argmin(np.abs(z_centers - d))) for d in target_depths_m]
depth_labels_sel = [f"{int(z_centers[k])} m" for k in depth_idx]

specs = [
    ("NO3",   "NO$_3$ (mmol N m$^{-3}$)",        "YlGn_r",   False),
    ("Phyto", "Phytoplankton (mmol N m$^{-3}$)",  "YlGn",     False),
    ("Zoo",   "Zooplankton (mmol N m$^{-3}$)",    "Oranges",  False),
    ("Det",   "Detritus (mmol N m$^{-3}$)",       "YlOrRd",   False),
    ("DIC",   "DIC anomaly (mmol C m$^{-3}$)",    "RdBu_r",   True),
    ("ALK",   "ALK anomaly (mmol eq m$^{-3}$)",   "RdBu_r",   True),
    # T: sequential colormap, physical range [-2, 30] clips the Drake-Passage hot spot
    ("T",     "Temperature (°C)",                 "RdYlBu_r", False),
]

proj     = ccrs.PlateCarree()
data_crs = ccrs.PlateCarree()
ncols, nrows = 4, 2
out_dir = "experiments"
os.makedirs(out_dir, exist_ok=True)

for key, title, cmap, diverging in specs:
    data_3d = ann[key] if key in ann else T_ann

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 3.2),
        subplot_kw=dict(projection=proj),
        gridspec_kw=dict(hspace=0.45, wspace=0.08),
    )

    # Roll data by N_LON//2 so lon-0 grid (0→360°) maps to -180→180° extent
    data_3d_rolled = np.roll(data_3d, N_LON // 2, axis=1)

    for panel_i, (k, ax) in enumerate(zip(depth_idx, axes.flatten())):
        data_k = data_3d_rolled[:, :, k]
        # Shared symmetric or one-sided limits across all 8 depths
        valid = data_k[np.isfinite(data_k)]
        if valid.size == 0:
            ax.set_visible(False)
            continue
        p1  = float(np.nanpercentile(data_k, 2))
        p99 = float(np.nanpercentile(data_k, 98))
        if diverging:
            vabs = max(abs(p1), abs(p99), 1e-6)
            vmin, vmax = -vabs, vabs
        elif key == "T":
            # Physical temperature range; clips numerical hot-spots
            vmin = max(p1, -2.0)
            vmax = min(p99, 30.0)
        else:
            vmin = 0.0
            vmax = max(p99, 1e-6)

        im = ax.imshow(
            data_k, origin="lower",
            extent=[-180, 180, -90, 90],
            cmap=plt.get_cmap(cmap).with_extremes(bad="#c8c8c8"),
            vmin=vmin, vmax=vmax,
            transform=data_crs, interpolation="nearest",
        )
        ax.add_feature(cfeature.LAND,      facecolor="#c8c8c8", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5,        zorder=3)
        ax.set_extent([-180, 180, -90, 90], crs=data_crs)
        ax.gridlines(linewidth=0.3, color="grey", alpha=0.5,
                     xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30))
        ax.set_title(depth_labels_sel[panel_i], fontsize=10, fontweight="bold", pad=3)
        cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.05, shrink=0.9, aspect=25)
        cb.ax.tick_params(labelsize=6)
        # Tight tick formatting
        cb.ax.xaxis.set_major_formatter(
            plt.matplotlib.ticker.FormatStrFormatter("%.2g"))

    fig.suptitle(
        f"{title}\nlegoESM 3-D ocean + BGC  —  year-5 annual mean  "
        f"({N_LAT}×{N_LON}, {N_LEV} lev)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out = os.path.join(out_dir, f"npzd_3d_{key}.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ------------------------------------------------------------------
# Bonus: zonal-mean section for NO3 and T (depth vs latitude)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lat_deg = np.degrees(np.asarray(grid.lat))

for ax, key, label, cmap_s, div in [
    (axes[0], "NO3",  "NO$_3$ (mmol N m$^{-3}$)", "YlGn_r", False),
    (axes[1], "T",    "Temperature (°C)",          "RdYlBu_r", False),
]:
    data_3d = ann[key] if key in ann else T_ann
    zonal = np.nanmean(data_3d, axis=1)          # (nlat, nlev)
    depths = np.abs(np.array(z_coord.z_full_ref))
    LON_M, DEP_M = np.meshgrid(lat_deg, depths)
    p1  = float(np.nanpercentile(zonal, 2))
    p99 = float(np.nanpercentile(zonal, 98))
    if div:
        vabs = max(abs(p1), abs(p99))
        vmin2, vmax2 = -vabs, vabs
    elif key == "T":
        vmin2 = max(p1, -2.0)
        vmax2 = min(p99, 30.0)
    else:
        vmin2, vmax2 = 0.0, max(p99, 1e-6)
    im = ax.contourf(lat_deg, depths, zonal.T,
                     levels=20, cmap=cmap_s, vmin=vmin2, vmax=vmax2)
    ax.set_ylim(depths.max(), depths.min())
    ax.set_xlabel("Latitude (°)", fontsize=10)
    ax.set_ylabel("Depth (m)",    fontsize=10)
    ax.set_title(label, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.9)

fig.suptitle("Zonal-mean sections — year-5 annual mean", fontsize=13, fontweight="bold")
out_sec = os.path.join(out_dir, "npzd_3d_zonal_section.png")
plt.savefig(out_sec, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_sec}")

print("All done.")
