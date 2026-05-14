"""Diagnostic script: compare latlon C-grid vs MPAS global barotropic wind.

Runs both grids for 10 days with step-by-step diagnostics to find
where the solutions diverge.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp

from legoesm.ocean.vertical import create_ocean_z_star

# ── Shared parameters ──────────────────────────────────────────────
NLEV = 10
H_MAX = 5500.0
DT = 300.0
DAYS = 10.0
N_STEPS = int(DAYS * 86400 / DT)
T_UNIFORM = 10.0
S_UNIFORM = 35.0

z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
print(f"z_coord.dz_ref = {np.asarray(z_coord.dz_ref)}")
print(f"Top layer thickness: {float(z_coord.dz_ref[0]):.2f} m")
print(f"dt={DT}s, {N_STEPS} steps, {DAYS} days")
print()


def _create_simplified_continent_mask(lon_deg, lat_deg,
                                       continent_lon_west=20.0,
                                       continent_lon_east=60.0,
                                       continent_lat_south=-55.0,
                                       polar_cap_lat=80.0):
    lon = jnp.asarray(lon_deg)
    lat = jnp.asarray(lat_deg)
    ocean = jnp.ones_like(lat)
    ocean = jnp.where(jnp.abs(lat) > polar_cap_lat, 0.0, ocean)
    in_continent = (
        (lon >= continent_lon_west) & (lon <= continent_lon_east) &
        (lat >= continent_lat_south)
    )
    ocean = jnp.where(in_continent, 0.0, ocean)
    return ocean


# ── Physics config (same for both) ─────────────────────────────────
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig, PrescribedForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig, LinearDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig

physics = OceanPhysicsConfig(
    surface_forcing=SurfaceForcingConfig(
        scheme="prescribed",
        prescribed=PrescribedForcingConfig(
            wind_profile="global_wind",
            tau_max=0.1,
        ),
    ),
    vertical_mixing=VerticalMixingConfig(scheme="none"),
    lateral_mixing=LateralMixingConfig(scheme="none"),
    bottom_drag=BottomDragConfig(
        scheme="linear",
        linear=LinearDragConfig(r=1e-4),
    ),
    convection=OceanConvectionConfig(scheme="none"),
    shortwave_penetration=None,
)


# ── Setup latlon C-grid ─────────────────────────────────────────────
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.core.field import Field

ll_grid = create_latlon_grid(36, 72)
ll_config = LatLonCGridOceanConfig(n_barotropic_substeps=30, physics=physics, A_h=5e5)
ll_model = LatLonCGridOceanModel(ll_grid, z_coord, ll_config)

ll_state = rest_state_latlon_cgrid_ocean(
    ll_grid, z_coord, T_water_init_C=T_UNIFORM, T_deep=T_UNIFORM, S_uniform=S_UNIFORM)
lon_2d = np.asarray(ll_grid.lon, dtype=np.float64) * 180 / np.pi
lat_1d = np.asarray(ll_grid.lat, dtype=np.float64) * 180 / np.pi
lon_grid, lat_grid = np.meshgrid(lon_2d, lat_1d)
ll_mask = _create_simplified_continent_mask(lon_grid, lat_grid)
mask_typed = ll_mask.astype(ll_state.eta.data.dtype)
u_mask_new, v_mask_new = compute_face_masks(mask_typed)
ll_state = ll_state._replace(
    land_mask=Field(data=mask_typed),
    u_mask=Field(data=u_mask_new),
    v_mask=Field(data=v_mask_new),
)

# ── Setup MPAS ──────────────────────────────────────────────────────
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.init_mpas import rest_state_mpas_ocean

mpas_mesh = create_voronoi_mesh(3)
mpas_config = MPASOceanConfig(n_barotropic_substeps=30, physics=physics, A_h=5e5)
mpas_model = MPASOceanModel(mpas_mesh, z_coord, mpas_config)

mpas_state = rest_state_mpas_ocean(
    mpas_mesh, z_coord, T_water_init_C=T_UNIFORM, T_deep=T_UNIFORM, S_uniform=S_UNIFORM)
lon_deg_c = np.asarray(mpas_mesh.lonCell, dtype=np.float64) * 180 / np.pi
lat_deg_c = np.asarray(mpas_mesh.latCell, dtype=np.float64) * 180 / np.pi
mpas_mask = _create_simplified_continent_mask(lon_deg_c, lat_deg_c)
mpas_state = mpas_state._replace(
    land_mask=Field(data=mpas_mask.astype(mpas_state.eta.data.dtype)))

# ── Check wind forcing magnitude ────────────────────────────────────
print("=" * 70)
print("WIND FORCING CHECK")
print("=" * 70)

# Latlon: what does the physics pipeline produce?
from legoesm.ocean.physics.combined import make_ocean_physics
ll_phys_fn = make_ocean_physics(physics)

# Create cell-center proxy state for latlon
u_cc = 0.5 * (ll_state.u.data[:, :-1, :] + ll_state.u.data[:, 1:, :])
v_cc = 0.5 * (ll_state.v.data[:-1, :, :] + ll_state.v.data[1:, :, :])
cc_state = ll_state._replace(
    u=ll_state.u.replace(data=u_cc),
    v=ll_state.v.replace(data=v_cc),
)
ll_phys = ll_phys_fn(cc_state, ll_grid, z_coord)
print(f"\nLatlon physics output shapes:")
print(f"  du_dt: {ll_phys.du_dt.data.shape}  (cell centers)")
print(f"  max|du_dt|: {float(jnp.max(jnp.abs(ll_phys.du_dt.data))):.6e}")
print(f"  max|dv_dt|: {float(jnp.max(jnp.abs(ll_phys.dv_dt.data))):.6e}")
# Wind stress itself
lat_rad = jnp.asarray(ll_grid.grid_lat)
taper = jnp.cos(lat_rad) ** 2
tau_x_ll = -0.1 * jnp.cos(2.0 * lat_rad) * taper
print(f"  max|tau_x|: {float(jnp.max(jnp.abs(tau_x_ll))):.4f} N/m^2")

# MPAS: what does the physics pipeline produce?
from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
mpas_phys_fn = make_mpas_ocean_physics(physics)
mpas_phys = mpas_phys_fn(mpas_state, mpas_mesh, z_coord)
print(f"\nMPAS physics output shapes:")
print(f"  du_dt: {mpas_phys.du_dt.data.shape}  (edge normals)")
print(f"  max|du_dt|: {float(jnp.max(jnp.abs(mpas_phys.du_dt.data))):.6e}")
# Wind stress
lat_mpas = jnp.asarray(mpas_mesh.latCell)
taper_m = jnp.cos(lat_mpas) ** 2
tau_x_mpas = -0.1 * jnp.cos(2.0 * lat_mpas) * taper_m
print(f"  max|tau_x|: {float(jnp.max(jnp.abs(tau_x_mpas))):.4f} N/m^2")

# ── Check tendencies after first step ───────────────────────────────
print("\n" + "=" * 70)
print("BAROCLINIC TENDENCIES (before first step)")
print("=" * 70)

from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import latlon_cgrid_ocean_baroclinic_tendencies
ll_tend = latlon_cgrid_ocean_baroclinic_tendencies(
    ll_state, ll_grid, z_coord, ll_config,
    physics_fn=ll_phys_fn)
print(f"\nLatlon C-grid tendencies:")
print(f"  max|du_dt|: {float(jnp.max(jnp.abs(ll_tend.du_dt.data))):.6e}")
print(f"  max|dv_dt|: {float(jnp.max(jnp.abs(ll_tend.dv_dt.data))):.6e}")
print(f"  max|deta_dt|: {float(jnp.max(jnp.abs(ll_tend.deta_dt.data))):.6e}")

from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
mpas_tend = mpas_ocean_baroclinic_tendencies(
    mpas_state, mpas_mesh, z_coord, mpas_config,
    physics_fn=mpas_phys_fn)
print(f"\nMPAS tendencies:")
print(f"  max|du_dt|: {float(jnp.max(jnp.abs(mpas_tend.du_dt.data))):.6e}")
print(f"  max|deta_dt|: {float(jnp.max(jnp.abs(mpas_tend.deta_dt.data))):.6e}")


# ── Time loop with diagnostics ──────────────────────────────────────
print("\n" + "=" * 70)
print("TIME EVOLUTION (step-by-step for first 10 steps, then every 50)")
print("=" * 70)

def ll_diagnostics(state):
    eta = np.asarray(state.eta.data)
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    mask = np.asarray(state.land_mask.data)
    # Speed proxy: max of |u| and |v| at surface
    u_sfc = np.abs(u[:, :, 0])  # (n_lat, n_lon+1)
    v_sfc = np.abs(v[:, :, 0])  # (n_lat+1, n_lon)
    max_u = float(np.max(u_sfc))
    max_v = float(np.max(v_sfc))
    max_speed = max(max_u, max_v)
    ocean = mask > 0.5
    mean_eta = float(np.mean(eta[ocean])) if np.any(ocean) else 0.0
    max_eta = float(np.max(np.abs(eta[ocean]))) if np.any(ocean) else 0.0
    # KE proxy (surface only)
    u_cc = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_cc = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    ke = 0.5 * (u_cc**2 + v_cc**2)
    mean_ke = float(np.mean(ke[ocean])) if np.any(ocean) else 0.0
    return max_speed, max_eta, mean_eta, mean_ke

def mpas_diagnostics(state):
    eta = np.asarray(state.eta.data)
    u = np.asarray(state.u.data)  # (nEdges, nlev) - edge-normal velocity
    mask = np.asarray(state.land_mask.data)
    edge_mask = np.asarray(state.edge_mask.data) if hasattr(state, 'edge_mask') else None
    max_u = float(np.max(np.abs(u[:, 0])))
    ocean = mask > 0.5
    mean_eta = float(np.mean(eta[ocean])) if np.any(ocean) else 0.0
    max_eta = float(np.max(np.abs(eta[ocean]))) if np.any(ocean) else 0.0
    mean_ke = float(np.mean(u[:, 0]**2)) * 0.5  # rough proxy
    return max_u, max_eta, mean_eta, mean_ke

print(f"\n{'Step':>6} {'Day':>7} | {'LL max_spd':>10} {'LL max_eta':>10} {'LL mean_eta':>11} {'LL mean_KE':>10}"
      f" | {'MP max_spd':>10} {'MP max_eta':>10} {'MP mean_eta':>11} {'MP mean_KE':>10}")
print("-" * 130)

ll_s = ll_state
mpas_s = mpas_state

for step in range(N_STEPS):
    # Print diagnostics
    if step < 10 or step % 50 == 0 or step == N_STEPS - 1:
        ll_spd, ll_me, ll_mea, ll_ke = ll_diagnostics(ll_s)
        mp_spd, mp_me, mp_mea, mp_ke = mpas_diagnostics(mpas_s)
        day = step * DT / 86400
        print(f"{step:6d} {day:7.2f} | {ll_spd:10.4e} {ll_me:10.4e} {ll_mea:11.4e} {ll_ke:10.4e}"
              f" | {mp_spd:10.4e} {mp_me:10.4e} {mp_mea:11.4e} {mp_ke:10.4e}")

    ll_s = ll_model.step(ll_s, DT)
    mpas_s = mpas_model.step(mpas_s, DT)

# Final
ll_spd, ll_me, ll_mea, ll_ke = ll_diagnostics(ll_s)
mp_spd, mp_me, mp_mea, mp_ke = mpas_diagnostics(mpas_s)
day = N_STEPS * DT / 86400
print(f"{N_STEPS:6d} {day:7.2f} | {ll_spd:10.4e} {ll_me:10.4e} {ll_mea:11.4e} {ll_ke:10.4e}"
      f" | {mp_spd:10.4e} {mp_me:10.4e} {mp_mea:11.4e} {mp_ke:10.4e}")
