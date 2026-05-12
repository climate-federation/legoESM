"""Single-level global barotropic wind: isolate cos(angleEdge) effect.

With 1 level, wind and bottom drag act on the same layer.
Steady-state: tau_x/(rho_0*H) = r*u  →  u = tau_x/(rho_0*H*r)
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig, PrescribedForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig, LinearDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.core.field import Field

# ── Parameters ──────────────────────────────────────────────────────
NLEV = 1
H_MAX = 5500.0
DT = 300.0
DAYS = 30.0
N_STEPS = int(DAYS * 86400 / DT)
T_UNIFORM = 10.0
S_UNIFORM = 35.0
RHO0 = constants.rho_ocean
R_DRAG = 1e-4
TAU_MAX = 0.1

z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
dz0 = float(z_coord.dz_ref[0])
print(f"Single level: dz = {dz0:.1f} m (= H_max = {H_MAX} m)")
print(f"dt={DT}s, {N_STEPS} steps, {DAYS} days")
print(f"Analytical steady state: u = tau_max/(rho_0*H*r) = {TAU_MAX/(RHO0*H_MAX*R_DRAG):.6f} m/s")
print()

physics = OceanPhysicsConfig(
    surface_forcing=SurfaceForcingConfig(
        scheme="prescribed",
        prescribed=PrescribedForcingConfig(wind_profile="global_wind", tau_max=TAU_MAX),
    ),
    vertical_mixing=VerticalMixingConfig(scheme="none"),
    lateral_mixing=LateralMixingConfig(scheme="none"),
    bottom_drag=BottomDragConfig(scheme="linear", linear=LinearDragConfig(r=R_DRAG)),
    convection=OceanConvectionConfig(scheme="none"),
    shortwave_penetration=None,
)

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

# ── Setup latlon C-grid ─────────────────────────────────────────────
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks

ll_grid = create_latlon_grid(36, 72)
ll_config = LatLonCGridOceanConfig(n_barotropic_substeps=30, physics=physics, A_h=5e5)
ll_model = LatLonCGridOceanModel(ll_grid, z_coord, ll_config)
ll_state = rest_state_latlon_cgrid_ocean(
    ll_grid, z_coord, T_surface=T_UNIFORM, T_deep=T_UNIFORM, S_uniform=S_UNIFORM)
lon_2d = np.asarray(ll_grid.lon) * 180 / np.pi
lat_1d = np.asarray(ll_grid.lat) * 180 / np.pi
lon_g, lat_g = np.meshgrid(lon_2d, lat_1d)
mask_ll = _create_simplified_continent_mask(lon_g, lat_g)
mask_typed = mask_ll.astype(ll_state.eta.data.dtype)
u_mask_new, v_mask_new = compute_face_masks(mask_typed)
ll_state = ll_state._replace(
    land_mask=Field(data=mask_typed),
    u_mask=Field(data=u_mask_new), v_mask=Field(data=v_mask_new))

# ── Setup MPAS ──────────────────────────────────────────────────────
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.init_mpas import rest_state_mpas_ocean

mpas_mesh = create_voronoi_mesh(3)
mpas_config = MPASOceanConfig(n_barotropic_substeps=30, physics=physics, A_h=5e5)
mpas_model = MPASOceanModel(mpas_mesh, z_coord, mpas_config)
mpas_state = rest_state_mpas_ocean(
    mpas_mesh, z_coord, T_surface=T_UNIFORM, T_deep=T_UNIFORM, S_uniform=S_UNIFORM)
lon_deg_c = np.asarray(mpas_mesh.lonCell) * 180 / np.pi
lat_deg_c = np.asarray(mpas_mesh.latCell) * 180 / np.pi
mpas_mask = _create_simplified_continent_mask(lon_deg_c, lat_deg_c)
mpas_state = mpas_state._replace(
    land_mask=Field(data=mpas_mask.astype(mpas_state.eta.data.dtype)))

# ── Time loop ───────────────────────────────────────────────────────
print(f"{'Step':>6} {'Day':>7} | {'LL max_spd':>10} {'LL max_eta':>10} {'LL mean_eta':>11}"
      f" | {'MP max_spd':>10} {'MP max_eta':>10} {'MP mean_eta':>11}")
print("-" * 100)

diag_every = max(1, N_STEPS // 30)

for step in range(N_STEPS + 1):
    if step % diag_every == 0 or step == N_STEPS:
        ll_eta = np.asarray(ll_state.eta.data)
        ll_u = np.asarray(ll_state.u.data)
        ll_v = np.asarray(ll_state.v.data)
        ll_ocean = np.asarray(mask_ll) > 0.5
        ll_spd = max(float(np.max(np.abs(ll_u))), float(np.max(np.abs(ll_v))))
        ll_me = float(np.max(np.abs(ll_eta[ll_ocean]))) if np.any(ll_ocean) else 0
        ll_mea = float(np.mean(ll_eta[ll_ocean])) if np.any(ll_ocean) else 0

        mp_eta = np.asarray(mpas_state.eta.data)
        mp_u = np.asarray(mpas_state.u.data)
        mp_ocean = np.asarray(mpas_mask) > 0.5
        mp_spd = float(np.max(np.abs(mp_u)))
        mp_me = float(np.max(np.abs(mp_eta[mp_ocean]))) if np.any(mp_ocean) else 0
        mp_mea = float(np.mean(mp_eta[mp_ocean])) if np.any(mp_ocean) else 0

        day = step * DT / 86400
        print(f"{step:6d} {day:7.2f} | {ll_spd:10.4e} {ll_me:10.4e} {ll_mea:11.4e}"
              f" | {mp_spd:10.4e} {mp_me:10.4e} {mp_mea:11.4e}")

    if step < N_STEPS:
        ll_state = ll_model.step(ll_state, DT)
        mpas_state = mpas_model.step(mpas_state, DT)

# ── Final analysis ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)
print(f"Analytical steady-state |u| = tau_max/(rho*H*r) = {TAU_MAX/(RHO0*H_MAX*R_DRAG):.6f} m/s")
print(f"Latlon max speed: {ll_spd:.6f} m/s")
print(f"MPAS max speed:   {mp_spd:.6f} m/s")
print(f"Ratio LL/MP: {ll_spd/max(mp_spd, 1e-20):.3f}")
print(f"If cos^2 effect: expected ratio ~ {1.0/0.5:.1f}")
