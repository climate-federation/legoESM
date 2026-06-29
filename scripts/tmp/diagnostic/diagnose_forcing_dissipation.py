"""Diagnose forcing and dissipation differences between latlon C-grid and MPAS.

Key questions:
1. Does the cos(angleEdge) projection on MPAS reduce total wind power input?
2. Is bottom drag effective on both grids?
3. Is lateral viscosity (A_h) effective on both grids?
4. What is the total KE budget (wind input vs dissipation)?
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.vertical import create_ocean_z_star

NLEV = 10
H_MAX = 5500.0
DT = 300.0
T_UNIFORM = 10.0
S_UNIFORM = 35.0

z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)

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

# ── Physics config ──────────────────────────────────────────────────
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig, PrescribedForcingConfig
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig, LinearDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig

physics = OceanPhysicsConfig(
    surface_forcing=SurfaceForcingConfig(
        scheme="prescribed",
        prescribed=PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1),
    ),
    vertical_mixing=VerticalMixingConfig(scheme="none"),
    lateral_mixing=LateralMixingConfig(scheme="none"),
    bottom_drag=BottomDragConfig(scheme="linear", linear=LinearDragConfig(r=1e-4)),
    convection=OceanConvectionConfig(scheme="none"),
    shortwave_penetration=None,
)

# ═══════════════════════════════════════════════════════════════════
# 1. MPAS: Quantify cos(angleEdge) projection effect
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. MPAS EDGE ANGLE ANALYSIS")
print("=" * 70)

from legoesm.grids.voronoi import create_voronoi_mesh
mpas_mesh = create_voronoi_mesh(3)

angles = np.asarray(mpas_mesh.angleEdge)
cos_angles = np.cos(angles)
sin_angles = np.sin(angles)

print(f"nEdges = {mpas_mesh.nEdges}")
print(f"angleEdge stats:")
print(f"  mean = {np.mean(angles):.4f} rad ({np.degrees(np.mean(angles)):.1f} deg)")
print(f"  std  = {np.std(angles):.4f} rad ({np.degrees(np.std(angles)):.1f} deg)")
print(f"cos(angleEdge) stats:")
print(f"  mean = {np.mean(cos_angles):.4f}")
print(f"  mean|cos| = {np.mean(np.abs(cos_angles)):.4f}")
print(f"  <cos^2> = {np.mean(cos_angles**2):.4f}")
print(f"sin(angleEdge) stats:")
print(f"  mean = {np.mean(sin_angles):.4f}")
print(f"  <sin^2> = {np.mean(sin_angles**2):.4f}")
print()

# For a purely zonal wind (tau_y=0), the edge-normal force is tau_x * cos(angleEdge).
# The total wind power input ∝ sum(tau_n * u_n * dvEdge * dz_0).
# If edges are uniformly oriented, <cos^2> = 0.5, meaning only half the
# KE input compared to a grid where all faces are zonal.
# On a C-grid latlon, ALL u-faces are zonal (cos=1) and v-faces get zero tau.
# Effective forcing ratio: MPAS gets <cos^2> of the latlon forcing.
print(f">>> Effective MPAS wind power fraction vs latlon: <cos^2(angleEdge)> = {np.mean(cos_angles**2):.4f}")
print(f"    (1.0 means same total input, 0.5 means half)")
print()

# Check if edge mask affects this
c1 = np.asarray(mpas_mesh.cellsOnEdge[0])
c2 = np.asarray(mpas_mesh.cellsOnEdge[1])
lat_deg_c = np.asarray(mpas_mesh.latCell) * 180 / np.pi
mask_cells = np.asarray(_create_simplified_continent_mask(
    np.asarray(mpas_mesh.lonCell) * 180 / np.pi, lat_deg_c))
edge_mask = mask_cells[c1] * mask_cells[c2]
ocean_edges = edge_mask > 0.5
print(f"Ocean edges: {np.sum(ocean_edges)} / {mpas_mesh.nEdges}")
print(f"<cos^2> on ocean edges only: {np.mean(cos_angles[ocean_edges]**2):.4f}")
print()

# ═══════════════════════════════════════════════════════════════════
# 2. Compute total wind power input on each grid
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("2. TOTAL WIND POWER INPUT COMPARISON")
print("=" * 70)

# --- Latlon ---
from legoesm.grids.latlon import create_latlon_grid
ll_grid = create_latlon_grid(36, 72)

lat_rad_ll = np.asarray(ll_grid.grid_lat)  # (n_lat, n_lon) in radians
taper_ll = np.cos(lat_rad_ll) ** 2
tau_x_ll = -0.1 * np.cos(2.0 * lat_rad_ll) * taper_ll

dz0 = float(z_coord.dz_ref[0])
rho0 = constants.rho_ocean
du_dt_wind_ll = tau_x_ll / (rho0 * dz0)  # cell center tendency

# Cell areas
area_ll = np.asarray(ll_grid.area)  # (n_lat, n_lon)
lon_2d = np.asarray(ll_grid.lon) * 180 / np.pi
lat_1d = np.asarray(ll_grid.lat) * 180 / np.pi
lon_g, lat_g = np.meshgrid(lon_2d, lat_1d)
mask_ll_2d = np.asarray(_create_simplified_continent_mask(lon_g, lat_g))

# Wind power = integral(tau_x * du_dt_wind * rho0 * dz0 * area) over ocean
# = integral(tau_x^2 / (rho0 * dz0) * area)
# But more directly: total forcing rate = sum(du_dt_wind * area) over ocean
total_du_dt_area_ll = float(np.sum(np.abs(du_dt_wind_ll) * area_ll * mask_ll_2d))
total_tau_area_ll = float(np.sum(np.abs(tau_x_ll) * area_ll * mask_ll_2d))
# Wind power input rate: P = sum(tau_x * u_surface * area) -- but u=0 initially
# Better metric: total forcing magnitude = sum(|tau_x|/(rho*dz) * area)
print(f"\nLatlon C-grid:")
print(f"  Total ocean area: {float(np.sum(area_ll * mask_ll_2d)):.4e} m^2")
print(f"  max|du_dt_wind|: {float(np.max(np.abs(du_dt_wind_ll))):.6e} m/s^2")
print(f"  mean|du_dt_wind| (ocean): {float(np.mean(np.abs(du_dt_wind_ll[mask_ll_2d > 0.5]))):.6e} m/s^2")
print(f"  sum(|du_dt_wind| * area) (ocean): {total_du_dt_area_ll:.4e} m^3/s^2")

# --- MPAS ---
lat_rad_m = np.asarray(mpas_mesh.latCell)
taper_m = np.cos(lat_rad_m) ** 2
tau_x_m = -0.1 * np.cos(2.0 * lat_rad_m) * taper_m

tau_x_e = 0.5 * (tau_x_m[c1] + tau_x_m[c2])
tau_n = tau_x_e * cos_angles  # + tau_y_e * sin_angles, but tau_y=0

du_dt_wind_m = tau_n / (rho0 * dz0)
dvEdge = np.asarray(mpas_mesh.dvEdge)
# MPAS "area" equivalent for edges: dvEdge * dcEdge, but for total forcing
# the relevant metric is: sum(|du_dt_wind| * dvEdge * dcEdge) or similar
# More directly: area per cell
areaCell = np.asarray(mpas_mesh.areaCell)
total_ocean_area_m = float(np.sum(areaCell * mask_cells))

# For MPAS: du_dt is at edges, but the KE is reconstructed at cells.
# Total edge forcing = sum(|tau_n/(rho*dz)| * dvEdge * dz)
dcEdge = np.asarray(mpas_mesh.dcEdge)
edge_area = dvEdge * dcEdge  # rough edge "area of influence"
total_du_dt_area_m = float(np.sum(np.abs(du_dt_wind_m) * edge_area * edge_mask))

print(f"\nMPAS:")
print(f"  Total ocean area: {total_ocean_area_m:.4e} m^2")
print(f"  max|du_dt_wind|: {float(np.max(np.abs(du_dt_wind_m))):.6e} m/s^2")
print(f"  mean|du_dt_wind| (ocean edges): {float(np.mean(np.abs(du_dt_wind_m[ocean_edges]))):.6e} m/s^2")
print(f"  sum(|du_dt_wind| * edge_area) (ocean): {total_du_dt_area_m:.4e}")

# The key comparison: mean forcing per edge
print(f"\n>>> mean|du_dt_wind| ratio (latlon/MPAS): "
      f"{float(np.mean(np.abs(du_dt_wind_ll[mask_ll_2d > 0.5]))) / float(np.mean(np.abs(du_dt_wind_m[ocean_edges]))):.3f}")
print()

# ═══════════════════════════════════════════════════════════════════
# 3. Bottom drag effectiveness comparison
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("3. BOTTOM DRAG CHECK")
print("=" * 70)
print()

# Run both grids for 5 days, then check bottom drag tendency magnitude
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
from legoesm.core.field import Field

ll_config = LatLonCGridOceanConfig.from_flat(n_barotropic_substeps=30, physics=physics, A_h=5e5)
ll_model = LatLonCGridOceanModel(ll_grid, z_coord, ll_config)
ll_state = rest_state_latlon_cgrid_ocean(
    ll_grid, z_coord, T_water_init_C=T_UNIFORM, T_deep=T_UNIFORM, S_uniform=S_UNIFORM)
mask_typed = mask_ll_2d.astype(ll_state.eta.data.dtype)
u_mask_new, v_mask_new = compute_face_masks(jnp.asarray(mask_typed))
ll_state = ll_state._replace(
    land_mask=Field(data=jnp.asarray(mask_typed)),
    u_mask=Field(data=u_mask_new), v_mask=Field(data=v_mask_new))

from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.init_mpas import rest_state_mpas_ocean

mpas_config = MPASOceanConfig(n_barotropic_substeps=30, physics=physics, A_h=5e5)
mpas_model = MPASOceanModel(mpas_mesh, z_coord, mpas_config)
mpas_state = rest_state_mpas_ocean(
    mpas_mesh, z_coord, T_water_init_C=T_UNIFORM, T_deep=T_UNIFORM, S_uniform=S_UNIFORM)
mpas_state = mpas_state._replace(
    land_mask=Field(data=jnp.asarray(mask_cells.astype(mpas_state.eta.data.dtype))))

# Spin up for 5 days
print("Spinning up both grids for 5 days...")
n5 = int(5 * 86400 / DT)
for i in range(n5):
    ll_state = ll_model.step(ll_state, DT)
    mpas_state = mpas_model.step(mpas_state, DT)
    if (i+1) % 500 == 0:
        print(f"  step {i+1}/{n5}")

print("\nAfter 5 days of spinup:")
ll_u = np.asarray(ll_state.u.data)
ll_v = np.asarray(ll_state.v.data)
mp_u = np.asarray(mpas_state.u.data)

print(f"  LL max|u|: {np.max(np.abs(ll_u)):.4e}, max|v|: {np.max(np.abs(ll_v)):.4e}")
print(f"  MP max|u_edge|: {np.max(np.abs(mp_u)):.4e}")

# Check bottom drag tendency
r_drag = 1e-4
# Latlon: bottom drag applied to cell-center velocity, then interpolated to faces
ll_u_cc = 0.5 * (ll_u[:, :-1, :] + ll_u[:, 1:, :])
ll_v_cc = 0.5 * (ll_v[:-1, :, :] + ll_v[1:, :, :])
ll_drag_du_cc = -r_drag * ll_u_cc[:, :, -1]  # bottom level
ll_drag_dv_cc = -r_drag * ll_v_cc[:, :, -1]

# MPAS: bottom drag applied directly to edge velocity
mp_drag_du = -r_drag * mp_u[:, -1]

print(f"\n  Bottom drag tendency (bottom level):")
print(f"    LL max|drag_du|: {np.max(np.abs(ll_drag_du_cc)):.6e} m/s^2")
print(f"    LL max|drag_dv|: {np.max(np.abs(ll_drag_dv_cc)):.6e} m/s^2")
print(f"    MP max|drag_du|: {np.max(np.abs(mp_drag_du)):.6e} m/s^2")

# Compare bottom-level velocity (where drag acts)
print(f"\n  Bottom level velocity magnitude:")
ll_u_bot = ll_u_cc[:, :, -1]
ll_v_bot = ll_v_cc[:, :, -1]
ll_speed_bot = np.sqrt(ll_u_bot**2 + ll_v_bot**2)
mp_speed_bot = np.abs(mp_u[:, -1])
ocean_ll = mask_ll_2d > 0.5
print(f"    LL mean|speed_bot| (ocean): {np.mean(ll_speed_bot[ocean_ll]):.6e}")
print(f"    LL max|speed_bot|: {np.max(ll_speed_bot):.6e}")
print(f"    MP mean|speed_bot| (ocean edges): {np.mean(mp_speed_bot[ocean_edges]):.6e}")
print(f"    MP max|speed_bot|: {np.max(mp_speed_bot):.6e}")

# Compare surface vs bottom velocity ratio
ll_u_sfc = ll_u_cc[:, :, 0]
ll_v_sfc = ll_v_cc[:, :, 0]
ll_speed_sfc = np.sqrt(ll_u_sfc**2 + ll_v_sfc**2)
mp_speed_sfc = np.abs(mp_u[:, 0])
print(f"\n  Surface velocity magnitude:")
print(f"    LL mean|speed_sfc| (ocean): {np.mean(ll_speed_sfc[ocean_ll]):.6e}")
print(f"    LL max|speed_sfc|: {np.max(ll_speed_sfc):.6e}")
print(f"    MP mean|speed_sfc| (ocean edges): {np.mean(mp_speed_sfc[ocean_edges]):.6e}")
print(f"    MP max|speed_sfc|: {np.max(mp_speed_sfc):.6e}")

print(f"\n  Surface/bottom speed ratio:")
print(f"    LL: {np.max(ll_speed_sfc) / max(np.max(ll_speed_bot), 1e-20):.1f}")
print(f"    MP: {np.max(mp_speed_sfc) / max(np.max(mp_speed_bot), 1e-20):.1f}")

# ═══════════════════════════════════════════════════════════════════
# 4. Vertical structure: is flow barotropic or baroclinic?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. VERTICAL STRUCTURE CHECK")
print("=" * 70)

# Print velocity profile at the location of max surface speed
ll_speed_sfc_2d = ll_speed_sfc
imax_ll = np.unravel_index(np.argmax(ll_speed_sfc_2d), ll_speed_sfc_2d.shape)
print(f"\nLatlon: max speed location = lat_idx={imax_ll[0]}, lon_idx={imax_ll[1]}")
print(f"  lat={lat_1d[imax_ll[0]]:.1f} deg, lon={lon_2d[imax_ll[1]]:.1f} deg")
print(f"  u profile: {ll_u_cc[imax_ll[0], imax_ll[1], :]}")
print(f"  v profile: {ll_v_cc[imax_ll[0], imax_ll[1], :]}")

imax_mp = np.argmax(np.abs(mp_u[:, 0]))
lat_edge_mp = 0.5 * (lat_deg_c[c1[imax_mp]] + lat_deg_c[c2[imax_mp]])
lon_edge_mp = 0.5 * (np.asarray(mpas_mesh.lonCell)[c1[imax_mp]] + np.asarray(mpas_mesh.lonCell)[c2[imax_mp]]) * 180 / np.pi
print(f"\nMPAS: max speed edge = {imax_mp}")
print(f"  lat~{lat_edge_mp:.1f} deg, lon~{lon_edge_mp:.1f} deg")
print(f"  u_edge profile: {mp_u[imax_mp, :]}")

# ═══════════════════════════════════════════════════════════════════
# 5. Wind forcing: where is it actually applied in the vertical?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. WIND FORCING VERTICAL DISTRIBUTION")
print("=" * 70)

# Latlon: wind du_dt applied to all levels or just surface?
from legoesm.ocean.physics.combined import make_ocean_physics
ll_phys_fn = make_ocean_physics(physics)
u_cc_jnp = 0.5 * (ll_state.u.data[:, :-1, :] + ll_state.u.data[:, 1:, :])
v_cc_jnp = 0.5 * (ll_state.v.data[:-1, :, :] + ll_state.v.data[1:, :, :])
cc_state = ll_state._replace(
    u=ll_state.u.replace(data=u_cc_jnp),
    v=ll_state.v.replace(data=v_cc_jnp),
)
ll_phys = ll_phys_fn(cc_state, ll_grid, z_coord)
ll_phys_du = np.asarray(ll_phys.du_dt.data)
print(f"\nLatlon physics du_dt per level (max abs):")
for k in range(NLEV):
    print(f"  level {k}: max|du_dt| = {np.max(np.abs(ll_phys_du[:, :, k])):.6e}")

from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
mp_phys_fn = make_mpas_ocean_physics(physics)
mp_phys = mp_phys_fn(mpas_state, mpas_mesh, z_coord)
mp_phys_du = np.asarray(mp_phys.du_dt.data)
print(f"\nMPAS physics du_dt per level (max abs):")
for k in range(NLEV):
    print(f"  level {k}: max|du_dt| = {np.max(np.abs(mp_phys_du[:, k])):.6e}")
