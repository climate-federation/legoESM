"""MOC diagnostic — 1-year ocean-only run, plots the overturning streamfunction.

Runs the same fixed physics as npzd_3d_ocean.py (SSS restoring + K_h=2e4)
but only 1 year, accumulating the time-mean meridional velocity to compute:

  Ψ(lat, z) = ∫_{x_W}^{x_E} ∫_z^0 <v> dz' dx   [Sv]

Produces:
  experiments/moc_streamfunction.png  — global MOC (lat vs depth)
  experiments/moc_barotropic.png      — barotropic stream function (lat vs lon)
  experiments/moc_zonal_T.png         — zonal-mean T section for context

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python experiments/diag_moc.py
"""

from __future__ import annotations
import gc, os, sys, time

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

from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean, replace_land_mask,
    idealized_bathymetry_latlon_cgrid,
)
from legoesm.ocean.init_woa import init_ocean_from_woa
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_forcings as go_forcings, create_eos_config as go_eos,
)
from legoesm.ocean.experiments.global_barotropic_wind import create_simplified_continent_mask
from legoesm.core.field import Field
import dataclasses as _dc

# ── grid ──────────────────────────────────────────────────────────────────────
N_LAT, N_LON, N_LEV, H_MAX = 36, 72, 20, 4000.0
DT   = 1800.0
DAYS = 365        # 1-year run

grid    = create_latlon_grid(N_LAT, N_LON)
z_coord = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX, dz_surface=10.0, dz_deep=400.0)

# ── land mask ─────────────────────────────────────────────────────────────────
go_cfg  = GlobalOverturningConfig(H_max=H_MAX, n_levels=N_LEV)
H_bathy, _ = idealized_bathymetry_latlon_cgrid(grid, H_max=H_MAX, land_lat_threshold=80.0)
lon_d = np.degrees(np.asarray(grid.lon2d))
lat_d = np.degrees(np.asarray(grid.lat2d))
land_mask = jnp.array(create_simplified_continent_mask(lon_d, lat_d, go_cfg).astype(np.float64))

# ── initial state ─────────────────────────────────────────────────────────────
state = rest_state_latlon_cgrid_ocean(grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
state = replace_land_mask(state, land_mask)
T_woa, S_woa = init_ocean_from_woa(grid, z_coord, T_path=None, S_path=None)
state = state._replace(
    T=Field(jnp.array(T_woa), name="T", dims=state.T.dims, units=state.T.units),
    S=Field(jnp.array(S_woa), name="S", dims=state.S.dims, units=state.S.units),
)

# ── physics (same fixes as npzd_3d_ocean.py) ─────────────────────────────────
go_cfg_r = _dc.replace(go_cfg, tau_T_days=5.0)
physics  = go_forcings("latlon", grid, go_cfg_r)
eos_cfg  = go_eos(go_cfg)
_r  = physics.surface_forcing.restoring._replace(tau_S=60.0 * 86400.0)
_sf = physics.surface_forcing._replace(restoring=_r)
physics = physics._replace(surface_forcing=_sf)

model_cfg = LatLonCGridOceanConfig(
    n_barotropic_substeps=30, physics=physics,
    A_h=2e5, K_h=2e4, A_v=1e-3, K_v=1e-5,
    bottom_drag_r=1.1e-3, eos="linear", eos_linear=eos_cfg,
    tracer_advection="dst3",
)
model = LatLonCGridOceanModel(grid, z_coord, model_cfg)
print("Ocean model built.")

# ── time loop ─────────────────────────────────────────────────────────────────
n_steps    = int(DAYS * 86400.0 / DT)
save_every = max(1, int(2 * 86400.0 / DT))   # every 2 days

# V is on N-S faces: (nlat+1, nlon, nlev).  We accumulate the interior rows
# (1:-1) to get cell-centre v: shape (nlat, nlon, nlev)
sum_V = np.zeros((N_LAT, N_LON, N_LEV))
sum_T = np.zeros((N_LAT, N_LON, N_LEV))
sum_h = np.zeros((N_LAT, N_LON, N_LEV))
n_saved = 0

print(f"Running {DAYS}-day ocean-only run ({n_steps} steps)…")
t0 = time.time()
for k in range(n_steps):
    state = model.step(state, dt=DT)
    if (k + 1) % save_every == 0:
        v_face = np.asarray(state.v.data)          # (nlat+1, nlon, nlev)
        v_c    = 0.5 * (v_face[:-1] + v_face[1:]) # (nlat, nlon, nlev) cell-centre
        eta    = state.eta.data
        Hb     = state.H_bathy.data
        h_k    = np.asarray(compute_layer_thickness(eta, Hb, z_coord))
        sum_V += v_c
        sum_T += np.asarray(state.T.data)
        sum_h += h_k
        n_saved += 1
    if (k + 1) % max(1, n_steps // 10) == 0:
        pct = 100.0 * (k + 1) / n_steps
        print(f"  {pct:5.1f}%  elapsed {time.time()-t0:.0f}s")
    if (k + 1) % max(1, n_steps // 5) == 0:
        jax.block_until_ready(state.T.data)
        jax.clear_caches(); gc.collect()

print(f"Done in {time.time()-t0:.1f}s.  Accumulated {n_saved} snapshots.")

# ── diagnostics ───────────────────────────────────────────────────────────────
V_mean = sum_V / n_saved          # (nlat, nlon, nlev) m/s
T_mean = sum_T / n_saved
h_mean = sum_h / n_saved          # (nlat, nlon, nlev) layer thickness [m]

# Ocean mask
om2d = np.asarray(land_mask)      # (nlat, nlon)
om3d = om2d[:, :, None]

# Cell width in x direction for the zonal integral: dx (nlat, nlon)
dx2d = np.asarray(grid.dx)        # (nlat, nlon) metres
lat_deg = np.degrees(np.asarray(grid.lat))   # (nlat,)
depths  = np.abs(np.array(z_coord.z_full_ref))  # (nlev,) positive downward

# ── MOC streamfunction ────────────────────────────────────────────────────────
# Ψ(lat, z) = ∫_x ∫_z^0 v dz' dx   [m³/s] → convert to Sv (1 Sv = 1e6 m³/s)
# Discretely: sum v * dx * h_k over x, then cumulative sum from surface downward.
# v_transport[lat, lev] = Σ_lon  v * dx * h_k  (land=0)
v_masked = V_mean * om3d          # zero out land
v_transport = np.sum(v_masked * dx2d[:, :, None] * h_mean, axis=1)  # (nlat, nlev) m²/s

# Cumulative sum from top (lev=0) downward → streamfunction at each level
psi = np.cumsum(v_transport, axis=1)  # (nlat, nlev) m³/s
psi_Sv = psi / 1e6                    # Sv

# ── barotropic stream function ────────────────────────────────────────────────
# Ψ_BT(lat) = ∫_x ∫_{-H}^0 v dz dx  = Σ_lev v_transport summed over all levels
# then cumulative sum in y → stream function on the (lat, lon) plane
# Simpler: column-integrated v times dx, then running lon-integral from west.
psi_bt_col = np.sum(v_masked * h_mean, axis=2)   # (nlat, nlon)  m/s·m = m²/s
# Zonal cumulative sum (east-to-west gives stream-function in x-y plane)
psi_bt = np.cumsum(psi_bt_col * dx2d, axis=1) / 1e6   # Sv

# ── zonal-mean T ─────────────────────────────────────────────────────────────
T_zonal = np.nanmean(np.where(om3d > 0.5, T_mean, np.nan), axis=1)  # (nlat, nlev)

# ── plot MOC streamfunction ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
vmax = max(abs(np.nanpercentile(psi_Sv, 2)), abs(np.nanpercentile(psi_Sv, 98)), 1.0)
cf = ax.contourf(lat_deg, depths, psi_Sv.T,
                 levels=np.linspace(-vmax, vmax, 21), cmap="RdBu_r", extend="both")
ax.contour(lat_deg, depths, psi_Sv.T,
           levels=[0.0], colors="k", linewidths=1.0, linestyles="--")
ax.set_ylim(depths.max(), 0)
ax.set_xlabel("Latitude (°)", fontsize=11)
ax.set_ylabel("Depth (m)", fontsize=11)
ax.set_title("Global Meridional Overturning Circulation\n"
             "legoESM 5° idealized ocean — year-1 time mean  [Sv]",
             fontsize=12, fontweight="bold")
plt.colorbar(cf, ax=ax, label="Ψ [Sv]", shrink=0.85)
fig.tight_layout()
out = "experiments/moc_streamfunction.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out}")

# ── plot zonal-mean T section ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
cf2 = ax.contourf(lat_deg, depths, T_zonal.T, levels=20, cmap="RdYlBu_r")
ax.contour(lat_deg, depths, T_zonal.T, levels=[2, 5, 10, 15, 20], colors="k",
           linewidths=0.5, alpha=0.7)
ax.set_ylim(depths.max(), 0)
ax.set_xlabel("Latitude (°)", fontsize=11)
ax.set_ylabel("Depth (m)", fontsize=11)
ax.set_title("Zonal-mean Temperature (°C) — year-1 time mean",
             fontsize=12, fontweight="bold")
plt.colorbar(cf2, ax=ax, label="T [°C]", shrink=0.85)
fig.tight_layout()
out2 = "experiments/moc_zonal_T.png"
fig.savefig(out2, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out2}")

# ── plot barotropic stream function ──────────────────────────────────────────
import cartopy.crs as ccrs, cartopy.feature as cfeature
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(11, 5), subplot_kw=dict(projection=proj))
data_r = np.roll(psi_bt, N_LON // 2, axis=1)
vmax_bt = max(abs(np.nanpercentile(data_r, 2)), abs(np.nanpercentile(data_r, 98)), 1.0)
im = ax.imshow(data_r, origin="lower", extent=[-180, 180, -90, 90],
               cmap=plt.get_cmap("RdBu_r").with_extremes(bad="#c8c8c8"),
               vmin=-vmax_bt, vmax=vmax_bt, transform=proj, interpolation="nearest")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
ax.set_extent([-180, 180, -90, 90], crs=proj)
ax.gridlines(linewidth=0.3, color="grey", alpha=0.5,
             xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30))
ax.set_title("Barotropic stream function [Sv] — year-1 time mean\n"
             "(gyres visible; positive = clockwise)",
             fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.85, label="Ψ_BT [Sv]")
out3 = "experiments/moc_barotropic.png"
fig.savefig(out3, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {out3}")
print("All done.")
