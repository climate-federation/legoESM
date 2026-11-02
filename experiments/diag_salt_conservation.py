"""Salt-conservation diagnostic — produces figures showing the S blow-up.

Runs the ocean-only setup and, every few days, records:
  - total volume-integrated salt inventory  Σ (S · area · h_k)
  - global Smin / Smax
  - total heat inventory (control: T is restored, should be bounded)

Then saves:
  experiments/diag_salt_conservation.png
      (a) total salt vs time (should be flat; if it drifts → spurious source)
      (b) Smin / Smax envelope vs time
  experiments/diag_salt_map.png
      surface-salinity map at the final recorded step, showing WHERE the
      spurious salt accumulates.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python experiments/diag_salt_conservation.py
    TRACER_ADV=tvd JAX_ENABLE_X64=1 .venv/bin/python experiments/diag_salt_conservation.py
"""

from __future__ import annotations

import os
import sys

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
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean,
    replace_land_mask,
    idealized_bathymetry_latlon_cgrid,
)
from legoesm.ocean.init_woa import init_ocean_from_woa
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_forcings as go_forcings,
    create_eos_config as go_eos,
)
from legoesm.ocean.experiments.global_barotropic_wind import (
    create_simplified_continent_mask,
)
from legoesm.core.field import Field

N_LAT, N_LON = 36, 72
N_LEV = 20
H_MAX = 4000.0
DT    = 1800.0
DAYS  = 365 * 2          # 2 years is enough to see the drift
SAMPLE_EVERY_DAYS = 5.0
ADV = os.environ.get("TRACER_ADV", "dst3")

grid   = create_latlon_grid(N_LAT, N_LON)
z_coord = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX,
                               dz_surface=10.0, dz_deep=400.0)

go_cfg = GlobalOverturningConfig(H_max=H_MAX, n_levels=N_LEV)
H_bathy, _land_flat = idealized_bathymetry_latlon_cgrid(grid, H_max=H_MAX,
                                                        land_lat_threshold=80.0)
lon_deg = np.degrees(np.asarray(grid.lon2d))
lat_deg = np.degrees(np.asarray(grid.lat2d))
cont_mask = create_simplified_continent_mask(lon_deg, lat_deg, go_cfg)
land_mask = jnp.array(cont_mask.astype(np.float64))

state = rest_state_latlon_cgrid_ocean(
    grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0)
state = replace_land_mask(state, land_mask)

T_woa, S_woa = init_ocean_from_woa(grid, z_coord, T_path=None, S_path=None)
state = state._replace(
    T=Field(jnp.array(T_woa), name="T", dims=state.T.dims, units=state.T.units),
    S=Field(jnp.array(S_woa), name="S", dims=state.S.dims, units=state.S.units),
)

import dataclasses as _dc
go_cfg_strong_restore = _dc.replace(go_cfg, tau_T_days=5.0)
physics = go_forcings("latlon", grid, go_cfg_strong_restore)
eos_cfg = go_eos(go_cfg)

# --- FIX: enable weak sea-surface-salinity restoring ---
# The default config disables SSS restoring (tau_S=1e30) and salinity is a
# passive tracer (beta_S=0 in the linear EOS), so grid-scale advection noise
# in S has no damping and grows without bound.  A physical SSS restoring of
# ~60 days keeps S bounded near the target, exactly as in standard OGCMs.
TAU_S_DAYS = float(os.environ.get("TAU_S_DAYS", "60.0"))
if TAU_S_DAYS > 0:
    _r = physics.surface_forcing.restoring._replace(tau_S=TAU_S_DAYS * 86400.0)
    _sf = physics.surface_forcing._replace(restoring=_r)
    physics = physics._replace(surface_forcing=_sf)
    print(f"SSS restoring enabled: tau_S = {TAU_S_DAYS} days")
else:
    print("SSS restoring DISABLED (tau_S = 1e30)")

model_cfg = LatLonCGridOceanConfig(
    n_barotropic_substeps=30,
    physics=physics,
    A_h=2e5, K_h=float(os.environ.get("K_H", "1e3")),
    A_v=1e-3, K_v=1e-5,
    bottom_drag_r=1.1e-3,
    eos="linear", eos_linear=eos_cfg,
    tracer_advection=ADV,
)
model = LatLonCGridOceanModel(grid, z_coord, model_cfg)
print(f"tracer_advection = {ADV}, K_h = {model_cfg.K_h}")

# Cell area (nlat, nlon), broadcast over levels
area2d = np.asarray(grid.area)
if area2d.ndim == 1:
    area2d = area2d.reshape(N_LAT, N_LON)
area3d = area2d[:, :, None]
om3d = np.asarray(land_mask)[:, :, None]

def salt_inventory(S, h_k):
    """Volume-integrated salt Σ S·area·h over ocean cells."""
    return float(np.nansum(np.where(om3d > 0.5, S * area3d * h_k, 0.0)))

def heat_inventory(T, h_k):
    return float(np.nansum(np.where(om3d > 0.5, T * area3d * h_k, 0.0)))

n_steps = int(DAYS * 86400.0 / DT)
sample_every = max(1, int(SAMPLE_EVERY_DAYS * 86400.0 / DT))

days_hist, salt_hist, heat_hist = [], [], []
smin_hist, smax_hist = [], []
last_S_surf = None

# initial inventories
h0 = np.asarray(compute_layer_thickness(
    state.eta.data, state.H_bathy.data, z_coord))
salt0 = salt_inventory(np.asarray(state.S.data), h0)
heat0 = heat_inventory(np.asarray(state.T.data), h0)
print(f"Initial salt inventory = {salt0:.6e}")

for k in range(n_steps):
    state = model.step(state, dt=DT)

    if (k + 1) % sample_every == 0:
        t_days = (k + 1) * DT / 86400.0
        S = np.asarray(state.S.data)
        T = np.asarray(state.T.data)
        h_k = np.asarray(compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord))
        salt = salt_inventory(S, h_k)
        heat = heat_inventory(T, h_k)
        days_hist.append(t_days)
        salt_hist.append(salt)
        heat_hist.append(heat)
        smin_hist.append(float(np.nanmin(np.where(om3d > 0.5, S, np.nan))))
        smax_hist.append(float(np.nanmax(np.where(om3d > 0.5, S, np.nan))))
        last_S_surf = np.where(om3d[:, :, 0] > 0.5, S[:, :, 0], np.nan)

        if not np.all(np.isfinite(S)):
            print(f"Non-finite S at day {t_days:.1f} — stopping sampling.")
            break

days_hist = np.array(days_hist)
salt_hist = np.array(salt_hist)
heat_hist = np.array(heat_hist)
smin_hist = np.array(smin_hist)
smax_hist = np.array(smax_hist)

salt_drift_pct = 100.0 * (salt_hist - salt0) / abs(salt0)
heat_drift_pct = 100.0 * (heat_hist - heat0) / abs(heat0)

# ------------------------------------------------------------------
# Figure 1: conservation time series
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax1.plot(days_hist, salt_drift_pct, "-o", ms=3, color="crimson",
         label="salt inventory drift")
ax1.plot(days_hist, heat_drift_pct, "-s", ms=3, color="steelblue",
         label="heat inventory drift (T is restored → control)")
ax1.axhline(0.0, color="k", lw=0.8, ls="--")
ax1.set_ylabel("Inventory drift from t=0  [%]")
ax1.set_title(f"Volume-integrated tracer conservation  (advection = {ADV})\n"
              "A conservative scheme should keep the salt line flat at 0%",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

ax2.plot(days_hist, smax_hist, "-o", ms=3, color="darkred",  label="global S max")
ax2.plot(days_hist, smin_hist, "-o", ms=3, color="navy",     label="global S min")
ax2.axhline(35.0, color="grey", lw=1.0, ls=":", label="initial max ≈ 35.5")
ax2.axhline(0.0,  color="grey", lw=0.8, ls="--")
ax2.set_xlabel("Day")
ax2.set_ylabel("Salinity  [psu]")
ax2.set_title("Global salinity extrema — monotone advection must not exceed the "
              "initial max", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig.tight_layout()
out1 = f"experiments/diag_salt_conservation_{ADV}.png"
fig.savefig(out1, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")

# ------------------------------------------------------------------
# Figure 2: surface-salinity map at final recorded step
# ------------------------------------------------------------------
proj = ccrs.PlateCarree()
fig2, ax = plt.subplots(figsize=(11, 5), subplot_kw=dict(projection=proj))
data_rolled = np.roll(last_S_surf, N_LON // 2, axis=1)
im = ax.imshow(data_rolled, origin="lower", extent=[-180, 180, -90, 90],
               cmap=plt.get_cmap("RdYlBu_r").with_extremes(bad="#c8c8c8"),
               transform=proj, interpolation="nearest")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
ax.set_extent([-180, 180, -90, 90], crs=proj)
ax.gridlines(linewidth=0.3, color="grey", alpha=0.5,
             xlocs=range(-180, 181, 60), ylocs=range(-60, 61, 30))
ax.set_title(f"Surface salinity at day {days_hist[-1]:.0f}  (advection = {ADV})\n"
             "grid-scale checkerboard mode grows in the passive, undamped tracer",
             fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.85, label="S [psu]")
out2 = f"experiments/diag_salt_map_{ADV}.png"
fig2.savefig(out2, dpi=140, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

print(f"\nSummary ({ADV}):")
print(f"  salt drift at day {days_hist[-1]:.0f}: {salt_drift_pct[-1]:+.3f} %")
print(f"  heat drift at day {days_hist[-1]:.0f}: {heat_drift_pct[-1]:+.3f} %")
print(f"  S range: [{smin_hist[-1]:.1f}, {smax_hist[-1]:.1f}] psu "
      f"(initial max ≈ 35.5)")
print("Done.")
