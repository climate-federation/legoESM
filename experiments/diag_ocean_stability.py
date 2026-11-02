"""Stability diagnostic for the ocean-only 3D run.

Runs the same LatLonCGridOceanModel setup as ``ocean_only_3d.py`` but
monitors the quantities that reveal a numerical blow-up:
  - max horizontal current speed and WHERE it occurs
  - max |eta| (free-surface height)
  - min layer thickness (z-star cells can collapse → divide-by-thin-layer)
  - min/max T and S

Prints every ``PRINT_EVERY_DAYS`` days and STOPS as soon as any field
becomes non-finite or S leaves a sane range, reporting the location.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python experiments/diag_ocean_stability.py
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

N_LAT, N_LON = 36, 72
N_LEV = 20
H_MAX = 4000.0
DT    = 1800.0
DAYS  = 365 * 3
PRINT_EVERY_DAYS = 20.0
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

from legoesm.core.field import Field
T_woa, S_woa = init_ocean_from_woa(grid, z_coord, T_path=None, S_path=None)
state = state._replace(
    T=Field(jnp.array(T_woa), name="T", dims=state.T.dims, units=state.T.units),
    S=Field(jnp.array(S_woa), name="S", dims=state.S.dims, units=state.S.units),
)

import dataclasses as _dc
go_cfg_strong_restore = _dc.replace(go_cfg, tau_T_days=5.0)
physics = go_forcings("latlon", grid, go_cfg_strong_restore)
eos_cfg = go_eos(go_cfg)

model_cfg = LatLonCGridOceanConfig(
    n_barotropic_substeps=30,
    physics=physics,
    A_h=2e5, K_h=1e3, A_v=1e-3, K_v=1e-5,
    bottom_drag_r=1.1e-3,
    eos="linear", eos_linear=eos_cfg,
    tracer_advection=os.environ.get("TRACER_ADV", "dst3"),
)
model = LatLonCGridOceanModel(grid, z_coord, model_cfg)
print(f"tracer_advection = {model_cfg.tracer_advection}")

lat2d = np.degrees(np.asarray(grid.lat2d))
lon2d = np.degrees(np.asarray(grid.lon2d))

n_steps = int(DAYS * 86400.0 / DT)
print_every = max(1, int(PRINT_EVERY_DAYS * 86400.0 / DT))
print(f"Diagnostic run: {n_steps} steps, dt={DT}s, print every {PRINT_EVERY_DAYS} d")

# Ocean-cell mask and reference cell volumes for salt-conservation check
om3 = np.asarray(land_mask)[:, :, None]                 # (nlat,nlon,1)
dz_ref = np.asarray(z_coord.dz_ref)                     # (nlev,)
area2d = np.asarray(grid.area) if hasattr(grid, "area") else np.ones((N_LAT, N_LON))
if area2d.ndim == 1:
    area2d = np.broadcast_to(area2d[:, None], (N_LAT, N_LON))
cell_vol = (area2d[:, :, None] * dz_ref[None, None, :]) * (om3 > 0.5)  # (nlat,nlon,nlev)
tot_vol = cell_vol.sum()

def total_salt(Sarr):
    return float(np.nansum(np.where(om3 > 0.5, Sarr, 0.0) * cell_vol))

S0 = np.asarray(state.S.data)
salt0 = total_salt(S0)

print(f"{'day':>7} {'maxspd':>8} {'@lat':>6} {'@lon':>6} {'@lev':>4} "
      f"{'max|eta|':>9} {'min_h':>7} {'Tmin':>6} {'Tmax':>6} {'Smin':>8} {'Smax':>8} "
      f"{'saltRel':>10}")

for k in range(n_steps):
    state = model.step(state, dt=DT)

    if (k + 1) % print_every == 0 or k == 0:
        t_days = (k + 1) * DT / 86400.0
        u = np.asarray(state.u.data)   # (nlat, nlon+1, nlev) E-W faces
        v = np.asarray(state.v.data)   # (nlat+1, nlon, nlev) N-S faces
        T = np.asarray(state.T.data)
        S = np.asarray(state.S.data)
        eta = np.asarray(state.eta.data)
        h_k = np.asarray(compute_layer_thickness(
            jnp.array(eta), jnp.array(np.asarray(state.H_bathy.data)), z_coord))

        # Average C-grid face velocities to cell centres
        u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])   # (nlat, nlon, nlev)
        v_c = 0.5 * (v[:-1, :, :] + v[1:, :, :])   # (nlat, nlon, nlev)
        spd = np.sqrt(u_c**2 + v_c**2)
        # mask land
        om = np.asarray(land_mask)[:, :, None]
        spd_m = np.where(om > 0.5, spd, np.nan)
        imax = np.unravel_index(np.nanargmax(spd_m), spd_m.shape)
        maxspd = spd_m[imax]

        print(f"{t_days:7.1f} {maxspd:8.3f} "
              f"{lat2d[imax[0], imax[1]]:6.1f} {lon2d[imax[0], imax[1]]:6.1f} "
              f"{imax[2]:4d} "
              f"{np.nanmax(np.abs(eta)):9.2f} {np.nanmin(h_k):7.2f} "
              f"{np.nanmin(T):6.1f} {np.nanmax(T):6.1f} "
              f"{np.nanmin(S):8.1f} {np.nanmax(S):8.1f} "
              f"{total_salt(S) / salt0:10.6f}")

        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(T))
                and np.all(np.isfinite(S)) and np.all(np.isfinite(eta))):
            print(f"*** NON-FINITE detected at day {t_days:.1f} — stopping.")
            break
        if maxspd > 20.0 or np.nanmin(S) < 0.0:
            print(f"*** BLOW-UP (maxspd={maxspd:.1f}, Smin={np.nanmin(S):.1f}) "
                  f"at day {t_days:.1f} — location lat={lat2d[imax[0],imax[1]]:.1f} "
                  f"lon={lon2d[imax[0],imax[1]]:.1f} lev={imax[2]} — stopping.")
            break

print("Diagnostic complete.")
