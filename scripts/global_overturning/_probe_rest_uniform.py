#!/usr/bin/env python
"""Rest-state test with UNIFORM T/S (no stratification).

If this passes but the stratified version fails, the PGF is the culprit.
If this also fails, a more fundamental operator is broken at the fold.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.tripole import create_tripole_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean,
    replace_land_mask,
)
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_eos_config,
)


GRID_FILE = Path("data/grids/eORCA1.2_mesh_mask.nc")


def _reconstruct_bathymetry(grid_file):
    import netCDF4
    ds = netCDF4.Dataset(str(grid_file), "r")
    mbathy = np.asarray(ds.variables["mbathy"][0])
    gdept_1d = np.asarray(ds.variables["gdept_1d"][0])
    tmask_surf = np.asarray(ds.variables["tmask"][0, 0])
    ds.close()
    H_bathy = np.zeros_like(mbathy, dtype=np.float64)
    n_lev = len(gdept_1d)
    for j in range(mbathy.shape[0]):
        for i in range(mbathy.shape[1]):
            k = int(mbathy[j, i])
            if k > 0:
                H_bathy[j, i] = gdept_1d[min(k, n_lev - 1)]
    land_mask = tmask_surf.astype(np.float64)
    return H_bathy, land_mask


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dt = 600.0
    nlev = 20

    print("Loading ORCA1 tripolar grid...")
    geom = create_tripole_grid(str(GRID_FILE))

    dx_floor = 1000.0
    geom = geom._replace(
        dx_T=jnp.maximum(geom.dx_T, dx_floor),
        dy_T=jnp.maximum(geom.dy_T, dx_floor),
        area_T=jnp.maximum(geom.area_T, dx_floor * dx_floor),
        dx_u=jnp.maximum(geom.dx_u, dx_floor),
        dy_u=jnp.maximum(geom.dy_u, dx_floor),
        dx_v=jnp.maximum(geom.dx_v, dx_floor),
        dy_v=jnp.maximum(geom.dy_v, dx_floor),
        area_q=jnp.maximum(geom.area_q, dx_floor * dx_floor),
    )

    H_bathy_raw, land_mask_raw = _reconstruct_bathymetry(GRID_FILE)
    config = GlobalOverturningConfig(H_max=float(H_bathy_raw.max()), n_levels=nlev)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    eos_config = create_eos_config(config)

    # Test A: uniform T/S, no PGF activity expected
    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=None,
        A_h=2e5, A_v=1e-3, K_v=1e-5,
        A_h_lat_scaling=True,
        A_h_floor=2000.0,
        eos="linear", eos_linear=eos_config,
        barotropic_solver="implicit_cn",
    )

    model = LatLonCGridOceanModel(geom, z_coord, ocean_config)

    # UNIFORM T and S everywhere — no horizontal gradients
    state = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        T_surface=15.0, T_deep=15.0,  # uniform
        S_uniform=35.0,
    )
    H_bathy_clamped = np.clip(H_bathy_raw, 0, config.H_max)
    lat_deg_2d = np.asarray(geom.lat_T) * 180 / np.pi
    south_cap_mask = lat_deg_2d < -75.0
    H_bathy_clamped[south_cap_mask] = 0.0
    land_mask_raw[south_cap_mask] = 0.0
    H_shelf_min = 500.0
    too_shallow = (H_bathy_clamped > 0) & (H_bathy_clamped < H_shelf_min)
    H_bathy_clamped[too_shallow] = 0.0
    land_mask_raw[too_shallow] = 0.0
    state = state._replace(H_bathy=Field(jnp.array(H_bathy_clamped)))
    state = replace_land_mask(state, jnp.array(land_mask_raw), grid=geom)

    step_jit = jax.jit(model.step)

    lat_T_deg = np.asarray(geom.lat_T) * 180 / np.pi

    print(f"\n=== Test A: UNIFORM T/S, no forcing, {n_steps} steps ===")
    print(f"{'Step':>5} {'max|u|':>12} {'max|v|':>12} {'max|eta|':>12} "
          f"{'u_argmax(j,i)':>15} {'lat':>7}")
    print("-" * 90)

    for step in range(n_steps):
        state = step_jit(state, dt)
        jax.block_until_ready(state.eta.data)
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        eta = np.asarray(state.eta.data)
        max_u = float(np.nanmax(np.abs(u)))
        max_v = float(np.nanmax(np.abs(v)))
        max_eta = float(np.nanmax(np.abs(eta)))
        u_abs_2d = np.nanmax(np.abs(u), axis=-1)
        u_jj, u_ii = np.unravel_index(np.nanargmax(u_abs_2d), u_abs_2d.shape)
        u_lat = lat_T_deg[min(u_jj, lat_T_deg.shape[0]-1),
                          min(u_ii, lat_T_deg.shape[1]-1)]
        print(f"{step+1:>5} {max_u:>12.4e} {max_v:>12.4e} {max_eta:>12.4e} "
              f"  ({u_jj:>3},{u_ii:>3})     {u_lat:>7.1f}")
        if np.isnan(max_u) or max_u > 1e-6:
            print(f"\n*** NOT at rest! ***")
            break
    else:
        print(f"\n=== PASS (max|u| = {max_u:.4e}) ===")

    # Test B: stratified but FLAT bottom (no bathymetry variation)
    print(f"\n=== Test B: STRATIFIED T, FLAT bottom (H=5902m), no forcing ===")
    state2 = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        T_surface=config.T_surface, T_deep=config.T_surface,
        S_uniform=config.S_uniform,
    )
    # Flat bottom: all ocean cells have H_max depth
    H_flat = np.where(land_mask_raw > 0, config.H_max, 0.0)
    # Reapply masks
    H_flat[south_cap_mask] = 0.0
    state2 = state2._replace(H_bathy=Field(jnp.array(H_flat)))
    state2 = replace_land_mask(state2, jnp.array(land_mask_raw), grid=geom)
    # Add stratification
    z_full = np.asarray(z_coord.z_full_ref)
    decay = np.exp(z_full / config.T_scale_depth)
    T_profile = config.T_deep + (config.T_surface - config.T_deep) * decay
    T_data2 = np.array(state2.T.data)
    for k in range(z_coord.n_levels):
        T_data2[..., k] = T_profile[k]
    state2 = state2._replace(
        T=Field(jnp.array(T_data2), name="T",
                dims=state2.T.dims, units=state2.T.units),
    )

    model2 = LatLonCGridOceanModel(geom, z_coord, ocean_config)
    step_jit2 = jax.jit(model2.step)

    print(f"{'Step':>5} {'max|u|':>12} {'max|v|':>12} {'max|eta|':>12} "
          f"{'u_argmax(j,i)':>15} {'lat':>7}")
    print("-" * 90)

    for step in range(n_steps):
        state2 = step_jit2(state2, dt)
        jax.block_until_ready(state2.eta.data)
        u = np.asarray(state2.u.data)
        v = np.asarray(state2.v.data)
        eta = np.asarray(state2.eta.data)
        max_u = float(np.nanmax(np.abs(u)))
        max_v = float(np.nanmax(np.abs(v)))
        max_eta = float(np.nanmax(np.abs(eta)))
        u_abs_2d = np.nanmax(np.abs(u), axis=-1)
        u_jj, u_ii = np.unravel_index(np.nanargmax(u_abs_2d), u_abs_2d.shape)
        u_lat = lat_T_deg[min(u_jj, lat_T_deg.shape[0]-1),
                          min(u_ii, lat_T_deg.shape[1]-1)]
        print(f"{step+1:>5} {max_u:>12.4e} {max_v:>12.4e} {max_eta:>12.4e} "
              f"  ({u_jj:>3},{u_ii:>3})     {u_lat:>7.1f}")
        if np.isnan(max_u) or max_u > 1e-6:
            print(f"\n*** NOT at rest! ***")
            break
    else:
        print(f"\n=== PASS (max|u| = {max_u:.4e}) ===")


if __name__ == "__main__":
    main()
