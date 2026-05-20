#!/usr/bin/env python
"""Single-step diagnostics for ORCA1 tripolar blowup investigation.

Runs the first N steps one at a time (no scan) and prints per-step
max|u|, max|eta|, argmax location (j,i), and lat/lon of the hotspot.
"""

from __future__ import annotations

import os
import sys
import time
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
    create_gm_redi_config,
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
    n_diag_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dt = 600.0
    nlev = 20

    print("Loading ORCA1 tripolar grid...")
    geom = create_tripole_grid(str(GRID_FILE))
    lat_T_deg = np.asarray(geom.lat_T) * 180 / np.pi
    lon_T_deg = np.asarray(geom.lon_T) * 180 / np.pi

    # Metric floor
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

    # Bathymetry
    H_bathy_raw, land_mask_raw = _reconstruct_bathymetry(GRID_FILE)
    config = GlobalOverturningConfig(H_max=float(H_bathy_raw.max()), n_levels=nlev)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )

    # Physics
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig, OceanConvectionConfig,
    )

    tau_T_seconds = config.tau_T_days * 86400.0
    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="two_belt", tau_max=config.tau_max,
            ),
            restoring=RestoringConfig(
                tau_T=tau_T_seconds, tau_S=1e30,
                T_star_eq=config.T_star_eq, T_star_pole=config.T_star_pole,
                S_star=config.S_uniform, T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0, K_bg=1e-5),
        ),
        shortwave_penetration=None,
    )
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=config.A_h, A_v=config.A_v, K_v=config.K_v,
        A_h_lat_scaling=True,
        A_h_floor=2000.0,
        C_smag_lap=0.15,
        B_h=5.0e9,
        B_h_barotropic=1.0e14,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        bottom_drag_bg_velocity=0.1,
        eos="linear", eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        barotropic_solver="implicit_cn",
        implicit_vertical_mixing=True,
        use_conservation_fixer=True,
    )

    model = LatLonCGridOceanModel(geom, z_coord, ocean_config)

    # Initial conditions
    state = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        T_surface=config.T_surface, T_deep=config.T_surface,
        S_uniform=config.S_uniform,
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

    # Add stratification
    z_full = np.asarray(z_coord.z_full_ref)
    decay = np.exp(z_full / config.T_scale_depth)
    T_profile = config.T_deep + (config.T_surface - config.T_deep) * decay
    T_data = np.array(state.T.data)
    for k in range(z_coord.n_levels):
        T_data[..., k] = T_profile[k]
    state = state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units),
    )

    # JIT-compile a single step
    step_jit = jax.jit(model.step)

    print(f"\n=== Diagnosing first {n_diag_steps} steps (dt={dt}s) ===")
    print(f"{'Step':>5} {'max|u|':>12} {'max|v|':>12} {'max|eta|':>12} "
          f"{'u_argmax(j,i)':>15} {'lat':>7} {'lon':>7} "
          f"{'eta_argmax(j,i)':>17} {'lat':>7} {'lon':>7}")
    print("-" * 120)

    for step in range(n_diag_steps):
        state = step_jit(state, dt)
        jax.block_until_ready(state.eta.data)

        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        eta = np.asarray(state.eta.data)

        # Max over all levels for u, v
        u_abs = np.abs(u)
        v_abs = np.abs(v)
        max_u = float(np.nanmax(u_abs))
        max_v = float(np.nanmax(v_abs))
        max_eta = float(np.nanmax(np.abs(eta)))

        # Argmax for u (3D: j, i, k) — collapse to j,i
        u_abs_2d = np.nanmax(u_abs, axis=-1)  # max over levels
        u_jj, u_ii = np.unravel_index(np.nanargmax(u_abs_2d), u_abs_2d.shape)

        # Argmax for eta (2D)
        eta_abs = np.abs(eta)
        e_jj, e_ii = np.unravel_index(np.nanargmax(eta_abs), eta_abs.shape)

        # Lat/lon at T-points (u-point is shifted but close enough)
        u_lat = lat_T_deg[min(u_jj, lat_T_deg.shape[0]-1),
                          min(u_ii, lat_T_deg.shape[1]-1)]
        u_lon = lon_T_deg[min(u_jj, lon_T_deg.shape[0]-1),
                          min(u_ii, lon_T_deg.shape[1]-1)]
        e_lat = lat_T_deg[e_jj, e_ii]
        e_lon = lon_T_deg[e_jj, e_ii]

        print(f"{step+1:>5} {max_u:>12.4e} {max_v:>12.4e} {max_eta:>12.4e} "
              f"  ({u_jj:>3},{u_ii:>3})     {u_lat:>7.1f} {u_lon:>7.1f} "
              f"    ({e_jj:>3},{e_ii:>3})     {e_lat:>7.1f} {e_lon:>7.1f}")

        if np.isnan(max_u) or max_u > 1e10:
            print(f"\n*** BLOWUP at step {step+1} ***")
            # Print top-5 hotspots
            u_flat = u_abs_2d.ravel()
            top5 = np.argsort(u_flat)[-5:][::-1]
            print("Top-5 |u| hotspots:")
            for rank, idx in enumerate(top5):
                jj, ii = np.unravel_index(idx, u_abs_2d.shape)
                print(f"  #{rank+1}: ({jj},{ii}) lat={lat_T_deg[min(jj,lat_T_deg.shape[0]-1), min(ii,lat_T_deg.shape[1]-1)]:.1f} "
                      f"lon={lon_T_deg[min(jj,lon_T_deg.shape[0]-1), min(ii,lon_T_deg.shape[1]-1)]:.1f} "
                      f"|u|={u_flat[idx]:.4e}")
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
