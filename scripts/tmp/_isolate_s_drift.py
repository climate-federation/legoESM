#!/usr/bin/env python
"""Isolation experiments for MPAS S drift diagnosis.

Runs 10-day MPAS simulations with components disabled one at a time.
Reports column-mean S deficit at shallow vs deep cells.

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/tmp/_isolate_s_drift.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, KPPConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    LateralMixingConfig, GMRediConfig, VisbeckConfig,
)
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import (
    OceanConvectionConfig, EnhancedDiffusionConfig,
)
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate


SUBDIVISION = 5
N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
DT = 1200.0
SNAP_FRAC = 0.30
NORTH_CAP_LAT = 80.0
DAYS = 10  # shorter for faster iteration


def snap_partial_cells(H_bathy, z_coord, min_frac=SNAP_FRAC):
    abs_z_half = jnp.abs(z_coord.z_half_ref)
    nlev = z_coord.n_levels
    n_above = jnp.sum(abs_z_half[None, :] < H_bathy[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    abs_z_at_bottom = abs_z_half[bottom_level]
    dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_bathy - abs_z_at_bottom
    frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]
    z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_bathy - z_upper < z_lower - H_bathy,
                           z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_bathy > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_bathy)
    return jnp.where(H_new <= 0, 0.0, H_new)


def build_experiment(name, etopo_path):
    """Build (config, physics) for each isolation experiment."""

    # Full baseline physics
    full_physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind", tau_max=0.1,
                tropical_wind_scale=0.5, tropical_wind_lat_deg=15.0,
            ),
            restoring=RestoringConfig(
                tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0,
                S_star=35.0, T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(
            scheme="kpp", kpp=KPPConfig(K_conv=1.0),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
        ),
        shortwave_penetration=None,
    )

    # Full baseline model config
    base_cfg = dict(
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        A_h=1e4,
        A_v=1e-4,
        C_smag_lap=0.33,
        K_v=1e-5,
        bottom_drag_r=1e-3,
        bottom_drag_bbl_thickness=100.0,
        bottom_drag_bg_velocity=0.1,
        K_zeta_bih=1e14,
        equatorial_visc_boost=0.0,
        pgf_scheme="adcroft",
        implicit_vertical_mixing=True,
        tracer_advection="tvd",
        gm_redi=GMRediConfig(
            kappa_GM=600.0, kappa_Redi=600.0, S_max=0.005,
            visbeck=VisbeckConfig(enabled=False),
            slope_scheme="centered",
        ),
    )

    if name == "baseline":
        return MPASOceanConfig(**base_cfg, physics=full_physics), "Full config"

    elif name == "no_gmredi":
        cfg = dict(base_cfg)
        cfg["gm_redi"] = None
        return MPASOceanConfig(**cfg, physics=full_physics), "No GM/Redi"

    elif name == "no_kpp":
        physics = full_physics._replace(
            vertical_mixing=VerticalMixingConfig(scheme="none"),
            convection=OceanConvectionConfig(scheme="none"),
        )
        return MPASOceanConfig(**base_cfg, physics=physics), "No KPP/convection"

    elif name == "no_forcing":
        physics = full_physics._replace(
            surface_forcing=SurfaceForcingConfig(scheme="none"),
        )
        return MPASOceanConfig(**base_cfg, physics=physics), "No surface forcing"

    elif name == "no_implicit":
        cfg = dict(base_cfg)
        cfg["implicit_vertical_mixing"] = False
        return MPASOceanConfig(**cfg, physics=full_physics), "Explicit vertical mixing"

    elif name == "upwind":
        cfg = dict(base_cfg)
        cfg["tracer_advection"] = "upwind"
        return MPASOceanConfig(**cfg, physics=full_physics), "Upwind (no TVD)"

    elif name == "no_visc":
        cfg = dict(base_cfg)
        cfg["A_h"] = 0.0
        cfg["C_smag_lap"] = 0.0
        cfg["K_zeta_bih"] = 0.0
        return MPASOceanConfig(**cfg, physics=full_physics), "No horiz viscosity"

    elif name == "no_drag":
        cfg = dict(base_cfg)
        cfg["bottom_drag_r"] = 0.0
        cfg["bottom_drag_bg_velocity"] = 0.0
        return MPASOceanConfig(**cfg, physics=full_physics), "No bottom drag"

    else:
        raise ValueError(f"Unknown experiment: {name}")


def run_one(name, mesh, pc_coord, z_coord, H_snapped, ocean_mask, etopo_path):
    """Run a single isolation experiment and return S stats."""
    config, desc = build_experiment(name, etopo_path)
    model = MPASOceanModel(mesh, pc_coord, config)

    # Initial condition
    T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, N_LEVELS))
    T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(pc_coord.is_active,
                        jnp.full_like(T_data, 35.0), 0.0)

    state = rest_state_mpas_ocean(
        mesh, z_coord, T_surface=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=H_MAX, land_lat_threshold=90.0,
    )
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(data=H_snapped.astype(dtype), name="H_bathy",
                      dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask.astype(dtype)),
        T=Field(data=T_data.astype(dtype), name="T",
                dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data.astype(dtype), name="S",
                dims=("nCells", "nlev"), units="PSU"),
    )

    n_steps = int(DAYS * 86400 / DT)
    t0 = time.time()
    for k in range(n_steps):
        state = model.step(state, dt=DT)
    wall = time.time() - t0

    # S stats
    mask_np = np.asarray(ocean_mask) > 0.5
    H_np = np.asarray(H_snapped)
    S_np = np.asarray(state.S.data)
    shallow = mask_np & (H_np < 1000) & (H_np > 0)
    deep = mask_np & (H_np > 4000)
    active = S_np != 0
    col_mean = np.where(active, S_np, np.nan)
    col_mean = np.nanmean(col_mean, axis=1)

    S_shallow = col_mean[shallow].mean()
    S_deep = col_mean[deep].mean()
    S_sfc_std = S_np[:, 0][mask_np].std()
    max_u = float(jnp.max(jnp.abs(state.u.data)))

    return {
        "name": name,
        "desc": desc,
        "S_shallow": S_shallow,
        "S_deep": S_deep,
        "deficit": S_shallow - S_deep,
        "S_sfc_std": S_sfc_std,
        "max_u": max_u,
        "wall": wall,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--etopo",
                   default="/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc")
    p.add_argument("--experiments", nargs="+",
                   default=["baseline", "no_gmredi", "no_kpp",
                            "no_forcing", "no_implicit", "upwind",
                            "no_visc", "no_drag"])
    args = p.parse_args()

    print(f"=== S drift isolation ({DAYS} days each) ===\n")

    # Build grid and bathymetry once
    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
    z_coord = create_ocean_z_star(n_levels=N_LEVELS, H_max=H_MAX,
                                  dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP)
    bathy_cfg = BathymetryConfig(
        source="file", path=args.etopo,
        H_max=H_MAX, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True,
    )
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    lat_cell = np.degrees(np.asarray(mesh.latCell))
    north_cap_mask = jnp.asarray(lat_cell <= NORTH_CAP_LAT,
                                  dtype=H_bathy_raw.dtype)
    H_bathy_raw = H_bathy_raw * north_cap_mask
    ocean_mask = ocean_mask * north_cap_mask
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord)
    ocean_mask = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    results = []
    for name in args.experiments:
        print(f"--- Running: {name} ---")
        r = run_one(name, mesh, pc_coord, z_coord, H_snapped, ocean_mask,
                    args.etopo)
        results.append(r)
        print(f"  {r['desc']}: S_shallow={r['S_shallow']:.7f}, "
              f"S_deep={r['S_deep']:.7f}, "
              f"deficit={r['deficit']:.7f}, "
              f"sfc_std={r['S_sfc_std']:.7f}, "
              f"max|u|={r['max_u']:.3f}, "
              f"wall={r['wall']:.0f}s\n")

    print("\n=== Summary ===")
    print(f"{'Experiment':<20s} {'Desc':<25s} {'S_shallow':>12s} {'S_deep':>12s} "
          f"{'Deficit':>12s} {'Sfc_std':>12s} {'max|u|':>8s}")
    print("-" * 105)
    for r in results:
        print(f"{r['name']:<20s} {r['desc']:<25s} {r['S_shallow']:12.7f} "
              f"{r['S_deep']:12.7f} {r['deficit']:12.7f} "
              f"{r['S_sfc_std']:12.7f} {r['max_u']:8.3f}")


if __name__ == "__main__":
    main()
