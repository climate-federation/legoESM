"""Dominant momentum balance by depth and regime.

Computes all budget terms from the state, reports which terms dominate
at each depth/regime, and classifies the physical balance.

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_budget_dominant_balance.py
"""
from __future__ import annotations

import os, sys, time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax; import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from legoesm.core.field import Field
from legoesm.core.operators_voronoi import (
    gradient_edge_3d, tangential_velocity_3d,
)
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
from legoesm.ocean.dynamics.ocean_pe_mpas import (
    mpas_ocean_baroclinic_tendencies, _vertical_diffusion,
)
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
    compute_max_level_edge_bot, min_cell_to_edge,
)
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, KPPConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import (
    OceanConvectionConfig, EnhancedDiffusionConfig)
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate,
    compute_ocean_jacobian, compute_layer_thickness,
)


def snap_partial_cells(H_bathy, z_coord, min_frac=0.30):
    abs_z_half = jnp.abs(z_coord.z_half_ref); nlev = z_coord.n_levels
    n_above = jnp.sum(abs_z_half[None, :] < H_bathy[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    abs_z_at_bottom = abs_z_half[bottom_level]; dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_bathy - abs_z_at_bottom; frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]; z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_bathy - z_upper < z_lower - H_bathy, z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_bathy > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_bathy)
    return jnp.where(H_new <= 0, 0.0, H_new)


def load_restart(restart_path, template_state):
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    replacements = {}
    for f in template_state._fields:
        if f not in data: continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"): continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = Field(data=arr, name=obj.name, dims=obj.dims, units=obj.units)
    return template_state._replace(**replacements), restart_day


def main():
    mesh = create_voronoi_mesh(subdivision_level=5)
    z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0, dz_surface=20.0, dz_deep=500.0)
    bathy_cfg = BathymetryConfig(source="file",
        path="/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc",
        H_max=5500.0, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True)
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord, min_frac=0.30)
    ocean_mask_new = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="combined",
            prescribed=PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1),
            restoring=RestoringConfig(tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0, S_star=35.0, T_profile="cosine")),
        vertical_mixing=VerticalMixingConfig(scheme="kpp", kpp=KPPConfig(K_conv=1.0)),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0)),
        shortwave_penetration=None)

    config = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        A_h=1e4, A_v=1e-3, K_v=1e-5,
        bottom_drag_r=1e-3, bottom_drag_bbl_thickness=100.0,
        C_smag=0.2, K_zeta_bih=1e14,
        equatorial_visc_boost=0.0, pgf_scheme="centered",
        implicit_vertical_mixing=True,
        physics=physics)

    # Template + restart
    T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, 20))
    T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(pc_coord.is_active, jnp.full_like(T_data, 35.0), 0.0)
    state = rest_state_mpas_ocean(mesh, z_coord, T_surface=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=5500.0, land_lat_threshold=90.0)
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(data=H_snapped.astype(dtype), name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask_new.astype(dtype)),
        T=Field(data=T_data.astype(dtype), name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data.astype(dtype), name="S", dims=("nCells", "nlev"), units="PSU"))

    state, restart_day = load_restart(
        "results/ocean/global_overturning_mpas_etopo_Av1e2/realistic_physics_test/restart_day003652.npz",
        state)
    print(f"Loaded year {restart_day/365.25:.1f}")

    mask = state.land_mask.data
    c1 = mesh.cellsOnEdge[0]; c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    # --- Compute u_bar and u_prime ---
    u_3d = state.u.data
    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, pc_coord,
                                  min_water_column_m=config.min_water_column_m)
    h_e = min_cell_to_edge(h_k, mesh)
    H_e = jnp.maximum(jnp.sum(h_e, axis=1), 1e-10)
    u_bar = jnp.sum(u_3d * h_e, axis=1) / H_e * edge_mask
    u_prime = u_3d - u_bar[:, None]

    # --- Per-level edge mask ---
    bot_e = compute_max_level_edge_bot(pc_coord.bottom_level, mesh)
    nlev = u_3d.shape[1]
    k_idx = jnp.arange(nlev, dtype=bot_e.dtype)
    edge_mask_3d = (k_idx[None, :] <= bot_e[:, None]).astype(dtype) * edge_mask[:, None]

    # --- 1. PE tendency decomposition (PGF, wind, A_h, K_zeta, drag, nonlinear) ---
    physics_fn = make_mpas_ocean_physics(physics, implicit_vertical_mixing=True)
    physics_fn_no_wind = make_mpas_ocean_physics(
        physics._replace(surface_forcing=SurfaceForcingConfig(scheme="restoring",
            restoring=RestoringConfig(tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0, S_star=35.0, T_profile="cosine"))),
        implicit_vertical_mixing=True)

    print("Computing PE tendencies...")
    t0 = time.time()
    tend_full = mpas_ocean_baroclinic_tendencies(state, mesh, pc_coord, config,
                                                  physics_fn=physics_fn)
    tend_no_wind = mpas_ocean_baroclinic_tendencies(state, mesh, pc_coord, config,
                                                     physics_fn=physics_fn_no_wind)
    tend_no_pgf = mpas_ocean_baroclinic_tendencies(state, mesh, pc_coord,
                    config._replace(pgf_scheme="zero"), physics_fn=physics_fn)
    tend_no_Ah = mpas_ocean_baroclinic_tendencies(state, mesh, pc_coord,
                    config._replace(A_h=0.0), physics_fn=physics_fn)
    tend_no_kzeta = mpas_ocean_baroclinic_tendencies(state, mesh, pc_coord,
                    config._replace(K_zeta_bih=0.0), physics_fn=physics_fn)
    tend_no_drag = mpas_ocean_baroclinic_tendencies(state, mesh, pc_coord,
                    config._replace(bottom_drag_r=0.0), physics_fn=physics_fn)
    print(f"  Done in {time.time()-t0:.1f}s")

    # du_dt_full includes F_slow_u (depth-mean) + du_dt_3d (perturbation)
    # Reconstruct the FULL tendency before depth-mean removal
    du_full = tend_full.du_dt.data + tend_full.F_slow_u.data[:, None]

    # Decompose
    wind_3d = du_full - (tend_no_wind.du_dt.data + tend_no_wind.F_slow_u.data[:, None])
    pgf_3d = du_full - (tend_no_pgf.du_dt.data + tend_no_pgf.F_slow_u.data[:, None])
    Ah_3d = du_full - (tend_no_Ah.du_dt.data + tend_no_Ah.F_slow_u.data[:, None])
    kzeta_3d = du_full - (tend_no_kzeta.du_dt.data + tend_no_kzeta.F_slow_u.data[:, None])
    drag_3d = du_full - (tend_no_drag.du_dt.data + tend_no_drag.F_slow_u.data[:, None])
    nonlinear_3d = du_full - wind_3d - pgf_3d - Ah_3d - kzeta_3d - drag_3d

    # --- 2. Coriolis ---
    f_e = mesh.fEdge[:, None].astype(dtype)  # (nEdges, 1)
    coriolis_prime = f_e * tangential_velocity_3d(u_prime, mesh) * edge_mask_3d
    coriolis_bar = (mesh.fEdge * jnp.squeeze(tangential_velocity_3d(
        u_bar[:, None], mesh), axis=-1) * edge_mask)

    # --- 3. Barotropic SSH gradient ---
    # eta is (nCells,) — gradient gives (nEdges,). Broadcast to all levels.
    eta_2d = state.eta.data[:, None]  # (nCells, 1)
    g_grad_eta_1d = config.g * gradient_edge_3d(eta_2d, mesh)[:, 0] * edge_mask  # (nEdges,)
    g_grad_eta_3d = jnp.broadcast_to(g_grad_eta_1d[:, None], u_3d.shape)

    # --- Coordinates and regime masks ---
    lat_e = np.degrees(np.asarray(mesh.latEdge))
    ocean_edge = np.asarray(edge_mask) > 0.5
    z_full = np.asarray(z_coord.z_full_ref)

    regimes = {
        "Equatorial (|lat|<10°)": ocean_edge & (np.abs(lat_e) < 10),
        "Subtropical (20-40°)": ocean_edge & (np.abs(lat_e) >= 20) & (np.abs(lat_e) < 40),
        "Subpolar (40-60°)": ocean_edge & (np.abs(lat_e) >= 40) & (np.abs(lat_e) < 60),
        "ACC (55-70°S)": ocean_edge & (lat_e < -55) & (lat_e > -70),
    }

    depth_bands = {
        "Surface (0-50m)": [0, 1, 2],          # levels 0-2: 0-106m
        "Upper (50-500m)": [3, 4, 5],           # levels 3-5: 194-450m
        "Mid (500-2000m)": [6, 7, 8, 9, 10],   # levels 6-10: 618-1558m
        "Deep (2000-4000m)": [11, 12, 13, 14, 15],  # levels 11-15: 1860-3334m
        "Abyss (>4000m)": [16, 17, 18, 19],    # levels 16-19: 3769-5236m
    }

    # All budget terms (3D)
    terms = {
        "Baro PGF (-g∇η)": np.asarray(-g_grad_eta_3d),
        "Bclnc PGF (p'/ρ₀)": np.asarray(pgf_3d),
        "Wind stress": np.asarray(wind_3d),
        "Coriolis(u')": np.asarray(coriolis_prime),
        "Coriolis(ū)": np.asarray(jnp.broadcast_to(coriolis_bar[:, None], u_3d.shape)),
        "A_h viscosity": np.asarray(Ah_3d),
        "K_zeta biharm": np.asarray(kzeta_3d),
        "Bottom drag": np.asarray(drag_3d),
        "Nonlinear(PV+KE)": np.asarray(nonlinear_3d),
    }

    def rms(arr, edge_mask_r, level_list):
        vals = []
        for k in level_list:
            vals.append(arr[edge_mask_r, k])
        v = np.concatenate(vals)
        return np.sqrt(np.mean(v**2)) if len(v) > 0 else 0.0

    # --- Print budget tables ---
    print("\n" + "=" * 100)
    print("DOMINANT MOMENTUM BALANCE BY DEPTH AND REGIME")
    print("=" * 100)

    for regime_name, regime_mask in regimes.items():
        print(f"\n{'─' * 100}")
        print(f"  {regime_name}")
        print(f"{'─' * 100}")
        header = f"  {'Depth band':>20}" + "".join(f"{k:>14}" for k in terms.keys())
        print(header)

        for depth_name, levels in depth_bands.items():
            vals = [rms(v, regime_mask, levels) for v in terms.values()]
            max_val = max(vals) if max(vals) > 0 else 1.0
            # Mark dominant terms (>20% of max)
            row = f"  {depth_name:>20}"
            for v in vals:
                frac = v / max_val
                marker = " *" if frac > 0.20 else "  "
                row += f"  {v:10.2e}{marker}"
            print(row)

        # Summary: what's the dominant balance?
        print(f"\n  Dominant balance summary for {regime_name}:")
        for depth_name, levels in depth_bands.items():
            vals = {k: rms(v, regime_mask, levels) for k, v in terms.items()}
            sorted_terms = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            top = sorted_terms[:3]
            top_str = " ≈ ".join(f"{k}({v:.1e})" for k, v in top)
            print(f"    {depth_name:>20}: {top_str}")

    # --- Total pressure gradient (barotropic + baroclinic) ---
    print("\n" + "=" * 100)
    print("TOTAL PRESSURE GRADIENT vs CORIOLIS (geostrophic balance check)")
    print("=" * 100)

    total_pgf = np.asarray(-g_grad_eta_3d) + np.asarray(pgf_3d)
    total_coriolis = np.asarray(coriolis_prime) + np.asarray(
        jnp.broadcast_to(coriolis_bar[:, None], u_3d.shape))

    print(f"\n  {'Regime':>25} {'Depth':>20} {'RMS ∇p':>12} {'RMS f×u':>12} {'Ratio':>8} {'Balance':>15}")
    print(f"  {'':>25} {'':>20} {'[m/s²]':>12} {'[m/s²]':>12}")
    for regime_name, regime_mask in regimes.items():
        for depth_name, levels in depth_bands.items():
            rms_pgf = rms(total_pgf, regime_mask, levels)
            rms_cor = rms(total_coriolis, regime_mask, levels)
            ratio = rms_pgf / rms_cor if rms_cor > 0 else float('inf')
            if ratio > 0.8 and ratio < 1.2:
                balance = "GEOSTROPHIC"
            elif ratio > 2:
                balance = "ageostrophic"
            else:
                balance = "partial"
            print(f"  {regime_name:>25} {depth_name:>20} {rms_pgf:12.2e} {rms_cor:12.2e} {ratio:8.2f} {balance:>15}")

    print(f"\n  * = term contributes >20% of the dominant term at that depth/regime")
    print(f"  Geostrophic: ratio ∇p/f×u in [0.8, 1.2]")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
