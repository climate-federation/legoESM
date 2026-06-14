"""Global momentum budget diagnostic — decompose by regime.

Computes the explicit tendency decomposition at a snapshot and reports
which terms dominate at the equator, mid-latitudes, WBCs, and ACC.

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/tmp/_momentum_budget_global.py \
        results/ocean/global_overturning_mpas_etopo_Av1e2/realistic_physics_test/restart_day003652.npz
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
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
from legoesm.ocean.dynamics.ocean_pe_mpas import mpas_ocean_baroclinic_tendencies
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, KPPConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate


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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("restart")
    args = p.parse_args()

    mesh = create_voronoi_mesh(subdivision_level=5)
    z_coord = create_ocean_z_star(n_levels=20, H_max=5500.0, dz_surface=20.0, dz_deep=500.0)
    bathy_cfg = BathymetryConfig(source="file", path="/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc",
        H_max=5500.0, H_min=10.0, smoothing_passes=2, r_factor_max=0.2, depth_is_negative=True)
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord, min_frac=0.30)
    ocean_mask_new = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    # Physics (just wind + restoring for the tendency call — KPP/convection are implicit)
    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="combined",
            prescribed=PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1),
            restoring=RestoringConfig(tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0, S_star=35.0, T_profile="cosine")),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None)

    # Base config (A_h=1e4 production)
    base = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        A_h=1e4, A_v=1e-3, K_v=1e-5,
        bottom_drag_r=1e-3, bottom_drag_bbl_thickness=100.0,
        C_smag=0.2, K_zeta_bih=1e14,
        equatorial_visc_boost=0.0, pgf_scheme="centered",
        implicit_vertical_mixing=True,  # A_v excluded from tendency
        physics=physics)

    # Template + restart
    T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, 20))
    T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(pc_coord.is_active, jnp.full_like(T_data, 35.0), 0.0)
    state = rest_state_mpas_ocean(mesh, z_coord, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=5500.0, land_lat_threshold=90.0)
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(data=H_snapped.astype(dtype), name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask_new.astype(dtype)),
        T=Field(data=T_data.astype(dtype), name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data.astype(dtype), name="S", dims=("nCells", "nlev"), units="PSU"))

    state, restart_day = load_restart(args.restart, state)
    print(f"Loaded state at year {restart_day/365.25:.1f}")
    print(f"  max|u| = {float(jnp.max(jnp.abs(state.u.data))):.4f} m/s")

    # Build physics function (includes wind + restoring)
    physics_fn = make_mpas_ocean_physics(physics, implicit_vertical_mixing=True)

    # Compute tendencies with different configs to isolate terms
    # "no_wind" uses a physics without wind to isolate wind contribution
    physics_no_wind = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(scheme="restoring",
            restoring=RestoringConfig(tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0, S_star=35.0, T_profile="cosine")),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None)
    physics_fn_no_wind = make_mpas_ocean_physics(physics_no_wind, implicit_vertical_mixing=True)

    configs = {
        "full":     (base, physics_fn),
        "no_wind":  (base, physics_fn_no_wind),
        "no_pgf":   (base._replace(pgf_scheme="zero"), physics_fn),
        "no_Ah":    (base._replace(A_h=0.0), physics_fn),
        "no_smag":  (base._replace(C_smag=0.0), physics_fn),
        "no_kzeta": (base._replace(K_zeta_bih=0.0), physics_fn),
        "no_drag":  (base._replace(bottom_drag_r=0.0), physics_fn),
    }

    tendencies = {}
    print("\nComputing tendencies (with wind)...")
    for name, (cfg, phys_fn) in configs.items():
        t0 = time.time()
        tend = mpas_ocean_baroclinic_tendencies(state, mesh, pc_coord, cfg,
                                                physics_fn=phys_fn, surface_forcing=None)
        dt = time.time() - t0
        tendencies[name] = tend
        mu = float(jnp.max(jnp.abs(tend.du_dt.data)))
        mf = float(jnp.max(jnp.abs(tend.F_slow_u.data)))
        print(f"  {name:>10}: max|du_dt|={mu:.4e}  max|F_slow|={mf:.4e}  ({dt:.1f}s)")

    # Decompose
    full = tendencies["full"].du_dt.data
    full_Fs = tendencies["full"].F_slow_u.data

    pgf = full - tendencies["no_pgf"].du_dt.data
    Ah_visc = full - tendencies["no_Ah"].du_dt.data
    smag = full - tendencies["no_smag"].du_dt.data
    kzeta = full - tendencies["no_kzeta"].du_dt.data
    drag = full - tendencies["no_drag"].du_dt.data
    wind = full - tendencies["no_wind"].du_dt.data
    other = full - pgf - Ah_visc - smag - kzeta - drag - wind  # PV flux + KE grad + vert_adv

    # Edge coordinates
    lat_e = np.degrees(np.asarray(mesh.latEdge))
    lon_e = np.degrees(np.asarray(mesh.lonEdge))

    # RMS magnitude per component
    def rms(x, mask=None):
        if mask is not None:
            x = x[mask]
        return float(jnp.sqrt(jnp.mean(x**2)))

    # Define regimes by edge latitude
    c0 = np.asarray(mesh.cellsOnEdge[0]); c1 = np.asarray(mesh.cellsOnEdge[1])
    ocean_edge = (np.asarray(ocean_mask_new)[c0] > 0.5) & (np.asarray(ocean_mask_new)[c1] > 0.5)

    equatorial = ocean_edge & (np.abs(lat_e) < 10)
    subtropical = ocean_edge & (np.abs(lat_e) >= 20) & (np.abs(lat_e) < 40)
    subpolar = ocean_edge & (np.abs(lat_e) >= 40) & (np.abs(lat_e) < 60)
    acc = ocean_edge & (lat_e < -55) & (lat_e > -70)

    # Surface level (k=0) — where wind drives
    components = {
        "PGF": np.asarray(pgf[:, 0]),
        "Wind": np.asarray(wind[:, 0]),
        "A_h": np.asarray(Ah_visc[:, 0]),
        "K_zeta": np.asarray(kzeta[:, 0]),
        "Drag": np.asarray(drag[:, 0]),
        "Smag": np.asarray(smag[:, 0]),
        "Other(PV+KE)": np.asarray(other[:, 0]),
        "Full": np.asarray(full[:, 0]),
    }

    print("\n" + "=" * 80)
    print("SURFACE (k=0) MOMENTUM BUDGET — RMS by regime [m/s²]")
    print("=" * 80)
    header = f"{'Regime':>15} " + " ".join(f"{k:>12}" for k in components.keys())
    print(header)
    print("-" * len(header))

    for regime_name, mask_r in [("Equatorial", equatorial),
                                  ("Subtropical", subtropical),
                                  ("Subpolar", subpolar),
                                  ("ACC", acc),
                                  ("Global ocean", ocean_edge)]:
        vals = [rms(v, mask_r) for v in components.values()]
        row = f"{regime_name:>15} " + " ".join(f"{v:12.4e}" for v in vals)
        print(row)

    # Depth-averaged (F_slow_u) budget
    pgf_Fs = full_Fs - tendencies["no_pgf"].F_slow_u.data
    wind_Fs = full_Fs - tendencies["no_wind"].F_slow_u.data
    Ah_Fs = full_Fs - tendencies["no_Ah"].F_slow_u.data
    smag_Fs = full_Fs - tendencies["no_smag"].F_slow_u.data
    kzeta_Fs = full_Fs - tendencies["no_kzeta"].F_slow_u.data
    drag_Fs = full_Fs - tendencies["no_drag"].F_slow_u.data
    other_Fs = full_Fs - pgf_Fs - wind_Fs - Ah_Fs - smag_Fs - kzeta_Fs - drag_Fs

    components_Fs = {
        "PGF": np.asarray(pgf_Fs),
        "Wind": np.asarray(wind_Fs),
        "A_h": np.asarray(Ah_Fs),
        "K_zeta": np.asarray(kzeta_Fs),
        "Drag": np.asarray(drag_Fs),
        "Smag": np.asarray(smag_Fs),
        "Other": np.asarray(other_Fs),
        "Full": np.asarray(full_Fs),
    }

    print("\n" + "=" * 80)
    print("DEPTH-MEAN (F_slow_u) MOMENTUM BUDGET — RMS by regime [m/s²]")
    print("=" * 80)
    header = f"{'Regime':>15} " + " ".join(f"{k:>12}" for k in components_Fs.keys())
    print(header)
    print("-" * len(header))

    for regime_name, mask_r in [("Equatorial", equatorial),
                                  ("Subtropical", subtropical),
                                  ("Subpolar", subpolar),
                                  ("ACC", acc),
                                  ("Global ocean", ocean_edge)]:
        vals = [rms(v, mask_r) for v in components_Fs.values()]
        row = f"{regime_name:>15} " + " ".join(f"{v:12.4e}" for v in vals)
        print(row)

    print("\nNote: A_v (implicit) is NOT in this budget — it's applied as a")
    print("separate backward-Euler step. Wind IS included (via physics_fn).")
    print("Coriolis is in the barotropic solver — not shown here.")


if __name__ == "__main__":
    main()
