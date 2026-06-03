"""Momentum budget diagnostic for MPAS ETOPO blowup.

Loads a restart near the onset of instability, computes one baroclinic
tendency call, and decomposes the momentum budget into individual terms
at the fastest edges. Reports which term is driving the acceleration.

Strategy: run the full tendency with different configs to isolate terms:
- Full tendency (reference)
- pgf="zero" → difference = PGF contribution
- A_h=0 → difference = viscosity contribution
- A_v=0 → difference = vertical diffusion
- bottom_drag_r=0 → difference = bottom drag

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_momentum_budget.py
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
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate


def snap_partial_cells(H_bathy, z_coord, min_frac=0.30):
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
    H_snapped = jnp.where(H_bathy - z_upper < z_lower - H_bathy, z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_bathy > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_bathy)
    return jnp.where(H_new <= 0, 0.0, H_new)


def load_restart(restart_path, template_state):
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    restart_step = int(data["step"])
    replacements = {}
    for f in template_state._fields:
        if f not in data:
            continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = Field(data=arr, name=obj.name, dims=obj.dims, units=obj.units)
    state = template_state._replace(**replacements)
    return state, restart_day, restart_step


def main():
    # --- Setup ---
    mesh = create_voronoi_mesh(subdivision_level=5)
    z_coord = create_ocean_z_star(n_levels=10, H_max=5500.0)
    bathy_cfg = BathymetryConfig(
        source="file", path="/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc",
        H_max=5500.0, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True,
    )
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord, min_frac=0.30)
    ocean_mask_new = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    # --- Physics config ---
    physics_cfg = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1),
            restoring=RestoringConfig(tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0, S_star=35.0, T_profile="cosine"),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )

    # --- Base config (the run that blew up at day 232) ---
    base_config = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        A_h=5e4,
        A_v=1e-2,
        K_v=1e-4,
        bottom_drag_r=1e-3,
        bottom_drag_bbl_thickness=100.0,
        equatorial_visc_boost=0.0,
        pgf_scheme="centered",
        physics=physics_cfg,
    )

    # --- Template state ---
    T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, 10))
    T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(pc_coord.is_active, jnp.full_like(T_data, 35.0), 0.0)
    state = rest_state_mpas_ocean(mesh, z_coord, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=5500.0, land_lat_threshold=90.0)
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(data=H_snapped.astype(dtype), name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask_new.astype(dtype)),
        T=Field(data=T_data.astype(dtype), name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data.astype(dtype), name="S", dims=("nCells", "nlev"), units="PSU"),
    )

    # --- Load restart at day 150 (onset of exponential growth) ---
    restart_path = "outputs/mpas_etopo_spinup_bdrag1e-3_topo/restarts/restart_day000150.npz"
    state, restart_day, _ = load_restart(restart_path, state)
    print(f"Loaded state at day {restart_day:.0f}")
    print(f"  max|u| = {float(jnp.max(jnp.abs(state.u.data))):.4f} m/s")
    print(f"  max|eta| = {float(jnp.max(jnp.abs(state.eta.data))):.4f} m")

    # --- Identify fastest edges ---
    u_abs = jnp.abs(state.u.data)
    u_max_per_edge = jnp.max(u_abs, axis=1)
    top_edges = jnp.argsort(u_max_per_edge)[-20:][::-1]

    lon_e = np.degrees(np.asarray(mesh.lonEdge))
    lat_e = np.degrees(np.asarray(mesh.latEdge))

    print(f"\n  Top-10 fastest edges:")
    print(f"  {'rank':>4} {'lon':>6} {'lat':>6} {'|u|':>8} {'level':>5}")
    for i in range(10):
        eidx = int(top_edges[i])
        max_lev = int(jnp.argmax(u_abs[eidx]))
        print(f"  {i+1:4d} {lon_e[eidx]:6.1f} {lat_e[eidx]:6.1f} "
              f"{float(u_max_per_edge[eidx]):8.4f} {max_lev:5d}")

    # --- Compute tendencies with different configs ---
    # We compute the baroclinic tendency (du_dt) and depth-mean forcing (F_slow_u)
    # under various configs, then difference to isolate each term.

    # Note: we pass physics_fn=None so wind is excluded from all configs.
    # Since wind is the same in all, the DIFFERENCES isolate each term correctly.
    # The "other" column will include PV flux + KE grad + vert_adv (no wind).

    configs = {
        "full": base_config,
        "no_pgf": base_config._replace(pgf_scheme="zero"),
        "no_visc": base_config._replace(A_h=0.0),
        "no_Av": base_config._replace(A_v=0.0),
        "no_drag": base_config._replace(bottom_drag_r=0.0),
    }

    tendencies = {}
    print("\n  Computing tendencies...")
    for name, cfg in configs.items():
        t0 = time.time()
        tend = mpas_ocean_baroclinic_tendencies(
            state, mesh, pc_coord, cfg,
            physics_fn=None,
            surface_forcing=None,
        )
        dt = time.time() - t0
        tendencies[name] = tend
        mu_tend = float(jnp.max(jnp.abs(tend.du_dt.data)))
        mu_Fslow = float(jnp.max(jnp.abs(tend.F_slow_u.data)))
        print(f"    {name:>10}: max|du_dt|={mu_tend:.4e}  max|F_slow|={mu_Fslow:.4e}  ({dt:.1f}s)")

    # --- Decompose at fastest edges ---
    # The contributions are:
    # PGF = full.du_dt - no_pgf.du_dt (includes both baroclinic du_dt and F_slow)
    # visc = full.du_dt - no_visc.du_dt
    # A_v = full.du_dt - no_Av.du_dt
    # drag = full.du_dt - no_drag.du_dt
    # residual (PV flux + KE grad + vert_adv + wind) = no_pgf with no visc/drag/Av

    full_du = tendencies["full"].du_dt.data
    full_Fs = tendencies["full"].F_slow_u.data

    # For the 3D baroclinic tendency
    pgf_contrib = full_du - tendencies["no_pgf"].du_dt.data
    visc_contrib = full_du - tendencies["no_visc"].du_dt.data
    Av_contrib = full_du - tendencies["no_Av"].du_dt.data
    drag_contrib = full_du - tendencies["no_drag"].du_dt.data

    # For the depth-mean F_slow_u (what the barotropic solver sees)
    pgf_Fs = full_Fs - tendencies["no_pgf"].F_slow_u.data
    visc_Fs = full_Fs - tendencies["no_visc"].F_slow_u.data
    Av_Fs = full_Fs - tendencies["no_Av"].F_slow_u.data
    drag_Fs = full_Fs - tendencies["no_drag"].F_slow_u.data

    # --- Report budget at top edges ---
    print("\n" + "=" * 80)
    print("MOMENTUM BUDGET at top-10 fastest edges (3D baroclinic du_dt)")
    print("=" * 80)
    print(f"  {'rank':>4} {'lon':>5} {'lat':>5} {'full':>9} {'PGF':>9} "
          f"{'visc':>9} {'A_v':>9} {'drag':>9} {'other':>9}")

    for i in range(10):
        eidx = int(top_edges[i])
        max_lev = int(jnp.argmax(u_abs[eidx]))
        # Get the tendency at the level where velocity is max
        f_val = float(full_du[eidx, max_lev])
        p_val = float(pgf_contrib[eidx, max_lev])
        v_val = float(visc_contrib[eidx, max_lev])
        a_val = float(Av_contrib[eidx, max_lev])
        d_val = float(drag_contrib[eidx, max_lev])
        other = f_val - p_val - v_val - a_val - d_val
        print(f"  {i+1:4d} {lon_e[eidx]:5.1f} {lat_e[eidx]:5.1f} "
              f"{f_val:9.2e} {p_val:9.2e} {v_val:9.2e} "
              f"{a_val:9.2e} {d_val:9.2e} {other:9.2e}")

    print("\n" + "=" * 80)
    print("DEPTH-MEAN FORCING (F_slow_u → barotropic solver) at top-10 edges")
    print("=" * 80)
    print(f"  {'rank':>4} {'lon':>5} {'lat':>5} {'full':>9} {'PGF':>9} "
          f"{'visc':>9} {'A_v':>9} {'drag':>9} {'other':>9}")

    for i in range(10):
        eidx = int(top_edges[i])
        f_val = float(full_Fs[eidx])
        p_val = float(pgf_Fs[eidx])
        v_val = float(visc_Fs[eidx])
        a_val = float(Av_Fs[eidx])
        d_val = float(drag_Fs[eidx])
        other = f_val - p_val - v_val - a_val - d_val
        print(f"  {i+1:4d} {lon_e[eidx]:5.1f} {lat_e[eidx]:5.1f} "
              f"{f_val:9.2e} {p_val:9.2e} {v_val:9.2e} "
              f"{a_val:9.2e} {d_val:9.2e} {other:9.2e}")

    # --- Sign analysis: is the net tendency ACCELERATING the flow? ---
    print("\n" + "=" * 80)
    print("SIGN ANALYSIS: is du/dt aligned with u (accelerating)?")
    print("=" * 80)
    for i in range(10):
        eidx = int(top_edges[i])
        max_lev = int(jnp.argmax(u_abs[eidx]))
        u_val = float(state.u.data[eidx, max_lev])
        dudt_val = float(full_du[eidx, max_lev])
        aligned = "ACCEL" if u_val * dudt_val > 0 else "DECEL"
        # Which term drives the acceleration?
        p_val = float(pgf_contrib[eidx, max_lev])
        v_val = float(visc_contrib[eidx, max_lev])
        a_val = float(Av_contrib[eidx, max_lev])
        d_val = float(drag_contrib[eidx, max_lev])
        terms = {"PGF": p_val, "visc": v_val, "A_v": a_val, "drag": d_val}
        # Find which terms are accelerating (same sign as u)
        accel_terms = {k: v for k, v in terms.items() if v * u_val > 0}
        decel_terms = {k: v for k, v in terms.items() if v * u_val < 0}
        accel_str = ", ".join(f"{k}={v:.2e}" for k, v in
                             sorted(accel_terms.items(), key=lambda x: abs(x[1]), reverse=True))
        decel_str = ", ".join(f"{k}={v:.2e}" for k, v in
                             sorted(decel_terms.items(), key=lambda x: abs(x[1]), reverse=True))
        print(f"  edge {i+1}: u={u_val:+.3f} du/dt={dudt_val:+.2e} [{aligned}]")
        print(f"    accelerating: {accel_str}")
        print(f"    decelerating: {decel_str}")

    # --- Global budget statistics ---
    print("\n" + "=" * 80)
    print("GLOBAL max magnitude of each term")
    print("=" * 80)
    print(f"  Full du_dt:  {float(jnp.max(jnp.abs(full_du))):.4e}")
    print(f"  PGF:         {float(jnp.max(jnp.abs(pgf_contrib))):.4e}")
    print(f"  Visc (A_h):  {float(jnp.max(jnp.abs(visc_contrib))):.4e}")
    print(f"  Vert (A_v):  {float(jnp.max(jnp.abs(Av_contrib))):.4e}")
    print(f"  Bottom drag: {float(jnp.max(jnp.abs(drag_contrib))):.4e}")
    print(f"  F_slow_u:    {float(jnp.max(jnp.abs(full_Fs))):.4e}")
    print(f"  PGF in Fs:   {float(jnp.max(jnp.abs(pgf_Fs))):.4e}")


if __name__ == "__main__":
    main()
