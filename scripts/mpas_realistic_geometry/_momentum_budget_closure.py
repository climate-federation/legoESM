#!/usr/bin/env python
"""Full momentum budget closure at every grid point -- approximate and exact.

Part A: Approximate budget
  Calls mpas_ocean_baroclinic_tendencies to get du_dt_3d and F_slow_u, then
  computes the remaining terms (Coriolis on u_prime, g*grad(eta), explicit A_v)
  that are NOT in the tendency function but are applied in the step() split.

Part B: Exact budget
  Runs one full model.step() from the same restart, computes
  du/dt_exact = (u_new - u_old) / dt, and compares with the approximate sum.

Part C: Comparison
  Reports RMS of each term by regime, approximate closure residual,
  exact closure residual, and the "implicit correction" = exact - approximate.

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_momentum_budget_closure.py
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
from legoesm.core.operators_voronoi import (
    gradient_edge_3d,
    tangential_velocity_3d,
)
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.dynamics.ocean_pe_mpas import (
    _vertical_diffusion,
    mpas_ocean_baroclinic_tendencies,
)
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
    compute_max_level_edge_bot,
    min_cell_to_edge,
)
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.mpas_physics import make_mpas_ocean_physics
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig,
    RestoringConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    KPPConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.vertical import (
    compute_layer_thickness,
    compute_ocean_jacobian,
    create_ocean_z_star,
    create_partial_cell_coordinate,
)
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    implicit_vertical_diffusion_ocean,
    build_dz_half,
)

# ============================================================================
# Configuration
# ============================================================================

RESTART_PATH = (
    "results/ocean/global_overturning_mpas_etopo_Av1e2/"
    "realistic_physics_test/restart_day003652.npz"
)
ETOPO_PATH = "/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc"
SUBDIVISION = 5
N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
DT = 300.0  # seconds
SNAP_FRAC = 0.30


# ============================================================================
# Helpers
# ============================================================================


def snap_partial_cells(H_bathy, z_coord, min_frac=SNAP_FRAC):
    """Round H_bathy to nearest interface when bottom partial cell < min_frac."""
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
    H_snapped = jnp.where(
        H_bathy - z_upper < z_lower - H_bathy, z_upper, z_lower
    )
    needs_snap = (frac < min_frac) & (frac > 0) & (H_bathy > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_bathy)
    return jnp.where(H_new <= 0, 0.0, H_new)


def load_restart(restart_path, template_state):
    """Load restart npz into template state."""
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    replacements = {}
    for f in template_state._fields:
        if f not in data:
            continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = Field(
            data=arr, name=obj.name, dims=obj.dims, units=obj.units
        )
    return template_state._replace(**replacements), restart_day


def rms(x, mask=None):
    """Root-mean-square of x over masked entries."""
    if mask is not None:
        x = x[mask]
    n = max(x.size, 1)
    return float(jnp.sqrt(jnp.sum(x ** 2) / n))


def rms_3d(x_3d, edge_mask_2d):
    """RMS across (nEdges, nlev) where edge_mask_2d selects edges."""
    vals = x_3d[edge_mask_2d]  # (n_selected, nlev)
    n = max(vals.size, 1)
    return float(jnp.sqrt(jnp.sum(vals ** 2) / n))


# ============================================================================
# Main
# ============================================================================


def main():
    t_wall_start = time.time()

    # ---- Grid ----
    print("=" * 80)
    print("MPAS MOMENTUM BUDGET CLOSURE DIAGNOSTIC")
    print("=" * 80)
    print(f"\nBuilding mesh and bathymetry (ico{SUBDIVISION}, {N_LEVELS} levels)...")
    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
    z_coord = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP,
    )
    bathy_cfg = BathymetryConfig(
        source="file",
        path=ETOPO_PATH,
        H_max=H_MAX,
        H_min=10.0,
        smoothing_passes=2,
        r_factor_max=0.2,
        depth_is_negative=True,
    )
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord)
    ocean_mask_new = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    n_ocean = int(jnp.sum(ocean_mask_new > 0.5))
    print(f"  Mesh: nCells={mesh.nCells}, nEdges={mesh.nEdges}")
    print(f"  Ocean cells: {n_ocean}/{mesh.nCells}")

    # ---- Physics config ----
    # Wind + restoring for the tendency call; KPP and convection go through
    # implicit path (handled in step()).
    physics_cfg = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind", tau_max=0.1,
            ),
            restoring=RestoringConfig(
                tau_T=2592000.0,
                tau_S=2592000.0,
                T_star_eq=25.0,
                T_star_pole=0.0,
                S_star=35.0,
                T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(
            scheme="kpp",
            kpp=KPPConfig(K_conv=1.0),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="enhanced_diffusion"),
        shortwave_penetration=None,
    )

    # ---- Model config (A_h=1e4 realistic physics) ----
    config = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        A_h=1e4,
        A_v=1e-3,
        K_v=1e-5,
        bottom_drag_r=1e-3,
        bottom_drag_bbl_thickness=100.0,
        C_smag=0.2,
        K_zeta_bih=1e14,
        equatorial_visc_boost=0.0,
        pgf_scheme="centered",
        implicit_vertical_mixing=True,
        physics=physics_cfg,
    )

    # ---- Build template state and load restart ----
    T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, N_LEVELS))
    T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(
        pc_coord.is_active, jnp.full_like(T_data, 35.0), 0.0
    )

    state = rest_state_mpas_ocean(
        mesh, z_coord, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=H_MAX, land_lat_threshold=90.0,
    )
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(
            data=H_snapped.astype(dtype),
            name="H_bathy",
            dims=("nCells",),
            units="m",
        ),
        land_mask=Field(data=ocean_mask_new.astype(dtype)),
        T=Field(
            data=T_data.astype(dtype),
            name="T",
            dims=("nCells", "nlev"),
            units="degC",
        ),
        S=Field(
            data=S_data.astype(dtype),
            name="S",
            dims=("nCells", "nlev"),
            units="PSU",
        ),
    )

    state, restart_day = load_restart(RESTART_PATH, state)
    print(f"\nLoaded restart at year {restart_day / 365.25:.1f} (day {restart_day:.0f})")
    print(f"  max|u| = {float(jnp.max(jnp.abs(state.u.data))):.4f} m/s")
    print(f"  max|eta| = {float(jnp.max(jnp.abs(state.eta.data))):.4f} m")

    # ---- Precompute shared geometry ----
    u_3d = state.u.data  # (nEdges, nlev)
    eta = state.eta.data  # (nCells,)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    # Per-level edge mask for partial cells
    bot_e = compute_max_level_edge_bot(pc_coord.bottom_level, mesh)
    nlev = u_3d.shape[1]
    k_idx = jnp.arange(nlev, dtype=bot_e.dtype)
    edge_mask_3d = (
        (k_idx[None, :] <= bot_e[:, None]).astype(u_3d.dtype)
        * edge_mask[:, None]
    )

    # Layer thicknesses
    h_k = compute_layer_thickness(
        eta, H_bathy, pc_coord, min_water_column_m=config.min_water_column_m,
    )
    h_e_3d = min_cell_to_edge(h_k, mesh)
    _u_pair = jnp.sum(
        jnp.stack([h_e_3d, u_3d * h_e_3d], axis=-1), axis=1,
    )
    H_e = jnp.maximum(_u_pair[..., 0], config.min_water_column_m)
    u_bar = _u_pair[..., 1] / jnp.maximum(H_e, 1e-10)
    u_bar = u_bar * edge_mask
    u_prime = u_3d - u_bar[:, jnp.newaxis]

    # Jacobian for explicit A_v computation
    jacobian = compute_ocean_jacobian(
        eta, H_bathy, pc_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # ====================================================================
    # PART A: APPROXIMATE BUDGET
    # ====================================================================
    print("\n" + "=" * 80)
    print("PART A: APPROXIMATE BUDGET (from tendency function + explicit terms)")
    print("=" * 80)

    # A1: Baroclinic tendencies (du_dt_3d and F_slow_u)
    # Build physics function to include wind
    physics_fn = make_mpas_ocean_physics(
        physics_cfg, implicit_vertical_mixing=True,
    )

    print("\n  Computing baroclinic tendencies...")
    t0 = time.time()
    tend = mpas_ocean_baroclinic_tendencies(
        state, mesh, pc_coord, config,
        physics_fn=physics_fn,
        surface_forcing=None,
    )
    t1 = time.time()
    du_dt_3d = tend.du_dt.data  # (nEdges, nlev) — baroclinic PERTURBATION tendency
    F_slow_u = tend.F_slow_u.data  # (nEdges,) — depth-mean forcing for barotropic
    print(f"    Tendency computed in {t1 - t0:.1f}s")
    print(f"    max|du_dt_3d| = {float(jnp.max(jnp.abs(du_dt_3d))):.4e}")
    print(f"    max|F_slow_u| = {float(jnp.max(jnp.abs(F_slow_u))):.4e}")

    # A2: Coriolis on u_prime: f_edge * tangential_velocity(u_prime)
    # The forward-backward Matsuno in step() does:
    #   v_t_old = tangential_velocity_3d(u_prime)
    #   u_prime_star = u_prime + dt * f * v_t_old
    #   v_t_star = tangential_velocity_3d(u_prime_star)
    #   u_prime_new = u_prime + dt * f * 0.5*(v_t_old + v_t_star)
    # For the approximate budget, use the leading-order term: f * v_t(u_prime)
    print("  Computing Coriolis on u_prime...")
    f_e = mesh.fEdge[:, jnp.newaxis]  # (nEdges, 1)
    v_t_prime = tangential_velocity_3d(u_prime, mesh)
    coriolis_u_prime = f_e * v_t_prime * edge_mask_3d  # (nEdges, nlev)
    print(f"    max|f*v_t(u')| = {float(jnp.max(jnp.abs(coriolis_u_prime))):.4e}")

    # A3: g * grad(eta) — the barotropic pressure gradient
    # This is handled by the barotropic solver, NOT in the tendency function.
    print("  Computing g*grad(eta)...")
    g_grad_eta = config.g * gradient_edge_3d(
        eta[:, jnp.newaxis] * jnp.ones((1, nlev)),
        mesh,
    ) * edge_mask_3d  # (nEdges, nlev)
    # Note: this is the same at every level (barotropic)
    print(f"    max|g*grad(eta)| = {float(jnp.max(jnp.abs(g_grad_eta))):.4e}")

    # A4: Explicit vertical diffusion (forward-Euler approximation of what
    # the implicit solver does).
    # When implicit_vertical_mixing=True, A_v is NOT in the tendency.
    # The step() function applies backward-Euler implicit. Here we compute
    # the forward-Euler version for comparison.
    print("  Computing explicit A_v (forward-Euler approx)...")
    # Use per-edge actual dz from min-cell-to-edge (matches implicit solver)
    dz_edge = jnp.maximum(h_e_3d, 1e-10)
    dz_half_edge = build_dz_half(dz_edge)
    # Build per-edge A_v with sub-seafloor zeroing
    k_half = jnp.arange(nlev - 1, dtype=bot_e.dtype)
    active_half_edge = (k_half[None, :] < bot_e[:, None]).astype(u_3d.dtype)
    A_v_edge = jnp.where(active_half_edge > 0.5, config.A_v, 0.0)

    # We need to add KPP viscosity to be consistent with what the implicit
    # solver actually uses. Build the KPP profiles.
    model = MPASOceanModel(mesh, pc_coord, config)
    if model._kpp_profiles_fn is not None:
        print("    Adding KPP viscosity profiles to A_v...")
        A_v_kpp_cells, K_v_kpp_cells = model._kpp_profiles_fn(
            state, mesh, pc_coord, None,
        )
        A_v_kpp_edge = 0.5 * (A_v_kpp_cells[c1] + A_v_kpp_cells[c2])
        A_v_kpp_edge = A_v_kpp_edge * active_half_edge
        A_v_edge = A_v_edge + A_v_kpp_edge
        print(f"    max KPP A_v at edges = {float(jnp.max(A_v_kpp_edge)):.4e}")

    # Also add convective-adjustment viscosity if present
    if model._conv_config is not None:
        print("    Note: convective adjustment K is tracer-only (no A_v).")

    # Compute explicit diffusion: d/dz(A_v * du_prime/dz)
    # This mirrors the forward-Euler version of the implicit solver.
    # Build the tendency using the same stencil as the implicit solver:
    #   flux_{k+1/2} = A_v * (u[k] - u[k+1]) / dz_half
    #   tendency[k] = (flux_{k-1/2} - flux_{k+1/2}) / dz[k]
    _dzh_safe = jnp.maximum(dz_half_edge, 1e-10)
    _flux_Av = A_v_edge * (u_prime[:, :-1] - u_prime[:, 1:]) / _dzh_safe
    _flux_above_Av = jnp.pad(_flux_Av, ((0, 0), (1, 0)))
    _flux_below_Av = jnp.pad(_flux_Av, ((0, 0), (0, 1)))
    _dz_safe = jnp.maximum(dz_edge, 1e-10)
    explicit_Av = ((_flux_above_Av - _flux_below_Av) / _dz_safe) * edge_mask_3d
    print(f"    max|explicit_A_v| = {float(jnp.max(jnp.abs(explicit_Av))):.4e}")

    # A5: Approximate total tendency for the FULL velocity (not just perturbation)
    # The baroclinic step updates u via:
    #   u_star = u + dt * du_dt_3d
    # Then implicit A_v, then Coriolis on u_prime.
    # The full 3D tendency (including barotropic PGF from the barotropic solver)
    # would be:
    #   du_full_3d = du_dt_3d + F_slow_u  (reconstruction of du_dt_full from PE func)
    #             + coriolis_u_prime  (from Matsuno step)
    #             + explicit_Av  (forward-Euler approx of implicit)
    #             - g_grad_eta  (barotropic PGF, applied via barotropic solver)
    #
    # But the barotropic solver also handles:
    #   - Coriolis on u_bar (f * v_t(u_bar)) — online in substeps
    #   - g * grad(eta_new) — the implicit PGF on the updated eta
    #
    # The cleanest way to think about this:
    # du_dt_full_3d_from_PE = du_dt_3d + F_slow_u[:, None]
    #   (this is the TOTAL tendency from mpas_ocean_baroclinic_tendencies,
    #    before removing the depth mean; F_slow_u IS the depth mean)
    #
    # So: du_dt_full_from_PE = du_dt_3d + F_slow_u[:, None]
    #     = PGF + PV_flux + visc + vert_adv + drag + wind + restoring
    #     (everything EXCEPT Coriolis and A_v)
    #
    # Then the missing terms are:
    #   1. Coriolis on u_prime: f * v_t(u_prime)
    #   2. Explicit A_v: d/dz(A_v du'/dz) [forward-Euler approx]
    #   3. Coriolis on u_bar: f * v_t(u_bar) [from barotropic solver]
    #   4. Barotropic PGF: -g * grad(eta) [from barotropic solver]
    #
    # For the APPROXIMATE budget, we sum:
    #   approx_du_dt = du_dt_full_from_PE + coriolis_total + explicit_Av
    # where coriolis_total = coriolis_u_prime + coriolis_u_bar (depth-uniform)

    # Coriolis on u_bar
    v_t_bar = tangential_velocity_3d(
        u_bar[:, jnp.newaxis] * jnp.ones((1, nlev)), mesh,
    )
    coriolis_u_bar = f_e * v_t_bar * edge_mask_3d  # (nEdges, nlev)
    print(f"    max|f*v_t(u_bar)| = {float(jnp.max(jnp.abs(coriolis_u_bar))):.4e}")

    # Total Coriolis
    coriolis_total = coriolis_u_prime + coriolis_u_bar

    # Full PE tendency (before depth-mean removal)
    du_dt_full_from_PE = du_dt_3d + F_slow_u[:, jnp.newaxis]

    # Approximate total: PE + Coriolis + explicit_Av - g*grad(eta)
    # Note: g*grad(eta) enters with a MINUS sign in the momentum equation
    # (it's a restoring force), but the barotropic solver applies it
    # implicitly. In the budget, the barotropic solver produces:
    #   u_bar_new - u_bar_old = dt * (F_slow_u + f*v_t(u_bar) - g*grad(eta))
    # So the barotropic contribution to du/dt is:
    #   F_slow_u + f*v_t(u_bar) - g*grad(eta)
    # And the baroclinic contribution is:
    #   du_dt_3d + f*v_t(u_prime) + A_v
    approx_du_dt = (
        du_dt_full_from_PE  # = du_dt_3d + F_slow_u (PGF+PV+visc+vert_adv+drag+wind)
        + coriolis_total  # f*v_t(u_prime) + f*v_t(u_bar)
        + explicit_Av  # forward-Euler A_v
        - g_grad_eta  # barotropic PGF
    )

    # ====================================================================
    # PART B: EXACT BUDGET (one model step)
    # ====================================================================
    print("\n" + "=" * 80)
    print("PART B: EXACT BUDGET (one full model step)")
    print("=" * 80)

    print("  Running one model step (dt=300s)...")
    t0 = time.time()
    state_new = model.step(state, dt=DT)
    t1 = time.time()
    print(f"    Step completed in {t1 - t0:.1f}s")

    u_old = state.u.data
    u_new = state_new.u.data
    exact_du_dt = (u_new - u_old) / DT
    print(f"    max|exact_du_dt| = {float(jnp.max(jnp.abs(exact_du_dt))):.4e}")
    print(f"    max|u_new - u_old| = {float(jnp.max(jnp.abs(u_new - u_old))):.4e}")

    # ====================================================================
    # PART C: COMPARISON
    # ====================================================================
    print("\n" + "=" * 80)
    print("PART C: BUDGET COMPARISON BY REGIME")
    print("=" * 80)

    # Residuals
    approx_residual = approx_du_dt - exact_du_dt
    implicit_correction = exact_du_dt - approx_du_dt  # = -(approx - exact)

    # Define regimes by edge latitude
    lat_e = np.degrees(np.asarray(mesh.latEdge))
    c0_np = np.asarray(c1)
    c1_np = np.asarray(c2)
    ocean_mask_np = np.asarray(ocean_mask_new)
    ocean_edge = (ocean_mask_np[c0_np] > 0.5) & (ocean_mask_np[c1_np] > 0.5)

    equatorial = ocean_edge & (np.abs(lat_e) < 10)
    subtropical = ocean_edge & (np.abs(lat_e) >= 20) & (np.abs(lat_e) < 40)
    subpolar = ocean_edge & (np.abs(lat_e) >= 40) & (np.abs(lat_e) < 60)
    acc = ocean_edge & (lat_e < -55) & (lat_e > -70)

    regimes = [
        ("Equatorial", equatorial),
        ("Subtropical", subtropical),
        ("Subpolar", subpolar),
        ("ACC", acc),
        ("Global ocean", ocean_edge),
    ]

    # Components for budget table
    # Convert all to numpy for the regime-masked RMS
    components_3d = {
        "PE tendency": np.asarray(du_dt_full_from_PE),
        "Coriolis(u')": np.asarray(coriolis_u_prime),
        "Coriolis(ubar)": np.asarray(coriolis_u_bar),
        "g*grad(eta)": np.asarray(g_grad_eta),
        "Expl A_v": np.asarray(explicit_Av),
        "Approx sum": np.asarray(approx_du_dt),
        "Exact du/dt": np.asarray(exact_du_dt),
        "Approx resid": np.asarray(approx_residual),
        "Impl correct": np.asarray(implicit_correction),
    }

    # ---- Table 1: RMS of each term at surface (k=0) ----
    print("\n  TABLE 1: RMS at surface (k=0) [m/s^2]")
    print("  " + "-" * 130)
    header = f"  {'Regime':>15}"
    for name in components_3d:
        header += f" {name:>14}"
    print(header)
    print("  " + "-" * 130)

    for regime_name, mask_r in regimes:
        row = f"  {regime_name:>15}"
        for name, arr in components_3d.items():
            val = rms(arr[:, 0], mask_r)
            row += f" {val:14.4e}"
        print(row)

    # ---- Table 2: RMS across all levels ----
    print(f"\n  TABLE 2: RMS across all active levels [m/s^2]")
    print("  " + "-" * 130)
    print(header)
    print("  " + "-" * 130)

    for regime_name, mask_r in regimes:
        row = f"  {regime_name:>15}"
        for name, arr in components_3d.items():
            val = rms_3d(arr, mask_r)
            row += f" {val:14.4e}"
        print(row)

    # ---- Table 3: Decompose the PE tendency further ----
    # Run "no_*" configs to isolate individual terms within du_dt_full_from_PE
    print("\n\n  Decomposing PE tendency into individual terms...")

    # Physics without wind
    physics_no_wind = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="restoring",
            restoring=RestoringConfig(
                tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0,
                S_star=35.0, T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )
    physics_fn_no_wind = make_mpas_ocean_physics(
        physics_no_wind, implicit_vertical_mixing=True,
    )

    # Compute tendencies with different configs to isolate
    decomp_configs = {
        "no_wind": (config, physics_fn_no_wind),
        "no_pgf": (config._replace(pgf_scheme="zero"), physics_fn),
        "no_Ah": (config._replace(A_h=0.0), physics_fn),
        "no_smag": (config._replace(C_smag=0.0), physics_fn),
        "no_kzeta": (config._replace(K_zeta_bih=0.0), physics_fn),
        "no_drag": (config._replace(bottom_drag_r=0.0), physics_fn),
    }

    decomp_tends = {}
    for name, (cfg, pfn) in decomp_configs.items():
        t0 = time.time()
        t_d = mpas_ocean_baroclinic_tendencies(
            state, mesh, pc_coord, cfg,
            physics_fn=pfn, surface_forcing=None,
        )
        decomp_tends[name] = t_d
        dt_s = time.time() - t0
        print(f"    {name:>10}: {dt_s:.1f}s")

    # Extract full (du_dt + F_slow) for each config
    full_total = np.asarray(du_dt_full_from_PE)  # already computed
    decomp_full = {}
    for name, td in decomp_tends.items():
        decomp_full[name] = np.asarray(td.du_dt.data + td.F_slow_u.data[:, jnp.newaxis])

    # Differences (term = full - no_term)
    pe_components = {
        "PGF": full_total - decomp_full["no_pgf"],
        "Wind": full_total - decomp_full["no_wind"],
        "A_h": full_total - decomp_full["no_Ah"],
        "Smag": full_total - decomp_full["no_smag"],
        "K_zeta": full_total - decomp_full["no_kzeta"],
        "Drag": full_total - decomp_full["no_drag"],
    }
    pe_components["Other(PV+KE+vadv)"] = (
        full_total
        - pe_components["PGF"]
        - pe_components["Wind"]
        - pe_components["A_h"]
        - pe_components["Smag"]
        - pe_components["K_zeta"]
        - pe_components["Drag"]
    )

    print(f"\n  TABLE 3: PE tendency decomposition — RMS at surface (k=0) [m/s^2]")
    print("  " + "-" * 120)
    header3 = f"  {'Regime':>15}"
    for name in pe_components:
        header3 += f" {name:>14}"
    header3 += f" {'Total PE':>14}"
    print(header3)
    print("  " + "-" * 120)

    for regime_name, mask_r in regimes:
        row = f"  {regime_name:>15}"
        for name, arr in pe_components.items():
            val = rms(arr[:, 0], mask_r)
            row += f" {val:14.4e}"
        val = rms(full_total[:, 0], mask_r)
        row += f" {val:14.4e}"
        print(row)

    print(f"\n  TABLE 4: PE tendency decomposition — RMS all levels [m/s^2]")
    print("  " + "-" * 120)
    print(header3)
    print("  " + "-" * 120)

    for regime_name, mask_r in regimes:
        row = f"  {regime_name:>15}"
        for name, arr in pe_components.items():
            val = rms_3d(arr, mask_r)
            row += f" {val:14.4e}"
        val = rms_3d(full_total, mask_r)
        row += f" {val:14.4e}"
        print(row)

    # ---- Summary statistics ----
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (global ocean, all levels)")
    print("=" * 80)

    all_terms = {
        "PE tendency (du_dt_3d + F_slow)": du_dt_full_from_PE,
        "Coriolis on u_prime [f*v_t(u')]": coriolis_u_prime,
        "Coriolis on u_bar [f*v_t(ubar)]": coriolis_u_bar,
        "Barotropic PGF [-g*grad(eta)]": -g_grad_eta,
        "Explicit A_v [FE approx]": explicit_Av,
        "APPROXIMATE sum": approx_du_dt,
        "EXACT du/dt = (u_new-u_old)/dt": exact_du_dt,
        "Approx closure residual": approx_residual,
        "Implicit correction (exact-approx)": implicit_correction,
    }

    print(f"\n  {'Term':>45}   {'RMS':>12}   {'Max|x|':>12}")
    print("  " + "-" * 75)
    for name, arr in all_terms.items():
        arr_np = np.asarray(arr)
        vals_ocean = arr_np[ocean_edge]
        r = float(np.sqrt(np.mean(vals_ocean ** 2)))
        m = float(np.max(np.abs(vals_ocean)))
        print(f"  {name:>45}   {r:12.4e}   {m:12.4e}")

    # ---- Closure quality metrics ----
    print("\n" + "=" * 80)
    print("CLOSURE QUALITY")
    print("=" * 80)

    exact_rms = float(np.sqrt(
        np.mean(np.asarray(exact_du_dt)[ocean_edge] ** 2)
    ))
    approx_resid_rms = float(np.sqrt(
        np.mean(np.asarray(approx_residual)[ocean_edge] ** 2)
    ))
    impl_corr_rms = float(np.sqrt(
        np.mean(np.asarray(implicit_correction)[ocean_edge] ** 2)
    ))

    print(f"\n  RMS(exact du/dt):           {exact_rms:.6e}")
    print(f"  RMS(approx residual):       {approx_resid_rms:.6e}")
    print(f"  RMS(implicit correction):   {impl_corr_rms:.6e}")
    print(f"  Relative closure error:     {approx_resid_rms / max(exact_rms, 1e-30):.6e}")
    print(f"  Implicit/exact ratio:       {impl_corr_rms / max(exact_rms, 1e-30):.6e}")

    # ---- Where is the implicit correction largest? ----
    print("\n" + "=" * 80)
    print("WHERE IS THE IMPLICIT CORRECTION LARGEST?")
    print("=" * 80)

    impl_abs = np.abs(np.asarray(implicit_correction))
    impl_max_per_edge = np.max(impl_abs, axis=1)
    top_edges = np.argsort(impl_max_per_edge)[-10:][::-1]

    lon_e_deg = np.degrees(np.asarray(mesh.lonEdge))
    lat_e_deg = np.degrees(np.asarray(mesh.latEdge))

    print(f"\n  {'Rank':>4} {'Lon':>6} {'Lat':>6} {'|impl_corr|':>12} {'|exact|':>12} "
          f"{'|approx|':>12} {'|u|':>10} {'Level':>5}")
    print("  " + "-" * 75)
    for i, eidx in enumerate(top_edges):
        lev = int(np.argmax(impl_abs[eidx]))
        ic = float(impl_abs[eidx, lev])
        ex = float(np.abs(np.asarray(exact_du_dt)[eidx, lev]))
        ap = float(np.abs(np.asarray(approx_du_dt)[eidx, lev]))
        u_mag = float(np.abs(np.asarray(u_3d)[eidx, lev]))
        print(f"  {i + 1:4d} {lon_e_deg[eidx]:6.1f} {lat_e_deg[eidx]:6.1f} "
              f"{ic:12.4e} {ex:12.4e} {ap:12.4e} {u_mag:10.4f} {lev:5d}")

    # ---- Closure by level ----
    print("\n" + "=" * 80)
    print("CLOSURE BY VERTICAL LEVEL (global ocean)")
    print("=" * 80)
    print(f"\n  {'Level':>5} {'RMS exact':>12} {'RMS approx':>12} {'RMS resid':>12} "
          f"{'RMS impl_c':>12} {'Rel err':>10}")
    print("  " + "-" * 70)

    for k in range(nlev):
        exact_k = np.asarray(exact_du_dt[:, k])[ocean_edge]
        approx_k = np.asarray(approx_du_dt[:, k])[ocean_edge]
        resid_k = np.asarray(approx_residual[:, k])[ocean_edge]
        impl_k = np.asarray(implicit_correction[:, k])[ocean_edge]

        rms_ex = float(np.sqrt(np.mean(exact_k ** 2)))
        rms_ap = float(np.sqrt(np.mean(approx_k ** 2)))
        rms_re = float(np.sqrt(np.mean(resid_k ** 2)))
        rms_im = float(np.sqrt(np.mean(impl_k ** 2)))
        rel = rms_re / max(rms_ex, 1e-30)

        print(f"  {k:5d} {rms_ex:12.4e} {rms_ap:12.4e} {rms_re:12.4e} "
              f"{rms_im:12.4e} {rel:10.4f}")

    wall_total = time.time() - t_wall_start
    print(f"\nTotal wall time: {wall_total / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
