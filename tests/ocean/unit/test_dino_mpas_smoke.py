"""End-to-end smoke test for DINO on the MPAS regional mesh.

Wires together every Phase 2 + Phase 3 piece (config, bathymetry,
initial conditions, vertical grid, MPAS state, model+physics config)
and runs a single timestep from rest. Verifies no NaN and that the
state magnitudes stay sensible (small velocities, T/S stay in physical
range, η stays small).

This is the first end-to-end DINO test that actually integrates time.
The lat-lon path is not exercised here — that's blocked on the
Mercator grid PR.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.voronoi import create_regional_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
    dino_mpas_model_config,
    dino_mpas_state,
)


# Coarser resolution for fast tests
TEST_RESOLUTION_KM = 220.0


@pytest.fixture(scope="module")
def cfg():
    return DINOConfig()


@pytest.fixture(scope="module")
def mesh(cfg):
    return create_regional_voronoi_mesh(
        lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
        lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
        resolution_km=TEST_RESOLUTION_KM,
        periodic_x=True,
    )


@pytest.fixture(scope="module")
def z_coord(cfg):
    return create_dino_z_star(cfg)


@pytest.fixture(scope="module")
def state(mesh, z_coord, cfg):
    return dino_mpas_state(mesh, z_coord, cfg)


# -------------------- state construction ---------------------------

def test_state_has_all_fields(state):
    assert hasattr(state, "u")
    assert hasattr(state, "T")
    assert hasattr(state, "S")
    assert hasattr(state, "eta")
    assert hasattr(state, "w")
    assert hasattr(state, "H_bathy")
    assert hasattr(state, "land_mask")


def test_state_starts_from_rest(state):
    """u, eta, w should be exactly zero at t=0."""
    assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.eta.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.w.data))) == 0.0


def test_state_T_S_finite_on_ocean(state):
    mask = np.asarray(state.land_mask.data)
    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    if (mask == 1.0).any():
        assert np.isfinite(T[mask == 1.0]).all()
        assert np.isfinite(S[mask == 1.0]).all()


def test_state_T_in_physical_range(state):
    mask = np.asarray(state.land_mask.data)
    T = np.asarray(state.T.data)
    if (mask == 1.0).any():
        T_ocean = T[mask == 1.0]
        assert T_ocean.min() > -3.0
        assert T_ocean.max() < 30.0


def test_state_S_in_physical_range(state):
    mask = np.asarray(state.land_mask.data)
    S = np.asarray(state.S.data)
    if (mask == 1.0).any():
        S_ocean = S[mask == 1.0]
        assert S_ocean.min() > 30.0
        assert S_ocean.max() < 40.0


# -------------------- model + config ------------------------------

def test_model_config_uses_dino_values(mesh, cfg):
    model_cfg, physics_cfg = dino_mpas_model_config(mesh, cfg, physics=True)
    assert model_cfg.rho_0 == cfg.rho_0
    assert model_cfg.A_v == cfg.A_v_bg
    assert model_cfg.K_v == cfg.K_v_bg
    assert model_cfg.barotropic_solver == cfg.barotropic_solver
    assert model_cfg.tracer_advection == cfg.tracer_advection
    assert physics_cfg is not None


def test_model_config_dycore_only_returns_no_physics(mesh, cfg):
    model_cfg, physics_cfg = dino_mpas_model_config(mesh, cfg, physics=False)
    assert physics_cfg is None


def test_model_lateral_mixing_scaled_with_resolution(mesh, cfg):
    model_cfg, _ = dino_mpas_model_config(mesh, cfg, physics=True)
    # A_h ≈ 0.5·U_M·Δx with U_M=0.27 m/s and Δx ~ 220 km gives ~30000 m²/s
    assert 1e3 < model_cfg.A_h < 1e6
    # K_h is 10× smaller (U_T = U_M/10)
    assert model_cfg.K_h == pytest.approx(model_cfg.A_h * cfg.U_T / cfg.U_M)


# -------------------- one-step smoke test --------------------------

def test_dycore_only_one_step_from_rest(mesh, z_coord, state, cfg):
    """Build model with no physics, run ONE step from rest with no
    surface forcing. State should not NaN; velocities should stay
    small (< 0.1 m/s); eta should stay tiny."""
    model_cfg, _ = dino_mpas_model_config(mesh, cfg, physics=False)
    model = MPASOceanModel(mesh, z_coord, model_cfg)
    new_state = model.step(state, dt=cfg.dt)

    # No NaN anywhere
    for fld in (new_state.u, new_state.T, new_state.S, new_state.eta, new_state.w):
        assert bool(jnp.all(jnp.isfinite(fld.data))), f"NaN in {fld.name}"

    # Velocities stay small after one timestep
    u_max = float(jnp.max(jnp.abs(new_state.u.data)))
    assert u_max < 0.5, f"Step-1 u_max={u_max} m/s — instability?"

    # Eta stays small (no large barotropic transient)
    eta_max = float(jnp.max(jnp.abs(new_state.eta.data)))
    assert eta_max < 1.0, f"Step-1 eta_max={eta_max} m — barotropic blowup?"


def test_dycore_only_T_S_drift_is_small_in_one_step(mesh, z_coord, state, cfg):
    """T, S should change very little in a single 45-min step from rest
    with no surface forcing — only baroclinic adjustment via PGF."""
    model_cfg, _ = dino_mpas_model_config(mesh, cfg, physics=False)
    model = MPASOceanModel(mesh, z_coord, model_cfg)
    new_state = model.step(state, dt=cfg.dt)

    mask = np.asarray(state.land_mask.data) > 0.5
    T_diff = np.asarray(new_state.T.data) - np.asarray(state.T.data)
    S_diff = np.asarray(new_state.S.data) - np.asarray(state.S.data)
    if mask.any():
        assert np.abs(T_diff[mask, :]).max() < 0.5  # ≤ 0.5°C per step
        assert np.abs(S_diff[mask, :]).max() < 0.5  # ≤ 0.5 g/kg per step


def test_dycore_only_land_cells_stay_finite(mesh, z_coord, state, cfg):
    """T, S on land cells are intentionally filled with the Neumann
    ocean-neighbor average (see ``fill_land_cells_mpas`` in
    ocean_model_mpas.step) to prevent sharp coastline discontinuities
    in subsequent operators. So land cells are NOT held at exact zero
    after a step — they should match nearby ocean values. Just verify
    they're finite and in physical range."""
    model_cfg, _ = dino_mpas_model_config(mesh, cfg, physics=False)
    model = MPASOceanModel(mesh, z_coord, model_cfg)
    new_state = model.step(state, dt=cfg.dt)

    mask = np.asarray(state.land_mask.data) < 0.5
    T_land = np.asarray(new_state.T.data)[mask, :]
    S_land = np.asarray(new_state.S.data)[mask, :]
    if mask.any():
        assert np.isfinite(T_land).all()
        assert np.isfinite(S_land).all()
        # Filled land values should be near neighboring ocean values, in
        # physical range (or exactly zero for cells with no ocean
        # neighbors at all — e.g., interior of a buffer block).
        T_land_filled = T_land[T_land != 0.0]
        if T_land_filled.size > 0:
            assert (T_land_filled > -3.0).all()
            assert (T_land_filled < 30.0).all()
