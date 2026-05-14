"""End-to-end smoke test for DINO on the lat-lon Mercator grid.

Builds the DINO basin on a Mercator lat-lon grid using:
  - create_mercator_grid (Mercator-PR #262)
  - dino_bathymetry (Phase 2B)
  - dino_initial_T_S (Phase 2D)
  - seam-wall land mask (analogous to MPAS Phase 3)
  - LatLonCGridOceanModel with A_h_lat_scaling=True (Phase 1B,
    free in legoESM via cos(lat) scaling — see plan)

Then runs a single timestep from rest. Verifies no NaN, |u|, |eta|
small, T/S stay in physical range.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
    dino_lat_lon_grid,
    dino_lat_lon_initial_state_arrays,
    dino_lat_lon_model_config,
    dino_lat_lon_state,
)


# Coarser resolution for fast tests: 25 zonal cells = 2° equatorial spacing.
TEST_N_LON = 25


@pytest.fixture(scope="module")
def cfg():
    return DINOConfig()


@pytest.fixture(scope="module")
def grid(cfg):
    return dino_lat_lon_grid(cfg, n_lon=TEST_N_LON)


@pytest.fixture(scope="module")
def z_coord(cfg):
    return create_dino_z_star(cfg)


@pytest.fixture(scope="module")
def state(grid, z_coord, cfg):
    return dino_lat_lon_state(grid, z_coord, cfg)


# -------------------- grid construction -----------------------------

def test_grid_dimensions_match_dino_domain(grid, cfg):
    # Mercator grid should cover ±70° latitude with n_lon zonal cells
    assert grid.n_lon == TEST_N_LON
    # n_lat depends on lat_max_deg via the Mercator placement formula
    assert grid.n_lat > 0
    # Latitudes within [-70°, 70°]
    lat_deg = np.degrees(np.asarray(grid.lat))
    assert lat_deg.min() >= -cfg.lat_max_deg - 1.0
    assert lat_deg.max() <= cfg.lat_max_deg + 1.0


def test_grid_dx_dy_isotropic(grid):
    """Mercator: dx(j) ≈ dy(j) per row (the whole point)."""
    dx = np.asarray(grid.dx)  # (n_lat, n_lon)
    dy = np.asarray(grid.dy)  # (n_lat,)
    # Compare per row
    rel_err = np.abs(dx[:, 0] - dy) / dy
    assert rel_err.max() < 0.05  # within 5%


# -------------------- initial-state arrays --------------------------

def test_initial_state_arrays_shapes(grid, z_coord, cfg):
    T, S, H_bathy, mask = dino_lat_lon_initial_state_arrays(grid, z_coord, cfg)
    assert T.shape == (grid.n_lat, grid.n_lon, cfg.n_levels)
    assert S.shape == T.shape
    assert H_bathy.shape == (grid.n_lat, grid.n_lon)
    assert mask.shape == (grid.n_lat, grid.n_lon)


def test_seam_wall_outside_channel(grid, z_coord, cfg):
    """Westernmost lon column should be land outside channel band."""
    _, _, _, mask = dino_lat_lon_initial_state_arrays(grid, z_coord, cfg)
    mask_np = np.asarray(mask)
    lat_deg = np.degrees(np.asarray(grid.lat))
    in_channel = (
        (lat_deg >= cfg.channel_lat_south_deg)
        & (lat_deg <= cfg.channel_lat_north_deg)
    )
    # i=0 column outside channel = land
    assert (mask_np[~in_channel, 0] == 0.0).all()
    # i=0 column INSIDE channel = ocean
    if in_channel.any():
        assert (mask_np[in_channel, 0] == 1.0).all()


def test_basin_interior_is_ocean(grid, z_coord, cfg):
    _, _, _, mask = dino_lat_lon_initial_state_arrays(grid, z_coord, cfg)
    mask_np = np.asarray(mask)
    # Middle column should be all ocean
    middle = grid.n_lon // 2
    assert (mask_np[:, middle] == 1.0).all()


def test_T_S_finite_on_ocean(grid, z_coord, cfg):
    T, S, _, mask = dino_lat_lon_initial_state_arrays(grid, z_coord, cfg)
    mask_np = np.asarray(mask) > 0.5
    if mask_np.any():
        T_ocean = np.asarray(T)[mask_np]
        S_ocean = np.asarray(S)[mask_np]
        assert np.isfinite(T_ocean).all()
        assert np.isfinite(S_ocean).all()
        assert T_ocean.min() > -3.0
        assert T_ocean.max() < 30.0
        assert S_ocean.min() > 30.0
        assert S_ocean.max() < 40.0


# -------------------- model + config -------------------------------

def test_model_config_uses_dino_values(grid, cfg):
    model_cfg, physics_cfg = dino_lat_lon_model_config(grid, cfg, physics=True)
    assert model_cfg.rho_0 == cfg.rho_0
    assert model_cfg.A_v == cfg.A_v_bg
    assert model_cfg.K_v == cfg.K_v_bg
    assert model_cfg.barotropic_solver == cfg.barotropic_solver
    assert model_cfg.tracer_advection == cfg.tracer_advection
    assert model_cfg.pgf_scheme == cfg.pgf_scheme
    assert model_cfg.eos == "wright"
    assert model_cfg.A_h_lat_scaling is True  # Phase 1B!
    assert physics_cfg is not None


def test_A_h_base_matches_DINO_formula(grid, cfg):
    """Verify A_h_base = 0.5·U_M·R·Δλ so that A_h(j)=A_h_base·cos(φ)
    equals 0.5·U_M·dx(j) for Mercator dx(j)=R·cos(φ)·Δλ."""
    model_cfg, _ = dino_lat_lon_model_config(grid, cfg, physics=True)
    expected_A_h_base = 0.5 * cfg.U_M * grid.radius * grid.dlon
    assert model_cfg.A_h == pytest.approx(expected_A_h_base, rel=1e-9)


def test_K_h_base_matches_DINO_formula(grid, cfg):
    model_cfg, _ = dino_lat_lon_model_config(grid, cfg, physics=True)
    expected_K_h_base = 0.5 * cfg.U_T * grid.radius * grid.dlon
    assert model_cfg.K_h == pytest.approx(expected_K_h_base, rel=1e-9)


def test_dycore_only_returns_no_physics(grid, cfg):
    model_cfg, physics_cfg = dino_lat_lon_model_config(grid, cfg, physics=False)
    assert model_cfg.physics is None
    assert physics_cfg is None


# -------------------- one-step smoke ---------------------------------

def test_state_starts_from_rest(state):
    assert float(jnp.max(jnp.abs(state.u.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.v.data))) == 0.0
    assert float(jnp.max(jnp.abs(state.eta.data))) == 0.0


def test_dycore_only_one_step_from_rest(grid, z_coord, state, cfg):
    """Build dycore-only model, run ONE step from rest. State should
    not NaN; |u|, |v| < 0.5 m/s; |eta| < 1 m."""
    model_cfg, _ = dino_lat_lon_model_config(grid, cfg, physics=False)
    model = LatLonCGridOceanModel(grid, z_coord, model_cfg)
    new_state = model.step(state, dt=cfg.dt)

    for fld_name in ("u", "v", "T", "S", "eta"):
        fld = getattr(new_state, fld_name)
        assert bool(jnp.all(jnp.isfinite(fld.data))), f"NaN in {fld_name}"

    u_max = float(jnp.max(jnp.abs(new_state.u.data)))
    v_max = float(jnp.max(jnp.abs(new_state.v.data)))
    eta_max = float(jnp.max(jnp.abs(new_state.eta.data)))
    assert u_max < 0.5, f"Step-1 u_max={u_max} m/s — instability?"
    assert v_max < 0.5, f"Step-1 v_max={v_max} m/s — instability?"
    assert eta_max < 1.0, f"Step-1 eta_max={eta_max} m — barotropic blowup?"
