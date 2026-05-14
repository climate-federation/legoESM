"""Unit tests for DINO MPAS regional-mesh wiring (Phase 3).

The MPAS path uses a regional spherical Voronoi mesh with
``periodic_x=True`` covering the basin's longitude range. To get the
DINO topology (closed walls everywhere except the channel band), a
LAND MASK is applied: cells within one cell-width of the periodic
seam AND outside the channel band are land.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.voronoi import create_regional_voronoi_mesh
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
    dino_bathymetry,
    dino_mpas_initial_state_arrays,
    dino_mpas_land_mask,
)


# Reduce mesh size for faster tests; 220 km ≈ 2° resolution.
TEST_RESOLUTION_KM = 220.0


def _make_test_mesh(cfg: DINOConfig | None = None):
    if cfg is None:
        cfg = DINOConfig()
    return create_regional_voronoi_mesh(
        lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
        lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
        resolution_km=TEST_RESOLUTION_KM,
        periodic_x=True,
    )


# -------------------- mesh creation ---------------------------------

def test_mesh_can_be_created_with_dino_domain():
    mesh = _make_test_mesh()
    assert mesh.nCells > 100
    assert mesh.nEdges > 100


def test_mesh_lon_in_basin_range():
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    # mesh stores lon in [0, 2π); wrap to (-180, 180] to compare with cfg
    lon = (np.degrees(np.asarray(mesh.lonCell)) + 180.0) % 360.0 - 180.0
    assert lon.min() >= cfg.lon_west_deg - 1e-3
    assert lon.max() <= cfg.lon_east_deg + 1e-3


# -------------------- land mask -------------------------------------

def test_land_mask_buffer_cells_at_lat_extremes():
    """Cells outside the lat band [-70, 70] should be land (buffer)."""
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    mask = np.asarray(dino_mpas_land_mask(mesh, cfg))
    lat = np.degrees(np.asarray(mesh.latCell))
    outside_band = (lat < -cfg.lat_max_deg) | (lat > cfg.lat_max_deg)
    if outside_band.any():
        assert (mask[outside_band] == 0.0).all()


def test_land_mask_seam_outside_channel_is_land():
    """Cells near the periodic seam, but outside the channel band,
    should be land (forming the wall)."""
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    mask = np.asarray(dino_mpas_land_mask(mesh, cfg))
    lon = (np.degrees(np.asarray(mesh.lonCell)) + 180.0) % 360.0 - 180.0
    lat = np.degrees(np.asarray(mesh.latCell))
    # Take cells on the western edge (within 1° of -50) at temperate
    # latitudes (well outside the channel)
    near_seam = lon - cfg.lon_west_deg < 1.0
    outside_channel = (lat > cfg.channel_lat_north_deg + 5) | (
        lat < cfg.channel_lat_south_deg - 5
    )
    candidates = near_seam & outside_channel
    if candidates.any():
        # At least some of these must be land (the wall)
        assert (mask[candidates] == 0.0).any()


def test_land_mask_seam_inside_channel_is_ocean():
    """Cells near the periodic seam, INSIDE the channel band,
    should be ocean — the channel is re-entrant."""
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    mask = np.asarray(dino_mpas_land_mask(mesh, cfg))
    lon = (np.degrees(np.asarray(mesh.lonCell)) + 180.0) % 360.0 - 180.0
    lat = np.degrees(np.asarray(mesh.latCell))
    near_seam = lon - cfg.lon_west_deg < 1.0
    in_channel = (
        (lat >= cfg.channel_lat_south_deg)
        & (lat <= cfg.channel_lat_north_deg)
    )
    candidates = near_seam & in_channel
    if candidates.any():
        # All candidates must be OCEAN (no wall in the channel)
        assert (mask[candidates] == 1.0).all()


def test_land_mask_basin_interior_is_ocean():
    """Cells in the deep basin interior (mid-lon, mid-lat) should be ocean."""
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    mask = np.asarray(dino_mpas_land_mask(mesh, cfg))
    lon = (np.degrees(np.asarray(mesh.lonCell)) + 180.0) % 360.0 - 180.0
    lat = np.degrees(np.asarray(mesh.latCell))
    interior = (
        (lon > cfg.lon_west_deg + 10) & (lon < cfg.lon_east_deg - 10)
        & (lat > -50) & (lat < 50)
    )
    if interior.any():
        assert (mask[interior] == 1.0).all()


def test_land_mask_seam_strip_width_param_controls_wall():
    """A wider strip should produce more land cells near the seam."""
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    mask_narrow = np.asarray(dino_mpas_land_mask(mesh, cfg, seam_strip_width_deg=1.0))
    mask_wide = np.asarray(dino_mpas_land_mask(mesh, cfg, seam_strip_width_deg=5.0))
    # Wider strip → more land
    assert (mask_wide == 0.0).sum() >= (mask_narrow == 0.0).sum()


def test_land_mask_dtype_and_shape():
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    mask = dino_mpas_land_mask(mesh, cfg)
    assert mask.shape == (mesh.nCells,)
    # Boolean-like values only
    unique = jnp.unique(mask)
    assert all(float(v) in (0.0, 1.0) for v in np.asarray(unique))


# -------------------- combined initial state ------------------------

def test_initial_state_arrays_shapes():
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    z = create_dino_z_star(cfg)
    T, S, H_bathy, land_mask = dino_mpas_initial_state_arrays(mesh, z, cfg)
    assert T.shape == (mesh.nCells, cfg.n_levels)
    assert S.shape == (mesh.nCells, cfg.n_levels)
    assert H_bathy.shape == (mesh.nCells,)
    assert land_mask.shape == (mesh.nCells,)


def test_initial_state_T_S_zero_on_land():
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    z = create_dino_z_star(cfg)
    T, S, H_bathy, land_mask = dino_mpas_initial_state_arrays(mesh, z, cfg)
    mask = np.asarray(land_mask)
    if (mask == 0.0).any():
        T_np = np.asarray(T)
        S_np = np.asarray(S)
        assert (T_np[mask == 0.0, :] == 0.0).all()
        assert (S_np[mask == 0.0, :] == 0.0).all()


def test_initial_state_T_S_finite_on_ocean():
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    z = create_dino_z_star(cfg)
    T, S, _, land_mask = dino_mpas_initial_state_arrays(mesh, z, cfg)
    mask = np.asarray(land_mask)
    T_np = np.asarray(T)
    S_np = np.asarray(S)
    if (mask == 1.0).any():
        assert np.isfinite(T_np[mask == 1.0]).all()
        assert np.isfinite(S_np[mask == 1.0]).all()


def test_initial_state_H_bathy_zero_on_land_else_in_basin_range():
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    z = create_dino_z_star(cfg)
    _, _, H_bathy, land_mask = dino_mpas_initial_state_arrays(mesh, z, cfg)
    H_np = np.asarray(H_bathy)
    mask = np.asarray(land_mask)
    if (mask == 0.0).any():
        assert (H_np[mask == 0.0] == 0.0).all()
    if (mask == 1.0).any():
        # Ocean cells: H_shallow ≤ H ≤ H_deep (sill cells can shoal to
        # H_sill but at the resolution of this test the sill is barely
        # resolved, so a generous lower bound of H_shallow - 1 m is
        # appropriate).
        assert (H_np[mask == 1.0] >= cfg.H_shallow - 1.0).all()
        assert (H_np[mask == 1.0] <= cfg.H_deep + 1.0).all()


def test_initial_state_T_at_equator_matches_1d_profile():
    """An ocean cell near the equator should have T close to the
    1D equatorial profile (no meridional gradient at equator)."""
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    z = create_dino_z_star(cfg)
    T, _, _, land_mask = dino_mpas_initial_state_arrays(mesh, z, cfg)
    lat = np.degrees(np.asarray(mesh.latCell))
    mask = np.asarray(land_mask)
    # Find the cell closest to lat=0 that is ocean
    candidates = (mask == 1.0) & (np.abs(lat) < 5)
    if candidates.any():
        idx = np.where(candidates)[0][np.argmin(np.abs(lat[candidates]))]
        # Surface T at the equator should be ~23.5°C (per the 1D profile)
        T_surface = float(T[idx, 0])
        assert 22.0 < T_surface < 25.0


def test_initial_state_T_at_pole_isothermal():
    """An ocean cell near the pole (within band) should have a
    near-isothermal column at the bottom value (~4°C)."""
    cfg = DINOConfig()
    mesh = _make_test_mesh(cfg)
    z = create_dino_z_star(cfg)
    T, _, _, land_mask = dino_mpas_initial_state_arrays(mesh, z, cfg)
    lat = np.degrees(np.asarray(mesh.latCell))
    mask = np.asarray(land_mask)
    candidates = (mask == 1.0) & (np.abs(lat) > 65)
    if candidates.any():
        idx = np.where(candidates)[0][np.argmax(np.abs(lat[candidates]))]
        T_col = np.asarray(T[idx, :])
        assert T_col.std() < 0.5
        assert 3.0 < T_col[0] < 5.0
