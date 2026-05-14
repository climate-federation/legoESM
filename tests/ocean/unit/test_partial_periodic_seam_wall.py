"""Unit tests for the partial-periodic seam-wall mask helpers in
``legoesm.ocean.init_mpas`` and ``legoesm.ocean.init_latlon_cgrid``
(promoted from DINO 2026-05-14).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_mercator_grid
from legoesm.grids.voronoi import create_regional_voronoi_mesh
from legoesm.ocean.init_latlon_cgrid import partial_periodic_seam_wall_latlon
from legoesm.ocean.init_mpas import partial_periodic_seam_wall_mpas


# ---------- MPAS path -----------------------------------------------

@pytest.fixture(scope="module")
def mpas_mesh():
    """Regional Voronoi mesh covering [-50°, 0°] × [-70°, 70°]."""
    return create_regional_voronoi_mesh(
        lon_range=(-50.0, 0.0),
        lat_range=(-70.0, 70.0),
        resolution_km=220.0,
        periodic_x=True,
    )


def test_mpas_seam_wall_outside_open_band_is_land(mpas_mesh):
    mask = np.asarray(partial_periodic_seam_wall_mpas(
        mpas_mesh,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
    ))
    lon = (np.degrees(np.asarray(mpas_mesh.lonCell)) + 180.0) % 360.0 - 180.0
    lat = np.degrees(np.asarray(mpas_mesh.latCell))
    # Near the seam (lon ≈ -50, within 1°), well outside the band → land
    near_seam = (lon - (-50.0)) < 1.0
    outside_band = (lat > -40) | (lat < -70)
    candidates = near_seam & outside_band
    if candidates.any():
        assert (mask[candidates] == 0.0).any()


def test_mpas_seam_inside_open_band_is_ocean(mpas_mesh):
    mask = np.asarray(partial_periodic_seam_wall_mpas(
        mpas_mesh,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
    ))
    lon = (np.degrees(np.asarray(mpas_mesh.lonCell)) + 180.0) % 360.0 - 180.0
    lat = np.degrees(np.asarray(mpas_mesh.latCell))
    near_seam = (lon - (-50.0)) < 1.0
    in_band = (lat >= -65.0) & (lat <= -45.0)
    candidates = near_seam & in_band
    if candidates.any():
        assert (mask[candidates] == 1.0).all()


def test_mpas_interior_far_from_seam_is_ocean(mpas_mesh):
    mask = np.asarray(partial_periodic_seam_wall_mpas(
        mpas_mesh,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
    ))
    lon = (np.degrees(np.asarray(mpas_mesh.lonCell)) + 180.0) % 360.0 - 180.0
    lat = np.degrees(np.asarray(mpas_mesh.latCell))
    interior = (lon > -40) & (lon < -10) & (lat > -50) & (lat < 50)
    if interior.any():
        assert (mask[interior] == 1.0).all()


def test_mpas_seam_strip_width_param(mpas_mesh):
    """Wider strip should make more land cells."""
    m_narrow = np.asarray(partial_periodic_seam_wall_mpas(
        mpas_mesh, -65.0, -45.0, seam_strip_width_deg=1.0,
    ))
    m_wide = np.asarray(partial_periodic_seam_wall_mpas(
        mpas_mesh, -65.0, -45.0, seam_strip_width_deg=5.0,
    ))
    assert (m_wide == 0.0).sum() >= (m_narrow == 0.0).sum()


def test_mpas_base_mask_combined(mpas_mesh):
    """When a base mask is provided, cells already land stay land."""
    # Pre-mask the northern half as land
    lat = jnp.degrees(mpas_mesh.latCell)
    base = (lat <= 0.0).astype(jnp.float32)
    out = np.asarray(partial_periodic_seam_wall_mpas(
        mpas_mesh,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
        base_mask=base,
    ))
    lat_np = np.degrees(np.asarray(mpas_mesh.latCell))
    # Anywhere north of equator must be land
    assert (out[lat_np > 0.0] == 0.0).all()


# ---------- Lat-lon path --------------------------------------------

@pytest.fixture(scope="module")
def lat_lon_grid():
    return create_mercator_grid(
        n_lon=25, lat_max_deg=70.0, lon_west_deg=-50.0, lon_east_deg=0.0,
    )


def test_latlon_seam_outside_band_is_land(lat_lon_grid):
    mask = np.asarray(partial_periodic_seam_wall_latlon(
        lat_lon_grid,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
    ))
    lat_deg = np.degrees(np.asarray(lat_lon_grid.lat))
    outside_band = (lat_deg > -45.0) | (lat_deg < -65.0)
    # Column 0 outside the band must be land
    assert (mask[outside_band, 0] == 0.0).all()


def test_latlon_seam_inside_band_is_ocean(lat_lon_grid):
    mask = np.asarray(partial_periodic_seam_wall_latlon(
        lat_lon_grid,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
    ))
    lat_deg = np.degrees(np.asarray(lat_lon_grid.lat))
    in_band = (lat_deg >= -65.0) & (lat_deg <= -45.0)
    if in_band.any():
        # Column 0 inside the band must be ocean
        assert (mask[in_band, 0] == 1.0).all()


def test_latlon_interior_columns_all_ocean(lat_lon_grid):
    mask = np.asarray(partial_periodic_seam_wall_latlon(
        lat_lon_grid,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
    ))
    # Columns 1..n_lon-1 are all ocean (only column 0 is the wall)
    assert (mask[:, 1:] == 1.0).all()


def test_latlon_seam_column_index_param(lat_lon_grid):
    """Picking a different column should put the wall there."""
    mask = np.asarray(partial_periodic_seam_wall_latlon(
        lat_lon_grid,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
        seam_column_index=12,
    ))
    lat_deg = np.degrees(np.asarray(lat_lon_grid.lat))
    outside_band = (lat_deg > -45.0) | (lat_deg < -65.0)
    assert (mask[outside_band, 12] == 0.0).all()
    # Column 0 is now ocean
    assert (mask[:, 0] == 1.0).all()


def test_latlon_base_mask_intersection(lat_lon_grid):
    """Cells already land in base_mask stay land."""
    base = jnp.ones((lat_lon_grid.n_lat, lat_lon_grid.n_lon))
    base = base.at[0, :].set(0.0)  # bottom row entirely land
    out = np.asarray(partial_periodic_seam_wall_latlon(
        lat_lon_grid,
        open_lat_south_deg=-65.0,
        open_lat_north_deg=-45.0,
        base_mask=base,
    ))
    assert (out[0, :] == 0.0).all()
