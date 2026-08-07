"""Land-only source regrid (CRU-JRA coastal init-NaN fix).

CRU-JRA is land-only (ocean = NaN).  A model column whose nearest source cells are
all ocean regridded to NaN -> a NaN cold-start -> that column NaN'd for the whole
run (the mid-latitude/coastal chronic-NaN cells).  Building the KD-tree from LAND
source cells only makes every column draw its nearest ACTUAL land forcing.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from legoesm.grids.regridding import (
    compute_latlon_to_voronoi_weights,
    regrid_scalar_nan_aware,
)


def _setup():
    nlat, nlon = 8, 8
    src_lat = np.radians(np.linspace(-70.0, 70.0, nlat))
    src_lon = np.radians(np.linspace(0.0, 315.0, nlon))
    field = np.full((nlat, nlon), np.nan)        # ocean = NaN
    field[1, 1] = 280.0                            # two isolated land cells
    field[6, 6] = 300.0
    land = np.isfinite(field)
    # a target in the far corner, surrounded by ocean source cells
    tgt_lat = np.radians(np.array([-70.0]))
    tgt_lon = np.radians(np.array([0.0]))
    return src_lat, src_lon, field, land, tgt_lat, tgt_lon


def test_all_ocean_neighbours_give_nan_without_mask():
    src_lat, src_lon, field, land, tgt_lat, tgt_lon = _setup()
    w = compute_latlon_to_voronoi_weights(src_lat, src_lon, tgt_lat, tgt_lon, k_neighbors=4)
    out = np.asarray(regrid_scalar_nan_aware(jnp.asarray(field[..., None]), w))
    assert np.isnan(out).all()                    # the bug: all-ocean neighbours -> NaN


def test_src_valid_gives_finite_land_forcing():
    src_lat, src_lon, field, land, tgt_lat, tgt_lon = _setup()
    w = compute_latlon_to_voronoi_weights(
        src_lat, src_lon, tgt_lat, tgt_lon, k_neighbors=4, src_valid=land)
    out = np.asarray(regrid_scalar_nan_aware(jnp.asarray(field[..., None]), w))
    assert np.all(np.isfinite(out))               # the fix: nearest LAND, finite
    assert 279.0 < float(out.ravel()[0]) < 301.0  # drawn from the two land cells


def test_src_indices_point_only_to_land_cells():
    """With src_valid, every neighbour index must reference a finite (land) cell."""
    src_lat, src_lon, field, land, tgt_lat, tgt_lon = _setup()
    w = compute_latlon_to_voronoi_weights(
        src_lat, src_lon, tgt_lat, tgt_lon, k_neighbors=4, src_valid=land)
    flat = field.ravel()
    idx = np.asarray(w.src_indices).ravel()
    assert np.all(np.isfinite(flat[idx]))
