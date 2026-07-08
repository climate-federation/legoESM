"""The synthetic land-sea mask generator (make_synthetic_land_mask.py).

Locks: the written NetCDF carries an `lsm(lat, lon)` field with a clear land/ocean split
that legoesm.grids.topography.load_land_fraction auto-detects + regrids to a model grid
yielding BOTH land and ocean columns (so --ocean-only has columns to exclude)."""
from __future__ import annotations

import numpy as np
import pytest


def test_dataset_has_a_clear_land_ocean_split():
    from scripts.data.make_synthetic_land_mask import build_synthetic_land_mask_dataset

    ds = build_synthetic_land_mask_dataset(nlat=24, nlon=48,
                                           land_lat_min=20.0, land_lat_max=60.0)
    assert ds["lsm"].dims == ("lat", "lon") and ds["lsm"].shape == (24, 48)
    lsm = np.asarray(ds["lsm"].values)
    assert lsm.min() == 0.0 and lsm.max() == 1.0          # both ocean AND land present
    # land ONLY in the 20-60N band (the loader handles lat ordering; the band must be NH)
    lat = np.asarray(ds["lat"].values)
    land_rows = lsm.max(axis=1) > 0.5
    assert np.all((lat[land_rows] >= 20.0) & (lat[land_rows] <= 60.0))

    with pytest.raises(ValueError, match="non-empty land band"):
        build_synthetic_land_mask_dataset(land_lat_min=60.0, land_lat_max=20.0)


def test_written_mask_loads_and_regrids_to_a_mixed_land_ocean_field(tmp_path):
    """The end-to-end contract the smoke relies on: the written file is readable by
    load_land_fraction and regrids to a model grid with BOTH land and ocean columns, so
    --ocean-only ranks a STRICT SUBSET (the iter-486 ocean-only-excludes-land test)."""
    from legoesm.grids.factory import create_grid
    from legoesm.grids.topography import load_land_fraction

    from scripts.data.make_synthetic_land_mask import main

    out = tmp_path / "land.nc"
    assert main([str(out)]) == 0
    assert out.is_file()
    grid = create_grid("latlon", resolution=8)
    f_land = np.asarray(load_land_fraction(grid, str(out)))
    n_land = int((f_land > 0.5).sum())
    assert 0 < n_land < f_land.size                       # a strict land/ocean mix on the grid
