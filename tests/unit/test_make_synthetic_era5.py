"""Unit test for ``scripts/data/make_synthetic_era5.py`` — the synthetic ERA5 smoke
store that must load through the campaign's own ERA5 loader.
"""

from __future__ import annotations

import numpy as np


def test_build_synthetic_era5_dataset_has_required_vars():
    from scripts.data.make_synthetic_era5 import build_synthetic_era5_dataset

    ds = build_synthetic_era5_dataset(nlat=8, nlon=12, ntime=3)
    for v in ("temperature", "u_component_of_wind", "v_component_of_wind",
              "specific_humidity", "surface_pressure", "skin_temperature"):
        assert v in ds.data_vars
    assert ds.temperature.dims == ("time", "level", "lat", "lon")
    assert ds.surface_pressure.dims == ("time", "lat", "lon")
    assert ds.sizes["time"] == 3 and ds.sizes["lat"] == 8 and ds.sizes["lon"] == 12
    # T varies by level (the vertical interp is genuinely exercised, not a uniform field);
    # lat is the ERA5 90→-90 descending convention.
    t = ds.temperature.values
    assert not np.allclose(t[:, 0], t[:, -1])
    assert ds.lat.values[0] > ds.lat.values[-1]


def test_synthetic_store_loads_via_campaign_era5_loader(tmp_path):
    """End-to-end: the written zarr loads through the campaign's load_era5_slice into a
    valid ERA5Slice (levels sorted to ascending Pa, finite T, right shapes) — so it is a
    genuine smoke-test input for the full campaign loop, not merely an xarray object."""
    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice

    from scripts.data.make_synthetic_era5 import main

    zp = tmp_path / "syn.zarr"
    assert main([str(zp), "--nlat", "10", "--nlon", "20", "--ntime", "2"]) == 0
    sl = load_era5_slice(TrainingERA5Config(zarr_store=str(zp)), 0)
    assert tuple(sl.T.shape) == (10, 20, 13)          # (lat, lon, level), WB2 13 levels
    assert tuple(sl.p_s.shape) == (10, 20)
    assert bool(np.all(np.isfinite(sl.T))) and bool(np.all(np.isfinite(sl.p_s)))
    assert np.all(np.diff(sl.plev_Pa) > 0)            # ascending Pa (sorted from hPa-descending)
