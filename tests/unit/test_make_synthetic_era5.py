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


def test_synthetic_load_preserves_field_to_level_correspondence(tmp_path):
    """The descending→ascending level reorder must keep each field value tied to its
    OWN pressure level — the silent-catastrophic-bug surface of the ERA5 ingest.

    ``era5_to_state`` ``np.sort``s the levels to ascending Pa AND reverses the field
    axis (``data[..., ::-1]``) when the source levels are descending (the ERA5/WB2
    convention).  The sort and the reverse are INDEPENDENT operations: if the field
    reversal regressed (or were dropped) the loaded ``plev_Pa`` would STILL come out
    ascending — so the existing ``np.all(diff(plev_Pa) > 0)`` check would pass — while
    the T/u/v/q values would sit at the WRONG levels, silently comparing model-top
    against ERA5-surface and corrupting every column bias.  The synthetic T profile is
    ``220 + 70·(p/1000)^0.3`` — monotonically WARMER at higher pressure (≈290 K at the
    1000 hPa surface, ≈248 K at 50 hPa aloft).  So after load the level-mean T MUST
    increase with ``plev_Pa``; a broken reversal would leave the 290 K surface value at
    the 5000 Pa (50 hPa) slot and make it DECREASE — a decisive, non-vacuous catch.
    """
    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice

    from scripts.data.make_synthetic_era5 import main

    zp = tmp_path / "syn_order.zarr"
    assert main([str(zp), "--nlat", "6", "--nlon", "8", "--ntime", "1"]) == 0
    sl = load_era5_slice(TrainingERA5Config(zarr_store=str(zp)), 0)

    plev = np.asarray(sl.plev_Pa)
    t_profile = np.asarray(sl.T).mean(axis=(0, 1))    # level-mean T (n_lev,)
    assert plev[0] < plev[-1]                          # ascending: top (50 hPa) → surface
    # The coldest air sits at the LOWEST pressure (aloft); the warmest at the HIGHEST
    # (surface) — i.e. the field rode the reorder with its level, not against it.
    assert np.all(np.diff(t_profile) > 0)             # strictly warmer with pressure
    assert 244.0 < t_profile[0] < 252.0               # ~248 K at 50 hPa (NOT the 290 K sfc)
    assert 288.0 < t_profile[-1] < 291.0              # ~290 K at 1000 hPa surface
    assert t_profile[-1] - t_profile[0] > 30.0        # decisive surface-aloft contrast
