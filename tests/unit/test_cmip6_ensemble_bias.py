"""cmip6_ensemble_bias: window, regrid, bias and z-score on a known answer."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_SRC = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    "scripts", "validate", "amip_bias", "cmip6_ensemble_bias.py"))


def _load():
    spec = importlib.util.spec_from_file_location("_cmip6_ens_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_window_selection_and_coverage(tmp_path):
    xr = pytest.importorskip("xarray")
    ce = _load()
    # 1-deg synthetic CMIP6 file, Dec 2000 - Feb 2001; January carries 7.
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    vals = np.stack([np.full((lat.size, lon.size), x) for x in (3.0, 7.0, 11.0)])
    cftime = pytest.importorskip("cftime")
    time = [cftime.DatetimeNoLeap(y, m, 16) for y, m in
            ((2000, 12), (2001, 1), (2001, 2))]
    da = xr.DataArray(vals, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lat, "lon": lon},
                      attrs={"units": "%"})
    f = tmp_path / "clt_Amon_X_amip_r1i1p1f1_gn_200012-200102.nc"
    da.to_dataset(name="clt").to_netcdf(f)
    got, _, _ = ce.window_mean([str(f)], "clt", [(2001, 1)])
    np.testing.assert_allclose(got, 7.0)
    got, _, _ = ce.window_mean([str(f)], "clt", [(2000, 12), (2001, 1)])
    np.testing.assert_allclose(got, 5.0)
    with pytest.raises(LookupError):              # month absent -> left out
        ce.window_mean([str(f)], "clt", [(2001, 3)])
    with pytest.raises(SystemExit):               # outside AMIP -> fatal
        ce.check_window([(2015, 1)])
    ce.check_window([(1979, 1), (2014, 12)])


def test_regrid_bias_and_zscore_known_answer():
    ce = _load()
    # model grid 5 deg; source 1 deg field constant within each 5-deg box,
    # so the area-weighted binning must return the box value exactly.
    mlat = np.arange(-87.5, 90.0, 5.0)
    mlon = np.arange(2.5, 360.0, 5.0)
    box = np.add.outer(np.arange(mlat.size), 0.1 * np.arange(mlon.size))
    src = np.repeat(np.repeat(box, 5, axis=0), 5, axis=1)
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    np.testing.assert_allclose(ce.rb.bin_to_model(src, lat, lon, mlat, mlon),
                               box, rtol=1e-12)

    ref = box
    models = np.stack([box + 1.0, box + 3.0, box + 5.0])  # biases 1, 3, 5
    lego = box + 8.0                                     # bias 8
    lego[0, 0] = np.nan                                  # one masked cell
    st = ce.ensemble_stats(lego, ref, models)
    assert not st["valid"][0, 0] and st["valid"].sum() == box.size - 1
    v = st["valid"]
    np.testing.assert_allclose(st["b_mmm"][v], 3.0)
    np.testing.assert_allclose(st["spread"][v], 2.0)     # ddof=1 sd of 1,3,5
    np.testing.assert_allclose(st["z"][v], 4.0)
    assert np.isnan(st["z"][0, 0])
    np.testing.assert_allclose(ce.area_mean(st["b_lego"], mlat, v), 8.0)
    np.testing.assert_allclose(ce.area_rms(st["b_mod"][2], mlat, v), 5.0)
    assert ce.rank_of(4.0, [1.0, 3.0, 5.0]) == 3
    assert ce.rank_of(0.5, [1.0, 3.0, 5.0]) == 1


def test_lowest_f_member(tmp_path):
    ce = _load()
    for m in ("r1i1p1f10", "r1i1p1f2", "r2i1p1f1"):
        (tmp_path / "I" / "M" / "amip" / m).mkdir(parents=True)
    assert ce.list_models(str(tmp_path)) == {"M": ("I", "r1i1p1f2")}
