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
    # 1-deg synthetic CMIP6 file: Jan 2000 = 3, Jan 2001 = 7, Feb 2001 = 11.
    # Two Januaries, so a month-only (year-blind) selection gives the wrong 5.
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    vals = np.stack([np.full((lat.size, lon.size), x) for x in (3.0, 7.0, 11.0)])
    cftime = pytest.importorskip("cftime")
    time = [cftime.DatetimeNoLeap(y, m, 16) for y, m in
            ((2000, 1), (2001, 1), (2001, 2))]
    da = xr.DataArray(vals, dims=("time", "lat", "lon"),
                      coords={"time": time, "lat": lat, "lon": lon},
                      attrs={"units": "%"})
    f = tmp_path / "clt_Amon_X_amip_r1i1p1f1_gn_200001-200102.nc"
    da.to_dataset(name="clt").to_netcdf(f)
    got, _, _ = ce.window_mean([str(f)], "clt", [(2001, 1)])
    np.testing.assert_allclose(got, 7.0)
    got, _, _ = ce.window_mean([str(f)], "clt", [(2000, 1), (2001, 2)])
    np.testing.assert_allclose(got, 7.0)
    with pytest.raises(LookupError):              # month absent -> left out
        ce.window_mean([str(f)], "clt", [(2001, 3)])
    with pytest.raises(SystemExit):               # outside AMIP -> fatal
        ce.check_window([(2015, 1)])
    ce.check_window([(1979, 1), (2014, 12)])


def test_regrid_bias_and_zscore_known_answer():
    pytest.importorskip("xarray")                 # regional_bias imports it
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
    lego[0, 0] = np.nan                                  # masked in legoESM
    models[1, 3, 4] = np.nan                             # masked in one model
    st = ce.ensemble_stats(lego, ref, models)
    assert not st["valid"][0, 0] and not st["valid"][3, 4]
    assert st["valid"].sum() == box.size - 2
    v = st["valid"]
    np.testing.assert_allclose(st["b_mmm"][v], 3.0)
    np.testing.assert_allclose(st["spread"][v], 2.0)     # ddof=1 sd of 1,3,5
    np.testing.assert_allclose(st["z"][v], 4.0)
    assert st["z_ok"][v].all()
    # a cell where the models agree (spread 0.1 < 0.1 x median 2) is masked
    agree = models.copy()
    agree[:, 5, 6] = ref[5, 6] + np.array([3.0, 3.1, 3.2])
    sa = ce.ensemble_stats(lego, ref, agree)
    assert sa["valid"][5, 6] and not sa["z_ok"][5, 6]
    assert np.isnan(sa["z"][5, 6]) and np.isfinite(sa["z"][5, 7])
    assert np.isnan(st["z"][0, 0])
    np.testing.assert_allclose(ce.area_mean(st["b_lego"], mlat, v), 8.0)
    np.testing.assert_allclose(ce.area_rms(st["b_mod"][2], mlat, v), 5.0)
    assert ce.rank_of(4.0, [1.0, 3.0, 5.0]) == 3
    assert ce.rank_of(0.5, [1.0, 3.0, 5.0]) == 1


def test_lowest_f_member(tmp_path):
    pytest.importorskip("xarray")
    ce = _load()
    for m in ("r1i1p1f10", "r1i1p1f2", "r2i1p1f1"):
        (tmp_path / "I" / "M" / "amip" / m).mkdir(parents=True)
    assert ce.list_models(str(tmp_path)) == {"M": ("I", "r1i1p1f2")}


def test_model_field_albedo_is_ratio_of_binned_fluxes(tmp_path):
    """File-backed: month selection, binning with cos weights, ratio order."""
    xr = pytest.importorskip("xarray")
    cftime = pytest.importorskip("cftime")
    ce = _load()
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    mlat = np.arange(-87.5, 90.0, 5.0)
    mlon = np.arange(2.5, 360.0, 5.0)
    # rsdt and the albedo both vary inside each 5-deg box and are correlated,
    # so the mean of ratios differs from the ratio of binned means.
    k = np.arange(lat.size) % 5
    rsdt = (100.0 + 60.0 * k)[:, None] + 10.0 * np.cos(np.deg2rad(lon))[None, :]
    alb = np.broadcast_to((0.1 + 0.1 * k)[:, None], rsdt.shape)
    time = [cftime.DatetimeNoLeap(2000, 1, 16), cftime.DatetimeNoLeap(2001, 1, 16)]
    files = {}
    for v, f in (("rsut", alb * rsdt), ("rsdt", rsdt)):
        da = xr.DataArray(np.stack([f * 0.5, f]), dims=("time", "lat", "lon"),
                          coords={"time": time, "lat": lat, "lon": lon},
                          attrs={"units": "W m-2"})
        p = tmp_path / f"{v}_Amon_M_amip_r1i1p1f1_gn_200001-200112.nc"
        da.to_dataset(name=v).to_netcdf(p)
        files[v] = [str(p)]
    got = ce.model_field("albedo", files, [(2001, 1)], mlat, mlon)
    # independent answer: cos-lat weighted 5x5 block means, then the ratio
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(rsdt)

    def block(a):
        num = (a * w).reshape(36, 5, 72, 5).sum(axis=(1, 3))
        return num / w.reshape(36, 5, 72, 5).sum(axis=(1, 3))
    want = block(alb * rsdt) / block(rsdt)
    np.testing.assert_allclose(got, want, rtol=1e-10)
    assert not np.allclose(got, block(alb), rtol=1e-6)   # ratio order matters
