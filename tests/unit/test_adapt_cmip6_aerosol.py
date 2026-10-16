"""``scripts/data/adapt_cmip6_aerosol.py``: the reference MACv2-SP column port
and the 1850-2014 aerosol product it writes on the Kinne MACv1 background."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
netCDF4 = pytest.importorskip("netCDF4")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_PY = _REPO / "scripts" / "data" / "adapt_cmip6_aerosol.py"
_MACV2SP = pathlib.Path(os.environ.get(
    "LEGOESM_MACV2SP_FILE",
    "/burg-archive/glab/users/pg2328/legoESM/data/cmip6_forcing/aerosol/raw_macv2sp/MACv2.0-SP_v1.nc"))
_LNWL = [230, 300, 400, 550, 700, 1000, 1270, 1460, 1780, 2050, 2320, 2790, 3470, 8000]
_WL1 = [0.2, 0.263, 0.345, 0.442, 0.625, 0.778, 1.242, 1.299, 1.626, 1.942, 2.151, 2.5, 3.077, 3.846]
_WL2 = _WL1[1:] + [12.195]


def _mod():
    spec = importlib.util.spec_from_file_location("adapt_cmip6_aerosol", _PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# Reference sp_aop_profile (mo_simple_plumes_v1.f90, MPI-M-MACv2-SP-1-0 v20170201,
# real*8) on one 60-level terrain-following column per row, from
# scripts/validate/macv2sp_oracle.F90 < macv2sp_oracle_cases.txt:
# lat, lon (0-360), oro [m], year_fr, lambda [nm] -> column aod, ssa, asy.
FORTRAN = [
    (49.4, 20.6, 300.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (49.4, 20.6, 300.0, 1991.6, 550.0, 0.3406950375475875, 0.9300000055780814, 0.6299999952316284),
    (49.4, 20.6, 300.0, 2014.98, 300.0, 0.2860421025878955, 0.9300000066787475, 0.6299999952316285),
    (49.4, 20.6, 300.0, 1979.04, 1000.0, 0.11311916020054616, 0.8200468038857129, 0.5270958127269618),
    (49.4, 20.6, 300.0, 1950.81, 3470.0, 0.0028676947247612407, 0.09834066366799363, 0.28295985794584166),
    (45.0, 270.0, 0.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (45.0, 270.0, 0.0, 1991.6, 550.0, 0.09823630303695326, 0.9299999536815379, 0.6299999952316283),
    (45.0, 270.0, 0.0, 2014.98, 300.0, 0.07113234894768156, 0.929999988495965, 0.6299999952316282),
    (45.0, 270.0, 0.0, 1979.04, 1000.0, 0.01923856673048615, 0.8200467946628395, 0.5270958127269616),
    (45.0, 270.0, 0.0, 1950.81, 3470.0, 0.0016328106337198088, 0.09834066018570087, 0.28295985794584155),
    (40.0, 300.0, 0.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (40.0, 300.0, 0.0, 1991.6, 550.0, 0.07313604064850987, 0.9299898194437991, 0.6299999952316283),
    (40.0, 300.0, 0.0, 2014.98, 300.0, 0.05431433680463763, 0.9299965411342614, 0.6299999952316284),
    (40.0, 300.0, 0.0, 1979.04, 1000.0, 0.016029596943310742, 0.8200452061209421, 0.5270958127269615),
    (40.0, 300.0, 0.0, 1950.81, 3470.0, 0.0011127814862585022, 0.09833989435276616, 0.2829598579458418),
    (40.1, 277.5, 200.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (40.1, 277.5, 200.0, 1991.6, 550.0, 0.1669990117899409, 0.9299996596903529, 0.6299999952316286),
    (40.1, 277.5, 200.0, 2014.98, 300.0, 0.1208965027856351, 0.9299998858808479, 0.6299999952316281),
    (40.1, 277.5, 200.0, 1979.04, 1000.0, 0.032704919376960205, 0.8200467430546702, 0.5270958127269616),
    (40.1, 277.5, 200.0, 1950.81, 3470.0, 0.002775811929035695, 0.09834063985815589, 0.2829598579458419),
    (30.0, 114.0, 1500.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (30.0, 114.0, 1500.0, 1991.6, 550.0, 0.4121729261207523, 0.9299943870726163, 0.6299999952316283),
    (30.0, 114.0, 1500.0, 2014.98, 300.0, 1.9047810736114488, 0.9299989436229847, 0.6299999952316285),
    (30.0, 114.0, 1500.0, 1979.04, 1000.0, 0.07827589347649974, 0.8200454115076868, 0.5270958127269615),
    (30.0, 114.0, 1500.0, 1950.81, 3470.0, 0.0012825412758165388, 0.0983375276720125, 0.28295985794584166),
    (23.3, 88.0, 0.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (23.3, 88.0, 0.0, 1991.6, 550.0, 0.22080937083825242, 0.929966524050087, 0.6299999952316284),
    (23.3, 88.0, 0.0, 2014.98, 300.0, 1.0698853508872417, 0.9299939637920365, 0.6299999952316284),
    (23.3, 88.0, 0.0, 1979.04, 1000.0, 0.03998697267326318, 0.8200381036591369, 0.5270958127269618),
    (23.3, 88.0, 0.0, 1950.81, 3470.0, 0.0006794610751506535, 0.09832176947830967, 0.2829598579458416),
    (-10.0, 298.0, 500.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (-10.0, 298.0, 500.0, 1991.6, 550.0, 0.10293671658048344, 0.8700281286472132, 0.6299999952316285),
    (-10.0, 298.0, 500.0, 2014.98, 300.0, 0.017860742453988828, 0.8707815620656769, 0.6299999952316288),
    (-10.0, 298.0, 500.0, 1979.04, 1000.0, 0.0012816496467787013, 0.6973975975241375, 0.5270958127269614),
    (-10.0, 298.0, 500.0, 1950.81, 3470.0, 0.0006787459863897312, 0.05208835065394593, 0.28295985794584166),
    (-3.5, 16.0, 800.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (-3.5, 16.0, 800.0, 1991.6, 550.0, 0.3431656416487574, 0.8720781362402305, 0.6299999952316289),
    (-3.5, 16.0, 800.0, 2014.98, 300.0, 0.4723882897078799, 0.8773454330709071, 0.6299999952316281),
    (-3.5, 16.0, 800.0, 1979.04, 1000.0, 0.01659727873651934, 0.7125196065774598, 0.5270958127269618),
    (-3.5, 16.0, 800.0, 1950.81, 3470.0, 0.0010130409169082258, 0.053744057896953454, 0.28295985794584155),
    (-20.0, 135.0, 300.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (-20.0, 135.0, 300.0, 1991.6, 550.0, 0.10985181942161812, 0.9299805970461136, 0.6299999952316285),
    (-20.0, 135.0, 300.0, 2014.98, 300.0, 0.21759087473359376, 0.9299914374218227, 0.6299999952316283),
    (-20.0, 135.0, 300.0, 1979.04, 1000.0, 0.014178160747405789, 0.820039727371969, 0.5270958127269615),
    (-20.0, 135.0, 300.0, 1950.81, 3470.0, 0.001065442314847348, 0.09833718884227817, 0.2829598579458418),
    (-40.0, 200.0, 0.0, 1850.5, 550.0, 0.0, 1.0, 0.0),
    (-40.0, 200.0, 0.0, 1991.6, 550.0, 4.1851686139557826e-05, 0.930000007152557, 0.6299999952316286),
    (-40.0, 200.0, 0.0, 2014.98, 300.0, 8.126679173485834e-05, 0.9300000071525573, 0.6299999952316284),
    (-40.0, 200.0, 0.0, 1979.04, 1000.0, 5.486672124684108e-06, 0.8200468040487309, 0.5270958127269619),
    (-40.0, 200.0, 0.0, 1950.81, 3470.0, 4.149127678053892e-07, 0.0983406638829728, 0.2829598579458417),
]


@pytest.mark.skipif(not _MACV2SP.exists(), reason="MACv2.0-SP_v1.nc not staged here")
@pytest.mark.parametrize("east", (False, True))
def test_column_optics_match_the_reference_fortran(east):
    """Every oracle column, with the grid longitude given in -180..180 too: the
    port maps it to the reference's 0-360 (the Europe plume's 260-degree wrap is
    not convention-invariant)."""
    m = _mod()
    with xr.open_dataset(_MACV2SP, decode_times=False) as ds:
        p = m.read_plumes(ds)
    for lat, lon, _oro, yfr, lam, aod, ssa, asy in FORTRAN:
        lon_in = lon if east else (lon - 360.0 if lon > 180.0 else lon)
        cw = m.column_aod550(p, m.time_weights(p, yfr), m.feature_gaussians(p, lat, lon_in))
        got = [float(x[0]) for x in m.column_optics(m.plume_band_sums(p, cw, [lam]))]
        np.testing.assert_allclose(got, [aod, ssa, asy], rtol=1e-12, atol=1e-18,
                                   err_msg=str((lat, lon_in, yfr, lam)))


def _params_ds(year_weight_1851=1.0):
    """Two-plume MACv2-SP-shaped parameter set (dimension names as the real file)."""
    npl, nf, ny = 2, 2, 251
    pf = ("plume_number", "plume_feature")
    yw = np.zeros((npl, ny), np.float32)
    yw[:, 1:] = year_weight_1851
    yw[:, 200:] = np.nan                           # the v1 file masks years past 2016
    yw[0, 100] = np.nan                            # an interior hole (1950) must refuse too
    return xr.Dataset({
        "years": (("years",), np.arange(1850, 1850 + ny)),
        "plume_lat": (("plume_number",), [30.0, -10.0]),
        "plume_lon": (("plume_number",), [20.0, 300.0]),
        "aod_spmx": (("plume_number",), [0.3, 0.2]),
        "ssa550": (("plume_number",), [0.93, 0.9]),
        "asy550": (("plume_number",), [0.63, 0.6]),
        "angstrom": (("plume_number",), [2.0, 1.5]),
        "sig_lon_E": (pf, [[20.0, 10.0], [15.0, 15.0]]),
        "sig_lon_W": (pf, [[10.0, 10.0], [15.0, 15.0]]),
        "sig_lat_E": (pf, [[8.0, 8.0], [10.0, 10.0]]),
        "sig_lat_W": (pf, [[8.0, 8.0], [10.0, 10.0]]),
        "theta": (pf, [[0.2, 0.0], [0.0, 0.1]]),
        "ftr_weight": (pf, [[0.7, 0.3], [0.5, 0.5]]),
        "year_weight": (("plume_number", "years"), yw),
        "ann_cycle": (("plume_number", "year_fr", "plume_feature"),
                      np.ones((npl, 52, nf), np.float32)),
    })


def _write_macv1(path, kind, aod, rng):
    lat = np.array([67.5, 22.5, -22.5, -67.5])
    lon = np.arange(-157.5, 180.0, 45.0)
    with netCDF4.Dataset(path, "w") as nc:
        nc.history = "ncecat " + " ".join(f"gt_{kind}_{w:05d}nm.nc" for w in _LNWL)
        for d, n in (("lon", 8), ("lat", 4), ("time", 12), ("record", 14)):
            nc.createDimension(d, n)
        nc.createVariable("lat", "f4", ("lat",))[:] = lat
        nc.createVariable("lon", "f4", ("lon",))[:] = lon
        shape = (14, 12, 4, 8)
        for name, val in (("aod", aod * (1.0 + rng.random(shape))),
                          ("ssa", 0.8 + 0.15 * rng.random(shape)),
                          ("asy", 0.6 + 0.1 * rng.random(shape))):
            nc.createVariable(name, "f4", ("record", "time", "lat", "lon"))[:] = val


@pytest.fixture
def product(tmp_path):
    m = _mod()
    rng = np.random.default_rng(0)
    _params_ds().to_netcdf(tmp_path / "sp.nc")
    _write_macv1(tmp_path / "g30_pre_9999.nc", "p", 0.05, rng)
    _write_macv1(tmp_path / "g30_cor_2000.nc", "c", 0.03, rng)
    with netCDF4.Dataset(tmp_path / "tmpl.nc", "w") as nc:
        nc.createDimension("lnwl", 14)
        for name, val in (("lnwl", _LNWL), ("wl1_sun", _WL1), ("wl2_sun", _WL2)):
            nc.createVariable(name, "f4", ("lnwl",))[:] = val
    fin, tot = m.build(tmp_path / "sp.nc", tmp_path, tmp_path / "tmpl.nc", tmp_path, 1850, 1851)
    return m, tmp_path, fin, tot


def _read(path, name):
    with netCDF4.Dataset(path) as nc:
        return np.asarray(nc[name][:], dtype=np.float64)


def test_1850_is_the_background_and_plumes_add_after(product):
    """year_weight(1850) = 0 in MACv2-SP: 1850 is the MACv1 pre-industrial fine
    aerosol exactly; 1851 adds the plume column (largest at the plume centre);
    tot adds the coarse mode."""
    m, d, fin, tot = product
    pre = _read(d / "g30_pre_9999.nc", "aod").transpose(1, 0, 2, 3)     # (12, nb, lat, lon)
    cor = _read(d / "g30_cor_2000.nc", "aod").transpose(1, 0, 2, 3)
    a_fin, a_tot = _read(fin, "aod"), _read(tot, "aod")
    assert a_fin.shape == (24, 14, 4, 8)
    np.testing.assert_array_equal(a_fin[:12], pre)
    plume = a_fin[12:] - pre
    assert plume.min() >= -1e-6 and plume[:, 3].max() > 0.05
    lat, lon = _read(fin, "lat"), _read(fin, "lon")
    i, j = np.unravel_index(plume[0, 3].argmax(), plume[0, 3].shape)
    assert (lat[i], lon[j]) == (22.5, 22.5)                             # nearest the 30N/20E plume
    np.testing.assert_allclose(a_tot, a_fin + np.concatenate([cor, cor]), rtol=1e-6)
    t = _read(fin, "time")
    assert t[0] == 15.5 and t[13] == 365 + 31 + 14.0                    # mid-Feb 1851
    ssa, asy = _read(fin, "ssa"), _read(fin, "asy")
    assert np.all((ssa > 0) & (ssa <= 1)) and np.all((asy > 0) & (asy < 1))


def test_product_round_trips_through_the_aerosol_loader(product):
    """The loader reads both products: 24 months, the 550 nm band for CCN, and
    its slab-wise zonal mean equals a whole-array nanmean bitwise."""
    from legoesm.forcing.external import _load_monthly_zonal_anchored

    _m, _d, fin, tot = product
    for path in (fin, tot):
        mid, first, lat, vis = _load_monthly_zonal_anchored(str(path), "aod", "vis")
        assert mid.shape == (24,) and str(first).startswith("1850-01")
        np.testing.assert_array_equal(vis, np.nanmean(_read(path, "aod"), axis=-1)[:, 3])
        _, _, _, gray = _load_monthly_zonal_anchored(str(path), "aod", "gray")
        assert gray.shape == (24, 4) and np.all(gray > 0)


def test_time_weights_follow_the_reference_indexing(product):
    """FLOOR(year_fr)-1849 and FLOOR(frac*52)+1; a masked year and a year off the
    axis refuse; a month is the mean of its daily 12 UTC samples."""
    m = product[0]
    p = m.read_plumes(_params_ds(year_weight_1851=2.0))
    p["ann_cycle"][:, 4, :] = 3.0                                       # week 5 only
    np.testing.assert_array_equal(m.time_weights(p, 1851.0 + 4.5 / 52), 6.0)
    np.testing.assert_array_equal(m.time_weights(p, 1851.0 + 3.99 / 52), 2.0)
    np.testing.assert_array_equal(m.time_weights(p, 1850.9), 0.0)
    for bad in (1849.5, 2051.5):
        with pytest.raises(ValueError):
            m.time_weights(p, bad)
    feb = m.month_time_weights(p, 1851, 2)
    days = [1851.0 + (31 + d + 0.5) / 365 for d in range(28)]
    np.testing.assert_allclose(feb, np.mean([m.time_weights(p, y) for y in days], axis=0))


def test_bad_year_range_refuses_before_writing(product):
    """Reversed years and years without MACv2-SP weights refuse before any
    output exists; a finished build leaves no temporary file behind."""
    m, d, _fin, _tot = product
    assert not list(d.glob("*.partial"))
    for years in ((1851, 1850), (1849, 1850), (2049, 2050), (1940, 1960)):
        out = d / f"out{years[0]}"
        out.mkdir()
        with pytest.raises(ValueError):
            m.build(d / "sp.nc", d, d / "tmpl.nc", out, *years)
        assert not list(out.iterdir())


@pytest.mark.parametrize("width", (0.0, -5.0, np.nan))
def test_reader_refuses_a_non_positive_plume_width(width):
    m = _mod()
    ds = _params_ds()
    ds["sig_lat_W"][0, 1] = width
    with pytest.raises(ValueError):
        m.read_plumes(ds)


def test_record_order_must_match_the_band_template(tmp_path):
    m = _mod()
    _write_macv1(tmp_path / "g30_pre_9999.nc", "p", 0.05, np.random.default_rng(1))
    with pytest.raises(ValueError, match="record wavelengths"):
        m._read_macv1(tmp_path / "g30_pre_9999.nc", _LNWL[::-1])


def test_product_check_reports_global_means(product, capsys):
    """scripts/validate/macv2sp_product_check.py: cos(lat)-weighted annual 550 nm
    means; 1850 fin is the pre-industrial background, the plume adds after."""
    spec = importlib.util.spec_from_file_location(
        "macv2sp_product_check", _REPO / "scripts" / "validate" / "macv2sp_product_check.py")
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    lat = np.array([60.0, 0.0])
    assert chk.global_mean(np.array([[1.0, 1.0], [3.0, 3.0]]), lat) == pytest.approx(
        (0.5 * 1 + 1.0 * 3) / 1.5)
    _m, d, fin, tot = product
    gm, cover = chk.annual_gm_550(fin, (1850, 1851))
    assert cover == (1850, 1851, 24)
    days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31.0])
    pre = np.tensordot(days / 365.0, _read(d / "g30_pre_9999.nc", "aod")[3], axes=1)
    assert gm[1850] == pytest.approx(chk.global_mean(pre, _read(fin, "lat")), rel=1e-12)
    assert gm[1851] > gm[1850]
    assert chk.main([str(fin), str(tot), "--years", "1851"]) == 0      # 1850 not listed
    out = capsys.readouterr().out
    assert f"anthropogenic (fin - fin[1850]) {gm[1851] - gm[1850]:.4f}" in out
