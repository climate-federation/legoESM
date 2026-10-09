"""SW aerosol band collapse, stratospheric volcanic SW, and the CCN input.

- solar-weighted gray / 550 nm band from the file's own ``wl1_sun``/``wl2_sun``
  (order-independent; no edges -> old flat mean);
- volcanic SW (``ext_sun``) placed at its stratospheric pressure, not spread
  through the troposphere, and excluded from ``aerosol_ccn_aod``;
- ``aerosol_ccn_aod`` drives the AOD->CCN proxy when present, byte-identical
  sum(aerosol_od) when absent;
- ``--aerosol-ccn-file`` -> ExperimentConfig -> AerosolConfig.ccn_path -> CCN.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

# Three bands deliberately NOT in wavelength order: near-IR, visible, UV [um].
_WL1 = np.array([0.778, 0.442, 0.263])
_WL2 = np.array([1.242, 0.625, 0.345])
_LAT = np.linspace(-90.0, 90.0, 7)


def _kinne_file(path, band_aod, edges=True):
    """(time=12, lnwl, lat) AOD file, band values constant in time/lat."""
    nb = len(band_aod)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lnwl", nb)
        ds.createDimension("lat", _LAT.size)
        ds.createVariable("time", "f8", ("time",))[:] = 15.5 + 30.4375 * np.arange(12)
        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
        ds.createVariable("aod", "f8", ("time", "lnwl", "lat"))[:] = np.broadcast_to(
            np.asarray(band_aod, float)[None, :, None], (12, nb, _LAT.size))
        if edges:
            ds.createVariable("wl1_sun", "f8", ("lnwl",))[:] = _WL1[:nb]
            ds.createVariable("wl2_sun", "f8", ("lnwl",))[:] = _WL2[:nb]
    return str(path)


def _volc_file(path, ext=0.01, alt=(18.0, 20.0, 22.0, 24.0)):
    """CMIP6-shaped ``ext_sun(solar_bands, lat, altitude, month)`` [1/km]."""
    with netCDF4.Dataset(path, "w") as ds:
        for d, n in (("solar_bands", 3), ("lat", _LAT.size), ("altitude", len(alt)), ("month", 12)):
            ds.createDimension(d, n)
        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
        ds.createVariable("altitude", "f8", ("altitude",))[:] = np.asarray(alt)
        ds.createVariable("ext_sun", "f8", ("solar_bands", "lat", "altitude", "month"))[:] = ext
        ds.createVariable("wl1_sun", "f8", ("solar_bands",))[:] = _WL1
        ds.createVariable("wl2_sun", "f8", ("solar_bands",))[:] = _WL2
    return str(path)


def test_solar_weighted_and_visible_band_collapse(tmp_path):
    from legoesm.forcing.external import (
        _SOLAR_PLANCK_TEMP_K,
        _load_monthly_zonal_anchored,
        _planck_band_weights,
    )
    tau = np.array([0.05, 0.20, 0.40])  # near-IR, vis, UV
    p = _kinne_file(tmp_path / "k.nc", tau)
    w = _planck_band_weights(_WL1, _WL2, _SOLAR_PLANCK_TEMP_K)
    gray = _load_monthly_zonal_anchored(p, "aod")[3]
    np.testing.assert_allclose(gray, (w * tau).sum() / w.sum(), rtol=1e-12)
    assert not np.isclose(gray[0, 0], tau.mean())  # really weighted
    vis = _load_monthly_zonal_anchored(p, "aod", "vis")[3]
    np.testing.assert_allclose(vis, 0.20)  # band containing 550 nm
    # Band order is taken from the edges, not assumed: reversed file, same answer.
    pr = str(tmp_path / "k_rev.nc")
    with netCDF4.Dataset(pr, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lnwl", 3)
        ds.createDimension("lat", _LAT.size)
        ds.createVariable("time", "f8", ("time",))[:] = 15.5 + 30.4375 * np.arange(12)
        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
        ds.createVariable("aod", "f8", ("time", "lnwl", "lat"))[:] = np.broadcast_to(
            tau[::-1][None, :, None], (12, 3, _LAT.size))
        ds.createVariable("wl1_sun", "f8", ("lnwl",))[:] = _WL1[::-1]
        ds.createVariable("wl2_sun", "f8", ("lnwl",))[:] = _WL2[::-1]
    np.testing.assert_allclose(_load_monthly_zonal_anchored(pr, "aod")[3], gray, rtol=1e-12)
    np.testing.assert_allclose(_load_monthly_zonal_anchored(pr, "aod", "vis")[3], 0.20)
    # No edges -> old flat band mean (backward compatible), also for "vis".
    pn = _kinne_file(tmp_path / "k_noedge.nc", tau, edges=False)
    np.testing.assert_allclose(_load_monthly_zonal_anchored(pn, "aod")[3], tau.mean())
    np.testing.assert_allclose(_load_monthly_zonal_anchored(pn, "aod", "vis")[3], tau.mean())


def _driver_forcing(cfg, ncol=_LAT.size, temp_fn=None, **kw):
    """Run the real driver _setup_external_forcing + _precompute_external_forcing
    (+ ``_owned_temp``) on a minimal stand-in; ``temp_fn(p_full)`` -> the
    state's grid-point T (None -> state without T); ``kw`` -> precompute."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.grids.vertical import make_cam6_l32_levels
    fake = SimpleNamespace(config=cfg)
    ModelDriver._setup_external_forcing(fake)
    fake.sigma = make_cam6_l32_levels()
    p_s = jnp.full((ncol,), 1.0e5)
    fake.state = SimpleNamespace() if temp_fn is None else SimpleNamespace(
        T=SimpleNamespace(data=temp_fn(fake.sigma.pressure_at_full(p_s))))
    fake._owned_face_ids = None
    fake._gather_spmd_tree_to_host = lambda x: x
    fake._owned_temp = lambda: ModelDriver._owned_temp(fake)
    lat = jnp.radians(jnp.asarray(_LAT))
    _, aer, _ = ModelDriver._precompute_external_forcing(fake, 15.5, p_s, lat, **kw)
    p_full = np.asarray(fake.sigma.pressure_at_full(p_s)).reshape(ncol, -1)
    return fake, np.asarray(aer), p_full


def _cfg(**kw):
    from legoesm.driver.config import ExperimentConfig
    return ExperimentConfig(radiation="rrtmgp", aerosol_forcing="external", **kw)


def test_volcanic_sw_above_tropopause_and_out_of_ccn(tmp_path):
    trop = _kinne_file(tmp_path / "t.nc", [0.05, 0.20, 0.40])
    volc = _volc_file(tmp_path / "v.nc")  # 17-25 km, column ~0.08
    f0, aer0, _ = _driver_forcing(_cfg(aerosol_file=trop))
    f1, aer1, p_full = _driver_forcing(_cfg(aerosol_file=trop, volcanic_aerosol_file=volc))
    dv = aer1 - aer0
    np.testing.assert_allclose(dv.sum(-1), 0.01 * 8.0, rtol=1e-6)  # all of it lands
    assert np.abs(dv[p_full > 3.0e4]).max() < 1e-12  # none below 300 hPa
    # CCN: tropospheric 550 nm column, volcanic excluded.
    np.testing.assert_allclose(np.asarray(f1._aerosol_ccn_aod), 0.20, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(f1._aerosol_ccn_aod),
                                  np.asarray(f0._aerosol_ccn_aod))


def test_ccn_aod_used_when_present_byte_identical_when_absent():
    from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
        ccn_from_aod,
        specified_nc_field,
    )
    from legoesm.atmosphere.physics.microphysics.arg_activation import (
        ActivationConfig,
        activated_nc_field,
    )
    aer = jnp.asarray(np.random.default_rng(0).uniform(0, 0.02, (4, 6)))
    shape = (4, 6)
    old = specified_nc_field(aer, shape)
    np.testing.assert_array_equal(
        activated_nc_field(ActivationConfig(), shape, aerosol_od=aer), old)
    np.testing.assert_array_equal(
        activated_nc_field(ActivationConfig(), shape, aerosol_od=aer, ccn_aod=None), old)
    ccn = jnp.asarray([0.01, 0.05, 0.1, 0.3])
    new = activated_nc_field(ActivationConfig(), shape, aerosol_od=aer, ccn_aod=ccn)
    np.testing.assert_allclose(new, jnp.broadcast_to(ccn_from_aod(ccn)[:, None], shape))


def test_ccn_file_flows_from_cli_to_ccn(tmp_path):
    from scripts.run import run_amip
    fine = _kinne_file(tmp_path / "fin.nc", [0.05, 0.10, 0.40])
    tot = _kinne_file(tmp_path / "tot.nc", [0.15, 0.30, 0.50])
    parser = run_amip.build_arg_parser()
    assert run_amip.build_config_from_args(parser.parse_args([])).aerosol_ccn_file == ""
    cli = run_amip.build_config_from_args(parser.parse_args(
        ["--aerosol-file", tot, "--aerosol-ccn-file", fine]))
    assert cli.aerosol_ccn_file == fine
    f, _, _ = _driver_forcing(_cfg(aerosol_file=tot, aerosol_ccn_file=fine))
    assert f._aerosol_config.path == tot and f._aerosol_config.ccn_path == fine
    np.testing.assert_allclose(np.asarray(f._aerosol_ccn_aod), 0.10, rtol=1e-12)
    g, _, _ = _driver_forcing(_cfg(aerosol_file=tot))  # unset -> path
    np.testing.assert_allclose(np.asarray(g._aerosol_ccn_aod), 0.30, rtol=1e-12)


def test_volcanic_weight_zero_below_tropopause():
    """CMIP6 strat-aerosol notes: 1 above the tropopause, 0.5 in the layer
    holding it, 0 below (codex 2026-10-08: the file's tail reaches 5 km)."""
    from legoesm.forcing.surface_utils import stratospheric_layer_weight
    p_half = jnp.array([[1e3, 5e3, 9e3, 11e3, 2e4, 5e4, 1e5]])  # Pa, TOA first
    w = np.asarray(stratospheric_layer_weight(p_half, jnp.array([0.0])))[0]
    # equator: tropopause 100 hPa sits in the 90-110 hPa layer
    np.testing.assert_array_equal(w, [1.0, 1.0, 0.5, 0.0, 0.0, 0.0])
    # pole: tropopause 300 hPa -> 200-500 hPa layer is the transition one
    w_p = np.asarray(stratospheric_layer_weight(p_half, jnp.array([np.pi / 2])))[0]
    np.testing.assert_array_equal(w_p, [1.0, 1.0, 1.0, 1.0, 0.5, 0.0])


def _std_temp(p, z_tp_m=12.0e3, t0=288.0, lapse=6.5e-3):
    """Analytic column: ``lapse`` K/m from t0 at 1000 hPa to ``z_tp_m``, then
    isothermal (T = t0 (p/p0)^(R lapse/g) closed form); z_tp_m=inf -> none.
    Returns (T, tropopause pressure [Pa])."""
    from legoesm import constants
    k = constants.R_d * lapse / constants.g
    t_tp = t0 - lapse * z_tp_m
    p_tp = 1.0e5 * (max(t_tp, 1e-9) / t0) ** (1.0 / k)
    return jnp.maximum(t0 * (p / 1.0e5) ** k, t_tp), p_tp


def test_wmo_tropopause_analytic_column_and_layer_weight():
    """6.5 K/km to 12 km then isothermal: WMO tropopause found within one
    layer of the truth, either vertical order, under jit; the layer weight is
    1 above / 0.5 in the found layer / 0 below."""
    from legoesm.forcing.surface_utils import (
        stratospheric_layer_weight,
        wmo_tropopause_pressure,
    )
    from legoesm.grids.vertical import make_cam6_l32_levels

    sig = make_cam6_l32_levels()
    p_s = jnp.full((2,), 1.0e5)
    p_full, p_half = sig.pressure_at_full(p_s), sig.pressure_at_half(p_s)
    temp, p_true = _std_temp(p_full)
    p_wmo = np.asarray(jax.jit(wmo_tropopause_pressure)(p_full, temp))
    ph = np.asarray(p_half[0])
    lay = lambda pp: int(np.searchsorted(np.sort(ph), pp)) - 1  # noqa: E731
    assert abs(lay(p_wmo[0]) - lay(p_true)) <= 1, (p_wmo[0], p_true)
    np.testing.assert_array_equal(  # order-independent
        np.asarray(wmo_tropopause_pressure(p_full[:, ::-1], temp[:, ::-1])), p_wmo)
    w = np.asarray(stratospheric_layer_weight(
        p_half, jnp.array([0.0, np.pi / 2]), p_full, temp))
    for wc in w:  # same tropopause at equator and pole: T decides, not lat
        pf = np.asarray(p_full[0])
        np.testing.assert_array_equal(wc[pf < p_wmo[0]], 1.0)
        np.testing.assert_array_equal(wc[pf > p_wmo[0]], 0.0)
        np.testing.assert_array_equal(wc[pf == p_wmo[0]], 0.5)


def test_wmo_tropopause_fallback_to_climatology():
    """No qualifying level in 500-50 hPa (constant 6.5 K/km to the top) ->
    NaN, and the layer weight is exactly the climatological one."""
    from legoesm.forcing.surface_utils import (
        stratospheric_layer_weight,
        wmo_tropopause_pressure,
    )
    from legoesm.grids.vertical import make_cam6_l32_levels

    sig = make_cam6_l32_levels()
    p_s = jnp.full((2,), 1.0e5)
    p_full, p_half = sig.pressure_at_full(p_s), sig.pressure_at_half(p_s)
    temp, _ = _std_temp(p_full, z_tp_m=np.inf)
    assert np.isnan(np.asarray(wmo_tropopause_pressure(p_full, temp))).all()
    lat = jnp.array([0.0, np.pi / 2])
    np.testing.assert_array_equal(
        np.asarray(stratospheric_layer_weight(p_half, lat, p_full, temp)),
        np.asarray(stratospheric_layer_weight(p_half, lat)))


def test_driver_volcanic_cut_at_state_tropopause(tmp_path):
    """The driver threads the state T: a 19 km (~60 hPa) WMO tropopause cuts
    part of the 18-24 km volcanic layer the climatological (100 hPa) one keeps."""
    volc = _volc_file(tmp_path / "v.nc")
    trop = _kinne_file(tmp_path / "t.nc", [0.05, 0.20, 0.40])
    cfg = _cfg(aerosol_file=trop, volcanic_aerosol_file=volc)
    _, aer0, _ = _driver_forcing(_cfg(aerosol_file=trop))
    _, aer_c, _ = _driver_forcing(cfg)
    _, aer_w, p_full = _driver_forcing(
        cfg, temp_fn=lambda p: _std_temp(p, z_tp_m=19.0e3, t0=300.0)[0])
    np.testing.assert_allclose((aer_c - aer0).sum(-1), 0.01 * 8.0, rtol=1e-6)
    dv = aer_w - aer0
    assert (dv.sum(-1) < 0.99 * 0.08).all() and (dv.sum(-1) > 0.0).all()
    p_tp = _std_temp(1.0e5, z_tp_m=19.0e3, t0=300.0)[1]
    assert np.abs(dv[p_full > 2.0 * p_tp]).max() < 1e-12
    # Explicit ``temp=`` (spectral caller) == state T; off-grid shape -> climatology.
    tp = _std_temp(jnp.asarray(p_full), z_tp_m=19.0e3, t0=300.0)[0]
    np.testing.assert_array_equal(_driver_forcing(cfg, temp=tp)[1], aer_w)
    np.testing.assert_array_equal(_driver_forcing(cfg, temp=tp[:, :-1])[1], aer_c)


def test_owned_temp_selects_owned_faces():
    from legoesm.driver.model_driver import ModelDriver
    temp = jnp.arange(24.0).reshape(6, 2, 2)
    fake = SimpleNamespace(state=SimpleNamespace(T=SimpleNamespace(data=temp)),
                           _owned_face_ids=jnp.array([4, 1]))
    np.testing.assert_array_equal(ModelDriver._owned_temp(fake), temp[jnp.array([4, 1])])
    assert ModelDriver._owned_temp(SimpleNamespace(state=SimpleNamespace())) is None


def test_wmo_tropopause_checks_every_level_within_2km():
    """WMO needs the mean lapse rate to EVERY level within 2 km <= 2 K/km, not
    just to the 2 km point: 6.5 K/km to 10 km, isothermal to 10.5, -3 K by 11,
    isothermal above.  The 10 km level passes the endpoint test (1.5 K/km to
    12 km) but not 10->11 km (3 K/km); the tropopause is the 11 km level."""
    from legoesm.forcing.surface_utils import wmo_tropopause_pressure

    from legoesm import constants

    z = np.arange(0.0, 20001.0, 250.0)
    temp = np.interp(z, [0.0, 1e4, 1.05e4, 1.1e4, 2e4], [288.0, 223.0, 223.0, 220.0, 220.0])
    dlnp = constants.g * np.diff(z) / (constants.R_d * 0.5 * (temp[1:] + temp[:-1]))
    p = 1.0e5 * np.exp(-np.concatenate([[0.0], np.cumsum(dlnp)]))
    p_tp = float(wmo_tropopause_pressure(jnp.asarray(p[None]), jnp.asarray(temp[None]))[0])
    np.testing.assert_allclose(p_tp, p[z == 1.1e4][0], rtol=1e-12)
