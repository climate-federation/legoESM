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


def _driver_forcing(cfg, ncol=_LAT.size):
    """Run the real driver _setup_external_forcing + _precompute_external_forcing
    on a minimal stand-in (only ``config`` / ``sigma`` are read)."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.grids.vertical import make_cam6_l32_levels
    fake = SimpleNamespace(config=cfg)
    ModelDriver._setup_external_forcing(fake)
    fake.sigma = make_cam6_l32_levels()
    p_s = jnp.full((ncol,), 1.0e5)
    lat = jnp.radians(jnp.asarray(_LAT))
    _, aer, _ = ModelDriver._precompute_external_forcing(fake, 15.5, p_s, lat)
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
