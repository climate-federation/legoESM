"""Volcanic stratospheric aerosol in the SHORTWAVE with the CMIP6 file's own
per-band extinction / single-scattering albedo / asymmetry.

- loader: bands sorted to RRTMGP order and checked against its band limits,
  moments (tau, tau*ssa, tau*ssa*g), refusals (no SSA/g, wrong bands, g<0);
- the band-limit table equals both RRTMGP SW gas-optics files;
- RRTMGP: moments carrying the grey ssa/g reproduce the grey path; the file's
  non-absorbing sulfate absorbs less than the grey 0.93; SW flux closure;
  jit parity; finite non-zero gradient;
- gray / zero radiation refuse the field.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
# RRTMG-SW / CMIP6-file band edges [um], in the file's own (E3SM) order.
_WL1 = np.array([3.07692, 2.5, 2.15054, 1.94175, 1.62602, 1.2987, 1.24224,
                 0.77821, 0.625, 0.441501, 0.344828, 0.263158, 0.2, 3.84615])
_WL2 = np.array([3.84615, 3.07692, 2.5, 2.15054, 1.94175, 1.62602, 1.2987,
                 1.24224, 0.77821, 0.625, 0.441501, 0.344828, 0.263158, 12.1951])
_LAT = np.linspace(-90.0, 90.0, 7)
_PROD_FILE = "/glade/work/pg2328/amip_inputs/volcanic_cmip6_from_cesm_1978-2014.nc"


def volc_sw_file(path, ext=0.01, omega=None, g=None, alt=(18.0, 20.0, 22.0, 24.0),
                 wl=(_WL1, _WL2), drop=()):
    """CMIP6-shaped volcanic file: ``ext_sun/omega_sun/g_sun(solar_bands,
    lat, altitude, month)``; per-band ``omega``/``g`` (file band order)."""
    nb = len(wl[0])
    omega = np.full(nb, 0.99) if omega is None else np.asarray(omega, float)
    g = np.full(nb, 0.7) if g is None else np.asarray(g, float)
    shape = (nb, _LAT.size, len(alt), 12)
    with netCDF4.Dataset(path, "w") as ds:
        for d, n in (("solar_bands", nb), ("lat", _LAT.size),
                     ("altitude", len(alt)), ("month", 12)):
            ds.createDimension(d, n)
        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
        ds.createVariable("altitude", "f8", ("altitude",))[:] = np.asarray(alt)
        dims = ("solar_bands", "lat", "altitude", "month")
        fields = {"ext_sun": np.broadcast_to(np.asarray(ext, float), shape),
                  "omega_sun": np.broadcast_to(omega[:, None, None, None], shape),
                  "g_sun": np.broadcast_to(g[:, None, None, None], shape)}
        for k, v in fields.items():
            if k not in drop:
                ds.createVariable(k, "f8", dims)[:] = v
        ds.createVariable("wl1_sun", "f8", ("solar_bands",))[:] = wl[0]
        ds.createVariable("wl2_sun", "f8", ("solar_bands",))[:] = wl[1]
    return str(path)


def _load(path):
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
    _load_volcanic_sw_optics_profile.cache_clear()
    return _load_volcanic_sw_optics_profile(path)


def test_loader_sorts_bands_and_builds_moments(tmp_path):
    omega = np.linspace(0.2, 1.0, 14)
    g = np.linspace(0.3, 0.8, 14)
    _, _, lat, mom, p_edges, band_wn = _load(
        volc_sw_file(tmp_path / "v.nc", omega=omega, g=g))
    assert mom.shape == (12, _LAT.size, 4, 3, 14) and mom.dtype == np.float32
    assert np.all(np.diff(p_edges) > 0)           # top first
    assert np.all(np.diff(band_wn[:, 0]) > 0)     # ascending wavenumber
    srt = np.argsort(1.0e4 / np.maximum(_WL1, _WL2))
    assert srt[0] == 13                           # 820-2600 cm-1 band first
    tau = 0.01 * 2.0                               # ext [1/km] * dz 2 km
    np.testing.assert_allclose(mom[0, 0, 0, 0], tau, rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 1], tau * omega[srt], rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 2], tau * omega[srt] * g[srt], rtol=1e-6)


@pytest.mark.parametrize("kw, match", [
    ({"drop": ("omega_sun",)}, "omega_sun"),
    ({"drop": ("g_sun",)}, "g_sun"),
    ({"g": np.full(14, -0.1)}, "Negative"),
    ({"wl": (_WL1[:13], _WL2[:13])}, "map 1:1"),
    ({"wl": (_WL1 * 1.05, _WL2 * 1.05)}, "map 1:1"),
    ({"wl": (_WL1, np.where(_WL2 == 0.441501, 0.4428, _WL2))}, "map 1:1"),
])
def test_loader_refuses(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        _load(volc_sw_file(tmp_path / "v.nc", **kw))


def test_loader_accepts_the_files_three_decimal_wavelengths(tmp_path):
    """The ETH/IACETH v4 file stores edges to 3 decimals in um (0.442 for
    22650 cm-1, 1.13e-3 off in wavenumber); that rounding must load."""
    _, _, _, _, _, band_wn = _load(volc_sw_file(
        tmp_path / "v.nc", wl=(np.round(_WL1, 3), np.round(_WL2, 3))))
    assert band_wn.shape == (14, 2)


def test_legacy_column_file_is_not_an_optics_source(tmp_path):
    p = str(tmp_path / "legacy.nc")
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", 3)
        ds.createVariable("lat", "f8", ("lat",))[:] = [-60.0, 0.0, 60.0]
        ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01
    assert _load(p) is None


@pytest.mark.parametrize("table", ["rrtmgp-gas-sw-g112.nc", "rrtmgp-gas-sw-g224.nc"])
def test_band_limit_table_equals_rrtmgp_files(table):
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        _default_data_path,
    )
    from legoesm.forcing.external import _RRTMGP_SW_BAND_LIMITS_CM1
    from legoesm.forcing.external import _open_forcing_dataset
    ds = _open_forcing_dataset(_default_data_path(table))
    lims = np.asarray(ds["bnd_limits_wavenumber"].values)
    ds.close()
    np.testing.assert_array_equal(lims, np.asarray(_RRTMGP_SW_BAND_LIMITS_CM1))


@pytest.mark.skipif(not os.path.exists(_PROD_FILE), reason="production file not staged")
def test_production_file_maps_and_sulfate_scatters():
    _, _, _, mom, _, band_wn = _load(_PROD_FILE)
    assert band_wn.shape == (14, 2)
    tau = mom[..., 0, :].astype(np.float64).sum(axis=2)   # (t, lat, band)
    ssa = mom[..., 1, :].astype(np.float64).sum(axis=2) / tau
    # Visible bands (ascending index 9-11 = 12850-29000 cm-1): non-absorbing.
    assert ssa[..., 9:12].min() > 0.999


def _columns(ncol=2, nlev=10):
    T = jnp.broadcast_to(jnp.linspace(210.0, 290.0, nlev)[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to(jnp.geomspace(5.0e2, 1.0e5, nlev + 1)[None, :],
                              (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    return dict(T=T, p_full=p_full, p_half=p_half,
                sfc_temperature=jnp.full((ncol,), 295.0),
                q_v=jnp.full((ncol, nlev), 2.0e-3),
                cos_zenith=jnp.array([0.6, 0.3])[:ncol],
                aerosol_optical_depth=jnp.full((ncol, nlev), 0.01))


def _volc_moments(kw, tau_b, ssa_b, g_b, layers=slice(1, 4)):
    ncol, nlev = kw["T"].shape
    tau = jnp.zeros((ncol, nlev, 14)).at[:, layers, :].set(jnp.asarray(tau_b))
    return jnp.stack([tau, tau * jnp.asarray(ssa_b),
                      tau * jnp.asarray(ssa_b) * jnp.asarray(g_b)], axis=2)


def _solver():
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    cfg = RRTMGPConfig()
    return RRTMGP.from_legoesm_config(cfg), cfg


def test_grey_moments_reproduce_grey_path():
    """Volcanic tau with the grey ssa/g == the same tau in aerosol_od."""
    solver, cfg = _solver()
    kw = _columns()
    volc = _volc_moments(kw, 0.05, cfg.aerosol_ssa, cfg.aerosol_g)
    a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
    kw_grey = {**kw, "aerosol_optical_depth":
               kw["aerosol_optical_depth"] + volc[:, :, 0, 0]}
    b = solver.solve_columns(**kw_grey)
    for f in ("sw_flux_up", "sw_flux_down", "sw_heating_rate"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=1e-10,
                                   atol=1e-10)


def test_file_optics_change_sw_and_close_budget():
    """Non-absorbing volcanic sulfate (ssa 1) absorbs less SW and reflects
    more than the same optical depth at the grey ssa 0.93; both close the SW
    column budget (TOA net - surface net = sum of layer heating * cp dp / g)."""
    from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rc
    solver, cfg = _solver()
    kw = _columns()
    grey = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=_volc_moments(
        kw, 0.1, cfg.aerosol_ssa, cfg.aerosol_g))
    file_ = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=_volc_moments(
        kw, 0.1, 1.0, 0.72))
    dp = kw["p_half"][:, 1:] - kw["p_half"][:, :-1]

    def absorbed(o):
        net = o.sw_flux_down - o.sw_flux_up
        return net[:, 0] - net[:, -1]

    for o in (grey, file_):
        col = jnp.sum(o.sw_heating_rate * dp, axis=1) * rc.CP_D / rc.G
        incident = o.sw_flux_down[:, 0]
        assert np.all(np.abs(np.asarray(absorbed(o) - col))
                      <= 1e-10 * np.asarray(incident) + 1e-10)
    assert np.all(np.asarray(absorbed(grey) - absorbed(file_)) > 0.5)
    assert np.all(np.asarray(file_.sw_flux_up[:, 0] - grey.sw_flux_up[:, 0]) > 0.5)


def test_jit_parity_and_gradient():
    solver, _ = _solver()
    kw = _columns()

    def toa_up(scale):
        volc = _volc_moments(kw, 0.08 * scale, 0.99, 0.7)
        return jnp.sum(solver.solve_columns(
            **kw, aerosol_volcanic_moments_sw=volc).sw_flux_up[:, 0])

    eager = toa_up(1.0)
    np.testing.assert_allclose(jax.jit(toa_up)(1.0), eager, rtol=1e-12)
    g = jax.grad(toa_up)(1.0)
    assert np.isfinite(g) and g > 0.0
    fd = (toa_up(1.0 + 1e-5) - toa_up(1.0 - 1e-5)) / 2e-5
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_wrong_band_count_raises():
    solver, _ = _solver()
    kw = _columns()
    with pytest.raises(ValueError, match="trailing shape"):
        solver.solve_columns(**kw, aerosol_volcanic_moments_sw=jnp.zeros(
            kw["T"].shape + (3, 13)))


@pytest.mark.parametrize("builder", ["_build_gray_radiation_fn",
                                     "_build_none_radiation_fn"])
def test_non_rrtmgp_radiation_refuses(builder):
    from legoesm.driver import physics_pipeline as pp
    from legoesm.driver.config import ExperimentConfig
    fn, _ = getattr(pp, builder)(ExperimentConfig(radiation="gray"))
    kw = _columns()
    ncol, nlev = kw["T"].shape
    args = (kw["T"], kw["p_full"], kw["p_half"], kw["q_v"],
            kw["sfc_temperature"], jnp.zeros(ncol), jnp.zeros(ncol), 180.0,
            43200.0, jnp.full((ncol,), 0.1), jnp.ones(ncol),
            jnp.zeros((ncol, nlev)), jnp.zeros((ncol, nlev)), jnp.zeros(0))
    with pytest.raises(ValueError, match="volcanic SW"):
        fn(*args, aerosol_sw_volc_col=jnp.zeros((ncol, nlev, 3, 14)))


def test_probe_runs_on_synthetic_file(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe", _REPO / "scripts/validate/probe_volcanic_sw_optics.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.main(volc_sw_file(tmp_path / "v.nc", omega=np.full(14, 1.0)), 1979, 1)
    out = capsys.readouterr().out
    assert "file-grey" in out and "ssa 1.0000" in out


def test_band_mapping_matches_per_band_grey_path():
    """Absorbing volcanic aerosol in ONE band == the grey aerosol with that
    band's per-band ssa (RRTMGPConfig.aerosol_ssa_bands, an independent path
    through the same g-point -> band map): a wrong band order or a fixed band
    for every g-point fails."""
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    solver, cfg = _solver()
    kw = {**_columns(), "aerosol_optical_depth": jnp.zeros((2, 10))}
    for k in (0, 10):
        ssa = np.ones(14)
        ssa[k] = 0.4
        volc = _volc_moments(kw, 0.2, ssa, cfg.aerosol_g)
        a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
        ref = RRTMGP.from_legoesm_config(cfg._replace(
            aerosol_ssa_bands=tuple(ssa))).solve_columns(
            **{**kw, "aerosol_optical_depth": volc[:, :, 0, 0]})
        np.testing.assert_allclose(a.sw_flux_down, ref.sw_flux_down,
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(a.sw_flux_up, ref.sw_flux_up,
                                   rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scheme", ["gray", "simple_lw"])
def test_integration_non_rrtmgp_refuses(scheme):
    from legoesm.atmosphere.physics.radiation import integration
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    kw = _columns()
    with pytest.raises(ValueError, match="aerosol_sw_volc"):
        integration._call_radiation_backend(
            RadiationConfig(scheme=scheme), kw["T"], kw["p_full"], kw["p_half"],
            kw["sfc_temperature"], jnp.zeros(2), kw["q_v"], jnp.full(2, 400.0),
            aerosol_sw_volc=jnp.zeros((2, 10, 3, 14)))


def test_volcanic_chunking_is_exact(tmp_path, monkeypatch):
    from legoesm.forcing import surface_utils as su
    from legoesm.forcing.external import AerosolConfig
    p = volc_sw_file(tmp_path / "v.nc", omega=np.linspace(0.3, 1.0, 14))
    acfg = AerosolConfig(enabled=True, volcanic_enabled=True, volcanic_path=p,
                         use_reference_if_missing=True)
    ncol, nlev = 5, 12
    lat = jnp.radians(jnp.linspace(-70.0, 70.0, ncol))
    p_half = jnp.broadcast_to(jnp.geomspace(1.0e2, 1.0e5, nlev + 1), (ncol, nlev + 1))
    w = su.stratospheric_layer_weight(p_half, lat)
    args = (acfg, 15.0, lat, p_half, w)
    kw = dict(active=True, lw_active=False, dtype=jnp.float64)
    whole = su.aerosol_layer_fields(*args, **kw)[3]
    monkeypatch.setattr(su, "_VOLC_SW_COL_CHUNK", 2)
    np.testing.assert_array_equal(su.aerosol_layer_fields(*args, **kw)[3], whole)
    assert whole.shape == (ncol, nlev, 3, 14) and float(whole[..., 0, :].sum()) > 0
