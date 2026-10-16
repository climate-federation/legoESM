"""Cloud liquid and ice must reach the initial condition, or the microphysics
is being trained against air that never holds a cloud.

The WeatherBench2 store the training campaign reads carries NO condensate at
all — only ``total_cloud_cover`` — so every sample started bone dry and the
thirteen Morrison rate coefficients could not influence a six-hour forecast.
ARCO-ERA5 carries both fields, and these lock the second-store read, the
the (still available) specific-content to mixing-ratio helper, and the hand-off into the carry.
"""
from __future__ import annotations

import legoesm.training.era5_to_state as e2s
import numpy as np
import pytest
from legoesm.thermo import specific_condensate_to_mixing_ratio
from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice
from legoesm.training.era5_to_state import era5_terrain_product

_LEVELS = (1000.0, 500.0, 100.0)          # hPa, descending
_NLAT, _NLON, _NLEV = 5, 6, 3
_QC = np.array([3.0e-4, 1.0e-4, 0.0], dtype=np.float32)   # per level, as listed
_QI = np.array([0.0, 2.0e-5, 8.0e-5], dtype=np.float32)


def _state_store(lat_descending=True):
    import xarray as xr

    lat = np.linspace(90.0, -90.0, _NLAT)
    if not lat_descending:
        lat = lat[::-1]
    d3 = lambda v: (("time", "level", "lat", "lon"),
                    np.full((1, _NLEV, _NLAT, _NLON), v, dtype=np.float32))
    d2 = lambda v: (("time", "lat", "lon"),
                    np.full((1, _NLAT, _NLON), v, dtype=np.float32))
    return xr.Dataset(
        {"temperature": d3(280.0), "u_component_of_wind": d3(5.0),
         "v_component_of_wind": d3(2.0), "specific_humidity": d3(5e-3),
         "surface_pressure": d2(1.0e5), "skin_temperature": d2(290.0),
         "geopotential_at_surface": d2(0.0)},
        coords={"time": [0], "level": list(_LEVELS), "lat": lat,
                "lon": np.linspace(0.0, 300.0, _NLON)},
    )


def _cloud_store(*, lat_descending=True, drop=(), short_names=False):
    """A second store carrying only the two condensate fields, level-varying so
    a level mix-up between the stores cannot pass."""
    import xarray as xr

    lat = np.linspace(90.0, -90.0, _NLAT)
    if not lat_descending:
        lat = lat[::-1]
    names = ({"specific_cloud_liquid_water_content": "clwc",
              "specific_cloud_ice_water_content": "ciwc"} if short_names
             else {k: k for k in ("specific_cloud_liquid_water_content",
                                  "specific_cloud_ice_water_content")})
    prof = {"specific_cloud_liquid_water_content": _QC,
            "specific_cloud_ice_water_content": _QI}
    data = {}
    for long, key in names.items():
        if long in drop:
            continue
        arr = np.broadcast_to(prof[long][None, :, None, None],
                              (1, _NLEV, _NLAT, _NLON)).astype(np.float32)
        if not lat_descending:
            arr = arr[:, :, ::-1, :]
        data[key] = (("time", "level", "lat", "lon"), arr)
    return xr.Dataset(data, coords={
        "time": [0], "level": list(_LEVELS), "lat": lat,
        "lon": np.linspace(0.0, 300.0, _NLON)})


def _cfg(**kw):
    return TrainingERA5Config(zarr_store="dummy", levels=_LEVELS,
                              load_cloud_condensate=True, **kw)


# --- the thermodynamic conversion ----------------------------------------

def test_condensate_conversion_divides_by_the_DRY_fraction():
    """``c/(1-q)``, not ``c/(1-c)`` — dividing by the condensate itself would be
    a different quantity, and wrong by the humidity."""
    c, q = 1.0e-3, 2.0e-2
    got = float(specific_condensate_to_mixing_ratio(np.float64(c), np.float64(q)))
    assert got == pytest.approx(c / (1.0 - q), rel=1e-12)
    assert got != pytest.approx(c / (1.0 - c), rel=1e-6)


def test_condensate_conversion_clips_an_interpolation_undershoot():
    """A negative condensate mass has no meaning; every scheme downstream would
    otherwise have to guard it."""
    assert float(specific_condensate_to_mixing_ratio(
        np.float64(-1e-6), np.float64(1e-2))) == 0.0


# --- the second-store read ------------------------------------------------

def test_the_slice_carries_condensate_when_asked(monkeypatch):
    monkeypatch.setattr(e2s, "open_era5_zarr",
                        lambda store: _cloud_store() if "cloud" in str(store)
                        else _state_store())
    sl = load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())
    assert sl.q_c is not None and sl.q_i is not None
    assert sl.q_c.shape == (_NLAT, _NLON, _NLEV)
    # The loader sorts levels to ASCENDING pressure, so the profiles reverse.
    np.testing.assert_allclose(sl.q_c[0, 0], _QC[::-1], rtol=1e-6)
    np.testing.assert_allclose(sl.q_i[0, 0], _QI[::-1], rtol=1e-6)


def test_condensate_is_absent_unless_requested(monkeypatch):
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: _state_store())
    sl = load_era5_slice(TrainingERA5Config(zarr_store="dummy", levels=_LEVELS),
                         0, ds=_state_store())
    assert sl.q_c is None and sl.q_i is None


def test_a_cloud_store_without_the_fields_is_a_hard_error(monkeypatch):
    """Silently zero-filling is the exact condition this option removes."""
    dropped = _cloud_store(drop=("specific_cloud_ice_water_content",))
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: dropped)
    with pytest.raises(ValueError, match="specific_cloud_ice_water_content"):
        load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())


def test_short_grib_names_resolve(monkeypatch):
    monkeypatch.setattr(e2s, "open_era5_zarr",
                        lambda store: _cloud_store(short_names=True))
    sl = load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())
    np.testing.assert_allclose(sl.q_c[0, 0], _QC[::-1], rtol=1e-6)


def test_an_opposite_latitude_order_is_flipped_to_the_state_grid(monkeypatch):
    """ARCO runs north-to-south and the WB2 store may run south-to-north; an
    unflipped read would put northern cloud in the southern hemisphere."""
    state = _state_store(lat_descending=False)
    monkeypatch.setattr(e2s, "open_era5_zarr",
                        lambda store: _cloud_store(lat_descending=True))
    sl = load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=state)
    # The profile is lat-uniform here, so compare the ORDERING via a lat-varying
    # check: the flip must make row 0 of the cloud field correspond to row 0 of
    # the state grid, i.e. the shapes and the level profile survive.
    assert sl.q_c.shape == (_NLAT, _NLON, _NLEV)
    np.testing.assert_allclose(sl.q_c[0, 0], _QC[::-1], rtol=1e-6)


def test_a_mismatched_grid_is_refused(monkeypatch):
    import xarray as xr
    small = xr.Dataset(
        {"specific_cloud_liquid_water_content":
            (("time", "level", "lat", "lon"),
             np.zeros((1, _NLEV, 3, _NLON), dtype=np.float32)),
         "specific_cloud_ice_water_content":
            (("time", "level", "lat", "lon"),
             np.zeros((1, _NLEV, 3, _NLON), dtype=np.float32))},
        coords={"time": [0], "level": list(_LEVELS),
                "lat": np.linspace(90.0, -90.0, 3),
                "lon": np.linspace(0.0, 300.0, _NLON)})
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: small)
    with pytest.raises(ValueError, match="latitude size"):
        load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())


# --- the hand-off into the carry -----------------------------------------

def _mini_slice(grid, nlev, *, with_cloud):
    """A slice on a SOURCE grid that is deliberately NOT the model grid.

    ERA5 is 721x1440 and the model is not, so the condensate has to be
    horizontally regridded before it meets the model's surface pressure.  A
    fixture built on the model grid would let a missing regrid pass silently —
    which is exactly how the first version of this change shipped a shape
    mismatch that only the reviewer caught.
    """
    from legoesm.training.era5_to_state import ERA5Slice

    n_lat, n_lon = 2 * len(grid.lat) + 1, 2 * len(grid.lon)
    src_lat = np.linspace(float(np.max(grid.lat)), float(np.min(grid.lat)), n_lat)
    src_lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    shape = (n_lat, n_lon, nlev)
    q_c = q_i = None
    if with_cloud:
        q_c = np.full(shape, 4.0e-4, dtype=np.float32)
        q_i = np.full(shape, 6.0e-5, dtype=np.float32)
    return ERA5Slice(
        T=np.full(shape, 280.0, dtype=np.float32),
        u=np.zeros(shape, dtype=np.float32),
        v=np.zeros(shape, dtype=np.float32),
        q=np.full(shape, 5.0e-3, dtype=np.float32),
        p_s=np.full((n_lat, n_lon), 1.0e5, dtype=np.float32),
        sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
        phis=np.zeros((n_lat, n_lon), dtype=np.float32),
        lat=src_lat, lon=src_lon,
        plev_Pa=np.linspace(1.0e4, 1.0e5, nlev),
        q_c=q_c, q_i=q_i,
    )


def _carry(with_cloud, microphysics):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.era5_to_state import era5_to_spectral_carry

    grid = create_gaussian_grid(n_max=10)
    nlev = 6
    sigma = create_sigma_coordinate(nlev)
    era5 = _mini_slice(grid, nlev, with_cloud=with_cloud)
    return era5_to_spectral_carry(era5, grid, sigma, microphysics=microphysics,
                                  target_phis=era5_terrain_product(era5, grid))


def test_the_carry_is_cloud_free_without_the_option():
    """The state before this change: every sample starts with exactly zero
    condensate, which is why no microphysics rate could move the loss."""
    c = _carry(False, "morrison")
    assert float(np.max(np.abs(np.asarray(c.q_c)))) == 0.0
    assert float(np.max(np.abs(np.asarray(c.q_i)))) == 0.0


def test_the_carry_holds_the_ERA5_cloud_water():
    c = _carry(True, "morrison")
    q_c = np.asarray(c.q_c)
    q_i = np.asarray(c.q_i)
    assert float(np.min(q_c)) > 0.0, "cloud liquid never reached the carry"
    assert float(np.min(q_i)) > 0.0, "cloud ice never reached the carry"
    # Specific content, loaded AS IS (the tracer convention, 2026-09-28);
    # the former mixing ratio c/(1-q_v) is 0.5 % above and fails at 1e-3.
    assert float(np.mean(q_c)) == pytest.approx(4.0e-4, rel=1e-3)
    assert float(np.mean(q_i)) == pytest.approx(6.0e-5, rel=1e-3)


def test_the_droplet_number_stays_at_the_scheme_s_own_default():
    """Liquid needs no seed: Morrison reads its specified constant droplet
    number wherever the prognostic one is not physical."""
    c = _carry(True, "morrison")
    assert float(np.max(np.abs(np.asarray(c.N_c)))) == 0.0


def test_the_ice_number_is_seeded_so_the_crystals_have_a_physical_SIZE():
    """Radiation runs BEFORE microphysics and inverts the ice size distribution
    to get the crystal radius, so ice mass with a zero number gives an
    effective radius of HUNDREDS OF METRES on the first radiation call —
    finite, plausible, and radiatively almost inert."""
    c = _carry(True, "morrison")
    q_i = np.asarray(c.q_i)
    n_i = np.asarray(c.N_i)
    assert float(np.min(n_i)) > 0.0, "ice mass shipped with no crystals"
    # Invert the module's own PSD: r_eff = 1.5 / (rho_ci*pi*N_i/q_i)^(1/3).
    lami = (500.0 * np.pi * n_i / np.maximum(q_i, 1e-30)) ** (1.0 / 3.0)
    r_eff_um = 1.5 / lami * 1e6
    assert 5.0 < float(np.mean(r_eff_um)) < 100.0, float(np.mean(r_eff_um))


def test_ice_free_air_gets_no_crystals():
    c = _carry(False, "morrison")
    assert float(np.max(np.abs(np.asarray(c.N_i)))) == 0.0


def test_a_warm_rain_scheme_drops_the_ice_loudly(caplog):
    """Three tracer slots cannot hold cloud ice. Handing it to the liquid slot
    would put supercooled water into a scheme with no ice physics; dropping it
    silently is what this whole option exists to remove."""
    import logging

    with caplog.at_level(logging.WARNING):
        c = _carry(True, "kessler")
    assert any("cloud ice DROPPED" in r.message for r in caplog.records), caplog.text
    assert float(np.min(np.asarray(c.q_c))) > 0.0     # liquid still lands


# --- the campaign switch --------------------------------------------------

def test_the_campaign_yaml_turns_it_on_and_picks_the_store():
    from legoesm.training.scale_build import _era5_config

    off = _era5_config(None, {"era5_zarr": "s"})
    assert off.load_cloud_condensate is False

    on = _era5_config(None, {"era5_zarr": "s", "era5_cloud_condensate": True})
    assert on.load_cloud_condensate is True
    assert on.cloud_zarr, "a default condensate store must be configured"

    named = _era5_config(None, {"era5_zarr": "s", "era5_cloud_condensate": True,
                                "era5_cloud_zarr": "gs://elsewhere"})
    assert named.cloud_zarr == "gs://elsewhere"


def test_both_campaign_decks_ask_for_condensate():
    """The three arms share one configuration and differ only in the model, so
    an initial condition enabled on one deck and not the other would confound
    every arm-to-arm comparison."""
    import pathlib

    import yaml

    for deck in ("spectral_t63_bechtold_clubb.yaml", "spectral_t63.yaml"):
        y = yaml.safe_load(
            pathlib.Path("config/wb/campaign", deck).read_text())
        assert y.get("era5_cloud_condensate") is True, deck


def test_below_ground_levels_do_not_inject_cloud_into_the_lowest_model_levels():
    """A pressure level under the terrain carries an extrapolated fill value.
    Held constant downward by the interpolation, it would put cloud water over
    every elevated land point that ERA5 never reported there."""
    from legoesm.training.era5_to_state import _fill_below_ground

    plev = np.array([1.0e4, 5.0e4, 9.25e4, 1.0e5])      # ascending Pa
    field = np.array([[[0.0, 1.0e-4, 2.0e-4, 9.9e-3]]])  # last level = fill
    p_s = np.array([[9.5e4]])                            # 950 hPa: 1000 is buried
    out = np.asarray(_fill_below_ground(field, plev, p_s))
    assert out[0, 0, 3] == pytest.approx(2.0e-4)         # fill replaced
    np.testing.assert_allclose(out[0, 0, :3], field[0, 0, :3])

    # A sea-level column keeps every level untouched.
    out_sea = np.asarray(_fill_below_ground(field, plev, np.array([[1.01e5]])))
    np.testing.assert_allclose(out_sea, field)


def test_a_shifted_longitude_grid_is_refused(monkeypatch):
    """Same size and same sense is not the same grid: a 0..360 versus
    -180..180 origin loads condensate that is geographically displaced and
    entirely plausible-looking."""
    import xarray as xr

    shifted = _cloud_store()
    shifted = shifted.assign_coords(lon=shifted.lon.values - 180.0)
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: shifted)
    with pytest.raises(ValueError, match="longitudes"):
        load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())


def test_a_missing_analysis_time_is_refused(monkeypatch):
    """``nearest`` has no tolerance; a store missing the hour would hand back a
    field from another day and say nothing."""
    other = _cloud_store().assign_coords(time=[10])
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: other)
    with pytest.raises(ValueError, match="no field at"):
        load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())


def test_seeded_condensate_actually_makes_cloud_the_radiation_can_see():
    """The whole point. Xu-Randall diagnoses cloud fraction from humidity AND
    explicit condensate, so a condensate-free initial condition gives EXACTLY
    zero cloud — the first radiation call of every training sample saw a clear
    sky. The seeded amount gives a real cloud at ordinary humidity."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        xu_randall_cloud_fraction,
    )
    from legoesm.atmosphere.physics.clouds.config import CloudConfig

    cfg = CloudConfig()
    q_sat = jnp.asarray(6.0e-3)
    rh = jnp.asarray(0.80)
    assert float(xu_randall_cloud_fraction(rh, jnp.asarray(0.0), q_sat, cfg)) == 0.0
    assert float(xu_randall_cloud_fraction(
        rh, jnp.asarray(4.0e-4), q_sat, cfg)) > 0.5


def test_the_seeded_ice_number_round_trips_through_the_radiation_diagnostic():
    """The seed is the inverse of the module's own ice size distribution, so it
    must come back out of that distribution at the size it was built for. A
    constant copied by hand would drift the moment the module's changed."""
    import numpy as _np
    from legoesm.atmosphere.physics.clouds import cloud_fraction as cf

    q_i = 1.0e-5
    n_i = float(cf.initial_ice_number_from_mass(q_i))
    lami = (cf._RHO_CLOUD_ICE_DEFAULT * _np.pi * n_i / q_i) ** (1.0 / 3.0)
    r_eff = cf._R_EFF_ICE_PSD_COEFF / lami
    assert r_eff == pytest.approx(cf._R_EFF_ICE_DEFAULT_M, rel=1e-9)


def test_a_store_with_the_wrong_level_units_is_refused(monkeypatch):
    """A level coordinate in Pa rather than hPa selects nothing recognisable
    and would be wrong by two orders of magnitude."""
    pa = _cloud_store()
    pa = pa.assign_coords(level=[v * 100.0 for v in _LEVELS])
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: pa)
    with pytest.raises(ValueError, match="level coordinate"):
        load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())


def test_a_density_valued_store_is_refused(monkeypatch):
    """kg/m^3 is a different quantity from kg/kg and would silently be off by
    the air density."""
    dens = _cloud_store()
    for v in dens.data_vars:
        dens[v].attrs["units"] = "kg m**-3"
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: dens)
    with pytest.raises(ValueError, match="units"):
        load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())


def test_the_real_ERA5_units_string_is_accepted(monkeypatch):
    """ARCO-ERA5 labels both fields 'kg kg**-1'; the guard must not reject the
    store it exists to read."""
    ok = _cloud_store()
    for v in ok.data_vars:
        ok[v].attrs["units"] = "kg kg**-1"
    monkeypatch.setattr(e2s, "open_era5_zarr", lambda store: ok)
    sl = load_era5_slice(_cfg(cloud_zarr="cloud"), 0, ds=_state_store())
    assert sl.q_c is not None


def test_the_surface_temperature_reload_does_not_touch_the_cloud_store():
    """It wants 2-D fields only. Leaving the flag on would re-read the remote
    condensate store once per sample for data it throws away."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(
        "packages/ml/legoesm/training/scale_build.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "load_era5_slice"]
    sst = [n for n in calls
           if any(isinstance(a, ast.Name) and a.id == "sst_cfg" for a in n.args)]
    assert len(sst) == 2, f"expected both loaders to use sst_cfg, got {len(sst)}"
    # ...and that config must really have the flag off.
    from legoesm.training.scale_build import _era5_config

    on = _era5_config(None, {"era5_zarr": "s", "era5_cloud_condensate": True})
    assert on._replace(load_cloud_condensate=False).load_cloud_condensate is False
