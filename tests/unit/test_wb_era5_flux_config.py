"""WeatherBench ERA5-surface-flux + spatial-embedding config plumbing.

Pins the scale_build helpers (``wb_needs_land_frac``, ``_era5_config``,
campaign-YAML key validation), the committed campaign decks' flag settings,
and the inert_params
policy-freeze helpers on the real classical parameter pytree.
"""
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import pytest
import yaml
from legoesm.training import inert_params, scale_build
from legoesm.training.model_registry import build_variant

REPO_ROOT = Path(__file__).resolve().parents[2]
DECKS = sorted((REPO_ROOT / "config" / "wb" / "campaign").glob("*.yaml"))


# ---------------------------------------------------------------- helpers

def test_wb_needs_land_frac_follows_physics_fluxes_or_the_learned_arm_flag():
    assert scale_build.wb_needs_land_frac("physics", {"era5_surface_fluxes": True})
    assert scale_build.wb_needs_land_frac("physics", {"era5_surface_fluxes": False})
    assert scale_build.wb_needs_land_frac("physics", {})
    assert scale_build.wb_needs_land_frac(
        "neural_gcm", {"neural_gcm": {"spatial_embedding": True}})
    assert scale_build.wb_needs_land_frac(
        "sfno", {"sfno": {"spatial_embedding": True}})
    assert not scale_build.wb_needs_land_frac(
        "sfno", {"neural_gcm": {"spatial_embedding": True}})
    # the classical arm never reads the learned-arm block
    assert scale_build.wb_needs_land_frac(
        "physics", {"sfno": {"spatial_embedding": True}})


def test_era5_config_maps_the_flux_keys():
    from types import SimpleNamespace

    base = {"era5_zarr": "dummy"}
    cfg = scale_build._era5_config(None, base)
    assert cfg.load_surface_fluxes is False
    assert scale_build._era5_config(SimpleNamespace(mode="physics"), base).load_land_frac
    default_flux_zarr = cfg.flux_zarr
    cfg = scale_build._era5_config(
        None, {**base, "era5_surface_fluxes": True,
               "era5_flux_zarr": "gs://bucket/flux"})
    assert cfg.load_surface_fluxes is True
    assert cfg.flux_zarr == "gs://bucket/flux" != default_flux_zarr
    # An explicit "" means "the state store is the flux store" and must
    # survive the mapping (a falsy check would silently restore ARCO).
    cfg = scale_build._era5_config(None, {**base, "era5_flux_zarr": ""})
    assert cfg.flux_zarr == ""


# ---------------------------------------------------------------- YAML keys

def test_decks_found():
    assert DECKS, "no campaign decks under config/wb/campaign"


@pytest.mark.parametrize("deck", DECKS, ids=[d.name for d in DECKS])
def test_committed_decks_declare_the_new_flags_explicitly(deck):
    """Every campaign deck names both choices (no hidden default) and
    validates; the values are the deck's to choose."""
    yml = yaml.safe_load(deck.read_text())
    assert isinstance(yml["era5_surface_fluxes"], bool)
    assert isinstance(yml["neural_gcm"]["spatial_embedding"], bool)
    assert isinstance(yml["sfno"]["spatial_embedding"], bool)
    scale_build.validate_wb_campaign_yaml(yml)


def test_one_committed_deck_selects_the_prescribed_fluxes():
    """A deck selects prescribed fluxes; the base deck disables embedding."""
    flags = {d.name: yaml.safe_load(d.read_text()) for d in DECKS}
    assert any(y["era5_surface_fluxes"] is True for y in flags.values())
    assert flags["spectral_t63.yaml"]["neural_gcm"]["spatial_embedding"] is False
    assert flags["spectral_t63.yaml"]["sfno"]["spatial_embedding"] is False
    # the lat-lon deck cannot run either option: it must not select them
    latlon = yaml.safe_load(
        (REPO_ROOT / "config" / "wb" / "scale" / "train_07deg.yaml").read_text())
    assert latlon["era5_surface_fluxes"] is False
    assert latlon["neural_gcm"]["spatial_embedding"] is False
    assert latlon["sfno"]["spatial_embedding"] is False


def test_validate_accepts_the_new_keys_and_rejects_a_typo():
    yml = yaml.safe_load(DECKS[0].read_text())
    yml["era5_surface_fluxes"] = True
    yml["era5_flux_zarr"] = "gs://some/zarr"
    scale_build.validate_wb_campaign_yaml(yml)
    bad = dict(yml)
    bad["era5_surface_flux"] = True
    with pytest.raises(SystemExit, match="era5_surface_flux"):
        scale_build.validate_wb_campaign_yaml(bad)
    bad = yaml.safe_load(DECKS[0].read_text())
    bad["sfno"]["spatial_embeding"] = True
    with pytest.raises(SystemExit, match="spatial_embeding"):
        scale_build.validate_wb_campaign_yaml(bad)


# ---------------------------------------------------------------- policy freeze

_SURFACE_KNOBS = (
    "surface_Cd_neutral", "surface_Ch_neutral", "surface_z0",
    "surface_z0h_z0_ratio", "rrtmgp_sfc_albedo", "rrtmgp_sfc_emissivity",
)


def _leaf_names(tree):
    arrays = eqx.partition(tree, eqx.is_inexact_array)[0]
    return [jax.tree_util.keystr(p)
            for p, _ in jax.tree.leaves_with_path(arrays)]


def test_prescribed_surface_frozen_names_are_exactly_the_surface_leaves():
    params = build_variant("classical", nlev=4)
    names = inert_params.prescribed_surface_frozen_names(params)
    assert names == sorted(names)
    for knob in _SURFACE_KNOBS:
        assert sum(knob in n for n in names) >= 1, knob
    # nothing else: every returned name carries a marker, and a non-surface
    # knob (a convection timescale) is NOT in the list
    markers = inert_params.PRESCRIBED_SURFACE_LEAF_MARKERS
    assert all(any(m in n for m in markers) for n in names)
    assert not any("tiedtke" in n for n in names)
    assert any("tiedtke" in n for n in _leaf_names(params))


def test_apply_forced_freeze_removes_the_surface_leaves_from_the_trainable_set():
    params = build_variant("classical", nlev=4)
    forced = inert_params.prescribed_surface_frozen_names(params)
    arr, static, merged = inert_params.apply_forced_freeze(params, [], forced)
    all_names = _leaf_names(params)
    live = _leaf_names(arr)
    assert len(live) == len(all_names) - len(forced)
    assert not any(any(m in n for m in inert_params.PRESCRIBED_SURFACE_LEAF_MARKERS)
                   for n in live)
    assert merged == sorted(forced)
    # recombining gives back the original pytree
    assert eqx.tree_equal(eqx.combine(arr, static), params)


def test_apply_forced_freeze_unions_with_the_measured_list():
    params = build_variant("classical", nlev=4)
    names = _leaf_names(params)
    measured = [names[0]]
    forced = inert_params.prescribed_surface_frozen_names(params)
    arr, _static, merged = inert_params.apply_forced_freeze(
        params, measured + forced[:1], forced)
    assert merged == sorted(set(measured) | set(forced))
    assert len(_leaf_names(arr)) == len(names) - len(merged)


# ---------------------------------------------------------------- shared forcing

def test_build_spectral_forcing_carries_exactly_what_the_slice_carries():
    import jax.numpy as jnp
    import numpy as np
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.era5_to_state import ERA5Slice

    grid = create_gaussian_grid(n_max=8)
    n_lat, n_lon = 9, 12
    lat = np.deg2rad(np.linspace(90.0, -90.0, n_lat))
    lon = np.deg2rad(np.linspace(0.0, 330.0, n_lon))
    z3 = np.zeros((n_lat, n_lon, 3), np.float32)
    two = np.full((n_lat, n_lon), 2.0, np.float32)
    base = dict(T=z3, u=z3, v=z3, q=z3, p_s=two, sst=np.full((n_lat, n_lon), 290.0, np.float32),
                phis=two, lat=lat, lon=lon, plev_Pa=np.array([300.0, 500.0, 850.0]))
    t = np.datetime64("2019-03-01T06:00")
    plain = scale_build.build_spectral_forcing(ERA5Slice(**base), grid, t, 2019)
    assert set(plain) == {"T_sfc", "sic", "day_of_year", "seconds_of_day"}
    ncol = len(grid.lat) * len(grid.lon)
    assert plain["T_sfc"].shape == (ncol,)
    fluxy = scale_build.build_spectral_forcing(
        ERA5Slice(**base, sfc_shf=two, sfc_lhf=two, sfc_tau_x=two,
                  sfc_tau_y=-two, sfc_sw_up=two, sfc_sw_down=two,
                  sfc_lw_up=two, land_frac=np.full((n_lat, n_lon), 1.5, np.float32)),
        grid, t, 2019)
    assert set(fluxy) == set(plain) | {
        "sfc_shf", "sfc_lhf", "sfc_tau_x", "sfc_tau_y", "sfc_sw_up",
        "sfc_sw_down", "sfc_lw_up", "land_frac"}
    assert float(jnp.max(fluxy["sfc_tau_y"])) == -2.0
    assert float(jnp.max(fluxy["land_frac"])) == 1.0  # clipped to [0, 1]
    assert fluxy["land_frac"].shape == (ncol,)

    # Pin mask columns to the existing SST convention on a non-square grid.
    field = np.linspace(0.1, 0.9, n_lat * n_lon).reshape(n_lat, n_lon)
    ordered = scale_build.build_spectral_forcing(
        ERA5Slice(**{**base, "sst": field}, land_frac=field), grid, t, 2019)
    from legoesm.atmosphere.physics.convection.integration import land_fraction_for_columns

    assert grid.n_lat != grid.n_lon
    mask = land_fraction_for_columns(grid._replace(land_frac=ordered["land_frac"]), ncol)
    np.testing.assert_array_equal(mask, ordered["T_sfc"])
    # A transposed/F-order flatten must not satisfy the same comparison.
    scrambled = np.asarray(mask).reshape(grid.n_lat, grid.n_lon).T.reshape(-1)
    assert not np.array_equal(scrambled, ordered["T_sfc"])


def test_classical_spectral_sample_land_fraction_reaches_convection(monkeypatch, caplog):
    """Real ERA5 extraction/regridding and rollout; only expensive physics is stubbed."""
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.physics.convection.config import BechtoldConfig
    from legoesm.atmosphere.physics.convection.integration import land_fraction_for_columns
    from legoesm.training import aimip_params, era5_to_state
    from scripts.data.make_synthetic_era5 import build_synthetic_era5_dataset
    from tests.unit.test_spectral_surface_and_moisture import _marked_physics, _rad_from_calendar

    ds = build_synthetic_era5_dataset(nlat=8, nlon=12, ntime=2)
    ds = ds.assign_coords(time=np.array(["2019-01-01", "2019-01-01T06"],
                                       dtype="datetime64[ns]"))
    ds["land_sea_mask"] = (("lat", "lon"), np.broadcast_to(
        np.linspace(0.0, 1.0, 8)[:, None], (8, 12)))
    monkeypatch.setattr(era5_to_state, "open_era5_zarr", lambda _: ds)
    cfg = SimpleNamespace(mode="physics", training_core="spectral", smoke=True,
                          multi_step_hours=(6,), n_days=1)
    yml = {"era5_zarr": "synthetic", "era5_surface_fluxes": False,
           "train_years": [2019], "radiation": "gray",
           "n_lat": 8, "n_lon": 16, "nlev": 4,
           "spectral": {"n_max": 5, "dt": 60.0}, "loss": {},
           "classical": {"convection": "bechtold",
                         "trainable_schemes": False}}
    scheme = BechtoldConfig(use_ifs_land_rhebc=True)
    calls = []

    def factory(*args, **kwargs):
        fn = _marked_physics(0.0)
        original = fn.with_phys_state

        def with_state(state, physics_grid, sigma, ps):
            lf = land_fraction_for_columns(physics_grid, ncol, scheme)
            assert lf is not None, "convection received no land fraction"
            assert lf.shape == (ncol,)
            calls.append(True)
            # A changed/misordered mask poisons the forecast, even under JIT.
            tend, ps = original(state, physics_grid, sigma, ps)
            return tend._replace(T_hat=tend.T_hat.replace(data=jnp.where(
                jnp.all(lf == forcing["land_frac"]), tend.T_hat.data, jnp.nan))), ps

        fn.with_phys_state = with_state
        return fn, _rad_from_calendar(0.0)

    monkeypatch.setattr(aimip_params, "make_aimip_classical_spectral_physics", factory)
    _, grid, sigma, params, make_run, _, _ = scale_build.build_mode_components(cfg, yml)
    [(ic, _, forcing)] = scale_build._load_era5_samples_spectral(cfg, yml, grid, sigma)
    ncol = grid.n_lat * grid.n_lon
    assert "land_frac" in forcing, "classical sample dropped ERA5 land fraction"
    lf = np.asarray(forcing["land_frac"])
    assert lf.shape == (ncol,) and np.isfinite(lf).all()
    assert 0.0 <= lf.min() < lf.max() <= 1.0
    run = make_run(params)
    for missing in (None, {k: v for k, v in forcing.items() if k != "land_frac"},
                    {**forcing, "land_frac": None}):
        with pytest.raises(ValueError, match="_load_era5_samples_spectral"):
            run.raw(ic, 1, missing)
    result = jax.jit(lambda ic, forcing: run.raw(ic, 1, forcing))(ic, forcing)
    assert np.isfinite(np.asarray(result.T)).all()
    assert calls, "convection land-fraction consumer never ran"
    assert "grid carries no land_frac" not in caplog.text
    assert getattr(grid, "land_frac", None) is None  # no shared-grid mutation
