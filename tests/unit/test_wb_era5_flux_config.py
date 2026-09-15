"""WeatherBench ERA5-surface-flux + spatial-embedding config plumbing.

Pins the scale_build helpers (``wb_needs_land_frac``, ``_era5_config``,
campaign-YAML key validation), the committed campaign decks' flag settings
(spatial embedding ON, prescribed fluxes OFF), and the inert_params
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

def test_wb_needs_land_frac_follows_fluxes_or_the_learned_arm_flag():
    assert scale_build.wb_needs_land_frac("physics", {"era5_surface_fluxes": True})
    assert not scale_build.wb_needs_land_frac("physics", {"era5_surface_fluxes": False})
    assert not scale_build.wb_needs_land_frac("physics", {})
    assert scale_build.wb_needs_land_frac(
        "neural_gcm", {"neural_gcm": {"spatial_embedding": True}})
    assert scale_build.wb_needs_land_frac(
        "sfno", {"sfno": {"spatial_embedding": True}})
    assert not scale_build.wb_needs_land_frac(
        "sfno", {"neural_gcm": {"spatial_embedding": True}})
    # the classical arm never reads the learned-arm block
    assert not scale_build.wb_needs_land_frac(
        "physics", {"sfno": {"spatial_embedding": True}})


def test_era5_config_maps_the_flux_keys():
    base = {"era5_zarr": "dummy"}
    cfg = scale_build._era5_config(None, base)
    assert cfg.load_surface_fluxes is False
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
    """A knob ships with a config that selects it (CLAUDE.md), and the
    spectral campaign default keeps the spatial embedding ON."""
    flags = {d.name: yaml.safe_load(d.read_text()) for d in DECKS}
    assert any(y["era5_surface_fluxes"] is True for y in flags.values())
    assert flags["spectral_t63.yaml"]["neural_gcm"]["spatial_embedding"] is True
    assert flags["spectral_t63.yaml"]["sfno"]["spatial_embedding"] is True
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


# ---------------------------------------------------------------- core gate

@pytest.mark.parametrize("mode,yml_extra", [
    ("physics", {"era5_surface_fluxes": True}),
    ("neural_gcm", {"neural_gcm": {"spatial_embedding": True}}),
    ("sfno", {"sfno": {"spatial_embedding": True}}),
])
def test_latlon_core_refuses_the_spectral_only_options(mode, yml_extra):
    from types import SimpleNamespace

    yml = yaml.safe_load(DECKS[0].read_text())
    yml["neural_gcm"]["spatial_embedding"] = False
    yml["sfno"]["spatial_embedding"] = False
    for k, v in yml_extra.items():
        if isinstance(v, dict):
            yml[k].update(v)
        else:
            yml[k] = v
    cfg = SimpleNamespace(mode=mode, training_core="latlon")
    with pytest.raises(SystemExit, match="spectral"):
        scale_build.build_mode_components(cfg, yml)


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
