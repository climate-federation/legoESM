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


# ---------------------------------------------------------------- YAML keys

def test_decks_found():
    assert DECKS, "no campaign decks under config/wb/campaign"


@pytest.mark.parametrize("deck", DECKS, ids=[d.name for d in DECKS])
def test_committed_decks_set_the_new_flags(deck):
    yml = yaml.safe_load(deck.read_text())
    assert yml["era5_surface_fluxes"] is False
    assert yml["neural_gcm"]["spatial_embedding"] is True
    assert yml["sfno"]["spatial_embedding"] is True
    scale_build.validate_wb_campaign_yaml(yml)


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
