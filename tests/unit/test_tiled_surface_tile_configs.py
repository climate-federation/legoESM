"""What the tiled surface actually runs on each tile (#1320).

The #1320 quantification probe was retracted because it RECONSTRUCTED the
production configuration instead of reading it: it gave the ice tile a
Monin-Obukhov scheme with a roughness of 1e-3, when the real tiled path runs
the constant-coefficient law there, and it used a gustiness depth the
production deck does not set.  A reconstruction that is wrong in three places
produces a number nobody can act on.

These pin the three tile laws at the single function that now owns them, so
the next probe can call it rather than guess.
"""

from __future__ import annotations

import pytest

from legoesm.driver.physics_pipeline import tiled_surface_tile_configs


@pytest.fixture
def ocean_cfg():
    """An OCEAN tile law with every experiment-level knob set to a NON-default
    value, so a tile that silently ignores one is visible."""
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    return SurfaceLayerConfig()._replace(
        bulk_scheme="coare3",
        gustiness_w_zi=300.0,
        thermo_convention="aerobulk",
    )


def test_the_ice_tile_runs_the_constant_law_not_the_ocean_scheme(ocean_cfg):
    """The single fact the retracted probe got wrong."""
    _oc, ice, _ln = tiled_surface_tile_configs(ocean_cfg, ocean_cfg)
    assert ice.bulk_scheme == "constant", (
        "sea ice runs the constant-coefficient surface layer; giving it the "
        "ocean's scheme is what invalidated the #1320 probe")


def test_the_ice_tile_differs_from_the_ocean_tile_in_EXACTLY_that(ocean_cfg):
    """Nothing else about the ice tile is invented either -- no roughness of
    its own, no gustiness of its own, no thermodynamic convention of its own.

    The inequality matters as much as the equality: codex mutated the helper to
    return the OCEAN config for the ice tile and four of the five tests here
    still passed, this one among them, because "differs in exactly one field"
    is also satisfied by differing in NONE.
    """
    _oc, ice, _ln = tiled_surface_tile_configs(ocean_cfg, ocean_cfg)
    assert ice != ocean_cfg, (
        "the ice tile is the ocean tile -- it must carry the constant law")
    assert ice._replace(bulk_scheme=ocean_cfg.bulk_scheme) == ocean_cfg


def test_the_ocean_tile_is_the_experiment_config_untouched(ocean_cfg):
    """The ocean tile is whatever the run resolved -- the helper must not
    'normalise' a scheme or drop the gustiness depth on the way through."""
    oc, _ice, _ln = tiled_surface_tile_configs(ocean_cfg, ocean_cfg)
    assert oc is ocean_cfg


def test_the_land_tile_is_passed_through_not_derived_here(ocean_cfg):
    """The land law has ONE owner (``_land_tile_surface_cfg``), shared with the
    unified slab surface-energy balance; this helper must not fork it."""
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig
    land = SurfaceLayerConfig()._replace(
        bulk_scheme="most", z0=1.5e-2, gustiness_w_zi=0.0,
        thermo_convention="legoesm")
    _oc, _ice, ln = tiled_surface_tile_configs(ocean_cfg, land)
    assert ln is land


def test_the_pipeline_uses_this_helper_rather_than_its_own_copy():
    """Non-vacuity: if the tiled path stopped calling the helper, these tests
    would pin a function nobody runs -- the exact shape of a gate that cannot
    fail."""
    import inspect
    from legoesm.driver import physics_pipeline as pp

    src = inspect.getsource(pp.PhysicsPipeline._tiled_surface_flux)
    assert "tiled_surface_tile_configs" in src, (
        "_tiled_surface_flux no longer builds its tile configs through the "
        "shared helper, so these assertions no longer describe the run")
    assert 'bulk_scheme="constant"' not in src, (
        "the ice-tile law is being rebuilt inside _tiled_surface_flux again; "
        "that is the second implementation this helper exists to remove")


# ---------------------------------------------------------------------------
# The whole chain, from the experiment config -- nothing left to reconstruct
# ---------------------------------------------------------------------------

def _experiment_config(**overrides):
    from legoesm.driver.config import ExperimentConfig
    return ExperimentConfig()._replace(**overrides)


def test_resolution_from_the_experiment_config_carries_the_surface_knobs():
    """The experiment's surface settings must REACH the ocean tile.

    The retracted probe's second and third errors were here, not in the tile
    derivation: it used a gustiness depth production does not set. Exposing
    only the tile split would have left that guess intact -- both reviewers
    said so, which is why this path exists.
    """
    from legoesm.driver.physics_pipeline import resolve_tiled_surface_configs

    # ExperimentConfig defaults to turbulence="none", which carries NO
    # surface-layer config at all -- pick a scheme that has one, because the
    # question here is whether the experiment's knobs REACH it.
    cfg = _experiment_config(turbulence="louis",
                             surface_bulk_scheme="coare3",
                             surface_gustiness_zi=300.0)
    ocean, ice, land = resolve_tiled_surface_configs(cfg)
    assert ocean is not None, (
        "turbulence='louis' carries a surface layer; resolving to None means "
        "the chain is broken, not that the scheme has no surface")
    assert ocean.bulk_scheme == "coare3"
    assert ocean.gustiness_w_zi == 300.0
    # ... and the tiles derived from it stay consistent
    assert ice.bulk_scheme == "constant"
    assert land.bulk_scheme == "most"
    assert land.gustiness_w_zi == 0.0, (
        "the land tile must not inherit the ocean's convective gustiness")


def test_defaults_resolve_without_inventing_anything():
    from legoesm.driver.physics_pipeline import (
        _DEFAULT_SURFACE_Z0_LAND, resolve_tiled_surface_configs,
    )

    # The DEFAULT experiment selects turbulence="none": no surface layer
    # exists, and the resolver says so rather than inventing one.
    assert resolve_tiled_surface_configs(_experiment_config()) == (
        None, None, None)

    # With a scheme that HAS a surface layer, the defaults resolve and the
    # land roughness is the named default rather than a number someone guessed.
    ocean, ice, land = resolve_tiled_surface_configs(
        _experiment_config(turbulence="louis"))
    assert ocean is not None
    assert ice.bulk_scheme == "constant"
    assert land.z0 == _DEFAULT_SURFACE_Z0_LAND


def test_the_pipeline_land_law_and_the_resolver_agree():
    """One owner: the pipeline's land tile must BE the shared derivation, not a
    copy of it -- the divergence this helper exists to prevent."""
    import inspect
    from legoesm.driver import physics_pipeline as pp

    src = inspect.getsource(pp.PhysicsPipeline._land_tile_surface_cfg)
    assert "land_tile_surface_cfg(" in src
    assert 'bulk_scheme="most"' not in src, (
        "the land law is being rebuilt inside the pipeline again")
