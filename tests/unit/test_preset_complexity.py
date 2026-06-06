"""preset_complexity: the model-wide ModelComplexity dial reaching CoupledConfig."""

from __future__ import annotations

import pytest

from legoesm.components import ModelComplexity
from legoesm.driver.coupled_config import CoupledConfig, preset_complexity
from legoesm.land import LandConfig, MultiLayerLandConfig


def test_idealized_is_fixed_sst_and_slab_land() -> None:
    cfg = preset_complexity(ModelComplexity.IDEALIZED)
    assert isinstance(cfg, CoupledConfig)
    assert cfg.ocean_config.mode == "fixed"
    assert cfg.land_mode == "slab"
    assert isinstance(cfg.land_config, LandConfig)


def test_intermediate_is_slab_ocean_and_multilayer_land() -> None:
    cfg = preset_complexity(ModelComplexity.INTERMEDIATE)
    assert cfg.ocean_config.mode == "slab"
    assert cfg.land_mode == "multilayer"
    assert isinstance(cfg.land_config, MultiLayerLandConfig)


def test_full_is_rejected_pointing_at_the_resolver() -> None:
    """The coupled driver has no full-3D ocean slot (SimpleOceanConfig only)."""
    with pytest.raises(ValueError, match="full"):
        preset_complexity(ModelComplexity.FULL)


def test_string_level_accepted() -> None:
    cfg = preset_complexity("idealized")
    assert cfg.ocean_config.mode == "fixed"


def test_overrides_applied_last() -> None:
    cfg = preset_complexity(
        ModelComplexity.IDEALIZED, carbon_active=True, f_land_mode="zero"
    )
    assert cfg.carbon_active is True
    assert cfg.f_land_mode == "zero"


@pytest.mark.parametrize(
    "level", [ModelComplexity.IDEALIZED, ModelComplexity.INTERMEDIATE]
)
def test_ocean_config_is_accepted_by_make_ocean(level) -> None:
    """The preset's ocean_config builds a real ocean (grounded vs the live factory)."""
    from legoesm.ocean.simple_ocean import make_ocean

    cfg = preset_complexity(level)
    assert callable(make_ocean(cfg.ocean_config))
    # ocean_mode is the decorative log label, kept in the documented domain
    assert cfg.ocean_mode in ("slab", "two_layer")
