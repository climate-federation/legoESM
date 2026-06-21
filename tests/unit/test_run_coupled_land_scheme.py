"""--land-scheme override mapping for run_coupled (coupler land-tile selection)."""
import pytest

from scripts.run.run_coupled import land_scheme_overrides, _LAND_SCHEMES
from legoesm.land.config import LandConfig, MultiLayerLandConfig


def test_multilayer_selects_richards_config():
    ov = land_scheme_overrides("multilayer")
    assert ov["land_mode"] == "multilayer"
    assert isinstance(ov["land_config"], MultiLayerLandConfig)


def test_slab_selects_slab_config():
    ov = land_scheme_overrides("slab")
    assert ov["land_mode"] == "slab"
    assert isinstance(ov["land_config"], LandConfig)
    assert not isinstance(ov["land_config"], MultiLayerLandConfig)


def test_mode_and_config_type_agree():
    # the coupler dispatches on config TYPE; mode label must match it for every scheme
    for s in _LAND_SCHEMES:
        ov = land_scheme_overrides(s)
        is_ml = isinstance(ov["land_config"], MultiLayerLandConfig)
        assert (ov["land_mode"] == "multilayer") == is_ml


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        land_scheme_overrides("bucket2")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
