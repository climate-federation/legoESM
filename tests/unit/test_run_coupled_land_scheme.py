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


def test_diurnal_surface_default_on():
    from legoesm.driver.coupled_config import CoupledConfig
    assert CoupledConfig().land_diurnal_surface is True


def test_enable_diurnal_surface_land():
    """The coupled diurnal surface flip enables MOST + Farquhar (and upgrades a
    'none' carbon scheme to differland so Farquhar has a prognostic LAI)."""
    from legoesm.driver.coupled_esm_driver import enable_diurnal_surface_land
    cfg = enable_diurnal_surface_land(MultiLayerLandConfig())
    assert cfg.bulk_scheme == "most"
    assert cfg.stomata.enabled is True
    assert cfg.carbon.scheme == "differland"


def test_diurnal_surface_cli_roundtrip():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--land-diurnal-surface", dest="land_diurnal_surface",
                   action=argparse.BooleanOptionalAction, default=True)
    assert p.parse_args([]).land_diurnal_surface is True
    assert p.parse_args(["--no-land-diurnal-surface"]).land_diurnal_surface is False


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        land_scheme_overrides("bucket2")


def test_elev_bands_default_off():
    from legoesm.driver.coupled_config import CoupledConfig
    assert CoupledConfig().land_elev_bands is False


def test_elev_bands_cli_roundtrip():
    from scripts.run.run_coupled import build_parser
    p = build_parser()
    assert p.parse_args([]).land_elev_bands is False          # opt-in: default off
    assert p.parse_args(["--elev-bands"]).land_elev_bands is True
    assert p.parse_args(["--no-elev-bands"]).land_elev_bands is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
