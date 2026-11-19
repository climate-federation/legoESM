"""#2574: every lane whose surface fluxes read the stable MOST selector pins
Beljaars-Holtslag 1991 in its committed deck."""
from __future__ import annotations

import pathlib

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config

_REPO = pathlib.Path(__file__).resolve().parents[2]
_BH91 = "beljaars_holtslag1991"


def _parser(driver):
    if driver == "amip":
        from scripts.run import run_amip
        return run_amip.build_arg_parser()
    if driver == "coupled":
        from scripts.run import run_coupled
        return run_coupled.build_parser()
    raise ValueError(driver)


@pytest.mark.parametrize("deck, driver", [
    ("config/amip/amip_sota.yaml", "amip"),
    ("config/amip/amip_realism_c12.yaml", "amip"),
    ("config/cmip/cmip_ocean_3D.yaml", "coupled"),
    ("config/cmip/cmip_ocean_slab.yaml", "coupled"),
])
def test_deck_pins_bh91(deck, driver):
    cfg = load_yaml_config(str(_REPO / deck), _parser(driver))
    assert cfg["surface_stability_scheme"] == _BH91


def test_omip_deck_pins_bh91():
    # The deck carries the key itself; run_omip's parser default stays dyer1974.
    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    raw = read_yaml_with_includes(str(_REPO / "config/omip/omip_example.yaml"))
    assert raw["surface_stability_scheme"] == _BH91
