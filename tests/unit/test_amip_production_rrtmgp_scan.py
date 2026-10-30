"""The MPAS production deck scans the RRTMGP column recurrence; the lat-lon
and FV3-duo decks that include it keep the code default.

Scan was measured on the MPAS lane only (CPU, res6: step -4% at 1 node, -7%
at 8; round-off-only difference, dry mass unchanged).  The code default
(``None``: unrolled on CPU, scan on GPU) stays the record for the lanes it
was not measured on, so their decks name it explicitly.
"""
from __future__ import annotations

import pathlib

import pytest

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"


@pytest.mark.parametrize("deck, expected", [
    ("amip_production.yaml", True),
    ("amip_production_seaice_melt_albedo.yaml", True),
    ("amip_production_latlon48.yaml", None),
    ("amip_production_fv3duo_c24.yaml", None),
])
def test_deck_resolves_rrtmgp_use_scan(deck, expected):
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import build_arg_parser
    parser = build_arg_parser()
    keys = load_yaml_config(_AMIP / deck, parser)
    parser.set_defaults(**keys)  # same two-pass load as run_amip.main
    assert parser.parse_args([]).rrtmgp_use_scan is expected
