"""run_coupled shares run_amip's parser (composable physics P5a).

Every atmosphere key an AMIP deck carries is a coupled key, so a coupled deck
can ``include:`` the AMIP production deck; the coupled CLI's own spellings and
its own defaults are kept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config
from scripts.run.run_amip import build_arg_parser as amip_parser
from scripts.run.run_coupled import _COUPLED_PARSER_DEFAULTS
from scripts.run.run_coupled import build_parser as coupled_parser

REPO = Path(__file__).resolve().parents[2]


def _actions(parser):
    return {a.dest: a for a in parser._actions if a.dest != "help"}


def test_every_amip_dest_is_a_coupled_dest():
    amip, coupled = _actions(amip_parser()), _actions(coupled_parser())
    missing = sorted(set(amip) - set(coupled))
    assert not missing, missing
    for dest, a in amip.items():
        if dest != "grid_type":   # the coupled lane couples four grids only
            assert coupled[dest].choices == a.choices, dest
        assert coupled[dest].type == a.type, dest
    assert sorted(coupled["grid_type"].choices) == [
        "cubed_sphere", "gaussian", "latlon", "voronoi"]


@pytest.mark.parametrize("deck", [
    "config/amip/amip_production.yaml",
    "config/amip/amip_production_latlon48.yaml",
])
def test_amip_production_decks_validate_against_the_coupled_parser(deck):
    keys = load_yaml_config(str(REPO / deck), coupled_parser())
    assert "convection" in keys and "grid_type" in keys


def test_coupled_defaults_are_pinned_and_differ_from_amip():
    """Non-vacuous: every pinned default must differ from run_amip's, else the
    pin is dead and the row belongs in a deck instead."""
    amip, coupled = _actions(amip_parser()), _actions(coupled_parser())
    args = coupled_parser().parse_args([])
    for dest, value in _COUPLED_PARSER_DEFAULTS.items():
        assert getattr(args, dest) == value, dest
        assert amip[dest].default != value, f"{dest}: pin equals run_amip's default"
    assert coupled["grid_type"].default == "cubed_sphere"


@pytest.mark.parametrize("argv, dest, want", [
    (["--grid", "latlon"], "grid_type", "latlon"),
    (["-n", "8"], "resolution", 8),
    (["-o", "out"], "output", "out"),
    (["--polar-filter"], "use_polar_filter", True),
    (["--no-polar-filter"], "use_polar_filter", False),
    (["--conv-cloud-max", "0.5"], "conv_cloud_max", 0.5),
    (["--cloud-conv-cloud-max", "0.5"], "conv_cloud_max", 0.5),
    (["--conv-cloud-condensate", "1e-4"], "conv_cloud_condensate", 1e-4),
    (["--rrtmgp-gpoint-checkpoint"], "rrtmgp_gpoint_checkpoint", True),
    (["--no-rrtmgp-gpoint-checkpoint"], "rrtmgp_gpoint_checkpoint", False),
    (["--unfused-radiation"], "unfused_radiation", True),
    (["--no-unfused-radiation"], "unfused_radiation", False),
])
def test_coupled_spellings_land_on_the_shared_dest(argv, dest, want):
    assert getattr(coupled_parser().parse_args(argv), dest) == want


def test_alias_refuses_an_existing_option_string():
    from scripts.run.run_coupled import _alias
    with pytest.raises(ValueError, match="already an option"):
        _alias(coupled_parser(), "nlev", "--resolution")


def test_aliases_do_not_change_the_default():
    args = coupled_parser().parse_args([])
    assert args.unfused_radiation is True
    assert args.rrtmgp_gpoint_checkpoint == _actions(amip_parser())["rrtmgp_gpoint_checkpoint"].default


def test_grid_alias_is_a_real_option_string():
    """``--grid`` would also reach grid_type as an argparse abbreviation of
    ``--grid-type``; the alias is what keeps it exact (and keeps ``-n``/``-o``)."""
    coupled = _actions(coupled_parser())
    assert "--grid" in coupled["grid_type"].option_strings
    assert "-n" in coupled["resolution"].option_strings
    assert "-o" in coupled["output"].option_strings


def test_old_coupled_yaml_keys_still_load(tmp_path):
    from scripts.run.run_coupled import _COUPLED_RENAMED_KEYS
    p = tmp_path / "old.yaml"
    p.write_text("grid: voronoi\npolar_filter: false\ncloud_conv_cloud_max: 0.4\n")
    out = load_yaml_config(str(p), coupled_parser(), renamed_keys=_COUPLED_RENAMED_KEYS)
    assert out == {"grid_type": "voronoi", "use_polar_filter": False,
                   "conv_cloud_max": 0.4}
    p.write_text("grid: voronoi\ngrid_type: latlon\n")
    with pytest.raises(SystemExit, match="old name"):
        load_yaml_config(str(p), coupled_parser(), renamed_keys=_COUPLED_RENAMED_KEYS)
    with pytest.raises(SystemExit, match="unknown key"):
        load_yaml_config(str(p), coupled_parser())


@pytest.mark.parametrize("argv", [["--truncation", "21"], ["--land-model", "slab"]])
def test_unresolved_amip_flags_are_refused(argv):
    from scripts.run import run_coupled
    with pytest.raises(SystemExit, match="post-parse"):
        run_coupled.main(argv, build_only=True)


def test_yaml_grid_type_key_and_grid_flag_agree():
    p = coupled_parser()
    deck = load_yaml_config(str(REPO / "config/cmip/cmip_ocean_slab.yaml"), p)
    assert deck["grid_type"] == "latlon"
    p.set_defaults(**deck)
    assert p.parse_args([]).grid_type == "latlon"
    assert p.parse_args(["--grid", "voronoi"]).grid_type == "voronoi"
