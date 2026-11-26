"""run_coupled refuses radiation='ecrad_ifs' (vote R1a): the coupled loop is
not verified to carry ecRad's between-call state, so the scheme is neither
offered on the command line nor accepted from a deck default."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_refuses_ecrad_mod", REPO / "scripts" / "run" / "run_coupled.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_cli_does_not_offer_ecrad_ifs():
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--radiation", "ecrad_ifs"])


def test_deck_default_is_refused_at_config_build():
    p = mod.build_parser()
    p.set_defaults(radiation="ecrad_ifs")     # a deck row bypasses choices=
    with pytest.raises(SystemExit, match="ecrad_carry"):
        mod._coupled_atm_config(p.parse_args([]))
