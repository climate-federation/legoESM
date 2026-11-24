"""``--sw-skin-fraction`` (jerlov_2band) and ``ohlmann_2003`` are read only on the tripole lane with
jra55_do_tropical forcing; every other combination is refused before the model is built."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run.run_omip import check_shortwave_options


def _args(sw, skin, forcing="jra55_do_tropical"):
    return SimpleNamespace(sw_penetration=sw, sw_skin_fraction=skin, forcing_mode=forcing)


@pytest.mark.parametrize("sw,skin,grid,forcing,match", [
    ("auto", 0.0, "tripole", "jra55_do_tropical", "--sw-skin-fraction"),
    ("jerlov_2band", 0.0, "latlon", "jra55_do_tropical", "--sw-skin-fraction"),
    ("jerlov_2band", 0.0, "mpas", "jra55_do_tropical", "--sw-skin-fraction"),
    ("jerlov_2band", 0.0, "tripole", "restoring", "--sw-skin-fraction"),
    ("ohlmann_2003", None, "mpas", "jra55_do_tropical", "ohlmann_2003"),
    ("ohlmann_2003", None, "latlon", "jra55_do_tropical", "ohlmann_2003"),
    ("ohlmann_2003", None, "tripole", "restoring", "ohlmann_2003"),
])
def test_unread_shortwave_options_are_refused(sw, skin, grid, forcing, match):
    with pytest.raises(ValueError, match=match):
        check_shortwave_options(_args(sw, skin, forcing), grid)


@pytest.mark.parametrize("sw,skin", [("jerlov_2band", 0.0), ("ohlmann_2003", None), ("auto", None)])
def test_read_shortwave_options_pass_on_tripole_jra55(sw, skin):
    check_shortwave_options(_args(sw, skin), "tripole")
