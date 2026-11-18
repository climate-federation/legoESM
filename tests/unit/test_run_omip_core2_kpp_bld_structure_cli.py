"""``--kpp-bld-structure`` on the tripole driver selects ``KPPConfig.bld_structure``."""
from __future__ import annotations

import pytest

from legoesm.ocean.physics.vertical_mixing.config import KPPConfig


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_unset_keeps_the_kppconfig_default():
    assert _core2().build_tripole_vmix_config("kpp").kpp == KPPConfig()


@pytest.mark.parametrize("mode", ["legacy", "cvmix"])
def test_flag_reaches_the_kpp_config(mode):
    vm = _core2().build_tripole_vmix_config("kpp", kpp_bld_structure=mode)
    assert vm.kpp.bld_structure == mode
    assert vm.kpp._replace(bld_structure=KPPConfig().bld_structure) == KPPConfig()


def test_parser_choices_and_default():
    p = _core2()._build_arg_parser()
    assert p.parse_args([]).kpp_bld_structure is None
    assert p.parse_args(["--kpp-bld-structure", "cvmix"]).kpp_bld_structure == "cvmix"
    with pytest.raises(SystemExit):
        p.parse_args(["--kpp-bld-structure", "mom6"])


@pytest.mark.parametrize("grid,vmix", [("tripole", "tke"), ("tripole", "none"),
                                       ("mpas", "none"), ("latlon_bathy", "none")])
def test_refused_where_it_would_be_discarded(grid, vmix):
    with pytest.raises(ValueError, match="--kpp-bld-structure"):
        _core2()._validate_kpp_bld_structure(grid, vmix, "cvmix")


def test_accepted_on_tripole_kpp_and_when_unset():
    v = _core2()._validate_kpp_bld_structure
    v("tripole", "kpp", "cvmix")
    v("mpas", "none", None)
