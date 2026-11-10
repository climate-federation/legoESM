"""``--tke-buoyancy-sink``: which discretisation of the TKE buoyancy term runs.

NEMO puts the whole ``-avt*rn2`` term on the right-hand side explicitly
(``zdftke.F90:417-420``); the card's inherited default charges the stable
part against the new energy on the diagonal (``implicit_linearized``). The
closure already carries both branches (``tke.py`` ``tke_buoyancy_sink``);
this module pins the flag that selects between them from the command line.
"""
from __future__ import annotations

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_card_default_is_unchanged_without_the_flag():
    assert _core2().orca1_zdftke_config().tke_buoyancy_sink == "implicit_linearized"


def test_flag_reaches_the_closure_config():
    cfg = _core2().orca1_zdftke_config(buoyancy_sink="nemo_explicit")
    assert cfg.tke_buoyancy_sink == "nemo_explicit"


def test_explicit_default_value_is_also_accepted():
    cfg = _core2().orca1_zdftke_config(buoyancy_sink="implicit_linearized")
    assert cfg.tke_buoyancy_sink == "implicit_linearized"


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="buoyancy_sink"):
        _core2().orca1_zdftke_config(buoyancy_sink="explicit")


def test_it_rides_the_tripole_vmix_builder():
    vm = _core2().build_tripole_vmix_config("tke", tke_buoyancy_sink="nemo_explicit")
    assert vm.tke.tke_buoyancy_sink == "nemo_explicit"


def test_builder_refuses_it_without_the_tke_closure():
    with pytest.raises(ValueError, match="--tke-buoyancy-sink"):
        _core2().build_tripole_vmix_config("kpp", tke_buoyancy_sink="nemo_explicit")


def test_parser_exposes_both_choices_and_defaults_to_none():
    p = _core2()._build_arg_parser()
    a = p.parse_args(["--tke-buoyancy-sink", "nemo_explicit"])
    assert a.tke_buoyancy_sink == "nemo_explicit"
    assert p.parse_args([]).tke_buoyancy_sink is None
    with pytest.raises(SystemExit):
        p.parse_args(["--tke-buoyancy-sink", "patankar"])


def test_card_grid_guard_refuses_it_off_the_tke_closure():
    with pytest.raises(SystemExit, match="--tke-buoyancy-sink"):
        _core2()._validate_tke_card_grid("tripole", "kpp", None, None, None, None,
                                         None, None, None,
                                         tke_buoyancy_sink="nemo_explicit")
