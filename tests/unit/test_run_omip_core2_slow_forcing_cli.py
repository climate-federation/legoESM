"""``--barotropic-slow-forcing-depth-evaluation`` reaches BarotropicConfig.

Decision 90 gave the field no default and the RK3 barotropic consumer raises
when it is unset; the OMIP driver had no way to state it, so every
``--momentum-rk3`` tripole card refused to start after the main merge.
"""
from __future__ import annotations

import pytest


def _parser():
    import scripts.run.run_omip_core2 as core2
    return core2._build_arg_parser()


def test_flag_has_no_default_and_parses_both_choices():
    p = _parser()
    assert p.parse_args(["--grid", "tripole"]).barotropic_slow_forcing_depth_evaluation is None
    for v in ("nemo_literal", "min_rule_live"):
        a = p.parse_args(["--grid", "tripole", "--barotropic-slow-forcing-depth-evaluation", v])
        assert a.barotropic_slow_forcing_depth_evaluation == v
    with pytest.raises(SystemExit):
        p.parse_args(["--grid", "tripole", "--barotropic-slow-forcing-depth-evaluation", "x"])


def test_replace_flat_routes_it_into_the_barotropic_subconfig():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    c = LatLonCGridOceanConfig().replace_flat(
        barotropic_slow_forcing_depth_evaluation="nemo_literal")
    assert c.barotropic.barotropic_slow_forcing_depth_evaluation == "nemo_literal"
    assert LatLonCGridOceanConfig().barotropic.barotropic_slow_forcing_depth_evaluation == ""


def test_both_tripole_builders_pass_it_through():
    import inspect
    import scripts.run.run_omip_core2 as core2
    src = inspect.getsource(core2)
    assert src.count('("barotropic_slow_forcing_depth_evaluation",\n') == 2
    assert src.count("barotropic_slow_forcing_depth_evaluation="
                     "args.barotropic_slow_forcing_depth_evaluation") == 2



def test_nemo_literal_without_carried_mode_is_refused():
    import inspect
    import scripts.run.run_omip_core2 as core2
    from legoesm.ocean.state import BarotropicConfig
    bt = BarotropicConfig()
    with pytest.raises(SystemExit, match="carried external mode"):
        core2.assert_slow_forcing_pair_resolved(
            bt._replace(barotropic_slow_forcing_depth_evaluation="nemo_literal"))
    core2.assert_slow_forcing_pair_resolved(
        bt._replace(barotropic_slow_forcing_depth_evaluation="nemo_literal",
                    nemo_prognostic_barotropic_state=True))
    core2.assert_slow_forcing_pair_resolved(
        bt._replace(barotropic_slow_forcing_depth_evaluation="min_rule_live"))
    core2.assert_slow_forcing_pair_resolved(None)
    # main checks the RESOLVED model config (after any --config YAML rebuild)
    src = inspect.getsource(core2.main)
    i_yaml = src.index("ocean override: {sorted(_ovr)}")
    i_chk = src.index("assert_slow_forcing_pair_resolved(")
    assert i_chk > i_yaml
