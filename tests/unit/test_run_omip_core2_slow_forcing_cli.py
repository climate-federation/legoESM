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
    assert bt.barotropic_solver == "explicit_substep"   # the refused case below IS the window
    with pytest.raises(SystemExit, match="carried external mode"):
        core2.assert_slow_forcing_pair_resolved(
            bt._replace(barotropic_slow_forcing_depth_evaluation="nemo_literal"))
    with pytest.raises(SystemExit, match="inert"):
        core2.assert_slow_forcing_pair_resolved(
            bt._replace(barotropic_slow_forcing_depth_evaluation="nemo_literal",
                        barotropic_solver="implicit_unsplit"))
    # implicit CN: no window to seed, so the reference mean alone is allowed
    core2.assert_slow_forcing_pair_resolved(
        bt._replace(barotropic_slow_forcing_depth_evaluation="nemo_literal",
                    barotropic_solver="implicit_cn"))
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


def _tiny(carried):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=3, H_max=300.0)
    return rest_state_latlon_cgrid_ocean(
        grid, z, T_water_init_C=10.0, T_deep=10.0, H_max=300.0,
        land_lat_threshold=70.0, nemo_prognostic_barotropic_velocity=carried)


def test_carried_flag_parses_and_routes_into_the_barotropic_subconfig():
    import inspect
    import scripts.run.run_omip_core2 as core2
    from legoesm.ocean.state import LatLonCGridOceanConfig
    p = _parser()
    assert p.parse_args(["--grid", "tripole"]).nemo_carried_external_mode is False
    assert p.parse_args(["--grid", "tripole", "--nemo-carried-external-mode"]).nemo_carried_external_mode
    c = LatLonCGridOceanConfig().replace_flat(nemo_prognostic_barotropic_state=True)
    assert c.barotropic.nemo_prognostic_barotropic_state is True
    src = inspect.getsource(core2)
    assert src.count('("nemo_prognostic_barotropic_state",\n                               nemo_carried_external_mode)') == 2
    assert src.count("nemo_carried_external_mode=(True if args.nemo_carried_external_mode else None)") == 2


def test_attach_allocates_the_rest_pair_only_when_the_config_carries_it():
    import numpy as np
    import scripts.run.run_omip_core2 as core2
    from legoesm.ocean.state import LatLonCGridOceanConfig
    plain, ref = _tiny(False), _tiny(True)
    off = LatLonCGridOceanConfig()
    on = off.replace_flat(nemo_prognostic_barotropic_state=True)
    assert core2.attach_nemo_carried_pair(plain, off).uu_b is None
    got = core2.attach_nemo_carried_pair(plain, on)
    for a, b in ((got.uu_b, ref.uu_b), (got.vv_b, ref.vv_b)):
        assert a.data.shape == b.data.shape and a.data.dtype == b.data.dtype
        assert a.dims == b.dims and a.staggering == b.staggering
        assert np.count_nonzero(np.asarray(a.data)) == 0
    with pytest.raises(SystemExit, match="half"):
        core2.attach_nemo_carried_pair(plain._replace(uu_b=ref.uu_b), on)
    assert core2.attach_nemo_carried_pair(ref, on) is ref
    for kw in ({"balanced_init": True}, {"n_gpus": 2}):
        with pytest.raises(SystemExit, match="refused"):
            core2.attach_nemo_carried_pair(plain, on, **kw)
    for solver in ("implicit_cn", "rigid_lid", "implicit_unsplit"):
        with pytest.raises(SystemExit, match="split-explicit"):
            core2.attach_nemo_carried_pair(
                plain, on.replace_flat(barotropic_solver=solver))
    moving = plain._replace(u=plain.u.replace(data=plain.u.data.at[2, 3, 0].set(0.1)))
    with pytest.raises(SystemExit, match="not at rest"):
        core2.attach_nemo_carried_pair(moving, on)
    # main attaches on the RESOLVED config, before the restart loader uses the template
    import inspect
    src = inspect.getsource(core2.main)
    assert src.index("state = attach_nemo_carried_pair(state,") < src.index("load_run_restart(")


@pytest.mark.parametrize("argv", [
    ["--grid", "mpas", "--nemo-carried-external-mode"],
    ["--grid", "tripole", "--nemo-carried-external-mode"],          # no --momentum-rk3
])
def test_carried_flag_refused_where_nothing_consumes_it(argv, monkeypatch):
    import sys
    import scripts.run.run_omip_core2 as core2
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", *argv])
    with pytest.raises(SystemExit, match="nemo-carried-external-mode is (wired|refused)"):
        core2.main()
