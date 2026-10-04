"""``--nemo-een-coriolis``: NEMO ORCA1 ln_dynvor_een on the tripole lane.

Off by default (production keeps the 4-point-averaged Matsuno split); on, the
three config fields NEMO's EEN needs move together, and a non-tripole grid is
refused instead of silently ignoring the flag.
"""
from __future__ import annotations

import sys

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_flag_defaults_off():
    args = _core2()._build_arg_parser().parse_args(["--grid", "tripole"])
    assert args.nemo_een_coriolis is False


def test_non_tripole_grid_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "mpas",
                                      "--nemo-een-coriolis"])
    with pytest.raises(SystemExit, match="tripole only"):
        _core2().main()


def test_the_three_fields_land_on_the_ocean_config():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    base = LatLonCGridOceanConfig()
    assert (base.vorticity_scheme, base.coriolis_scheme) == ("al81", "matsuno_split")
    cfg = base.replace_flat(vorticity_scheme="een_total",
                            coriolis_scheme="explicit_ab2",
                            een_metric_weighting="nemo")
    assert (cfg.vorticity_scheme, cfg.coriolis_scheme, cfg.een_metric_weighting) \
        == ("een_total", "explicit_ab2", "nemo")


def test_main_forwards_the_flag_to_build_tripole():
    import inspect
    assert "nemo_een_coriolis=args.nemo_een_coriolis" in inspect.getsource(_core2().main)
    assert "nemo_een_coriolis" in inspect.signature(_core2().build_tripole).parameters


def test_vertical_momentum_flag_defaults_off_and_is_forwarded():
    import inspect
    args = _core2()._build_arg_parser().parse_args(["--grid", "tripole"])
    assert args.vertical_momentum_scheme is None
    assert "vertical_momentum_scheme=args.vertical_momentum_scheme" in inspect.getsource(_core2().main)


def test_vertical_momentum_flag_refused_off_tripole(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "fesom",
                                      "--vertical-momentum-scheme", "nemo_advective"])
    with pytest.raises(SystemExit, match="tripole and mpas only"):
        _core2().main()


def test_een_refused_with_rk3(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "tripole",
                                      "--nemo-een-coriolis", "--momentum-rk3"])
    with pytest.raises(SystemExit, match="forward-Euler"):
        _core2().main()


def test_vertical_momentum_flag_reaches_the_mpas_builder():
    import inspect
    assert "vertical_momentum_scheme" in inspect.signature(_core2().build_mpas_ocean).parameters
    assert inspect.getsource(_core2().main).count(
        "vertical_momentum_scheme=args.vertical_momentum_scheme") == 2


def test_dm2dc_hold_needs_dm2dc_and_a_day_divisor(monkeypatch):
    for extra in (["--dm2dc-hold-s", "14400"],                      # no --dm2dc
                  ["--dm2dc", "--dm2dc-hold-s", "7000", "--dt", "100"],   # not a day divisor
                  ["--dm2dc", "--dm2dc-hold-s", "14400", "--dt", "7000"]):  # not a dt multiple
        monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", "--grid", "tripole", *extra])
        with pytest.raises(SystemExit, match="dm2dc-hold-s"):
            _core2().main()


def test_hold_window_mean_conserves_the_daily_shortwave():
    """Block means over 4 h must integrate to the same daily SW as the per-step
    windows (both normalise to a daily mean of 1)."""
    import numpy as np
    from legoesm.ocean.forcing.diurnal_cycle import diurnal_sw_factor
    lon, lat = np.array([230.0, 10.0]), np.array([0.0, 45.0])
    blocks = [diurnal_sw_factor(lon, lat, day_of_year=15, t_frac_lo=b / 6, t_frac_up=(b + 1) / 6)
              for b in range(6)]
    steps = [diurnal_sw_factor(lon, lat, day_of_year=15, t_frac_lo=k / 96, t_frac_up=(k + 1) / 96)
             for k in range(96)]
    np.testing.assert_allclose(np.mean(blocks, 0), np.mean(steps, 0), rtol=1e-10)
    assert float(np.max(blocks)) < float(np.max(steps))   # the noon peak is smoothed
