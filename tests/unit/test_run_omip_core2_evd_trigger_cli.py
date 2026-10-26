"""``--convection-*`` trigger flags reach ``EnhancedDiffusionConfig``, are
refused under ``--convection none``, and the top-interface occupancy
diagnostic counts what it says it counts.

The EVD module carried NEMO's zdfevd trigger fields (``n2_mode``,
``n2_eos_form``, ``smooth_transition``, ``n2_threshold``,
``two_level_trigger``) for weeks while the driver exposed only K_conv/K_bg, so
every EVD arm ran the in-situ smooth default that never fires in a stably
stratified column.  These tests hold the wiring down.
"""
from __future__ import annotations

import numpy as np
import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def _args(*extra):
    return _core2()._build_arg_parser().parse_args(
        ["--grid", "tripole", *extra])


def test_nemo_trigger_flags_reach_the_config():
    core2 = _core2()
    a = _args("--convection", "enhanced_diffusion",
              "--convection-K-conv", "100", "--convection-K-bg", "0",
              "--convection-n2-mode", "nemo_bn2",
              "--convection-n2-eos", "teos10",
              "--convection-trigger", "hard",
              "--convection-n2-threshold=-1e-12",
              "--convection-evd-composition", "nemo_replace")
    cfg = core2.build_enhanced_diffusion_config(a)
    assert cfg.K_conv == 100.0 and cfg.K_bg == 0.0
    assert cfg.n2_mode == "nemo_bn2" and cfg.n2_eos_form == "teos10"
    assert cfg.smooth_transition is False
    assert cfg.n2_threshold == -1e-12
    # decision 94: NEMO's zdfevd REPLACES avt, and selecting NEMO's trigger
    # from the command line must be able to say so.
    assert cfg.evd_composition == "nemo_replace"
    assert cfg.two_level_trigger is False
    # the two-level (before-state) arm cannot run on this driver's outer
    # integrators; it must refuse rather than silently fire now-only
    with pytest.raises(SystemExit, match="two-level"):
        core2.build_enhanced_diffusion_config(
            _args("--convection", "enhanced_diffusion",
                  "--convection-two-level"))


def test_bare_k_flags_keep_the_config_defaults():
    core2 = _core2()
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    cfg = core2.build_enhanced_diffusion_config(
        _args("--convection", "enhanced_diffusion", "--convection-K-conv", "7"))
    d = EnhancedDiffusionConfig()
    assert cfg.K_conv == 7.0
    assert (cfg.n2_mode, cfg.smooth_transition, cfg.n2_threshold,
            cfg.two_level_trigger) == (d.n2_mode, d.smooth_transition,
                                       d.n2_threshold, d.two_level_trigger)


def test_convection_none_returns_none_and_refuses_trigger_flags():
    core2 = _core2()
    assert core2.build_enhanced_diffusion_config(_args()) is None
    with pytest.raises(SystemExit, match="convection-n2-mode"):
        core2.build_enhanced_diffusion_config(
            _args("--convection-n2-mode", "nemo_bn2"))
    # the occupancy diagnostic is allowed on a control (no convection): it is
    # the measurement of whether the closure alone reaches the EVD range
    assert core2.build_enhanced_diffusion_config(
        _args("--evd-occupancy-every-hours", "1")) is None
    # a zero threshold is a real value, not "unset" (0.0 == False trap)
    with pytest.raises(SystemExit, match="convection-n2-threshold"):
        core2.build_enhanced_diffusion_config(
            _args("--convection-n2-threshold", "0.0"))


def test_occupancy_counts_the_first_interface_in_the_box_and_globally():
    core2 = _core2()
    lat = np.array([[0.0, 0.0, 10.0], [1.0, 1.0, 40.0]])
    lon = np.array([[230.0, 250.0, 230.0], [235.0, 100.0, 230.0]])
    land = np.ones_like(lat)
    land[1, 2] = 0.0                       # dry column: never counted
    K_H = np.zeros(lat.shape + (4,))
    K_H[0, 0, 0] = 100.0                   # box column, first interface fires
    K_H[1, 0, 1] = 100.0                   # box column, only a deeper interface
    K_H[0, 2, 0] = 100.0                   # outside the box (10N), fires
    K_H[1, 2, 0] = 100.0                   # dry, fires -> ignored
    f_box, f_glob, f_box3 = core2.evd_top_interface_occupancy(
        K_H, land, lat, lon, 100.0)
    # box = (0,0) and (1,0): one of two fires at the top interface
    assert f_box == pytest.approx(0.5)
    # wet = 5 columns; fired at the top: (0,0) and (0,2)
    assert f_glob == pytest.approx(2.0 / 5.0)
    # any-of-top-3: (1,0) fires at interface 1 -> both box columns count
    assert f_box3 == pytest.approx(1.0)
    # a half-strength K does not count as the convective branch
    K_H[0, 0, 0] = 40.0
    assert core2.evd_top_interface_occupancy(K_H, land, lat, lon, 100.0)[0] == 0.0


def test_trigger_occupancy_counts_unstable_top_interfaces():
    """The occupancy instrument measures NEMO's own trigger: a column with an
    inversion in the top cell fires at interface 1, a stratified one never
    does, and a dry column is never counted."""
    core2 = _core2()
    gdept = np.array([0.51, 1.56, 2.68, 3.87])
    gdepw = np.array([1.08, 2.10, 3.22])
    #            box column, inverted   box column, stable   outside the box
    T = np.array([[[22.0, 25.9, 25.6, 25.3],
                   [25.9, 25.7, 25.5, 25.2],
                   [22.0, 25.9, 25.6, 25.3]]])
    S = np.full_like(T, 35.0)
    lat = np.array([[0.0, 1.0, 20.0]])
    lon = np.array([[230.0, 235.0, 230.0]])
    land = np.ones_like(lat)
    f_box, f_glob, f_box3 = core2.evd_trigger_occupancy(
        T, S, gdept, gdepw, land, lat, lon)
    assert f_box == pytest.approx(0.5)          # one of the two box columns
    assert f_glob == pytest.approx(2.0 / 3.0)   # the off-box column fires too
    assert f_box3 == pytest.approx(0.5)         # nothing fires deeper
    land[0, 0] = 0.0                            # dry: the firing column leaves
    assert core2.evd_trigger_occupancy(
        T, S, gdept, gdepw, land, lat, lon)[0] == pytest.approx(0.0)


def test_trigger_ladders_come_from_the_shared_helper_on_a_partial_cell_coord():
    """The production tripole grid is a partial-cell coordinate, which carries
    no ``z_center_ref``; the ladders must come from the same helper the EVD
    trigger itself uses, or the sampler dies on its first call (it did)."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import nemo_bn2_depth_ladders
    from legoesm.ocean.vertical import (
        create_ocean_z_star, create_partial_cell_coordinate,
    )
    zs = create_ocean_z_star(n_levels=5, H_max=100.0)
    zc = create_partial_cell_coordinate(zs, jnp.full((2, 3), 80.0))
    assert not hasattr(zc, "z_center_ref")
    gdept, gdepw = (np.asarray(x) for x in nemo_bn2_depth_ladders(zc))
    assert gdept.shape == (5,) and gdepw.shape == (4,)
    assert np.all(np.diff(gdept) > 0) and np.all(gdept > 0)
    # and they are directly usable by the sampler
    T = np.tile(np.array([22.0, 25.9, 25.6, 25.3]), (2, 3, 1))
    S = np.full_like(T, 35.0)
    lat = np.zeros((2, 3)); lon = np.full((2, 3), 230.0)
    f_box, _, _ = _core2().evd_trigger_occupancy(
        T, S, gdept, gdepw, np.ones((2, 3)), lat, lon)
    assert f_box == pytest.approx(1.0)     # every column is inverted on top
