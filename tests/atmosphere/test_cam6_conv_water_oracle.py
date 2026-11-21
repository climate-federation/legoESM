"""CAM6 convective water in the radiation (cloud_fraction.cam6_conv_water_4rad).

Oracle: CAM ``conv_water.F90`` at cam_cesm2_1_rel_60, compiled unmodified in
fp64 with stand-in modules (scripts/validate/oracle/cam6_conv_water), run on
12 columns of the A2 C1 smoke dump (4 warm pool, 4 SE Pacific, 4 elsewhere)
and 8 synthetic columns that reach every branch.  Fixture:
tests/atmosphere/oracle_fixtures/cam6_conv_water.npz.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    cam6_conv_water_4rad, compute_cloud_properties)
from tests.unit import _cam6_rad_columns as cc

jax.config.update("jax_enable_x64", True)

FIX = os.path.join(os.path.dirname(__file__), "oracle_fixtures", "cam6_conv_water.npz")
RTOL = 1e-10


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIX))


def test_port_matches_the_fortran(fx):
    liq, ice = cam6_conv_water_4rad(
        *(jnp.asarray(fx[k]) for k in ("ls_liq", "ls_ice", "ast", "dp_frac", "dp_icwmr")),
        float(fx["frac_limit"]))
    for got, want in ((liq, fx["totg_liq"]), (ice, fx["totg_ice"])):
        got = np.asarray(got)
        scale = np.maximum(np.abs(want), 1e-300)
        rel = np.where(want != 0, np.abs(got - want) / scale, np.abs(got))
        assert rel.max() <= RTOL, rel.max()
        np.testing.assert_array_equal(got == 0.0, want == 0.0)


def test_fixture_is_not_vacuous(fx):
    n = int(fx["n_dumped"])
    assert n >= 10 and fx["ast"].shape[0] >= 18
    # convection adds water in the dumped columns, and the stratiform-only
    # branch drops stratus under frac_limit (synthetic column 1)
    cu = (fx["dp_frac"] > fx["frac_limit"]) & (fx["dp_icwmr"] > 1e-12)
    assert cu[:n].sum() > 20
    tot = fx["totg_liq"] + fx["totg_ice"]
    ls = fx["ls_liq"] + fx["ls_ice"]
    assert np.any(tot[:n][cu[:n]] > 1.5 * ls[:n][cu[:n]])
    small = (fx["ast"] < fx["frac_limit"]) & (ls > 0) & ~cu
    assert small.any() and np.all(tot[small] == 0.0)


def _props(cfg, c):
    return compute_cloud_properties(
        c["T"], c["p_full"], c["q_v"], c["p_half"][:, 1:] - c["p_half"][:, :-1],
        cfg, q_cloud=c["q_c"], q_ice=c["q_i"], n_ice=c["n_i"],
        cloud_fraction_override=c["alst"], lat=c["lat"],
        conv_mass_flux_up=c["conv_mass_flux_up"], conv_icwmr=c["conv_icwmr"],
        p_half=c["p_half"], clubb_config=cc.CLUBB)


def test_cloud_call_uses_cam6_in_cloud_water_over_cld():
    """cloud_diagnostics.F90:391-404: in-cloud water = min(totg/max(1e-4, CLD),
    5 g/kg), grid path = in-cloud path x CLD; the radii stay stratiform."""
    c = cc.columns(2)
    off = _props(cc.cloud_config(), c)
    on = _props(cc.cloud_config(cam6_conv_water_in_rad=1), c)
    liq, ice = cam6_conv_water_4rad(c["q_c"], c["q_i"], off.ast, off.deepcu,
                                   c["conv_icwmr"], 0.01,
                                   log_width=cc.cloud_config().cam6_rad_gate_log_width)
    dpg = (c["p_half"][:, 1:] - c["p_half"][:, :-1]) / constants.g
    cld = on.cloud_fraction
    want_l = jnp.minimum(liq / jnp.maximum(cld, 1e-4), 5e-3) * dpg * cld
    want_i = jnp.minimum(ice / jnp.maximum(cld, 1e-4), 5e-3) * dpg * cld
    np.testing.assert_allclose(np.asarray(on.lwp), np.asarray(want_l), rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.asarray(on.iwp), np.asarray(want_i), rtol=1e-12, atol=0)
    np.testing.assert_array_equal(np.asarray(on.cloud_fraction), np.asarray(off.cloud_fraction))
    np.testing.assert_array_equal(np.asarray(on.r_eff_liq), np.asarray(off.r_eff_liq))
    np.testing.assert_array_equal(np.asarray(on.r_eff_ice), np.asarray(off.r_eff_ice))
    # convection adds water: more total path where deepcu is on
    deep = np.asarray(off.deepcu) > 0.05
    assert deep.any()
    assert (np.asarray(on.lwp + on.iwp)[deep].sum()
            > 1.2 * np.asarray(off.lwp + off.iwp)[deep].sum())


def test_default_is_unchanged_and_bad_settings_refused():
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    assert CloudConfig().cam6_conv_water_in_rad == 0
    c = cc.columns(3)
    with pytest.raises(ValueError, match="mode 1"):
        _props(cc.cloud_config(cam6_conv_water_in_rad=2), c)
    with pytest.raises(ValueError, match="cam6_clubb"):
        compute_cloud_properties(
            c["T"], c["p_full"], c["q_v"], c["p_half"][:, 1:] - c["p_half"][:, :-1],
            CloudConfig(scheme="sundqvist", cam6_conv_water_in_rad=1),
            q_cloud=c["q_c"], q_ice=c["q_i"])
    with pytest.raises(ValueError, match="frac_limit"):
        cam6_conv_water_4rad(c["q_c"], c["q_i"], c["alst"], c["alst"], c["conv_icwmr"], 0.0)


def test_deck_keys_reach_the_radiation_cloud_config():
    from legoesm.driver.model_driver import _standalone_cloud_config
    from tests.unit.test_cam6_cloud_fraction import _exp_cfg
    cfg = _exp_cfg(cloud_cam6_conv_water_in_rad=1, cloud_cam6_conv_water_frac_limit=0.02)
    cfg.validate_strict()
    cc_ = _standalone_cloud_config(cfg, "cam6_clubb")
    assert cc_.cam6_conv_water_in_rad == 1 and cc_.cam6_conv_water_frac_limit == 0.02
    assert _standalone_cloud_config(_exp_cfg(), "cam6_clubb").cam6_conv_water_in_rad == 0
    with pytest.raises(ValueError, match="cloud_scheme='cam6_clubb'"):
        _exp_cfg(cloud_scheme="sundqvist", use_clubb_cloud_fraction=False,
                 cloud_cam6_conv_water_in_rad=1).validate_strict()
    with pytest.raises(ValueError, match="mode 2 is not ported"):
        _exp_cfg(cloud_cam6_conv_water_in_rad=2).validate_strict()


def test_cli_parses_the_keys():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("_run_amip_cw", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod.build_arg_parser().parse_args(
        ["--cloud-cam6-conv-water-in-rad", "1", "--cloud-cam6-conv-water-frac-limit", "0.01"])
    assert a.cloud_cam6_conv_water_in_rad == 1 and a.cloud_cam6_conv_water_frac_limit == 0.01
