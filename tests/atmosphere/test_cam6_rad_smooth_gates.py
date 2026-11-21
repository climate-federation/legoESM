"""RULE 5 smooth default of the CAM6 convective-water and snow-in-radiation
thresholds (CloudConfig.cam6_rad_gate_log_width).

Three checks per switch: at width 0 the port is the CAM6 oracle (the oracle
tests in test_cam6_conv_water_oracle.py / test_cam6_snow_rad.py, plus the
config plumbing below); at the default width the gap to the oracle is pinned
PER CASE (each oracle fixture column / point) in
oracle_fixtures/cam6_rad_smooth_gaps.npz, to 1e-6 relative with a 1e-12
absolute floor; and the gradient through every switch is finite, matches
finite differences and is non-zero inside the ramp band.

Regenerate the stored gaps after a change to the smooth form or the default
width:  JAX_ENABLE_X64=1 python -m tests.atmosphere.test_cam6_rad_smooth_gates
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    cam6_conv_water_4rad, cam6_snow_for_radiation, compute_cloud_properties)
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.core.smooth import cap_ln, floor_ln

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(__file__)
CW_FIX = os.path.join(_HERE, "oracle_fixtures", "cam6_conv_water.npz")
SN_FIX = os.path.join(_HERE, "oracle_fixtures", "cam6_snow_rad.npz")
GAPS = os.path.join(_HERE, "oracle_fixtures", "cam6_rad_smooth_gaps.npz")
W = CloudConfig().cam6_rad_gate_log_width
_CW_IN = ("ls_liq", "ls_ice", "ast", "dp_frac", "dp_icwmr")
_SN_IN = ("cld", "deepcu", "qliq", "qs", "ns", "T", "p", "pdel")


def _cw(fx, w):
    return cam6_conv_water_4rad(*(jnp.asarray(fx[k]) for k in _CW_IN),
                                float(fx["frac_limit"]), log_width=w)


def _sn(fx, w):
    return cam6_snow_for_radiation(*(jnp.asarray(fx[k]) for k in _SN_IN), log_width=w)


def _gaps():
    """Per-case |smooth - hard| at the default width: one entry per conv-water
    fixture column (relative to that column's largest hard value) and per snow
    fixture point (relative to the field's largest hard value over the
    fixture, since many points are exactly 0 at the hard limit)."""
    cw, sn = dict(np.load(CW_FIX)), dict(np.load(SN_FIX))
    out = {}
    for name, h, s in zip(("cw_liq", "cw_ice"), _cw(cw, 0.0), _cw(cw, W)):
        h, s = np.asarray(h), np.asarray(s)
        out[name] = (np.abs(s - h).max(axis=1)
                     / np.maximum(np.abs(h).max(axis=1), 1e-300))
    for name, h, s in zip(("sn_cldfsnow", "sn_icswp", "sn_des"), _sn(sn, 0.0), _sn(sn, W)):
        h, s = np.asarray(h), np.asarray(s)
        out[name] = np.abs(s - h) / max(float(np.abs(h).max()), 1e-300)
    return out


def test_default_width_is_the_approved_value():
    assert W == 0.01


def test_hard_limit_reaches_the_oracle_through_the_config():
    """width 0 in the CloudConfig reaches both ports (the oracle tests call the
    functions directly at their hard default)."""
    from tests.unit import _cam6_rad_columns as cc
    c = cc.columns(2)
    dp = c["p_half"][:, 1:] - c["p_half"][:, :-1]

    def props(**kw):
        return compute_cloud_properties(
            c["T"], c["p_full"], c["q_v"], dp, cc.cloud_config(**kw),
            q_cloud=c["q_c"], q_ice=c["q_i"], n_ice=c["n_i"],
            cloud_fraction_override=c["alst"], lat=c["lat"],
            conv_mass_flux_up=c["conv_mass_flux_up"], conv_icwmr=c["conv_icwmr"],
            p_half=c["p_half"], clubb_config=cc.CLUBB)

    dpg = dp / constants.g
    for w in (0.0, W):
        on = props(cam6_conv_water_in_rad=1, cam6_rad_gate_log_width=w)
        liq, _ = cam6_conv_water_4rad(c["q_c"], c["q_i"], on.ast, on.deepcu,
                                      c["conv_icwmr"], 0.01, log_width=w)
        cld = on.cloud_fraction
        want = cap_ln(liq / floor_ln(cld, 1e-4, w), 5e-3, w) * dpg * cld
        np.testing.assert_allclose(np.asarray(on.lwp), np.asarray(want),
                                   rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="cam6_rad_gate_log_width"):
        props(cam6_conv_water_in_rad=1, cam6_rad_gate_log_width=-1.0)


def test_smooth_default_gap_matches_the_recorded_gap_per_case():
    want = dict(np.load(GAPS))
    got = _gaps()
    assert sorted(want) == sorted(got)
    for k in got:
        np.testing.assert_allclose(got[k], want[k], rtol=1e-6, atol=1e-12, err_msg=k)
    # non-vacuous: the smooth default does differ from CAM somewhere
    assert max(float(np.max(v)) for v in got.values()) > 0.0


def _in_band_cw():
    """One column per conv-water switch, each input sitting inside its ramp."""
    fl, b = 0.01, np.exp(0.5 * W)
    ls_liq = jnp.array([2e-5, 2e-5, 2e-5])
    ls_ice = jnp.array([1e-5, 1e-5, 1e-5])
    ast = jnp.array([fl * b, 0.3, 0.3])
    # dp0 = dp_frac * ramp(dp_frac) must itself pass frac_limit for the
    # cumulus weight to move (CAM's two-stage test), so the upper band
    dp_frac = jnp.array([0.0, fl * np.exp(0.95 * W), 0.2])
    dp_icwmr = jnp.array([0.0, 1e-3, 1e-12 * b])
    return ls_liq, ls_ice, ast, dp_frac, dp_icwmr


def test_conv_water_gradient_through_every_switch():
    args = _in_band_cw()

    def f(*a):
        liq, ice = cam6_conv_water_4rad(*a, 0.01, log_width=W)
        return liq + ice
    # inputs span 1e-12..0.3: check in ln space so one finite-difference step
    # (eps in ln x, well inside the ramp width W) fits every input
    pos = tuple(jnp.where(a > 0, a, 1e-30) for a in args)

    def f_ln(*la):
        return f(*(jnp.exp(x) for x in la)) / 1e-5
    check_grads(f_ln, tuple(jnp.log(a) for a in pos), order=1, modes=("fwd", "rev"),
                eps=1e-6, atol=1e-6, rtol=1e-3)
    # non-zero derivative through the switched input of each column
    for col, arg in ((0, 2), (1, 3), (2, 4)):
        g = jax.grad(lambda x: f(*[x if i == arg else a for i, a in enumerate(args)])[col])(
            args[arg])
        assert np.isfinite(np.asarray(g)).all() and float(g[col]) != 0.0


def test_snow_gradient_through_every_switch():
    b = np.exp(0.5 * W)
    one = jnp.ones(4)
    cld = jnp.array([1e-4 * b, 0.5, 1e-5, 0.5])
    deepcu = jnp.array([0.0, 1e-4 * b, 0.0, 0.0])
    qliq = jnp.array([0.0, 0.0, 0.0, 1e-10 * b])
    qs = jnp.array([1e-6, 1e-6, 1e-6 * b, 1e-7 * b])
    args = (cld, deepcu, qliq, qs, 1e5 * one, 250.0 * one, 5e4 * one, 1e3 * one)

    def f(*a):
        cf, sp, de = cam6_snow_for_radiation(*a, log_width=W)
        return cf + sp + 1e-3 * de
    pos = tuple(jnp.where(a > 0, a, 1e-30) for a in args[:4])
    check_grads(lambda *la: f(*(jnp.exp(x) for x in la), *args[4:]),
                tuple(jnp.log(a) for a in pos), order=1, modes=("fwd", "rev"),
                eps=1e-6, atol=1e-6, rtol=1e-3)
    for col, arg in ((0, 0), (1, 1), (2, 3), (3, 2)):
        g = jax.grad(lambda x: f(*[x if i == arg else a for i, a in enumerate(args)])[col])(
            args[arg])
        assert np.isfinite(np.asarray(g)).all() and float(g[col]) != 0.0


def test_width_reaches_the_cloud_config_from_the_deck_key_and_the_cli():
    from legoesm.driver.model_driver import _standalone_cloud_config
    from scripts.run import run_amip
    from tests.unit.test_cam6_cloud_fraction import _exp_cfg
    cfg = _exp_cfg(cloud_cam6_conv_water_in_rad=1, cloud_cam6_rad_gate_log_width=0.0)
    assert _standalone_cloud_config(cfg, "cam6_clubb").cam6_rad_gate_log_width == 0.0
    assert _standalone_cloud_config(_exp_cfg(), "cam6_clubb").cam6_rad_gate_log_width == W
    a = run_amip.build_arg_parser().parse_args(["--cloud-cam6-rad-gate-log-width", "0"])
    assert run_amip.build_config_from_args(a).cloud_cam6_rad_gate_log_width == 0.0


def test_snow_and_specified_number_refused_off_the_hydrostatic_lane():
    from legoesm.atmosphere.physics.radiation.integration import (
        _refuse_hydrostatic_only_cloud_options as refuse)
    from tests.unit import _cam6_rad_columns as cc
    refuse(cc.radiation_config(cc.cloud_config()), "spectral")
    for kw in (dict(cam6_snow_in_rad=True), dict(cam6_rad_specified_nc=True)):
        with pytest.raises(ValueError, match="hydrostatic radiation lane only"):
            refuse(cc.radiation_config(cc.cloud_config(**kw)), "spectral")


if __name__ == "__main__":
    np.savez(GAPS, **_gaps())
    print({k: float(np.max(v)) for k, v in _gaps().items()})
