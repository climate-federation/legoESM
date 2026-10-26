"""RULE 5 tests of the smoothed Bigg rain-freezing -4 C cut (T < 269.15 K in
CAM6 MG2 and in SAM M2005): C2 step, 0 at 269.15 K and full 2 K below in
production (``RAMP_WIDTH_PER_SHARPNESS / melt_sharpness``), user decisions
2026-10-22 (15a CAM lane, 3a SAM lane).

CAM lane oracle: ``mg2_prognostic_nc_frz.npz``, MG2's own Bigg rates and
in-precip rain (qric, nric, lamr) tapped before its frozen-rain split and rain
limiter (tap_frz).  SAM lane: no SAM oracle fixture exists in the repository;
its hard limit is SAM's gate itself (module_mp_graupel.F90:3296).  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402

from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FX = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_frz.npz"
HARD = 1.0e9
PROD = _production_morrison().melt_sharpness
# measured 2026-10-22 at the production melt_sharpness 2 (2 K); re-measure on change
GAP_MG2 = np.array([2.212334589792078e-07, 4.576061279692622e-19, 5.639053734011146e-06,
                    3.270257259811646e-05, 1.7336744112709187e-06, 6.106610810517941e-06])
GAP_SAM = np.array([2.1442994313251643e-07, 0.0, 7.601625914325994e-06,
                    3.270257259811647e-05, 8.304480512665032e-07, 8.376760986906147e-06])


def _fx():
    return dict(np.load(FX))


def _mg2(fx, ms):
    rho, T = jnp.asarray(fx["col_rho"]), jnp.asarray(fx["col_T"])
    return W.rain_freezing_bigg_mg2(
        jnp.asarray(fx["ora_frz_qric"]), jnp.asarray(fx["ora_frz_nric"]) * rho,
        jnp.asarray(fx["ora_frz_lamr"]), T, rho, jnp.ones_like(T),
        _cfg()._replace(melt_sharpness=ms))


def _sam(fx, ms):
    return W.rain_freezing_bigg(jnp.asarray(fx["col_q_r"]), jnp.asarray(fx["col_N_r"]),
                                jnp.asarray(fx["col_T"]), jnp.asarray(fx["col_rho"]),
                                _cfg()._replace(melt_sharpness=ms))


def _col_gap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max(axis=1) / np.maximum(np.abs(b).max(axis=1), 1e-300)


def test_fixture_covers_the_step():
    fx = _fx()
    T = fx["col_T"]
    assert (fx["ora_frz_mnuccr"] > 0).sum() >= 100
    assert ((T > 267.15) & (T < 269.15) & (fx["ora_frz_qric"] >= W.MG2_QSMALL)).sum() >= 10
    assert ((T >= 269.15) & (fx["ora_frz_qric"] >= W.MG2_QSMALL)).sum() >= 10


def test_cam_lane_hard_limit_matches_mg2():
    fx = _fx()
    m, n = _mg2(fx, HARD)
    assert _col_gap(m, fx["ora_frz_mnuccr"]).max() <= 1e-12
    assert _col_gap(np.asarray(n) / fx["col_rho"], fx["ora_frz_nnuccr"]).max() <= 1e-12


def test_cam_lane_default_gap_per_column():
    fx = _fx()
    gap = _col_gap(_mg2(fx, PROD)[0], fx["ora_frz_mnuccr"])
    print("Bigg MG2 gap per column", [float(x) for x in gap])
    np.testing.assert_allclose(gap, GAP_MG2, rtol=1e-6, atol=1e-12)


def test_sam_lane_hard_limit_is_sams_gate():
    T = jnp.asarray([268.0, 269.149, 269.15, 271.0])
    m, n = W.rain_freezing_bigg(jnp.full(4, 1e-4), jnp.full(4, 1e5), T, jnp.ones(4),
                                _cfg()._replace(melt_sharpness=HARD))
    assert np.all(np.asarray(m[:1]) > 0) and np.all(np.asarray(m[2:]) == 0) and np.all(np.asarray(n[2:]) == 0)


def test_infinite_sharpness_is_the_strict_cut():
    """melt_sharpness = inf is CAM's ``t < 269.15``: finite, 0 at 269.15 K."""
    T = jnp.asarray([268.0, 269.149, 269.15, 271.0])
    s = np.asarray(W.bigg_cold_step(T, _cfg()._replace(melt_sharpness=float("inf"))))
    assert np.array_equal(s, [1.0, 1.0, 0.0, 0.0])


def test_sam_lane_default_gap_per_column():
    fx = _fx()
    gap = _col_gap(_sam(fx, PROD)[0], _sam(fx, HARD)[0])
    print("Bigg SAM gap per column", [float(x) for x in gap])
    np.testing.assert_allclose(gap, GAP_SAM, rtol=1e-6, atol=1e-12)


def test_gradients_through_the_step():
    T = jnp.asarray([266.5, 267.6, 268.4, 269.0, 270.0])
    cfg = _cfg()._replace(melt_sharpness=PROD)
    check_grads(lambda t: W.bigg_cold_step(t, cfg), (T,), order=2, modes=["rev"],
                eps=1e-3, atol=1e-6, rtol=1e-4)
    one = jnp.ones_like(T)

    def mg2(t):
        return W.rain_freezing_bigg_mg2(1e-4 * one, 1e5 * one, 3000.0 * one, t, one, one, cfg)[0] * 1e6

    def sam(t):
        return W.rain_freezing_bigg(1e-4 * one, 1e5 * one, t, one, cfg)[0] * 1e6

    for f in (mg2, sam):
        check_grads(f, (T,), order=2, modes=["rev"], eps=1e-3, atol=1e-6, rtol=1e-3)
        g = np.asarray(jax.grad(lambda t: jnp.sum(f(t)))(T))
        assert np.all(np.isfinite(g)) and np.all(g[1:4] != 0.0) and g[4] == 0.0


def test_rain_at_qsmall_freezes_nothing_and_stays_finite():
    """lamr = 0 at q_ric == qsmall; the rate is 0, also above -4 C, and finite."""
    T = jnp.asarray([260.0, 275.0])
    one = jnp.ones_like(T)
    m, n = W.rain_freezing_bigg_mg2(W.MG2_QSMALL * one, 1e3 * one, 0.0 * one, T, one, one,
                                    _cfg()._replace(melt_sharpness=PROD))
    assert np.all(np.asarray(m) == 0.0) and np.all(np.asarray(n) == 0.0)
