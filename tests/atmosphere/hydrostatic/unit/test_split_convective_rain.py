"""Unit tests for the shared convective rain-split helper (output.py).

``split_convective_rain`` is the ONE definition of the in-updraft
precipitation split used by every mass-flux scheme that detrains to cloud
water (Tiedtke, Bechtold). Lock its mass-conservation contract + the
disabled-default (byte-identical legacy) path.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection.output import (
    convective_autoconversion_split,
    split_convective_rain,
)


def _dqc(ncol=3, nlev=8, amp=2e-6):
    # a mostly-positive condensate source with one negative cell to exercise
    # the non-negative clamp
    x = amp * jnp.linspace(-0.5, 1.0, ncol * nlev).reshape(ncol, nlev)
    return x


def test_disabled_default_is_legacy_noop():
    dqc = _dqc()
    dq_c, dq_r = split_convective_rain(dqc, 0.0)
    assert dq_r is None
    # cloud = max(dqc, 0) exactly (the pre-split behaviour)
    np.testing.assert_array_equal(np.asarray(dq_c),
                                  np.asarray(jnp.maximum(dqc, 0.0)))


@pytest.mark.parametrize("pe", [0.1, 0.5, 0.9, 1.0])
def test_mass_conserved_and_split_fraction(pe):
    dqc = _dqc()
    dq_c, dq_r = split_convective_rain(dqc, pe)
    pos = jnp.maximum(dqc, 0.0)
    # mass: cloud + rain == positive condensate, to machine precision
    np.testing.assert_allclose(np.asarray(dq_c + dq_r), np.asarray(pos),
                               rtol=0, atol=1e-20)
    # the rain fraction is exactly pe of the positive condensate
    np.testing.assert_allclose(np.asarray(dq_r), np.asarray(pos * pe),
                               rtol=1e-12, atol=1e-20)
    # both non-negative
    assert float(jnp.min(dq_c)) >= 0.0 and float(jnp.min(dq_r)) >= 0.0


def test_negative_source_clamped():
    dq_c, dq_r = split_convective_rain(-1.0 * jnp.ones((2, 3)), 0.5)
    assert float(jnp.max(dq_c)) == 0.0
    assert float(jnp.max(dq_r)) == 0.0


def test_efficiency_one_sends_all_to_rain():
    dqc = _dqc()
    dq_c, dq_r = split_convective_rain(dqc, 1.0)
    assert float(jnp.max(dq_c)) == 0.0
    np.testing.assert_allclose(np.asarray(dq_r),
                               np.asarray(jnp.maximum(dqc, 0.0)),
                               rtol=1e-12, atol=1e-20)


# --- physical (Sundqvist) autoconversion split -------------------------------
_QCRIT = 5.0e-4   # kg/kg
_PEMAX = 0.9


def test_autoconv_mass_conserved_and_nonneg():
    dqc = jnp.maximum(_dqc(), 0.0)
    q_c_u = jnp.linspace(0.0, 3e-3, dqc.size).reshape(dqc.shape)
    dq_c, dq_r = convective_autoconversion_split(dqc, q_c_u, _QCRIT, _PEMAX)
    # cloud + rain == positive condensate, to machine precision (no source/sink)
    np.testing.assert_allclose(np.asarray(dq_c + dq_r), np.asarray(dqc),
                               rtol=0, atol=1e-20)
    assert float(jnp.min(dq_c)) >= 0.0 and float(jnp.min(dq_r)) >= 0.0


def test_autoconv_zero_qcu_is_all_anvil():
    # q_c_u -> 0: pe -> 0, no convective rain (condensate detrains as anvil)
    dqc = jnp.maximum(_dqc(), 0.0)
    dq_c, dq_r = convective_autoconversion_split(
        dqc, jnp.zeros_like(dqc), _QCRIT, _PEMAX)
    np.testing.assert_allclose(np.asarray(dq_r), 0.0, atol=1e-30)
    np.testing.assert_allclose(np.asarray(dq_c), np.asarray(dqc), atol=1e-20)


def test_autoconv_saturates_at_pe_max():
    # q_c_u >> q_c_crit: rain fraction -> pe_max, and never exceeds it
    dqc = jnp.ones((2, 4)) * 1e-6
    big = jnp.ones_like(dqc) * (50.0 * _QCRIT)
    _dc, dq_r = convective_autoconversion_split(dqc, big, _QCRIT, _PEMAX)
    frac = np.asarray(dq_r) / np.asarray(dqc)
    np.testing.assert_allclose(frac, _PEMAX, rtol=1e-4)
    assert float(jnp.max(dq_r / dqc)) <= _PEMAX + 1e-6


def test_autoconv_monotonic_in_qcu():
    # more updraft cloud water -> larger precipitating fraction
    dqc = jnp.ones((1, 6)) * 1e-6
    q_c_u = jnp.linspace(0.0, 2e-3, 6).reshape(1, 6)
    _dc, dq_r = convective_autoconversion_split(dqc, q_c_u, _QCRIT, _PEMAX)
    r = np.asarray(dq_r).ravel()
    assert np.all(np.diff(r) >= -1e-20)


def test_autoconv_negative_source_clamped():
    dq_c, dq_r = convective_autoconversion_split(
        -jnp.ones((2, 3)), jnp.ones((2, 3)) * 1e-3, _QCRIT, _PEMAX)
    assert float(jnp.max(dq_c)) == 0.0 and float(jnp.max(dq_r)) == 0.0


def test_autoconv_qcrit_zero_is_finite():
    # divide-by-zero floor: q_c_crit == 0 must not produce NaN/Inf
    dqc = jnp.ones((2, 3)) * 1e-6
    dq_c, dq_r = convective_autoconversion_split(
        dqc, jnp.ones_like(dqc) * 1e-3, 0.0, _PEMAX)
    assert bool(jnp.all(jnp.isfinite(dq_c))) and bool(jnp.all(jnp.isfinite(dq_r)))


def test_autoconv_differentiable():
    import jax

    def loss(qcrit):
        dqc = jnp.ones((2, 3)) * 1e-6
        _dc, dq_r = convective_autoconversion_split(
            dqc, jnp.ones_like(dqc) * 8e-4, qcrit, _PEMAX)
        return jnp.sum(dq_r)

    g = jax.grad(loss)(_QCRIT)
    assert bool(jnp.isfinite(g))
