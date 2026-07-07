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

from legoesm.atmosphere.physics.convection.output import split_convective_rain


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
