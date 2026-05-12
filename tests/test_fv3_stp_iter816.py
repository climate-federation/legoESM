"""FV3_3D iter 816: significant_tornado_parameter_fv3 (Thompson STP).

STP = (CAPE/1500) · (SRH_1km/150) · (BWD_6km/12) · LCL_term.

Tests
-----

1. ``test_stp_classic_tornadic``: textbook violent tornado → STP ≥ 3.
2. ``test_stp_high_lcl_zero``: LCL > 2000 m → STP = 0.
3. ``test_stp_low_lcl_saturates``: LCL < 1000 m → term clamped to 1.
4. ``test_stp_zero_cape``: CAPE=0 → STP=0.
5. ``test_stp_bwd_cap``: BWD=50 capped at 30.
6. ``test_stp_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import significant_tornado_parameter_fv3


def test_stp_classic_tornadic():
    """Classic May Plains tornadic: CAPE=3000, SRH=200, BWD=24, LCL=800
    → STP = 2 · 1.333 · 2 · 1 = 5.333."""
    cape = jnp.array([3000.0])
    srh = jnp.array([200.0])
    u_s = jnp.array([24.0])
    v_s = jnp.array([0.0])
    lcl = jnp.array([800.0])
    stp = significant_tornado_parameter_fv3(cape, srh, u_s, v_s, lcl)
    expected = (3000.0 / 1500.0) * (200.0 / 150.0) * (24.0 / 12.0) * 1.0
    np.testing.assert_allclose(np.asarray(stp), [expected], rtol=1e-12)
    assert float(stp[0]) >= 3.0  # significant-tornado threshold


def test_stp_high_lcl_zero():
    """LCL > 2000 m → STP = 0 (LCL too high)."""
    cape = jnp.array([3000.0])
    srh = jnp.array([200.0])
    u_s = jnp.array([24.0])
    v_s = jnp.array([0.0])
    lcl = jnp.array([2500.0])
    stp = significant_tornado_parameter_fv3(cape, srh, u_s, v_s, lcl)
    np.testing.assert_allclose(np.asarray(stp), [0.0], atol=1e-12)


def test_stp_low_lcl_saturates():
    """LCL < 1000 m → LCL_term = 1 (saturated).
    Same STP for LCL=500 and LCL=900 with all else equal."""
    cape = jnp.array([3000.0, 3000.0])
    srh = jnp.array([200.0, 200.0])
    u_s = jnp.array([24.0, 24.0])
    v_s = jnp.array([0.0, 0.0])
    lcl = jnp.array([500.0, 900.0])
    stp = significant_tornado_parameter_fv3(cape, srh, u_s, v_s, lcl)
    np.testing.assert_allclose(
        np.asarray(stp[0:1]), np.asarray(stp[1:2]), rtol=1e-12
    )


def test_stp_zero_cape():
    """CAPE=0 → STP=0."""
    cape = jnp.array([0.0])
    srh = jnp.array([200.0])
    u_s = jnp.array([24.0])
    v_s = jnp.array([0.0])
    lcl = jnp.array([800.0])
    stp = significant_tornado_parameter_fv3(cape, srh, u_s, v_s, lcl)
    np.testing.assert_allclose(np.asarray(stp), [0.0], atol=1e-15)


def test_stp_bwd_cap():
    """BWD=50 capped at 30 → BWD factor = 30/12 = 2.5.
    CAPE=1500, SRH=150, LCL=500, BWD=50 → STP = 1·1·2.5·1 = 2.5."""
    cape = jnp.array([1500.0])
    srh = jnp.array([150.0])
    u_s = jnp.array([50.0])
    v_s = jnp.array([0.0])
    lcl = jnp.array([500.0])
    stp = significant_tornado_parameter_fv3(cape, srh, u_s, v_s, lcl)
    np.testing.assert_allclose(np.asarray(stp), [2.5], rtol=1e-12)


def test_stp_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=816)
    n_x, n_y = 6, 8
    cape = jnp.asarray(rng.uniform(500.0, 4500.0, size=(n_x, n_y)))
    srh = jnp.asarray(rng.uniform(0.0, 400.0, size=(n_x, n_y)))
    u_s = jnp.asarray(rng.uniform(-40.0, 40.0, size=(n_x, n_y)))
    v_s = jnp.asarray(rng.uniform(-40.0, 40.0, size=(n_x, n_y)))
    lcl = jnp.asarray(rng.uniform(500.0, 2500.0, size=(n_x, n_y)))
    stp = significant_tornado_parameter_fv3(cape, srh, u_s, v_s, lcl)
    assert stp.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(stp))
    assert jnp.all(stp >= 0.0)
