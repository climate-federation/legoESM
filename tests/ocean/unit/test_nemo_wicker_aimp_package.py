"""Literal controls for NEMO 5.0.2's unbranched RK3 ``ln_zad_Aimp`` package."""

import numpy as np
import jax.numpy as jnp

from legoesm.ocean.vertical import nemo_wicker_aimp_partition_transport
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    implicit_vertical_diffusion_nemo_momentum,
    implicit_vertical_diffusion_nemo_tracer_pair,
)


def _partition(w_value, lower_inflow):
    h = jnp.ones((1, 2, 2), dtype=jnp.float64)
    mfu = jnp.zeros((1, 3, 2), dtype=jnp.float64)
    mfu = mfu.at[0, 1, 1].set(lower_inflow)
    mfv = jnp.zeros((2, 2, 2), dtype=jnp.float64)
    w = jnp.zeros((1, 2, 3), dtype=jnp.float64)
    w = w.at[0, 0, 1].set(w_value)
    return nemo_wicker_aimp_partition_transport(
        mfu, mfv, w, h, jnp.ones_like(w), jnp.ones((1, 2)),
        jnp.ones((1, 3)), jnp.ones((2, 2)), 1.0,
    )


def test_partition_matches_literal_threshold_formula_and_upstream_cell():
    # Positive w selects the lower cell's Cu_h=0.55.  Thus Cu_min=.4,
    # Cu_max=.55, Cu_cut=.7 and Cu_v=.9 takes the high branch: 7/18.
    positive = _partition(0.9, 0.55)
    np.testing.assert_allclose(
        np.asarray(positive.fraction)[0, 0, 1], 7.0 / 18.0,
        rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        np.asarray(positive.w_explicit + positive.w_implicit),
        np.asarray(positive.w_explicit / (1.0 - positive.fraction)),
        rtol=0.0, atol=2.0e-15)

    # Synthetic wrong-time/side control: reversing w selects the upper cell,
    # whose Cu_h is zero.  Cu_v=.9 is then on the smooth branch and MUST not
    # reproduce the positive-w coefficient.
    negative = _partition(-0.9, 0.55)
    expected = 1.0 / (1.0 + 4.0 * 1.1 * 0.3 / 0.1**2)
    np.testing.assert_allclose(
        np.asarray(negative.fraction)[0, 0, 1], expected,
        rtol=0.0, atol=2.0e-15)
    assert positive.fraction[0, 0, 1] != negative.fraction[0, 0, 1]


def test_partition_has_no_spurious_bottom_up_coefficient_recurrence():
    base = _partition(0.9, 0.0)
    h = jnp.ones((1, 2, 3), dtype=jnp.float64)
    mfu = jnp.zeros((1, 3, 3), dtype=jnp.float64)
    mfv = jnp.zeros((2, 2, 3), dtype=jnp.float64)
    w = jnp.zeros((1, 2, 4), dtype=jnp.float64)
    w = w.at[0, 0, 1].set(50.0).at[0, 0, 2].set(0.9)
    got = nemo_wicker_aimp_partition_transport(
        mfu, mfv, w, h, jnp.ones_like(w), jnp.ones((1, 2)),
        jnp.ones((1, 3)), jnp.ones((2, 2)), 1.0)
    np.testing.assert_array_equal(
        np.asarray(got.fraction)[0, 0, 2],
        np.asarray(base.fraction)[0, 0, 1])


def test_literal_tracer_matrix_fuses_diffusion_and_implicit_transport():
    h = np.array([2.0, 3.0, 4.0])
    field_t = np.array([1.0, 4.0, 2.0])
    field_s = np.array([35.0, 34.0, 36.0])
    k = np.array([0.2, 0.3])
    e3w = np.array([2.5, 3.5])
    wi = np.array([0.0, -0.4, 0.7, 0.0])
    dt = 0.5
    diff = -dt * k / e3w
    lower = np.array([0.0, diff[0], diff[1]])
    upper = np.array([diff[0], diff[1], 0.0])
    diagonal = h - lower - upper
    lower += dt * np.minimum(wi[:-1], 0.0)
    upper -= dt * np.maximum(wi[1:], 0.0)
    diagonal += dt * (np.maximum(wi[:-1], 0.0) - np.minimum(wi[1:], 0.0))
    matrix = np.diag(diagonal) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    expected_t = np.linalg.solve(matrix, h * field_t)
    expected_s = np.linalg.solve(matrix, h * field_s)
    got_t, got_s = implicit_vertical_diffusion_nemo_tracer_pair(
        jnp.asarray(h * field_t), jnp.asarray(h * field_s), jnp.asarray(k),
        jnp.asarray(h), jnp.asarray(e3w), dt,
        jnp.ones(3, dtype=bool), implicit_w=jnp.asarray(wi))
    np.testing.assert_allclose(got_t, expected_t, rtol=0.0, atol=2.0e-15)
    # Commit 101d3383f deliberately keeps the tracer pair's two solves
    # separate.  Their independent roundoff is one binary64 ulp here.
    np.testing.assert_array_max_ulp(np.asarray(got_s), expected_s, maxulp=1)
    # Planted omission: removing wi must move the result.
    zero_t, _ = implicit_vertical_diffusion_nemo_tracer_pair(
        jnp.asarray(h * field_t), jnp.asarray(h * field_s), jnp.asarray(k),
        jnp.asarray(h), jnp.asarray(e3w), dt, jnp.ones(3, dtype=bool))
    assert np.max(np.abs(np.asarray(zero_t - got_t))) > 1.0e-3


def test_literal_momentum_matrix_fuses_adaptive_transport():
    field = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    h = jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float64)
    avm = jnp.asarray([0.0, 0.0], dtype=jnp.float64)
    wi = jnp.asarray([0.0, 0.6, -0.8, 0.0], dtype=jnp.float64)
    got = implicit_vertical_diffusion_nemo_momentum(
        field, avm, h, jnp.asarray([2.5, 3.5]), 0.25,
        jnp.ones(3, dtype=bool), implicit_w=wi)
    top, bottom = np.asarray(wi[:-1]), np.asarray(wi[1:])
    lower = 0.25 * np.minimum(top, 0.0) / np.asarray(h)
    upper = -0.25 * np.maximum(bottom, 0.0) / np.asarray(h)
    diag = 1.0 + 0.25 * (
        np.maximum(top, 0.0) - np.minimum(bottom, 0.0)) / np.asarray(h)
    matrix = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    np.testing.assert_allclose(got, np.linalg.solve(matrix, np.asarray(field)),
                               rtol=0.0, atol=2.0e-15)
