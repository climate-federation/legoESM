"""Unit tests for maximum-random cloud overlap (issue #689).

The retired ``clt`` diagnostic used a near-binary condensate mask + pure random
overlap, saturating total cloud cover to ~100% wherever a column held any trace
of condensate.  These tests pin the replacement ``maximum_random_overlap``
reduction: a single cloudy layer maps to *its own* fraction (no saturation),
vertically-adjacent layers overlap maximally, separated groups overlap
randomly, and the result is always in [0, 1].
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.diagnostics.cloud_overlap import maximum_random_overlap


def test_clear_column_is_zero():
    cf = jnp.zeros((8,))
    assert float(maximum_random_overlap(cf)) == pytest.approx(0.0)


def test_single_fully_cloudy_layer_is_one():
    # One saturated layer anywhere in the column -> clt = 1 (that layer).
    for k in (0, 3, 7):
        cf = np.zeros(8)
        cf[k] = 1.0
        assert float(maximum_random_overlap(jnp.asarray(cf))) == pytest.approx(1.0)


def test_single_partial_layer_equals_its_fraction():
    # The anti-saturation property: one partially-cloudy layer (rest clear)
    # must report exactly its own fraction, NOT ~1 (the old binary-mask bug).
    for k in (0, 4, 7):
        cf = np.zeros(8)
        cf[k] = 0.3
        assert float(maximum_random_overlap(jnp.asarray(cf))) == pytest.approx(0.3)


def test_adjacent_layers_maximum_overlap():
    # Two vertically-adjacent identical layers overlap MAXIMALLY:
    # clt = 0.5, not the 1 - (1-0.5)(1-0.5) = 0.75 of pure random overlap.
    cf = np.zeros(8)
    cf[3] = 0.5
    cf[4] = 0.5
    assert float(maximum_random_overlap(jnp.asarray(cf))) == pytest.approx(0.5)


def test_separated_groups_random_overlap():
    # Two cloud groups separated by a clear layer overlap RANDOMLY:
    # clt = 1 - (1-0.5)(1-0.5) = 0.75.
    cf = np.zeros(8)
    cf[2] = 0.5
    # layer 3 clear
    cf[4] = 0.5
    assert float(maximum_random_overlap(jnp.asarray(cf))) == pytest.approx(0.75)


def test_result_bounded_and_no_nan_with_saturated_layers():
    # Full column of saturated + partial layers must stay in [0, 1] and be
    # NaN-free (the (1 - cf) denominator guard).
    rng = np.random.default_rng(0)
    cf = rng.uniform(0.0, 1.0, size=(5, 7, 12))
    cf[..., 3] = 1.0  # force a fully-cloudy layer (denominator -> 0)
    out = np.asarray(maximum_random_overlap(jnp.asarray(cf)))
    assert out.shape == (5, 7)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert np.all(out == pytest.approx(1.0))  # any saturated layer -> overcast


def test_values_clipped_to_unit_interval():
    # Out-of-range inputs are clipped before the recursion.
    cf = jnp.asarray([[-0.2, 1.4, 0.0]])
    out = float(maximum_random_overlap(cf)[0])
    assert out == pytest.approx(1.0)  # the 1.4 -> 1.0 layer makes it overcast


def test_monotonic_in_single_layer_fraction():
    # clt is monotone non-decreasing in a lone cloudy layer's fraction.
    vals = [float(maximum_random_overlap(jnp.asarray(np.array([0.0, f, 0.0]))))
            for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals)
    assert vals[0] == pytest.approx(0.0) and vals[-1] == pytest.approx(1.0)
