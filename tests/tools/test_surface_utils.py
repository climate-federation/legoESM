"""Unit tests for ``legoesm.forcing.surface_utils.snow_fraction`` (PR A #8).

The hard step ``where(T_low < T_freeze, 1, 0)`` was replaced by a smooth
Wigmosta 1994 / Dai 2008 ramp shared by the coupled and earth-system drivers.
These tests pin the ramp bounds, monotonicity, the exact transition edges, and
— the whole point of the change — a NON-ZERO gradient inside the mixed-phase
band so training/DA paths keep ``d(snow)/d(T_low)``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.forcing.surface_utils import snow_fraction

TF = float(constants.T_freeze)


def test_bounds_all_snow_below_all_rain_above():
    assert float(snow_fraction(jnp.array(TF - 10.0), TF)) == 1.0
    assert float(snow_fraction(jnp.array(TF + 10.0), TF)) == 0.0


def test_matches_clip_ramp_formula():
    T = jnp.linspace(TF - 5.0, TF + 5.0, 50)
    expected = jnp.clip((TF + 2.0 - T) / 4.0, 0.0, 1.0)
    np.testing.assert_array_equal(
        np.asarray(snow_fraction(T, TF)), np.asarray(expected)
    )


def test_monotone_non_increasing_in_T():
    T = jnp.linspace(TF - 5.0, TF + 5.0, 100)
    sf = np.asarray(snow_fraction(T, TF))
    assert np.all(np.diff(sf) <= 1e-12)


def test_transition_edges_and_midpoint():
    assert float(snow_fraction(jnp.array(TF + 2.0), TF)) == pytest.approx(0.0)
    assert float(snow_fraction(jnp.array(TF - 2.0), TF)) == pytest.approx(1.0)
    assert float(snow_fraction(jnp.array(TF), TF)) == pytest.approx(0.5)


def test_gradient_nonzero_in_band():
    """d(snow)/d(T_low) must be nonzero in the mixed-phase band — the hard step
    it replaced had zero gradient everywhere. Slope = -1/transition_width."""
    g = jax.grad(lambda t: snow_fraction(t, TF))(jnp.array(TF + 0.5))
    assert float(g) == pytest.approx(-0.25)


def test_gradient_zero_outside_band():
    g_cold = jax.grad(lambda t: snow_fraction(t, TF))(jnp.array(TF - 10.0))
    g_warm = jax.grad(lambda t: snow_fraction(t, TF))(jnp.array(TF + 10.0))
    assert float(g_cold) == 0.0
    assert float(g_warm) == 0.0
