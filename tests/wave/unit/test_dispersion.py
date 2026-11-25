"""Dispersion solve, group velocity and depth-refraction derivative."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.wave.constants import GRAVITY
from legoesm.wave.physics.dispersion import (
    calculate_depth_dispersion_derivative,
    compute_group_velocity,
    solve_wavenumber,
)


def test_deep_water_limit():
    """kd >> 1: k -> omega^2/g and Cg -> omega/(2k)."""
    freq = jnp.asarray([0.1])
    depth = jnp.full((1, 2, 2, 1, 1), 4000.0)
    k = solve_wavenumber(freq, depth)
    omega = 2.0 * jnp.pi * freq[0]
    expected = omega**2 / GRAVITY
    assert np.allclose(np.asarray(k), float(expected), rtol=1e-3)

    cg = compute_group_velocity(freq, k, depth)
    assert np.allclose(np.asarray(cg), float(omega / (2.0 * expected)), rtol=2e-3)


def test_lands_do_not_leak_nan_wavenumbers():
    """NaN depth fails the wet test and returns exactly zero, not NaN.

    The port substitutes finite depth/frequency before the bisection and
    then masks the result with ``where(wet, k, 0)``; no NaN can reach the
    physics.
    """
    freq = jnp.asarray([0.1])
    depth = jnp.full((1, 2, 2, 1, 1), jnp.nan)
    k = solve_wavenumber(freq, depth)
    assert np.all(np.asarray(k) == 0.0)


def test_group_velocity_zero_on_land():
    freq = jnp.asarray([0.1])
    depth = jnp.zeros((1, 2, 2, 1, 1))
    k = jnp.zeros((1, 2, 2, 1, 1))
    cg = compute_group_velocity(freq, k, depth)
    assert np.all(np.asarray(cg) == 0.0)


def test_depth_derivative_zero_in_deep_water():
    freq = jnp.asarray([0.1])
    depth = jnp.full((1, 2, 2, 1, 1), 4000.0)
    k = solve_wavenumber(freq, depth)
    cg = compute_group_velocity(freq, k, depth)
    dsdd = calculate_depth_dispersion_derivative(cg, k, freq, depth)
    assert np.all(np.asarray(dsdd) == 0.0)


def test_depth_derivative_matches_the_analytic_shallow_water_value():
    """5 m / 0.1 Hz: check DsDd against n analytic in the solved k.

    A sign-flipped finite-depth term in Cg is clipped to zero by the
    derivative's non-negative floor, so a ``>= 0`` assertion cannot catch
    it.  This test pins the value instead.
    """
    freq = jnp.asarray([0.1])
    depth = jnp.full((1, 2, 2, 1, 1), 5.0)
    k = solve_wavenumber(freq, depth)
    cg = compute_group_velocity(freq, k, depth)
    dsdd = calculate_depth_dispersion_derivative(cg, k, freq, depth)

    kd = float(np.asarray(k).ravel()[0]) * 5.0
    omega = 2.0 * np.pi * 0.1
    n = 0.5 + kd / np.sinh(2.0 * kd)
    cg_expected = n * omega / float(np.asarray(k).ravel()[0])
    dsdd_expected = (cg_expected * float(np.asarray(k).ravel()[0]) - 0.5 * omega) / 5.0
    assert np.isclose(float(np.asarray(cg).ravel()[0]), cg_expected, rtol=1e-5)
    assert np.isclose(float(np.asarray(dsdd).ravel()[0]), dsdd_expected, rtol=1e-5)
