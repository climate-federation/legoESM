"""Item 6: the FV3 C-D grid shallow-water mass-conservation fixer must keep the
fluid depth ``h >= 0`` (a bare additive ``h + correction`` over a thin layer can
drive ``h`` negative -> mass leaks through a "hole") WHILE conserving column
mass exactly.

These tests pin ``_apply_mass_conserving_floor`` directly:
  * no-flooring fast path is BIT-IDENTICAL to the legacy additive correction
    (so existing W2/W5 regression baselines are unchanged on the normal path);
  * thin-layer / negative-correction case stays >= 0 AND conserves mass;
  * the fix is differentiable (finite grad, no NaN).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    _apply_mass_conserving_floor,
    _H_FLOOR_M,
)


def _mass(h, area):
    return float(jnp.sum(jnp.asarray(h) * jnp.asarray(area)))


def test_no_flooring_is_bit_identical_to_additive():
    """When the additive correction never drives any cell below the floor, the
    helper returns EXACTLY ``h + correction`` (the legacy behaviour)."""
    rng = np.random.default_rng(0)
    h = jnp.asarray(rng.uniform(1000.0, 5000.0, size=(6, 8, 8)))
    area = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 8, 8)))
    total_area = jnp.sum(area)
    mass_new = jnp.sum(h * area)
    # Small positive target perturbation -> uniform positive correction, no
    # cell can go negative (all h are O(1000) m).
    mass_target = mass_new * 1.0001
    correction = (mass_target - mass_new) / total_area
    legacy = h + correction
    out = _apply_mass_conserving_floor(h, area, total_area, mass_target, mass_new)
    # Bit-identical on the no-flooring path.
    assert np.array_equal(np.asarray(out), np.asarray(legacy))


def test_thin_layer_stays_nonnegative_and_conserves_mass():
    """A column with a very thin layer and a NEGATIVE mass correction (mass must
    be REMOVED) would drive the thin cell negative under bare addition; the
    floor-then-renormalize keeps it >= 0 and conserves total mass exactly."""
    # One pathologically thin cell, rest moderate depth.
    h_np = np.full((1, 1, 6), 100.0)
    h_np[0, 0, 2] = 0.5  # thin layer
    h = jnp.asarray(h_np)
    area = jnp.ones((1, 1, 6))
    total_area = jnp.sum(area)
    mass_new = jnp.sum(h * area)
    # Remove enough mass that a uniform correction (-large) would push the thin
    # cell below zero: correction = (target - new)/total_area must exceed 0.5
    # in magnitude.  Remove 6 m of column-mean depth -> correction = -6 < -0.5.
    mass_target = mass_new - 6.0 * float(total_area)
    correction = (mass_target - mass_new) / total_area
    # Confirm the BARE additive correction would go negative (non-vacuous).
    assert float((h + correction)[0, 0, 2]) < 0.0

    out = _apply_mass_conserving_floor(h, area, total_area, mass_target, mass_new)
    # Positivity: every cell >= floor.
    assert bool(jnp.all(out >= _H_FLOOR_M - 1e-12)), np.asarray(out)
    # Mass conservation to accumulator precision.
    assert _mass(out, area) == pytest.approx(float(mass_target), rel=1e-10, abs=1e-6)


def test_floor_zero_with_positive_target_is_strictly_nonnegative():
    """With ``h_floor == 0`` and ``mass_target > 0`` the renorm scale lands in
    (0, 1], so the output is the additive field scaled down -> strictly >= 0
    even when MANY cells would have gone negative."""
    rng = np.random.default_rng(3)
    h = jnp.asarray(rng.uniform(0.1, 10.0, size=(6, 5, 5)))
    area = jnp.asarray(rng.uniform(0.5, 1.5, size=(6, 5, 5)))
    total_area = jnp.sum(area)
    mass_new = jnp.sum(h * area)
    # Aggressively remove ~80% of the mass: many thin cells would go negative.
    mass_target = mass_new * 0.2
    out = _apply_mass_conserving_floor(h, area, total_area, mass_target, mass_new)
    assert bool(jnp.all(out >= 0.0)), float(jnp.min(out))
    assert _mass(out, area) == pytest.approx(float(mass_target), rel=1e-10, abs=1e-6)


def test_fixer_is_differentiable_through_flooring_branch():
    """jax.grad of a scalar of the fixed depth w.r.t. the input depth is finite
    (no NaN/Inf) even when the flooring branch is active."""
    h0 = jnp.asarray(np.array([[[100.0, 0.5, 100.0, 100.0]]]))
    area = jnp.ones((1, 1, 4))
    total_area = jnp.sum(area)

    def scalar(h):
        mass_new = jnp.sum(h * area)
        mass_target = mass_new - 6.0 * total_area  # forces flooring
        out = _apply_mass_conserving_floor(h, area, total_area, mass_target, mass_new)
        return jnp.sum(out ** 2)

    g = jax.grad(scalar)(h0)
    assert bool(jnp.all(jnp.isfinite(g))), np.asarray(g)


def test_fixer_gradient_finite_at_zero_headroom_boundary():
    """Codex round-1 #1: the empty-headroom boundary (excess_mass == 0, the
    feasible dry column h_add==0, mass_target==0) must NOT seed a NaN adjoint.

    A bare ``deficit/excess_mass`` inside ``jnp.where`` still evaluates the
    divide on the dead branch and back-props NaN; the safe-denominator guard
    fixes it.  Differentiate at exactly the all-zero-target boundary and assert
    the gradient is finite.
    """
    h0 = jnp.asarray(np.array([[[1.0, 1.0, 1.0, 1.0]]]))
    area = jnp.ones((1, 1, 4))
    total_area = jnp.sum(area)

    def scalar(h):
        mass_new = jnp.sum(h * area)
        # Remove ALL mass -> h_add == 0 everywhere -> excess_mass == 0.
        mass_target = jnp.asarray(0.0)
        out = _apply_mass_conserving_floor(
            h, area, total_area, mass_target, mass_new
        )
        return jnp.sum(out ** 2)

    val = float(scalar(h0))
    g = jax.grad(scalar)(h0)
    assert np.isfinite(val), val
    assert bool(jnp.all(jnp.isfinite(g))), np.asarray(g)


def test_fixer_value_finite_when_target_exceeds_removable():
    """Zero-headroom value path: when mass_target == 0 the output is the floor
    (all zeros) and stays finite (no inf/nan from the guarded divide)."""
    h0 = jnp.asarray(np.array([[[2.0, 0.3, 2.0]]]))
    area = jnp.ones((1, 1, 3))
    total_area = jnp.sum(area)
    mass_new = jnp.sum(h0 * area)
    out = _apply_mass_conserving_floor(
        h0, area, total_area, jnp.asarray(0.0), mass_new
    )
    assert bool(jnp.all(jnp.isfinite(out))), np.asarray(out)
    assert bool(jnp.all(out >= 0.0)), np.asarray(out)
