"""Behavioural tests for the consolidated stomatal-conductance kernels in the
single neutral ``legoesm.land.stomata`` module (explicit slope/intercept form).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land import stomata as sto
from legoesm.land.stomata import ball_berry_gs, medlyn_gs


def test_ball_berry_floors_at_intercept_when_no_assimilation():
    gs = ball_berry_gs(jnp.array(-5.0), jnp.array(0.8), jnp.array(400.0), 9.0, 0.01)
    assert float(gs) == pytest.approx(0.01)


def test_ball_berry_increases_with_A_and_RH():
    base = float(ball_berry_gs(jnp.array(10.0), jnp.array(0.5), jnp.array(400.0), 9.0, 0.01))
    hi_A = float(ball_berry_gs(jnp.array(20.0), jnp.array(0.5), jnp.array(400.0), 9.0, 0.01))
    hi_RH = float(ball_berry_gs(jnp.array(10.0), jnp.array(0.9), jnp.array(400.0), 9.0, 0.01))
    assert hi_A > base and hi_RH > base


def test_medlyn_decreases_with_VPD_and_floors_vpd():
    lo = float(medlyn_gs(jnp.array(10.0), jnp.array(0.5), jnp.array(400.0), 4.0, 0.01))
    hi = float(medlyn_gs(jnp.array(10.0), jnp.array(2.0), jnp.array(400.0), 4.0, 0.01))
    assert lo > hi  # higher VPD -> more closure
    at_floor = float(medlyn_gs(jnp.array(10.0), jnp.array(sto._VPD_FLOOR_KPA), jnp.array(400.0), 4.0, 0.01))
    below = float(medlyn_gs(jnp.array(10.0), jnp.array(1e-6), jnp.array(400.0), 4.0, 0.01))
    assert below == pytest.approx(at_floor)  # VPD floor guards 1/sqrt(VPD)


def test_medlyn_floors_at_g0_when_no_assimilation():
    gs = medlyn_gs(jnp.array(-3.0), jnp.array(1.0), jnp.array(400.0), 4.0, 0.01)
    assert float(gs) == pytest.approx(0.01)


def test_per_leaf_class_slope_intercept():
    # The explicit-arg form is what lets the two-leaf canopy pass distinct
    # C3 vs C4 (slope, intercept) that a single scalar config could not.
    c3 = float(ball_berry_gs(jnp.array(10.0), jnp.array(0.7), jnp.array(400.0), 9.0, 0.01))
    c4 = float(ball_berry_gs(jnp.array(10.0), jnp.array(0.7), jnp.array(400.0), 4.0, 0.04))
    assert c3 != c4


def test_differentiable():
    g = jax.grad(lambda A: ball_berry_gs(A, jnp.array(0.7), jnp.array(400.0), 9.0, 0.01))(10.0)
    assert np.isfinite(float(g)) and float(g) > 0.0
    gm = jax.grad(lambda A: medlyn_gs(A, jnp.array(1.0), jnp.array(400.0), 4.0, 0.01))(10.0)
    assert np.isfinite(float(gm)) and float(gm) > 0.0
