"""FV3_3D iter 766: vapor_pressure_from_q_fv3 + iter-763 refactor.

Identity from r = q/(1−q):

    e = p · q / (ε + q·(1−ε))   ⇔   e = p · r / (ε·1000 + r)·1000/1000

where ε = R_d/R_v = ``constants.epsilon``.

Replaces inline Bolton-paper literal 622 in iter-763
``lcl_temperature_fv3`` with ``constants.epsilon``.

Tests
-----

1. ``test_vapor_pressure_zero_q``: q=0 → e≈0.
2. ``test_vapor_pressure_monotonic_in_q``: ∂e/∂q > 0.
3. ``test_vapor_pressure_scales_with_p``: e ∝ p.
4. ``test_vapor_pressure_iter763_refactor_consistent``: LCL still
   physically plausible (close to old 622-literal output to <0.1 K).
5. ``test_vapor_pressure_shapes_3d``.
6. ``test_vapor_pressure_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    lcl_temperature_fv3,
    vapor_pressure_from_q_fv3,
)


def test_vapor_pressure_zero_q():
    """q→0 → e→0 (e ≈ p·q/ε for small q)."""
    p = jnp.full((3,), 1000.0)  # mb
    q = jnp.array([0.0, 1e-10, 1e-8])
    e = vapor_pressure_from_q_fv3(p, q)
    # Linear regime: e ≈ p·q/ε ≈ 1610·q  (for ε=0.622, p=1000 mb)
    eps = constants.epsilon
    expected_max_linear = 1000.0 * 1e-8 / eps * 1.01  # +1% slack
    assert jnp.all(e < expected_max_linear)
    assert e[0] < 1e-8  # q=0 (clamped at 1e-12 → e ≈ 1.6e-9)


def test_vapor_pressure_monotonic_in_q():
    """∂e/∂q > 0."""
    p = jnp.full((4,), 1000.0)
    q1 = jnp.array([0.001, 0.005, 0.010, 0.020])
    q2 = q1 + 0.001
    e1 = vapor_pressure_from_q_fv3(p, q1)
    e2 = vapor_pressure_from_q_fv3(p, q2)
    assert jnp.all(e2 > e1)


def test_vapor_pressure_scales_with_p():
    """e ∝ p at fixed q."""
    q = jnp.full((3,), 0.010)
    p_a = jnp.array([500.0, 1000.0, 1500.0])
    p_b = 2.0 * p_a
    e_a = vapor_pressure_from_q_fv3(p_a, q)
    e_b = vapor_pressure_from_q_fv3(p_b, q)
    np.testing.assert_allclose(np.asarray(e_b), 2.0 * np.asarray(e_a), rtol=1e-12)


def test_vapor_pressure_iter763_refactor_consistent():
    """iter-763 LCL output after refactor close to pre-refactor (<0.1 K).

    Pre-refactor used literal 622.0 (rounded ε·1000); post-refactor
    uses constants.epsilon=0.621980.  Relative diff in e is ~3e-5,
    so T_LCL drift << 0.1 K.
    """
    T = jnp.array([290.0, 295.0, 300.0])
    p_mb = jnp.array([1000.0, 850.0, 700.0])
    q = jnp.array([0.010, 0.012, 0.015])
    t_lcl = lcl_temperature_fv3(T, p_mb, q)
    # Recompute with pre-refactor formula (literal 622) for delta check
    r = q / (1.0 - q) * 1000.0
    e_old = p_mb * r / (622.0 + r)
    t_lcl_old = 2840.0 / (3.5 * jnp.log(T) - jnp.log(e_old) - 4.805) + 55.0
    assert jnp.all(jnp.abs(t_lcl - t_lcl_old) < 0.1)


def test_vapor_pressure_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=766)
    n_x, n_y, km = 4, 5, 20
    p = jnp.asarray(rng.uniform(100.0, 1000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(n_x, n_y, km)))
    e = vapor_pressure_from_q_fv3(p, q)
    assert e.shape == (n_x, n_y, km)


def test_vapor_pressure_finite():
    """No NaN/Inf for realistic atmospheric range."""
    rng = np.random.default_rng(seed=767)
    p = jnp.asarray(rng.uniform(50.0, 1050.0, size=(8, 30)))
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(8, 30)))
    e = vapor_pressure_from_q_fv3(p, q)
    assert jnp.all(jnp.isfinite(e))
    assert jnp.all(e >= 0.0)
