"""FV3_3D iter 846: tcre_remaining_budget_fv3.

E_remaining = 1000·(ΔT_target − ΔT_current)/TCRE  [Gt-CO₂].

Tests
-----

1. ``test_ar6_15c_canonical``: target=1.5, current=1.1, TCRE=0.45
   → 889 Gt-CO₂.
2. ``test_ar6_2c_canonical``: target=2.0, current=1.1 → ~2000 Gt-CO₂.
3. ``test_overshoot_negative``: target<current → E<0.
4. ``test_at_target_zero``: target=current → E=0.
5. ``test_tcre_inverse_scaling``: 2× TCRE → ½ budget.
6. ``test_tcre_low_high_range``: AR6 0.27 vs 0.63 spreads budget 2.3×.
7. ``test_tcre_zero_floored``: TCRE=0 → finite (no NaN).
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import tcre_remaining_budget_fv3


def test_ar6_15c_canonical():
    """ΔT_target=1.5, ΔT_current=1.1, TCRE=0.45 → 889 Gt-CO₂."""
    e = tcre_remaining_budget_fv3(
        jnp.array([1.5]), jnp.array([1.1]),
    )
    expected = 1000.0 * (1.5 - 1.1) / 0.45
    np.testing.assert_allclose(np.asarray(e), [expected], rtol=1e-12)
    assert 880.0 < float(e[0]) < 895.0


def test_ar6_2c_canonical():
    """ΔT_target=2.0, ΔT_current=1.1 → 2000 Gt-CO₂."""
    e = tcre_remaining_budget_fv3(
        jnp.array([2.0]), jnp.array([1.1]),
    )
    np.testing.assert_allclose(np.asarray(e), [2000.0], rtol=1e-12)


def test_overshoot_negative():
    """target<current → E<0 (overshoot)."""
    e = tcre_remaining_budget_fv3(
        jnp.array([1.0]), jnp.array([1.1]),
    )
    assert float(e[0]) < 0.0


def test_at_target_zero():
    """target=current → E=0."""
    e = tcre_remaining_budget_fv3(
        jnp.array([1.5]), jnp.array([1.5]),
    )
    np.testing.assert_allclose(np.asarray(e), [0.0], atol=1e-12)


def test_tcre_inverse_scaling():
    """2× TCRE → ½ budget."""
    e1 = tcre_remaining_budget_fv3(
        jnp.array([1.5]), jnp.array([1.1]),
        tcre=jnp.array([0.45]),
    )
    e2 = tcre_remaining_budget_fv3(
        jnp.array([1.5]), jnp.array([1.1]),
        tcre=jnp.array([0.90]),
    )
    np.testing.assert_allclose(np.asarray(e2), 0.5 * np.asarray(e1),
                                rtol=1e-12)


def test_tcre_low_high_range():
    """AR6 likely range 0.27–0.63 spans budget by 2.3×."""
    e_low = tcre_remaining_budget_fv3(
        jnp.array([1.5]), jnp.array([1.1]),
        tcre=jnp.array([0.27]),
    )
    e_high = tcre_remaining_budget_fv3(
        jnp.array([1.5]), jnp.array([1.1]),
        tcre=jnp.array([0.63]),
    )
    ratio = float(e_low[0] / e_high[0])
    np.testing.assert_allclose(ratio, 0.63 / 0.27, rtol=1e-12)


def test_tcre_zero_floored():
    """TCRE=0 → finite (no NaN, no div-by-0)."""
    e = tcre_remaining_budget_fv3(
        jnp.array([1.5]), jnp.array([1.1]),
        tcre=jnp.array([0.0]),
    )
    assert jnp.all(jnp.isfinite(e))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=846)
    n_x, n_y = 6, 8
    dt_target = jnp.asarray(rng.uniform(1.5, 3.0, size=(n_x, n_y)))
    dt_current = jnp.asarray(rng.uniform(0.8, 1.4, size=(n_x, n_y)))
    tcre = jnp.asarray(rng.uniform(0.25, 0.65, size=(n_x, n_y)))
    e = tcre_remaining_budget_fv3(dt_target, dt_current, tcre)
    assert e.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(e))
