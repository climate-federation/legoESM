"""Unit tests for ``legoesm.atmosphere.physics.convection._triggers``.

The eight smooth-trigger primitives (``smooth_step``, ``smooth_max``,
``smooth_min``, ``smooth_positive_part``, ``smooth_level_indicator``,
``smooth_lowest_crossing_index``, ``cape_trigger``) are the foundation
that every new convection scheme uses to replace ``if`` / ``where``
discontinuities with differentiable approximations.  These tests pin
their mathematical contract:

* monotonicity / boundedness of step;
* upper-bound property and sharpness limit of ``smooth_max``;
* exact softplus identity for ``smooth_positive_part``;
* per-level indicator level count;
* fractional crossing index against analytical reference values
  (single, multiple, none, exact-on-level);
* finite, non-zero gradients at the threshold (the AD-safety
  property that motivates these helpers).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection import _triggers as T


# ---------------------------------------------------------------------------
# smooth_step / smooth_heaviside
# ---------------------------------------------------------------------------

def test_smooth_step_monotonicity_and_bounds():
    xs = jnp.linspace(-5.0, 5.0, 41)
    out = T.smooth_step(xs, sharpness=2.0)
    # Strictly increasing.
    assert jnp.all(jnp.diff(out) > 0)
    # Bounded in (0, 1).
    assert jnp.all(out > 0.0)
    assert jnp.all(out < 1.0)


def test_smooth_step_zero_gives_one_half():
    """sigmoid(s * 0) = 1/2 for any sharpness."""
    for s in (0.5, 1.0, 5.0, 100.0):
        assert pytest.approx(0.5, abs=1e-12) == float(
            T.smooth_step(jnp.asarray(0.0), s)
        )


def test_smooth_step_grad_at_zero_is_sharpness_over_four():
    """d sigmoid(sx)/dx at x=0 equals s/4."""
    for s in (1.0, 5.0, 10.0):
        g = float(jax.grad(lambda x: T.smooth_step(x, s))(jnp.asarray(0.0)))
        assert pytest.approx(s / 4.0, rel=1e-10) == g


def test_smooth_heaviside_is_alias_for_smooth_step():
    xs = jnp.linspace(-2.0, 2.0, 11)
    assert jnp.allclose(
        T.smooth_heaviside(xs, sharpness=3.5),
        T.smooth_step(xs, sharpness=3.5),
    )


# ---------------------------------------------------------------------------
# smooth_max / smooth_min
# ---------------------------------------------------------------------------

def test_smooth_max_upper_bound():
    """smooth_max(a, b, s) >= max(a, b) for all finite s > 0."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = float(rng.uniform(-10, 10))
        b = float(rng.uniform(-10, 10))
        for s in (0.5, 1.0, 5.0):
            sm = float(T.smooth_max(jnp.asarray(a), jnp.asarray(b), s))
            assert sm >= max(a, b) - 1e-12


def test_smooth_max_sharpness_limit():
    """As sharpness → ∞, smooth_max → max."""
    a, b = 1.0, 2.0
    sm = float(T.smooth_max(jnp.asarray(a), jnp.asarray(b), 100.0))
    assert pytest.approx(2.0, abs=1e-6) == sm


def test_smooth_max_symmetric_in_arguments():
    a, b = 1.7, -0.3
    s = 2.0
    sm_ab = float(T.smooth_max(jnp.asarray(a), jnp.asarray(b), s))
    sm_ba = float(T.smooth_max(jnp.asarray(b), jnp.asarray(a), s))
    assert pytest.approx(sm_ab, rel=1e-12) == sm_ba


def test_smooth_min_dual_to_smooth_max():
    """smooth_min(a, b, s) = -smooth_max(-a, -b, s)."""
    a, b = 0.4, 1.2
    s = 3.0
    sm_min = float(T.smooth_min(jnp.asarray(a), jnp.asarray(b), s))
    expected = -float(T.smooth_max(jnp.asarray(-a), jnp.asarray(-b), s))
    assert pytest.approx(expected, rel=1e-12) == sm_min


# ---------------------------------------------------------------------------
# smooth_positive_part
# ---------------------------------------------------------------------------

def test_smooth_positive_part_softplus_identity():
    """smooth_positive_part(x, s) == softplus(s*x)/s."""
    rng = np.random.default_rng(1)
    xs = jnp.asarray(rng.uniform(-5, 5, size=17))
    for s in (0.5, 1.0, 10.0):
        out = T.smooth_positive_part(xs, s)
        ref = jax.nn.softplus(s * xs) / s
        assert jnp.allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_smooth_positive_part_at_zero():
    """smooth_positive_part(0, s) = log(2)/s."""
    for s in (1.0, 10.0, 100.0):
        out = float(T.smooth_positive_part(jnp.asarray(0.0), s))
        assert pytest.approx(np.log(2.0) / s, rel=1e-10) == out


def test_smooth_positive_part_sharpness_limit():
    """As sharpness → ∞, smooth_positive_part(x, s) → max(x, 0)."""
    for x, expected in [(-1.0, 0.0), (0.5, 0.5), (3.0, 3.0)]:
        out = float(T.smooth_positive_part(jnp.asarray(x), 200.0))
        assert pytest.approx(expected, abs=1e-3) == out


def test_smooth_positive_part_grad_finite_everywhere():
    grad_fn = jax.grad(lambda x: T.smooth_positive_part(x, 5.0))
    for x in (-10.0, -1.0, 0.0, 0.5, 10.0):
        g = float(grad_fn(jnp.asarray(x)))
        assert np.isfinite(g)


# ---------------------------------------------------------------------------
# smooth_level_indicator
# ---------------------------------------------------------------------------

def test_smooth_level_indicator_above_count():
    """For a sharp sigmoid, the column-sum of indicators equals the
    number of levels above the threshold."""
    profile = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    ind = T.smooth_level_indicator(
        profile, threshold=2.5, sharpness=50.0, direction="above",
    )
    # Levels with values 3, 4, 5 are above 2.5 ⇒ count 3.
    assert pytest.approx(3.0, abs=1e-3) == float(jnp.sum(ind))


def test_smooth_level_indicator_below_complement():
    profile = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    above = T.smooth_level_indicator(profile, 2.5, 50.0, direction="above")
    below = T.smooth_level_indicator(profile, 2.5, 50.0, direction="below")
    # Above + below = 1 at every level (modulo sigmoid floating-point).
    assert jnp.allclose(above + below, 1.0, atol=1e-6)


def test_smooth_level_indicator_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        T.smooth_level_indicator(jnp.zeros((1, 3)), 0.0, 1.0, direction="sideways")


# ---------------------------------------------------------------------------
# smooth_lowest_crossing_index — the heart of the plume-localization machinery
# ---------------------------------------------------------------------------

def _make_buoyancy_like_profile(nlev: int, val_top: float, val_surf: float):
    """Surface-last profile with a single linear gradient.

    Returns a ``(1, nlev)`` profile increasing with altitude when
    ``val_top > val_surf`` (the canonical LFC orientation).
    """
    return jnp.linspace(val_top, val_surf, nlev)[None, :]


def test_smooth_lowest_crossing_single_crossing_at_high_sharpness():
    """For a single, monotone upward crossing, the diagnosed index
    converges to the analytical crossing point as sharpness → ∞."""
    nlev = 8
    profile = _make_buoyancy_like_profile(nlev, val_top=2.0, val_surf=-2.0)
    # Reversed (surface-first): -2 → 2 linearly.  Threshold 0.5
    # crosses at fractional surface-first index k where
    #   profile_rev[k] = 0.5  ⇔  -2 + (4/(nlev-1)) * k = 0.5
    #   k = 2.5 * (nlev-1) / 4 = 4.375.
    # Surface-last index = (nlev-1) - 4.375 = 2.625.
    idx = float(T.smooth_lowest_crossing_index(profile, 0.5, sharpness=100.0)[0])
    assert pytest.approx(2.625, abs=0.01) == idx


def test_smooth_lowest_crossing_picks_lowest_among_multiple():
    """When two upward crossings exist, the soft 'first crossing
    probability' formulation selects the lowest-altitude one."""
    # surface-last; surface-first reversed view alternates around 0.5:
    # rev = [-0.5, 0.7, 0.3, 0.7, 0.4, 0.6, -0.5, -1].
    # First upward crossing at pair k=0: -0.5 → 0.7 crosses 0.5 at
    # frac = (0.5 - (-0.5)) / (0.7 - (-0.5)) = 0.833;
    # idx_surface_first = 0.833; surface-last = 7 - 0.833 = 6.167.
    profile = jnp.array(
        [[-1.0, -0.5, 0.6, 0.4, 0.7, 0.3, 0.7, -0.5]], dtype=jnp.float64,
    )
    idx = float(T.smooth_lowest_crossing_index(profile, 0.5, sharpness=100.0)[0])
    assert pytest.approx(6.167, abs=0.05) == idx


def test_smooth_lowest_crossing_no_crossing_falls_back_to_surface():
    """A profile entirely below the threshold returns ~``nlev-1``
    (surface in surface-last coords) via the no-crossing fallback."""
    nlev = 8
    profile = jnp.full((1, nlev), -2.0, dtype=jnp.float64)
    idx = float(T.smooth_lowest_crossing_index(profile, 0.5, sharpness=10.0)[0])
    assert pytest.approx(float(nlev - 1), abs=0.01) == idx


def test_smooth_lowest_crossing_grad_finite_at_threshold():
    """Gradient of the diagnosed index w.r.t. the threshold is finite
    and non-zero — the AD-safety property that motivates the smooth
    primitive over a hard ``argmax``."""
    profile = _make_buoyancy_like_profile(8, val_top=2.0, val_surf=-2.0)

    def f(thr):
        return T.smooth_lowest_crossing_index(profile, thr, sharpness=10.0)[0]

    g = float(jax.grad(f)(jnp.asarray(0.5)))
    assert np.isfinite(g)
    assert abs(g) > 1e-6, "Gradient should be non-trivial at the threshold"


def test_smooth_lowest_crossing_rejects_short_profiles():
    with pytest.raises(ValueError, match="nlev >= 2"):
        T.smooth_lowest_crossing_index(jnp.zeros((1, 1)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# cape_trigger (thin convenience wrapper)
# ---------------------------------------------------------------------------

def test_cape_trigger_equals_smooth_step_of_difference():
    cape = jnp.array([0.0, 50.0, 100.0, 200.0])
    threshold = 70.0
    sharpness = 0.05
    out = T.cape_trigger(cape, threshold, sharpness)
    expected = T.smooth_step(cape - threshold, sharpness)
    assert jnp.allclose(out, expected)


def test_cape_trigger_grad_at_threshold_finite_nonzero():
    """``jax.grad`` w.r.t. each of the three arguments is finite at
    ``cape == threshold`` — protects against the failure mode where a
    naïve hard trigger (``cape > threshold``) zeros the gradient."""
    sharpness = 0.02
    threshold = 70.0

    cape = jnp.asarray(70.0)
    g_cape = float(jax.grad(lambda c: T.cape_trigger(c, threshold, sharpness))(cape))
    g_thr = float(jax.grad(lambda t: T.cape_trigger(cape, t, sharpness))(jnp.asarray(threshold)))
    g_s = float(jax.grad(lambda s: T.cape_trigger(cape, threshold, s))(jnp.asarray(sharpness)))

    for label, g in [("cape", g_cape), ("threshold", g_thr), ("sharpness", g_s)]:
        assert np.isfinite(g), f"grad w.r.t. {label} not finite"
    assert abs(g_cape) > 1e-6
    assert abs(g_thr) > 1e-6
