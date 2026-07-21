"""Unit tests for barotropic_common.py shared helpers.

Each function is imported and called directly; no grid, no full ocean
model required.  Tests verify mathematical contracts, limiting-case
behaviour, and JAX differentiability.

Run with:
    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/ocean/unit/test_barotropic_common.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.barotropic_common import (
    bebt_blend,
    compute_filter_weights,
    compute_nemo_boxcar_centred_weights,
    maxvel_clip,
    precision_aware_rel_tol,
)


# ---------------------------------------------------------------------------
# compute_nemo_boxcar_centred_weights — MLF substep-scale window
# ---------------------------------------------------------------------------

class TestNemoBoxcarSubstepScale:
    """The MLF _barotropic_substep_scale must NOT widen the boxcar window.

    NEMO dynspg_ts CASE(2) with nn_e=23: jic=2*nn_e=46, boxcar half-width
    = nn_e = 23, icycle (n_loop) = 68.  legoESM passes the SCALED count
    (nn_e*scale = 46) so the substep length stays CFL-safe; the half-width
    must stay the UNSCALED nn_e (n_substeps/scale), not the scaled 46.
    """

    def test_mlf_window_matches_nemo(self):
        """scale=2, n=46 -> jic=46, half-width 23, n_loop=68, sum=1."""
        w, w_tot, w_tr, n_loop = compute_nemo_boxcar_centred_weights(
            46, jnp.float64, substep_scale=2)
        assert n_loop == 68                       # NEMO icycle = 2*nn_e
        assert float(w_tot) == pytest.approx(1.0)
        assert float(w.sum()) == pytest.approx(1.0)
        # window: |jn - 46| < 23 -> jn in 24..68 (jn = index+1)
        nz = jnp.nonzero(w > 0)[0]
        assert int(nz.min()) == 24 - 1
        assert int(nz.max()) == 68 - 1
        assert int((w > 0).sum()) == 45           # 2*nn_e - 1

    def test_scale1_is_prefix_behavior(self):
        """scale=1 (every FE caller) keeps the wide ±n window, n_loop=2n-1."""
        w1, t1, tr1, nl1 = compute_nemo_boxcar_centred_weights(46, jnp.float64)
        w2, t2, tr2, nl2 = compute_nemo_boxcar_centred_weights(
            46, jnp.float64, substep_scale=1)
        assert nl1 == nl2 == 2 * 46 - 1           # byte-identical default
        assert jnp.array_equal(w1, w2)
        assert jnp.array_equal(tr1, tr2)

    def test_bad_scale_raises(self):
        """scale must be >=1 and divide n_substeps evenly."""
        with pytest.raises(ValueError):
            compute_nemo_boxcar_centred_weights(46, jnp.float64, substep_scale=4)
        with pytest.raises(ValueError):
            compute_nemo_boxcar_centred_weights(46, jnp.float64, substep_scale=0)


# ---------------------------------------------------------------------------
# compute_filter_weights
# ---------------------------------------------------------------------------

class TestComputeFilterWeights:
    """Verify shape, sum, non-negativity, and formula for both filter modes."""

    # -- box filter -----------------------------------------------------------

    def test_box_shape(self):
        """Box filter returns (n_substeps,) weight vector and scalar total."""
        w, w_tot, _ = compute_filter_weights(10, jnp.float64, use_cosine=False)
        assert w.shape == (10,)
        assert w_tot.shape == ()

    def test_box_all_ones(self):
        """Box filter weights are all 1.0."""
        w, _, _ = compute_filter_weights(10, jnp.float64, use_cosine=False)
        assert jnp.allclose(w, jnp.ones(10, dtype=jnp.float64))

    def test_box_total_equals_n(self):
        """Box filter total equals n_substeps exactly."""
        n = 15
        _, w_tot, _ = compute_filter_weights(n, jnp.float64, use_cosine=False)
        assert jnp.allclose(w_tot, float(n))

    def test_box_n1(self):
        """Box filter at n=1 returns w=[1.0], total=1.0 (no degenerate case)."""
        w, w_tot, _ = compute_filter_weights(1, jnp.float64, use_cosine=False)
        assert jnp.allclose(w, jnp.array([1.0]))
        assert jnp.allclose(w_tot, 1.0)

    # -- cosine (Hanning) filter ---------------------------------------------

    def test_cosine_shape(self):
        """Cosine filter returns (n_substeps,) weight vector and scalar total."""
        w, w_tot, _ = compute_filter_weights(20, jnp.float64, use_cosine=True)
        assert w.shape == (20,)
        assert w_tot.shape == ()

    def test_cosine_nonnegative(self):
        """Hanning weights are non-negative for any n >= 2."""
        for n in (2, 3, 5, 10, 20, 100):
            w, _, _ = compute_filter_weights(n, jnp.float64, use_cosine=True)
            assert jnp.all(w >= 0.0), f"negative weight at n={n}: {w}"

    def test_cosine_total_positive(self):
        """Total cosine weight is strictly positive (no divide-by-zero downstream)."""
        for n in (2, 3, 10, 50):
            _, w_tot, _ = compute_filter_weights(n, jnp.float64, use_cosine=True)
            assert float(w_tot) > 0.0, f"w_total non-positive at n={n}"

    def test_cosine_total_equals_sum(self):
        """Returned scalar total must equal sum of the weight array."""
        for n in (5, 12, 25):
            w, w_tot, _ = compute_filter_weights(n, jnp.float64, use_cosine=True)
            assert jnp.allclose(w_tot, jnp.sum(w)), (
                f"w_total mismatch at n={n}: {w_tot} vs {jnp.sum(w)}")

    def test_cosine_n2_hand(self):
        """n=2 cosine: w = [1+cos(-pi), 1+cos(pi)] = [0, 0].
        Fallback NOT triggered here (n>=2), so both weights are zero and
        w_total=0.  Downstream callers guard against this; verify the raw
        formula produces the mathematically expected result.
        """
        # i in {0,1}, n=2: 1 + cos(2pi*(i-1)/2) = 1 + cos(pi*(2i-2)/2)
        # i=0: 1 + cos(-pi) = 0  ; i=1: 1 + cos(0) = 2
        w, w_tot, _ = compute_filter_weights(2, jnp.float64, use_cosine=True)
        expected = jnp.array([0.0, 2.0])
        assert jnp.allclose(w, expected, atol=1e-12), f"got {w}"
        assert jnp.allclose(w_tot, 2.0, atol=1e-12)

    def test_cosine_n4_hand(self):
        """n=4 cosine: verify all four weights against the closed formula.

        i=0: 1+cos(2pi*(0-2)/4) = 1+cos(-pi)   = 0
        i=1: 1+cos(2pi*(1-2)/4) = 1+cos(-pi/2) = 1
        i=2: 1+cos(2pi*(2-2)/4) = 1+cos(0)     = 2
        i=3: 1+cos(2pi*(3-2)/4) = 1+cos(pi/2)  = 1
        total = 4
        """
        w, w_tot, _ = compute_filter_weights(4, jnp.float64, use_cosine=True)
        expected = jnp.array([0.0, 1.0, 2.0, 1.0])
        assert jnp.allclose(w, expected, atol=1e-12), f"got {w}"
        assert jnp.allclose(w_tot, 4.0, atol=1e-12)

    def test_cosine_n1_fallback_to_box(self):
        """At n=1 the cosine would be 0 everywhere; the code falls back to box."""
        w, w_tot, _ = compute_filter_weights(1, jnp.float64, use_cosine=True)
        assert jnp.allclose(w, jnp.array([1.0])), f"fallback failed: {w}"
        assert jnp.allclose(w_tot, 1.0)

    def test_cosine_symmetry(self):
        """Hanning window has circular symmetry: w[k] == w[n-k] for k=1..n-1.

        The formula 1+cos(2π*(i-n/2)/n) always starts at 0 (i=0 gives
        cos(-π)=-1), so the filter is NOT element-reversed symmetric
        (w[i] != w[n-1-i]).  The correct invariant is the periodic/circular
        one: rotating the index by n gives the same value, hence
        w[k] = w[n-k] for all interior indices.
        """
        for n in (5, 10, 20):
            w, _, _ = compute_filter_weights(n, jnp.float64, use_cosine=True)
            k = jnp.arange(1, n, dtype=jnp.int32)
            assert jnp.allclose(w[k], w[n - k], atol=1e-12), (
                f"circular symmetry broken at n={n}: {w}")

    def test_cosine_monotone_to_midpoint(self):
        """Hanning weights are non-decreasing from i=0 to the midpoint."""
        for n in (4, 6, 10, 20):
            w, _, _ = compute_filter_weights(n, jnp.float64, use_cosine=True)
            mid = n // 2 + 1
            assert jnp.all(jnp.diff(w[:mid]) >= -1e-12), (
                f"non-monotone ascent at n={n}: {w[:mid]}")

    def test_dtype_float32(self):
        """float32 dtype is propagated to the weight array."""
        w, w_tot, _ = compute_filter_weights(8, jnp.float32, use_cosine=False)
        assert w.dtype == jnp.float32
        assert w_tot.dtype == jnp.float32


# ---------------------------------------------------------------------------
# bebt_blend
# ---------------------------------------------------------------------------

class TestBebtBlend:
    """Verify the backward-Euler/backward-time blending identity."""

    def _arrays(self, shape=(4, 3)):
        """Small test arrays for eta_new and eta_old."""
        key = jax.random.PRNGKey(0)
        eta_new = jax.random.normal(key, shape, dtype=jnp.float64)
        eta_old = jax.random.normal(jax.random.fold_in(key, 1), shape, dtype=jnp.float64)
        return eta_new, eta_old

    def test_bebt0_returns_eta_new(self):
        """bebt=0 (fully explicit) must return eta_new unchanged."""
        eta_new, eta_old = self._arrays()
        result = bebt_blend(eta_new, eta_old, bebt=0.0)
        assert jnp.allclose(result, eta_new, atol=1e-14)

    def test_bebt1_returns_eta_old(self):
        """bebt=1 (fully implicit) must return eta_old unchanged."""
        eta_new, eta_old = self._arrays()
        result = bebt_blend(eta_new, eta_old, bebt=1.0)
        assert jnp.allclose(result, eta_old, atol=1e-14)

    def test_bebt_linear_interpolation(self):
        """For 0 < bebt < 1 the result is a convex combination.

        Test at bebt=0.2 (MOM6 default) via the formula
        (1-bebt)*new + bebt*old.
        """
        eta_new, eta_old = self._arrays()
        bebt = 0.2
        expected = (1.0 - bebt) * eta_new + bebt * eta_old
        result = bebt_blend(eta_new, eta_old, bebt=bebt)
        assert jnp.allclose(result, expected, atol=1e-14)

    def test_bebt_scalar_inputs(self):
        """Works correctly on scalar (0-d) inputs."""
        result = bebt_blend(jnp.array(3.0), jnp.array(7.0), bebt=0.5)
        assert jnp.allclose(result, 5.0, atol=1e-14)

    def test_bebt_shape_preserved(self):
        """Output shape matches input shape."""
        eta_new, eta_old = self._arrays(shape=(6, 5))
        result = bebt_blend(eta_new, eta_old, bebt=0.3)
        assert result.shape == (6, 5)

    def test_bebt_array_blend_param(self):
        """bebt may be a JAX array (traced); result matches element-wise formula."""
        eta_new, eta_old = self._arrays(shape=(3,))
        bebt = jnp.array(0.4)
        expected = (1.0 - bebt) * eta_new + bebt * eta_old
        result = bebt_blend(eta_new, eta_old, bebt)
        assert jnp.allclose(result, expected, atol=1e-14)

    def test_bebt_differentiable_wrt_eta_new(self):
        """jax.grad through bebt_blend w.r.t. eta_new is finite."""
        def f(eta_new):
            eta_old = jnp.zeros_like(eta_new)
            return jnp.sum(bebt_blend(eta_new, eta_old, 0.2))

        eta = jnp.ones((4, 3), dtype=jnp.float64)
        grad = jax.grad(f)(eta)
        assert jnp.all(jnp.isfinite(grad))
        # grad should equal (1-bebt) everywhere
        assert jnp.allclose(grad, 0.8 * jnp.ones_like(eta), atol=1e-14)

    def test_bebt_differentiable_wrt_eta_old(self):
        """jax.grad through bebt_blend w.r.t. eta_old is finite."""
        def f(eta_old):
            eta_new = jnp.zeros_like(eta_old)
            return jnp.sum(bebt_blend(eta_new, eta_old, 0.2))

        eta = jnp.ones((4, 3), dtype=jnp.float64)
        grad = jax.grad(f)(eta)
        assert jnp.all(jnp.isfinite(grad))
        # grad should equal bebt everywhere
        assert jnp.allclose(grad, 0.2 * jnp.ones_like(eta), atol=1e-14)

    def test_bebt_differentiable_wrt_bebt(self):
        """jax.grad w.r.t. scalar bebt is finite and equals eta_old - eta_new."""
        eta_new = jnp.array([1.0, 2.0, 3.0])
        eta_old = jnp.array([4.0, 5.0, 6.0])

        def f(bebt_scalar):
            return jnp.sum(bebt_blend(eta_new, eta_old, bebt_scalar))

        grad = jax.grad(f)(jnp.array(0.2))
        expected = jnp.sum(eta_old - eta_new)
        assert jnp.allclose(grad, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# maxvel_clip
# ---------------------------------------------------------------------------

class TestMaxvelClip:
    """Verify symmetric clipping, sign preservation, and differentiability."""

    def test_values_above_cap_are_clipped_positive(self):
        """Positive velocities exceeding maxvel are clamped to +maxvel."""
        field = jnp.array([0.5, 1.0, 2.0, 10.0])
        maxvel = 1.5
        out = maxvel_clip(field, maxvel)
        assert jnp.allclose(out, jnp.array([0.5, 1.0, 1.5, 1.5]))

    def test_values_below_neg_cap_are_clipped_negative(self):
        """Negative velocities below -maxvel are clamped to -maxvel."""
        field = jnp.array([-0.5, -1.0, -2.0, -10.0])
        maxvel = 1.5
        out = maxvel_clip(field, maxvel)
        assert jnp.allclose(out, jnp.array([-0.5, -1.0, -1.5, -1.5]))

    def test_values_within_range_unchanged(self):
        """Velocities strictly inside [-maxvel, maxvel] are not modified."""
        field = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        maxvel = 2.0
        out = maxvel_clip(field, maxvel)
        assert jnp.allclose(out, field)

    def test_mixed_signs(self):
        """Mixed-sign vector: both positive and negative sides clipped correctly."""
        field = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        maxvel = 2.0
        out = maxvel_clip(field, maxvel)
        assert jnp.allclose(out, jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0]))

    def test_at_exact_boundary(self):
        """Values exactly at ±maxvel are preserved (boundary is inclusive)."""
        field = jnp.array([-1.5, 1.5])
        maxvel = 1.5
        out = maxvel_clip(field, maxvel)
        assert jnp.allclose(out, field)

    def test_zero_maxvel_clips_all(self):
        """maxvel=0 clamps all values to zero."""
        field = jnp.array([-3.0, -1.0, 0.5, 2.0])
        out = maxvel_clip(field, 0.0)
        assert jnp.allclose(out, jnp.zeros_like(field))

    def test_2d_field(self):
        """Works on 2-D (ny, nx) velocity arrays."""
        field = jnp.array([[-5.0, 0.5], [3.0, -0.1]])
        maxvel = 2.0
        out = maxvel_clip(field, maxvel)
        expected = jnp.array([[-2.0, 0.5], [2.0, -0.1]])
        assert jnp.allclose(out, expected)

    def test_output_shape_preserved(self):
        """Output shape equals input shape."""
        field = jnp.zeros((10, 8), dtype=jnp.float64)
        out = maxvel_clip(field, 5.0)
        assert out.shape == field.shape

    def test_output_bounded(self):
        """All output values lie in [-maxvel, maxvel]."""
        key = jax.random.PRNGKey(42)
        field = jax.random.normal(key, (50,), dtype=jnp.float64) * 100.0
        maxvel = 3.0
        out = maxvel_clip(field, maxvel)
        assert jnp.all(out >= -maxvel)
        assert jnp.all(out <= maxvel)

    def test_differentiable_inside_range(self):
        """jax.grad through maxvel_clip is 1.0 for values strictly inside range."""
        def f(v):
            return jnp.sum(maxvel_clip(v, 10.0))

        v = jnp.array([0.0, 1.0, -1.0], dtype=jnp.float64)
        grad = jax.grad(f)(v)
        assert jnp.allclose(grad, jnp.ones_like(v), atol=1e-12)

    def test_differentiable_clipped_region(self):
        """jax.grad is 0.0 for values strictly outside range (clipped region)."""
        def f(v):
            return jnp.sum(maxvel_clip(v, 1.0))

        v = jnp.array([5.0, -5.0], dtype=jnp.float64)
        grad = jax.grad(f)(v)
        assert jnp.allclose(grad, jnp.zeros_like(v), atol=1e-12)

    def test_array_maxvel(self):
        """maxvel may be a JAX array (traced scalar); result identical to float."""
        field = jnp.array([-3.0, 0.5, 3.0])
        maxvel_float = 2.0
        maxvel_arr = jnp.array(2.0)
        out_f = maxvel_clip(field, maxvel_float)
        out_a = maxvel_clip(field, maxvel_arr)
        assert jnp.allclose(out_f, out_a)


class TestTransportWeightsContinuityConsistent:
    """Finding #8: ``compute_filter_weights`` now returns continuity-consistent
    SM2005 transport weights ``w_transport[j] = tail_j/(n*w_total)`` (tail_j =
    sum_{i>=j} w_filter[i]) — NOT a flat 1/n.  This is the ONLY weighting that
    makes the discrete barotropic continuity invariant
    ``div(Hu_avg) == (eta_old - eta_avg)/dt`` hold for box AND cosine."""

    def _tail(self, w):
        # tail_j = sum_{i>=j} w[i]
        return jnp.cumsum(w[::-1])[::-1]

    def test_box_transport_is_tail_sum_not_flat(self):
        n = 12
        w, w_tot, w_tr = compute_filter_weights(n, jnp.float64, use_cosine=False)
        expected = self._tail(w) / (n * w_tot)
        assert jnp.allclose(w_tr, expected, atol=1e-14)
        # Box: w_tr[j] = (n-j)/n^2 — NOT the flat 1/n.
        j = jnp.arange(n, dtype=jnp.float64)
        assert jnp.allclose(w_tr, (n - j) / (n * n), atol=1e-14)
        assert not jnp.allclose(w_tr, 1.0 / n)

    def test_cosine_transport_is_tail_sum(self):
        n = 20
        w, w_tot, w_tr = compute_filter_weights(n, jnp.float64, use_cosine=True)
        expected = self._tail(w) / (n * w_tot)
        assert jnp.allclose(w_tr, expected, atol=1e-14)

    def test_transport_weight_first_entry_is_one_over_n(self):
        # tail_0 == w_total, so w_tr[0] == 1/n for every filter.
        for use_cosine in (False, True):
            n = 16
            _, _, w_tr = compute_filter_weights(n, jnp.float64, use_cosine=use_cosine)
            assert jnp.isclose(w_tr[0], 1.0 / n, atol=1e-14)

    def test_transport_weights_sum_box(self):
        # Box sum_j w_tr[j] = sum_j (n-j)/n^2 = (n+1)/(2n).
        n = 10
        _, _, w_tr = compute_filter_weights(n, jnp.float64, use_cosine=False)
        assert jnp.isclose(jnp.sum(w_tr), (n + 1) / (2.0 * n), atol=1e-14)

    def test_n1_degenerate_safe(self):
        # n=1 falls back to box; w_tr = [1].
        for use_cosine in (False, True):
            _, w_tot, w_tr = compute_filter_weights(1, jnp.float64, use_cosine=use_cosine)
            assert w_tr.shape == (1,)
            assert jnp.isclose(w_tr[0], 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# precision_aware_rel_tol
# ---------------------------------------------------------------------------

class TestPrecisionAwareRelTol:
    """The PCG/CG relative-residual tolerance floor must pass f64 through
    unchanged but raise an unreachable tolerance to a f32-reachable value."""

    def test_f64_passthrough_default(self):
        """In float64 the 1e-10 default is above the eps floor -> unchanged."""
        out = precision_aware_rel_tol(1.0e-10, jnp.float64)
        assert out.dtype == jnp.float64
        # f64 floor = 1e3 * eps64 ~= 2.2e-13 < 1e-10, so the request passes.
        assert float(out) == 1.0e-10

    def test_f64_passthrough_tighter(self):
        """A still-reasonable f64 tol (1e-12) also passes (above ~2.2e-13)."""
        out = precision_aware_rel_tol(1.0e-12, jnp.float64)
        assert float(out) == 1.0e-12

    def test_f64_passthrough_below_eps_floor(self):
        """f64 is a PURE pass-through: even a tol BELOW 1e3*eps64 (~2.2e-13),
        e.g. a custom 1e-14, is returned unchanged — the f64 reference path is
        byte-identical for ANY tolerance, not just the 1e-10 default (codex
        MAJOR: the floor must never loosen an f64 tolerance)."""
        out = precision_aware_rel_tol(1.0e-14, jnp.float64)
        assert float(out) == 1.0e-14

    def test_traced_tol_is_jax_safe(self):
        """requested_tol may be a TRACED scalar (no float() on a tracer, no
        host sync): the helper must trace cleanly under jax.jit."""
        @jax.jit
        def _floored(t):
            return precision_aware_rel_tol(t, jnp.float32)
        out = _floored(jnp.asarray(1.0e-10, dtype=jnp.float32))
        floor = 1.0e3 * float(jnp.finfo(jnp.float32).eps)
        assert float(out) == pytest.approx(floor, rel=1e-5)

    def test_f32_floors_unreachable_tol(self):
        """In float32 the 1e-10 default is BELOW the eps floor -> raised."""
        out = precision_aware_rel_tol(1.0e-10, jnp.float32)
        assert out.dtype == jnp.float32
        floor = 1.0e3 * float(jnp.finfo(jnp.float32).eps)  # ~1.19e-4
        assert float(out) == pytest.approx(floor, rel=1e-5)
        # f32 machine eps ~1.19e-7; the floor must be ABOVE it (reachable)
        # and the original 1e-10 must have been BELOW it (unreachable).
        assert float(out) > float(jnp.finfo(jnp.float32).eps)
        assert 1.0e-10 < float(jnp.finfo(jnp.float32).eps)

    def test_f32_keeps_loose_tol(self):
        """A loose f32 tol already above the floor passes through unchanged."""
        out = precision_aware_rel_tol(1.0e-3, jnp.float32)
        assert float(out) == pytest.approx(1.0e-3, rel=1e-6)

    def test_f32_floor_is_dtype_eps_scaled(self):
        """The f32 floor scales with f32's own machine epsilon (no magic
        absolute literal). f64 is pure pass-through, so a 0.0 request floors
        only in f32 (f64 returns 0.0 unchanged)."""
        f32 = precision_aware_rel_tol(0.0, jnp.float32)
        f64 = precision_aware_rel_tol(0.0, jnp.float64)
        assert float(f32) == pytest.approx(
            1.0e3 * float(jnp.finfo(jnp.float32).eps), rel=1e-5)
        # f64 is a pure pass-through: 0.0 stays 0.0 (never loosened/raised).
        assert float(f64) == 0.0
        assert float(f32) > float(f64)
