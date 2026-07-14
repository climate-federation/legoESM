"""Tracer positivity filter tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.shared.tracer_positivity import (
    apply_positive_filter_state, clip_positive,
    clip_positive_compensated,
)

jax.config.update("jax_enable_x64", True)


def test_clip_positive_pointwise():
    """Negatives → 0; positives untouched."""
    q = jnp.array([-1.0, 0.0, 0.5, 1.0, -1.0e-12])
    out = clip_positive(q)
    np.testing.assert_array_equal(
        np.asarray(out),
        np.array([0.0, 0.0, 0.5, 1.0, 0.0]),
    )


def test_clip_positive_compensated_preserves_mean_uniform_weights():
    """Uniform weights: sum(q_out) == sum(q_in) — the compensated
    clip preserves global mass."""
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal(100))  # mean ≈ 0
    out = clip_positive_compensated(q)
    np.testing.assert_allclose(
        float(jnp.sum(out)), float(jnp.sum(q)),
        rtol=1.0e-12, atol=1.0e-12,
    )
    assert float(jnp.min(out)) >= 0.0


def test_clip_positive_compensated_preserves_weighted_mean():
    """With per-cell weights, sum(q_out · w) == sum(q_in · w)."""
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal(50))
    w = jnp.asarray(rng.uniform(0.5, 2.0, 50))
    out = clip_positive_compensated(q, weights=w)
    np.testing.assert_allclose(
        float(jnp.sum(out * w)), float(jnp.sum(q * w)),
        rtol=1.0e-12, atol=1.0e-10,
    )
    assert float(jnp.min(out)) >= 0.0


def test_clip_positive_compensated_all_positive_no_change():
    """When the input has NO negatives, the compensated filter is
    a no-op (no deficit → no rescaling)."""
    q = jnp.array([0.1, 0.5, 0.7, 1.0])
    out = clip_positive_compensated(q)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(q), rtol=1.0e-12, atol=1.0e-12,
    )


def test_clip_positive_compensated_handles_zero_positive_mass():
    """Pathological: all input <= 0 → output is all zeros (no positive
    mass to rescale; mass is dropped to preserve non-negativity)."""
    q = jnp.array([-1.0, -2.0, -0.5])
    out = clip_positive_compensated(q)
    np.testing.assert_array_equal(
        np.asarray(out), np.zeros(3),
    )


def test_clip_positive_compensated_jax_grad_finite():
    """jax.grad through the compensated clip on a smooth loss."""
    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.standard_normal(20))

    def loss_fn(x):
        return jnp.sum(clip_positive_compensated(x) ** 2)

    g = jax.grad(loss_fn)(q)
    assert g.shape == q.shape
    assert bool(jnp.all(jnp.isfinite(g)))


# --------------------------------------------------------------------- #
# State-level filter                                                    #
# --------------------------------------------------------------------- #


@pytest.fixture
def plane_state_with_negative_qv():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        make_flat_plane_terrain_metric, make_rest_state,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_height_coordinate
    grid = create_plane_grid(
        nx=4, ny=4, nlev=4, dx=2_000.0, dy=2_000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(4, H=4_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    # Allocate 3 tracers; inject negative q_v at a few cells.
    tracers = jnp.zeros((4, 4, 4, 3), dtype=jnp.float64).at[
        ..., 0,
    ].set(0.01)
    tracers = tracers.at[0, 0, 0, 0].set(-0.001)   # negative q_v
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )
    return state, grid, hc, tm


def test_apply_positive_filter_state_clip_mode_zeros_negatives(
    plane_state_with_negative_qv,
):
    state, *_ = plane_state_with_negative_qv
    out = apply_positive_filter_state(
        state, tracer_slots_to_filter=(0,), mode="clip",
    )
    assert float(jnp.min(out.tracers.data[..., 0])) >= 0.0
    # The originally-negative cell should now be 0.
    np.testing.assert_allclose(
        float(out.tracers.data[0, 0, 0, 0]), 0.0,
    )


def test_apply_positive_filter_state_compensated_preserves_mean(
    plane_state_with_negative_qv,
):
    state, *_ = plane_state_with_negative_qv
    sum_before = float(jnp.sum(state.tracers.data[..., 0]))
    out = apply_positive_filter_state(
        state, tracer_slots_to_filter=(0,), mode="compensated",
    )
    sum_after = float(jnp.sum(out.tracers.data[..., 0]))
    np.testing.assert_allclose(
        sum_after, sum_before, rtol=1.0e-12, atol=1.0e-10,
    )
    assert float(jnp.min(out.tracers.data[..., 0])) >= 0.0


def test_apply_positive_filter_state_unfiltered_slots_untouched(
    plane_state_with_negative_qv,
):
    """Only slot 0 was selected; slots 1, 2 must be identical."""
    state, *_ = plane_state_with_negative_qv
    out = apply_positive_filter_state(
        state, tracer_slots_to_filter=(0,), mode="clip",
    )
    np.testing.assert_array_equal(
        np.asarray(out.tracers.data[..., 1]),
        np.asarray(state.tracers.data[..., 1]),
    )
    np.testing.assert_array_equal(
        np.asarray(out.tracers.data[..., 2]),
        np.asarray(state.tracers.data[..., 2]),
    )


def test_apply_positive_filter_state_default_filters_all_slots(
    plane_state_with_negative_qv,
):
    state, *_ = plane_state_with_negative_qv
    # Inject a tiny negative into slot 1 too.
    t = state.tracers.data.at[2, 2, 2, 1].set(-1.0e-6)
    state = state._replace(tracers=state.tracers.replace(data=t))
    out = apply_positive_filter_state(state, mode="clip")
    assert float(jnp.min(out.tracers.data)) >= 0.0


def test_apply_positive_filter_state_rejects_bad_mode(
    plane_state_with_negative_qv,
):
    state, *_ = plane_state_with_negative_qv
    with pytest.raises(ValueError, match="Unknown positivity"):
        apply_positive_filter_state(state, mode="bogus")


def test_apply_positive_filter_state_rejects_oob_slot(
    plane_state_with_negative_qv,
):
    state, *_ = plane_state_with_negative_qv
    with pytest.raises(ValueError, match="out of range"):
        apply_positive_filter_state(
            state, tracer_slots_to_filter=(99,), mode="clip",
        )


def test_apply_positive_filter_state_rejects_non_field_state():
    """Codex iter-1: state without the required Field protocol raises
    TypeError instead of silently AttributeError-ing."""
    class _Dummy:
        tracers = jnp.zeros((4, 4, 4, 3))
    with pytest.raises(TypeError, match="tracers.*Field"):
        apply_positive_filter_state(_Dummy(), mode="clip")


def test_compensated_rejects_mismatched_weights_shape():
    """Codex iter-1: weights must match q.shape exactly (no
    broadcasting allowed → otherwise the conservation invariant
    becomes ambiguous)."""
    q = jnp.ones(10)
    w_bad = jnp.ones(5)
    with pytest.raises(ValueError, match="weights.shape"):
        clip_positive_compensated(q, weights=w_bad)


def test_compensated_rejects_negative_weights():
    """Codex iter-1: negative weights make D, pos_mass + scale
    physically meaningless. Reject at host check time."""
    q = jnp.array([1.0, -0.5, 0.3])
    w_bad = jnp.array([1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="weights must be non-negative"):
        clip_positive_compensated(q, weights=w_bad)


def test_compensated_skips_negative_weight_check_under_jit():
    """Codex iter-2: pin that the documented JIT-skip behaviour
    matches the docstring. Under jax.jit, the host-side non-
    negativity check is skipped (TracerArrayConversionError trapped)
    — caller is responsible. Without this guard, the validation
    would crash every JIT trace that uses dynamic weights."""
    q = jnp.array([1.0, -0.5, 0.3])
    bad_weights = jnp.array([1.0, -1.0, 1.0])

    @jax.jit
    def _compensated_jit(q, w):
        return clip_positive_compensated(q, weights=w)

    # Under JIT the negative-weight host check is skipped — no
    # ValueError. The output is mathematically nonsense (the
    # documented gap) but does not crash.
    out = _compensated_jit(q, bad_weights)
    assert out.shape == q.shape
    # Output is finite (no NaN from the validator).
    assert bool(jnp.all(jnp.isfinite(out)))


def test_compensated_mostly_negative_input_zeros_column():
    """Codex iter-1 conservation contract: when sum(q·w) < 0, the
    compensated filter cannot preserve the (negative) mean while
    enforcing non-negativity → it drops mass to zero, recording the
    "input was malformed" signal. This is the documented degenerate
    behaviour."""
    q = jnp.array([1.0, -2.0, -3.0])     # sum = -4 (negative)
    out = clip_positive_compensated(q)
    np.testing.assert_array_equal(np.asarray(out), np.zeros(3))
