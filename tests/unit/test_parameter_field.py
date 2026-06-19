"""Unit tests for :mod:`legoesm.training.parameter_field`.

Stage 7 of ``docs/COMPARE_REANALYSIS.md``: assemble a spatially-varying GCM
parameter field from per-column LES diagnoses — static scatter and the
environment-kernel generalization.  Checks placement, background fallback,
kernel-regression recovery at sample locations, smoothness, validity masking,
and differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.training.parameter_field import (
    environment_kernel_field,
    scatter_column_field,
)


def test_scatter_places_values_on_background():
    field = scatter_column_field(
        (2, 3),
        flat_indices=jnp.array([1, 5]),
        values=jnp.array([10.0, 20.0]),
        background=-1.0,
    )
    assert field.shape == (2, 3)
    f = np.asarray(field)
    assert f[0, 1] == pytest.approx(10.0)
    assert f[1, 2] == pytest.approx(20.0)
    # everything else is background
    assert f[0, 0] == pytest.approx(-1.0)
    assert (f == -1.0).sum() == 4


def test_scatter_valid_mask_drops_invalid():
    field = scatter_column_field(
        (2, 2),
        flat_indices=jnp.array([0, 3]),
        values=jnp.array([5.0, 7.0]),
        background=0.0,
        valid=jnp.array([True, False]),
    )
    f = np.asarray(field)
    assert f.reshape(-1)[0] == pytest.approx(5.0)
    assert f.reshape(-1)[3] == pytest.approx(0.0)  # invalid -> background


def test_scatter_differentiable_wrt_values():
    def loss(v):
        return jnp.sum(scatter_column_field((2, 2), jnp.array([0, 3]), v) ** 2)

    g = jax.grad(loss)(jnp.array([3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(g), [6.0, 8.0], rtol=1e-12)


def test_kernel_field_recovers_value_at_sample():
    # One sample; a grid column sitting exactly at its environment gets the
    # sample value (single-sample N-W regression = the value, weights cancel).
    grid_env = jnp.array([[300.0, 1000.0], [280.0, 0.0]])
    sample_env = jnp.array([[300.0, 1000.0]])
    sample_values = jnp.array([0.5])
    L = jnp.array([5.0, 500.0])
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L, background=-9.0
    )
    assert field.shape == (2,)
    assert float(field[0]) == pytest.approx(0.5, rel=1e-10)
    # The far column (~4.5 normalized σ away, single sample) is below the
    # 3σ total-weight floor ⇒ NOT extrapolated to ⇒ background.
    assert float(field[1]) == pytest.approx(-9.0)


def test_kernel_field_array_background_per_column_fallback():
    # Array background (accumulated round-k field): a column with NO env-similar
    # diagnosis this round retains its prior per-column value, while a column
    # near a sample is overwritten by the regression. Enables multi-round
    # campaign accumulation under strategy="environment".
    grid_env = jnp.array([[300.0, 1000.0], [0.0, 0.0]])  # col 0 near sample, col 1 far
    sample_env = jnp.array([[300.0, 1000.0]])
    sample_values = jnp.array([0.9])
    L = jnp.array([1.0, 1.0])
    prior = jnp.array([-1.0, -2.0])  # flat (ncol,) accumulated background
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L, background=prior,
    )
    assert float(field[0]) == pytest.approx(0.9, abs=1e-6)   # regressed
    assert float(field[1]) == pytest.approx(-2.0)            # prior retained


def test_kernel_field_array_background_row_major_ordering():
    # A genuinely 2-D (2, 3) background where C-order and F-order flattening
    # diverge: the kernel must flatten row-major (C-order) to match the flat
    # column contract. Only the column near the sample is overwritten; every
    # other column keeps its own prior value at the C-order position.
    grid_env = jnp.array([
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [0.0, 0.0], [300.0, 1000.0], [0.0, 0.0],   # flat index 4 near sample
    ])
    sample_env = jnp.array([[300.0, 1000.0]])
    sample_values = jnp.array([0.9])
    L = jnp.array([1.0, 1.0])
    prior_2d = jnp.arange(6.0).reshape(2, 3) - 10.0   # C-order: [-10..-5]
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L, background=prior_2d,
    )
    expected = np.array([-10.0, -9.0, -8.0, -7.0, 0.9, -5.0])  # C-order, idx 4 regressed
    np.testing.assert_allclose(np.asarray(field), expected, atol=1e-6)


def test_kernel_field_array_background_preserves_float64():
    # An accumulated float64 background must NOT be downcast to the float32 env/
    # sample dtype at no-neighbor fallback columns — else it diverges from the same
    # background used by the line-search base and partial steps drift untouched
    # columns. Requires x64 to express the mixed precision.
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("requires JAX_ENABLE_X64=1 to exercise mixed precision")
    grid_env = jnp.array([[0.0, 0.0]], dtype=jnp.float32)
    sample_env = jnp.array([[1.0e6, 1.0e6]], dtype=jnp.float32)  # far ⇒ no neighbor
    sample_values = jnp.array([0.5], dtype=jnp.float32)
    L = jnp.array([1.0, 1.0], dtype=jnp.float32)
    bg = jnp.asarray([0.123456789012345], dtype=jnp.float64)    # not exact in f32
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L, background=bg)
    assert field.dtype == jnp.float64
    np.testing.assert_array_equal(np.asarray(field), np.asarray(bg))  # exact, no downcast
    # ...while a scalar (weakly-typed) background does NOT spuriously promote the
    # common-case float32 kernel to float64 (locks the result_type neutrality).
    assert jnp.result_type(jnp.float32, 0.0) == jnp.float32
    f32_field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L, background=0.0)
    assert f32_field.dtype == jnp.float32


def test_kernel_field_array_background_size_mismatch_raises():
    grid_env = jnp.array([[300.0, 1000.0], [0.0, 0.0]])
    sample_env = jnp.array([[300.0, 1000.0]])
    with pytest.raises(ValueError, match="array background size"):
        environment_kernel_field(
            grid_env, sample_env, jnp.array([0.9]),
            length_scales=jnp.array([1.0, 1.0]),
            background=jnp.array([1.0, 2.0, 3.0]),  # size 3 != ncol 2
        )


def test_kernel_field_background_when_no_neighbor():
    grid_env = jnp.array([[300.0, 1000.0]])
    # Sample is astronomically far in normalized space ⇒ ~zero weight.
    sample_env = jnp.array([[0.0, 0.0]])
    sample_values = jnp.array([0.5])
    L = jnp.array([1.0, 1.0])  # tiny length scale ⇒ huge normalized distance
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L, background=-9.0,
    )
    assert float(field[0]) == pytest.approx(-9.0)


def test_kernel_field_nonfinite_grid_column_is_bg_and_ad_safe():
    """A NON-FINITE grid column (e.g. a NaN SST over land in the full-grid env
    predictors) must (a) fall back to background in the FORWARD field and (b) NOT leak
    a NaN adjoint into sample_values. Before the grid_finite sanitization the NaN
    kernel weights gave d(field)/d(values) = 0·NaN = NaN even though the forward value
    was correctly bg — breaking the docstring's AD-safety-w.r.t.-grid_env claim. The
    grid-side analog of the invalid-SAMPLE mask + the fail-loud sample-side guard in
    assemble_feedback_field (iter 179)."""
    grid_env = jnp.array([[280.0, 100.0, 2.0], [jnp.nan, 100.0, 2.0]])  # col 1 NaN env
    sample_env = jnp.array([[280.0, 100.0, 2.0]])
    L = jnp.array([5.0, 50.0, 2.0])

    def field_of(v):
        return environment_kernel_field(
            grid_env, sample_env, v, length_scales=L,
            valid=jnp.array([True]), background=0.4)

    field = field_of(jnp.array([10.0]))
    assert float(field[0]) == pytest.approx(10.0, abs=1e-6)   # valid col regressed
    assert float(field[1]) == pytest.approx(0.4)              # NaN-env col → background
    g = jax.grad(lambda v: jnp.sum(field_of(v)))(jnp.array([10.0]))
    assert bool(jnp.all(jnp.isfinite(g)))                     # no 0·NaN gradient leak
    # The coverage mask reports the NaN-env column as a fallback (not covered).
    _, cov = environment_kernel_field(
        grid_env, sample_env, jnp.array([10.0]), length_scales=L,
        valid=jnp.array([True]), background=0.4, return_coverage=True)
    assert bool(cov[0]) and not bool(cov[1])


def test_kernel_field_weighted_between_two_samples():
    # Two samples; a column equidistant in env space ⇒ average of the two.
    grid_env = jnp.array([[0.0]])
    sample_env = jnp.array([[-1.0], [1.0]])
    sample_values = jnp.array([2.0, 4.0])
    L = jnp.array([1.0])
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L
    )
    assert float(field[0]) == pytest.approx(3.0, rel=1e-10)


def test_kernel_field_closer_sample_dominates():
    grid_env = jnp.array([[0.1]])  # nearer to sample at 0 than at 2
    sample_env = jnp.array([[0.0], [2.0]])
    sample_values = jnp.array([1.0, 5.0])
    L = jnp.array([1.0])
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L
    )
    assert float(field[0]) < 3.0  # weighted toward the nearer (value 1.0)
    assert float(field[0]) > 1.0


def test_kernel_field_valid_mask_drops_sample():
    grid_env = jnp.array([[0.0]])
    sample_env = jnp.array([[0.0], [0.0]])
    sample_values = jnp.array([1.0, 100.0])
    L = jnp.array([1.0])
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=L,
        valid=jnp.array([True, False]),
    )
    assert float(field[0]) == pytest.approx(1.0, rel=1e-10)  # invalid dropped


def test_kernel_field_nan_invalid_sample_not_contaminating():
    """A NaN in an INVALID sample value must not poison the valid output."""
    grid_env = jnp.array([[0.0]])
    sample_env = jnp.array([[0.0], [0.0]])
    sample_values = jnp.array([1.0, jnp.nan])  # second is invalid + NaN
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=jnp.array([1.0]),
        valid=jnp.array([True, False]),
    )
    assert float(field[0]) == pytest.approx(1.0, rel=1e-12)
    assert jnp.isfinite(field[0])


def test_kernel_field_nan_invalid_env_not_contaminating():
    grid_env = jnp.array([[0.0]])
    sample_env = jnp.array([[0.0], [jnp.nan]])  # invalid sample has NaN env
    sample_values = jnp.array([2.0, 3.0])
    field = environment_kernel_field(
        grid_env, sample_env, sample_values, length_scales=jnp.array([1.0]),
        valid=jnp.array([True, False]),
    )
    assert float(field[0]) == pytest.approx(2.0, rel=1e-12)


def test_kernel_field_threshold_boundary():
    """A single sample just inside 3σ gets the value; just outside → background."""
    L = jnp.array([1.0])
    sample_env = jnp.array([[0.0]])
    sample_values = jnp.array([0.5])
    inside = environment_kernel_field(
        jnp.array([[2.9]]), sample_env, sample_values, length_scales=L,
        background=-9.0,
    )
    outside = environment_kernel_field(
        jnp.array([[3.1]]), sample_env, sample_values, length_scales=L,
        background=-9.0,
    )
    assert float(inside[0]) == pytest.approx(0.5, rel=1e-10)
    assert float(outside[0]) == pytest.approx(-9.0)


def test_kernel_field_differentiable_and_ad_safe():
    grid_env = jnp.array([[300.0], [250.0]])
    sample_env = jnp.array([[300.0]])
    L = jnp.array([10.0])

    def loss(v):
        return jnp.sum(
            environment_kernel_field(grid_env, sample_env, v, length_scales=L)
        )

    g = jax.grad(loss)(jnp.array([0.7]))
    assert bool(jnp.all(jnp.isfinite(g)))

    # AD-safe AND exactly zero gradient in the no-neighbor (background) branch.
    def loss_bg(v):
        far = jnp.array([[0.0]])
        return jnp.sum(
            environment_kernel_field(
                jnp.array([[1e6]]), far, v, length_scales=jnp.array([1.0])
            )
        )

    g2 = jax.grad(loss_bg)(jnp.array([0.7]))
    assert jnp.isfinite(g2[0])
    np.testing.assert_allclose(np.asarray(g2), [0.0], atol=1e-12)


def test_kernel_field_jit():
    grid_env = jnp.array([[300.0], [250.0]])
    sample_env = jnp.array([[300.0]])
    sample_values = jnp.array([0.5])
    L = jnp.array([10.0])
    f = jax.jit(
        lambda ge, se, sv: environment_kernel_field(ge, se, sv, length_scales=L)
    )(grid_env, sample_env, sample_values)
    assert bool(jnp.all(jnp.isfinite(f)))
    # Exact-match column recovers the sample value under jit.
    assert float(f[0]) == pytest.approx(0.5, rel=1e-10)
