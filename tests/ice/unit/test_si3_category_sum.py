"""Shared SI3 category sum follows gfortran ``SUM``: +0 seed, then left-to-right (DECISION 180)."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

from legoesm.ice.si3_orca2_state import fortran_sum_categories  # noqa: E402


def _bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


@pytest.mark.parametrize("axis", [-1, 2])
def test_all_negative_zero_categories_sum_to_positive_zero_under_production_jit(axis):
    arr = jnp.full((4, 3, 5), -0.0, dtype=jnp.float64)
    got = jax.jit(fortran_sum_categories, static_argnums=1)(arr, axis)
    assert not np.signbit(np.asarray(got)).any(), "jit: all -0 categories must sum to +0"
    with jax.disable_jit():
        eager = fortran_sum_categories(arr, axis)
    np.testing.assert_array_equal(_bits(got), _bits(eager))


def test_nested_sum_and_left_to_right_order_under_production_jit():
    nested = jax.jit(lambda a: fortran_sum_categories(fortran_sum_categories(a, axis=-1), axis=-1))
    assert not np.signbit(np.asarray(nested(jnp.full((2, 3, 4), -0.0)))).any()
    # ((+0 + 1e16) + 1) + -1e16 == 0 only in Fortran's order; other groupings give 1 or 2.
    vals = jnp.array([[1e16, 1.0, -1e16]])
    np.testing.assert_array_equal(np.asarray(jax.jit(fortran_sum_categories)(vals)), [0.0])
    mixed = jnp.array([[-0.0, -2.5, 2.5]])  # (+0 + -0) + -2.5 + 2.5 = +0
    assert not np.signbit(np.asarray(jax.jit(fortran_sum_categories)(mixed))).any()
