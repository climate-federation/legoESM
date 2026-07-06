"""Unit tests for :func:`legoesm.atmosphere.physics._shared.broadcast_column_param`.

The enabler for per-column coefficient promotion (docs/COMPARE_REANALYSIS.md):
a scalar coefficient stays byte-identical; a per-column ``(ncol,)`` field is
reshaped to broadcast over the vertical axis.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics._shared import broadcast_column_param


def test_scalar_passes_through_unchanged():
    like = jnp.ones((3, 5))
    out = broadcast_column_param(jnp.asarray(0.1), like)
    assert out.ndim == 0
    assert float(out) == pytest.approx(0.1)
    # scalar * (ncol, nlev) broadcasts as before (byte-identical path).
    np.testing.assert_allclose(np.asarray(out * like), 0.1 * np.ones((3, 5)))


def test_per_column_reshapes_to_broadcast_over_vertical():
    like = jnp.ones((3, 5))
    value = jnp.array([1.0, 2.0, 3.0])  # (ncol,)
    out = broadcast_column_param(value, like)
    assert out.shape == (3, 1)
    prod = out * like  # (3, 5): each column scaled by its value
    np.testing.assert_allclose(np.asarray(prod[0]), 1.0)
    np.testing.assert_allclose(np.asarray(prod[1]), 2.0)
    np.testing.assert_allclose(np.asarray(prod[2]), 3.0)


def test_per_column_broadcasts_over_higher_rank():
    like = jnp.ones((2, 4, 3))  # (ncol, nlev, n)
    out = broadcast_column_param(jnp.array([5.0, 7.0]), like)
    assert out.shape == (2, 1, 1)
    assert float((out * like)[1, 2, 1]) == pytest.approx(7.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="!= column count"):
        broadcast_column_param(jnp.array([1.0, 2.0]), jnp.ones((3, 5)))


def test_2d_value_raises():
    with pytest.raises(ValueError, match="scalar or 1-D"):
        broadcast_column_param(jnp.ones((3, 5)), jnp.ones((3, 5)))


def test_jit_scalar_and_per_column():
    """ndim is static even under jit, so both paths trace cleanly."""
    import jax

    like = jnp.ones((3, 5))
    scalar_out = jax.jit(lambda v: broadcast_column_param(v, like) * like)(
        jnp.asarray(0.2))
    np.testing.assert_allclose(np.asarray(scalar_out), 0.2)
    pc_out = jax.jit(lambda v: broadcast_column_param(v, like) * like)(
        jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(np.asarray(pc_out[1]), 2.0)
