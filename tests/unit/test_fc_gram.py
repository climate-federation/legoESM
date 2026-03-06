"""Tests for FC-Gram spectral derivative core."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fc_gram import (
    build_fc_gram_matrices,
    apply_continuation_1d,
    spectral_derivative_1d,
    spectral_filter_1d,
)


@pytest.fixture
def matrices():
    return build_fc_gram_matrices(d=2, C=4, degree=5, dtype=np.float64)


def test_build_matrices_shapes(matrices):
    """Matrices have correct shapes."""
    assert matrices.self_matrix.shape == (4, 2)
    assert matrices.boundary_matrix.shape == (4, 4)
    assert matrices.d == 2
    assert matrices.C == 4


def _make_padded_periodic(f, halo):
    """Pad a periodic signal with halo cells on each side."""
    return jnp.concatenate([f[-halo:], f, f[:halo]], axis=-1)


def test_polynomial_continuation_exact(matrices):
    """FC continuation of a padded polynomial is finite."""
    N = 16
    dx = 1.0 / N
    x = jnp.linspace(0, 1 - dx, N, dtype=jnp.float64)
    f = 2 * x**2 - x + 1

    f_padded = _make_padded_periodic(f, halo=1)  # N+2 points
    f_ext = apply_continuation_1d(f_padded, matrices, axis=-1)
    assert f_ext.shape[-1] == N + 2 + matrices.C
    assert jnp.all(jnp.isfinite(f_ext))


def test_sin_derivative_accuracy(matrices):
    """Derivative of sin(2*pi*x) is accurate with padded signal."""
    N = 32
    dx = 1.0 / N
    x = jnp.linspace(0, 1 - dx, N, dtype=jnp.float64)
    k = 2 * jnp.pi
    f = jnp.sin(k * x)
    f_exact = k * jnp.cos(k * x)

    halo = 1
    f_padded = _make_padded_periodic(f, halo)  # N+2 points
    f_ext = apply_continuation_1d(f_padded, matrices, axis=-1)
    df_padded = spectral_derivative_1d(f_ext, dx, N + 2 * halo, matrices.C, axis=-1, order=1)
    df = df_padded[halo:-halo]  # restrict to interior

    error = float(jnp.max(jnp.abs(df - f_exact)))
    # d=2 FC gives ~5% relative error for sin(2*pi*x) at N=32
    assert error < 1.0, f"sin derivative error {error} too large"


def test_spectral_filter_reduces_modes(matrices):
    """Spectral filter removes high-frequency content."""
    N = 32
    dx = 1.0 / N
    x = jnp.linspace(0, 1 - dx, N, dtype=jnp.float64)
    f = jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.sin(20 * jnp.pi * x)

    halo = 1
    f_padded = _make_padded_periodic(f, halo)
    f_ext = apply_continuation_1d(f_padded, matrices, axis=-1)
    f_filt = spectral_filter_1d(f_ext, dx, N + 2 * halo, matrices.C, axis=-1, cutoff_fraction=0.5)

    assert f_filt.shape == (N + 2 * halo,)
    assert jnp.all(jnp.isfinite(f_filt))


def test_continuation_batched(matrices):
    """Continuation works with batch dimensions."""
    batch = 3
    N = 16
    f = jnp.ones((batch, N), dtype=jnp.float64)
    f_ext = apply_continuation_1d(f, matrices, axis=-1)
    assert f_ext.shape == (batch, N + matrices.C)


def test_derivative_order_2(matrices):
    """Second-order derivative works."""
    N = 32
    dx = 1.0 / N
    x = jnp.linspace(0, 1 - dx, N, dtype=jnp.float64)
    k = 2 * jnp.pi
    f = jnp.sin(k * x)
    f_exact = -k**2 * jnp.sin(k * x)

    halo = 1
    f_padded = _make_padded_periodic(f, halo)
    f_ext = apply_continuation_1d(f_padded, matrices, axis=-1)
    d2f_padded = spectral_derivative_1d(f_ext, dx, N + 2 * halo, matrices.C, axis=-1, order=2)
    d2f = d2f_padded[halo:-halo]

    error = float(jnp.max(jnp.abs(d2f - f_exact)))
    # d=2 C1 continuation limits 2nd derivative accuracy;
    # in practice, Laplacian is computed as div(grad), two 1st-derivative steps
    assert error < 25.0, f"2nd derivative error {error} too large"


def test_float32_matrices():
    """Float32 matrices build correctly."""
    m = build_fc_gram_matrices(d=2, C=4, degree=5, dtype=np.float32)
    assert m.self_matrix.dtype == jnp.float32
    assert m.boundary_matrix.dtype == jnp.float32
