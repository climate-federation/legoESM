"""Direct contract tests for shared NEMO source-statement rounding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.source_rounding import nemo_source_round


@pytest.fixture(autouse=True)
def _enable_x64():
    """Run these binary64 bit-preservation checks in float64 (restored after).

    Without this, the jnp.float64 inputs below are silently truncated to
    float32 before nemo_source_round ever sees them, so the bit-preservation
    assertions would pass vacuously on already-rounded-off float32 bits.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _bits(values: jax.Array) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def test_nemo_source_round_preserves_binary64_bits_and_jit_parity() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values_np = np.array(
        [-np.inf, -2.0, -subnormal, -0.0, subnormal, 3.0, np.inf],
        dtype=np.float64,
    )
    values = jnp.asarray(values_np, dtype=jnp.float64)

    eager = nemo_source_round(values)
    compiled = jax.jit(nemo_source_round)(values)

    # Ground truth computed purely in NumPy (never touches JAX/x64), so a
    # float64->float32->float64 round-trip on `values` itself cannot also
    # corrupt this comparison target. The subnormal (~4.94e-324) underflows
    # to exactly 0.0 under float32, so a truncation bug flips its bit
    # pattern away from expected_bits and this assertion catches it.
    expected_bits = values_np.view(np.uint64)
    np.testing.assert_array_equal(_bits(eager), expected_bits)
    np.testing.assert_array_equal(_bits(compiled), expected_bits)
    np.testing.assert_array_equal(_bits(compiled), _bits(eager))


def test_nemo_source_round_has_identity_gradient_for_finite_inputs() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values = jnp.asarray([-2.0, -subnormal, -0.0, subnormal, 3.0], dtype=jnp.float64)

    gradient = jax.grad(lambda x: jnp.sum(nemo_source_round(x)))(values)

    np.testing.assert_array_equal(np.asarray(gradient), np.ones(5, dtype=np.float64))
