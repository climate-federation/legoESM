"""Direct contract tests for shared NEMO source-statement rounding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.source_rounding import nemo_source_round


def _bits(values: jax.Array) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def test_nemo_source_round_preserves_binary64_bits_and_jit_parity() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values = jnp.asarray(
        [-np.inf, -2.0, -subnormal, -0.0, subnormal, 3.0, np.inf],
        dtype=jnp.float64,
    )

    eager = nemo_source_round(values)
    compiled = jax.jit(nemo_source_round)(values)

    np.testing.assert_array_equal(_bits(eager), _bits(values))
    np.testing.assert_array_equal(_bits(compiled), _bits(values))
    np.testing.assert_array_equal(_bits(compiled), _bits(eager))


def test_nemo_source_round_has_identity_gradient_for_finite_inputs() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values = jnp.asarray([-2.0, -subnormal, -0.0, subnormal, 3.0], dtype=jnp.float64)

    gradient = jax.grad(lambda x: jnp.sum(nemo_source_round(x)))(values)

    np.testing.assert_array_equal(np.asarray(gradient), np.ones(5, dtype=np.float64))
