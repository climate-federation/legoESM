"""NEMO source-statement materialization for bit-exact oracle paths.

This shared helper is intentionally separate from the precision-policy module:
the scalar-libm policy is imported byte-for-byte from GYRE commit 61180a6776c,
while lane 3b already depended on this Round-10 source-rounding identity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def nemo_source_round(value: jax.Array) -> jax.Array:
    """Materialize one NEMO source operation before its next consumer.

    XLA strips ``optimization_barrier`` from optimized HLO.  The surviving
    ``isfinite``/``select``/``copysign`` chain is the operative contraction
    guard.  It is an IEEE identity, including signed infinities and NaNs, and
    remains differentiable for finite physical inputs.
    """
    value = jax.lax.optimization_barrier(value)
    return jnp.where(
        jnp.isfinite(value),
        value,
        jnp.copysign(jnp.abs(value), value),
    )


__all__ = ("nemo_source_round",)
