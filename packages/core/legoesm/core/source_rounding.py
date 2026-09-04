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

    ``optimization_barrier`` alone does not provide the required compiled
    materialization.  The finite-classification ``select`` and ``copysign``
    copy are the operative guard.  They form an IEEE identity, including
    signed infinities and NaNs, and remain differentiable for finite physical
    inputs.
    """
    value = jax.lax.optimization_barrier(value)
    return jnp.where(
        jnp.isfinite(value),
        value,
        jnp.copysign(jnp.abs(value), value),
    )


__all__ = ("nemo_source_round",)
