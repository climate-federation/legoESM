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


def nemo_source_barrier(value: jax.Array) -> jax.Array:
    """Prevent XLA from rewriting one source operand before an FP operation."""
    return jax.lax.optimization_barrier(jnp.asarray(value))


def nemo_fortran_sign(magnitude: jax.Array, sign_source: jax.Array) -> jax.Array:
    """NEMO record-build ``SIGN`` semantics, including ``-0``.

    The record build defines ``SIGN(a, b)`` in ``lib_fortran`` using
    ``b >= 0``.  Therefore both ``+0`` and ``-0`` select ``ABS(a)``; this is
    deliberately different from :func:`jax.numpy.copysign` for ``b == -0``.
    """
    magnitude = jnp.abs(magnitude)
    return jnp.where(sign_source >= 0.0, magnitude, -magnitude)


def nemo_fortran_max(first: jax.Array, second: jax.Array) -> jax.Array:
    """Record-object ``MAXSD`` ordering: the second operand wins a tie."""
    return jnp.where(first > second, first, second)


def nemo_fortran_min(first: jax.Array, second: jax.Array) -> jax.Array:
    """Record-object ``MINSD`` ordering: the second operand wins a tie."""
    return jnp.where(first < second, first, second)


__all__ = (
    "nemo_fortran_max",
    "nemo_fortran_min",
    "nemo_fortran_sign",
    "nemo_source_barrier",
    "nemo_source_round",
)
