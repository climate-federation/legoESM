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


def fortran_min(a, b):
    """Fortran ``MIN(a, b)`` as gfortran ``minsd`` compiles it: ``b`` only when strictly smaller.

    A tie (including ``+0``/``-0``) or a NaN operand returns ``a``.  gfortran may
    swap the operands of a given ``MIN`` site, so each caller cites its object code.
    """
    return jnp.where(b < a, b, a)


def fortran_max(a, b):
    """Fortran ``MAX(a, b)`` as gfortran ``maxsd`` compiles it: ``b`` only when strictly larger."""
    return jnp.where(b > a, b, a)


def fortran_sum0(terms):
    """Fortran ``SUM(x(:))`` over the last axis: ``+0`` then each term in index order."""
    # Materialized seed: XLA otherwise folds 0 + x to x, which keeps a -0 term's sign.
    acc = nemo_source_round(jnp.zeros(terms.shape[:-1], dtype=terms.dtype))
    for k in range(terms.shape[-1]):
        acc = nemo_source_round(acc + terms[..., k])
    return acc


def ieee_is_zero(value):
    """``value == 0._wp`` on the binary64 bit pattern (a positive subnormal stays nonzero)."""
    bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
    return (bits << jnp.uint64(1)) == jnp.uint64(0)


__all__ = ("fortran_max", "fortran_min", "fortran_sum0", "ieee_is_zero", "nemo_source_round")
