"""Quasi-random (low-discrepancy) sampling for variance reduction.

The oracle (microhh) uses Sobol QRNG per thread to converge the ray tracer in
fewer photons. A full Sobol needs the Joe-Kuo direction-number tables (external
data); this module provides a self-contained, fully-jittable alternative — the
Halton sequence (radical inverse in bases 2 and 3) with a per-column
Cranley-Patterson rotation (random shift mod 1) to decorrelate columns and act
as the random scramble. Applied to the DOMINANT smooth dimension (the photon
launch position within a column), it lowers the variance of column/surface/TOD
fluxes at fixed photon count — i.e. fewer photons (and less GPU memory) for a
target error. The deep multiple-scattering walk stays on counter-based threefry,
where QRNG offers little (high effective dimension); this "padded QRNG" split is
standard MC-rendering practice. Swapping Halton for table-driven Sobol is a drop
-in upgrade if the Joe-Kuo tables are added.
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array

# Digit counts cover photon-per-column indices up to base**n_digits (2^24 ~ 16M,
# 3^16 ~ 43M) — far beyond any per-column photon count.
_N_DIGITS_BASE2 = 24
_N_DIGITS_BASE3 = 16


def _radical_inverse(i: Array, base: int, n_digits: int) -> Array:
  """Van der Corput radical inverse of integer ``i`` in ``base`` (in [0,1))."""
  i = i.astype(jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
  result = jnp.zeros(i.shape, dtype=jnp.float64 if jax.config.jax_enable_x64
                     else jnp.float32)
  inv_base = 1.0 / base
  f = inv_base
  x = i
  for _ in range(n_digits):
    d = x % base
    result = result + d.astype(result.dtype) * f
    x = x // base
    f = f * inv_base
  return result


def halton_2d(index: Array, rot: Array) -> tuple[Array, Array]:
  """Scrambled Halton 2D point for each ``index`` (base 2, base 3) with a
  Cranley-Patterson rotation ``rot`` (shape ``(...,2)``, each in [0,1)).

  Returns ``(hx, hy)`` in [0,1), low-discrepancy across ``index`` per fixed
  ``rot`` and decorrelated across distinct ``rot`` (e.g. per column).
  """
  h2 = _radical_inverse(index, 2, _N_DIGITS_BASE2)
  h3 = _radical_inverse(index, 3, _N_DIGITS_BASE3)
  hx = jnp.mod(h2 + rot[..., 0], 1.0)
  hy = jnp.mod(h3 + rot[..., 1], 1.0)
  return hx, hy
