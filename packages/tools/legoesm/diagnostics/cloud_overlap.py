"""Vertical cloud-overlap reductions for total-cloud-cover diagnostics.

The model carries a fractional *layer* cloud fraction (sundqvist / xu_randall /
resolved; see ``atmosphere.physics.clouds.cloud_fraction``).  The CMIP total
cloud cover ``clt`` is the vertical overlap of that layer field.  Random overlap
saturates to ~1 as soon as any single layer is cloudy and over-counts
vertically-correlated cloud; the operational choice is **maximum-random**
overlap (contiguous cloudy layers overlap maximally, cloud groups separated by
clear layers overlap randomly).

Pure, AD-/JIT-safe ``jnp`` (usable from the host-side diagnostics collector and,
if ever needed, inside a traced graph).  See issue #689 for why the previous
near-binary condensate mask was retired.
"""
from __future__ import annotations

import jax.numpy as jnp

# Smallest clear-sky fraction retained per layer so the max-random recursion's
# division by ``(1 - cf)`` never blows up when a layer is fully cloudy
# (``cf = 1``): there the running clear-sky product must collapse to 0 (total
# cover = 1), which the ``jnp.where`` guard enforces exactly (NaN-free).
_OVERLAP_EPS = 1.0e-6


def maximum_random_overlap(cloud_fraction: jnp.ndarray) -> jnp.ndarray:
    """Total column cloud cover from layer cloud fractions (maximum-random).

    Geleyn & Hollingsworth (1979); Hogan & Illingworth (2000).  Standard
    clear-sky recursion over the vertical axis (``axis=-1``; legoESM
    convention: index 0 = model top, index -1 = surface — the recursion is
    symmetric in direction, so orientation only has to be *consistent*):

        ``C_clr  = (1 - cf_0)``
        ``C_clr *= (1 - max(cf_{k-1}, cf_k)) / (1 - cf_{k-1})``  for k = 1..N-1
        ``C_tot  = 1 - C_clr``

    Adjacent cloudy layers overlap maximally (the ``max`` in the numerator);
    a clear layer (``cf_{k-1} = 0``) resets the denominator to 1 so the next
    cloud group multiplies in *randomly*.

    Parameters
    ----------
    cloud_fraction : jnp.ndarray
        Layer cloud fraction in [0, 1], vertical on the LAST axis, shape
        ``(..., nlev)``.  Values outside [0, 1] are clipped.

    Returns
    -------
    jnp.ndarray
        Total cloud cover in [0, 1], shape ``(...)`` (vertical axis removed).

    Examples
    --------
    - a single fully-cloudy layer (rest clear) -> ``clt = 1`` (= that layer);
    - a single partially-cloudy layer ``cf`` -> ``clt = cf``;
    - two adjacent layers ``cf = 0.5`` -> ``clt = 0.5`` (maximum overlap,
      *not* the 0.75 of pure random overlap);
    - a clear column -> ``clt = 0``.
    """
    cf = jnp.clip(cloud_fraction, 0.0, 1.0)
    nlev = cf.shape[-1]
    if nlev == 0:
        return jnp.zeros(cf.shape[:-1], dtype=cf.dtype)

    # Clear-sky fraction contributed by the topmost layer.
    clr = 1.0 - cf[..., 0]
    cf_prev = cf[..., 0]
    for k in range(1, nlev):
        cf_k = cf[..., k]
        denom = 1.0 - cf_prev
        # Incremental clear-sky factor between adjacent layers.  When the layer
        # above is (near) fully cloudy the column is overcast from there down,
        # so the clear product must stay 0: the guard returns 0 rather than a
        # 0/0 NaN.
        ratio = jnp.where(
            denom > _OVERLAP_EPS,
            (1.0 - jnp.maximum(cf_prev, cf_k)) / jnp.maximum(denom, _OVERLAP_EPS),
            0.0,
        )
        clr = clr * ratio
        cf_prev = cf_k
    return jnp.clip(1.0 - clr, 0.0, 1.0)
