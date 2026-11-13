"""Pytree state of the wave component.

Only the directional spectrum is prognostic.  Depth and the sea mask are
carried alongside it because every step needs them and because a
checkpoint must restore a self-consistent set; they are not advanced.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class WaveState(NamedTuple):
    spectra: jax.Array   # (B, Nlon, Nlat, Ndir, Nfreq) [m2 s rad-1]
    depth: jax.Array     # (B, Nlon, Nlat) [m], positive where sea
    mask: jax.Array      # (B, Nlon, Nlat) bool, True where sea
    spatial_phase: jax.Array | int = 0


def init_wave_state(
    depth: jax.Array,
    *,
    n_dir: int,
    n_freq: int,
    dtype=jnp.float32,
) -> WaveState:
    """Start from a calm sea; ``depth`` sets the mask."""
    depth = jnp.asarray(depth, dtype=dtype)
    while depth.ndim < 3:
        depth = depth[None, ...]
    if depth.ndim != 3:
        raise ValueError(f"depth must be 2-D (Nlon, Nlat) or 3-D (B,Nlon,Nlat); got {depth.shape}")
    mask = depth > 0
    spectra = jnp.zeros(
        (depth.shape[0], depth.shape[1], depth.shape[2], n_dir, n_freq), dtype=dtype)
    return WaveState(spectra=spectra, depth=depth, mask=mask,
                     spatial_phase=jnp.asarray(0, dtype=jnp.int32))
