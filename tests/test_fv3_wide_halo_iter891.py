"""FV3_3D iter 891: wide-halo pad_halo smoke test for nord>=2 restructure.

Foundational step toward FV3-faithful expanding-halo Laplacian
iteration (sw_core.F90:1746-1782 with nt = nord-n decreasing).

Verifies ``pad_halo(data, halo=k)`` already works for k ∈ {1, 2, 3}
(declared in halo.py:523-525, supports halo=1/2/3 via
_pad_halo_local_h2 / _pad_halo_local_h3 dispatch).

Confirms the foundation:
  1. ``pad_halo(divg_d, halo=2)`` returns shape ``(6, n+4, n+4)`` —
     the wider-halo pre-pad needed for nord=2 expanding-halo pattern.
  2. Interior (halo strip removed) matches the unpadded input.
  3. Padded array is finite for finite inputs (cube-vertex cells
     populated, not NaN).
  4. ``halo=3`` raises NotImplementedError → no — supported per
     halo.py:524.  ``halo=4`` raises NotImplementedError.

Once this foundation is verified, iter-892+ will implement the
expanding-halo Laplacian iteration using these wider pads.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.halo import pad_halo


def test_pad_halo_width_1_shape():
    """Canonical halo=1: (6, n, n) → (6, n+2, n+2)."""
    n = 8
    data = jnp.ones((6, n, n))
    padded = pad_halo(data, halo=1)
    assert padded.shape == (6, n + 2, n + 2)


def test_pad_halo_width_2_shape():
    """Wide halo=2: (6, n, n) → (6, n+4, n+4)."""
    n = 8
    data = jnp.ones((6, n, n))
    padded = pad_halo(data, halo=2)
    assert padded.shape == (6, n + 4, n + 4)


def test_pad_halo_width_3_shape():
    """Wide halo=3: (6, n, n) → (6, n+6, n+6)."""
    n = 8
    data = jnp.ones((6, n, n))
    padded = pad_halo(data, halo=3)
    assert padded.shape == (6, n + 6, n + 6)


def test_pad_halo_width_4_rejected():
    """halo>3 not supported per halo.py:524."""
    n = 8
    data = jnp.ones((6, n, n))
    with pytest.raises(NotImplementedError, match="halo=1, halo=2, and halo=3"):
        pad_halo(data, halo=4)


def test_pad_halo_interior_preserved():
    """Pad operation does not mutate interior pixels."""
    n = 8
    rng = np.random.default_rng(seed=891)
    data = jnp.asarray(rng.uniform(size=(6, n, n)))
    for halo in (1, 2, 3):
        padded = pad_halo(data, halo=halo)
        # Interior at offset [halo:n+halo, halo:n+halo] = original.
        interior = padded[:, halo:n + halo, halo:n + halo]
        np.testing.assert_array_equal(np.asarray(interior),
                                       np.asarray(data))


def test_pad_halo_finite_at_corners():
    """Cube-vertex halo cells are finite (not NaN) for halo=2 and 3."""
    n = 8
    rng = np.random.default_rng(seed=891)
    data = jnp.asarray(rng.uniform(size=(6, n, n)))
    for halo in (2, 3):
        padded = pad_halo(data, halo=halo)
        assert jnp.all(jnp.isfinite(padded)), \
            f"halo={halo}: cube-vertex cells contain NaN/Inf"


def test_pad_halo_width_2_face_neighbour_match_at_edges():
    """Sanity check: halo=2 west strip should reflect neighbour face
    data (not zeros / not all the same value).  This is a non-trivial
    smoke test that the wider-halo pad activates the cross-panel
    copying logic."""
    n = 8
    # Construct a face-distinct field: face f gets constant value f+1.
    data = jnp.broadcast_to(
        jnp.arange(1, 7, dtype=jnp.float64)[:, None, None],
        (6, n, n),
    )
    padded = pad_halo(data, halo=2)
    # West halo strip of face 0 should contain values from face 0's
    # west neighbour (which is NOT face 0's value of 1.0).
    west_strip = padded[0, 0, 2:n + 2]      # (n,) west halo cells of face 0
    # If west strip is all 1.0 (face 0's own value), copying did nothing.
    assert not jnp.all(west_strip == 1.0), \
        "halo=2 west strip of face 0 contains only own-face values; " \
        "cross-panel copy may not be active for halo=2."
