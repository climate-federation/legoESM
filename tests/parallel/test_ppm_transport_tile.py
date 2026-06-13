"""Cube tiled np>6 stage (task #3) — PPM transport tiling tests.

U1: ``_ppm_transport_1d(..., rd_prepadded=True)`` is BIT-IDENTICAL to the
default internal edge-pad when fed the same padded rdelta — the hook that
lets a sub-face TILE supply a REAL depth-1 neighbour-tile rdelta halo at
interior cuts (the internal edge-pad is wrong there: the upwind CFL cell
lives in the neighbour tile).  See docs/scaling/cube_transport_tiling_design.md.

(U2 will add the tile-vs-global PPM flux parity here.)
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import _ppm_transport_1d


def _bit_identical_for_axis(axis: int) -> None:
    rng = np.random.default_rng(0)
    eh = 2          # external (cross-face) halo, as d_sw3 supplies
    n, m = 8, 4     # interior cells along sweep axis, trailing dim
    if axis == 1:
        field = jnp.asarray(rng.standard_normal((6, n + 2 * eh, m)))
        courant = jnp.asarray(rng.standard_normal((6, n + 1, m)))
        rdelta = jnp.asarray(np.abs(rng.standard_normal((6, n, m))) + 0.1)
    else:
        field = jnp.asarray(rng.standard_normal((6, m, n + 2 * eh)))
        courant = jnp.asarray(rng.standard_normal((6, m, n + 1)))
        rdelta = jnp.asarray(np.abs(rng.standard_normal((6, m, n))) + 0.1)

    flux_def = _ppm_transport_1d(field, courant, rdelta, axis, external_halo=eh)

    # Replicate the function's internal rd edge-pad (done in its axis-1
    # working frame), then map back to the caller's axis so rd_prepadded
    # receives the identical rd_pad.
    rd_w = rdelta if axis == 1 else jnp.swapaxes(rdelta, 1, 2)
    rd_pad_w = jnp.pad(rd_w, [(0, 0), (1, 1), (0, 0)], mode="edge")
    rd_pad_caller = rd_pad_w if axis == 1 else jnp.swapaxes(rd_pad_w, 1, 2)

    flux_pp = _ppm_transport_1d(
        field, courant, rd_pad_caller, axis, external_halo=eh,
        rd_prepadded=True)

    np.testing.assert_array_equal(
        np.asarray(flux_def), np.asarray(flux_pp),
        err_msg=f"rd_prepadded flux != default (axis={axis})")


def test_rd_prepadded_bit_identical_axis1():
    _bit_identical_for_axis(1)


def test_rd_prepadded_bit_identical_axis2():
    _bit_identical_for_axis(2)


def test_rd_prepadded_wrong_length_raises():
    """codex U1 MEDIUM: a mis-sized prepadded rdelta must fail loudly, not
    silently truncate.  nn=8 ⇒ rd must be length nn+2=10; pass 9."""
    import pytest

    eh, n, m = 2, 8, 4
    field = jnp.zeros((6, n + 2 * eh, m))
    courant = jnp.zeros((6, n + 1, m))
    bad_rd = jnp.ones((6, n + 1, m))   # length nn+1=9, not nn+2=10
    with pytest.raises(ValueError, match="rd_prepadded expects"):
        _ppm_transport_1d(field, courant, bad_rd, 1, external_halo=eh,
                          rd_prepadded=True)
