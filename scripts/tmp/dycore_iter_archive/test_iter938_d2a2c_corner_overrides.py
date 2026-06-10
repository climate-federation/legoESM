"""Iter-938 sentinels for `_apply_fortran_d2a2c_corner_overrides`.

The helper ports Fortran sw_core.F90:3527-3545 (utmp x-direction)
and 3620-3639 (vtmp y-direction) cube-corner sign-flip overrides.
These tests verify the index map and sign convention against the
Fortran specification on synthetic inputs.

Each Fortran override has the form:

    SW: utmp(i,0)   = -vtmp(0, 1-i)        i in [-2, 0]
    SE: utmp(npx+i,0) = +vtmp(npx, i+1)    i in [0, 2]
    NE: utmp(npx+i,npy) = -vtmp(npx, npy-1-i)  i in [0, 2]
    NW: utmp(i,npy) = +vtmp(0, npy-1+i+1)  i in [-2, 0]

We have halo=2 (Python `pad_halo_vector` output shape (n+4, n+4))
so we port the 2 deepest cells per corner; the third (Fortran's
i=±2 cell) is out of our halo reach.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import _apply_fortran_d2a2c_corner_overrides


def _make_distinct_pads(n: int):
    """Build utmp_pad, vtmp_pad with a distinct value at every cell so the
    test can read back the exact value the override copied from."""
    utmp = jnp.arange(6 * (n + 4) * (n + 4), dtype=jnp.float64).reshape(
        6, n + 4, n + 4)
    vtmp = jnp.arange(
        6 * (n + 4) * (n + 4),
        2 * 6 * (n + 4) * (n + 4),
        dtype=jnp.float64,
    ).reshape(6, n + 4, n + 4)
    return utmp, vtmp


def test_iter938_sw_corner_utmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # SW utmp x-dir overrides (Fortran 3527):
    #   utmp(-1, 0) = -vtmp(0, 2) → utmp_pad[:,0,1] = -vtmp_pad[:,1,3]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, 0, 1]),
        -np.asarray(vtmp[:, 1, 3]),
    )
    #   utmp(0, 0)  = -vtmp(0, 1) → utmp_pad[:,1,1] = -vtmp_pad[:,1,2]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, 1, 1]),
        -np.asarray(vtmp[:, 1, 2]),
    )


def test_iter938_se_corner_utmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # SE utmp x-dir (Fortran 3532):
    #   utmp(npx, 0) = +vtmp(npx, 1) → utmp_pad[:,n+2,1] = +vtmp_pad[:,n+2,2]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, n + 2, 1]),
        +np.asarray(vtmp[:, n + 2, 2]),
    )
    #   utmp(npx+1, 0) = +vtmp(npx, 2) → utmp_pad[:,n+3,1] = +vtmp_pad[:,n+2,3]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, n + 3, 1]),
        +np.asarray(vtmp[:, n + 2, 3]),
    )


def test_iter938_ne_corner_utmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # NE utmp x-dir (Fortran 3537):
    #   utmp(npx, npy)   = -vtmp(npx, npy-1) → utmp_pad[:,n+2,n+2] = -vtmp_pad[:,n+2,n+1]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, n + 2, n + 2]),
        -np.asarray(vtmp[:, n + 2, n + 1]),
    )
    #   utmp(npx+1, npy) = -vtmp(npx, npy-2) → utmp_pad[:,n+3,n+2] = -vtmp_pad[:,n+2,n]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, n + 3, n + 2]),
        -np.asarray(vtmp[:, n + 2, n]),
    )


def test_iter938_nw_corner_utmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # NW utmp x-dir (Fortran 3543):
    #   utmp(-1, npy) = +vtmp(0, npy-2) → utmp_pad[:,0,n+2] = +vtmp_pad[:,1,n]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, 0, n + 2]),
        +np.asarray(vtmp[:, 1, n]),
    )
    #   utmp(0, npy)  = +vtmp(0, npy-1) → utmp_pad[:,1,n+2] = +vtmp_pad[:,1,n+1]
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, 1, n + 2]),
        +np.asarray(vtmp[:, 1, n + 1]),
    )


def test_iter938_sw_corner_vtmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # SW vtmp y-dir (Fortran 3620):
    # vtmp READS from utmp INTERIOR cells (not modified by utmp overrides).
    #   vtmp(0, -1) = -utmp(2, 0) → vtmp_pad[:,1,0] = -utmp_pad[:,3,1]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, 1, 0]),
        -np.asarray(utmp[:, 3, 1]),
    )
    #   vtmp(0, 0)  = -utmp(1, 0) → vtmp_pad[:,1,1] = -utmp_pad[:,2,1]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, 1, 1]),
        -np.asarray(utmp[:, 2, 1]),
    )


def test_iter938_nw_corner_vtmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # NW vtmp y-dir (Fortran 3625):
    #   vtmp(0, npy)   = +utmp(1, npy) → vtmp_pad[:,1,n+2] = +utmp_pad[:,2,n+2]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, 1, n + 2]),
        +np.asarray(utmp[:, 2, n + 2]),
    )
    #   vtmp(0, npy+1) = +utmp(2, npy) → vtmp_pad[:,1,n+3] = +utmp_pad[:,3,n+2]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, 1, n + 3]),
        +np.asarray(utmp[:, 3, n + 2]),
    )


def test_iter938_se_corner_vtmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # SE vtmp y-dir (Fortran 3630):
    #   vtmp(npx, -1) = +utmp(npx-2, 0) → vtmp_pad[:,n+2,0] = +utmp_pad[:,n,1]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, n + 2, 0]),
        +np.asarray(utmp[:, n, 1]),
    )
    #   vtmp(npx, 0)  = +utmp(npx-1, 0) → vtmp_pad[:,n+2,1] = +utmp_pad[:,n+1,1]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, n + 2, 1]),
        +np.asarray(utmp[:, n + 1, 1]),
    )


def test_iter938_ne_corner_vtmp_overrides():
    n = 8
    utmp, vtmp = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(utmp, vtmp, n)

    # NE vtmp y-dir (Fortran 3635):
    #   vtmp(npx, npy)   = -utmp(npx-1, npy) → vtmp_pad[:,n+2,n+2] = -utmp_pad[:,n+1,n+2]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, n + 2, n + 2]),
        -np.asarray(utmp[:, n + 1, n + 2]),
    )
    #   vtmp(npx, npy+1) = -utmp(npx-2, npy) → vtmp_pad[:,n+2,n+3] = -utmp_pad[:,n,n+2]
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, n + 2, n + 3]),
        -np.asarray(utmp[:, n, n + 2]),
    )


def test_iter938_helper_is_pure_function():
    """The helper `_apply_fortran_d2a2c_corner_overrides(utmp_pad,
    vtmp_pad, n)` is a STANDALONE module-level function (not wired
    into `d2a2c_vect`).  This test confirms its public contract:
    inputs unchanged at non-corner cells, outputs match Fortran at
    the 8 corner positions per axis.

    Codex iter-938 stop-time finding: wiring this helper into
    `d2a2c_vect` would be output-dead because the downstream
    edge_interpolate4 j-slicing reads only interior j ∈ [2, n+1]
    while the corner overrides write to padded j=1 and j=n+2.
    iter-938b removed the dead wiring; the helper remains as a
    documented Fortran-arithmetic reference for iter-939+ to wire in
    after the edge_interpolate4 j-slice refactor lands.
    """
    n = 8
    utmp_in, vtmp_in = _make_distinct_pads(n)
    utmp_o, vtmp_o = _apply_fortran_d2a2c_corner_overrides(
        utmp_in, vtmp_in, n)

    # Cell at (i=2, j=2) is FAR from all corners — must be unchanged.
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, 2, 2]),
        np.asarray(utmp_in[:, 2, 2]),
    )
    np.testing.assert_array_equal(
        np.asarray(vtmp_o[:, 2, 2]),
        np.asarray(vtmp_in[:, 2, 2]),
    )
    # Centre interior cell at (n//2 + 2, n//2 + 2) — must be unchanged.
    cmid = n // 2 + 2
    np.testing.assert_array_equal(
        np.asarray(utmp_o[:, cmid, cmid]),
        np.asarray(utmp_in[:, cmid, cmid]),
    )
