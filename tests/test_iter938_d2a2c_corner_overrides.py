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


def test_iter938_d2a2c_default_off_bit_identical():
    """`_d2a2c_vect(...)` with default `apply_fortran_corner_overrides
    =False` must produce bit-identical output to the same call without
    the kwarg.  This guards the default-off contract.
    """
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState,
    )
    from legoesm.core.fv3_sw_core import _d2a2c_vect
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2,
    )

    n = 16
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    out_default = _d2a2c_vect(u_d, v_d, cdgrid)
    out_off_explicit = _d2a2c_vect(
        u_d, v_d, cdgrid, apply_fortran_corner_overrides=False)
    for a, b, name in zip(out_default, out_off_explicit,
                            ("ua", "va", "uc", "vc", "ut", "vt")):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"{name} differs at default-off",
        )


def test_iter938_d2a2c_flag_on_currently_no_op_on_uc_vc():
    """KNOWN INCOMPLETE PORT (iter-938).

    The utmp/vtmp halo overrides applied by `_apply_fortran_d2a2c_corner_
    overrides` write to halo cells at j=1 (Python padded south halo) and
    j=n+2 (north halo), but our Python `_d2a2c_vect` downstream stencils
    (ua/va computation step 3, uc/vc edge_interpolate4 step 4a/b) only
    read padded j in [2, n+1] (interior).  The overridden halo cells are
    therefore CURRENTLY NO-OP on uc/vc/ua/va output.

    To make the override effective, an additional port is required:
    Fortran sw_core.F90:3567-3582 also overrides ua/va halo cells with
    sign-flipped cross-component values; THOSE values feed into the
    edge_interpolate4 step that propagates to uc/vc.  iter-938 deferred
    that second port to a follow-up iter (iter-939+).

    This test pins the CURRENT no-op state so any future port that
    propagates the override into uc/vc is detected (and the editor
    must update this test alongside the propagation work).
    """
    from legoesm.core.fv3_sw_core import _d2a2c_vect
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2,
    )

    n = 16
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    ua_off, va_off, uc_off, vc_off, ut_off, vt_off = _d2a2c_vect(
        u_d, v_d, cdgrid, apply_fortran_corner_overrides=False)
    ua_on, va_on, uc_on, vc_on, ut_on, vt_on = _d2a2c_vect(
        u_d, v_d, cdgrid, apply_fortran_corner_overrides=True)

    # Pin the CURRENT no-op state.  Once iter-939+ adds the ua/va
    # halo-override propagation, these assertions will fail and the
    # editor must update this test alongside the propagation work.
    np.testing.assert_array_equal(np.asarray(ua_on), np.asarray(ua_off))
    np.testing.assert_array_equal(np.asarray(va_on), np.asarray(va_off))
    np.testing.assert_array_equal(np.asarray(uc_on), np.asarray(uc_off))
    np.testing.assert_array_equal(np.asarray(vc_on), np.asarray(vc_off))
    np.testing.assert_array_equal(np.asarray(ut_on), np.asarray(ut_off))
    np.testing.assert_array_equal(np.asarray(vt_on), np.asarray(vt_off))
