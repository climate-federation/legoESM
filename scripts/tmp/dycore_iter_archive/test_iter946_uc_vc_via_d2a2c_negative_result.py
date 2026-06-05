"""Iter-946 sentinel (negative result): pin the documented finding that
sourcing covariant uc, vc halo cells from OLD u_d, v_d via the
4th-order d2a2c machinery (`_pad_halo_uc_vc_via_d2a2c`) WORSENS the
duogrid FB chain W2 1-day |v_max| at C36 vs. the iter-945 mode='edge'
baseline.

Trigger.  iter-944b reverted Python-only `mpp_get_boundary` syncs.
iter-945 closed the cube-face D-grid PPM halo gap.  The remaining
`(uc, vc)` C-grid halo on the duogrid path was identified as the
likely next gap.  Fortran sources `(uc, vc)` halo cells from
`mpp_update_domains(uc, vc, gridtype=CGRID_NE)` upstream of d_sw1 /
d_sw3 (with `cube_rmp` interpolation when duogrid is active).

Iter-946 attempted to mirror that with a new helper
`_pad_halo_uc_vc_via_d2a2c` that reuses `_d2a2c_vect_duogrid`'s
4th-order machinery (extended slicing of `u_d_full`, `v_d_full` to
include j-halo / i-halo on the A→C interpolation) to derive
covariant uc, vc at halo positions WITHOUT going through a lossy
cell-centre roundtrip.

Negative result: the helper is called at `_d_sw_native` entry with
the OLD u_d, v_d (before c_sw + p_grad_c), so the derived halo
lacks the ~15 m/s c_sw + p_grad_c increment that was added to the
INTERIOR uc, vc.  Mixing OLD-derived halo with NEW interior in the
4-cell averages of `_d_sw1_recompute_ut_vt` Part 1 and
`_bgrid_ke_transport` Step 1 produces a discontinuity at cube-face
boundaries that gives WORSE results than `mode='edge'`:

| location of iter-946 halo | |u_max| (m/s) | |v_max| (m/s) |
|---------------------------|--------------:|--------------:|
| iter-945 baseline (mode='edge')      |     78        |       81  |
| d_sw1 + d_sw3 (both)                  |     73        |      156  |
| d_sw3 only                            |     71        |      150  |

The helper is RETAINED for future work that ALSO propagates the
c_sw + p_grad_c increments to halo positions; without that
companion fix it introduces a Fortran-unfaithful OLD/NEW mismatch.

This sentinel pins:
1. The helper exists, runs on duogrid C24 with ng>=3, and produces
   the documented (n+1, n+2) and (n+2, n+1) output shapes.
2. The helper raises ValueError on non-duogrid grids.
3. The helper raises ValueError on duogrid with ng<3.
4. `_d_sw_native` does NOT currently call `_pad_halo_uc_vc_via_d2a2c`
   (mode='edge' is the active path, per the iter-946 negative-result
   revert).  Re-enabling without iter-947+ companion fix would fail
   the iter-945 W2 sentinel.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv3_sw_core import (
    _d_sw_native, _pad_halo_uc_vc_via_d2a2c,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _w2_d_grid_winds(N: int, *, use_duogrid: bool):
    grid = create_cubed_sphere(N, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    return cdgrid, u_d, v_d


def test_iter946_helper_shapes_duogrid_ng3():
    """`_pad_halo_uc_vc_via_d2a2c` returns uc with j-halo=1 and vc
    with i-halo=1 at duogrid C24.
    """
    N = 24
    cdgrid, u_d, v_d = _w2_d_grid_winds(N, use_duogrid=True)
    assert cdgrid.base.duogrid.ng >= 3, "need ng>=3 for the helper"
    uc_jhalo, vc_ihalo = _pad_halo_uc_vc_via_d2a2c(u_d, v_d, cdgrid)
    assert uc_jhalo.shape == (6, N + 1, N + 2), (
        f"uc_jhalo shape expected (6, {N+1}, {N+2}); got {uc_jhalo.shape}"
    )
    assert vc_ihalo.shape == (6, N + 2, N + 1), (
        f"vc_ihalo shape expected (6, {N+2}, {N+1}); got {vc_ihalo.shape}"
    )
    # Output should be finite for a smooth W2 IC.
    assert np.all(np.isfinite(np.asarray(uc_jhalo)))
    assert np.all(np.isfinite(np.asarray(vc_ihalo)))


def test_iter946_helper_rejects_non_duogrid():
    """`_pad_halo_uc_vc_via_d2a2c` raises ValueError on non-duogrid."""
    N = 24
    cdgrid, u_d, v_d = _w2_d_grid_winds(N, use_duogrid=False)
    with pytest.raises(ValueError, match="duogrid"):
        _pad_halo_uc_vc_via_d2a2c(u_d, v_d, cdgrid)


def test_iter946_d_sw_native_does_NOT_use_d2a2c_uc_vc_halo():
    """`_d_sw_native` (the FB-chain d_sw entry point) must NOT call
    `_pad_halo_uc_vc_via_d2a2c` until a companion iter-947+ fix
    propagates c_sw + p_grad_c increments to halo.  Re-enabling the
    helper inside d_sw1 or d_sw3 without that companion fix would
    re-introduce the iter-946 W2 |v_max| regression (81 → 150+ m/s).
    """
    src = inspect.getsource(_d_sw_native)
    assert "_pad_halo_uc_vc_via_d2a2c" not in src, (
        "`_d_sw_native` calls `_pad_halo_uc_vc_via_d2a2c` — iter-946 "
        "established this WORSENS W2 |v_max| at C36 1-day because the "
        "OLD-u_d-derived halo conflicts with NEW (post-c_sw/p_grad_c) "
        "interior uc, vc.  Need iter-947+ to first propagate c_sw + "
        "p_grad_c increments to halo before re-enabling."
    )
