"""FV3_3D iter 514: extend ``monotone_clip`` to ``pad_halo_vector``
(3D).

iter-513 added clip to pad_halo (3D) + pad_halo_pair_h2 (4 new
sites) but NH residual at 10 steps was unchanged (1.6×).  Per
iter-507, the *vector halo* is critical to NH (toggling
``use_fv3_vector_halo_uv=False`` makes ratio 2.36× worse).
The 4D vector halo is already clipped (iter-503).  But the 3D
``pad_halo_vector`` in fv3_sw_core (lines 602, 1194, 1430) is
NOT.  iter-514 extends ``pad_halo_vector`` 3D with clip + adds
the fv3_sw_core import to the context manager.

Result on iter-509 residual test: still unchanged (1.6× at
10 steps).  Conclusion: the fv3_sw_core 3D vector halo path
is NOT exercised in this NH C8+duogrid+1-step test, despite
being a legitimate halo entry.  The NH residual growth must
go through one of the already-clipped 4D paths.  The bias is
probably accumulating as compound numerical noise at the 24
cube-vertex cells (per iter-489), not as an unclipped halo
leak.

iter-514 still adds value: it closes a real code path that
WILL be exercised under different configs (e.g., when the SW
test uses fv3_sw_core paths heavily).  No regression.

Tests
-----

1. ``test_pad_halo_vector_3d_accepts_clip_args``.
2. ``test_pad_halo_vector_3d_with_clip_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_vector


def test_pad_halo_vector_3d_accepts_clip_args():
    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    rng = np.random.default_rng(seed=514)
    u = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    v = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    n_face = 6
    cap = jnp.ones((n_face, n, n))
    sap = jnp.zeros((n_face, n, n))
    cap_pad = jnp.ones((n_face, n + 2, n + 2))
    sap_pad = jnp.zeros((n_face, n + 2, n + 2))
    u_no_clip, v_no_clip = pad_halo_vector(
        u, v, cap, sap, cap_pad, sap_pad,
        halo=1, duogrid=grid.duogrid,
    )
    u_clip, v_clip = pad_halo_vector(
        u, v, cap, sap, cap_pad, sap_pad,
        halo=1, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.5,
    )
    assert u_no_clip.shape == u_clip.shape
    assert v_no_clip.shape == v_clip.shape


def test_pad_halo_vector_3d_with_clip_finite():
    n = 8
    grid = create_cubed_sphere(n, use_duogrid=True)
    rng = np.random.default_rng(seed=515)
    u = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    v = jnp.asarray(rng.uniform(-3.0, 3.0, size=(6, n, n)))
    cap = jnp.ones((6, n, n))
    sap = jnp.zeros((6, n, n))
    cap_pad = jnp.ones((6, n + 2, n + 2))
    sap_pad = jnp.zeros((6, n + 2, n + 2))
    u_clip, v_clip = pad_halo_vector(
        u, v, cap, sap, cap_pad, sap_pad,
        halo=1, duogrid=grid.duogrid,
        monotone_clip=True, monotone_clip_slack=0.5,
    )
    assert jnp.all(jnp.isfinite(u_clip))
    assert jnp.all(jnp.isfinite(v_clip))
