"""FV3_3D iter 476: test ``pad_halo_vector_4d`` with duogrid
on a zero vector field — does the vector halo introduce
spurious non-zero values?

iter-475 showed scalar halo with duogrid is mostly clean (5e-
13 on constant, 3.5% overshoot on linear).  Next candidate
from iter-473: vector halo (``pad_halo_vector_4d``, used for
u/v cell→corner interpolation).

A zero vector field (u=0, v=0) under proper rotation should
remain zero in the halo (zero is a fixed point of rotation).

Tests
-----

1. ``test_pad_halo_vector_4d_no_duogrid_zero``.
2. ``test_pad_halo_vector_4d_duogrid_zero``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_vector_4d


def test_pad_halo_vector_4d_no_duogrid_zero():
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=False)
    u = jnp.zeros((6, n, n, nlev))
    v = jnp.zeros((6, n, n, nlev))
    u_padded, v_padded = pad_halo_vector_4d(
        u, v,
        cos_angle=grid.cos_angle,
        sin_angle=grid.sin_angle,
        cos_angle_padded=grid.cos_angle_padded,
        sin_angle_padded=grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
        halo=1, duogrid=None,
    )
    np.testing.assert_allclose(
        np.asarray(u_padded), 0.0, rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_allclose(
        np.asarray(v_padded), 0.0, rtol=1e-13, atol=1e-13,
    )


def test_pad_halo_vector_4d_duogrid_zero(capsys):
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    u = jnp.zeros((6, n, n, nlev))
    v = jnp.zeros((6, n, n, nlev))
    u_padded, v_padded = pad_halo_vector_4d(
        u, v,
        cos_angle=grid.cos_angle,
        sin_angle=grid.sin_angle,
        cos_angle_padded=grid.cos_angle_padded,
        sin_angle_padded=grid.sin_angle_padded,
        interp_offsets=None,
        halo=1, duogrid=grid.duogrid,
    )
    u_max = float(np.max(np.abs(np.asarray(u_padded))))
    v_max = float(np.max(np.abs(np.asarray(v_padded))))
    with capsys.disabled():
        print(
            f"\n[iter-476 vector halo duogrid zero field]"
            f"\n  max |u_padded|: {u_max:.4e}"
            f"\n  max |v_padded|: {v_max:.4e}"
        )
    np.testing.assert_allclose(
        np.asarray(u_padded), 0.0, rtol=1e-13, atol=1e-13,
        err_msg=(
            f"pad_halo_vector_4d (duogrid) introduced spurious "
            f"u_max = {u_max:.4e} on a zero vector field — BUG."
        ),
    )
    np.testing.assert_allclose(
        np.asarray(v_padded), 0.0, rtol=1e-13, atol=1e-13,
    )
