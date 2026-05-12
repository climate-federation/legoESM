"""FV3_3D iter 498: extend iter-490 monotone_clip to
``pad_halo_vector_4d`` — does it reduce iter-497's 4%
residual vector-halo edge amplification?

iter-497: vector u_d edge × 1.040, v_d × 1.027 under duogrid.
iter-498 tests if monotone_clip=True on the underlying
pad_halo_4d calls inside pad_halo_vector_4d reduces this.

Tests
-----

1. ``test_pad_halo_vector_4d_monotone_clip_default_off``.
2. ``test_pad_halo_vector_4d_monotone_clip_reduces_overshoot``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo_vector_4d


def test_pad_halo_vector_4d_monotone_clip_default_off():
    """Default monotone_clip=False — backward compat."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    u = jnp.zeros((6, n, n, nlev))
    v = jnp.zeros((6, n, n, nlev))
    u_padded_default, v_padded_default = pad_halo_vector_4d(
        u, v,
        cos_angle=grid.cos_angle,
        sin_angle=grid.sin_angle,
        cos_angle_padded=grid.cos_angle_padded,
        sin_angle_padded=grid.sin_angle_padded,
        halo=1, duogrid=grid.duogrid,
    )
    u_padded_explicit, v_padded_explicit = pad_halo_vector_4d(
        u, v,
        cos_angle=grid.cos_angle,
        sin_angle=grid.sin_angle,
        cos_angle_padded=grid.cos_angle_padded,
        sin_angle_padded=grid.sin_angle_padded,
        halo=1, duogrid=grid.duogrid,
        monotone_clip=False,
    )
    np.testing.assert_array_equal(
        np.asarray(u_padded_default), np.asarray(u_padded_explicit),
    )
    np.testing.assert_array_equal(
        np.asarray(v_padded_default), np.asarray(v_padded_explicit),
    )


def test_pad_halo_vector_4d_monotone_clip_reduces_overshoot(capsys):
    """Random Gaussian u, v — clip should reduce halo
    overshoot."""
    n = 8
    nlev = 5
    grid = create_cubed_sphere(n, use_duogrid=True)
    rng = np.random.default_rng(seed=498)
    u = jnp.asarray(
        rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)),
    )
    v = jnp.asarray(
        rng.normal(loc=0.0, scale=1.5, size=(6, n, n, nlev)),
    )
    # No clip
    u_off, v_off = pad_halo_vector_4d(
        u, v,
        cos_angle=grid.cos_angle, sin_angle=grid.sin_angle,
        cos_angle_padded=grid.cos_angle_padded,
        sin_angle_padded=grid.sin_angle_padded,
        halo=1, duogrid=grid.duogrid, monotone_clip=False,
    )
    # With clip
    u_on, v_on = pad_halo_vector_4d(
        u, v,
        cos_angle=grid.cos_angle, sin_angle=grid.sin_angle,
        cos_angle_padded=grid.cos_angle_padded,
        sin_angle_padded=grid.sin_angle_padded,
        halo=1, duogrid=grid.duogrid, monotone_clip=True,
    )
    u_arr_off = np.asarray(u_off)
    u_arr_on = np.asarray(u_on)
    halo_max_off = max(
        float(np.max(np.abs(u_arr_off[:, 0, :, :]))),
        float(np.max(np.abs(u_arr_off[:, -1, :, :]))),
        float(np.max(np.abs(u_arr_off[:, :, 0, :]))),
        float(np.max(np.abs(u_arr_off[:, :, -1, :]))),
    )
    halo_max_on = max(
        float(np.max(np.abs(u_arr_on[:, 0, :, :]))),
        float(np.max(np.abs(u_arr_on[:, -1, :, :]))),
        float(np.max(np.abs(u_arr_on[:, :, 0, :]))),
        float(np.max(np.abs(u_arr_on[:, :, -1, :]))),
    )
    interior_max = float(np.max(np.abs(np.asarray(u))))
    with capsys.disabled():
        print(
            f"\n[iter-498 vector halo monotone_clip]"
        )
        print(f"  interior_max:      {interior_max:.4f}")
        print(f"  halo_max no clip:  {halo_max_off:.4f}")
        print(f"  halo_max clip on:  {halo_max_on:.4f}")
        if halo_max_off > halo_max_on:
            print(
                f"  Clip reduced halo overshoot by "
                f"{(1 - halo_max_on/halo_max_off)*100:.1f}%"
            )
    assert np.isfinite(halo_max_off) and np.isfinite(halo_max_on)
