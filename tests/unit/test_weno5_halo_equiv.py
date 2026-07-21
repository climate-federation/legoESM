"""Halo-aware WENO5 advection must match the serial implementation
bit-for-bit on single rank (R6).

Convention: `layout.n_ranks == 1` + `jnp.pad(mode='wrap')` produces
the same neighbour values as the serial `jnp.roll(±k)` stencils.
Verify that the slice-based halo operator returns numerically
identical tendencies for a random field + co-located velocity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    _weno5_advection_x, _weno5_advection_y,
)
from legoesm.atmosphere.dynamics.les.plane_operators_halo import (
    weno5_advection_x_halo, weno5_advection_y_halo,
)

jax.config.update("jax_enable_x64", True)


def _pad_periodic(arr, halo):
    """Wrap-pad the leading 2 axes by ``halo`` cells."""
    pad_widths = [(halo, halo), (halo, halo)] + [(0, 0)] * (arr.ndim - 2)
    return jnp.pad(arr, pad_widths, mode="wrap")


@pytest.mark.parametrize("halo", [3, 4])
def test_weno5_x_halo_matches_serial(halo):
    rng = jax.random.PRNGKey(0)
    ny, nx, nlev = 8, 16, 4
    f = jax.random.normal(rng, (ny, nx, nlev), dtype=jnp.float64)
    u = 0.5 * jax.random.normal(
        jax.random.fold_in(rng, 1), (ny, nx, nlev), dtype=jnp.float64,
    )
    dx = 1000.0

    expected = _weno5_advection_x(f, u, dx)

    f_pad = _pad_periodic(f, halo)
    u_pad = _pad_periodic(u, halo)
    actual = weno5_advection_x_halo(f_pad, u_pad, dx, halo=halo)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected),
        rtol=1e-12, atol=1e-12,
        err_msg=f"WENO5-x halo (halo={halo}) deviates from serial",
    )


@pytest.mark.parametrize("halo", [3, 4])
def test_weno5_y_halo_matches_serial(halo):
    rng = jax.random.PRNGKey(7)
    ny, nx, nlev = 12, 8, 3
    f = jax.random.normal(rng, (ny, nx, nlev), dtype=jnp.float64)
    v = 0.5 * jax.random.normal(
        jax.random.fold_in(rng, 1), (ny, nx, nlev), dtype=jnp.float64,
    )
    dy = 750.0

    expected = _weno5_advection_y(f, v, dy)

    f_pad = _pad_periodic(f, halo)
    v_pad = _pad_periodic(v, halo)
    actual = weno5_advection_y_halo(f_pad, v_pad, dy, halo=halo)

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected),
        rtol=1e-12, atol=1e-12,
    )


def test_weno5_x_halo_rejects_halo_lt_3():
    rng = jax.random.PRNGKey(2)
    f = jax.random.normal(rng, (4, 6, 2), dtype=jnp.float64)
    u = jnp.zeros_like(f)
    f_pad = _pad_periodic(f, 2)
    u_pad = _pad_periodic(u, 2)
    with pytest.raises(ValueError, match="halo>=3"):
        weno5_advection_x_halo(f_pad, u_pad, 1000.0, halo=2)


def test_weno5_y_halo_rejects_halo_lt_3():
    rng = jax.random.PRNGKey(3)
    f = jax.random.normal(rng, (4, 6, 2), dtype=jnp.float64)
    v = jnp.zeros_like(f)
    f_pad = _pad_periodic(f, 1)
    v_pad = _pad_periodic(v, 1)
    with pytest.raises(ValueError, match="halo>=3"):
        weno5_advection_y_halo(f_pad, v_pad, 1000.0, halo=1)
