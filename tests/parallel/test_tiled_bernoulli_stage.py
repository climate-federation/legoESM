"""COMPOSED 3D-PE Bernoulli stage (B = KE + Phi) sub-face-tiled.

Composes two cc-local ops (dgrid_to_center_vector D->cc box-avg +
compute_geopotential[_hybrid] per-column integration) into ONE shard_map: the cc
tile is sliced ONCE, both ops are cc-local (no halo), so B/u_cc/v_cc are
bit-identical (EXACT) to the global composition sliced.  Covers BOTH sigma and
hybrid coordinates, host-composition + np24.

24 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_cdgrid import dgrid_to_center_vector
from legoesm.grids.vertical import (
    create_sigma_coordinate, compute_geopotential,
    make_hybrid_levels, compute_geopotential_hybrid,
)
from legoesm.parallel.tiled_production_cdgrid import make_tiled_bernoulli_stage_2d


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(10.0 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(10.0 * rng.standard_normal((6, n + 1, n + 1, nlev)))
    T = jnp.asarray(250.0 + 30.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    phis = jnp.asarray(1.0e3 * rng.standard_normal((6, n, n)))
    return u_d, v_d, T, p_s, phis


def _global_B(u_d, v_d, T, p_s, phis, coord, hybrid):
    u_cc, v_cc = dgrid_to_center_vector(u_d, v_d)
    geo = compute_geopotential_hybrid if hybrid else compute_geopotential
    Phi = geo(T, p_s, coord, phis)
    B = 0.5 * (u_cc ** 2 + v_cc ** 2) + Phi
    return [np.asarray(x) for x in (B, u_cc, v_cc)]


def _reassemble_cc(get_tile, kt):
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


@pytest.mark.parametrize("hybrid", [False, True])
def test_bernoulli_stage_np24(hybrid):
    kt, nl, nlev = 2, 6, 8
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    coord = make_hybrid_levels(nlev) if hybrid else create_sigma_coordinate(nlev)
    u_d, v_d, T, p_s, phis = _inputs(n, nlev, 41 if hybrid else 42)
    B_g, u_g, v_g = _global_B(u_d, v_d, T, p_s, phis, coord, hybrid)

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_bernoulli_stage_2d(mesh, coord, n, kt)
    B_s, u_s, v_s = stage(u_d, v_d, T, p_s, phis)

    def _re(arr):
        a = np.asarray(arr).reshape(6, kt, nl, kt, nl, nlev)
        return _reassemble_cc(lambda ti, tj: a[:, ti, :, tj, :, :], kt)

    np.testing.assert_array_equal(_re(B_s), B_g, err_msg="composed B != global")
    np.testing.assert_array_equal(_re(u_s), u_g, err_msg="composed u_cc != global")
    np.testing.assert_array_equal(_re(v_s), v_g, err_msg="composed v_cc != global")


@pytest.mark.parametrize("hybrid", [False, True])
def test_bernoulli_compose_host_body(hybrid):
    """Composition exactness WITHOUT 24 devices (slice-once chain on host)."""
    kt, nl, nlev = 3, 6, 8
    n = kt * nl
    coord = make_hybrid_levels(nlev) if hybrid else create_sigma_coordinate(nlev)
    u_d, v_d, T, p_s, phis = _inputs(n, nlev, 43 if hybrid else 44)
    B_g, u_g, v_g = _global_B(u_d, v_d, T, p_s, phis, coord, hybrid)
    geo = compute_geopotential_hybrid if hybrid else compute_geopotential

    def _chain(ti, tj, which):
        a_i, a_j = ti * nl, tj * nl
        u_cc, v_cc = dgrid_to_center_vector(
            u_d[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1],
            v_d[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1])
        if which == "u":
            return u_cc
        if which == "v":
            return v_cc
        Phi = geo(T[:, a_i:a_i + nl, a_j:a_j + nl],
                  p_s[:, a_i:a_i + nl, a_j:a_j + nl], coord,
                  phis[:, a_i:a_i + nl, a_j:a_j + nl])
        return 0.5 * (u_cc ** 2 + v_cc ** 2) + Phi

    for which, g in (("B", B_g), ("u", u_g), ("v", v_g)):
        t = _reassemble_cc(lambda ti, tj: _chain(ti, tj, which), kt)
        np.testing.assert_array_equal(
            t, g, err_msg=f"composed {which} host-body != global")
