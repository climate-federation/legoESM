"""3D PE geopotential (Simmons-Burridge) sub-face-tiled.

compute_geopotential is a per-COLUMN vertical integration (cumsum over levels +
alpha correction) — horizontally pointwise, NO stencil — so it tiles EXACTLY
(slice the cc horizontal tile; the vertical scan runs local per tile since nlev
is replicated, not sharded).  Bit-identity vs the global op, host-body + np24.

54 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_sigma_coordinate, compute_geopotential,
    make_hybrid_levels, compute_geopotential_hybrid,
)
from legoesm.parallel.tiled_production_cdgrid import (
    compute_geopotential_tile_2d, make_tiled_compute_geopotential_stage_2d,
    compute_geopotential_hybrid_tile_2d,
    make_tiled_compute_geopotential_hybrid_stage_2d,
)


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(250.0 + 30.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    phis = jnp.asarray(1.0e3 * rng.standard_normal((6, n, n)))
    return T, p_s, phis


def _reassemble_cc(get_tile, kt):
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


def test_geopotential_host_body_tiling():
    kt, nl, nlev = 3, 6, 8
    n = kt * nl
    sigma = create_sigma_coordinate(nlev)
    T, p_s, phis = _inputs(n, nlev, 81)
    Phi_g = np.asarray(compute_geopotential(T, p_s, sigma, phis))   # (6,n,n,nlev)

    def get_tile(ti, tj):
        return compute_geopotential_tile_2d(
            T, p_s, phis, sigma, ti * nl, tj * nl, nl)

    Phi_t = _reassemble_cc(get_tile, kt)
    assert Phi_t.shape == (6, n, n, nlev)
    np.testing.assert_allclose(
        Phi_t, Phi_g, rtol=0, atol=1e-9,
        err_msg="tiled geopotential host-body != global")


def test_geopotential_shard_map_np24():
    kt, nl, nlev = 2, 6, 8
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    sigma = create_sigma_coordinate(nlev)
    T, p_s, phis = _inputs(n, nlev, 82)
    Phi_g = np.asarray(compute_geopotential(T, p_s, sigma, phis))
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_compute_geopotential_stage_2d(mesh, sigma, n, kt)
    Phi_s = np.asarray(stage(T, p_s, phis)).reshape(6, kt, nl, kt, nl, nlev)
    Phi_t = _reassemble_cc(lambda ti, tj: Phi_s[:, ti, :, tj, :, :], kt)
    assert Phi_t.shape == (6, n, n, nlev)
    np.testing.assert_allclose(
        Phi_t, Phi_g, rtol=0, atol=1e-9,
        err_msg="tiled geopotential np24 != global")


# --- HYBRID coord (production OMIP vertical coord) — same per-column tiling ---
def test_geopotential_hybrid_host_body_tiling():
    kt, nl, nlev = 3, 6, 8
    n = kt * nl
    coord = make_hybrid_levels(nlev)
    T, p_s, phis = _inputs(n, nlev, 91)
    Phi_g = np.asarray(compute_geopotential_hybrid(T, p_s, coord, phis))

    def get_tile(ti, tj):
        return compute_geopotential_hybrid_tile_2d(
            T, p_s, phis, coord, ti * nl, tj * nl, nl)

    Phi_t = _reassemble_cc(get_tile, kt)
    assert Phi_t.shape == (6, n, n, nlev)
    np.testing.assert_allclose(
        Phi_t, Phi_g, rtol=0, atol=1e-9,
        err_msg="tiled hybrid geopotential host-body != global")


def test_geopotential_hybrid_shard_map_np24():
    kt, nl, nlev = 2, 6, 8
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    coord = make_hybrid_levels(nlev)
    T, p_s, phis = _inputs(n, nlev, 92)
    Phi_g = np.asarray(compute_geopotential_hybrid(T, p_s, coord, phis))
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_compute_geopotential_hybrid_stage_2d(mesh, coord, n, kt)
    Phi_s = np.asarray(stage(T, p_s, phis)).reshape(6, kt, nl, kt, nl, nlev)
    Phi_t = _reassemble_cc(lambda ti, tj: Phi_s[:, ti, :, tj, :, :], kt)
    assert Phi_t.shape == (6, n, n, nlev)
    np.testing.assert_allclose(
        Phi_t, Phi_g, rtol=0, atol=1e-9,
        err_msg="tiled hybrid geopotential np24 != global")
