"""3D PE vertical advection + omega (hybrid coord) sub-face-tiled.

vertical_advection_hybrid (-F*df/dp, upwind) and compute_omega_hybrid
(B_full*dp_s/dt + F_full) are per-COLUMN vertical ops: level-axis avg of the
half-level mass_flux, diff/upwind over levels, pressure_from_hybrid per column —
horizontally pointwise, NO stencil — so they tile EXACTLY (slice the cc tile;
the vertical work runs local per tile, nlev replicated).  Both ops are pure
mul/avg/diff/where per column (pressure_from_hybrid is linear A*p_ref+B*p_s), so
bit-identity holds EXACTLY.  Host-body + np24.

24 host CPU devices: ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.vertical import (
    make_hybrid_levels, vertical_advection_hybrid, compute_omega_hybrid,
)
from legoesm.parallel.tiled_production_cdgrid import (
    vertical_advection_hybrid_tile_2d, make_tiled_vertical_advection_hybrid_stage_2d,
    compute_omega_hybrid_tile_2d, make_tiled_compute_omega_hybrid_stage_2d,
)


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    field = jnp.asarray(250.0 + 30.0 * rng.standard_normal((6, n, n, nlev)))
    mass_flux = jnp.asarray(1.0 * rng.standard_normal((6, n, n, nlev + 1)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    dp_s_dt = jnp.asarray(1.0 * rng.standard_normal((6, n, n)))
    return field, mass_flux, p_s, dp_s_dt


def _reassemble_cc(get_tile, kt):
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


def test_vertical_advection_hybrid_host_body_tiling():
    kt, nl, nlev = 3, 6, 8
    n = kt * nl
    coord = make_hybrid_levels(nlev)
    field, mass_flux, p_s, _ = _inputs(n, nlev, 61)
    g = np.asarray(vertical_advection_hybrid(field, mass_flux, p_s, coord))

    def get_t(ti, tj):
        return vertical_advection_hybrid_tile_2d(
            field, mass_flux, p_s, coord, ti * nl, tj * nl, nl)

    t = _reassemble_cc(get_t, kt)
    assert t.shape == (6, n, n, nlev)
    np.testing.assert_array_equal(
        t, g, err_msg="tiled vertical_advection host-body != global (not bit-identical)")


def test_compute_omega_hybrid_host_body_tiling():
    kt, nl, nlev = 3, 6, 8
    n = kt * nl
    coord = make_hybrid_levels(nlev)
    _, mass_flux, p_s, dp_s_dt = _inputs(n, nlev, 62)
    g = np.asarray(compute_omega_hybrid(mass_flux, p_s, dp_s_dt, coord))

    def get_t(ti, tj):
        return compute_omega_hybrid_tile_2d(
            mass_flux, p_s, dp_s_dt, coord, ti * nl, tj * nl, nl)

    t = _reassemble_cc(get_t, kt)
    assert t.shape == (6, n, n, nlev)
    np.testing.assert_array_equal(
        t, g, err_msg="tiled omega host-body != global (not bit-identical)")


def test_vertical_advection_hybrid_shard_map_np24():
    kt, nl, nlev = 2, 6, 8
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    coord = make_hybrid_levels(nlev)
    field, mass_flux, p_s, _ = _inputs(n, nlev, 63)
    g = np.asarray(vertical_advection_hybrid(field, mass_flux, p_s, coord))
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_vertical_advection_hybrid_stage_2d(mesh, coord, n, kt)
    s = np.asarray(stage(field, mass_flux, p_s)).reshape(6, kt, nl, kt, nl, nlev)
    t = _reassemble_cc(lambda ti, tj: s[:, ti, :, tj, :, :], kt)
    assert t.shape == (6, n, n, nlev)
    np.testing.assert_array_equal(
        t, g, err_msg="tiled vertical_advection np24 != global (not bit-identical)")


def test_compute_omega_hybrid_shard_map_np24():
    kt, nl, nlev = 2, 6, 8
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    coord = make_hybrid_levels(nlev)
    _, mass_flux, p_s, dp_s_dt = _inputs(n, nlev, 64)
    g = np.asarray(compute_omega_hybrid(mass_flux, p_s, dp_s_dt, coord))
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_compute_omega_hybrid_stage_2d(mesh, coord, n, kt)
    s = np.asarray(stage(mass_flux, p_s, dp_s_dt)).reshape(6, kt, nl, kt, nl, nlev)
    t = _reassemble_cc(lambda ti, tj: s[:, ti, :, tj, :, :], kt)
    assert t.shape == (6, n, n, nlev)
    np.testing.assert_array_equal(
        t, g, err_msg="tiled omega np24 != global (not bit-identical)")
