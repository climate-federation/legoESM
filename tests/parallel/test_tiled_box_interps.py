"""P-ii: sub-face tiled PRODUCTION 4-point box interps (task (a)).

interp_corner_to_center (corner->cc, PURELY LOCAL — no halo, exact cc partition)
and interp_center_to_corner (cc->corner, needs the cc 1-cell halo: tile the
GLOBAL pad_halo_auto pre-pad; corner staggered -> lower-owns-shared).  Tile
kernels call the production ops on the sliced blocks (no dup numerics).  Tight-tol
parity vs the global op (rtol=0, atol=1e-12; exact is non-portable per U4a FMA).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_cdgrid import (
    interp_corner_to_center, interp_center_to_corner, pad_halo_auto,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.tiled_production_cdgrid import (
    interp_corner_to_center_tile_2d, interp_center_to_corner_tile_2d,
    make_tiled_interp_corner_to_center_stage_2d,
    make_tiled_interp_center_to_corner_stage_2d,
)


def _cube(kt, nl):
    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    return create_cubed_sphere_cdgrid(grid), n


def _reassemble_cc(get_tile, kt):
    """cc cells partition exactly → concat (nl,nl) blocks, no trim."""
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


def _reassemble_corner(get_tile, kt, nl):
    """corner staggered (n+1) → lower tile owns the shared face (trim last)."""
    rows = []
    for ti in range(kt):
        cols = []
        for tj in range(kt):
            blk = np.asarray(get_tile(ti, tj))            # (6, nl+1, nl+1)
            a_hi = nl + 1 if ti == kt - 1 else nl
            b_hi = nl + 1 if tj == kt - 1 else nl
            cols.append(blk[:, :a_hi, :b_hi])
        rows.append(np.concatenate(cols, axis=2))
    return np.concatenate(rows, axis=1)                   # (6, n+1, n+1)


def test_corner_to_center_host_body():
    kt, nl = 3, 6
    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(51)
    field_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))
    g = np.asarray(interp_corner_to_center(field_d))
    t = _reassemble_cc(
        lambda ti, tj: interp_corner_to_center_tile_2d(field_d, ti * nl, tj * nl, nl), kt)
    np.testing.assert_allclose(t, g, rtol=0, atol=1e-12,
                               err_msg="P-ii corner_to_center tiled != global")


def test_center_to_corner_host_body():
    kt, nl = 3, 6
    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(52)
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    g = np.asarray(interp_center_to_corner(field, cd))
    f_pad = pad_halo_auto(field, cd)                       # (6, n+2, n+2) global
    t = _reassemble_corner(
        lambda ti, tj: interp_center_to_corner_tile_2d(f_pad, cd, ti * nl, tj * nl, nl),
        kt, nl)
    np.testing.assert_allclose(t, g, rtol=0, atol=1e-12,
                               err_msg="P-ii center_to_corner tiled != global")


def test_box_interps_4d_host_body():
    """4D (n,n,nlev) parity — the box interps run 4D in the 3D dycore
    (theta_corner / zeta_corner)."""
    kt, nl, nlev = 3, 6, 4
    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(54)
    # corner -> cc (4D)
    field_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    g_cc = np.asarray(interp_corner_to_center(field_d))
    t_cc = _reassemble_cc(
        lambda ti, tj: interp_corner_to_center_tile_2d(field_d, ti * nl, tj * nl, nl), kt)
    np.testing.assert_allclose(t_cc, g_cc, rtol=0, atol=1e-12,
                               err_msg="P-ii 4D corner_to_center tiled != global")
    # cc -> corner (4D)
    field = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    g_co = np.asarray(interp_center_to_corner(field, cd))
    f_pad = pad_halo_auto(field, cd)                       # (6, n+2, n+2, nlev)
    t_co = _reassemble_corner(
        lambda ti, tj: interp_center_to_corner_tile_2d(f_pad, cd, ti * nl, tj * nl, nl),
        kt, nl)
    np.testing.assert_allclose(t_co, g_co, rtol=0, atol=1e-12,
                               err_msg="P-ii 4D center_to_corner tiled != global")


def test_box_interps_shard_map_np24():
    kt, nl = 2, 6
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, n = _cube(kt, nl)
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    rng = np.random.default_rng(53)

    # corner -> cc
    field_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))
    g_cc = np.asarray(interp_corner_to_center(field_d))
    s_cc = make_tiled_interp_corner_to_center_stage_2d(mesh, n, kt)
    cc = np.asarray(s_cc(field_d)).reshape(6, kt, nl, kt, nl)
    t_cc = _reassemble_cc(lambda ti, tj: cc[:, ti, :, tj, :], kt)
    np.testing.assert_allclose(t_cc, g_cc, rtol=0, atol=1e-12,
                               err_msg="P-ii corner_to_center shard_map != global")

    # cc -> corner
    field = jnp.asarray(rng.standard_normal((6, n, n)))
    g_co = np.asarray(interp_center_to_corner(field, cd))
    f_pad = pad_halo_auto(field, cd)
    s_co = make_tiled_interp_center_to_corner_stage_2d(mesh, cd, n, kt)
    co = np.asarray(s_co(f_pad)).reshape(6, kt, nl + 1, kt, nl + 1)
    t_co = _reassemble_corner(lambda ti, tj: co[:, ti, :, tj, :], kt, nl)
    np.testing.assert_allclose(t_co, g_co, rtol=0, atol=1e-12,
                               err_msg="P-ii center_to_corner shard_map != global")


def test_corner_to_center_shard_map_np24_4d():
    """4D ``(F, n+1, n+1, nlev)`` corner->cc shard_map np24 — the 3D PE dycore
    interp (dB cc) running on np=6*kt*kt devices (vertical replicated)."""
    kt, nl, nlev = 2, 6, 4
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, n = _cube(kt, nl)
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    rng = np.random.default_rng(534)
    field_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    g_cc = np.asarray(interp_corner_to_center(field_d))   # (6, n, n, nlev)
    s_cc = make_tiled_interp_corner_to_center_stage_2d(mesh, n, kt, nlev=nlev)
    cc = np.asarray(s_cc(field_d)).reshape(6, kt, nl, kt, nl, nlev)
    t_cc = _reassemble_cc(lambda ti, tj: cc[:, ti, :, tj, :, :], kt)
    assert t_cc.shape == (6, n, n, nlev)
    np.testing.assert_allclose(t_cc, g_cc, rtol=0, atol=1e-12,
                               err_msg="P-ii corner_to_center 4D shard_map != global")
