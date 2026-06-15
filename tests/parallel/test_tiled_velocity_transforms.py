"""P-v: sub-face tiled PRODUCTION velocity transforms (task (a)).

fv3_d2cc (D-grid edge-midpoint -> cc; PURELY LOCAL 2-pt avg, exact cc partition)
and fv3_cc2c (cc -> C-grid; VECTOR — tiles the GLOBAL pad_halo_vector pre-pad via
fv3_cc2c_core, staggered C-grid outputs -> lower-owns-shared).  Tile kernels call
the production ops / cores on the sliced blocks (no dup numerics).  Tight-tol
parity (rtol=0, atol=1e-12; exact non-portable per the U4a FMA finding).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_cdgrid import fv3_d2cc, fv3_cc2c
from legoesm.grids.halo import pad_halo_vector
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.tiled_production_cdgrid import (
    fv3_d2cc_tile_2d, fv3_cc2c_tile_2d,
    make_tiled_fv3_d2cc_stage_2d, make_tiled_fv3_cc2c_stage_2d,
)


def _cube(kt, nl):
    n = kt * nl
    return create_cubed_sphere_cdgrid(create_cubed_sphere(n=n, use_duogrid=True)), n


def _global_vec_pad(cd, u_cc, v_cc):
    grid = cd.base
    dg = grid.duogrid
    offsets = None if dg is not None else grid.halo_interp_offsets
    return pad_halo_vector(
        u_cc, v_cc, grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offsets, duogrid=dg)


def _reassemble_cc(get_tile, kt):
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


def _reassemble_stag(get_tile, kt, nl, stag):
    """stag='i' -> (n+1, n) lower-owns on i; stag='j' -> (n, n+1) lower-owns on j."""
    rows = []
    for ti in range(kt):
        cols = []
        for tj in range(kt):
            blk = np.asarray(get_tile(ti, tj))
            a_hi = (nl + 1 if ti == kt - 1 else nl) if stag == "i" else nl
            b_hi = (nl + 1 if tj == kt - 1 else nl) if stag == "j" else nl
            cols.append(blk[:, :a_hi, :b_hi])
        rows.append(np.concatenate(cols, axis=2))
    return np.concatenate(rows, axis=1)


def test_fv3_d2cc_host_body():
    kt, nl = 3, 6
    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(81)
    for shp in [((6, n, n + 1), (6, n + 1, n)),
                ((6, n, n + 1, 4), (6, n + 1, n, 4))]:
        u_d = jnp.asarray(rng.standard_normal(shp[0]))
        v_d = jnp.asarray(rng.standard_normal(shp[1]))
        gu, gv = fv3_d2cc(u_d, v_d, cd)
        tu = _reassemble_cc(lambda ti, tj: fv3_d2cc_tile_2d(u_d, v_d, ti * nl, tj * nl, nl)[0], kt)
        tv = _reassemble_cc(lambda ti, tj: fv3_d2cc_tile_2d(u_d, v_d, ti * nl, tj * nl, nl)[1], kt)
        np.testing.assert_allclose(tu, np.asarray(gu), rtol=0, atol=1e-12,
                                   err_msg="P-v fv3_d2cc u_cc tiled != global")
        np.testing.assert_allclose(tv, np.asarray(gv), rtol=0, atol=1e-12,
                                   err_msg="P-v fv3_d2cc v_cc tiled != global")


def test_fv3_cc2c_host_body():
    # fv3_cc2c is 3D-only: pad_halo_vector uses a 3D cos_angle metric and there
    # is no 4D caller (fv3_sw_tendencies + ocean barotropic both pass 3D winds).
    kt, nl = 3, 6
    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(82)
    u_cc = jnp.asarray(rng.standard_normal((6, n, n)))
    v_cc = jnp.asarray(rng.standard_normal((6, n, n)))
    gu, gv = fv3_cc2c(u_cc, v_cc, cd)
    u_pad, v_pad = _global_vec_pad(cd, u_cc, v_cc)
    blks = {(ti, tj): fv3_cc2c_tile_2d(u_pad, v_pad, cd.cosa_u, ti * nl, tj * nl, nl)
            for ti in range(kt) for tj in range(kt)}

    # duplicate shared faces bit-identical (codex P-v LOW): u_c i-staggered, v_c j.
    for ti in range(kt - 1):
        for tj in range(kt):
            np.testing.assert_allclose(
                np.asarray(blks[(ti, tj)][0])[:, nl, :],
                np.asarray(blks[(ti + 1, tj)][0])[:, 0, :], rtol=0, atol=1e-12,
                err_msg=f"cc2c u_c shared i-face (ti={ti},tj={tj}) != neighbour")
    for ti in range(kt):
        for tj in range(kt - 1):
            np.testing.assert_allclose(
                np.asarray(blks[(ti, tj)][1])[:, :, nl],
                np.asarray(blks[(ti, tj + 1)][1])[:, :, 0], rtol=0, atol=1e-12,
                err_msg=f"cc2c v_c shared j-face (ti={ti},tj={tj}) != neighbour")

    tu = _reassemble_stag(lambda ti, tj: blks[(ti, tj)][0], kt, nl, "i")
    tv = _reassemble_stag(lambda ti, tj: blks[(ti, tj)][1], kt, nl, "j")
    np.testing.assert_allclose(tu, np.asarray(gu), rtol=0, atol=1e-12,
                               err_msg="P-v fv3_cc2c u_c tiled != global")
    np.testing.assert_allclose(tv, np.asarray(gv), rtol=0, atol=1e-12,
                               err_msg="P-v fv3_cc2c v_c tiled != global")


def test_velocity_transforms_shard_map_np24():
    kt, nl = 2, 6
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, n = _cube(kt, nl)
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    rng = np.random.default_rng(83)

    # d2cc
    u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)))
    gu, gv = fv3_d2cc(u_d, v_d, cd)
    su, sv = make_tiled_fv3_d2cc_stage_2d(mesh, n, kt)(u_d, v_d)
    su = np.asarray(su).reshape(6, kt, nl, kt, nl)
    sv = np.asarray(sv).reshape(6, kt, nl, kt, nl)
    np.testing.assert_allclose(_reassemble_cc(lambda ti, tj: su[:, ti, :, tj, :], kt),
                               np.asarray(gu), rtol=0, atol=1e-12,
                               err_msg="P-v d2cc u_cc shard_map != global")
    np.testing.assert_allclose(_reassemble_cc(lambda ti, tj: sv[:, ti, :, tj, :], kt),
                               np.asarray(gv), rtol=0, atol=1e-12,
                               err_msg="P-v d2cc v_cc shard_map != global")

    # cc2c
    u_cc = jnp.asarray(rng.standard_normal((6, n, n)))
    v_cc = jnp.asarray(rng.standard_normal((6, n, n)))
    gcu, gcv = fv3_cc2c(u_cc, v_cc, cd)
    u_pad, v_pad = _global_vec_pad(cd, u_cc, v_cc)
    scu, scv = make_tiled_fv3_cc2c_stage_2d(mesh, cd, n, kt)(u_pad, v_pad)
    scu = np.asarray(scu).reshape(6, kt, nl + 1, kt, nl)
    scv = np.asarray(scv).reshape(6, kt, nl, kt, nl + 1)
    np.testing.assert_allclose(_reassemble_stag(lambda ti, tj: scu[:, ti, :, tj, :], kt, nl, "i"),
                               np.asarray(gcu), rtol=0, atol=1e-12,
                               err_msg="P-v cc2c u_c shard_map != global")
    np.testing.assert_allclose(_reassemble_stag(lambda ti, tj: scv[:, ti, :, tj, :], kt, nl, "j"),
                               np.asarray(gcv), rtol=0, atol=1e-12,
                               err_msg="P-v cc2c v_c shard_map != global")
