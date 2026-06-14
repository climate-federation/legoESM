"""P-iii: sub-face tiled PRODUCTION arakawa_lamb_gradient (default path).

cc B -> D-grid corner gradient (dB_dx, dB_dy_perp).  Tiles the GLOBAL
pad_halo_auto pre-pad B_pad + the per-tile corner matrices grad_c00..c11 via the
extracted arakawa_lamb_gradient_core (no dup).  Two staggered corner outputs ->
lower-owns-shared reassembly.  Tight-tol parity (rtol=0, atol=1e-12; exact is
non-portable per the U4a FMA finding).  (a2b/dir-aware cube-vertex diagnostics
are OFF in production and not tiled.)
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_cdgrid import arakawa_lamb_gradient, pad_halo_auto
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.tiled_production_cdgrid import (
    arakawa_lamb_gradient_tile_2d, make_tiled_arakawa_lamb_gradient_stage_2d,
)


def _cube(kt, nl):
    n = kt * nl
    return create_cubed_sphere_cdgrid(create_cubed_sphere(n=n, use_duogrid=True)), n


def _reassemble_corner(get_tile, kt, nl):
    """corner staggered (n+1) -> lower tile keeps its low edge (trim high edge
    of non-last tiles); duplicated shared faces are bit-identical."""
    rows = []
    for ti in range(kt):
        cols = []
        for tj in range(kt):
            blk = np.asarray(get_tile(ti, tj))            # (6, nl+1, nl+1[, nlev])
            a_hi = nl + 1 if ti == kt - 1 else nl
            b_hi = nl + 1 if tj == kt - 1 else nl
            cols.append(blk[:, :a_hi, :b_hi])
        rows.append(np.concatenate(cols, axis=2))
    return np.concatenate(rows, axis=1)


def _check(cd, n, kt, nl, B):
    gx, gy = arakawa_lamb_gradient(B, cd)
    gx, gy = np.asarray(gx), np.asarray(gy)
    B_pad = pad_halo_auto(B, cd)

    def get(ti, tj):
        return arakawa_lamb_gradient_tile_2d(
            B_pad, cd.grad_c00, cd.grad_c01, cd.grad_c10, cd.grad_c11,
            ti * nl, tj * nl, nl)

    tx = _reassemble_corner(lambda ti, tj: get(ti, tj)[0], kt, nl)
    ty = _reassemble_corner(lambda ti, tj: get(ti, tj)[1], kt, nl)
    np.testing.assert_allclose(tx, gx, rtol=0, atol=1e-12,
                               err_msg="P-iii dB_dx tiled != global")
    np.testing.assert_allclose(ty, gy, rtol=0, atol=1e-12,
                               err_msg="P-iii dB_dy_perp tiled != global")


def test_al_gradient_host_body_3d():
    kt, nl = 3, 6
    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(61)
    _check(cd, n, kt, nl, jnp.asarray(rng.standard_normal((6, n, n))))


def test_al_gradient_host_body_4d():
    kt, nl, nlev = 3, 6, 4
    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(62)
    _check(cd, n, kt, nl, jnp.asarray(rng.standard_normal((6, n, n, nlev))))


def test_al_gradient_shard_map_np24():
    kt, nl = 2, 6
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(63)
    B = jnp.asarray(rng.standard_normal((6, n, n)))
    gx, gy = arakawa_lamb_gradient(B, cd)
    gx, gy = np.asarray(gx), np.asarray(gy)
    B_pad = pad_halo_auto(B, cd)
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_arakawa_lamb_gradient_stage_2d(mesh, cd, n, kt)
    sx, sy = stage(B_pad)
    sx = np.asarray(sx).reshape(6, kt, nl + 1, kt, nl + 1)
    sy = np.asarray(sy).reshape(6, kt, nl + 1, kt, nl + 1)
    tx = _reassemble_corner(lambda ti, tj: sx[:, ti, :, tj, :], kt, nl)
    ty = _reassemble_corner(lambda ti, tj: sy[:, ti, :, tj, :], kt, nl)
    np.testing.assert_allclose(tx, gx, rtol=0, atol=1e-12,
                               err_msg="P-iii dB_dx shard_map != global")
    np.testing.assert_allclose(ty, gy, rtol=0, atol=1e-12,
                               err_msg="P-iii dB_dy_perp shard_map != global")


def test_al_gradient_shard_map_np24_4d():
    """4D ``(F, n+2, n+2, nlev)`` B_pad shard_map np24 — the 3D PE dycore A-L
    gradient (B and ln_ps PGF) on np=6*kt*kt devices (vertical replicated)."""
    kt, nl, nlev = 2, 6, 4
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, n = _cube(kt, nl)
    rng = np.random.default_rng(634)
    B = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    gx, gy = arakawa_lamb_gradient(B, cd)
    gx, gy = np.asarray(gx), np.asarray(gy)            # (6, n+1, n+1, nlev)
    B_pad = pad_halo_auto(B, cd)                        # (6, n+2, n+2, nlev)
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_arakawa_lamb_gradient_stage_2d(mesh, cd, n, kt, nlev=nlev)
    sx, sy = stage(B_pad)
    sx = np.asarray(sx).reshape(6, kt, nl + 1, kt, nl + 1, nlev)
    sy = np.asarray(sy).reshape(6, kt, nl + 1, kt, nl + 1, nlev)
    tx = _reassemble_corner(lambda ti, tj: sx[:, ti, :, tj, :, :], kt, nl)
    ty = _reassemble_corner(lambda ti, tj: sy[:, ti, :, tj, :, :], kt, nl)
    assert tx.shape == (6, n + 1, n + 1, nlev)
    np.testing.assert_allclose(tx, gx, rtol=0, atol=1e-12,
                               err_msg="P-iii dB_dx 4D shard_map != global")
    np.testing.assert_allclose(ty, gy, rtol=0, atol=1e-12,
                               err_msg="P-iii dB_dy_perp 4D shard_map != global")
