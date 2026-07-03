"""P-i: sub-face tiled PRODUCTION dgrid_vorticity (task (a), first production op).

dgrid_vorticity (operators_cdgrid) is the production cube cc-vorticity op
(fv3_sw_tendencies sec f). It is PURELY LOCAL — cc cell (i,j) reads only the
2x2 corner block [i:i+2,j:j+2] — so tiling is trivial approach-C: no halo, no
pre-pad, corners (n+1) sliced [a:a+nl+1], cc cells partition EXACTLY (no shared
face). Tight-tol parity vs the global op proves the tiling (rtol=0, atol=1e-12;
exact equality is non-portable — XLA FMA/codegen differs ~1 ULP between the
full-array and per-tile compiles, the U4a finding).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_cdgrid import dgrid_vorticity
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.parallel.tiled_production_cdgrid import (
    dgrid_vorticity_tile_2d, make_tiled_dgrid_vorticity_stage_2d,
)


def _setup(kt, nl, seed):
    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))  # corner winds
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1)))
    zeta_g = np.asarray(dgrid_vorticity(u_d, v_d, cd))          # (6, n, n)
    return cd, u_d, v_d, zeta_g


def _reassemble(get_tile, kt, nl):
    """cc cells partition exactly → concat tile blocks (nl,nl), NO trim."""
    rows = []
    for ti in range(kt):
        cols = [np.asarray(get_tile(ti, tj)) for tj in range(kt)]
        rows.append(np.concatenate(cols, axis=2))   # (6, nl, n)
    return np.concatenate(rows, axis=1)             # (6, n, n)


def test_dgrid_vorticity_host_body_tiling():
    kt, nl = 3, 6
    cd, u_d, v_d, zeta_g = _setup(kt, nl, seed=41)

    def get_tile(ti, tj):
        return dgrid_vorticity_tile_2d(
            u_d, v_d, cd.cosa_corner, cd.dx_edge_y, cd.dy_edge_x, cd.base.area,
            ti * nl, tj * nl, nl)

    zeta_t = _reassemble(get_tile, kt, nl)
    np.testing.assert_allclose(
        zeta_t, zeta_g, rtol=0, atol=1e-12,
        err_msg="P-i host-body tiled dgrid_vorticity != global")


def test_dgrid_vorticity_host_body_tiling_4d():
    """4D ``(F, n+1, n+1, nlev)`` corner winds — the 3D PE dycore case.

    The 3D ``fv3_hydrostatic_tendencies`` calls ``dgrid_vorticity`` on 4D D-grid
    winds (the metrics stay 2D-face, broadcast over nlev in the core).  The tile
    kernel's guard is HORIZONTAL-only (``[1:3]``) so it accepts the trailing
    nlev; this pins that the SAME kernel tiles the 4D op bit-identically — the
    foundation for the 3D-dycore tiling."""
    kt, nl, nlev = 3, 6, 5
    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(414)
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    zeta_g = np.asarray(dgrid_vorticity(u_d, v_d, cd))   # (6, n, n, nlev)

    def get_tile(ti, tj):
        return dgrid_vorticity_tile_2d(
            u_d, v_d, cd.cosa_corner, cd.dx_edge_y, cd.dy_edge_x, cd.base.area,
            ti * nl, tj * nl, nl)

    # cc cells partition exactly; _reassemble concats axes 1/2, keeps nlev.
    zeta_t = _reassemble(get_tile, kt, nl)
    assert zeta_t.shape == (6, n, n, nlev)
    np.testing.assert_allclose(
        zeta_t, zeta_g, rtol=0, atol=1e-12,
        err_msg="P-i 4D host-body tiled dgrid_vorticity != global")


def test_dgrid_vorticity_shard_map_stage_np24():
    kt, nl = 2, 6
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    cd, u_d, v_d, zeta_g = _setup(kt, nl, seed=42)
    n = kt * nl
    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_dgrid_vorticity_stage_2d(mesh, cd, n, kt)
    zeta_s = np.asarray(stage(u_d, v_d)).reshape(6, kt, nl, kt, nl)

    def get_tile(ti, tj):
        return zeta_s[:, ti, :, tj, :]

    zeta_t = _reassemble(get_tile, kt, nl)
    np.testing.assert_allclose(
        zeta_t, zeta_g, rtol=0, atol=1e-12,
        err_msg="P-i shard_map tiled dgrid_vorticity != global")


def test_dgrid_vorticity_shard_map_stage_np24_4d():
    """4D ``(F, n+1, n+1, nlev)`` shard_map np24 stage — the 3D PE dycore op
    running on np=6*kt*kt devices (the vertical axis is replicated, not tiled).
    First 3D-dycore op validated bit-identity at np24."""
    kt, nl, nlev = 2, 6, 4
    ndev = 6 * kt * kt
    if len(jax.devices()) < ndev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={ndev}")
    from jax.sharding import Mesh

    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cd = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(424)
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    zeta_g = np.asarray(dgrid_vorticity(u_d, v_d, cd))   # (6, n, n, nlev)

    dev = np.array(jax.devices()[:ndev]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_dgrid_vorticity_stage_2d(mesh, cd, n, kt, nlev=nlev)
    zeta_s = np.asarray(stage(u_d, v_d)).reshape(6, kt, nl, kt, nl, nlev)

    def get_tile(ti, tj):
        return zeta_s[:, ti, :, tj, :, :]

    zeta_t = _reassemble(get_tile, kt, nl)
    assert zeta_t.shape == (6, n, n, nlev)
    np.testing.assert_allclose(
        zeta_t, zeta_g, rtol=0, atol=1e-12,
        err_msg="P-i 4D shard_map tiled dgrid_vorticity != global")
