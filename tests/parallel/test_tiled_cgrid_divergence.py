"""P4 phase-1b: first tile-aware operator — C-grid divergence.

cgrid_divergence is flux-form (a cell reads only its four surrounding
C-grid faces), so a tile needs NO halo: its staggered u_c (nl+1, nl) /
v_c (nl, nl+1) blocks already carry the boundary faces (duplicated
shared face, Pace layout).  This pins that the tile-local core
(cgrid_divergence_local on a (1, nl[+1], nl[+1]) tile with per-tile
sliced metrics) reproduces the global cgrid_divergence restricted to
that tile, EXACTLY — the first proof of the per-tile-metric +
local-ownership + assembly mechanics the tiled tendency stage needs.

Also runs it inside an actual shard_map(face,tile_i,tile_j) on 24 host
devices (kt=2) to confirm the in-stage path.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.operators_cdgrid import (
    cgrid_divergence,
    cgrid_divergence_local,
)
from legoesm.parallel.mesh import staggered_tile_block, tiled_face_block

N = 12


@pytest.fixture(scope="module")
def setup():
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    rng = np.random.default_rng(4)
    u_c = jnp.asarray(rng.standard_normal((6, N + 1, N)))  # x-faces
    v_c = jnp.asarray(rng.standard_normal((6, N, N + 1)))  # y-faces
    div_global = np.asarray(cgrid_divergence(u_c, v_c, cdg))
    return cdg, u_c, v_c, div_global


@pytest.mark.parametrize("kt", (2, 3))
def test_tiled_local_divergence_matches_global(setup, kt):
    cdg, u_c, v_c, div_global = setup
    nl = N // kt
    dyx = cdg.dy_edge_x  # (6, n+1, n)
    dxy = cdg.dx_edge_y  # (6, n,   n+1)
    area = cdg.base.area  # (6, n, n)

    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                # Staggered tile blocks (duplicated shared face).
                u_t = staggered_tile_block(u_c[f], ti, tj, nl, 0)   # (nl+1,nl)
                v_t = staggered_tile_block(v_c[f], ti, tj, nl, 1)   # (nl,nl+1)
                dyx_t = staggered_tile_block(dyx[f], ti, tj, nl, 0)
                dxy_t = staggered_tile_block(dxy[f], ti, tj, nl, 1)
                area_t = tiled_face_block(area[f], ti, tj, nl, kt)  # (nl,nl)

                # Local core on a singleton-leading tile (mirrors the
                # face-only-sharded slab the stage slices per device).
                div_t = np.asarray(cgrid_divergence_local(
                    u_t[None], v_t[None], dyx_t[None], dxy_t[None],
                    area_t[None]))[0]

                want = div_global[
                    f, ti * nl:(ti + 1) * nl, tj * nl:(tj + 1) * nl]
                np.testing.assert_allclose(
                    div_t, want, rtol=0, atol=1e-13,
                    err_msg=f"tile f{f}({ti},{tj}) div != global")


def test_tiled_local_divergence_in_shardmap(setup):
    """Same op inside a real shard_map(face,tile_i,tile_j), 24 host
    devices, kt=2 — the in-stage path: each device slices its tile from
    the face-only-sharded staggered slabs via axis_index, runs the
    local core, returns its owned (nl,nl) block."""
    if len(jax.devices()) < 24:
        pytest.skip("needs --xla_force_host_platform_device_count=24")
    cdg, u_c, v_c, div_global = setup
    kt = 2
    nl = N // kt

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map

    dev = np.array(jax.devices()[:24]).reshape(6, kt, kt)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))

    # Staggered + centered metrics/state are face-only sharded (the
    # phase-1a policy): each device holds its whole face slab.
    face_only = NamedSharding(mesh, P("face", None, None))
    centered = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    u_sh = jax.device_put(u_c, face_only)
    v_sh = jax.device_put(v_c, face_only)
    dyx_sh = jax.device_put(cdg.dy_edge_x, face_only)
    dxy_sh = jax.device_put(cdg.dx_edge_y, face_only)
    area_sh = jax.device_put(cdg.base.area, centered)

    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh,
             in_specs=(fo, fo, fo, fo, co), out_specs=co, check_vma=False)
    def _stage(u, v, dyx, dxy, area):
        ti = jax.lax.axis_index("tile_i")
        tj = jax.lax.axis_index("tile_j")
        uf, vf = u[0], v[0]            # whole face slab (n+1,n)/(n,n+1)
        dyf, dxf = dyx[0], dxy[0]
        # Dynamic per-tile staggered slice (nl+1/nl on the stag axis).
        u_t = jax.lax.dynamic_slice(
            uf, (ti * nl, tj * nl), (nl + 1, nl))
        v_t = jax.lax.dynamic_slice(
            vf, (ti * nl, tj * nl), (nl, nl + 1))
        dy_t = jax.lax.dynamic_slice(
            dyf, (ti * nl, tj * nl), (nl + 1, nl))
        dx_t = jax.lax.dynamic_slice(
            dxf, (ti * nl, tj * nl), (nl, nl + 1))
        area_t = area[0]              # already this tile's (nl,nl)
        div = cgrid_divergence_local(
            u_t[None], v_t[None], dy_t[None], dx_t[None], area_t[None])
        return div  # (1, nl, nl)

    out = _stage(u_sh, v_sh, dyx_sh, dxy_sh, area_sh)
    got = np.asarray(out)
    np.testing.assert_allclose(got, div_global, rtol=0, atol=1e-13)
