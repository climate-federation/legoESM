"""P4 phase-1b: tiled VECTOR halo pad (d2a2c prerequisite).

Grid-aligned winds rotate across cube-face seams, so a scalar tiled pad
is wrong at panel boundaries.  make_tiled_pad_vector_body mirrors
grids.halo.pad_halo_vector (orthogonal rotation) around the shared
scalar tiled pad.  Pins that, assembled over tiles, it reproduces the
global pad_halo_vector EXACTLY on the duplicated-overlap interior (the
cube-corner cells use avg fill, compared loosely like the scalar lane).

24 host CPU devices (kt=2):
``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import (
    pad_halo_vector,
    compute_halo_interp_offsets,
    set_halo_backend,
)
from legoesm.parallel.cubesphere_exchange import make_tiled_pad_vector_body
from legoesm.parallel.mesh import tiled_padded_block

N = 24
KT = 2
NL = N // KT


@pytest.fixture(scope="module")
def mesh():
    if len(jax.devices()) < 24:
        pytest.skip("needs --xla_force_host_platform_device_count=24")
    set_halo_backend("local")
    dev = np.array(jax.devices()[:24]).reshape(6, KT, KT)
    return Mesh(dev, axis_names=("face", "tile_i", "tile_j"))


def test_tiled_vector_pad_matches_global(mesh):
    grid = create_cubed_sphere(N)
    rng = np.random.default_rng(8)
    u = jnp.asarray(rng.standard_normal((6, N, N)))
    v = jnp.asarray(rng.standard_normal((6, N, N)))
    offs = compute_halo_interp_offsets(N)

    # Global reference (orthogonal rotation, no duogrid/non-orth).
    u_g, v_g = pad_halo_vector(
        u, v, grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=offs, halo=1)
    u_g, v_g = np.asarray(u_g), np.asarray(v_g)  # (6, N+2, N+2)

    vbody = make_tiled_pad_vector_body(mesh, ndim=3, halo=1,
                                       with_offsets=True)

    # Per-tile metric stacks (interior cos/sin_angle + padded slices).
    ca = grid.cos_angle
    sa = grid.sin_angle
    cap = grid.cos_angle_padded
    sap = grid.sin_angle_padded
    fo = P("face", None, None)
    co = P("face", "tile_i", "tile_j")
    sh_fo = NamedSharding(mesh, fo)

    u_sh = jax.device_put(u, sh_fo)
    v_sh = jax.device_put(v, sh_fo)
    ca_sh = jax.device_put(ca, sh_fo)
    sa_sh = jax.device_put(sa, sh_fo)
    cap_sh = jax.device_put(cap, sh_fo)
    sap_sh = jax.device_put(sap, sh_fo)
    offs_j = jnp.asarray(offs)

    @partial(shard_map, mesh=mesh,
             in_specs=(fo, fo, fo, fo, fo, fo, P()),
             out_specs=(co, co), check_vma=False)
    def _stage(u_, v_, ca_, sa_, cap_, sap_, o):
        ti = jax.lax.axis_index("tile_i")
        tj = jax.lax.axis_index("tile_j")
        uf, vf = u_[0], v_[0]
        caf, saf = ca_[0], sa_[0]
        capf, sapf = cap_[0], sap_[0]
        u_t = jax.lax.dynamic_slice(uf, (ti * NL, tj * NL), (NL, NL))
        v_t = jax.lax.dynamic_slice(vf, (ti * NL, tj * NL), (NL, NL))
        ca_t = jax.lax.dynamic_slice(caf, (ti * NL, tj * NL), (NL, NL))
        sa_t = jax.lax.dynamic_slice(saf, (ti * NL, tj * NL), (NL, NL))
        # padded angle: nl+2 window, strided (tiled_padded_block math).
        cap_t = jax.lax.dynamic_slice(
            capf, (ti * NL, tj * NL), (NL + 2, NL + 2))
        sap_t = jax.lax.dynamic_slice(
            sapf, (ti * NL, tj * NL), (NL + 2, NL + 2))
        up, vp = vbody(u_t, v_t, ca_t, sa_t, cap_t, sap_t, o)
        return up[None], vp[None]

    up, vp = _stage(u_sh, v_sh, ca_sh, sa_sh, cap_sh, sap_sh, offs_j)
    up, vp = np.asarray(up), np.asarray(vp)  # (6, KT*(NL+2), ...)

    # Compare each tile's padded block against the global padded slice,
    # excluding the 4 cube/tile corner cells (avg-fill — same carve-out
    # as the scalar tiled-pad parity probe).
    blk = NL + 2
    worst = 0.0
    for f in range(6):
        for ti in range(KT):
            for tj in range(KT):
                gb = u_g[f, ti * NL: ti * NL + blk, tj * NL: tj * NL + blk]
                tb = up[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                m = np.ones_like(gb, dtype=bool)
                for ci, cj in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                    m[ci, cj] = False
                worst = max(worst, float(np.max(np.abs((tb - gb)[m]))))
                gbv = v_g[f, ti * NL:ti * NL + blk, tj * NL:tj * NL + blk]
                tbv = vp[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                worst = max(worst, float(np.max(np.abs((tbv - gbv)[m]))))
    assert worst < 1e-12, f"tiled vector pad vs global: {worst:.3e}"
