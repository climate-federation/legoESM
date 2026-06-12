"""P4 phase-1b: second tile-aware operator — C-grid Bernoulli gradient.

cgrid_gradient_2d pads a scalar (eta) once (halo=1) then differences
cc -> edge midpoints.  Unlike the divergence (no halo), this exercises
the scalar tiled pad (make_tiled_pad_body) in a real operator.  The
gradient uses [1:-1] in the cross direction, so corner halo cells never
feed it -> tile-local parity is EXACT (no corner carve-out).

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

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend, compute_halo_interp_offsets
from legoesm.core.operators_cdgrid import (
    cgrid_gradient_2d,
    cgrid_gradient_2d_local,
)
from legoesm.parallel.cubesphere_exchange import make_tiled_pad_body
from legoesm.parallel.mesh import staggered_tile_block, tiled_face_block

N = 24
KT = 2
NL = N // KT


@pytest.fixture(scope="module")
def setup():
    set_halo_backend("local")
    cdg = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    rng = np.random.default_rng(6)
    eta = jnp.asarray(rng.standard_normal((6, N, N)))
    dx_g, dy_g = cgrid_gradient_2d(eta, cdg)
    return cdg, eta, np.asarray(dx_g), np.asarray(dy_g)


def test_tiled_local_gradient_in_shardmap(setup):
    if len(jax.devices()) < 24:
        pytest.skip("needs --xla_force_host_platform_device_count=24")
    cdg, eta, dx_g, dy_g = setup
    offs = compute_halo_interp_offsets(N)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:  # pragma: no cover
        from jax.experimental.shard_map import shard_map

    dev = np.array(jax.devices()[:24]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    body = make_tiled_pad_body(mesh, ndim=3, halo=1, with_offsets=True)

    co = P("face", "tile_i", "tile_j")
    fo = P("face", None, None)
    eta_sh = jax.device_put(eta, NamedSharding(mesh, co))   # tile-sharded
    rdxc_sh = jax.device_put(cdg.rdxc, NamedSharding(mesh, fo))
    rdyc_sh = jax.device_put(cdg.rdyc, NamedSharding(mesh, fo))
    offs_j = jnp.asarray(offs)

    @partial(shard_map, mesh=mesh,
             in_specs=(co, fo, fo, P()), out_specs=(co, co),
             check_vma=False)
    def _stage(eta_t, rdxc, rdyc, o):
        ti = jax.lax.axis_index("tile_i")
        tj = jax.lax.axis_index("tile_j")
        eta_pad = body(eta_t[0], o)            # (nl+2, nl+2)
        rdxc_t = jax.lax.dynamic_slice(
            rdxc[0], (ti * NL, tj * NL), (NL + 1, NL))
        rdyc_t = jax.lax.dynamic_slice(
            rdyc[0], (ti * NL, tj * NL), (NL, NL + 1))
        dx, dy = cgrid_gradient_2d_local(
            eta_pad[None], rdxc_t[None], rdyc_t[None])
        return dx, dy  # (1, nl+1, nl), (1, nl, nl+1)

    dx_t, dy_t = _stage(eta_sh, rdxc_sh, rdyc_sh, offs_j)
    dx_t = np.asarray(dx_t)  # (6, KT*(nl+1), KT*nl)
    dy_t = np.asarray(dy_t)

    # Reassemble staggered outputs (duplicated shared face) and compare
    # to the global gradient; gradient is corner-independent -> exact.
    blk_u = NL + 1
    blk_v = NL + 1
    worst = 0.0
    for f in range(6):
        for ti in range(KT):
            for tj in range(KT):
                d = (f * KT + ti) * KT + tj
                # u-faces (nl+1, nl): global slice with the shared face.
                gx = dx_g[f, ti * NL: ti * NL + NL + 1,
                          tj * NL:(tj + 1) * NL]
                tx = dx_t[f, ti * blk_u:(ti + 1) * blk_u,
                         tj * NL:(tj + 1) * NL]
                worst = max(worst, float(np.max(np.abs(tx - gx))))
                gy = dy_g[f, ti * NL:(ti + 1) * NL,
                          tj * NL: tj * NL + NL + 1]
                ty = dy_t[f, ti * NL:(ti + 1) * NL,
                         tj * blk_v:(tj + 1) * blk_v]
                worst = max(worst, float(np.max(np.abs(ty - gy))))
    # f64 floor: the tiled scalar pad's offset-interpolation arithmetic
    # orders ppermute receives differently from the global pad, so the
    # halo cells differ by O(1e-12) (vs the divergence operator, which
    # is exact because it has no halo).  Well below any real error.
    assert worst < 1e-10, f"tiled gradient vs global: {worst:.3e}"
