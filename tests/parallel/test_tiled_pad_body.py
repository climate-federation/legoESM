"""P4 phase-1b: the unwrapped tiled pad body works inside an outer
shard_map and matches the wrapped exchange bit-for-bit.

``make_tiled_pad_body`` returns the tiled halo pad WITHOUT its own
shard_map wrapper, so the tiled FV3 tendency stage can call it from
inside its own ``shard_map(face,tile_i,tile_j)`` (a nested shard_map —
what the operators' SPMD pad does — would be illegal).  This pins:

1. wrapped ``_make_exchange_ppermute_tiled`` still equals the serial
   pad after the _build_tiled_pad refactor (refactor-safety);
2. the unwrapped body, invoked inside a SEPARATE outer shard_map over
   the same axes, produces the identical global padded result.

Runs on 24 host CPU devices (kt=2) — no MPI:
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
    from jax import shard_map  # JAX >= 0.8 top-level
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from legoesm.grids.halo import pad_halo_local, compute_halo_interp_offsets
from legoesm.parallel.cubesphere_exchange import (
    _make_exchange_ppermute_tiled,
    make_tiled_pad_body,
)

KT = 2
NL = 24
N = KT * NL


@pytest.fixture(scope="module")
def mesh():
    if len(jax.devices()) < 24:
        pytest.skip("needs 24 host devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
    dev = np.array(jax.devices()[:24]).reshape(6, KT, KT)
    return Mesh(dev, axis_names=("face", "tile_i", "tile_j"))


def test_wrapped_tiled_exchange_matches_serial(mesh):
    rng = np.random.default_rng(1)
    ref = jnp.asarray(rng.standard_normal((6, N, N)))
    offs = compute_halo_interp_offsets(N)
    serial = np.asarray(pad_halo_local(ref, interp_offsets=offs))

    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    ref_sh = jax.device_put(ref, sharding)
    fn = _make_exchange_ppermute_tiled(mesh, ndim=3, halo=1,
                                       with_offsets=True)
    out = fn(ref_sh, jnp.asarray(offs))
    blk = NL + 2
    for shard in out.addressable_shards:
        f = shard.index[0].start or 0
        ti = (shard.index[1].start or 0) // blk
        tj = (shard.index[2].start or 0) // blk
        got = np.asarray(shard.data)[0]
        want = serial[f, ti * NL: ti * NL + blk, tj * NL: tj * NL + blk]
        np.testing.assert_array_equal(got, want)


def test_unwrapped_body_inside_outer_shardmap_matches_wrapped(mesh):
    """The whole point: the body runs inside a DIFFERENT shard_map and
    gives the same padded blocks as the wrapped exchange."""
    rng = np.random.default_rng(2)
    ref = jnp.asarray(rng.standard_normal((6, N, N)))
    offs = jnp.asarray(compute_halo_interp_offsets(N))
    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    ref_sh = jax.device_put(ref, sharding)

    wrapped = _make_exchange_ppermute_tiled(
        mesh, ndim=3, halo=1, with_offsets=True)
    out_wrapped = wrapped(ref_sh, offs)

    body = make_tiled_pad_body(mesh, ndim=3, halo=1, with_offsets=True)

    # Outer shard_map (separate from the body's own) that calls the
    # unwrapped body on each device's local tile — exactly how the
    # tiled tendency stage will invoke it.
    out_blk = NL + 2
    in_sp = (P("face", "tile_i", "tile_j"), P())
    out_sp = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _stage(local_shard, o):
        return body(local_shard[0], o)[None]

    out_body = _stage(ref_sh, offs)

    for sw, sb in zip(out_wrapped.addressable_shards,
                      out_body.addressable_shards):
        np.testing.assert_array_equal(
            np.asarray(sw.data), np.asarray(sb.data),
            err_msg="unwrapped body != wrapped exchange")
    assert out_body.shape == (6, KT * out_blk, KT * out_blk)
