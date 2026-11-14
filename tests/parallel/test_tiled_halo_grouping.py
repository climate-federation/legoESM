"""Halo grouping: real closed-loop steps, AD and structural control.

Run via Slurm with 24/96 CPU virtual devices. No timing claim from this test.
"""
from __future__ import annotations

import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from legoesm.parallel import cubesphere_exchange as exchange


@pytest.fixture(autouse=True)
def restore_precision():
    from legoesm.core.precision import get_policy, set_policy
    from legoesm.grids.halo import get_halo_backend, set_halo_backend
    policy, x64 = get_policy(), jax.config.jax_enable_x64
    backend = get_halo_backend()
    yield
    set_halo_backend(backend)
    set_policy(policy)
    jax.config.update('jax_enable_x64', x64)


def test_bundle_uses_one_pad_and_preserves_dtypes():
    calls = []

    def pad(x, offsets):
        calls.append(jax.tree.map(lambda a: (a.shape, a.dtype), x))
        return jax.tree.map(lambda a: jnp.pad(a, ((1, 1), (1, 1), (0, 0))), x)

    fields = tuple(jnp.full((4, 4, w), i + 1, dtype=jnp.float32)
                   for i, w in enumerate((3, 3, 3, 3, 1, 3)))
    out = exchange.bundled_tiled_pad(pad, fields, None)
    assert calls == [((4, 4, 16), jnp.dtype('float32'))]
    for ref, got in zip(fields, out):
        np.testing.assert_array_equal(got[1:-1, 1:-1], ref)
        assert got.dtype == ref.dtype
    # Distinct dtypes must never share a promoted buffer.
    calls.clear()
    mixed = fields + (jnp.ones((4, 4, 1), dtype=jnp.int32),)
    out = exchange.bundled_tiled_pad(pad, mixed, None)
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert tuple(x.dtype for x in out) == tuple(x.dtype for x in mixed)


def test_real_mixed_dtype_pad_and_2d_vector():
    """Exercise actual tuple ppermutes, and the existing 2-D vector contract."""
    if len(jax.devices()) < 24:
        pytest.skip('needs 24 virtual devices')
    from functools import partial
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.grids.halo import compute_halo_interp_offsets
    from legoesm.parallel.shard_map_compat import shard_map
    from scripts.bench.metadata import count_collective_permutes

    set_policy(PrecisionPolicy.fp64())
    n, kt, nl = 24, 2, 12
    mesh = Mesh(np.array(jax.devices()[:24]).reshape(6, kt, kt),
                ('face', 'tile_i', 'tile_j'))
    spec = P('face', 'tile_i', 'tile_j', None)
    rng = np.random.default_rng(931)
    fields = tuple(jax.device_put(jnp.asarray(
        rng.normal(size=(6, n, n, width)),
        dtype=jnp.float64 if i == 4 else jnp.float32), NamedSharding(mesh, spec))
        for i, width in enumerate((4, 4, 4, 4, 1, 4)))
    offsets = jnp.asarray(compute_halo_interp_offsets(n))
    scalar = exchange.make_tiled_pad_body(mesh, 4, with_offsets=True)
    outputs, counts = [], []
    for mode in ('current', 'bundled'):
        @partial(shard_map, mesh=mesh,
                 in_specs=((spec,) * len(fields), P()),
                 out_specs=(spec,) * len(fields), check_vma=False)
        def stage(xs, offs):
            local = tuple(x[0] for x in xs)
            if mode == 'bundled':
                padded = exchange.bundled_tiled_pad(scalar, local, offs)
            else:
                padded = tuple(scalar(x, offs) for x in local)
            return tuple(x[None] for x in padded)
        compiled = jax.jit(stage).lower(fields, offsets).compile()
        counts.append(count_collective_permutes(compiled.as_text()))
        outputs.append(compiled(fields, offsets))
    for ref, a, b in zip(fields, *outputs):
        assert a.dtype == b.dtype == ref.dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    print(f'REAL_MIXED_PAD dtype_buffers=float32,float64 exact=True permutes={counts}', flush=True)

    vector_outputs = []
    spec3 = P('face', 'tile_i', 'tile_j')
    u, v = fields[0][..., 0], fields[1][..., 0]
    for mode in ('current', 'bundled'):
        vector = exchange.make_tiled_pad_vector_body(
            mesh, 3, with_offsets=True, halo_grouping=mode)
        @partial(shard_map, mesh=mesh, in_specs=(spec3, spec3, P()),
                 out_specs=(spec3, spec3), check_vma=False)
        def vstage(u, v, offs):
            ca = jnp.full((nl, nl), .8, dtype=u.dtype)
            sa = jnp.full((nl, nl), .6, dtype=u.dtype)
            cap = jnp.full((nl+2, nl+2), .8, dtype=u.dtype)
            sap = jnp.full((nl+2, nl+2), .6, dtype=u.dtype)
            up, vp = vector(u[0], v[0], ca, sa, cap, sap, offs)
            return up[None], vp[None]
        vector_outputs.append(jax.jit(vstage)(u, v, offsets))
    for a, b in zip(*vector_outputs):
        assert a.dtype == b.dtype == jnp.float32
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    print('BUNDLED_2D_VECTOR exact=True', flush=True)
