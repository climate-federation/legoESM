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
from legoesm.parallel.tiled_production_cdgrid import (
    expand_corners_to_blocks, make_tiled_fv3_hydrostatic_step_blocked_2d,
)


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


@pytest.mark.parametrize('kt', [2, 4])
def test_grouped_closed_loop_and_grad(kt, monkeypatch):
    if len(jax.devices()) < 6 * kt * kt:
        pytest.skip(f'needs {6 * kt * kt} virtual devices')
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.halo import set_halo_backend
    from legoesm.grids.vertical import create_sigma_coordinate

    dtype = jnp.float64 if os.environ.get('GROUPING_FP64') == '1' else jnp.float32
    set_policy(PrecisionPolicy.fp64() if dtype == jnp.float64 else PrecisionPolicy.fp32())
    set_halo_backend('local')
    n, nlev = 24, 4
    mesh = Mesh(np.array(jax.devices()[:6*kt*kt]).reshape(6, kt, kt),
                ('face', 'tile_i', 'tile_j'))
    cdgrid = create_cubed_sphere_cdgrid(create_cubed_sphere(n))
    coord = create_sigma_coordinate(nlev)
    rng = np.random.default_rng(783)
    arrays = [jnp.asarray(a, dtype=dtype) for a in (
        .1*rng.standard_normal((6,n+1,n+1,nlev)),
        .1*rng.standard_normal((6,n+1,n+1,nlev)),
        250+2*rng.standard_normal((6,n,n,nlev)),
        1e5+50*rng.standard_normal((6,n,n)),
        100*rng.standard_normal((6,n,n)))]
    arrays[:2] = [expand_corners_to_blocks(a, kt, n//kt) for a in arrays[:2]]
    arrays = tuple(jax.device_put(a, NamedSharding(mesh, P('face','tile_i','tile_j',
                                                        *([None] if a.ndim == 4 else []))))
                   for a in arrays)
    calls = []
    original_pad = exchange.make_tiled_pad_body

    def counted_pad(*args, **kwargs):
        body = original_pad(*args, **kwargs)
        def counted(x, offsets):
            calls.append(jax.tree.map(lambda a: a.shape[-1], x))
            return body(x, offsets)
        return counted

    monkeypatch.setattr(exchange, 'make_tiled_pad_body', counted_pad)
    steps = [jax.jit(make_tiled_fv3_hydrostatic_step_blocked_2d(
        mesh, cdgrid, coord, n, kt, nlev, p_floor=1., dt=30.,
        fix_mass=True, halo_grouping=mode)) for mode in ('current','bundled')]
    # Census the SAME compiled real step used for the trajectory.
    from scripts.bench.metadata import count_collective_permutes
    counts = []
    for mode, step in zip(('current', 'bundled'), steps):
        calls.clear()
        lowered = step.lower(*arrays)
        if mode == 'bundled':
            # Fail if scalar or vector bundling is replaced by per-field pads.
            scalar_widths = calls[0]
            assert scalar_widths == 5*nlev+1 or scalar_widths == (5*nlev, 1), calls
            assert calls == [scalar_widths, 2*nlev] * 3, calls
        else:
            assert len(calls) == 8*3, calls
        compiled = lowered.compile()
        hlo = compiled.as_text()
        counts.append(count_collective_permutes(hlo))
        path = os.environ.get('GROUPING_ARTIFACT_DIR')
        if path:
            from pathlib import Path
            Path(path, f'{mode}_kt{kt}_{np.dtype(dtype).name}.hlo').write_text(hlo)
    states = [arrays[:4], arrays[:4]]
    for _ in range(3):
        states = [step(*s, arrays[4]) for step, s in zip(steps, states)]
    maxdiff = max(float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
                  for a,b in zip(*states))
    print(f'GROUPING kt={kt} dtype={np.dtype(dtype).name} steps=3 max_abs_diff={maxdiff} permutes={counts}', flush=True)
    for a,b in zip(*states):
        assert np.isfinite(np.asarray(a)).all()
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    # Nonuniform weighted loss covers every cell including seams and corners.
    weights = tuple(jnp.asarray(rng.uniform(.5, 1.5, a.shape), dtype=dtype)
                    for a in arrays[:4])
    grads, bare_grads = [], []
    for step in steps:
        def loss(T):
            out = step(arrays[0], arrays[1], T, arrays[3], arrays[4])
            return sum(jnp.mean(w*x) for w,x in zip(weights, out))
        grads.append(jax.jit(jax.grad(loss))(arrays[2]))
        bare_grads.append(jax.grad(loss)(arrays[2]))
    grad_diff = float(np.max(np.abs(np.asarray(grads[0])-np.asarray(grads[1]))))
    print(f'GROUPING kt={kt} grad_max_abs_diff={grad_diff} grad_max={float(jnp.max(jnp.abs(grads[0])))}', flush=True)
    bare_diff = float(np.max(np.abs(np.asarray(bare_grads[0])-np.asarray(bare_grads[1]))))
    print(f'GROUPING kt={kt} bare_grad_max_abs_diff={bare_diff}', flush=True)
    for g in grads + bare_grads:
        assert np.isfinite(np.asarray(g)).all()
        assert np.any(np.asarray(g) != 0)
    # User-approved AD tolerances; forward comparison above stays bitwise.
    rtol, atol = (1e-6, 1e-7) if dtype == jnp.float32 else (1e-12, 1e-16)
    for pair in (grads, bare_grads):
        np.testing.assert_allclose(*[np.asarray(g) for g in pair],
                                   rtol=rtol, atol=atol)


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


def test_halo_grouping_rejects_unknown():
    from types import SimpleNamespace
    from legoesm.parallel.tiled_production_cdgrid import (
        make_tiled_fv3_hydrostatic_step_stage_2d,
    )
    # Satisfy earlier static guards so the actual selection check is reached.
    mesh = SimpleNamespace(devices=np.empty((6, 2, 2), dtype=object))
    cdgrid = SimpleNamespace(n=24, base=SimpleNamespace(duogrid=None))
    coord = SimpleNamespace(n_levels=4)
    with pytest.raises(ValueError, match='unknown halo_grouping'):
        exchange.make_tiled_pad_vector_body(mesh, 4, halo_grouping='invalid')
    for factory in (make_tiled_fv3_hydrostatic_step_stage_2d,
                    make_tiled_fv3_hydrostatic_step_blocked_2d):
        with pytest.raises(ValueError, match='unknown halo_grouping'):
            factory(mesh, cdgrid, coord, 24, 2, 4, p_floor=1., dt=30.,
                    halo_grouping='invalid')


def test_step_builders_default_to_bundled():
    import inspect
    from legoesm.atmosphere.dynamics.gcm import tiled_step_adapter as adapter
    from legoesm.parallel import tiled_production_cdgrid as tiled
    for builder in (adapter.make_tiled_cc_step, adapter.make_tiled_cc_loop,
                    tiled.make_tiled_fv3_hydrostatic_step_stage_2d,
                    tiled.make_tiled_fv3_hydrostatic_step_blocked_2d):
        default = inspect.signature(builder).parameters['halo_grouping'].default
        assert default == 'bundled', builder.__name__
