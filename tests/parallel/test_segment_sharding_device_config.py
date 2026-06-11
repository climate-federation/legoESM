"""build_segment_fn(device_config=...) sharding tripwire (2 CPU devices).

The moist/AMIP segment path was the THIRD single-process multi-GPU
replication site: ``build_segment_fn``'s jitted kernels had no
``in_shardings``/``out_shardings``, so production AMIP could enter with
a face-sharded state and still compile fully replicated carry compute
(every device computing the whole globe — the 1->2 GPU "exactly 0.500
efficiency at identical wall time" signature).  The fix threads an
optional ``device_config`` through ``build_segment_fn``; when its mesh
is live, both donating segment kernels AND the non-donating ``.raw``
variant pin explicit shardings READ DIRECTLY OFF the input carry/forcing
the driver already sharded upstream (``shard_pytree`` / ``shard_state``).
The jit copies that existing layout verbatim rather than re-deriving a
``create_output_shardings`` policy — the latter keys on ``shape[0] == 6``
and so misclassifies the moist/AMIP carry's FLATTENED cell-packed fields
(``conv_prog`` is ``[6*n*n, nlev]``), producing an ``in_sharding`` ``P()``
that disagreed with the arg's actual ``P("face")`` and made ``jax.jit``
raise "Sharding passed to jit does not match the sharding on the
respective arg" (jobs 8457808/8457809).  ``out_shardings`` equals the
carry's input layout because the scan body preserves carry shape.

This test forces 2 CPU devices and asserts:

* the donating segment kernel keeps every face-leading ``SegmentCarry``
  leaf genuinely face-sharded across 2 segments (2 addressable shards,
  per-shard leading dim 3 = 6 faces / 2 devices), verified with the
  SAME bench tripwire (``_assert_expected_sharding``) the scaling
  benchmark runs — when the INPUT carry is sharded (``shard_pytree``),
  exactly as the production driver passes it;
* sharded JIT wrappers are cached — a second segment with the same
  carry/forcing treedefs AND layout does NOT rebuild a wrapper
  (CLAUDE.md closure/recompile rules);
* a forcing pytree-structure change (optional ``sfc_T_override``
  flipping None -> array) gets its own cache entry and still works
  (forcing treedefs vary across segments in coupled runs);
* ``.raw`` under ``device_config`` is the non-donating SHARDED jit
  wrapper: face-sharded output for a sharded input, the input carry
  stays alive (donation would invalidate it), explicit ``.lower(...)``
  works, and it is still differentiable (non-donation, not non-jit, is
  the AD requirement) — under ``jax.grad`` the carry leaves are abstract
  tracers with no concrete sharding, so the wrapper infers the layout
  and gets its own cache entry (the per-leaf sharding cache key);
* ``device_config=None`` (default) is byte-identical legacy behaviour:
  the sharded-wrapper cache stays empty, ``.raw`` is the plain
  non-JIT/non-donating Python function the training drivers
  differentiate through, and the 2-device sharded run reproduces the
  default run's numbers to floating-point round-off.

Run standalone (canonical, deterministic device count)::

    XLA_FLAGS=--xla_force_host_platform_device_count=2 \\
        .venv/bin/python -m pytest \\
        tests/parallel/test_segment_sharding_device_config.py -v

In a full-suite run JAX may already be initialized before this module
imports (the ``setdefault`` below is then a no-op) — the multi-device
tests skip unless >=2 CPU devices are visible, same pattern as
``tests/parallel/test_sharded_step_sharding_tripwire.py``.  The
``device_config=None`` contract tests run on any device count.
"""

import os
import types

# Must be set BEFORE the first jax backend initialization in the process
# (same pattern as tests/parallel/test_sharded_step_sharding_tripwire.py).
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import pytest

N_DEVICES = 2


def _need_devices(n: int):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices "
            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
        )


def _segment_args():
    """Mock build_segment_fn kwargs (reuses the unit-test components)."""
    from tests.unit.test_compiled_segments import _make_segment_fn_args
    return _make_segment_fn_args()


def _make_carry(dev_config=None):
    """Standard tiny SegmentCarry (mirrors the unit-test packing).

    When *dev_config* is supplied, the carry is sharded across the mesh
    with the SAME ``shard_pytree`` the production driver applies at
    ``model_driver.py`` before the segment call — ``build_segment_fn``'s
    sharded wrapper reads the layout off this committed input rather than
    re-deriving one, so an unsharded carry would (correctly) produce an
    unsharded result.
    """
    from legoesm.driver.compiled_segments import pack_carry
    from tests.unit.test_compiled_segments import (
        _make_hydrostatic_state, N_FACES, N, NLEV,
    )
    shape_3d = (N_FACES, N, N, NLEV)
    shape_2d = (N_FACES, N, N)
    carry = pack_carry(
        _make_hydrostatic_state(),
        q_v=jnp.ones(shape_3d) * 0.01,
        q_c=jnp.zeros(shape_3d),
        q_r=jnp.zeros(shape_3d),
        held_dT_rad=jnp.zeros(shape_3d),
        held_sw_net_sfc=jnp.zeros(shape_2d),
        held_lw_net_sfc=jnp.zeros(shape_2d),
        held_sw_up_toa=jnp.zeros(shape_2d),
        held_lw_up_toa=jnp.zeros(shape_2d),
        held_sw_down_toa=jnp.zeros(shape_2d),
        step_index=0,
    )
    if dev_config is not None and dev_config.mesh is not None:
        from legoesm.parallel.mesh import shard_pytree
        carry = shard_pytree(carry, dev_config)
    return carry


def _copy(tree):
    """Leaf-copy a pytree so the original survives buffer donation."""
    return jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, tree)


def _assert_carry_face_sharded(carry, dev_config, *, where: str):
    """Bench tripwire + explicit shard-split check on a SegmentCarry."""
    from scripts.bench.run_levante_gpu_scaling import (
        _assert_expected_sharding,
    )
    # Same policy assertion the scaling benchmark runs.
    _assert_expected_sharding(carry, dev_config, where=where)

    # Belt-and-braces: every face-leading leaf is genuinely split — 2
    # addressable shards of 3 faces each, never a full-globe replica.
    face_leaves = [
        leaf for leaf in jax.tree.leaves(carry)
        if isinstance(leaf, jax.Array) and leaf.ndim >= 1
        and leaf.shape[0] == 6
    ]
    assert face_leaves, f"{where}: carry has no face-leading leaves?"
    for leaf in face_leaves:
        shards = leaf.addressable_shards
        assert len(shards) == N_DEVICES, where
        for shard in shards:
            assert shard.data.shape[0] == 6 // N_DEVICES, where
    # Scalars replicate (shared leaf policy).
    assert carry.step_index.sharding.is_fully_replicated, where


class TestSegmentDeviceConfigSharded:
    """device_config with a live mesh: sharded kernels + cached wrappers."""

    def test_segment_jit_keeps_carry_face_sharded_and_caches_wrapper(self):
        _need_devices(N_DEVICES)
        from legoesm.parallel.mesh import create_device_mesh
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import _make_forcing

        dev_config = create_device_mesh(n_devices=N_DEVICES)
        assert dev_config.mesh is not None
        run_segment = build_segment_fn(
            **_segment_args(), device_config=dev_config,
        )
        forcing = _make_forcing()

        out = run_segment(_make_carry(dev_config), 2, forcing)
        jax.block_until_ready(jax.tree.leaves(out))
        _assert_carry_face_sharded(out, dev_config, where="after segment 1")

        # Second segment: the donated input is the previous output; the
        # SAME cached wrapper must serve it (no rebuild per segment).
        out2 = run_segment(out, 2, forcing)
        jax.block_until_ready(jax.tree.leaves(out2))
        _assert_carry_face_sharded(out2, dev_config, where="after segment 2")
        assert int(out2.step_index) == 4

        cache = run_segment._sharded_jit_cache
        assert len(cache) == 1, (
            f"expected ONE cached sharded wrapper after two identical "
            f"segments, got {len(cache)} — wrapper rebuilt per segment "
            f"(recompile on every call)"
        )

    def test_forcing_treedef_change_gets_own_cache_entry(self):
        """Forcing pytrees vary (None -> array overrides): both must work."""
        _need_devices(N_DEVICES)
        from legoesm.parallel.mesh import create_device_mesh
        from legoesm.driver.compiled_segments import (
            build_segment_fn, pack_forcing,
        )
        from legoesm import constants
        from tests.unit.test_compiled_segments import (
            _make_forcing, N_FACES, N, NLEV,
        )

        dev_config = create_device_mesh(n_devices=N_DEVICES)
        run_segment = build_segment_fn(
            **_segment_args(), device_config=dev_config,
        )

        forcing_plain = _make_forcing()
        forcing_override = pack_forcing(
            sst=jnp.full((N_FACES, N, N), 300.0),
            sic=jnp.zeros((N_FACES, N, N)),
            day_of_year=1.0,
            seconds_of_day=0.0,
            solar_weights=jnp.ones(14),
            s_0=constants.S_0,
            o3_vmr=jnp.zeros((N_FACES, N, N, NLEV)),
            aerosol_od=jnp.zeros((N_FACES, N, N)),
            # Treedef change: optional leaf flips None -> array.
            sfc_T_override=jnp.full((N_FACES, N, N), 290.0),
        )

        out_a = run_segment(_make_carry(dev_config), 1, forcing_plain)
        out_b = run_segment(_make_carry(dev_config), 1, forcing_override)
        jax.block_until_ready(jax.tree.leaves(out_b))
        _assert_carry_face_sharded(out_a, dev_config, where="plain forcing")
        _assert_carry_face_sharded(out_b, dev_config, where="override forcing")
        assert len(run_segment._sharded_jit_cache) == 2, (
            "distinct forcing treedefs must map to distinct cached wrappers"
        )

    def test_raw_is_nondonating_sharded_and_differentiable(self):
        _need_devices(N_DEVICES)
        from legoesm.parallel.mesh import create_device_mesh
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import _make_forcing

        dev_config = create_device_mesh(n_devices=N_DEVICES)
        run_segment = build_segment_fn(
            **_segment_args(), device_config=dev_config,
        )
        forcing = _make_forcing()
        carry = _make_carry(dev_config)

        out = run_segment.raw(carry, 2, forcing)
        jax.block_until_ready(jax.tree.leaves(out))
        _assert_carry_face_sharded(out, dev_config, where="after .raw")

        # NON-donating: the input carry must still be readable after the
        # call (a donated buffer raises on first use).
        assert bool(jnp.isfinite(jnp.sum(carry.T)))

        # Explicit lowering of the cached non-donating sharded wrapper
        # works (AOT path used by inspection/bench tooling).
        (jitted,) = run_segment._sharded_jit_cache.values()
        lowered = jitted.lower(_make_carry(dev_config), 2, forcing)
        assert lowered is not None

        # Still differentiable: non-donation (not non-jit) is the AD
        # requirement — grad-of-jit composes.  Under ``jax.grad`` the
        # carry leaves are abstract tracers (no concrete sharding), so
        # the wrapper infers the layout — a separate cache entry from the
        # concrete sharded call above (see
        # ``test_raw_first_call_under_grad_builds_sharded_wrapper``).
        def loss(T0):
            out_g = run_segment.raw(carry._replace(T=T0), 1, forcing)
            return jnp.sum(out_g.T)

        g = jax.grad(loss)(carry.T)
        assert g.shape == carry.T.shape
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_raw_first_call_under_grad_then_concrete_keeps_both_layouts(self):
        """The FIRST ``.raw`` invocation happens inside ``jax.grad`` —
        the wrapper is built from abstract TRACER leaves, which have no
        concrete sharding, so the jit INFERS the layout (a grad trace
        never imposes a device mesh).  The wrapper must build and the
        grad must be finite (no crash from feeding an abstract-mesh
        sharding into ``in_shardings``).

        A subsequent CONCRETE call with a face-sharded carry reads that
        carry's real ``NamedSharding`` and pins it — a DIFFERENT
        executable (inferred-layout vs pinned-layout), so it gets its own
        cache entry (size 2) and the output is genuinely face-sharded.
        This is the per-leaf-sharding cache key doing its job: the two
        layouts must never collide on one wrapper.
        """
        _need_devices(N_DEVICES)
        from legoesm.parallel.mesh import create_device_mesh
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import _make_forcing

        dev_config = create_device_mesh(n_devices=N_DEVICES)
        run_segment = build_segment_fn(
            **_segment_args(), device_config=dev_config,
        )
        forcing = _make_forcing()
        carry = _make_carry()  # unsharded: leaves become tracers under grad
        assert run_segment._sharded_jit_cache == {}  # genuinely fresh

        def loss(T0):
            out_g = run_segment.raw(carry._replace(T=T0), 1, forcing)
            return jnp.sum(out_g.T)

        # First-ever call: wrapper built under the grad trace (inferred
        # layout — abstract tracers carry no concrete sharding).
        g = jax.grad(loss)(carry.T)
        assert bool(jnp.all(jnp.isfinite(g)))
        assert len(run_segment._sharded_jit_cache) == 1

        # Concrete call with a face-sharded carry: real shardings → new
        # cache entry, and the output is genuinely face-sharded.
        out = run_segment.raw(_make_carry(dev_config), 1, forcing)
        jax.block_until_ready(jax.tree.leaves(out))
        assert len(run_segment._sharded_jit_cache) == 2
        _assert_carry_face_sharded(
            out, dev_config, where="concrete sharded call after grad-built wrapper",
        )

    def test_flat_cell_packed_leaf_sharded_on_cell_axis_is_matched(self):
        """REGRESSION for the reported crash (jobs 8457808/8457809).

        The moist/AMIP ``conv_prog`` is a FLATTENED cell-packed leaf
        ``[n_cells, nlev]`` = ``[6*N*N, nlev]`` (leading dim 6*N*N, not
        6).  The old ``create_output_shardings`` policy keyed on
        ``shape[0] == 6`` and so derived a replicated ``P()`` in_sharding
        for it — but the driver had placed that leaf's cell axis on the
        ``"face"`` mesh axis (``P("face")``), so ``jax.jit`` raised
        "Sharding passed to jit does not match the sharding on the
        respective arg ... float32[..,nlev] ... spec=P() vs spec=P('face',)".

        Here we put exactly such a leaf — a ``[6*N*N, NLEV]`` array
        ``device_put`` onto ``P("face")`` (sharding its cell axis) — into
        the carry's ``conv_prog`` slot (the mock step passes ``conv_prog``
        through unchanged, so its shape is unconstrained) and assert the
        sharded segment MATCHES it: it runs without the mismatch crash and
        the output ``conv_prog`` is still cell-axis-sharded (2 shards of
        ``6*N*N/2`` rows).  The old code would have raised on this call.
        """
        _need_devices(N_DEVICES)
        from jax.sharding import NamedSharding, PartitionSpec
        from legoesm.parallel.mesh import create_device_mesh
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import (
            _make_forcing, N_FACES, N, NLEV,
        )

        dev_config = create_device_mesh(n_devices=N_DEVICES)
        run_segment = build_segment_fn(
            **_segment_args(), device_config=dev_config,
        )
        forcing = _make_forcing()

        # Flat cell-packed conv_prog, cell axis sharded across "face".
        n_cells = N_FACES * N * N
        assert n_cells % N_DEVICES == 0
        conv_prog_flat = jnp.arange(
            n_cells * NLEV, dtype=jnp.float32,
        ).reshape(n_cells, NLEV)
        conv_prog_sharded = jax.device_put(
            conv_prog_flat, NamedSharding(dev_config.mesh, PartitionSpec("face")),
        )
        # Sanity: the input genuinely shards the FLAT cell axis (the
        # layout create_output_shardings misreads as replicated).
        assert conv_prog_sharded.sharding.spec == PartitionSpec("face")
        assert len(conv_prog_sharded.addressable_shards) == N_DEVICES

        carry = _make_carry(dev_config)._replace(conv_prog=conv_prog_sharded)

        # The crash site: with the old policy this raised P() vs P('face').
        out = run_segment(carry, 2, forcing)
        jax.block_until_ready(jax.tree.leaves(out))

        # Output conv_prog preserves the cell-axis sharding (out==in).
        # NB: we deliberately do NOT run ``_assert_carry_face_sharded``
        # here — that tripwire compares against ``create_output_shardings``,
        # whose ``shape[0]==6`` classifier expects this flat ``[n_cells,
        # nlev]`` leaf to be REPLICATED ``P()``.  This whole test exists
        # because that classifier is wrong for cell-packed leaves; the fix
        # matches the leaf's ACTUAL ``P("face")`` layout instead, so we
        # assert that directly.
        assert isinstance(out.conv_prog.sharding, NamedSharding)
        assert out.conv_prog.sharding.spec == PartitionSpec("face")
        shards = out.conv_prog.addressable_shards
        assert len(shards) == N_DEVICES
        for shard in shards:
            assert shard.data.shape == (n_cells // N_DEVICES, NLEV)
        # The (6,N,N,*) dycore leaves stay genuinely face-sharded too
        # (checked directly, not via the create_output_shardings tripwire).
        for name in ("u", "v", "T", "q_v"):
            leaf = getattr(out, name)
            assert isinstance(leaf.sharding, NamedSharding), name
            assert leaf.sharding.spec[0] == "face", (name, leaf.sharding.spec)
            lshards = leaf.addressable_shards
            assert len(lshards) == N_DEVICES, name
            for shard in lshards:
                assert shard.data.shape[0] == N_FACES // N_DEVICES, name

    def test_raw_under_grad_with_committed_sharded_inputs(self):
        """A grad trace ENTERED with committed on-mesh ``NamedSharding``
        inputs (the closure captures a face-sharded carry).

        Under ``jax.grad`` the differentiated leaf becomes a tracer and
        the other captured leaves remain concrete, so the carry is a MIX
        of tracer + committed-sharded arrays.  ``_get_sharded_jit``
        detects the tracer and builds an UNPINNED jit (no ``in_shardings``)
        — pinning from a tracer's abstract sharding would disagree with
        the concrete array the grad transform actually feeds the inner
        jit at execution time and raise "Sharding passed to jit does not
        match the sharding on the respective arg".  The gradient must be
        finite and full-shape (this is the exact regression for that
        crash, seen on CPU job 8458320 before the unpinned-under-trace
        fix).
        """
        _need_devices(N_DEVICES)
        from legoesm.parallel.mesh import create_device_mesh
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import _make_forcing

        dev_config = create_device_mesh(n_devices=N_DEVICES)
        run_segment = build_segment_fn(
            **_segment_args(), device_config=dev_config,
        )
        forcing = _make_forcing()
        carry = _make_carry(dev_config)  # committed, face-sharded on-mesh

        def loss(T0):
            out_g = run_segment.raw(carry._replace(T=T0), 1, forcing)
            return jnp.sum(out_g.T)

        # T0 is itself a committed face-sharded array.
        g = jax.grad(loss)(carry.T)
        jax.block_until_ready(g)
        assert g.shape == carry.T.shape
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0


class TestCollectivePermuteCensus:
    """The bench HLO census regex (no devices needed — pure string fn)."""

    def test_counts_disambiguate_sync_and_async_forms(self):
        from scripts.bench.run_levante_gpu_scaling import (
            _count_collective_permute_ops,
        )
        hlo = "\n".join([
            # sync form: opcode application counted once, result-name
            # mentions ignored
            "%collective-permute.5 = f32[3,24,24]{2,1,0}"
            " collective-permute(f32[3,24,24]{2,1,0} %p),"
            " source_target_pairs={{0,1},{1,0}}",
            "%add.1 = f32[] add(f32[] %collective-permute.5, f32[] %c)",
            # async pair: -start/-done are NOT double-counted as the
            # plain opcode
            "%cps = (f32[3],f32[3]) collective-permute-start(f32[3] %q)",
            "%cpd = f32[3] collective-permute-done((f32[3],f32[3]) %cps)",
            "%cps2 = (f32[3],f32[3]) collective-permute-start(f32[3] %r)",
            "%cpd2 = f32[3] collective-permute-done((f32[3],f32[3]) %cps2)",
        ])
        counts = _count_collective_permute_ops(hlo)
        assert counts["collective-permute"] == 1
        assert counts["collective-permute-start"] == 2
        assert counts["collective-permute-done"] == 2

    def test_zero_on_collective_free_hlo(self):
        from scripts.bench.run_levante_gpu_scaling import (
            _count_collective_permute_ops,
        )
        counts = _count_collective_permute_ops(
            "%add.0 = f32[8] add(f32[8] %a, f32[8] %b)\n"
            "%ag = f32[6,8] all-gather(f32[3,8] %c), dimensions={0}\n"
        )
        assert counts == {
            "collective-permute": 0,
            "collective-permute-start": 0,
            "collective-permute-done": 0,
        }


class TestSegmentDeviceConfigNonePath:
    """device_config=None: byte-identical legacy behaviour (any device count)."""

    def test_default_raw_is_plain_function_and_cache_unused(self):
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import _make_forcing

        run_segment = build_segment_fn(**_segment_args())
        forcing = _make_forcing()

        out = run_segment(_make_carry(), 2, forcing)
        jax.block_until_ready(jax.tree.leaves(out))
        assert int(out.step_index) == 2

        # .raw is the plain non-JIT Python function (training AD
        # contract) — not a jit wrapper (those expose .lower).
        assert isinstance(run_segment.raw, types.FunctionType)
        assert not hasattr(run_segment.raw, "lower")

        # The sharded machinery never engages.
        carry = _make_carry()
        _ = run_segment.raw(carry, 1, forcing)
        assert run_segment._sharded_jit_cache == {}

        # Non-donating: input survives.
        assert bool(jnp.isfinite(jnp.sum(carry.T)))

    def test_default_raw_grad_contract(self):
        """The training contract: jax.grad flows through .raw."""
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import _make_forcing

        run_segment = build_segment_fn(**_segment_args())
        forcing = _make_forcing()
        carry = _make_carry()

        def loss(T0):
            out = run_segment.raw(carry._replace(T=T0), 2, forcing)
            return jnp.sum(out.T)

        g = jax.grad(loss)(carry.T)
        assert g.shape == carry.T.shape
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.max(jnp.abs(g))) > 0.0

    def test_sharded_run_matches_default_run_to_roundoff(self):
        """The 2-device face-sharded run reproduces the single-device run
        to floating-point round-off (2 CPU devices).

        Tolerance, not bit-equality: the default-precision segment runs
        in float32 (``_make_segment_fn_args`` builds an f32 state).  On a
        face-sharded carry the dycore stencils exchange halos across the
        two shards and the cubed-sphere edge/pole handling sums
        contributions in a DIFFERENT ORDER than the single-buffer run, so
        the results differ in the last few f32 ULPs — a benign reduction
        re-association, not a divergence.  A *misclassified* leaf (the
        bug this fix targets — a flat cell-packed field replicated where
        it should shard, or vice-versa) would instead produce a wholly
        wrong field, which the magnitude assertion below catches.  Under
        ``JAX_ENABLE_X64=1`` the same code path runs in f64 and the
        residual collapses to f64 round-off.
        """
        _need_devices(N_DEVICES)
        import numpy as np
        from legoesm.parallel.mesh import create_device_mesh
        from legoesm.driver.compiled_segments import build_segment_fn
        from tests.unit.test_compiled_segments import _make_forcing

        forcing = _make_forcing()
        args = _segment_args()
        dev_config = create_device_mesh(n_devices=N_DEVICES)

        out_default = build_segment_fn(**args).raw(_make_carry(), 3, forcing)
        # Genuinely sharded input — the production scenario the fix
        # targets: the carry is face-sharded upstream and the segment
        # matches that layout (NOT a replicated single-device run).
        out_sharded = build_segment_fn(
            **args, device_config=dev_config,
        ).raw(_make_carry(dev_config), 3, forcing)
        jax.block_until_ready(jax.tree.leaves(out_sharded))

        # Tolerance is keyed on the COMPUTE precision, not the output
        # dtype.  The test state (``_make_hydrostatic_state``) stores the
        # prognostics in float32, so the cubed-sphere stencil math whose
        # summation order differs between the 1-buffer and 2-shard runs is
        # float32 — the reduction-reorder residual is f32-level (~1e-6
        # relative / ~1e-8 absolute on a 1e-2 field, observed) EVEN under
        # ``JAX_ENABLE_X64=1`` (x64 only widens the accumulator scalars,
        # not the f32-stored state arrays, so the output dtype may be f64
        # while the diff stays f32-sized — keying tolerance on the output
        # dtype was wrong, CPU job 8458403).  A wholly-wrong field (a
        # mis-sharded leaf — the bug this fix targets) is orders of
        # magnitude larger and trips the assert.
        rtol, atol = 1e-4, 1e-6
        for name in ("u", "v", "T", "p_s", "q_v", "q_c", "q_r"):
            a = np.asarray(getattr(out_default, name))
            b = np.asarray(getattr(out_sharded, name))
            assert np.allclose(a, b, rtol=rtol, atol=atol), (
                f"sharded vs default mismatch in carry field {name!r} "
                f"beyond f32 round-off: max|Δ|={np.max(np.abs(a - b)):.3e}, "
                f"max|val|={np.max(np.abs(a)):.3e} (a wholly-wrong field "
                f"here means a leaf was mis-sharded, not reduction reorder)"
            )
        assert int(out_default.step_index) == int(out_sharded.step_index) == 3
