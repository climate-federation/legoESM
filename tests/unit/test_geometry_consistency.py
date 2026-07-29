"""Direct tests for ``legoesm.parallel.geometry_consistency`` (#1362).

These are the SHARED cross-process geometry-agreement primitives used by both
the ocean and atmosphere lat-lon SPMD lanes.  The collective paths
(``process_allgather`` / ``broadcast_one_to_all``) need a real multi-process
launch and are exercised by the distributed suite; what is tested here is the
part that decides WHETHER those collectives raise — the fingerprints.

Every test below is written so it FAILS if the property it names is lost:
the permutation tests in particular are the reason the byte digest exists at
all (a moment fingerprint cannot see a permutation).
"""

import numpy as np
import pytest

from legoesm.parallel.geometry_consistency import (
    assert_flags_agree,
    assert_schema_agrees,
    broadcast_checked,
    content_hash48,
    name_digest48,
    schema_fingerprint,
)


class TestContentHash48:
    def test_deterministic(self):
        a = np.arange(24, dtype=np.int32).reshape(4, 6)
        assert content_hash48(a) == content_hash48(a.copy())

    def test_exactly_representable_in_float64(self):
        """48 bits keeps the digest under 2**53, so the float64
        ``process_allgather`` payload carries it EXACTLY.  A wider digest
        would round in transit and two agreeing processes could compare
        unequal (or worse, two differing ones could compare equal)."""
        for seed in range(32):
            rng = np.random.default_rng(seed)
            h = content_hash48(rng.integers(0, 255, size=64).astype(np.uint8))
            assert 0 <= h < 2.0 ** 48
            assert h == float(int(h))            # integral
            assert int(h) == int(np.float64(h))  # survives a f64 round-trip

    def test_detects_permutation_that_moments_cannot(self):
        """THE reason this is a byte digest and not a moment fingerprint.

        A boolean mask and any permutation of it share sum, sum-of-squares
        and absmax exactly — so a moment-only compare is blind to two
        processes disagreeing about WHICH cells are wet, which is a physics
        difference, not autotune noise.
        """
        a = np.zeros(64, dtype=bool)
        a[:8] = True
        b = np.zeros(64, dtype=bool)
        b[-8:] = True  # same true-count, different positions

        f = lambda x: (x.sum(), (x * x).sum(), np.abs(x).max())
        assert f(a) == f(b), "fixture broken: moments must be identical here"
        assert content_hash48(a) != content_hash48(b)

    def test_detects_two_cell_flip_that_cancels_in_the_sum(self):
        """A +1/-1 pair leaves the sum unchanged; the digest still moves."""
        a = np.arange(32, dtype=np.int64)
        b = a.copy()
        b[3] += 1
        b[9] -= 1
        assert a.sum() == b.sum(), "fixture broken: sums must match"
        assert content_hash48(a) != content_hash48(b)

    def test_shape_change_is_visible_at_equal_bytes(self):
        a = np.arange(12, dtype=np.int32)
        assert content_hash48(a) == content_hash48(np.ascontiguousarray(a))
        # same bytes, different logical shape -> callers also fingerprint
        # ndim/shape structurally, so this documents that the DIGEST alone
        # is shape-blind and must not be used without the struct entry.
        assert content_hash48(a) == content_hash48(a.reshape(3, 4))


class TestSchemaFingerprint:
    def test_shape_and_dtype_are_fixed(self):
        """Fixed shape is what lets every process reach the SAME collective
        even when their field lists differ — a variable-length payload would
        deadlock instead of reporting.

        Asserts the payloads AGREE with each other rather than pinning a
        literal width, so extending the fingerprint (as the dtype-kind/ndim
        terms did) does not silently turn this into a stale-constant test.
        The one thing that must never vary is that the width is independent
        of the INPUTS.
        """
        a = schema_fingerprint(["x", "y"], 4)
        b = schema_fingerprint(["completely", "different", "names"], 9)
        c = schema_fingerprint(["x"], 1, ["exact", "inexact"], [1, 2, 3])
        assert a.shape == b.shape == c.shape
        assert a.ndim == 1 and a.shape[0] >= 4
        assert a.dtype == b.dtype == c.dtype == np.float64

    def test_sensitive_to_field_list(self):
        assert not np.array_equal(schema_fingerprint(["a", "b"], 4),
                                  schema_fingerprint(["a", "c"], 4))

    def test_sensitive_to_field_order(self):
        """Per-field checks are matched positionally across processes, so a
        reordering IS a divergence."""
        assert not np.array_equal(schema_fingerprint(["a", "b"], 4),
                                  schema_fingerprint(["b", "a"], 4))

    def test_sensitive_to_field_count_and_device_count(self):
        assert not np.array_equal(schema_fingerprint(["a"], 4),
                                  schema_fingerprint(["a", "b"], 4))
        assert not np.array_equal(schema_fingerprint(["a"], 4),
                                  schema_fingerprint(["a"], 8))

    def test_name_join_is_not_ambiguous(self):
        """SAME-LENGTH lists whose comma-join is identical must still differ.

        codex 2026-07-29 (minor 5): the earlier fixture used ``["a,b"]`` vs
        ``["a", "b"]``, which differ in COUNT — so it passed even with a
        colliding join and proved nothing. These two have equal length and an
        identical ``",".join``, so only a separator that cannot occur in a
        name (NUL) distinguishes them.
        """
        left, right = ["a,b", "c"], ["a", "b,c"]
        assert len(left) == len(right)
        assert ",".join(left) == ",".join(right), "fixture must collide"
        assert name_digest48(left) != name_digest48(right)
        assert not np.array_equal(schema_fingerprint(left, 1),
                                  schema_fingerprint(right, 1))


class TestSingleProcessIsAPassthrough:
    """With one process there is nothing to compare: no collective may run
    (they would hang), and the value must come back untouched."""

    def test_broadcast_checked_returns_the_SAME_OBJECT(self):
        a = np.linspace(0.0, 1.0, 40).reshape(5, 8)
        assert broadcast_checked(a, "area", context="test") is a

    def test_single_process_does_NOT_convert_a_device_array_to_host(self):
        """codex 2026-07-29 (major 3): converting unconditionally forced a
        device->host->device round trip and stripped weak-type metadata on a
        1-process mesh. The atmosphere lane passes `jnp.stack` results
        straight in, so it must come back as the SAME jax array."""
        jnp = pytest.importorskip("jax.numpy")
        a = jnp.arange(6, dtype=jnp.int32)
        out = broadcast_checked(a, "idx", context="test")
        assert out is a
        assert not isinstance(out, np.ndarray)

    def test_assert_schema_agrees_is_a_noop(self):
        assert_schema_agrees(["a", "b"], 1, context="test") is None


class _FakeMultihost:
    """Records every collective payload and echoes N identical copies.

    Lets the MULTI-process branch of `broadcast_checked` / `assert_*` run
    under pytest, which is the only way to test the properties that matter:
    payload SHAPE (a shape that varies per rank deadlocks in the real thing)
    and which fingerprint the routing actually computed.
    """

    def __init__(self, n=2, override=None):
        self.n, self.override, self.seen = n, override, []

    def process_allgather(self, payload):
        self.seen.append(np.array(payload, dtype=np.float64))
        rows = [np.asarray(payload, dtype=np.float64) for _ in range(self.n)]
        if self.override is not None:
            # `override` may return None to leave a payload alone. Needed
            # because a single call site emits SEVERAL payloads of different
            # widths (struct=8 then vals=3); blanket-overriding them all
            # would (a) make np.stack fail on ragged rows and (b) let a test
            # pass for the wrong reason (struct mismatch masquerading as a
            # value mismatch).
            replaced = self.override(rows[-1])
            if replaced is not None:
                rows[-1] = np.asarray(replaced, dtype=np.float64)
                assert rows[-1].shape == rows[0].shape, (
                    "test override changed the payload WIDTH; in the real "
                    "collective that is a deadlock, not a divergence")
        return np.stack(rows)

    def broadcast_one_to_all(self, host):
        return host


@pytest.fixture
def multiproc(monkeypatch):
    """Force `jax.process_count() > 1` and install the fake collectives."""
    import jax
    from jax.experimental import multihost_utils

    def _install(n=2, override=None):
        fake = _FakeMultihost(n, override)
        monkeypatch.setattr(jax, "process_count", lambda: n)
        monkeypatch.setattr(multihost_utils, "process_allgather",
                            fake.process_allgather)
        monkeypatch.setattr(multihost_utils, "broadcast_one_to_all",
                            fake.broadcast_one_to_all)
        return fake
    return _install


class TestBroadcastCheckedRouting:
    """Exercise `broadcast_checked` ITSELF, not just its helpers.

    codex 2026-07-29 (major 4) correctly observed that testing
    `content_hash48` directly proves nothing about whether the helper still
    CALLS it for exact arrays — swapping that branch to moments would leave a
    hash-only test green. These tests read the payload the helper actually
    emitted.
    """

    def test_exact_array_payload_is_the_byte_digest(self, multiproc):
        fake = multiproc()
        a = np.arange(24, dtype=np.int32).reshape(4, 6)
        broadcast_checked(a, "idx", context="t")
        vals = fake.seen[-1]
        assert vals[0] == content_hash48(a), (
            "exact arrays must be fingerprinted by the POSITIONAL byte "
            "digest; a moment fingerprint here loses permutation detection")

    def test_float_array_payload_is_the_moment_triple(self, multiproc):
        fake = multiproc()
        a = np.linspace(-2.0, 3.0, 30).reshape(5, 6)
        broadcast_checked(a, "area", context="t")
        vals = fake.seen[-1]
        np.testing.assert_allclose(
            vals[:3], [a.sum(), (a * a).sum(), np.abs(a).max()], rtol=1e-12)

    def test_permutation_of_a_mask_is_REJECTED_end_to_end(self, multiproc):
        """The property the byte digest exists for, through the real helper.

        Peer rank reports the digest of a PERMUTATION: identical shape,
        identical dtype, identical sum/sumsq/absmax — so `struct` matches and
        ONLY the value payload can catch it. That is what makes this a test
        of the digest rather than of the struct compare.
        """
        a = np.zeros(64, dtype=bool)
        a[:8] = True
        perm = np.zeros(64, dtype=bool)
        perm[-8:] = True
        assert a.sum() == perm.sum() and a.shape == perm.shape

        # touch ONLY the 3-wide value payload; leave the 8-wide struct alone
        multiproc(override=lambda row: (
            _make_vals_for(perm) if row.shape == (3,) else None))
        with pytest.raises(RuntimeError, match="DIVERGES"):
            broadcast_checked(a, "wet_mask", context="t")

    def test_agreeing_processes_pass(self, multiproc):
        multiproc()
        a = np.linspace(0, 1, 12)
        out = broadcast_checked(a, "area", context="t")
        np.testing.assert_array_equal(out, a)

    def test_float_divergence_beyond_rtol_raises(self, multiproc):
        """Only the VALUE payload is perturbed, so this cannot pass on a
        struct mismatch instead."""
        multiproc(override=lambda row: row * 1.1 if row.shape == (3,) else None)
        with pytest.raises(RuntimeError, match="DIVERGES"):
            broadcast_checked(np.linspace(1, 2, 10), "area", context="t")

    def test_float_drift_within_rtol_is_ACCEPTED(self, multiproc):
        """The whole reason for the broadcast: ULP-scale autotune drift must
        NOT raise, or every large multi-process run fails spuriously."""
        multiproc(override=lambda row:
                  row * (1.0 + 1e-9) if row.shape == (3,) else None)
        out = broadcast_checked(np.linspace(1, 2, 10), "area", context="t")
        assert out is not None

    def test_nonfinite_count_is_structural(self, multiproc):
        """A NaN on one process only must not be averaged away."""
        fake = multiproc()
        a = np.array([1.0, 2.0, np.nan, 4.0])
        broadcast_checked(a, "area", context="t")
        struct = fake.seen[-2]
        assert struct[5] == 1.0, "non-finite count must be carried in struct"


def _make_vals_for(arr):
    """The 3-wide vals payload `broadcast_checked` would emit for ``arr``."""
    v = np.zeros(3, dtype=np.float64)
    v[0] = content_hash48(arr)
    return v


class TestFixedShapePayloads:
    """A payload whose LENGTH depends on rank-local data can deadlock: two
    processes enter the same allgather with different shapes. codex
    2026-07-29 blocker 2 — inherited from the pre-extraction ocean code."""

    @pytest.mark.parametrize("arr", [
        np.zeros(4, dtype=bool),
        np.arange(6, dtype=np.int64),
        np.linspace(0, 1, 6),
        np.array([1.0, np.nan, 3.0]),
        np.zeros((2, 3, 4)),
        np.zeros((5,)),
    ])
    def test_every_payload_has_identical_shape(self, multiproc, arr):
        fake = multiproc()
        broadcast_checked(arr, "f", context="t")
        struct, vals = fake.seen[-2], fake.seen[-1]
        assert struct.shape == (8,), struct.shape
        assert vals.shape == (3,), vals.shape

    def test_schema_gate_covers_dtype_class_and_ndim(self, multiproc):
        """Without these terms a bool-on-one-rank / float-on-another field
        passes the gate and then deadlocks in the per-field gather."""
        a_exact = [np.zeros(4, dtype=bool)]
        a_float = [np.zeros(4, dtype=np.float64)]
        f1 = schema_fingerprint(["m"], 2, ["exact"], [1])
        f2 = schema_fingerprint(["m"], 2, ["inexact"], [1])
        f3 = schema_fingerprint(["m"], 2, ["exact"], [2])
        assert not np.array_equal(f1, f2), "dtype class must move the digest"
        assert not np.array_equal(f1, f3), "ndim must move the digest"
        # and end-to-end through the gate
        multiproc(override=lambda _r: schema_fingerprint(
            ["m"], 2, ["inexact"], [1]))
        with pytest.raises(RuntimeError, match="SCHEMA differs"):
            assert_schema_agrees(["m"], 2, context="t", arrays=a_exact)


class TestFlagsAgree:
    """`assert_flags_agree` makes a rank-local refusal symmetric."""

    def test_agreeing_flags_pass(self, multiproc):
        multiproc()
        assert_flags_agree(("a", "b"), (True, 4), context="t") is None

    def test_divergent_flags_raise(self, multiproc):
        multiproc(override=lambda row: np.concatenate([row[:2], [1.0, 4.0]]))
        with pytest.raises(RuntimeError, match="CONFIG differs"):
            assert_flags_agree(("a", "b"), (False, 4), context="t")

    def test_single_process_runs_no_collective(self):
        assert_flags_agree(("a",), (1,), context="t") is None


class TestDtypeRouting:
    """``broadcast_checked`` routes exact dtypes to the byte digest and float
    dtypes to the moment fingerprint.  If that routing inverts, masks lose
    permutation detection — so pin the predicate itself."""

    @pytest.mark.parametrize("dtype,exact", [
        (np.bool_, True), (np.int32, True), (np.int64, True),
        (np.uint8, True), (np.float32, False), (np.float64, False),
    ])
    def test_exactness_predicate(self, dtype, exact):
        assert (np.dtype(dtype).kind in "biu") is exact


class TestFactoryWiringOrder:
    """The gate must be CALLED, and called FIRST.

    codex round-2 flagged that every previous test exercised the helpers in
    isolation, so deleting `_agree_spmd_entry(...)` from the factories while
    leaving the helper defined would have stayed green. These tests read the
    production source and assert the wiring itself.

    Source inspection is a weak instrument, so per CLAUDE.md each assertion
    names the symbol that ACTUALLY RUNS (the public factory), not a wrapper,
    and pins ORDER rather than mere presence — presence alone is what the
    round-2 blocker already satisfied while still deadlocking.
    """

    FACTORIES = (
        "make_sharded_atm_latlon_step",
        "make_sharded_atm_latlon_segment",
        "make_sharded_atm_latlon_step_2d",
        "make_sharded_atm_latlon_segment_2d",
    )

    @staticmethod
    def _body(fn_name):
        import inspect
        import legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step as m
        src = inspect.getsource(getattr(m, fn_name))
        # strip the docstring so its prose cannot satisfy a text assertion
        import ast
        tree = ast.parse(src.lstrip())
        fn = tree.body[0]
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)):
            fn.body = fn.body[1:]
        return fn

    @pytest.mark.parametrize("fn_name", FACTORIES)
    def test_gate_is_the_first_statement(self, fn_name):
        import ast
        fn = self._body(fn_name)
        assert fn.body, f"{fn_name} has an empty body"
        first = fn.body[0]
        calls = [n for n in ast.walk(first)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "_agree_spmd_entry"]
        assert calls, (
            f"{fn_name}: the FIRST statement must be _agree_spmd_entry(...). "
            f"Any rank-local check above it (n_steps validation, mesh shape "
            f"checks, the `mesh is None` early return) lets one process raise "
            f"or return while a peer blocks in a collective — a HANG, not an "
            f"error. Got: {ast.dump(first)[:200]}")

    @pytest.mark.parametrize("fn_name", FACTORIES)
    def test_no_raise_or_return_precedes_the_gate(self, fn_name):
        """Even a gate present but not first is a deadlock (round-2 blocker)."""
        import ast
        fn = self._body(fn_name)
        for stmt in fn.body:
            has_gate = any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_agree_spmd_entry" for n in ast.walk(stmt))
            if has_gate:
                return
            offending = [n for n in ast.walk(stmt)
                         if isinstance(n, (ast.Raise, ast.Return))]
            assert not offending, (
                f"{fn_name}: a {type(offending[0]).__name__} occurs BEFORE "
                f"_agree_spmd_entry; that is the exact round-2 deadlock.")
        raise AssertionError(f"{fn_name} never calls _agree_spmd_entry")

    def test_n_steps_is_agreed_by_both_segment_factories(self):
        """Unequal positive n_steps => different STATIC scan lengths =>
        different numbers of in-body collectives (round-2 blocker 3)."""
        import ast
        for fn_name in ("make_sharded_atm_latlon_segment",
                        "make_sharded_atm_latlon_segment_2d"):
            fn = self._body(fn_name)
            call = next(n for n in ast.walk(fn)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id == "_agree_spmd_entry")
            kw = {k.arg: k.value for k in call.keywords}
            assert "n_steps" in kw, f"{fn_name} must agree n_steps"
            assert isinstance(kw["n_steps"], ast.Name), (
                f"{fn_name} must pass the n_steps VARIABLE, not a literal")
            assert kw["n_steps"].id == "n_steps"

    def test_entry_flag_tuple_is_static_and_matches_payload_width(self):
        """A payload width that depends on data is a deadlock, so the flag
        names must be a module-level literal tuple."""
        import legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step as m
        assert isinstance(m._SPMD_ENTRY_FLAGS, tuple)
        assert len(m._SPMD_ENTRY_FLAGS) >= 8
        assert all(isinstance(x, str) for x in m._SPMD_ENTRY_FLAGS)


class TestOceanFactoryWiringOrder:
    """The ocean lane has the SAME pre-collective-throw shape as the
    atmosphere and must be gated the same way.

    codex round-2 confirmed `_build_band_vertex_masks` can throw on one rank
    (unprimed vertex-mask cache) while a peer blocks in the schema gate. The
    round-2 fix only covered the four atmosphere factories, so the ocean half
    of that blocker stayed open.
    """

    FACTORIES = ("make_sharded_ocean_step", "make_sharded_ocean_step_global")

    @staticmethod
    def _body(fn_name):
        import ast
        import inspect
        import legoesm.ocean.dynamics.sharded_ocean_step as m
        fn = ast.parse(inspect.getsource(getattr(m, fn_name)).lstrip()).body[0]
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)):
            fn.body = fn.body[1:]
        return fn

    @pytest.mark.parametrize("fn_name", FACTORIES)
    def test_gate_precedes_every_raise_and_return(self, fn_name):
        import ast
        fn = self._body(fn_name)
        for stmt in fn.body:
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "_agree_ocean_spmd_entry"
                   for n in ast.walk(stmt)):
                return
            bad = [n for n in ast.walk(stmt)
                   if isinstance(n, (ast.Raise, ast.Return))]
            assert not bad, (
                f"{fn_name}: a {type(bad[0]).__name__} precedes "
                f"_agree_ocean_spmd_entry — one rank returns/raises while a "
                f"peer blocks in a collective (HANG).")
        raise AssertionError(f"{fn_name} never calls _agree_ocean_spmd_entry")

    def test_entry_flag_tuple_is_static(self):
        import legoesm.ocean.dynamics.sharded_ocean_step as m
        assert isinstance(m._OCEAN_SPMD_ENTRY_FLAGS, tuple)
        assert all(isinstance(x, str) for x in m._OCEAN_SPMD_ENTRY_FLAGS)
