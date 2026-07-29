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
    FLAG_ABSENT,
    FLAG_NEGATIVE,
    FLAG_OUT_OF_RANGE,
    FLAG_UNCOERCIBLE,
    assert_flags_agree,
    assert_schema_agrees,
    broadcast_checked,
    coerce_bool,
    coerce_count,
    config_digest48,
    content_hash48,
    name_digest48,
    safe_repr,
    schema_fingerprint,
    tree_schema_digest48,
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




class TestCoerceCount:
    """`coerce_count` is the reason the entry gates can no longer die while
    ASSEMBLING their collective payload (round-3 blocker 3), and — after
    round-4 blocker 1 — the reason they cannot silently ACCEPT a value that
    diverges downstream."""

    def test_good_values_pass_through_exactly(self):
        for v in (0, 1, 7, 2 ** 40, np.int64(5)):
            payload, problem = coerce_count(v)
            assert problem is None
            assert payload == float(int(v))

    def test_none_maps_to_the_absent_sentinel(self):
        assert coerce_count(None) == (FLAG_ABSENT, None)
        assert coerce_count(None, absent=0.0) == (0.0, None)

    def test_absent_cannot_alias_a_real_or_rejected_value(self):
        """codex round-4, blocker 1.  FLAG_ABSENT used to be ``-1.0``, so
        ``segment_steps=None`` on one rank and ``-1`` on another produced the
        SAME payload entry: the gate agreed, then one rank raised while its
        peer built the next collective."""
        absent, _ = coerce_count(None)
        neg, problem = coerce_count(-1)
        assert problem is not None, "-1 must be REJECTED, not encoded"
        assert neg == FLAG_NEGATIVE
        assert absent != neg, "absent and negative must not alias"

    @pytest.mark.parametrize("bad", [3.5, 3.0, "3", "bad", object(),
                                     float("nan"), float("inf"), 3.5 + 0j,
                                     np.array([1, 2]), np.array([3])])
    def test_non_integer_types_are_REFUSED(self, bad):
        """`int(3.5) == 3` made a rank carrying 3.5 indistinguishable from a
        peer carrying 3; the payloads agreed and then ``range(3.5)`` blew up on
        one rank alone.  A size-1 array coerces while a size-2 array raises, so
        accepting arrays makes the gate depend on rank-local shape."""
        payload, problem = coerce_count(bad)
        assert problem is not None, f"{bad!r} must be refused"
        assert payload == FLAG_UNCOERCIBLE

    def test_35_and_3_do_not_collide(self):
        """The decisive property, stated directly."""
        assert coerce_count(3.5)[0] != coerce_count(3)[0]

    def test_bool_is_refused_not_encoded_as_1(self):
        for v in (True, False, np.bool_(True)):
            payload, problem = coerce_count(v)
            assert problem is not None, "a bool is not a step count"
            assert payload == FLAG_UNCOERCIBLE

    def test_NEVER_raises_even_for_a_hostile_object(self):
        """A raise inside the payload build lands BEFORE the collective and
        hangs every peer, so no input may escape as an exception."""
        class Hostile:
            def __int__(self):
                raise RuntimeError("boom")

            def __repr__(self):
                raise RuntimeError("boom")

            def __bool__(self):
                raise RuntimeError("boom")

        payload, problem = coerce_count(Hostile())     # must not raise
        assert problem is not None and isinstance(payload, float)
        payload_b, problem_b = coerce_bool(Hostile())  # must not raise
        assert problem_b is not None and isinstance(payload_b, float)
        assert "unrepresentable" in safe_repr(Hostile())

    def test_value_above_2_53_is_refused_not_silently_aliased(self):
        """codex round-3 minor 2: above 2**53 two DIFFERENT counts map to the
        same float64 payload entry."""
        big = 2 ** 53 + 1
        assert float(big) == float(2 ** 53), "fixture: these MUST alias"
        payload, problem = coerce_count(big)
        assert payload == FLAG_OUT_OF_RANGE
        assert problem is not None and "2**53" in problem
        assert coerce_count(2 ** 53)[1] is None      # the boundary is fine

    def test_sentinels_are_all_distinct_and_unreachable(self):
        sentinels = (FLAG_ABSENT, FLAG_UNCOERCIBLE, FLAG_OUT_OF_RANGE,
                     FLAG_NEGATIVE)
        assert len(set(sentinels)) == len(sentinels)
        for s in sentinels:
            assert s < 0, "valid counts are >= 0, so a sentinel must be < 0"
            # and no accepted count can ever reach one
            assert coerce_count(abs(int(s)))[0] != s or \
                coerce_count(abs(int(s)))[1] is not None


class TestCoerceBool:
    """codex round-4, blocker 2: `bool(value)` RAISES on a multi-element array
    ("truth value ... is ambiguous"), and inside a gate that is a
    pre-collective throw, i.e. a hang."""

    def test_real_bools_encode(self):
        assert coerce_bool(True) == (1.0, None)
        assert coerce_bool(False) == (0.0, None)
        assert coerce_bool(np.bool_(True)) == (1.0, None)

    def test_none_is_the_absent_sentinel(self):
        assert coerce_bool(None) == (FLAG_ABSENT, None)

    @pytest.mark.parametrize("bad", [np.array([True, False]), 1, 0, "yes",
                                     [], object()])
    def test_non_bool_is_refused_without_raising(self, bad):
        payload, problem = coerce_bool(bad)
        assert problem is not None
        assert payload == FLAG_UNCOERCIBLE

    def test_multi_element_array_would_have_raised_under_bool(self):
        """Fixture proving the hazard is real, not hypothetical."""
        arr = np.array([True, False])
        with pytest.raises(ValueError):
            bool(arr)
        assert coerce_bool(arr)[1] is not None    # ... but this does not


class TestConfigDigest:
    """codex round-4, blocker 3.  Hand-listing `fold`/`anchor`/`polar` agreed
    only the fields somebody remembered; `fix_mass` (which gates a global-area
    psum), the integrator choice and `fix_moisture` were all missing.  ONE
    digest over every static scalar closes the class."""

    @staticmethod
    def _cfg(**kw):
        import types
        base = dict(fix_mass=False, outer_integrator="forward_euler",
                    use_polar_filter=False, dt=900.0, name="x")
        base.update(kw)
        ns = types.SimpleNamespace(**base)
        ns._fields = tuple(base)          # duck-type a NamedTuple
        return ns

    def test_a_feature_gating_bool_moves_the_digest(self):
        assert (config_digest48(self._cfg(fix_mass=True))
                != config_digest48(self._cfg(fix_mass=False)))

    def test_a_scheme_literal_moves_the_digest(self):
        assert (config_digest48(self._cfg(outer_integrator="ab2"))
                != config_digest48(self._cfg(outer_integrator="forward_euler")))

    def test_identical_configs_agree(self):
        assert config_digest48(self._cfg()) == config_digest48(self._cfg())

    def test_never_raises_on_a_hostile_or_plain_object(self):
        class Hostile:
            _fields = ("x",)

            @property
            def x(self):
                raise RuntimeError("boom")

        assert isinstance(config_digest48(Hostile()), float)
        assert isinstance(config_digest48(None), float)
        assert isinstance(config_digest48(object()), float)


class TestTreeSchemaDigest:
    """codex round-4, blocker 5: one replication runs PER LEAF, so the leaf
    schedule is rank-local data."""

    def test_extra_leaf_moves_the_digest(self):
        a = {"T": np.zeros((4, 8)), "q": np.zeros((4, 8))}
        b = {"T": np.zeros((4, 8)), "q": np.zeros((4, 8)),
             "o3": np.zeros((4, 8))}
        assert tree_schema_digest48(a) != tree_schema_digest48(b)

    def test_different_key_moves_the_digest(self):
        a = {"T": np.zeros((4, 8))}
        b = {"S": np.zeros((4, 8))}
        assert tree_schema_digest48(a) != tree_schema_digest48(b)

    def test_shape_and_dtype_move_the_digest(self):
        base = {"T": np.zeros((4, 8), dtype=np.float64)}
        assert (tree_schema_digest48(base)
                != tree_schema_digest48({"T": np.zeros((4, 9))}))
        assert (tree_schema_digest48(base)
                != tree_schema_digest48(
                    {"T": np.zeros((4, 8), dtype=np.float32)}))

    def test_insertion_order_alone_does_NOT_false_positive(self):
        """Dict key paths are canonicalised, so a different insertion order
        with the SAME key set must still agree — otherwise the gate would
        refuse healthy runs."""
        a = {"T": np.zeros(3), "q": np.zeros(3)}
        b = {"q": np.zeros(3), "T": np.zeros(3)}
        assert tree_schema_digest48(a) == tree_schema_digest48(b)

    def test_optional_none_field_moves_the_digest(self):
        assert (tree_schema_digest48({"a": np.zeros(3), "b": None})
                != tree_schema_digest48({"a": np.zeros(3),
                                         "b": np.zeros(3)}))

    def test_never_raises(self):
        assert isinstance(tree_schema_digest48(object()), float)



class TestSchemaDescriptorTolerance:
    """codex round-3 minor 3: `assert_schema_agrees` read `a.dtype`, which
    dies with a bare AttributeError on a plain Python scalar — and dying THERE
    is a pre-collective rank-local raise, i.e. a hang."""

    def test_plain_python_scalar_does_not_explode(self, multiproc):
        multiproc()
        # must not raise AttributeError; agreeing peers => no divergence
        assert_schema_agrees(["s"], 2, context="t", arrays=[1.5]) is None

    def test_scalar_is_classified_like_its_numpy_dtype(self, multiproc):
        fake = multiproc()
        assert_schema_agrees(["s"], 2, context="t", arrays=[1.5])
        from_scalar = fake.seen[-1]
        fake2 = multiproc()
        assert_schema_agrees(["s"], 2, context="t",
                             arrays=[np.float64(1.5)])
        np.testing.assert_array_equal(from_scalar, fake2.seen[-1])

    def test_bool_scalar_still_routes_to_the_exact_class(self, multiproc):
        fake = multiproc()
        assert_schema_agrees(["s"], 2, context="t", arrays=[True])
        exact_like = fake.seen[-1]
        fake2 = multiproc()
        assert_schema_agrees(["s"], 2, context="t", arrays=[1.5])
        assert not np.array_equal(exact_like, fake2.seen[-1]), (
            "a bool scalar and a float scalar must NOT share a schema digest "
            "— broadcast_checked routes them to different payload layouts")


# ---------------------------------------------------------------------------
# Entry-gate wiring: a STRICT, non-spoofable source check
# ---------------------------------------------------------------------------
# codex round-3 (MAJOR) refuted the round-2 instrument: it used
# `ast.walk(first_statement)` and accepted ANY nested occurrence, so
#
#     if mesh is None:
#         _agree_spmd_entry(...)     # gate on ONE branch only
#         return serial_step
#
# passed every assertion while still deadlocking.  The checker below demands
# the gate be a DIRECT top-level statement of the function body (an
# `ast.Expr` whose value is the call), which that shape does not satisfy.
# `test_checker_rejects_*` below are the NON-VACUITY self-tests: they feed the
# checker synthetically mutated trees and assert it says NO — and, for the
# spoof shape, that the OLD instrument said YES.


def _module_tree(mod):
    import ast
    import inspect
    return ast.parse(inspect.getsource(mod))


def _top_level_functions(tree) -> dict:
    import ast
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _public_entry_names(tree) -> set:
    """Public names a caller can reach: top-level defs AND module-level
    ALIASES (``need_rad_and_time = _need_rad_and_time``).

    Aliases matter because an alias is a public entry point that a
    FunctionDef-only scan does not see — exactly the kind of blind spot that
    let five paths stay unguarded through round 2.
    """
    import ast
    names = set(_top_level_functions(tree))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            names.update(t.id for t in node.targets
                         if isinstance(t, ast.Name))
    return {n for n in names if not n.startswith("_")}


def _strip_docstring(fn):
    """Return ``fn.body`` without a leading docstring.

    Prose must never be able to satisfy a wiring assertion (CLAUDE.md: a test
    that inspects source must name the symbol that RUNS).
    """
    import ast
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def _direct_gate_index(fn, gate_name):
    """Index of the DIRECT top-level statement that is exactly
    ``gate_name(...)``; ``None`` if there is no such statement.

    Deliberately NOT `ast.walk`: a call nested inside an ``If``/``Try``/loop
    runs on only some paths and is exactly the spoof codex round-3 flagged.
    """
    import ast
    for i, stmt in enumerate(_strip_docstring(fn)):
        if (isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == gate_name):
            return i
    return None


def _gate_call(fn, gate_name):
    import ast
    for stmt in _strip_docstring(fn):
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == gate_name):
            return stmt.value
    raise AssertionError(f"no direct {gate_name}(...) statement")


def _parse_fn(src):
    import ast
    import textwrap
    return ast.parse(textwrap.dedent(src)).body[0]


class TestGateCheckerIsNonVacuous:
    """The instrument itself, proven to reject what it claims to reject.

    Per CLAUDE.md a source-inspecting test is UNTRUSTED until shown to fail
    when the feature is removed.  These run that mutation IN MEMORY, so the
    proof is cheap and permanent rather than a one-off manual edit.
    """

    GOOD = """
        def f(model, mesh):
            '''doc'''
            _agree_spmd_entry(model, mesh, where="f")
            if mesh is None:
                return None
            return 1
    """

    def test_accepts_the_correct_shape(self):
        assert _direct_gate_index(_parse_fn(self.GOOD),
                                  "_agree_spmd_entry") == 0

    def test_rejects_a_gate_nested_in_a_branch(self):
        """THE round-3 MAJOR: a gate on one branch only.  The old
        `ast.walk(first_statement)` instrument ACCEPTED this."""
        import ast
        mutated = _parse_fn("""
            def f(model, mesh):
                if mesh is None:
                    _agree_spmd_entry(model, mesh, where="f")
                    return None
                return 1
        """)
        assert _direct_gate_index(mutated, "_agree_spmd_entry") is None
        old_instrument_accepts = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_agree_spmd_entry"
            for n in ast.walk(mutated.body[0]))
        assert old_instrument_accepts, (
            "fixture broken: this shape must be one the WEAK checker passed, "
            "otherwise the strengthening proves nothing")

    def test_rejects_a_gate_nested_in_a_try(self):
        mutated = _parse_fn("""
            def f(model, mesh):
                try:
                    _agree_spmd_entry(model, mesh, where="f")
                except Exception:
                    pass
                return 1
        """)
        assert _direct_gate_index(mutated, "_agree_spmd_entry") is None

    def test_rejects_outright_removal(self):
        mutated = _parse_fn("""
            def f(model, mesh):
                '''doc mentioning _agree_spmd_entry so prose cannot pass'''
                return 1
        """)
        assert _direct_gate_index(mutated, "_agree_spmd_entry") is None

    def test_rejects_a_gate_that_is_not_first(self):
        mutated = _parse_fn("""
            def f(model, mesh):
                if mesh is None:
                    return None
                _agree_spmd_entry(model, mesh, where="f")
                return 1
        """)
        assert _direct_gate_index(mutated, "_agree_spmd_entry") == 1


# (symbol -> gate that must be its FIRST direct top-level statement).
#
# ROUND-4 RETRACTION.  The round-3 allow-list claimed the SCATTER direction ran
# no collective.  That is FALSE for every entry that uses a raw
# ``jax.device_put`` of a full global array onto a cross-process
# ``NamedSharding``: the repo's own ``latlon_spmd.shard_leaf`` docstring records
# that XLA services exactly that with an ALL-GATHER.  Those entries are now
# GATED, and `test_exempt_entries_contain_no_mesh_device_put` mechanically
# refuses to let the claim be re-asserted.
_ATM_GATED = {
    "make_sharded_atm_latlon_step": "_agree_spmd_entry",
    "make_sharded_atm_latlon_segment": "_agree_spmd_entry",
    "make_sharded_atm_latlon_step_2d": "_agree_spmd_entry",
    "make_sharded_atm_latlon_segment_2d": "_agree_spmd_entry",
    # codex round-3 blocker 4: these validate and LOOP on n_steps themselves,
    # outside any factory, and are the production path via model_driver.
    "run_atm_latlon_spmd_segment": "_agree_spmd_entry",
    "run_atm_latlon_spmd": "_agree_spmd_entry",
    # round 4: GATHER runs a real cross-process replication (replicate_leaf ->
    # jit identity with replicated out_shardings); SCATTER runs one placement
    # per leaf, and the 2-D one is a raw cross-process device_put.
    "gather_state_atm_latlon": "_agree_mesh_entry",
    "gather_state_atm_latlon_2d": "_agree_mesh_entry",
    "gather_atm_latlon_to_hydrostatic": "_agree_mesh_entry",
    "shard_state_atm_latlon": "_agree_mesh_entry",
    "shard_state_atm_latlon_2d": "_agree_mesh_entry",
    "shard_hydrostatic_to_atm_latlon": "_agree_mesh_entry",
}

# SHRINK-ONLY.  Every entry is a CLAIM about what the function does, and
# round 4 proved that a plausible-sounding reason hides a real defect, so the
# reasons below are now also checked MECHANICALLY (see
# TestExemptionsAreVerifiedNotAsserted).
_ATM_UNGATED = {
    "lat_spec":
        "pure: returns P() from arr.ndim. It CAN raise (AttributeError) on a "
        "non-array, but it enters no collective and is called only with real "
        "leaves, so a raise here cannot strand a peer in a gather",
    "tile_spec": "pure: same contract as lat_spec",
    "atm_grid_array_field_names": "pure host introspection of grid._fields",
    "atm_latlon_geometry_bytes":
        "pure host byte accounting; it calls build_band_grids_atm, which can "
        "raise on indivisibility, but neither enters a collective",
    "unroll_to_dtype_fixed_point":
        "trace-time only (jax.eval_shape probe); no host collective",
    "state_finite_scalar":
        "emits an in-graph psum when called with an axis inside a shard_map; "
        "it is a TRACED helper, never a host entry point that runs a "
        "collective by itself",
    "build_band_grids_atm":
        "pure host geometry. It is public and CAN be called directly (the "
        "byte-accounting helper does), so its divisibility ValueError is only "
        "symmetric because the inputs it raises on (grid.n_lat, n_devices) "
        "are themselves agreed at whichever gated entry the caller used; it "
        "enters no collective of its own",
    "build_tile_grids_atm_2d":
        "pure host geometry; same contract as build_band_grids_atm",
    "build_sharded_held_suarez_state_atm_latlon":
        "band-LOCAL construction via jax.make_array_from_callback (#1100): "
        "each addressable shard is built from local data, no gather, and no "
        "replicated put",
}

_OCEAN_GATED = {
    "make_sharded_ocean_step": "_agree_ocean_spmd_entry",
    "make_sharded_ocean_step_global": "_agree_ocean_spmd_entry",
    "gather_state_latlon": "_agree_ocean_mesh_entry",
    "shard_state_latlon": "_agree_ocean_mesh_entry",
    "shard_forcing_latlon": "_agree_ocean_mesh_entry",
    "shard_forcing_stack_latlon": "_agree_ocean_mesh_entry",
}

_OCEAN_UNGATED = {
    "build_band_grids":
        "pure host geometry; enters no collective of its own (same contract "
        "as build_band_grids_atm)",
    "append_vface_wall_row":
        "pure array op (concatenate a zero row); no collective, no branch",
}

_OPSPLIT_GATED = {
    "make_sharded_operator_split_step": "_agree_opsplit_spmd_entry",
    "shard_operator_split_carry": "_agree_opsplit_mesh_entry",
    "shard_operator_split_forcing": "_agree_opsplit_mesh_entry",
}

_OPSPLIT_UNGATED = {
    "need_rad_and_time":
        "module ALIAS of the pure _need_rad_and_time cadence helper "
        "(jnp arithmetic on step_index); no collective, no raise",
}


def _atm_module():
    import legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step as m
    return m


def _ocean_module():
    import legoesm.ocean.dynamics.sharded_ocean_step as m
    return m


def _opsplit_module():
    import legoesm.driver.sharded_operator_split_step as m
    return m


_LANES = [
    ("atm", _atm_module, _ATM_GATED, _ATM_UNGATED),
    ("ocean", _ocean_module, _OCEAN_GATED, _OCEAN_UNGATED),
    ("opsplit", _opsplit_module, _OPSPLIT_GATED, _OPSPLIT_UNGATED),
]


class TestEveryPublicEntryPointIsClassified:
    """Fix the CLASS, not the instance.

    Round 1 patched individual refusal functions; round 2 found five more
    unguarded paths; round 3 found two more (the run wrappers).  The only way
    off that treadmill is a check that FAILS when a NEW public symbol appears
    ungated, instead of a list of the sites someone happened to look at.

    A new public function in any of these three lat-band SPMD lanes must
    either be gated or be classified in the shrink-only ``_*_UNGATED`` map
    with a reason that is TRUE of the code.
    """

    @pytest.mark.parametrize("lane,mod_fn,gated,ungated", _LANES,
                             ids=[l[0] for l in _LANES])
    def test_no_unclassified_public_entry_point(self, lane, mod_fn, gated,
                                                ungated):
        public = _public_entry_names(_module_tree(mod_fn()))
        classified = set(gated) | set(ungated)
        missing = public - classified
        assert not missing, (
            f"{lane}: public SPMD entry point(s) {sorted(missing)} are "
            f"neither gated nor classified. Every public entry of a lat-band "
            f"SPMD lane must agree its rank-local inputs before any "
            f"collective, or say in the allow-list why it needs no gate.")
        stale = classified - public
        assert not stale, (
            f"{lane}: {sorted(stale)} are listed but no longer public — the "
            f"allow-list must not carry dead entries that hide a rename")

    @pytest.mark.parametrize("lane,mod_fn,gated,ungated", _LANES,
                             ids=[l[0] for l in _LANES])
    def test_every_ungated_entry_has_a_real_reason(self, lane, mod_fn, gated,
                                                   ungated):
        for name, reason in ungated.items():
            assert isinstance(reason, str) and len(reason) > 20, (
                f"{lane}.{name}: an allow-list entry needs a REASON, and "
                f"every reason string is a claim that must be verified in "
                f"code before it is written")


class TestExemptionsAreVerifiedNotAsserted:
    """codex round-4 MAJOR: checking that a reason is a LONG STRING accepts a
    FALSE reason, and round 4 proved that is not hypothetical — the round-3
    list asserted "SCATTER, therefore no collective" for five entries that
    each `device_put` a full global array onto a cross-process
    `NamedSharding`, which the repo's own `shard_leaf` docstring records as an
    ALL-GATHER.  This test makes the specific claim MECHANICAL, so it cannot
    be re-asserted in prose.
    """

    @staticmethod
    def _mesh_device_put_calls(fn):
        """`jax.device_put(x, NamedSharding(...))` calls directly in ``fn``.

        Matches the SHAPE of the hazard (a put whose target sharding is built
        from a mesh), not a name, so renaming the local does not evade it.
        """
        import ast
        hits = []
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "device_put"):
                continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id == "NamedSharding"):
                    hits.append(node)
                    break
        return hits

    @pytest.mark.parametrize("lane,mod_fn,gated,ungated", _LANES,
                             ids=[l[0] for l in _LANES])
    def test_exempt_entries_contain_no_mesh_device_put(self, lane, mod_fn,
                                                       gated, ungated):
        fns = _top_level_functions(_module_tree(mod_fn()))
        for name in ungated:
            fn = fns.get(name)
            if fn is None:                    # a module alias, not a def
                continue
            hits = self._mesh_device_put_calls(fn)
            assert not hits, (
                f"{lane}.{name} is on the UNGATED allow-list but contains a "
                f"jax.device_put(..., NamedSharding(...)): under multi-process "
                f"that is serviced by an ALL-GATHER (see "
                f"latlon_spmd.shard_leaf), so the exemption reason is FALSE. "
                f"Gate it or migrate it to shard_leaf.")

    def test_the_detector_is_not_vacuous(self):
        """It must FIND the pattern it claims to reject."""
        fn = _parse_fn("""
            def f(state, mesh):
                return jax.device_put(state, NamedSharding(mesh, P()))
        """)
        assert self._mesh_device_put_calls(fn), (
            "fixture: the detector must match the hazard shape, else "
            "test_exempt_entries_contain_no_mesh_device_put passes vacuously")
        clean = _parse_fn("""
            def g(state):
                return state + 1
        """)
        assert not self._mesh_device_put_calls(clean)


class TestReturnedClosuresAreGated:
    """codex round-4, blocker 7 + MAJOR.

    The factories are gated, but the CLOSURES they return are public entry
    points too — a user holds ``step = make_...(...)`` and calls it — and they
    ran rank-local refusals (``refuse_unthreaded_stateful_physics``, the 2-D
    ``NotImplementedError``, ``_validate_forcing_layout``) BEFORE entering
    their collective program.  The top-level enumerator could not see them.
    """

    # (module, factory, returned-closure name, gate)
    CLOSURES = [
        ("atm", "make_sharded_atm_latlon_step", "sharded_step",
         "_agree_spmd_call"),
        ("atm", "make_sharded_atm_latlon_step_2d", "sharded_step",
         "_agree_spmd_call"),
        ("atm", "make_sharded_atm_latlon_segment", "segment",
         "_agree_spmd_call"),
        ("atm", "make_sharded_atm_latlon_segment_2d", "segment",
         "_agree_spmd_call"),
        ("ocean", "make_sharded_ocean_step", "sharded_step",
         "_agree_ocean_spmd_call"),
        ("ocean", "make_sharded_ocean_step_global", "sharded_step_global",
         "_agree_ocean_spmd_call"),
        ("opsplit", "make_sharded_operator_split_step", "sharded_split_step",
         "_agree_opsplit_mesh_entry"),
    ]

    _MODULES = {"atm": _atm_module, "ocean": _ocean_module,
                "opsplit": _opsplit_module}

    @pytest.mark.parametrize("lane,factory,closure,gate", CLOSURES,
                             ids=[f"{c[0]}.{c[1]}.{c[2]}" for c in CLOSURES])
    def test_closure_gate_is_its_first_direct_statement(self, lane, factory,
                                                        closure, gate):
        import ast
        fns = _top_level_functions(_module_tree(self._MODULES[lane]()))
        outer = fns[factory]
        inner = [n for n in ast.walk(outer)
                 if isinstance(n, ast.FunctionDef) and n.name == closure]
        assert inner, f"{factory} defines no closure named {closure}"
        # The SHARDED closure is the one that is returned; take the last
        # definition, which is the multi-device body (an early serial
        # fallback may share the name in a `mesh is None` branch).
        idx = _direct_gate_index(inner[-1], gate)
        assert idx == 0, (
            f"{lane}.{factory}.{closure}: {gate}(...) must be its FIRST "
            f"DIRECT statement (got index {idx}). The rank-local refusals "
            f"below it — stateful-physics refusal, forcing-layout validation, "
            f"the 2-D carry NotImplementedError — otherwise raise on one rank "
            f"while a peer enters the shard_map: a HANG.")

    @pytest.mark.parametrize("lane,factory,closure,gate", CLOSURES,
                             ids=[f"{c[0]}.{c[1]}.{c[2]}" for c in CLOSURES])
    def test_closure_is_actually_returned(self, lane, factory, closure, gate):
        """Guards against the gate being attached to a dead nested def."""
        import ast
        fns = _top_level_functions(_module_tree(self._MODULES[lane]()))
        returned = {n.value.id for n in ast.walk(fns[factory])
                    if isinstance(n, ast.Return)
                    and isinstance(n.value, ast.Name)}
        assert closure in returned, (
            f"{factory} does not return {closure}; the gate would be on a "
            f"function nobody calls")


class TestFactoryWiringOrder:
    """The gate must be CALLED, called FIRST, and called UNCONDITIONALLY.

    Strengthened after codex round-3 (MAJOR): the round-2 version used
    `ast.walk(first_statement)` and returned on the first nested hit, so a
    gate inside `if mesh is None:` satisfied it.  See
    :class:`TestGateCheckerIsNonVacuous` for the in-memory mutation proof that
    the current checker rejects that shape.
    """

    @pytest.mark.parametrize("fn_name", sorted(_ATM_GATED))
    def test_gate_is_the_first_direct_statement(self, fn_name):
        fns = _top_level_functions(_module_tree(_atm_module()))
        idx = _direct_gate_index(fns[fn_name], _ATM_GATED[fn_name])
        assert idx == 0, (
            f"{fn_name}: {_ATM_GATED[fn_name]}(...) must be the FIRST DIRECT "
            f"statement of the body (got index {idx}). A rank-local check "
            f"above it — n_steps validation, mesh shape checks, the "
            f"`mesh is None` early return — lets one process raise or return "
            f"while a peer blocks in a collective: a HANG, not an error. A "
            f"gate nested inside an `if` runs on only one path and is the "
            f"same defect.")

    @pytest.mark.parametrize("fn_name", sorted(_OCEAN_GATED))
    def test_ocean_gate_is_the_first_direct_statement(self, fn_name):
        fns = _top_level_functions(_module_tree(_ocean_module()))
        idx = _direct_gate_index(fns[fn_name], _OCEAN_GATED[fn_name])
        assert idx == 0, f"{fn_name}: gate must be first direct statement"

    @pytest.mark.parametrize("fn_name", sorted(_OPSPLIT_GATED))
    def test_opsplit_gate_is_the_first_direct_statement(self, fn_name):
        fns = _top_level_functions(_module_tree(_opsplit_module()))
        idx = _direct_gate_index(fns[fn_name], _OPSPLIT_GATED[fn_name])
        assert idx == 0, f"{fn_name}: gate must be first direct statement"

    def test_loop_count_inputs_are_passed_as_VARIABLES(self):
        """Unequal loop counts => different numbers of SPMD steps.

        Each of these functions must agree the loop-count argument it itself
        consumes (round-2 blocker 3 for the factories; round-3 blocker 4 for
        the run wrappers, which loop OUTSIDE any factory).
        """
        import ast
        expect = {
            "make_sharded_atm_latlon_segment": {"n_steps": "n_steps"},
            "make_sharded_atm_latlon_segment_2d": {"n_steps": "n_steps"},
            "run_atm_latlon_spmd_segment": {"n_steps": "n_steps"},
            "run_atm_latlon_spmd": {"n_steps": "n_steps",
                                    "segment_steps": "segment_steps",
                                    "compiled_segments": "compiled_segments"},
        }
        fns = _top_level_functions(_module_tree(_atm_module()))
        for fn_name, wanted in expect.items():
            call = _gate_call(fns[fn_name], "_agree_spmd_entry")
            kw = {k.arg: k.value for k in call.keywords}
            for arg, var in wanted.items():
                assert arg in kw, f"{fn_name} must agree {arg}"
                assert isinstance(kw[arg], ast.Name), (
                    f"{fn_name} must pass the {arg} VARIABLE, not a literal "
                    f"— a literal agrees nothing")
                assert kw[arg].id == var


# --- runtime (not source) checks on the gate payload ------------------------

class _StubGrid:
    def __init__(self, n_lat=8, n_lon=16, fold=None):
        self.n_lat, self.n_lon, self.fold = n_lat, n_lon, fold


class _StubConfig:
    anchor_mass_to_initial = False
    use_polar_filter = False


class _StubModel:
    def __init__(self, **grid_kw):
        self.grid = _StubGrid(**grid_kw)
        self.config = _StubConfig()


class _StubStatics:
    """Duck-typed ``_SplitStepStatics``: only the fields the gate digests."""

    _fields = ("dt", "fix_moisture", "qv_smooth_coeff", "owned_mask")

    def __init__(self, dt=900.0, fix_moisture=False, qv_smooth_coeff=0.0,
                 owned_mask=None):
        self.dt = dt
        self.fix_moisture = fix_moisture
        self.qv_smooth_coeff = qv_smooth_coeff
        self.owned_mask = owned_mask


class _StubMesh:
    """Only what the entry gates read: axis_names, shape, devices.size."""

    def __init__(self, axis_names=("lat",), sizes=(2,)):
        self.axis_names = tuple(axis_names)
        self.shape = dict(zip(self.axis_names, sizes))
        self.devices = np.zeros(sizes)


@pytest.fixture
def capture_flags(monkeypatch):
    """Capture the (names, values) the gate actually hands the collective."""
    def _install(module):
        seen = {}

        def _fake(names, values, *, context):
            seen["names"], seen["values"] = names, tuple(values)
            seen["context"] = context
        monkeypatch.setattr(module, "assert_flags_agree", _fake)
        return seen
    return _install


class TestEntryPayloadShape:
    def test_flag_names_and_values_have_the_SAME_length(self, capture_flags):
        """codex round-3 (MAJOR): the old test only asserted
        ``len(_SPMD_ENTRY_FLAGS) >= 8``, which cannot notice a value added
        without a name (or a name without a value).  This reads the payload
        the gate ACTUALLY builds at runtime and compares the two lengths."""
        m = _atm_module()
        seen = capture_flags(m)
        m._agree_spmd_entry(_StubModel(), _StubMesh(), n_steps=4, where="t")
        assert seen["names"] is m._SPMD_ENTRY_FLAGS, (
            "the gate must pass the module-level STATIC tuple, not a locally "
            "built one whose length could depend on data")
        assert len(seen["values"]) == len(m._SPMD_ENTRY_FLAGS)
        assert all(isinstance(v, float) for v in seen["values"])

    def test_the_length_check_is_not_vacuous(self):
        """Non-vacuity: the assertion above compares two INDEPENDENT lengths,
        so a mismatched pair must fail it."""
        names, values = ("a", "b", "c"), (1.0, 2.0)
        with pytest.raises(AssertionError):
            assert len(values) == len(names)

    @pytest.mark.parametrize("kwargs", [
        {},
        {"n_steps": 3},
        {"n_steps": 3, "segment_steps": 1, "compiled_segments": True,
         "has_physics_fn": True, "has_on_segment": False,
         "has_phys_state": None, "shard_geometry": True},
    ])
    def test_payload_width_is_INDEPENDENT_of_which_flags_are_supplied(
            self, capture_flags, kwargs):
        """A width that varies per call site is a deadlock: two processes
        entering the same allgather with different shapes."""
        m = _atm_module()
        seen = capture_flags(m)
        m._agree_spmd_entry(_StubModel(), _StubMesh(), where="t", **kwargs)
        assert len(seen["values"]) == len(m._SPMD_ENTRY_FLAGS)

    def test_mesh_None_keeps_the_same_width(self, capture_flags):
        m = _atm_module()
        seen = capture_flags(m)
        m._agree_spmd_entry(_StubModel(), None, where="t")
        assert len(seen["values"]) == len(m._SPMD_ENTRY_FLAGS)

    def test_ocean_and_opsplit_payload_widths_match_their_name_tuples(
            self, capture_flags):
        om = _ocean_module()
        seen = capture_flags(om)
        om._agree_ocean_spmd_entry(_StubModel(), _StubMesh(), where="t")
        assert seen["names"] is om._OCEAN_SPMD_ENTRY_FLAGS
        assert len(seen["values"]) == len(om._OCEAN_SPMD_ENTRY_FLAGS)

        cm = _opsplit_module()
        seen2 = capture_flags(cm)
        model = _StubModel()
        model._polar_mask = None
        cm._agree_opsplit_spmd_entry(model, _StubMesh(), _StubStatics(),
                                     fix_mass=True, rad_update_steps=4,
                                     ghg_keys=("co2",), where="t")
        assert seen2["names"] is cm._OPSPLIT_SPMD_ENTRY_FLAGS
        assert len(seen2["values"]) == len(cm._OPSPLIT_SPMD_ENTRY_FLAGS)


class TestAxisOrderIsAgreed:
    """codex round-3 BLOCKER 2.

    A ``(lat, lon)`` mesh of shape ``(2, 3)`` and a peer's ``(lon, lat)`` mesh
    of shape ``(3, 2)`` have the same axis COUNT and the same per-name sizes,
    so every count-based flag agreed. The gate passed; then ``_check_2d_mesh``
    rejected the swapped peer while the valid one walked into the geometry
    collective — a hang.
    """

    def test_swapped_axis_ORDER_changes_the_payload(self, capture_flags):
        m = _atm_module()
        seen = capture_flags(m)
        m._agree_spmd_entry(_StubModel(), _StubMesh(("lat", "lon"), (2, 3)),
                            where="t")
        ordered = seen["values"]
        seen2 = capture_flags(m)
        m._agree_spmd_entry(_StubModel(), _StubMesh(("lon", "lat"), (3, 2)),
                            where="t")
        swapped = seen2["values"]
        # the confound the old payload had: these agree on count and on the
        # per-NAME sizes, so only an ORDER-sensitive term can separate them
        names = m._SPMD_ENTRY_FLAGS
        for f in ("n_axes", "p_lat", "p_lon"):
            i = names.index(f)
            assert ordered[i] == swapped[i], (
                f"fixture broken: {f} must MATCH, else this test could pass "
                f"without any order sensitivity")
        assert ordered != swapped, (
            "the entry payload must fold in the ORDERED axis-name tuple")
        i = names.index("axis_names")
        assert ordered[i] != swapped[i]

    def test_ocean_gate_is_axis_order_sensitive_too(self, capture_flags):
        om = _ocean_module()
        seen = capture_flags(om)
        om._agree_ocean_spmd_entry(_StubModel(),
                                   _StubMesh(("lat", "lon"), (2, 3)),
                                   where="t")
        a = seen["values"]
        seen2 = capture_flags(om)
        om._agree_ocean_spmd_entry(_StubModel(),
                                   _StubMesh(("lon", "lat"), (3, 2)),
                                   where="t")
        assert a != seen2["values"]


class TestGateNeverRaisesBeforeItsCollective:
    """codex round-3 BLOCKER 3.

    ``int(n_steps)`` used to run while BUILDING the payload, so a rank passed
    ``"bad"`` died there while its peer blocked in ``process_allgather``.  The
    decisive assertion in each test below is that the collective RAN
    (``fake.seen`` non-empty) before the exception — a restored inline
    ``int()`` still raises ValueError, but with nothing gathered.
    """

    @pytest.mark.parametrize("bad", ["bad", float("nan"), object()])
    def test_uncoercible_n_steps(self, multiproc, bad):
        m = _atm_module()
        fake = multiproc()
        with pytest.raises(ValueError, match="n_steps"):
            m._agree_spmd_entry(_StubModel(), _StubMesh(), n_steps=bad,
                                where="ctx")
        assert fake.seen, (
            "the entry collective must run BEFORE the refusal, or one rank "
            "dies while its peers block forever")

    def test_out_of_range_n_steps(self, multiproc):
        m = _atm_module()
        fake = multiproc()
        with pytest.raises(ValueError, match="2\\*\\*53"):
            m._agree_spmd_entry(_StubModel(), _StubMesh(), n_steps=2 ** 60,
                                where="ctx")
        assert fake.seen

    def test_message_names_the_bad_value_and_the_context(self, multiproc):
        m = _atm_module()
        multiproc()
        with pytest.raises(ValueError) as e:
            m._agree_spmd_entry(_StubModel(), _StubMesh(), n_steps="bad",
                                where="run_atm_latlon_spmd")
        assert "'bad'" in str(e.value) and "run_atm_latlon_spmd" in str(e.value)

    def test_bad_segment_steps_also_deferred(self, multiproc):
        m = _atm_module()
        fake = multiproc()
        with pytest.raises(ValueError, match="segment_steps"):
            m._agree_spmd_entry(_StubModel(), _StubMesh(), n_steps=4,
                                segment_steps="two", where="ctx")
        assert fake.seen

    def test_bad_grid_dim_is_deferred_in_BOTH_lanes(self, multiproc):
        m = _atm_module()
        fake = multiproc()
        with pytest.raises(ValueError, match="grid.n_lat"):
            m._agree_spmd_entry(_StubModel(n_lat="eight"), _StubMesh(),
                                where="ctx")
        assert fake.seen

        om = _ocean_module()
        fake2 = multiproc()
        with pytest.raises(ValueError, match="grid.n_lat"):
            om._agree_ocean_spmd_entry(_StubModel(n_lat="eight"),
                                       _StubMesh(), where="ctx")
        assert fake2.seen

    def test_a_GOOD_value_raises_nothing(self, multiproc):
        m = _atm_module()
        multiproc()
        m._agree_spmd_entry(_StubModel(), _StubMesh(), n_steps=6,
                            segment_steps=2, where="ctx")


class TestOceanFactoryWiringOrder:
    """The ocean lane has the SAME pre-collective-throw shape as the
    atmosphere and must be gated the same way.

    codex round-2 confirmed `_build_band_vertex_masks` can throw on one rank
    (unprimed vertex-mask cache) while a peer blocks in the schema gate.
    Round-3 asked whether the ocean fix had actually landed; it had, in
    539c81a82, and these assertions pin it.
    """

    @pytest.mark.parametrize("fn_name", sorted(_OCEAN_GATED))
    def test_gate_precedes_every_raise_and_return(self, fn_name):
        import ast
        fns = _top_level_functions(_module_tree(_ocean_module()))
        body = _strip_docstring(fns[fn_name])
        idx = _direct_gate_index(fns[fn_name], _OCEAN_GATED[fn_name])
        assert idx is not None, f"{fn_name} never calls its gate directly"
        for stmt in body[:idx]:
            bad = [n for n in ast.walk(stmt)
                   if isinstance(n, (ast.Raise, ast.Return))]
            assert not bad, (
                f"{fn_name}: a {type(bad[0]).__name__} precedes the gate — "
                f"one rank returns/raises while a peer blocks (HANG).")

    def test_entry_flag_tuples_are_static_literals(self):
        for mod_fn, attr in ((_ocean_module, "_OCEAN_SPMD_ENTRY_FLAGS"),
                             (_ocean_module, "_OCEAN_MESH_ENTRY_FLAGS"),
                             (_ocean_module, "_OCEAN_CALL_ENTRY_FLAGS"),
                             (_atm_module, "_CALL_ENTRY_FLAGS"),
                             (_opsplit_module, "_OPSPLIT_MESH_ENTRY_FLAGS"),
                             (_atm_module, "_SPMD_ENTRY_FLAGS"),
                             (_atm_module, "_MESH_ENTRY_FLAGS"),
                             (_opsplit_module, "_OPSPLIT_SPMD_ENTRY_FLAGS")):
            t = getattr(mod_fn(), attr)
            assert isinstance(t, tuple) and t
            assert all(isinstance(x, str) for x in t)
            assert len(set(t)) == len(t), f"{attr} has duplicate flag names"


class TestOperatorSplitLaneIsGuarded:
    """Round 4, found by enumerating the class rather than the reported sites.

    ``make_sharded_operator_split_step`` is the THIRD lat-band SPMD lane. It
    reuses ``build_band_grids_atm`` and hands the per-process-recomputed stack
    to a REPLICATED ``device_put`` — the exact #1362 defect — and it had
    neither the guarded broadcast nor an entry gate.
    """

    def test_geometry_goes_through_the_guarded_broadcast(self):
        import ast
        fns = _top_level_functions(_module_tree(_opsplit_module()))
        fn = fns["make_sharded_operator_split_step"]
        called = {n.func.id for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "broadcast_checked" in called, (
            "the per-process band-geometry stack must be agreed + broadcast "
            "before the replicated device_put, or #1362 reproduces here")
        assert "assert_schema_agrees" in called, (
            "the polar-mask fields are CONDITIONAL on model._polar_mask, so "
            "the field LIST itself can differ across processes — that must "
            "fail in the fixed-shape schema gate, not desynchronise the "
            "per-field gathers")

    def test_no_replicated_device_put_of_unchecked_geometry(self):
        """The broadcast must actually WRAP the value that is put, not sit
        beside it."""
        import ast
        fns = _top_level_functions(_module_tree(_opsplit_module()))
        fn = fns["make_sharded_operator_split_step"]
        puts = [n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "device_put"]
        assert puts, "fixture: this lane must still device_put its geometry"
        guarded = [p for p in puts
                   if any(isinstance(c, ast.Call)
                          and isinstance(c.func, ast.Name)
                          and c.func.id == "broadcast_checked"
                          for c in ast.walk(p))]
        assert guarded, (
            "every replicated geometry device_put must take a "
            "broadcast_checked(...) value as its argument")
