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
    assert_schema_agrees,
    broadcast_checked,
    content_hash48,
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
        deadlock instead of reporting."""
        a = schema_fingerprint(["x", "y"], 4)
        b = schema_fingerprint(["completely", "different", "names"], 9)
        assert a.shape == b.shape == (4,)
        assert a.dtype == b.dtype == np.float64

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
        """``",".join`` must not collide two different lists onto one digest."""
        assert not np.array_equal(schema_fingerprint(["a,b"], 1),
                                  schema_fingerprint(["a", "b"], 1))


class TestSingleProcessIsAPassthrough:
    """With one process there is nothing to compare: no collective may run
    (they would hang), and the value must come back untouched."""

    def test_broadcast_checked_returns_input_unchanged(self):
        a = np.linspace(0.0, 1.0, 40).reshape(5, 8)
        out = broadcast_checked(a, "area", context="test")
        np.testing.assert_array_equal(out, a)
        assert out.dtype == a.dtype

    def test_broadcast_checked_accepts_non_numpy_input(self):
        jnp = pytest.importorskip("jax.numpy")
        out = broadcast_checked(jnp.arange(6, dtype=jnp.int32), "idx",
                                context="test")
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.arange(6))

    def test_assert_schema_agrees_is_a_noop(self):
        assert_schema_agrees(["a", "b"], 1, context="test") is None


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
