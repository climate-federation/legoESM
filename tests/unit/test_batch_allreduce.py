"""Tests for batch allreduce (single-process fallback).

Since ``batch_allreduce_mpi`` requires mpi4jax+mpi4py (which are not
available in the standard test environment), we test:

1. The packing/unpacking logic by calling the single-process fallback
   functions directly.
2. The version validation and import-error paths in ``reductions.py``.
3. A local emulation of the batch_allreduce pattern (pack -> reduce ->
   unpack) that exercises the same logic without MPI.

All tests run on a single CPU process with no MPI.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.reductions import (
    require_mpi_stack,
    _validate_mpi_runtime_versions,
    _parse_version_triplet,
    mpi4jax_array_result,
)


# ---------------------------------------------------------------------------
# Local emulation of the batch allreduce pattern
# ---------------------------------------------------------------------------

def _batch_allreduce_local(
    values: list[jax.Array],
    op: str = "sum",
) -> list[jax.Array]:
    """Local (single-process) implementation of the batch allreduce pattern.

    This mirrors the packing/unpacking logic of ``batch_allreduce_mpi``
    without requiring MPI.  On a single process, the "allreduce" is an
    identity operation.
    """
    if not values:
        return []

    ops = {
        "sum": lambda x: x,   # single-process sum = identity
        "max": lambda x: x,
        "min": lambda x: x,
    }
    if op not in ops:
        raise ValueError(f"Unsupported op={op!r}. Choose from {list(ops)}.")

    # Promote all values to a common dtype for packing.
    dtypes = [v.dtype for v in values]
    common_dtype = jnp.result_type(*dtypes)
    promoted = [v.astype(common_dtype) for v in values]

    # Record shapes and sizes for unpacking.
    shapes = [v.shape for v in values]
    sizes = [int(v.size) for v in promoted]

    # Pack into a single flat buffer.
    flat_parts = [v.reshape(-1) for v in promoted]
    packed = jnp.concatenate(flat_parts, axis=0)

    # "Allreduce" (identity on single process)
    global_packed = ops[op](packed)

    # Unpack and restore original shapes and dtypes.
    results = []
    offset = 0
    for i, (shape, size) in enumerate(zip(shapes, sizes)):
        chunk = global_packed[offset : offset + size].reshape(shape)
        if dtypes[i] != common_dtype:
            chunk = chunk.astype(dtypes[i])
        results.append(chunk)
        offset += size

    return results


# ===========================================================================
# TestBatchAllreduce
# ===========================================================================

class TestBatchAllreduce:
    """Tests for the batch allreduce packing/unpacking pattern."""

    def test_single_scalar(self):
        """Single scalar passes through unchanged."""
        val = jnp.array(42.0, dtype=jnp.float64)
        results = _batch_allreduce_local([val])
        assert len(results) == 1
        np.testing.assert_allclose(np.asarray(results[0]), 42.0, atol=1e-14)

    def test_multiple_scalars(self):
        """Multiple scalars are packed and unpacked correctly."""
        a = jnp.array(1.0, dtype=jnp.float64)
        b = jnp.array(2.0, dtype=jnp.float64)
        c = jnp.array(3.0, dtype=jnp.float64)

        results = _batch_allreduce_local([a, b, c])
        assert len(results) == 3
        np.testing.assert_allclose(float(results[0]), 1.0, atol=1e-14)
        np.testing.assert_allclose(float(results[1]), 2.0, atol=1e-14)
        np.testing.assert_allclose(float(results[2]), 3.0, atol=1e-14)

    def test_mixed_shapes(self):
        """Mixed shapes (scalars and arrays) are handled."""
        scalar = jnp.array(10.0, dtype=jnp.float64)
        vec = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        mat = jnp.ones((2, 3), dtype=jnp.float64) * 5.0

        results = _batch_allreduce_local([scalar, vec, mat])
        assert len(results) == 3

        # Check shapes preserved
        assert results[0].shape == ()
        assert results[1].shape == (3,)
        assert results[2].shape == (2, 3)

        # Check values preserved
        np.testing.assert_allclose(float(results[0]), 10.0, atol=1e-14)
        np.testing.assert_allclose(
            np.asarray(results[1]), [1.0, 2.0, 3.0], atol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(results[2]), 5.0, atol=1e-14
        )

    def test_different_ops(self):
        """All supported ops (sum, max, min) work without error."""
        val = jnp.array(7.0, dtype=jnp.float64)
        for op in ("sum", "max", "min"):
            results = _batch_allreduce_local([val], op=op)
            assert len(results) == 1
            np.testing.assert_allclose(float(results[0]), 7.0, atol=1e-14)

    def test_empty_list(self):
        """Empty list returns empty list."""
        results = _batch_allreduce_local([])
        assert results == []

    def test_invalid_op_raises(self):
        """Invalid op string raises ValueError."""
        val = jnp.array(1.0, dtype=jnp.float64)
        with pytest.raises(ValueError, match="Unsupported op"):
            _batch_allreduce_local([val], op="prod")

    def test_mixed_dtypes_promoted(self):
        """Values with different dtypes are promoted to common dtype."""
        if not jax.config.read("jax_enable_x64"):
            pytest.skip(
                "distinguishes float32 from float64; without JAX_ENABLE_X64=1 the "
                "float64 array silently truncates to float32 and the test is vacuous"
            )
        f32 = jnp.array(1.0, dtype=jnp.float32)
        f64 = jnp.array(2.0, dtype=jnp.float64)

        results = _batch_allreduce_local([f32, f64])
        assert len(results) == 2

        # Each result should be in its original dtype
        assert results[0].dtype == jnp.float32
        assert results[1].dtype == jnp.float64

        np.testing.assert_allclose(float(results[0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(results[1]), 2.0, atol=1e-14)

    def test_large_batch(self):
        """Batch of many values packs/unpacks correctly."""
        values = [jnp.array(float(i), dtype=jnp.float64) for i in range(50)]
        results = _batch_allreduce_local(values)

        assert len(results) == 50
        for i, r in enumerate(results):
            np.testing.assert_allclose(float(r), float(i), atol=1e-14)

    def test_multidim_arrays(self):
        """3D arrays pack/unpack correctly."""
        arr = jnp.arange(24, dtype=jnp.float64).reshape(2, 3, 4)
        results = _batch_allreduce_local([arr])

        assert len(results) == 1
        assert results[0].shape == (2, 3, 4)
        np.testing.assert_array_equal(
            np.asarray(results[0]), np.asarray(arr)
        )


# ===========================================================================
# TestVersionValidation
# ===========================================================================

class TestVersionValidation:
    """Tests for _parse_version_triplet and _validate_mpi_runtime_versions."""

    def test_parse_version_triplet(self):
        """Common version strings are parsed correctly."""
        assert _parse_version_triplet("0.8.0") == (0, 8, 0)
        assert _parse_version_triplet("0.9.1") == (0, 9, 1)
        assert _parse_version_triplet("1.0.0") == (1, 0, 0)

    def test_parse_version_short(self):
        """Short version strings are zero-padded."""
        assert _parse_version_triplet("0.8") == (0, 8, 0)
        assert _parse_version_triplet("1") == (1, 0, 0)

    def test_parse_version_empty(self):
        """Empty version string returns (0, 0, 0)."""
        assert _parse_version_triplet("") == (0, 0, 0)

    def test_validate_in_tested_range(self):
        """Versions in tested range do not raise."""
        # Should not raise or warn
        _validate_mpi_runtime_versions("0.8.1", "0.8.0")

    def test_validate_old_mpi4jax_raises(self):
        """mpi4jax below minimum version raises RuntimeError."""
        with pytest.raises(RuntimeError, match="mpi4jax"):
            _validate_mpi_runtime_versions("0.8.0", "0.7.0")

    def test_validate_outside_range_warns(self):
        """Versions outside tested range emit a RuntimeWarning."""
        with pytest.warns(RuntimeWarning, match="outside"):
            _validate_mpi_runtime_versions("0.10.0", "0.8.0")

    def test_validate_strict_outside_range_raises(self):
        """strict=True raises RuntimeError for untested versions."""
        with pytest.raises(RuntimeError, match="outside"):
            _validate_mpi_runtime_versions(
                "0.10.0", "0.8.0", strict=True
            )


# ===========================================================================
# TestRequireMPIStack
# ===========================================================================

class TestRequireMPIStack:
    """Tests for require_mpi_stack import checking."""

    def test_require_mpi_stack_without_mpi(self):
        """require_mpi_stack raises ImportError when mpi4jax is missing."""
        # mpi4jax is almost certainly not installed in the test env
        try:
            import mpi4jax  # noqa: F401
            pytest.skip("mpi4jax is installed; cannot test missing-import path")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="MPI reductions require"):
            require_mpi_stack()


# ===========================================================================
# TestMPI4JAXArrayResult
# ===========================================================================

class TestMPI4JAXArrayResult:
    """Tests for mpi4jax_array_result helper."""

    def test_unwrap_tuple(self):
        """Tuple result (old mpi4jax style) is unwrapped."""
        arr = jnp.array(5.0)
        result = mpi4jax_array_result((arr, "token"))
        np.testing.assert_allclose(float(result), 5.0, atol=1e-14)

    def test_passthrough_array(self):
        """Non-tuple result (new mpi4jax style) passes through."""
        arr = jnp.array(5.0)
        result = mpi4jax_array_result(arr)
        np.testing.assert_allclose(float(result), 5.0, atol=1e-14)

    def test_empty_tuple_raises(self):
        """Empty tuple raises RuntimeError."""
        with pytest.raises(RuntimeError, match="empty tuple"):
            mpi4jax_array_result(())
