"""Tests for MPI runtime version-guard behaviour.

These tests exercise :func:`legoesm.parallel.reductions._validate_mpi_runtime_versions`
directly — no MPI stack required.
"""

import warnings

import pytest

from legoesm.parallel.reductions import (
    _parse_version_triplet,
    _validate_mpi_runtime_versions,
    _format_range,
    _TESTED_JAX_MIN,
    _TESTED_JAX_MAX_EXCL,
    _TESTED_MPI4JAX_MIN,
    _TESTED_MPI4JAX_MAX_EXCL,
)


# -----------------------------------------------------------------------
# _parse_version_triplet
# -----------------------------------------------------------------------

class TestParseVersion:
    def test_simple(self):
        assert _parse_version_triplet("0.8.1") == (0, 8, 1)

    def test_major_minor_only(self):
        assert _parse_version_triplet("0.9") == (0, 9, 0)

    def test_four_part(self):
        assert _parse_version_triplet("0.9.0.1") == (0, 9, 0)

    def test_empty(self):
        assert _parse_version_triplet("") == (0, 0, 0)

    def test_dev_suffix(self):
        assert _parse_version_triplet("0.8.0.dev123") == (0, 8, 0)


# -----------------------------------------------------------------------
# _format_range
# -----------------------------------------------------------------------

class TestFormatRange:
    def test_basic(self):
        s = _format_range((0, 8, 0), (0, 9, 0))
        assert s == ">=0.8.0, <0.9.0"


# -----------------------------------------------------------------------
# _validate_mpi_runtime_versions
# -----------------------------------------------------------------------

class TestValidate:
    """Test the three outcomes: pass, warn, hard-fail."""

    def test_within_tested_range_passes_silently(self):
        """Versions inside the tested range produce no warning or error."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → exception
            _validate_mpi_runtime_versions("0.8.5", "0.8.0")

    def test_old_mpi4jax_hard_fails(self):
        """mpi4jax < 0.8 must raise RuntimeError regardless of strict."""
        with pytest.raises(RuntimeError, match="incompatible token semantics"):
            _validate_mpi_runtime_versions("0.8.0", "0.7.9")

    def test_old_mpi4jax_error_contains_fix(self):
        """The hard-error message must contain a remediation command."""
        with pytest.raises(RuntimeError, match="pip install"):
            _validate_mpi_runtime_versions("0.8.0", "0.7.0")

    def test_jax_outside_range_warns(self):
        """JAX outside tested range produces a RuntimeWarning."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.7.0", "0.8.0")

    def test_mpi4jax_above_range_warns(self):
        """mpi4jax above tested range produces a RuntimeWarning."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.8.0", "0.10.0")

    def test_warning_contains_fix(self):
        """Warning message must point to remediation."""
        with pytest.warns(RuntimeWarning, match="LEGOESM_MPI_STRICT_COMPAT"):
            _validate_mpi_runtime_versions("0.7.0", "0.8.0")

    def test_strict_mode_raises(self):
        """strict=True upgrades the warning to RuntimeError."""
        with pytest.raises(RuntimeError, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.7.0", "0.8.0", strict=True)

    def test_strict_with_mpi4jax_above(self):
        """strict=True with mpi4jax above range also raises."""
        with pytest.raises(RuntimeError, match="mpi4jax==0.10.0"):
            _validate_mpi_runtime_versions("0.8.0", "0.10.0", strict=True)

    # --- FFI generation (the SECOND compatible regime; iter 333) -------------
    @pytest.mark.parametrize("mpi4jax_v", ["0.9.0", "0.9.0.post1", "0.9.5"])
    def test_ffi_generation_passes_silently(self, mpi4jax_v):
        """jax 0.10.x PAIRED WITH mpi4jax 0.9.x (the FFI-based line) is verified-working
        (the distributed compare-reanalysis suite passes on jax 0.10.0 + mpi4jax
        0.9.0.post1) and must NOT warn — it is a compatible generation, not 'outside the
        tested range'. Pinning jax<0.10 (the legacy advice) would BREAK this stack."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning → exception
            _validate_mpi_runtime_versions("0.10.0", mpi4jax_v)

    def test_cross_pairing_ffi_jax_legacy_mpi4jax_warns(self):
        """jax 0.10 (FFI-era) with mpi4jax 0.8 (legacy custom-call, removed in jax 0.10)
        is the genuinely-incompatible CROSS pairing — it must still warn, not be silently
        accepted by the FFI short-circuit."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.10.0", "0.8.0")

    def test_cross_pairing_legacy_jax_ffi_mpi4jax_warns(self):
        """jax 0.8 (legacy) with mpi4jax 0.9 (FFI, needs jax>=0.10) is the other
        incompatible CROSS pairing — must still warn."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.8.0", "0.9.0")

    def test_future_jax_with_ffi_mpi4jax_warns_conservatively(self):
        """A jax beyond the verified FFI minor (0.11) warns again until re-verified — the
        FFI acceptance is capped at the tested versions' next minor (the safe direction)."""
        with pytest.warns(RuntimeWarning, match="outside legoESM's tested MPI range"):
            _validate_mpi_runtime_versions("0.11.0", "0.9.0")


# -----------------------------------------------------------------------
# Consistency: constants match pyproject.toml
# -----------------------------------------------------------------------

class TestConstantsConsistency:
    """Ensure runtime guard constants stay aligned with packaging metadata.

    If someone changes the pyproject.toml MPI extra version range,
    these tests will fail — reminding them to also update the runtime
    guards (and vice-versa).
    """

    def test_mpi4jax_min(self):
        assert _TESTED_MPI4JAX_MIN == (0, 8, 0), (
            "_TESTED_MPI4JAX_MIN must match pyproject.toml mpi4jax>=0.8"
        )

    def test_mpi4jax_max_excl(self):
        assert _TESTED_MPI4JAX_MAX_EXCL == (0, 10, 0), (
            "_TESTED_MPI4JAX_MAX_EXCL must match pyproject.toml mpi4jax<0.10 "
            "(mpi4jax 0.9.0 = the FFI rewrite, issue #567)"
        )

    def test_jax_tested_range(self):
        assert _TESTED_JAX_MIN == (0, 8, 0)
        assert _TESTED_JAX_MAX_EXCL == (0, 10, 0)
