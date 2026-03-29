"""Tests for CMOR output dtype handling.

Verifies that the CMOR output pipeline respects the precision policy
system rather than unconditionally forcing float32.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from legoesm.core.precision import (
    PrecisionPolicy, set_policy, clear_module_overrides,
)


class TestCMOROutputDtype:
    """Verify CMOR output respects precision policy."""

    def setup_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def teardown_method(self):
        set_policy(PrecisionPolicy.fp32())
        clear_module_overrides()

    def test_to_numpy_preserves_array(self):
        from legoesm.io.cmor_output import _to_numpy
        x = jnp.ones(5, dtype=jnp.float32)
        result = _to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_resolve_output_dtype_fp32_default(self):
        """Default output dtype should be float32 under fp32 policy."""
        from legoesm.io.cmor_output import _resolve_output_dtype
        dtype = _resolve_output_dtype(None)
        assert dtype == np.float32

    def test_resolve_output_dtype_fp64_policy(self):
        """Under fp64 policy, output dtype should be float64."""
        set_policy(PrecisionPolicy.fp64())
        from legoesm.io.cmor_output import _resolve_output_dtype
        dtype = _resolve_output_dtype(None)
        assert dtype == np.float64

    def test_resolve_output_dtype_explicit_override(self):
        """Explicit dtype spec should override policy."""
        from legoesm.io.cmor_output import _resolve_output_dtype
        dtype = _resolve_output_dtype("float64")
        assert dtype == np.float64

    def test_resolve_output_dtype_mixed_uses_fp32(self):
        """Mixed policy stores in float32, so output should be float32."""
        set_policy(PrecisionPolicy.mixed())
        from legoesm.io.cmor_output import _resolve_output_dtype
        dtype = _resolve_output_dtype(None)
        assert dtype == np.float32
