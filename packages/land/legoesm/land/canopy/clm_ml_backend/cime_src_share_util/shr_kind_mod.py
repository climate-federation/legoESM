"""JAX translation of shr_kind_mod module.

This module defines precision/kind constants for CTSM, providing standardized
numeric types for consistent precision across the model.

Fortran source: shr_kind_mod.F90, lines 1-11

The module establishes:
- SHR_KIND_R8: Double precision floating point (64-bit)
- SHR_KIND_IN: Native integer type (32-bit)

These constants are used throughout CTSM to ensure consistent numeric precision
and portability across different computing platforms.
"""

from typing import Final

import jax.numpy as jnp

# =============================================================================
# PRECISION/KIND CONSTANTS
# =============================================================================

# Fortran line 8: integer, parameter :: shr_kind_r8 = selected_real_kind(12)
# Double precision floating point type (8 bytes, ~15-17 decimal digits)
# Corresponds to Fortran REAL(r8) or REAL*8
SHR_KIND_R8: Final = jnp.float64

# Fortran line 9: integer, parameter :: SHR_KIND_IN = kind(1)
# Native integer type (4 bytes, range: -2^31 to 2^31-1)
# Corresponds to Fortran INTEGER or INTEGER*4
SHR_KIND_IN: Final = jnp.int32

# =============================================================================
# ALIASES FOR CONVENIENCE
# =============================================================================

# Common aliases used in translated modules
r8 = SHR_KIND_R8  # Alias for abortutils and other modules

# =============================================================================
# NOTES
# =============================================================================
#
# In Fortran, selected_real_kind(12) requests a real type with at least
# 12 decimal digits of precision, which maps to IEEE 754 double precision
# (binary64). JAX's float64 provides this precision.
#
# The native integer kind in Fortran (kind(1)) is typically 4 bytes on most
# modern systems, corresponding to int32. This provides sufficient range
# for array indexing and loop counters in CTSM.
#
# Usage in other modules:
#   from shr_kind_mod import SHR_KIND_R8, SHR_KIND_IN, r8
#
#   # For array declarations:
#   temperature = jnp.zeros(n, dtype=SHR_KIND_R8)
#   indices = jnp.arange(n, dtype=SHR_KIND_IN)
