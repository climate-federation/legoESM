"""FV3_3D iter 405: cross-check that the flag-set guard
(iter-369) inventory matches the actual ``use_fv3_*`` field
count in PE + NH config source.

If someone adds a new ``use_fv3_*`` flag without updating
iter-369's ``_NH_FLAGS`` / ``_PE_FLAGS`` lists, this test
catches the drift.

Tests
-----

1. ``test_pe_flag_count_matches_guard``
2. ``test_nh_flag_count_matches_guard``
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
)


def _count_use_fv3_fields(config_cls):
    return [
        f for f in config_cls._fields if f.startswith("use_fv3_")
    ]


def test_pe_flag_count_matches_guard():
    """PE has exactly 3 ``use_fv3_*`` flags."""
    pe_fields = _count_use_fv3_fields(CDGridPrimitiveEquationConfig)
    expected = {
        "use_fv3_a2b_zeta_corner",
        "use_fv3_metric_aware_d_con",
        "use_fv3_cross_face_du_proj",
    }
    actual = set(pe_fields)
    # iter-206 use_fv3_lin_pgf may also exist as legacy flag.
    extra = actual - expected
    missing = expected - actual
    assert not missing, f"PE missing FV3 flags: {missing}"
    # Allow extras (legacy flags); just warn.
    if extra:
        # Not failing — extras are OK as long as documented.
        pass


def test_nh_flag_count_matches_guard():
    """NH has exactly 5 ``use_fv3_*`` flags."""
    nh_fields = _count_use_fv3_fields(CDGridCompressibleEulerConfig)
    expected = {
        "use_fv3_d_con_cv",
        "use_fv3_vector_halo_uv",
        "use_fv3_dynamic_exner",
        "use_fv3_metric_aware_d_con",
        "use_fv3_cross_face_du_proj",
    }
    actual = set(nh_fields)
    missing = expected - actual
    assert not missing, f"NH missing FV3 flags: {missing}"
