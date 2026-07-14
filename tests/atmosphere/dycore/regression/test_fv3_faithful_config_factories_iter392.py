"""FV3_3D iter 392: tests for FV3-faithful config factories.

``make_fv3_faithful_pe_config`` + ``make_fv3_faithful_nh_config``
return configs with every FV3-fidelity flag enabled at
production-recommended values.  User-facing convenience.

Tests
-----

1. ``test_pe_factory_enables_all_flags``
2. ``test_nh_factory_enables_all_flags``
3. ``test_factory_accepts_overrides``
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


_PE_FLAGS = (
    "use_fv3_a2b_zeta_corner",
    "use_fv3_metric_aware_d_con",
    "use_fv3_cross_face_du_proj",
)
_NH_FLAGS = (
    "use_fv3_d_con_cv",
    "use_fv3_vector_halo_uv",
    "use_fv3_dynamic_exner",
    "use_fv3_metric_aware_d_con",
    "use_fv3_cross_face_du_proj",
)


def test_pe_factory_enables_all_flags():
    cfg = make_fv3_faithful_pe_config()
    for fld in _PE_FLAGS:
        assert getattr(cfg, fld) is True, (
            f"PE FV3-faithful factory must enable {fld}; "
            f"got {getattr(cfg, fld)}."
        )


def test_nh_factory_enables_all_flags():
    cfg = make_fv3_faithful_nh_config()
    for fld in _NH_FLAGS:
        assert getattr(cfg, fld) is True, (
            f"NH FV3-faithful factory must enable {fld}; "
            f"got {getattr(cfg, fld)}."
        )


def test_factory_accepts_overrides():
    pe = make_fv3_faithful_pe_config(damp_v=0.5, nord_v=2)
    assert pe.damp_v == 0.5
    assert pe.nord_v == 2
    # FV3-fidelity flags still ON.
    assert pe.use_fv3_metric_aware_d_con is True

    nh = make_fv3_faithful_nh_config(damp_w=0.5, ah_d_con=2.0)
    assert nh.damp_w == 0.5
    assert nh.ah_d_con == 2.0
    assert nh.use_fv3_metric_aware_d_con is True
