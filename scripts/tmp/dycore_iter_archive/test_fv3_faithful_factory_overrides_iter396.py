"""FV3_3D iter 396: factory override edge cases.

Verifies iter-392 ``make_fv3_faithful_*_config(**overrides)``
factories:

1. Allow disabling a single FV3 flag while keeping others ON.
2. Allow opt-out from ALL FV3 flags (overrides=all False).
3. Allow setting d_con knobs alongside flag enables.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


def test_pe_factory_can_disable_single_flag():
    """User can disable cross_face but keep other flags ON."""
    cfg = make_fv3_faithful_pe_config(
        use_fv3_cross_face_du_proj=False,
    )
    assert cfg.use_fv3_a2b_zeta_corner is True
    assert cfg.use_fv3_metric_aware_d_con is True
    assert cfg.use_fv3_cross_face_du_proj is False


def test_nh_factory_can_disable_single_flag():
    cfg = make_fv3_faithful_nh_config(
        use_fv3_metric_aware_d_con=False,
    )
    assert cfg.use_fv3_d_con_cv is True
    assert cfg.use_fv3_vector_halo_uv is True
    assert cfg.use_fv3_dynamic_exner is True
    assert cfg.use_fv3_metric_aware_d_con is False
    # FV3_3D iter-1077: re-enabled via pad_halo_dgrid_scalar_4d.
    assert cfg.use_fv3_cross_face_du_proj is True


def test_pe_factory_all_overrides_false():
    """User can opt out of every FV3 flag via overrides."""
    cfg = make_fv3_faithful_pe_config(
        use_fv3_a2b_zeta_corner=False,
        use_fv3_metric_aware_d_con=False,
        use_fv3_cross_face_du_proj=False,
    )
    assert cfg.use_fv3_a2b_zeta_corner is False
    assert cfg.use_fv3_metric_aware_d_con is False
    assert cfg.use_fv3_cross_face_du_proj is False


def test_nh_factory_with_d_con_knobs():
    cfg = make_fv3_faithful_nh_config(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005,
        corner_div_damp_d_con=1.0,
    )
    assert cfg.damp_v == 0.030
    assert cfg.damp_v_d_con == 1.0
    assert cfg.corner_div_damp_d2_bg == 0.0005
    assert cfg.corner_div_damp_d_con == 1.0
    # FV3 flags still ON.
    assert cfg.use_fv3_d_con_cv is True
