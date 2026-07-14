"""FV3_3D iter 418: executable test that iter-417 doc-example
code in FV3_3D.md compiles + runs.

Catches doc drift if API changes but example isn't updated.

Tests
-----

1. ``test_iter417_nh_example_runs`` — NH factory call from
   doc example produces valid config.
2. ``test_iter417_pe_example_runs`` — PE factory call same.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


def test_iter417_nh_example_runs():
    """Verbatim from iter-417 production-usage doc (updated
    iter-454)."""
    grid = create_cubed_sphere(n=8, use_duogrid=True)
    nh_cfg = make_fv3_faithful_nh_config(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    assert grid.duogrid is not None
    assert nh_cfg.damp_v == 0.030
    # iter-392 fidelity flags (the original 5):
    assert nh_cfg.use_fv3_d_con_cv is True
    assert nh_cfg.use_fv3_vector_halo_uv is True
    assert nh_cfg.use_fv3_dynamic_exner is True
    assert nh_cfg.use_fv3_metric_aware_d_con is True
    assert nh_cfg.use_fv3_cross_face_du_proj is True
    # iter-434/436/437/451/441/442 factory defaults (iter-454
    # additions to doc example):
    assert nh_cfg.d_con_top_zero_levels == 2
    assert nh_cfg.delt_max == 1.0
    assert nh_cfg.nord_v == 1
    assert nh_cfg.corner_div_damp_nord == 1
    assert nh_cfg.corner_div_damp_d4_bg == 0.16
    assert nh_cfg.use_fv3_sponge_damp_w is True
    assert nh_cfg.use_fv3_sponge_damp_v is True


def test_iter417_pe_example_runs():
    """Verbatim from iter-417 production-usage doc PE half
    (updated iter-454)."""
    grid = create_cubed_sphere(n=8, use_duogrid=True)
    pe_cfg = make_fv3_faithful_pe_config(
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
    )
    assert grid.duogrid is not None
    assert pe_cfg.damp_v == 0.030
    assert pe_cfg.use_fv3_a2b_zeta_corner is True
    assert pe_cfg.use_fv3_metric_aware_d_con is True
    assert pe_cfg.use_fv3_cross_face_du_proj is True
    # iter-434/436/437/451/443 factory defaults:
    assert pe_cfg.d_con_top_zero_levels == 2
    assert pe_cfg.delt_max == 1.0
    assert pe_cfg.nord_v == 1
    assert pe_cfg.corner_div_damp_nord == 1
    assert pe_cfg.corner_div_damp_d4_bg == 0.16
    assert pe_cfg.use_fv3_sponge_damp_v is True
