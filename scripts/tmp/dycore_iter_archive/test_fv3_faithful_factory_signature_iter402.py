"""FV3_3D iter 402: AST signature regression for iter-392
FV3-faithful config factories.

Asserts ``make_fv3_faithful_pe_config`` + ``make_fv3_faithful_nh_config``:
- Importable from their respective dycore modules
- Accept ``**overrides`` kwargs
- Return correct config type
- Default-factory result matches expected flag set

Mirror of iter-369 flag-set guard but at the factory level.
"""
from __future__ import annotations

import inspect

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
    make_fv3_faithful_pe_config,
)


def test_pe_factory_signature():
    sig = inspect.signature(make_fv3_faithful_pe_config)
    assert "overrides" in sig.parameters or "kwargs" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ), "PE factory must accept **overrides."


def test_nh_factory_signature():
    sig = inspect.signature(make_fv3_faithful_nh_config)
    assert any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ), "NH factory must accept **overrides."


def test_pe_factory_returns_correct_type():
    cfg = make_fv3_faithful_pe_config()
    assert isinstance(cfg, CDGridPrimitiveEquationConfig)


def test_nh_factory_returns_correct_type():
    cfg = make_fv3_faithful_nh_config()
    assert isinstance(cfg, CDGridCompressibleEulerConfig)


def test_factory_default_flag_set_complete():
    """Factories enable every FV3-fidelity flag.

    FV3_3D iter-1077: ``use_fv3_cross_face_du_proj`` re-enabled
    via the new ``pad_halo_dgrid_scalar_4d`` (iter-1076).  All FV3
    flags now enabled by default.
    """
    pe = make_fv3_faithful_pe_config()
    nh = make_fv3_faithful_nh_config()
    # PE FV3 flags
    assert pe.use_fv3_a2b_zeta_corner is True
    assert pe.use_fv3_metric_aware_d_con is True
    assert pe.use_fv3_cross_face_du_proj is True  # iter-1077
    # NH FV3 flags
    assert nh.use_fv3_d_con_cv is True
    assert nh.use_fv3_vector_halo_uv is True
    assert nh.use_fv3_dynamic_exner is True
    assert nh.use_fv3_metric_aware_d_con is True
    assert nh.use_fv3_cross_face_du_proj is True  # iter-1077
