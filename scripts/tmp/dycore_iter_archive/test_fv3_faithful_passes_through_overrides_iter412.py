"""FV3_3D iter 412: verify iter-392 factories pass through ALL
non-FV3-flag config knobs (hyperdiff, sponge, n_acoustic_substeps,
etc) via overrides.

Tests
-----

1. ``test_pe_factory_passes_hyperdiff_settings``
2. ``test_nh_factory_passes_acoustic_substeps``
3. ``test_factories_dont_clobber_user_values``
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


def test_pe_factory_passes_hyperdiff_settings():
    cfg = make_fv3_faithful_pe_config(
        hyperdiff_coeff=1e14,
        hyperdiff_ps_coeff=5e13,
    )
    assert cfg.hyperdiff_coeff == 1e14
    assert cfg.hyperdiff_ps_coeff == 5e13


def test_nh_factory_passes_acoustic_substeps():
    cfg = make_fv3_faithful_nh_config(
        n_acoustic_substeps=8,
        hyperdiff_coeff=2e14,
    )
    assert cfg.n_acoustic_substeps == 8
    assert cfg.hyperdiff_coeff == 2e14


def test_factories_dont_clobber_user_values():
    """User overrides take precedence over factory defaults."""
    # Factory enables a2b_zeta True; user disables it.
    pe = make_fv3_faithful_pe_config(
        use_fv3_a2b_zeta_corner=False,
        damp_v=0.045,
    )
    assert pe.use_fv3_a2b_zeta_corner is False  # override wins
    assert pe.damp_v == 0.045
    # Other flags still ON.
    assert pe.use_fv3_metric_aware_d_con is True

    # NH cv flag override.
    nh = make_fv3_faithful_nh_config(
        use_fv3_d_con_cv=False,
        damp_w=0.045,
    )
    assert nh.use_fv3_d_con_cv is False  # override wins
    assert nh.damp_w == 0.045
    assert nh.use_fv3_vector_halo_uv is True
