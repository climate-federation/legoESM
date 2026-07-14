"""FV3_3D iter 444: factory defaults expose FV3 production
sponge boost values.

Per FV3 ``fv_arrays.F90`` production defaults::

    real :: d2_bg_k1 = 4.    ! factor for d2_bg (k=1)
    real :: d2_bg_k2 = 2.    ! factor for d2_bg (k=2)

Both factories now set ``corner_div_damp_d2_bg_k1=4.0`` and
``corner_div_damp_d2_bg_k2=2.0``, together with the iter-441/
442/443 opt-in flags:
* NH: ``use_fv3_sponge_damp_w=True`` (FV3 ``damp_w = d2_divg``)
       ``use_fv3_sponge_damp_v=True`` (FV3 ``damp_vt = 0.5*d2_divg``)
* PE: ``use_fv3_sponge_damp_v=True`` (PE has no damp_w)

Tests
-----

1. ``test_nh_factory_default_d2_bg_k1`` — 4.0.
2. ``test_nh_factory_default_d2_bg_k2`` — 2.0.
3. ``test_nh_factory_default_sponge_damp_w_flag`` — True.
4. ``test_nh_factory_default_sponge_damp_v_flag`` — True.
5. ``test_pe_factory_default_d2_bg_k1`` — 4.0.
6. ``test_pe_factory_default_d2_bg_k2`` — 2.0.
7. ``test_pe_factory_default_sponge_damp_v_flag`` — True.
8. ``test_nh_factory_override_recovers_baseline``.
9. ``test_pe_factory_override_recovers_baseline``.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    make_fv3_faithful_pe_config,
)


def test_nh_factory_default_d2_bg_k1():
    """iter-452: rolled back from 4.0 → 0.0; FV3 production
    value assumes FV3's da_min_c × d2 normalization which
    legoESM does not match."""
    cfg = make_fv3_faithful_nh_config()
    assert cfg.corner_div_damp_d2_bg_k1 == 0.0


def test_nh_factory_default_d2_bg_k2():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.corner_div_damp_d2_bg_k2 == 0.0


def test_nh_factory_default_sponge_damp_w_flag():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.use_fv3_sponge_damp_w is True


def test_nh_factory_default_sponge_damp_v_flag():
    cfg = make_fv3_faithful_nh_config()
    assert cfg.use_fv3_sponge_damp_v is True


def test_pe_factory_default_d2_bg_k1():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.corner_div_damp_d2_bg_k1 == 0.0


def test_pe_factory_default_d2_bg_k2():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.corner_div_damp_d2_bg_k2 == 0.0


def test_pe_factory_default_sponge_damp_v_flag():
    cfg = make_fv3_faithful_pe_config()
    assert cfg.use_fv3_sponge_damp_v is True


def test_nh_factory_override_recovers_baseline():
    cfg = make_fv3_faithful_nh_config(
        corner_div_damp_d2_bg_k1=0.0,
        corner_div_damp_d2_bg_k2=0.0,
        use_fv3_sponge_damp_w=False,
        use_fv3_sponge_damp_v=False,
    )
    assert cfg.corner_div_damp_d2_bg_k1 == 0.0
    assert cfg.corner_div_damp_d2_bg_k2 == 0.0
    assert cfg.use_fv3_sponge_damp_w is False
    assert cfg.use_fv3_sponge_damp_v is False


def test_pe_factory_override_recovers_baseline():
    cfg = make_fv3_faithful_pe_config(
        corner_div_damp_d2_bg_k1=0.0,
        corner_div_damp_d2_bg_k2=0.0,
        use_fv3_sponge_damp_v=False,
    )
    assert cfg.corner_div_damp_d2_bg_k1 == 0.0
    assert cfg.corner_div_damp_d2_bg_k2 == 0.0
    assert cfg.use_fv3_sponge_damp_v is False
