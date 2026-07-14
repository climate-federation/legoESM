"""FV3_3D iter 317: PE+NH config default-parity guard for
shared d_con cluster knobs.

PE (``CDGridPrimitiveEquationConfig``) and NH
(``CDGridCompressibleEulerConfig``) share many knobs related
to the d_con cluster: corner-div damping coefficients,
damp_v / nord_v, damp_v_d_con, etc.  When a shared knob
default drifts on one side but not the other, latent bugs
appear in PE/NH parity tests.

iter-317 pins parity of defaults across the two configs for
every shared knob.  Catches silent default drift by AST-
level introspection (no runtime simulation needed).

Tests
-----

1. ``test_corner_div_damp_defaults_match`` — d2_bg, dddmp,
   d4_bg, nord match between PE+NH.
2. ``test_damp_v_defaults_match`` — damp_v, nord_v,
   damp_v_d_con match.
3. ``test_corner_div_damp_d_con_default_match`` —
   corner_div_damp_d_con / div_damp_d_con / ah_d_con match.
"""
from __future__ import annotations

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
)


def test_corner_div_damp_defaults_match():
    """PE and NH share corner-div damping defaults."""
    pe = CDGridPrimitiveEquationConfig()
    nh = CDGridCompressibleEulerConfig()
    for field in (
        "corner_div_damp_d2_bg",
        "corner_div_damp_dddmp",
        "corner_div_damp_d4_bg",
        "corner_div_damp_nord",
    ):
        pe_val = getattr(pe, field)
        nh_val = getattr(nh, field)
        assert pe_val == nh_val, (
            f"PE.{field}={pe_val} but NH.{field}={nh_val}; "
            f"shared knob defaults must match across both 3D "
            f"configs."
        )


def test_damp_v_defaults_match():
    """PE and NH share damp_v / nord_v / damp_v_d_con defaults."""
    pe = CDGridPrimitiveEquationConfig()
    nh = CDGridCompressibleEulerConfig()
    for field in ("damp_v", "nord_v", "damp_v_d_con"):
        pe_val = getattr(pe, field)
        nh_val = getattr(nh, field)
        assert pe_val == nh_val, (
            f"PE.{field}={pe_val} but NH.{field}={nh_val}; "
            f"shared damp_v knob defaults must match."
        )


def test_corner_div_damp_d_con_default_match():
    """PE and NH share d_con stack defaults."""
    pe = CDGridPrimitiveEquationConfig()
    nh = CDGridCompressibleEulerConfig()
    for field in (
        "corner_div_damp_d_con",
        "div_damp_d_con",
        "ah_d_con",
        "delt_max",
    ):
        pe_val = getattr(pe, field)
        nh_val = getattr(nh, field)
        assert pe_val == nh_val, (
            f"PE.{field}={pe_val} but NH.{field}={nh_val}; "
            f"d_con stack knob defaults must match across "
            f"both 3D paths."
        )


def test_a2b_zeta_corner_default_match():
    """PE and NH share use_fv3_a2b_zeta_corner default."""
    pe = CDGridPrimitiveEquationConfig()
    nh = CDGridCompressibleEulerConfig()
    pe_val = pe.use_fv3_a2b_zeta_corner
    nh_val = nh.use_fv3_a2b_zeta_corner
    assert pe_val == nh_val, (
        f"PE.use_fv3_a2b_zeta_corner={pe_val} but NH={nh_val}; "
        f"the iter-170/187 a2b_ord4 zeta interp gate must be "
        f"consistent."
    )


def test_smagorinsky_a_h_defaults_match():
    """PE and NH share smagorinsky_cs / A_h defaults."""
    pe = CDGridPrimitiveEquationConfig()
    nh = CDGridCompressibleEulerConfig()
    for field in ("smagorinsky_cs", "A_h"):
        pe_val = getattr(pe, field)
        nh_val = getattr(nh, field)
        assert pe_val == nh_val, (
            f"PE.{field}={pe_val} but NH.{field}={nh_val}; "
            f"Smagorinsky/A_h knob defaults must match."
        )
