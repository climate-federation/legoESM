"""FV3_3D iter 319: AST guard for d_con knob count.

The d_con cluster on PE has 4 knobs:

    * damp_v_d_con
    * corner_div_damp_d_con
    * div_damp_d_con
    * ah_d_con

NH has 5 knobs (the 4 PE + ``damp_w_d_con`` for w-damping):

    * damp_v_d_con
    * corner_div_damp_d_con
    * div_damp_d_con
    * ah_d_con
    * damp_w_d_con

This count is structurally pinned by the test matrix:
iter-272/273/274 (PE direction), iter-275 (NH trio), iter-276
(damp_v at production nord_v), iter-277 (NH damp_w).  When a
new d_con site is added (e.g., iter-238 metric-aware form), it
must be reflected here AND in the runtime tests.

Tests
-----

1. ``test_pe_d_con_knob_count`` — exactly 4 ``*_d_con`` knobs
   on PE config.
2. ``test_nh_d_con_knob_count`` — exactly 5 ``*_d_con`` knobs
   on NH config.
3. ``test_pe_d_con_subset_of_nh`` — every PE d_con knob also
   exists on NH.
4. ``test_nh_only_d_con_knob_is_damp_w`` — the unique NH-only
   d_con knob is ``damp_w_d_con``.
"""
from __future__ import annotations

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
)


def _d_con_field_names(config_cls):
    """Return all field names ending in '_d_con' (exclude
    ``use_fv3_*`` opt-in flags that happen to contain the
    ``_d_con`` substring — iter-339 added
    ``use_fv3_metric_aware_d_con``)."""
    return {
        f for f in config_cls._fields
        if f.endswith("_d_con") and not f.startswith("use_fv3_")
    }


def test_pe_d_con_knob_count():
    pe_d_con = _d_con_field_names(CDGridPrimitiveEquationConfig)
    expected = {
        "damp_v_d_con",
        "corner_div_damp_d_con",
        "div_damp_d_con",
        "ah_d_con",
    }
    assert pe_d_con == expected, (
        f"PE d_con knob set drift: expected {expected}, got "
        f"{pe_d_con}.  If a new d_con site is added, update "
        f"this guard AND iter-272/273/274/276 runtime tests."
    )


def test_nh_d_con_knob_count():
    nh_d_con = _d_con_field_names(CDGridCompressibleEulerConfig)
    expected = {
        "damp_v_d_con",
        "corner_div_damp_d_con",
        "div_damp_d_con",
        "ah_d_con",
        "damp_w_d_con",
    }
    assert nh_d_con == expected, (
        f"NH d_con knob set drift: expected {expected}, got "
        f"{nh_d_con}."
    )


def test_pe_d_con_subset_of_nh():
    pe_d_con = _d_con_field_names(CDGridPrimitiveEquationConfig)
    nh_d_con = _d_con_field_names(CDGridCompressibleEulerConfig)
    assert pe_d_con.issubset(nh_d_con), (
        f"PE d_con set ({pe_d_con}) must be subset of NH "
        f"({nh_d_con}); NH-only knobs reflect compressible "
        f"physics (damp_w) that PE doesn't have."
    )


def test_nh_only_d_con_knob_is_damp_w():
    pe_d_con = _d_con_field_names(CDGridPrimitiveEquationConfig)
    nh_d_con = _d_con_field_names(CDGridCompressibleEulerConfig)
    nh_only = nh_d_con - pe_d_con
    assert nh_only == {"damp_w_d_con"}, (
        f"NH-only d_con knob set must be exactly "
        f"{{'damp_w_d_con'}}; got {nh_only}."
    )
