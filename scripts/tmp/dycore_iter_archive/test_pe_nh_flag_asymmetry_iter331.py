"""FV3_3D iter 331: pin the PE / NH flag asymmetry for iter-320
``use_fv3_d_con_cv`` and iter-328 ``use_fv3_vector_halo_uv``.

Rationale
---------
iter-317 verified PE / NH config defaults match for the SHARED
knobs (corner_div_damp_*, damp_v / nord_v, damp_v_d_con,
corner_div_damp_d_con, div_damp_d_con, ah_d_con, delt_max,
use_fv3_a2b_zeta_corner, smagorinsky_cs / A_h).  iter-320 + 328
intentionally added NH-ONLY flags (PE doesn't need them):

* ``use_fv3_d_con_cv``: NH compressible NH dynamics conserves
  total energy via internal energy ``c_v · T`` (cv_air).  PE is
  hydrostatic and uses enthalpy ``c_p · T`` (cp_air) which is
  already FV3-faithful for the cp_air branch — no NH-style flag
  needed on the PE path.

* ``use_fv3_vector_halo_uv``: NH stores winds at cell centres and
  interpolates to D-grid corners; the scalar halo at this step
  bypasses face-local vector rotation.  PE stores winds at
  corners directly (D-grid native) — no center-to-corner
  interpolation, no vector halo gap.

iter-331 pins this intentional asymmetry so a future "parity"
refactor doesn't blindly add the NH flags to PE config.

Tests
-----

1. ``test_pe_does_not_expose_use_fv3_d_con_cv`` — PE config does
   NOT have the NH-only cv flag.
2. ``test_pe_does_not_expose_use_fv3_vector_halo_uv`` — PE config
   does NOT have the NH-only vector halo flag.
3. ``test_nh_exposes_both_nh_only_flags`` — NH config has both
   flags as part of the NH-only knob set.
4. ``test_nh_only_flags_default_off`` — both NH flags default to
   False (preserves baseline; iter-318 covers the cv flag, iter-
   331 redundantly pins the vector halo flag).
"""
from __future__ import annotations

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
)


_NH_ONLY_FV3_FLAGS = (
    "use_fv3_d_con_cv",
    "use_fv3_vector_halo_uv",
    "use_fv3_dynamic_exner",   # iter-336/337
)
# Flags PE + NH BOTH expose (iter-338 + iter-339 metric-aware,
# iter-370 cross-face du projection).
_SHARED_FV3_FLAGS = (
    "use_fv3_metric_aware_d_con",
    "use_fv3_cross_face_du_proj",
)


def test_pe_does_not_expose_use_fv3_dynamic_exner():
    """PE uses actual T (no Exner factor); dynamic Exner flag is
    NH-only.  iter-343 extension of iter-331 PE/NH asymmetry guard."""
    pe_cfg = CDGridPrimitiveEquationConfig()
    assert not hasattr(pe_cfg, "use_fv3_dynamic_exner"), (
        "CDGridPrimitiveEquationConfig must NOT expose "
        "use_fv3_dynamic_exner — PE is hydrostatic with actual T "
        "prognostic.  No Π factor in PE d_con denominator, so "
        "dynamic-vs-frozen-Π_ref flag has no PE meaning."
    )


def test_pe_does_not_expose_use_fv3_d_con_cv():
    """PE uses cp_air (FV3-faithful for hydrostatic branch); no
    cv-flag needed on PE."""
    pe_cfg = CDGridPrimitiveEquationConfig()
    assert not hasattr(pe_cfg, "use_fv3_d_con_cv"), (
        "CDGridPrimitiveEquationConfig must NOT expose "
        "use_fv3_d_con_cv — PE is hydrostatic and FV3 cp_air "
        "branch already matches PE's c_pd convention.  Adding "
        "the flag to PE would be a no-op at best and a "
        "thermodynamic bug at worst (would over-heat by "
        "c_p / c_v ≈ 1.40 if accidentally enabled)."
    )


def test_pe_does_not_expose_use_fv3_vector_halo_uv():
    """PE stores winds at corners (D-grid native); no
    center-to-corner interpolation, no vector halo gap."""
    pe_cfg = CDGridPrimitiveEquationConfig()
    assert not hasattr(pe_cfg, "use_fv3_vector_halo_uv"), (
        "CDGridPrimitiveEquationConfig must NOT expose "
        "use_fv3_vector_halo_uv — PE stores winds at D-grid "
        "corners, so there is no cell-centre → corner "
        "interpolation step and no scalar-halo basis-mismatch "
        "to fix."
    )


def test_nh_exposes_both_nh_only_flags():
    """NH config has both NH-only FV3-fidelity flags."""
    nh_cfg = CDGridCompressibleEulerConfig()
    for field in _NH_ONLY_FV3_FLAGS:
        assert hasattr(nh_cfg, field), (
            f"CDGridCompressibleEulerConfig must expose {field} — "
            f"iter-320/328 wiring depends on this name."
        )


def test_nh_only_flags_default_off():
    """Both NH-only FV3 flags default to False."""
    nh_cfg = CDGridCompressibleEulerConfig()
    for field in _NH_ONLY_FV3_FLAGS:
        val = getattr(nh_cfg, field)
        assert val is False, (
            f"NH config field {field} default is {val}; must be "
            f"False to preserve bit-for-bit baseline."
        )


def test_pe_and_nh_both_expose_shared_fv3_flags():
    """Shared FV3-fidelity flags (iter-338/339 metric-aware) live
    on BOTH PE and NH configs since the gap (cosa_s/rsin2 metric)
    affects both paths' damp_v_d_con sites."""
    pe_cfg = CDGridPrimitiveEquationConfig()
    nh_cfg = CDGridCompressibleEulerConfig()
    for field in _SHARED_FV3_FLAGS:
        assert hasattr(pe_cfg, field), (
            f"PE config must expose {field} (iter-338 wiring)."
        )
        assert hasattr(nh_cfg, field), (
            f"NH config must expose {field} (iter-339 wiring)."
        )
        # Both default False
        assert getattr(pe_cfg, field) is False
        assert getattr(nh_cfg, field) is False
