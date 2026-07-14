"""FV3_3D iter 318: AST guard for NH-only knob defaults.

NH (``CDGridCompressibleEulerConfig``) has knobs for vertical
velocity damping that PE doesn't (PE is hydrostatic, no w):

    * ``damp_w`` — vertical-velocity damping coefficient
    * ``nord_w`` — order of damp_w (default 2)
    * ``damp_w_d_con`` — KE→heat conversion for damp_w post-step

iter-203 / iter-258 / iter-275 / iter-277 / iter-280 cover
runtime behavior of these.  iter-318 pins the AST-level
defaults: damp_w / damp_w_d_con default to 0 (off) so adding
NH config to any test by default doesn't engage the path.

Tests
-----

1. ``test_damp_w_default_off`` — ``damp_w`` defaults to 0.0.
2. ``test_damp_w_d_con_default_off`` — ``damp_w_d_con``
   defaults to 0.0.
3. ``test_nord_w_default`` — ``nord_w`` defaults to 2.
"""
from __future__ import annotations

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
)


def test_damp_w_default_off():
    """``damp_w`` defaults to 0.0 (off).  Caller opts in by
    setting > 0."""
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.damp_w == 0.0, (
        f"damp_w default must be 0.0 (off); got {cfg.damp_w}.  "
        f"A non-zero default would silently engage the iter-203 "
        f"damp_w post-step KE→heat path everywhere."
    )


def test_damp_w_d_con_default_off():
    """``damp_w_d_con`` defaults to 0.0 (off)."""
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.damp_w_d_con == 0.0, (
        f"damp_w_d_con default must be 0.0 (off); got "
        f"{cfg.damp_w_d_con}.  A non-zero default would "
        f"silently activate the iter-203 KE→heat conversion."
    )


def test_nord_w_default():
    """``nord_w`` defaults to 2 (matching FV3 production)."""
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.nord_w == 2, (
        f"nord_w default must be 2 (matching FV3 production "
        f"and PE.nord_v); got {cfg.nord_w}."
    )


def test_nh_only_knobs_present_in_signature():
    """NH config exposes damp_w, nord_w, damp_w_d_con as
    fields (catches accidental rename / removal)."""
    cfg = CDGridCompressibleEulerConfig()
    for field in ("damp_w", "nord_w", "damp_w_d_con"):
        assert hasattr(cfg, field), (
            f"NH config must expose {field} field — iter-203/"
            f"258/275/277/280 callers depend on this name."
        )


def test_use_fv3_metric_aware_d_con_default_off():
    """``use_fv3_metric_aware_d_con`` defaults to False (iter 339).
    The opt-in flag swaps the iter-209 simpler ΔKE form for the
    FV3-faithful metric-aware form (cosa_s / rsin2 cross-terms)
    at the NH damp_v_d_con site.  Default False preserves
    bit-for-bit baseline."""
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.use_fv3_metric_aware_d_con is False, (
        f"use_fv3_metric_aware_d_con default must be False; got "
        f"{cfg.use_fv3_metric_aware_d_con}."
    )
    assert hasattr(cfg, "use_fv3_metric_aware_d_con"), (
        "NH config must expose use_fv3_metric_aware_d_con field — "
        "iter-339 wiring depends on this name."
    )


def test_use_fv3_dynamic_exner_default_off():
    """``use_fv3_dynamic_exner`` defaults to False (FV3_3D iter 336).
    The opt-in flag swaps frozen ``Π_ref`` for dynamic
    ``Π_total = Π_ref + π'`` at the 3 NH slow-tendency d_con sites
    (corner_div / div_damp / ah).  FV3 ``dyn_core.F90`` uses live
    ``pkz`` from current pressure.  Default False preserves
    bit-for-bit baseline."""
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.use_fv3_dynamic_exner is False, (
        f"use_fv3_dynamic_exner default must be False (preserves "
        f"bit-for-bit baseline); got {cfg.use_fv3_dynamic_exner}.  "
        f"A True default would silently introduce ~30 % heating "
        f"variation in the slow-tendency d_con sites."
    )
    assert hasattr(cfg, "use_fv3_dynamic_exner"), (
        "NH config must expose use_fv3_dynamic_exner field — "
        "iter-336 wiring depends on this name."
    )


def test_use_fv3_vector_halo_uv_default_off():
    """``use_fv3_vector_halo_uv`` defaults to False (FV3_3D iter 328).
    The opt-in flag switches NH cell-centre → D-grid corner
    interpolation of (u, v) from scalar halo (with duogrid routing)
    to vector-aware ``center_to_dgrid_vector`` which rotates
    components across cube-face boundaries (FV3 ``ext_vector``
    analogue at ``fv_duogrid.F90:626-975``).  Default False
    preserves bit-for-bit baseline; users opt in for FV3-faithful
    cube-edge vector halo."""
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.use_fv3_vector_halo_uv is False, (
        f"use_fv3_vector_halo_uv default must be False (preserves "
        f"bit-for-bit baseline); got {cfg.use_fv3_vector_halo_uv}.  "
        f"A True default would silently change the (u, v) → D-grid "
        f"corner halo behaviour at cube-face boundaries."
    )
    assert hasattr(cfg, "use_fv3_vector_halo_uv"), (
        "NH config must expose use_fv3_vector_halo_uv field — "
        "iter-328 wiring depends on this name."
    )


def test_use_fv3_d_con_cv_default_off():
    """``use_fv3_d_con_cv`` defaults to False (FV3_3D iter 320).
    The opt-in flag swaps c_pd → c_vd at all 5 NH d_con sites for
    FV3-faithful heating partition (FV3 dyn_core.F90:1795 cv_air
    branch).  Default False preserves bit-for-bit baseline; users
    opt in for FV3-faithful NH heat capacity."""
    cfg = CDGridCompressibleEulerConfig()
    assert cfg.use_fv3_d_con_cv is False, (
        f"use_fv3_d_con_cv default must be False (preserves "
        f"bit-for-bit baseline); got {cfg.use_fv3_d_con_cv}.  "
        f"A True default would silently change the heat partition "
        f"of every NH d_con site by a factor of c_pd/c_vd ≈ 1.40."
    )
    assert hasattr(cfg, "use_fv3_d_con_cv"), (
        "NH config must expose use_fv3_d_con_cv field — "
        "iter-320 wiring depends on this name."
    )
