"""Unit tests for :mod:`legoesm.training.promotable_params`.

Demonstrates the per-scheme promotion (Stage 7): a real production coefficient
(gray-radiation optical depth ``tau_equator``) accepts a per-column feedback
field with NO scheme-body change — the production scalar path stays identical,
and a non-uniform per-column field produces per-column-varying LW heating.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.gray import gray_radiation
from legoesm.training.promotable_params import (
    PROMOTABLE_FIELDS,
    apply_feedback_to_scheme,
    clip_field_to_promotable_bounds,
    promotable_field_names,
)


def test_registry_names():
    names = promotable_field_names()
    assert "gray_tau_equator" in names
    assert "gray_tau_pole" in names
    # sfc_albedo is deliberately NOT registered (shadowed by the explicit arg).
    assert "gray_sfc_albedo" not in names
    assert all(PROMOTABLE_FIELDS[k].body_safe for k in names)


def test_apply_expected_ncol_mismatch_raises():
    with pytest.raises(ValueError, match="expected_ncol"):
        apply_feedback_to_scheme(
            GrayRadiationConfig(), "gray_tau_equator", jnp.zeros(3),
            expected_ncol=2,
        )


def test_apply_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown promotable coefficient"):
        apply_feedback_to_scheme(GrayRadiationConfig(), "bogus", jnp.zeros(2))


def test_apply_splices_per_column_tau_equator():
    cfg = GrayRadiationConfig()
    field = jnp.array([7.2, 12.0])  # grid-shaped (ncol,)
    new = apply_feedback_to_scheme(cfg, "gray_tau_equator", field)
    assert new.tau_equator.shape == (2,)
    np.testing.assert_allclose(np.asarray(new.tau_equator), [7.2, 12.0])
    # original config unchanged (scalar production default).
    assert cfg.tau_equator == GrayRadiationConfig().tau_equator


def _gray_inputs(ncol=2, nlev=10):
    p_half = jnp.broadcast_to(jnp.linspace(1.0e3, 1.0e5, nlev + 1), (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.broadcast_to(jnp.linspace(230.0, 290.0, nlev), (ncol, nlev))
    sfc_T = jnp.full((ncol,), 290.0)
    lat = jnp.zeros((ncol,))  # equator for both, so tau_ref == tau_equator
    q_v = jnp.full((ncol, nlev), 5e-3)
    insol = jnp.full((ncol,), 400.0)
    return T, p_full, p_half, sfc_T, lat, q_v, insol


def test_uniform_percolumn_matches_scalar():
    """A uniform (ncol,) tau_equator == the scalar default (no regression)."""
    inp = _gray_inputs()
    cfg_scalar = GrayRadiationConfig()
    out_scalar = gray_radiation(*inp, config=cfg_scalar)

    cfg_uniform = apply_feedback_to_scheme(
        cfg_scalar, "gray_tau_equator",
        jnp.full((2,), cfg_scalar.tau_equator),
    )
    out_uniform = gray_radiation(*inp, config=cfg_uniform)

    np.testing.assert_allclose(
        np.asarray(out_uniform.lw_heating_rate),
        np.asarray(out_scalar.lw_heating_rate), rtol=1e-12,
    )


def test_tau_pole_promotion_reaches_physics():
    """tau_pole shares the same element-wise path; promote it at a pole column."""
    ncol, nlev = 2, 10
    inp = _gray_inputs(ncol, nlev)
    # Put both columns at the pole so tau_ref == tau_pole.
    T, p_full, p_half, sfc_T, _lat, q_v, insol = inp
    lat = jnp.full((ncol,), float(jnp.deg2rad(90.0)))
    inp_pole = (T, p_full, p_half, sfc_T, lat, q_v, insol)
    cfg = GrayRadiationConfig()
    cfg_pc = apply_feedback_to_scheme(
        cfg, "gray_tau_pole", jnp.array([cfg.tau_pole, 4.5])
    )
    out = gray_radiation(*inp_pole, config=cfg_pc)
    lw = np.asarray(out.lw_heating_rate)
    assert not np.allclose(lw[0], lw[1])  # per-column tau_pole reaches the LW


def test_nonuniform_percolumn_changes_lw_per_column():
    """A non-uniform per-column tau_equator produces per-column-varying LW
    heating — the feedback actually reaches the physics."""
    inp = _gray_inputs()
    cfg = GrayRadiationConfig()
    # column 0 keeps the default, column 1 gets a much larger optical depth.
    cfg_pc = apply_feedback_to_scheme(
        cfg, "gray_tau_equator", jnp.array([cfg.tau_equator, 14.0])
    )
    out = gray_radiation(*inp, config=cfg_pc)
    lw = np.asarray(out.lw_heating_rate)
    # The two columns now differ (identical inputs except tau_equator).
    assert not np.allclose(lw[0], lw[1])
    # Column 0 matches the scalar-default run.
    out_scalar = gray_radiation(*inp, config=cfg)
    np.testing.assert_allclose(lw[0], np.asarray(out_scalar.lw_heating_rate)[0],
                               rtol=1e-12)


def test_clip_field_to_promotable_bounds_clamps_ck():
    # C_K registered bounds are (0.1, 1.2); out-of-range diagnoses are clamped.
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    field = jnp.array([[0.05, 5.0], [0.4, 1.5]])   # below lo / far above / in / above
    clipped = clip_field_to_promotable_bounds(
        CLUBBLiteConfig(), "clubb_lite_C_K", field)
    np.testing.assert_allclose(
        np.asarray(clipped).reshape(-1), [0.1, 1.2, 0.4, 1.2])
    assert float(np.asarray(clipped).min()) >= 0.1
    assert float(np.asarray(clipped).max()) <= 1.2


def test_clip_field_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown promotable"):
        clip_field_to_promotable_bounds(GrayRadiationConfig(), "bogus",
                                        jnp.zeros((2, 2)))


def test_clip_field_clamps_gray_tau_generically():
    # The clamp is generic over registered coefficients: gray tau_equator bounds
    # are (2.0, 15.0), so an out-of-range field is clamped to that range.
    out = clip_field_to_promotable_bounds(
        GrayRadiationConfig(), "gray_tau_equator", jnp.array([1.0, 20.0, 7.0]))
    np.testing.assert_allclose(np.asarray(out), [2.0, 15.0, 7.0])


def test_clip_field_no_bounds_returns_unchanged():
    # When the target field has no resolvable scalar bounds in the config's spec
    # (param_field_bounds → None), the field is returned unchanged (no clamp).
    from legoesm.training.feedback import param_field_bounds

    class _NoSpec:                      # a config object with no __param_spec__
        C_K = 0.4
    assert param_field_bounds(_NoSpec(), "C_K") is None
    field = jnp.array([7.0, -3.0])
    out = clip_field_to_promotable_bounds(_NoSpec(), "clubb_lite_C_K", field)
    np.testing.assert_allclose(np.asarray(out), np.asarray(field))


def test_param_field_bounds_resolves_ck():
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.training.feedback import param_field_bounds
    assert param_field_bounds(CLUBBLiteConfig(), "C_K") == (0.1, 1.2)
    assert param_field_bounds(CLUBBLiteConfig(), "nonexistent_field") is None


def test_prandtl_promotion_registered_and_clamped():
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    assert "clubb_lite_Pr_t" in promotable_field_names()
    assert PROMOTABLE_FIELDS["clubb_lite_Pr_t"].field == "Pr_t"
    # Pr_t registered bounds (0.3, 1.5) ⇒ an out-of-range diagnosis is clamped.
    out = clip_field_to_promotable_bounds(
        CLUBBLiteConfig(), "clubb_lite_Pr_t", jnp.array([0.1, 2.0, 0.7]))
    np.testing.assert_allclose(np.asarray(out), [0.3, 1.5, 0.7])
    # the per-column Pr_t body wiring (uniform == scalar; per-column changes Kh)
    # is tested in tests/atmosphere/test_clubb_lite_ck_promotion.py.


def test_apply_feedback_independent_slots_order_independent():
    # The simultaneous multi-coefficient path applies C_K then Pr_t sequentially;
    # they are independent config slots, so the order must not matter (no clobber).
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    base = CLUBBLiteConfig()
    f_ck = jnp.array([0.5, 0.6, 0.7, 0.8])
    f_prt = jnp.array([0.7, 0.8, 0.9, 1.0])
    a = apply_feedback_to_scheme(
        apply_feedback_to_scheme(base, "clubb_lite_C_K", f_ck),
        "clubb_lite_Pr_t", f_prt)
    b = apply_feedback_to_scheme(
        apply_feedback_to_scheme(base, "clubb_lite_Pr_t", f_prt),
        "clubb_lite_C_K", f_ck)
    np.testing.assert_array_equal(np.asarray(a.C_K), np.asarray(b.C_K))
    np.testing.assert_array_equal(np.asarray(a.Pr_t), np.asarray(b.Pr_t))
    np.testing.assert_array_equal(np.asarray(a.C_K), np.asarray(f_ck))
    np.testing.assert_array_equal(np.asarray(a.Pr_t), np.asarray(f_prt))
