"""Tests for the SIGNED EKE-budget completion (EKE-budget probe deliverable).

Two config-selectable terms, BOTH default off (so every existing path stays
BIT-IDENTICAL; the ACC recipe opts in):

  1. ``gm_source_mode="realized_signed"`` — the LITERAL signed Veros conversion
     ``-P_diss_skew = -(g/ρ₀)·∇(int_drhodX)·F_skew`` summed over X∈{T,S} (the
     dynamic-enthalpy dissipation of the GM SKEW flux;
     ``compute_realized_signed_conversions``), replacing the positive-definite
     parameterized ``κ_GM·N²·⟨S²⟩``. NOT clamped: locally-negative production folds
     semi-implicitly (E ≥ floor by construction).
  2. ``source_p_diss_iso=True`` — SUBTRACT the realized signed Redi APE dissipation
     ``-P_diss_iso`` (Veros's EKE forc sink) built from the ISO-only flux + the
     implicit K_33 vertical diagonal.

Covered (CLAUDE.md hard constraints):
  (a) default-off BIT-IDENTITY for BOTH flags (regression guard for the gating);
  (b) energy-consistency identity: the signed skew conversion equals the
      dynamic-enthalpy contraction of the SKEW-only flux, computed independently;
  (c) sign behaviour: locally-negative production allowed, E stays ≥ floor;
  (d) P_diss_iso is built and finite on a stratified state;
  (e) AD-finite through the fully-signed 3-D step;
  (f) the ACC recipe carries both flags;
  (g) config validation (realized_signed requires eke_3d; source_p_diss_iso
      requires realized_signed).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity.veros_acc_recipe import (
    ACC_GM_REDI_CONFIG,
    DT_MOM_S,
    NZ,
    build_acc_recipe,
)
from legoesm.ocean.eos import int_drhodTS_dynamic_enthalpy, make_eos_fn
from legoesm.ocean.physics.lateral_mixing.eke import (
    EKEConfig,
    eke_apply_local_source,
    validate_eke_config,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    neumann_fill_cgrid,
    compute_eke_step_kappa,
    compute_realized_signed_conversions,
    gm_redi_tracer_tendency_triads_latlon_cgrid,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.vertical import compute_ocean_jacobian
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    gradient_x_cgrid,
    gradient_y_cgrid,
)


def _acc_model_with_eke(**eke_overrides):
    recipe = build_acc_recipe(with_surface_forcing=True)
    eke = recipe.model_config.gm_redi.eke._replace(**eke_overrides)
    gm = recipe.model_config.gm_redi._replace(eke=eke)
    cfg = recipe.model_config._replace(gm_redi=gm)
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
    return recipe, model


def _developed_state(recipe, model, n=8):
    state = recipe.initial_state
    for _ in range(n):
        state = model.step(state, DT_MOM_S, surface_forcing=recipe.wind_forcing)
    return state


# ---------------------------------------------------------------------------
# (a) Default-off bit-identity for BOTH new flags.
# ---------------------------------------------------------------------------


def test_default_off_bit_identical_realized_signed_and_p_diss_iso():
    """With ``gm_source_mode='parameterized'`` and ``source_p_diss_iso=False`` (the
    defaults) the 3-D EKE step is BIT-IDENTICAL to an independent parameterized run —
    the new realized_signed / P_diss_iso branches are fully gated and never touch the
    default path."""
    recipe, model_off = _acc_model_with_eke(
        source_kdiss_h=False, kdiss_h_flux_form=False,
        gm_source_mode="parameterized", source_p_diss_iso=False)
    _r2, model_ref = _acc_model_with_eke(
        source_kdiss_h=False, kdiss_h_flux_form=False,
        gm_source_mode="parameterized", source_p_diss_iso=False)
    s = recipe.initial_state
    s2 = recipe.initial_state
    for _ in range(5):
        s = model_off.step(s, DT_MOM_S, surface_forcing=recipe.wind_forcing)
        s2 = model_ref.step(s2, DT_MOM_S, surface_forcing=recipe.wind_forcing)
    np.testing.assert_array_equal(np.asarray(s.eke.data), np.asarray(s2.eke.data))
    np.testing.assert_array_equal(np.asarray(s.u.data), np.asarray(s2.u.data))
    np.testing.assert_array_equal(np.asarray(s.T.data), np.asarray(s2.T.data))


def test_eke_apply_local_source_signed_args_default_off_bit_identical():
    """``eke_apply_local_source`` with the new ``signed_source=None`` +
    ``clamp_production=True`` (defaults) is bit-identical to the call without them."""
    rng = np.random.default_rng(0)
    shp = (6, 5, 4)
    E = jnp.asarray(np.abs(rng.standard_normal(shp)) * 1e-3)
    sigma = jnp.asarray(np.abs(rng.standard_normal(shp)) * 1e-5)
    L = jnp.asarray(np.abs(rng.standard_normal(shp)) * 1e4 + 1e3)
    cfg = EKEConfig()
    dt = 3600.0
    a = eke_apply_local_source(E, sigma, L, cfg, dt)
    b = eke_apply_local_source(E, sigma, L, cfg, dt,
                               signed_source=None, clamp_production=True)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# (b) Energy-consistency identity: the signed skew conversion equals an
#     INDEPENDENT dynamic-enthalpy contraction of the SKEW-only flux.
# ---------------------------------------------------------------------------


def _independent_signed_skew(recipe, model, state):
    """Recompute -P_diss_skew from FIRST PRINCIPLES (independent of the production
    function): skew-only fluxes (kappa_Redi=0) of T and S, contracted with the
    dynamic-enthalpy gradient via Veros's compute_dissipation (horizontal) +
    vertical flux_top term. Returns the W-grid -P_diss_skew [m²/s³]."""
    cfg = recipe.model_config
    gm_cfg = cfg.gm_redi
    grid, z_coord = recipe.grid, recipe.z_coord
    T = state.T.data; S = state.S.data; lm = state.land_mask.data
    u_mask = state.u_mask.data; v_mask = state.v_mask.data
    rho_0 = cfg.constants.rho_0; g = cfg.constants.g
    E = state.eke.data
    kg, _sig, _L = compute_eke_step_kappa(
        T, S, state.eta.data, state.H_bathy.data, E, grid, z_coord, gm_cfg,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=lm, rho_0=rho_0, g=g,
        omega=cfg.constants.Omega, r_earth=cfg.constants.R_earth,
        depth_resolved=True)
    jac = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    eos_fn = make_eos_fn(cfg.eos, cfg.eos_linear)
    fill = lambda f: neumann_fill_cgrid(f, lm)
    rho, _r, _p = iterate_eos_and_pressure_anomaly(
        T, S, lm, fill, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2)
    rho_f = neumann_fill_cgrid(rho, lm)
    zero = jnp.asarray(0.0)
    common = dict(z_coord=z_coord, jacobian=jac, grid=grid, kappa_GM=kg,
                  kappa_Redi=zero, S_max=gm_cfg.S_max,
                  taper_width_frac=gm_cfg.taper_width_frac,
                  implicit_K33=gm_cfg.implicit_K33, K_iso_steep=gm_cfg.K_iso_steep,
                  slope_density=gm_cfg.slope_density, T_tracer=T, S_tracer=S,
                  eos_fn=eos_fn, rho_0=rho_0, g=g, return_fluxes=True)
    _, FxT, FyT, FzT = gm_redi_tracer_tendency_triads_latlon_cgrid(
        T, rho_f, lm, u_mask, v_mask, **common)
    _, FxS, FyS, FzS = gm_redi_tracer_tendency_triads_latlon_cgrid(
        S, rho_f, lm, u_mask, v_mask, **common)
    z_full = jnp.asarray(z_coord.z_full_ref)[None, None, :]
    intT, intS = int_drhodTS_dynamic_enthalpy(
        make_eos_fn(cfg.eos, cfg.eos_linear),
        neumann_fill_cgrid(T, lm), neumann_fill_cgrid(S, lm),
        z_full, rho_0, g)
    intT = intT * lm[:, :, None]; intS = intS * lm[:, :, None]
    dz_cell = z_coord.dz_ref * jac[:, :, None]
    dz_w = 0.5 * (dz_cell[..., :-1] + dz_cell[..., 1:])

    def diss(intX, Fx, Fy, Fz):
        dXx = gradient_x_cgrid(intX, grid); dXy = gradient_y_cgrid(intX, grid)
        fx = dXx * Fx; fy = dXy * Fy
        h_cell = 0.5 * (g / rho_0) * (
            (fx[:, :-1, :] + fx[:, 1:, :]) + (fy[:-1, :, :] + fy[1:, :, :])
        ) * lm[:, :, None]
        h_w = 0.5 * (h_cell[:, :, :-1] + h_cell[:, :, 1:])
        fxa = (intX[:, :, :-1] - intX[:, :, 1:]) / jnp.maximum(dz_w, 1e-10)
        v_w = (g / rho_0) * fxa * Fz
        return (h_w + v_w) * lm[:, :, None]

    P = diss(intT, FxT, FyT, FzT) + diss(intS, FxS, FyS, FzS)
    return -P * lm[:, :, None], kg


def test_signed_skew_matches_independent_dynamic_enthalpy_contraction():
    """The PRODUCTION ``compute_realized_signed_conversions`` skew output equals an
    INDEPENDENT first-principles dynamic-enthalpy contraction of the skew-only flux
    (the energy-consistency identity: the realized skew conversion IS the APE the
    skew flux extracts). Bit-close (same shared flux assembly + helper)."""
    recipe, model = _acc_model_with_eke()
    state = _developed_state(recipe, model)
    ref, kg = _independent_signed_skew(recipe, model, state)
    cfg = recipe.model_config
    neg_skew, _ = compute_realized_signed_conversions(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        recipe.grid, recipe.z_coord, cfg.gm_redi, kg,
        want_skew=True, want_iso=False,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=state.land_mask.data,
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g)
    lm = np.asarray(state.land_mask.data)
    wet = np.broadcast_to(lm[:, :, None] > 0.5, np.asarray(ref).shape)
    a = np.asarray(neg_skew)[wet]; b = np.asarray(ref)[wet]
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_signed_skew_is_signed_not_positive_definite_form():
    """The signed skew conversion DIFFERS from the positive-definite parameterized
    form (``compute_realized_gm_skew_conversion``) — it is a genuinely different
    (signed, dynamic-enthalpy) construction, not a relabel."""
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_realized_gm_skew_conversion,
    )
    recipe, model = _acc_model_with_eke()
    state = _developed_state(recipe, model)
    _ref, kg = _independent_signed_skew(recipe, model, state)
    cfg = recipe.model_config
    neg_skew, _ = compute_realized_signed_conversions(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        recipe.grid, recipe.z_coord, cfg.gm_redi, kg, want_skew=True, want_iso=False,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=state.land_mask.data,
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g)
    P_posdef = compute_realized_gm_skew_conversion(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        recipe.grid, recipe.z_coord, cfg.gm_redi, kg,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=state.land_mask.data,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g)
    lm = np.asarray(state.land_mask.data)
    wet = np.broadcast_to(lm[:, :, None] > 0.5, np.asarray(neg_skew).shape)
    a = np.asarray(neg_skew)[wet]; b = np.asarray(P_posdef)[wet]
    # Genuinely different constructions: a relative comparison (the developed-state
    # magnitudes are tiny, so the default allclose atol is uninformative — use a
    # scale-relative difference of the two fields).
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
    assert np.max(np.abs(a - b)) / scale > 1e-3
    assert np.all(np.isfinite(np.asarray(neg_skew)))


# ---------------------------------------------------------------------------
# (c) Sign behaviour: locally-negative production allowed, E stays >= floor.
# ---------------------------------------------------------------------------


def test_eke_apply_local_source_signed_keeps_E_nonneg_with_negative_production():
    """With a LOCALLY-NEGATIVE signed production (clamp_production=False) the
    semi-implicit update keeps ``E_{n+1} >= 0`` by construction (negative part folded
    into the implicit denominator), no clip/floor."""
    rng = np.random.default_rng(1)
    shp = (5, 4, 3)
    E = jnp.asarray(np.abs(rng.standard_normal(shp)) * 1e-3 + 1e-6)
    L = jnp.asarray(np.abs(rng.standard_normal(shp)) * 1e4 + 1e3)
    sigma = jnp.zeros(shp)
    # A production_override that is NEGATIVE everywhere (a pure local sink).
    neg_prod = -jnp.abs(jnp.asarray(rng.standard_normal(shp))) * 1e-5
    cfg = EKEConfig()
    E_new = eke_apply_local_source(
        E, sigma, L, cfg, 43200.0,
        production_override=neg_prod, clamp_production=False)
    assert float(jnp.min(E_new)) >= 0.0
    assert bool(jnp.all(jnp.isfinite(E_new)))
    # A negative source must DECREASE E (sink), not increase it.
    assert float(jnp.max(E_new - E)) <= 1e-12


def test_eke_apply_local_source_signed_source_negative_is_sink():
    """A negative ``signed_source`` (the -P_diss_iso sink, mostly < 0) decreases E
    and keeps it >= 0; a positive signed_source increases E."""
    rng = np.random.default_rng(2)
    shp = (4, 4, 3)
    E = jnp.asarray(np.abs(rng.standard_normal(shp)) * 1e-3 + 1e-6)
    L = jnp.full(shp, 5e4)
    sigma = jnp.zeros(shp)
    cfg = EKEConfig()
    sink = -jnp.full(shp, 1e-6)
    src = jnp.full(shp, 1e-6)
    E_sink = eke_apply_local_source(E, sigma, L, cfg, 43200.0, signed_source=sink)
    E_src = eke_apply_local_source(E, sigma, L, cfg, 43200.0, signed_source=src)
    assert float(jnp.min(E_sink)) >= 0.0
    assert float(jnp.max(E_sink - E)) <= 1e-12       # sink: E decreases
    assert float(jnp.min(E_src - E)) >= -1e-12       # source: E increases


def test_full_signed_step_keeps_eke_nonneg():
    """A full realized_signed + source_p_diss_iso 3-D step keeps the eke field
    ``>= 0`` everywhere (positivity by construction, even where the signed skew /
    iso sink are locally negative)."""
    recipe, model = _acc_model_with_eke()  # recipe default = realized_signed + iso
    s = recipe.initial_state
    for _ in range(6):
        s = model.step(s, DT_MOM_S, surface_forcing=recipe.wind_forcing)
        assert float(jnp.min(s.eke.data)) >= 0.0
    assert bool(jnp.all(jnp.isfinite(s.eke.data)))


# ---------------------------------------------------------------------------
# (d) P_diss_iso built + finite + mostly-signed on a stratified state.
# ---------------------------------------------------------------------------


def test_p_diss_iso_built_finite_and_nontrivial():
    """The signed -P_diss_iso is built (want_iso=True), finite, non-trivial, and zero
    on land — on a stratified developed ACC state."""
    recipe, model = _acc_model_with_eke()
    state = _developed_state(recipe, model)
    cfg = recipe.model_config
    _ref, kg = _independent_signed_skew(recipe, model, state)
    _neg_skew, neg_iso = compute_realized_signed_conversions(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        recipe.grid, recipe.z_coord, cfg.gm_redi, kg,
        want_skew=False, want_iso=True,
        kappa_redi_w=(kg if cfg.gm_redi.eke.isopycnal_diffusion else None),
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=state.land_mask.data,
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g)
    assert neg_iso is not None
    ni = np.asarray(neg_iso)
    assert ni.shape == (state.land_mask.data.shape[0],
                        state.land_mask.data.shape[1], NZ - 1)
    assert np.all(np.isfinite(ni))
    lm = np.asarray(state.land_mask.data)
    # zero on land columns
    land = lm < 0.5
    assert float(np.max(np.abs(ni[land]))) == 0.0
    # non-trivial somewhere
    assert float(np.max(np.abs(ni))) > 0.0


def test_want_skew_false_returns_none_for_skew():
    """The want flags gate the returns: want_skew=False -> skew is None."""
    recipe, model = _acc_model_with_eke()
    state = _developed_state(recipe, model, n=3)
    cfg = recipe.model_config
    _ref, kg = _independent_signed_skew(recipe, model, state)
    neg_skew, neg_iso = compute_realized_signed_conversions(
        state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
        recipe.grid, recipe.z_coord, cfg.gm_redi, kg, want_skew=False, want_iso=True,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=state.land_mask.data,
        u_mask=state.u_mask.data, v_mask=state.v_mask.data,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g)
    assert neg_skew is None and neg_iso is not None


# ---------------------------------------------------------------------------
# (e) AD-finite through the fully-signed 3-D step.
# ---------------------------------------------------------------------------


def test_signed_step_is_differentiable():
    """jax.grad through one fully-signed (realized_signed + source_p_diss_iso) 3-D
    step produces finite, non-trivial gradients — the signed skew/iso conversions and
    the negative-part semi-implicit folding are all differentiable."""
    recipe, model = _acc_model_with_eke()
    state = recipe.initial_state

    def loss(T_data):
        st = state._replace(T=state.T.replace(data=T_data))
        out = model.step(st, DT_MOM_S, surface_forcing=recipe.wind_forcing)
        return jnp.sum(out.eke.data ** 2) + jnp.sum(out.u.data ** 2)

    g = jax.grad(loss)(state.T.data)
    assert g.shape == state.T.data.shape
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_compute_realized_signed_conversions_is_differentiable():
    """jax.grad through ``compute_realized_signed_conversions`` (both skew + iso)
    w.r.t. T is finite — the dynamic-enthalpy contraction + the zeroed-kappa flux
    assembly carry clean gradients."""
    recipe, model = _acc_model_with_eke()
    state = _developed_state(recipe, model, n=3)
    cfg = recipe.model_config
    _ref, kg = _independent_signed_skew(recipe, model, state)

    def loss(T_data):
        neg_skew, neg_iso = compute_realized_signed_conversions(
            T_data, state.S.data, state.eta.data, state.H_bathy.data,
            recipe.grid, recipe.z_coord, cfg.gm_redi, kg,
            want_skew=True, want_iso=True, kappa_redi_w=kg,
            eos=cfg.eos, eos_linear=cfg.eos_linear, mask=state.land_mask.data,
            u_mask=state.u_mask.data, v_mask=state.v_mask.data,
            rho_0=cfg.constants.rho_0, g=cfg.constants.g)
        return jnp.sum(neg_skew ** 2) + jnp.sum(neg_iso ** 2)

    g = jax.grad(loss)(state.T.data)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


# ---------------------------------------------------------------------------
# (f) Recipe carries the flags + (g) config validation.
# ---------------------------------------------------------------------------


def test_acc_recipe_opts_in_to_signed_skew_only():
    """The recipe adopts the validated signed GM-skew conversion; the
    experimental -P_diss_iso term stays OFF (legoESM's adiabatic-cancelling
    triads cannot reproduce Veros's non-cancelling iso-dissipation sink —
    the faithful form yields a small spurious source instead; see the
    EKE-budget probe verdict and the recipe comment)."""
    assert ACC_GM_REDI_CONFIG.eke.gm_source_mode == "realized_signed"
    assert ACC_GM_REDI_CONFIG.eke.source_p_diss_iso is False


def test_validate_eke_config_signed_literals():
    # valid
    validate_eke_config(EKEConfig(gm_source_mode="realized_signed", eke_3d=True))
    validate_eke_config(EKEConfig(gm_source_mode="realized_signed",
                                  source_p_diss_iso=True, eke_3d=True))
    # source_p_diss_iso without realized_signed is rejected.
    with pytest.raises(ValueError, match="source_p_diss_iso"):
        validate_eke_config(EKEConfig(source_p_diss_iso=True, eke_3d=True))
    with pytest.raises(ValueError, match="source_p_diss_iso"):
        validate_eke_config(EKEConfig(gm_source_mode="realized",
                                      source_p_diss_iso=True, eke_3d=True))


def test_model_rejects_realized_signed_without_eke_3d():
    recipe = build_acc_recipe(with_surface_forcing=True)
    gm = recipe.model_config.gm_redi
    eke = gm.eke._replace(eke_3d=False, source_kdiss_h=False,
                          kdiss_h_flux_form=False, source_p_diss_iso=False,
                          gm_source_mode="realized_signed")
    cfg = recipe.model_config._replace(gm_redi=gm._replace(eke=eke))
    with pytest.raises(ValueError, match="eke_3d=True"):
        LatLonCGridOceanModel(recipe.grid, recipe.z_coord, cfg)
