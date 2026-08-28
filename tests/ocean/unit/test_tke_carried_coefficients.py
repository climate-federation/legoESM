"""NEMO pre-tke_avn avm_k/avt_k lifetime regression tests."""
from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
import legoesm.ocean.physics.vertical_mixing.tke as tke_mod
import legoesm.ocean.physics.vertical_mixing._shared as shared_mod
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
import legoesm.ocean.experiments.dino as dino_mod
from legoesm.ocean.experiments.dino import DINO_RECIPES, dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config
from legoesm.ocean.fidelity.veros_acc_recipe import ACC_TKE_CONFIG
from legoesm.ocean.fidelity.veros_acc_basic_recipe import ACC_BASIC_TKE_CONFIG
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig


def _column_kwargs(cfg):
    z = jnp.zeros((1, 3), dtype=jnp.float64)
    return dict(
        u_cell=jnp.asarray([[0.0, 1.0, 3.0]]),
        v_cell=z, T_cell=z, S_cell=z,
        rho_cell=jnp.full((1, 3), 1026.0),
        dz_half=jnp.asarray([[1.0, 2.0]]),
        tke_old=jnp.ones((1, 2)),
        tau_x_surface=None, tau_y_surface=None, dt=2.0, cfg=cfg,
        n_iterations=1,
    )


def test_carried_coefficients_feed_matched_step_and_postsolve_prandtl(monkeypatch):
    """Hand case: carried pair feeds shear/matrix/RHS; new en feeds tke_avn."""
    cfg = TKEConfig(
        prognostic=True,
        tke_preclosure_coeff_source="carried_previous_step",
        prandtl_mode="nemo_ri", prandtl_ri_coeff=1.0,
        kappa_convention="veros_sqrte", c_k=1.0,
        kappaM_min=0.0, kappaH_min=0.0,
        enable_kappaH_profile=False, tke_background=1.0e-12,
        surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0",
    )
    n2 = jnp.asarray([[2.0, 4.0]])
    avm = jnp.asarray([[3.0, 5.0]])
    avt = jnp.asarray([[7.0, 11.0]])
    monkeypatch.setattr(tke_mod, "_compute_N2", lambda *a, **k: n2)
    monkeypatch.setattr(
        tke_mod, "compute_mixing_lengths",
        lambda *a, **k: (jnp.ones_like(n2), jnp.ones_like(n2)))

    seen = {}

    def fake_solve(**kw):
        seen.update(kw)
        return jnp.full_like(kw["e_old"], 4.0)

    monkeypatch.setattr(tke_mod, "_solve_tke_backward_euler", fake_solve)
    out = tke_mod.tke_vertical_mixing(
        **_column_kwargs(cfg), dz_surface=jnp.asarray([0.5]),
        preclosure_K_M=avm, preclosure_K_H=avt,
        preclosure_K_M_surface=jnp.asarray([13.0]))

    # du/dz = [1,1], hence p_sh2 = carried avm exactly. The same pair is the
    # matrix (K_M_old) and stratification-RHS (K_H_old) input.
    np.testing.assert_array_equal(seen["P_s"], np.asarray([[3.0, 5.0]]))
    np.testing.assert_array_equal(seen["K_M_old"], avm)
    np.testing.assert_array_equal(seen["K_H_old"], avt)
    np.testing.assert_array_equal(seen["K_M_surface"], [13.0])
    # Post-solve e=4 gives raw avm_new=2. Pr=[2,4] from
    # rn2b*avm_old/p_sh2, so avt_new=[1,0.5].
    np.testing.assert_array_equal(out.tke_new, np.asarray([[4.0, 4.0]]))
    np.testing.assert_allclose(out.K_M, [[2.0, 2.0]], rtol=0, atol=0)
    np.testing.assert_allclose(out.K_H, [[1.0, 0.5]], rtol=0, atol=0)


def test_carried_selector_fails_on_missing_or_ignored_carry():
    faithful = TKEConfig(
        prognostic=True,
        tke_preclosure_coeff_source="carried_previous_step")
    with pytest.raises(ValueError, match="requires both preclosure"):
        tke_mod.tke_vertical_mixing(**_column_kwargs(faithful))

    legacy = TKEConfig(prognostic=True)
    with pytest.raises(ValueError, match="current_subiteration"):
        tke_mod.tke_vertical_mixing(
            **_column_kwargs(legacy),
            preclosure_K_M=jnp.ones((1, 2)),
            preclosure_K_H=jnp.ones((1, 2)))


def test_only_complete_dino_nemo_cards_change_coefficient_lifetime():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get(
            "tke_preclosure_coeff_source", "current_subiteration")
        assert resolved == (
            "carried_previous_step" if name in faithful
            else "current_subiteration"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_preclosure_coeff_source == resolved, name

    # The generic/ORCA-oriented fidelity recipe has a separate certificate and
    # deliberately retains the pre-fix numerical lifetime in this DINO lane.
    assert _nemo_tke_config().tke_preclosure_coeff_source == "current_subiteration"


def test_only_complete_dino_nemo_cards_freeze_step_entry_shear():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get(
            "tke_shear_evaluation_stage", "implicit_solve_state")
        assert resolved == (
            "step_entry" if name in faithful else "implicit_solve_state"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_shear_evaluation_stage == resolved, name
            expected_metric = ("nemo_qco_live_face" if name in faithful
                               else "tpoint_jacobian")
            assert built.tke_shear_metric_source == expected_metric, name

    # Every independently constructed TKE card reachable outside the two
    # complete DINO oracle recipes remains on the byte-identical legacy path.
    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_shear_evaluation_stage == "implicit_solve_state"
               for c in unchanged)
    assert all(c.tke_shear_metric_source == "tpoint_jacobian"
               for c in unchanged)


def test_live_face_metric_product_matches_hand_computed_sh2():
    # Two identical U faces, no V shear. du_now=[-2,-3], du_before=[-4,-6],
    # avm face sums=[6,10]. The 0.25 two-face collapse gives [24,90]
    # before division; NOW*BEFORE metrics are [2*4,3*5] => [3,6].
    u_now = jnp.asarray([[[0.0, 2.0, 5.0], [0.0, 2.0, 5.0]]])
    u_before = jnp.asarray([[[0.0, 4.0, 10.0], [0.0, 4.0, 10.0]]])
    v_now = jnp.zeros((2, 1, 3))
    v_before = jnp.zeros((2, 1, 3))
    masks_u = jnp.ones_like(u_now)
    masks_v = jnp.ones_like(v_now)
    avm = jnp.asarray([[[3.0, 5.0]]])
    e3un = jnp.broadcast_to(jnp.asarray([2.0, 3.0]), (1, 2, 2))
    e3ub = jnp.broadcast_to(jnp.asarray([4.0, 5.0]), (1, 2, 2))
    e3vn = jnp.ones((2, 1, 2))
    e3vb = jnp.ones((2, 1, 2))
    got = shared_mod.avm_weighted_shear_production(
        u_now, v_now, u_before, v_before, jnp.ones((1, 1, 2)),
        masks_u, masks_v, avm,
        face_metrics=(e3un, e3ub, e3vn, e3vb))
    np.testing.assert_array_equal(got, np.asarray([[[3.0, 6.0]]]))

    with pytest.raises(ValueError, match="wrong shape"):
        shared_mod.avm_weighted_shear_production(
            u_now, v_now, u_before, v_before, jnp.ones((1, 1, 2)),
            masks_u, masks_v, avm,
            face_metrics=(e3un[..., :1], e3ub, e3vn, e3vb))


def test_step_entry_p_sh2_is_frozen_for_rhs_and_prandtl(monkeypatch):
    """Hand case: p_sh2=[12,20] survives a deliberately different solve state."""
    cfg = TKEConfig(
        prognostic=True,
        tke_preclosure_coeff_source="carried_previous_step",
        tke_shear_production="nemo_face_native",
        tke_shear_avm_weighting="nemo_face",
        tke_shear_evaluation_stage="step_entry",
        prandtl_mode="nemo_ri", prandtl_ri_coeff=1.0,
        kappa_convention="veros_sqrte", c_k=1.0,
        kappaM_min=0.0, kappaH_min=0.0,
        enable_kappaH_profile=False,
    )
    frozen = jnp.asarray([[12.0, 20.0]])
    avm = jnp.asarray([[3.0, 5.0]])
    avt = jnp.asarray([[7.0, 11.0]])
    monkeypatch.setattr(tke_mod, "_compute_N2",
                        lambda *a, **k: jnp.asarray([[2.0, 4.0]]))
    monkeypatch.setattr(
        tke_mod, "compute_mixing_lengths",
        lambda *a, **k: (jnp.ones_like(frozen), jnp.ones_like(frozen)))

    prandtl_seen = []

    def fake_compute_K(*args, **kwargs):
        prandtl_seen.append(np.asarray(
            kwargs["p_sh2_override"](jnp.full_like(frozen, 999.0))))
        return jnp.ones_like(frozen), jnp.ones_like(frozen)

    solve_seen = {}
    monkeypatch.setattr(tke_mod, "compute_K_from_tke", fake_compute_K)
    monkeypatch.setattr(
        tke_mod, "_solve_tke_backward_euler",
        lambda **kw: solve_seen.setdefault("P_s", kw["P_s"]) * 0.0 + kw["e_old"])

    kw = _column_kwargs(cfg)
    # These cell-centred arrays intentionally describe a different
    # implicit-solve state.  Frozen mode must not use them for p_sh2.
    kw["u_cell"] = jnp.asarray([[0.0, 100.0, -50.0]])
    out = tke_mod.tke_vertical_mixing(
        **kw, u_before_cell=jnp.zeros((1, 3)),
        v_before_cell=jnp.zeros((1, 3)),
        preclosure_K_M=avm, preclosure_K_H=avt,
        precomputed_p_sh2=frozen)
    np.testing.assert_array_equal(solve_seen["P_s"], frozen)
    assert len(prandtl_seen) == 2  # pre-solve and post-solve tke_avn
    for seen in prandtl_seen:
        np.testing.assert_array_equal(seen, frozen)
    np.testing.assert_array_equal(out.tke_new, kw["tke_old"])


def test_step_entry_selector_guards_and_legacy_stage_bit_identity():
    base = TKEConfig(prognostic=True)
    explicit = base._replace(
        tke_shear_evaluation_stage="implicit_solve_state")
    old = tke_mod.tke_vertical_mixing(**_column_kwargs(base))
    selected = tke_mod.tke_vertical_mixing(**_column_kwargs(explicit))
    for field in ("K_M", "K_H", "tke_new", "l_eps"):
        np.testing.assert_array_equal(getattr(selected, field), getattr(old, field))

    with pytest.raises(ValueError, match="precomputed_p_sh2 was supplied"):
        tke_mod.tke_vertical_mixing(
            **_column_kwargs(base), precomputed_p_sh2=jnp.ones((1, 2)))
    bad = base._replace(tke_shear_evaluation_stage="not-a-stage")
    with pytest.raises(ValueError, match="Unknown TKEConfig.tke_shear"):
        tke_mod.tke_vertical_mixing(**_column_kwargs(bad))
    missing = TKEConfig(
        prognostic=True, tke_shear_evaluation_stage="step_entry",
        tke_shear_production="nemo_face_native",
        tke_shear_avm_weighting="nemo_face")
    with pytest.raises(ValueError, match="requires precomputed_p_sh2"):
        tke_mod.tke_vertical_mixing(**_column_kwargs(missing))


def test_explicit_legacy_selector_is_bit_identical_to_old_default():
    implicit_legacy = TKEConfig(
        prognostic=True, prandtl_mode="constant",
        kappa_convention="veros_sqrte", enable_kappaH_profile=False)
    explicit_legacy = implicit_legacy._replace(
        tke_preclosure_coeff_source="current_subiteration")
    old = tke_mod.tke_vertical_mixing(**_column_kwargs(implicit_legacy))
    selected = tke_mod.tke_vertical_mixing(**_column_kwargs(explicit_legacy))
    for field in ("K_M", "K_H", "tke_new", "l_eps"):
        np.testing.assert_array_equal(getattr(selected, field), getattr(old, field))


def test_carried_coefficients_are_jittable_and_differentiable():
    cfg = TKEConfig(
        prognostic=True,
        tke_preclosure_coeff_source="carried_previous_step",
        prandtl_mode="constant", kappa_convention="veros_sqrte",
        enable_kappaH_profile=False, bottom_tke_bc=False,
    )

    def loss(tke_old, avm, avt):
        kw = _column_kwargs(cfg)
        kw["tke_old"] = tke_old
        out = tke_mod.tke_vertical_mixing(
            **kw, preclosure_K_M=avm, preclosure_K_H=avt)
        return jnp.sum(out.tke_new) + jnp.sum(out.K_M) + jnp.sum(out.K_H)

    tke_old = jnp.asarray([[0.2, 0.3]], dtype=jnp.float64)
    avm = jnp.asarray([[0.01, 0.02]], dtype=jnp.float64)
    avt = jnp.asarray([[0.005, 0.006]], dtype=jnp.float64)
    value, grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(
        tke_old, avm, avt)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in grads)


def test_cold_start_carry_uses_nemo_wmask_on_partial_depth_columns():
    cfg = TKEConfig(
        prognostic=True,
        tke_preclosure_coeff_source="carried_previous_step",
        kappaM_min=3.0, kappaH_min=5.0)
    State = namedtuple(
        "CarryState", "T land_mask tke_avm tke_avt tke_avm_surface")
    state = State(
        T=Field(jnp.zeros((1, 2, 4)), "T", ("lat", "lon", "level"), "K"),
        land_mask=Field(jnp.asarray([[1.0, 0.0]]), "land_mask",
                        ("lat", "lon"), "1"),
        tke_avm=None, tke_avt=None, tke_avm_surface=None)
    is_active = jnp.asarray([[[True, True, False, False],
                              [False, False, False, False]]])
    dummy = SimpleNamespace(
        config=SimpleNamespace(
            physics=SimpleNamespace(
                vertical_mixing=SimpleNamespace(tke=cfg))),
        z_coord=SimpleNamespace(is_active=is_active),
        _tke_prognostic_active=lambda: True,
    )
    out = LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, state)
    np.testing.assert_array_equal(out.tke_avm.data, [[[3.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]]])
    np.testing.assert_array_equal(out.tke_avt.data, [[[5.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]]])
    np.testing.assert_array_equal(out.tke_avm_surface.data, [[3.0, 0.0]])

    partial = state._replace(tke_avm=out.tke_avm)
    with pytest.raises(ValueError, match="partially populated"):
        LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, partial)


def test_postsolve_carry_is_closure_output_not_evd_composite():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import (
        EnhancedDiffusionConfig, OceanConvectionConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.vertical_mixing.k_profiles import (
        compute_vertical_K_profiles,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(3, 4)
    z_coord = create_ocean_z_star(4, H_max=200.0)
    state = rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=200.0)
    # Warm-below-cold is statically unstable, forcing the deliberately huge
    # EVD value into the composed solve coefficients.
    T = jnp.broadcast_to(jnp.linspace(5.0, 15.0, 4), state.T.data.shape)
    u = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
    v = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
    shape = state.T.data.shape[:-1] + (3,)
    state = state._replace(
        T=state.T.replace(data=T), u=state.u.replace(data=u),
        v=state.v.replace(data=v),
        tke=Field(jnp.full(shape, 0.1), "tke", ("lat", "lon", "level"),
                  "m^2/s^2"),
        tke_avm=Field(jnp.full(shape, 0.01), "tke_avm",
                      ("lat", "lon", "level"), "m^2/s"),
        tke_avt=Field(jnp.full(shape, 0.005), "tke_avt",
                      ("lat", "lon", "level"), "m^2/s"),
    )
    cfg = TKEConfig(
        prognostic=True,
        tke_preclosure_coeff_source="carried_previous_step",
        prandtl_mode="constant", kappa_convention="veros_sqrte",
        enable_kappaH_profile=False, kappaM_max=1.0)
    physics = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="tke", tke=cfg),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(
                K_conv=10.0, nu_conv=10.0)),
    )
    K_total, A_total, carry = compute_vertical_K_profiles(
        state, z_coord, None, physics, tke_old=state.tke.data,
        dt_tke=1.0, return_tke=True)
    assert float(jnp.max(K_total)) >= 10.0
    assert float(jnp.max(A_total)) >= 10.0
    assert float(jnp.max(carry.K_M)) <= 1.0
    assert float(jnp.max(carry.K_H)) < 10.0
