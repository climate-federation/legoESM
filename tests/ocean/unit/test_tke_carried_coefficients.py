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
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
import legoesm.ocean.experiments.dino as dino_mod
from legoesm.ocean.experiments.dino import DINO_RECIPES, dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config
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
