"""NEMO pre-tke_avn avm_k/avt_k lifetime regression tests."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import legoesm.ocean.physics.vertical_mixing.tke as tke_mod
from legoesm.ocean.experiments.dino import DINO_RECIPES
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
        **_column_kwargs(cfg), preclosure_K_M=avm, preclosure_K_H=avt)

    # du/dz = [1,1], hence p_sh2 = carried avm exactly. The same pair is the
    # matrix (K_M_old) and stratification-RHS (K_H_old) input.
    np.testing.assert_array_equal(seen["P_s"], np.asarray([[3.0, 5.0]]))
    np.testing.assert_array_equal(seen["K_M_old"], avm)
    np.testing.assert_array_equal(seen["K_H_old"], avt)
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
