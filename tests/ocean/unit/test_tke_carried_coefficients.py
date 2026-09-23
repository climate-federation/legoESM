"""NEMO pre-tke_avn avm_k/avt_k lifetime regression tests."""
from __future__ import annotations

from collections import namedtuple
from types import MethodType, SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm import constants
import legoesm.ocean.physics.vertical_mixing.tke as tke_mod
import legoesm.ocean.physics.vertical_mixing._shared as shared_mod
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
import legoesm.ocean.experiments.dino as dino_mod
from legoesm.ocean.experiments.dino import DINO_RECIPES, dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config
from legoesm.ocean.fidelity.veros_acc_recipe import ACC_TKE_CONFIG
from legoesm.ocean.fidelity.veros_acc_basic_recipe import ACC_BASIC_TKE_CONFIG
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig


def _step_entry_helper_fixture(tke_cfg):
    from legoesm.ocean.vertical import (
        create_partial_cell_coordinate, create_z_star_from_thicknesses,
    )

    ny, nx, nz = 2, 2, 3
    gdept0 = np.broadcast_to(
        np.asarray([0.5, 2.0, 4.5]), (ny, nx, nz)).copy()
    gdepw0 = np.broadcast_to(
        np.asarray([0.0, 1.0, 3.0]), (ny, nx, nz)).copy()
    # Partial-cell-like column variation: replacing this with the 1-D ladder
    # changes the literal eosbn2 result and must turn the integration test red.
    gdepw0[0, 1, 1:] = np.asarray([0.9, 2.8])
    e3w0 = np.broadcast_to(
        np.asarray([1.0, 1.5, 2.5]), (ny, nx, nz)).copy()
    e3t0 = np.broadcast_to(
        np.asarray([1.0, 2.0, 3.0]), (ny, nx, nz)).copy()
    horizontal = np.ones((ny, nx))
    raw = create_z_star_from_thicknesses(
        [1.0, 2.0, 3.0], nemo_gdept_0_m=gdept0,
        nemo_gdepw_0_m=gdepw0,
        nemo_e3t_0_m=e3t0, nemo_e3w_0_m=e3w0, nemo_hu_0_m=horizontal,
        nemo_hv_0_m=horizontal, nemo_e1e2t_m=horizontal,
        nemo_e1e2u_m=horizontal, nemo_e1e2v_m=horizontal)
    H = jnp.full((ny, nx), raw.H_max)
    z_coord = create_partial_cell_coordinate(raw, H)
    cfg = SimpleNamespace(
        physics=SimpleNamespace(vertical_mixing=
            SimpleNamespace(scheme="tke", tke=tke_cfg)),
        constants=SimpleNamespace(g=constants.g))
    model = SimpleNamespace(config=cfg, z_coord=z_coord)
    dims3 = ("lat", "lon", "level")
    dims2 = ("lat", "lon")
    u_now = jnp.asarray([
        [[0.0, 1.0, 3.0], [0.2, 2.0, 5.0], [0.0, 1.0, 4.0]],
        [[0.5, 1.5, 4.0], [0.1, 1.0, 2.0], [0.5, 2.5, 3.5]],
    ])
    v_now = jnp.asarray([
        [[0.0, 0.5, 2.0], [0.2, 1.0, 3.0]],
        [[0.1, 1.0, 2.5], [0.4, 1.5, 4.0]],
        [[0.0, 0.7, 3.0], [0.3, 2.0, 3.5]],
    ])
    T_now = jnp.asarray([[[12.0, 11.0, 9.0], [13.0, 11.5, 8.5]],
                         [[10.0, 9.0, 7.5], [14.0, 12.0, 10.0]]])
    S_now = jnp.asarray([[[35.0, 35.1, 35.2], [34.9, 35.0, 35.3]],
                         [[35.2, 35.25, 35.4], [34.8, 35.0, 35.1]]])
    state = SimpleNamespace(
        T=Field(T_now, "T", dims3, "degC"),
        S=Field(S_now, "S", dims3, "PSU"),
        T_before=Field(T_now + 0.2, "T_before", dims3, "degC"),
        S_before=Field(S_now - 0.03, "S_before", dims3, "PSU"),
        H_bathy=Field(H, "H_bathy", dims2, "m"),
        eta=Field(jnp.asarray([[0.2, -0.1], [0.05, 0.15]]),
                  "eta", dims2, "m"),
        eta_before=Field(jnp.asarray([[-0.05, 0.1], [0.02, -0.08]]),
                         "eta_before", dims2, "m"),
        u=Field(u_now, "u", dims3, "m/s"),
        v=Field(v_now, "v", dims3, "m/s"),
        u_before=Field(u_now * 0.8 + 0.03, "u_before", dims3, "m/s"),
        v_before=Field(v_now * 1.1 - 0.02, "v_before", dims3, "m/s"),
        tke_avm=Field(jnp.asarray([[[0.3, 0.5], [0.2, 0.6]],
                                   [[0.4, 0.7], [0.25, 0.45]]]),
                      "tke_avm", dims3, "m2/s"),
    )
    model._n2_nemo_before_tracers = MethodType(
        LatLonCGridOceanModel._n2_nemo_before_tracers, model)
    return model, state


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
        dz_ref=jnp.asarray([1.0, 2.0, 3.0]), jacobian=jnp.ones((1,)),
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


def test_only_complete_dino_nemo_cards_select_literal_tke_matrix():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get("tke_matrix_evaluation", "factored")
        assert resolved == (
            "nemo_literal" if name in faithful else "factored"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_matrix_evaluation == resolved, name

    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_matrix_evaluation == "factored" for c in unchanged)


def test_only_complete_dino_nemo_cards_select_literal_tke_solver():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get("tke_solver_evaluation", "shared_thomas")
        assert resolved == (
            "nemo_literal" if name in faithful else "shared_thomas"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_solver_evaluation == resolved, name

    # These are the non-DINO consumers that reach the same production solve.
    # Their selector and therefore their numerical path remain unchanged.
    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_solver_evaluation == "shared_thomas" for c in unchanged)


@pytest.mark.parametrize("card_name", [
    "nemo_paper", "veros", "generic_nemo", "acc", "acc_basic",
])
def test_every_unchanged_tke_card_keeps_shared_solver_bits(card_name,
                                                           monkeypatch):
    dino_cards = {
        name: dino_mod._dino_vertical_mixing_config(
            dino_config_for_recipe(name)).tke
        for name in ("nemo_paper", "veros")
    }
    cards = {
        **dino_cards,
        "generic_nemo": _nemo_tke_config(),
        "acc": ACC_TKE_CONFIG,
        "acc_basic": ACC_BASIC_TKE_CONFIG,
    }
    cfg = cards[card_name]
    assert cfg.tke_solver_evaluation == "shared_thomas"

    def literal_must_not_run(*args, **kwargs):
        raise AssertionError(f"literal solver reached by unchanged {card_name}")

    monkeypatch.setattr(tke_mod, "_nemo_literal_tke_solve",
                        literal_must_not_run)
    common = dict(
        e_old=jnp.asarray([[1.0, 0.7]]),
        K_M_old=jnp.asarray([[0.2, 0.3]]),
        K_H_old=jnp.asarray([[0.1, 0.15]]),
        P_s=jnp.asarray([[0.01, 0.02]]),
        N2=jnp.asarray([[1.0e-5, -1.0e-5]]),
        l_eps=jnp.asarray([[1.0, 1.2]]),
        dz_half=jnp.asarray([[2.0, 3.0]]),
        surface_flux=jnp.asarray([0.001]), dt=2.0,
        surface_dirichlet=(
            jnp.asarray([0.8]) if cfg.surface_bc == "nemo_dirichlet"
            else None),
        surface_bc_level=cfg.tke_surface_bc_level,
    )
    implicit = tke_mod._solve_tke_backward_euler(cfg=cfg, **common)
    explicit = tke_mod._solve_tke_backward_euler(
        cfg=cfg._replace(tke_solver_evaluation="shared_thomas"), **common)
    np.testing.assert_array_equal(explicit, implicit)


def test_only_complete_dino_nemo_cards_select_literal_langmuir():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get("tke_langmuir_evaluation", "vectorized")
        assert resolved == (
            "nemo_literal" if name in faithful else "vectorized"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_langmuir_evaluation == resolved, name

    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_langmuir_evaluation == "vectorized" for c in unchanged)


def test_only_complete_dino_nemo_cards_select_literal_etau_exp():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get(
            "tke_etau_exponential_evaluation", "jax_expression")
        assert resolved == (
            "nemo_literal" if name in faithful else "jax_expression"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_etau_exponential_evaluation == resolved, name

    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_etau_exponential_evaluation == "jax_expression"
               for c in unchanged)


def test_only_complete_dino_nemo_cards_select_literal_htau():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get("tke_htau_evaluation", "jax_expression")
        assert resolved == (
            "nemo_literal" if name in faithful else "jax_expression"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_htau_evaluation == resolved, name

    # All independently constructed consumers retain their historical path.
    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_htau_evaluation == "jax_expression" for c in unchanged)


def test_only_complete_dino_nemo_cards_select_literal_raw_mxl():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get("tke_mxl_raw_evaluation", "factored")
        assert resolved == (
            "nemo_literal" if name in faithful else "factored"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_mxl_raw_evaluation == resolved, name
    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_mxl_raw_evaluation == "factored" for c in unchanged)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
def test_literal_raw_mxl_matches_hand_computed_source_order_and_red_control():
    e = jnp.asarray([[float.fromhex("0x1.f59eccb1e87d9p-15"), 2.0e-6]])
    n2 = jnp.asarray([[float.fromhex("-0x1.bb25cc435cf35p-32"), 1.0e-5]])
    dz = jnp.full((1, 3), 1.0e12)
    cfg = TKEConfig(tke_mxl_choice=3, mxl_min=0.01,
                    tke_mxl_raw_evaluation="nemo_literal")
    lk, _ = tke_mod.compute_mixing_lengths(e, n2, dz[..., :2], cfg,
                                            dz_cell=dz)
    rsmall = 0.5 * np.finfo(np.float64).eps
    raw = np.maximum(0.01, np.sqrt((2.0 * np.asarray(e))
                                   / np.maximum(np.asarray(n2), rsmall)))
    # With a huge dz allowance, the scan retains the raw physical interior;
    # the final carried slot is NEMO's untouched jpk pad.
    raw[..., -1] = 0.01
    np.testing.assert_array_equal(np.asarray(lk).view(np.uint64),
                                  raw.view(np.uint64))

    factored, _ = tke_mod.compute_mixing_lengths(
        e, n2, dz[..., :2],
        cfg._replace(tke_mxl_raw_evaluation="factored"), dz_cell=dz)
    assert np.asarray(factored)[0, 0].view(np.uint64) != raw[0, 0].view(np.uint64)

    with pytest.raises(ValueError, match="tke_mxl_raw_evaluation"):
        tke_mod.compute_mixing_lengths(
            e, n2, dz[..., :2],
            cfg._replace(tke_mxl_raw_evaluation="unknown"), dz_cell=dz)


def test_literal_mxl_ldown_keeps_jpk_terminal_seed_unmodified():
    e = jnp.asarray([[2.0, 3.0, 1.0e8]])
    n2 = jnp.full_like(e, 1.0e-12)
    dz_cell = jnp.asarray([[1.0, 2.0, 3.0, 4.0]])
    cfg = TKEConfig(tke_mxl_choice=3, mxl_min=0.01,
                    mxl0_min_m=0.01,
                    tke_mxl_raw_evaluation="nemo_literal")
    lk, _ = tke_mod.compute_mixing_lengths(
        e, n2, jnp.ones_like(e), cfg, dz_cell=dz_cell,
        l_surface_anchor=jnp.asarray([0.01]))
    # The final carried slot is NEMO's untouched jpk pad, not another raw
    # buoyancy-length row.  The old recurrence produced 4.01 here.
    assert float(lk[0, -1]) == 0.01
    legacy, _ = tke_mod.compute_mixing_lengths(
        e, n2, jnp.ones_like(e),
        cfg._replace(tke_mxl_raw_evaluation="factored"),
        dz_cell=dz_cell, l_surface_anchor=jnp.asarray([0.01]))
    assert float(legacy[0, -1]) != 0.01


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
def test_literal_etau_exp_matches_oracle_bits_and_one_ulp_control_fires():
    # First selected row-18 ownership-control operand, plus ordinary-range
    # values spanning the DINO ladder.  Expected bits are from the committed
    # NEMO EXP dump / linked _ZGVbN2v_exp receipt.
    argument = np.asarray([
        float.fromhex("-0x1.5fc93e9bbda71p-2"),
        float.fromhex("-0x1.063a29f68dac2p+3"),
        -10.0,
        -0.0001,
    ], dtype=np.float64)
    expected = np.asarray([
        float.fromhex("0x1.6b23619ba571ap-1"),
        float.fromhex("0x1.218df37ee5074p-12"),
        float.fromhex("0x1.7cd79b5647c9ap-15"),
        float.fromhex("0x1.fff2e4b97d31dp-1"),
    ], dtype=np.float64)
    actual = np.asarray(jax.jit(tke_mod._nemo_glibc234_vector_exp)(
        jnp.asarray(argument)))
    np.testing.assert_array_equal(actual.view(np.uint64),
                                  expected.view(np.uint64))

    planted = argument.copy()
    planted[0] = np.nextafter(planted[0], np.inf)
    planted_actual = np.asarray(tke_mod._nemo_glibc234_vector_exp(
        jnp.asarray(planted)))
    # Red-capable control: the same exactness assertion would fail after the
    # registered one-ULP argument plant.
    assert planted_actual[0].view(np.uint64) != expected[0].view(np.uint64)


def test_glibc_exp_table_full_payload_sha_and_mutation_control():
    import hashlib
    import struct

    from legoesm.ocean.physics.vertical_mixing._glibc234_exp_table import (
        GLIBC234_EXP_TABLE_BITS,
        GLIBC234_EXP_TABLE_SHA256,
    )

    payload = struct.pack("<1024Q", *GLIBC234_EXP_TABLE_BITS)
    assert hashlib.sha256(payload).hexdigest() == GLIBC234_EXP_TABLE_SHA256
    planted = bytearray(payload)
    planted[8 * 511] ^= 1
    assert hashlib.sha256(planted).hexdigest() != GLIBC234_EXP_TABLE_SHA256


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
def test_literal_etau_exp_jit_and_ad_are_finite():
    argument = jnp.asarray([-0.3, -1.0], dtype=jnp.float64)
    value = jax.jit(tke_mod._nemo_glibc234_vector_exp)(argument)
    forward = jax.jacfwd(tke_mod._nemo_glibc234_vector_exp)(argument)
    reverse = jax.jacrev(tke_mod._nemo_glibc234_vector_exp)(argument)
    assert np.all(np.isfinite(np.asarray(value)))
    assert np.all(np.isfinite(np.asarray(forward)))
    assert np.all(np.isfinite(np.asarray(reverse)))


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
def test_literal_etau_exp_exceptional_range_guard_and_float32_fallback():
    exceptional = jnp.asarray([-800.0, jnp.inf, -jnp.inf], dtype=jnp.float64)
    guarded = jax.jit(tke_mod._nemo_glibc234_vector_exp)(exceptional)
    fallback = jax.jit(jnp.exp)(exceptional)
    np.testing.assert_array_equal(
        np.asarray(guarded).view(np.uint64),
        np.asarray(fallback).view(np.uint64))

    # The dtype guard is independently red-capable: removing it would make
    # the uint64 bit assembly invalid for this float32 input.
    fp32 = jnp.asarray([-0.3, -80.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(tke_mod._nemo_glibc234_vector_exp(fp32)),
        np.asarray(jnp.exp(fp32)))


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
def test_literal_htau_sin_matches_glibc_vector_bits_and_ulp_control_fires():
    # Native DINO gphit phases at rows 0, 48, the equator, 150 and 198.
    # Expected values are the committed host glibc-2.34 _ZGVbN2v_sin receipt.
    phase = np.asarray([
        float.fromhex("-0x1.3819bd7e1e04dp+0"),
        float.fromhex("-0x1.9547a4d0d912dp-1"),
        0.0,
        float.fromhex("0x1.9547a4d0d912dp-1"),
        float.fromhex("0x1.3819bd7e1e04dp+0"),
    ], dtype=np.float64)
    expected = np.asarray([
        float.fromhex("-0x1.e0aaf9456437fp-1"),
        float.fromhex("-0x1.6c436eb8210dfp-1"),
        0.0,
        float.fromhex("0x1.6c436eb8210dfp-1"),
        float.fromhex("0x1.e0aaf9456437fp-1"),
    ], dtype=np.float64)
    actual = np.asarray(jax.jit(tke_mod._nemo_glibc234_vector_sin)(
        jnp.asarray(phase)))
    np.testing.assert_array_equal(actual.view(np.uint64),
                                  expected.view(np.uint64))

    planted = phase.copy()
    planted[0] = np.nextafter(planted[0], np.inf)
    planted_actual = np.asarray(tke_mod._nemo_glibc234_vector_sin(
        jnp.asarray(planted)))
    assert planted_actual[0].view(np.uint64) != expected[0].view(np.uint64)


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
def test_literal_htau_sin_jit_ad_and_range_guard():
    phase = jnp.asarray([-1.2, -0.3, 0.4, 1.2], dtype=jnp.float64)
    eager = tke_mod._nemo_glibc234_vector_sin(phase)
    compiled = jax.jit(tke_mod._nemo_glibc234_vector_sin)(phase)
    np.testing.assert_array_equal(np.asarray(eager).view(np.uint64),
                                  np.asarray(compiled).view(np.uint64))
    assert np.all(np.isfinite(np.asarray(
        jax.jacfwd(tke_mod._nemo_glibc234_vector_sin)(phase))))
    assert np.all(np.isfinite(np.asarray(
        jax.jacrev(tke_mod._nemo_glibc234_vector_sin)(phase))))

    outside = jnp.asarray([2.0, -2.0], dtype=jnp.float64)
    np.testing.assert_array_equal(
        np.asarray(tke_mod._nemo_glibc234_vector_sin(outside)).view(np.uint64),
        np.asarray(jnp.sin(outside)).view(np.uint64))
    fp32 = jnp.asarray([-0.3, 0.4], dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(tke_mod._nemo_glibc234_vector_sin(fp32)),
        np.asarray(jnp.sin(fp32)))


def test_etau_jax_expression_is_the_legacy_expression_byte_for_byte():
    e = jnp.asarray([[1.0e-6, 2.0e-6]])
    taum = jnp.asarray([0.08])
    depth = jnp.asarray([[4.0, 20.0]])
    cfg = TKEConfig(etau_mode="below_ml", etau_htau_mode="constant10m",
                    tke_etau_exponential_evaluation="jax_expression")
    actual = tke_mod.nemo_etau_injection(e, taum, depth, cfg)
    e_sfc = jnp.maximum(
        tke_mod._NEMO_TKE_EMIN0,
        tke_mod._NEMO_TKE_EBB / constants.rho_ocean * taum)
    expected = e + (cfg.etau_frac * e_sfc[..., None]
                    * jnp.exp(-depth / 10.0))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    with pytest.raises(ValueError, match="tke_etau_exponential_evaluation"):
        tke_mod.nemo_etau_injection(
            e, taum, depth,
            cfg._replace(tke_etau_exponential_evaluation="unknown"))
    with pytest.raises(ValueError, match="tke_etau_exponential_evaluation"):
        tke_mod.nemo_etau_injection(
            e, taum, depth,
            cfg._replace(etau_mode="none",
                         tke_etau_exponential_evaluation="unknown"))

    literal_cfg = cfg._replace(tke_etau_exponential_evaluation="nemo_literal")
    literal = tke_mod.nemo_etau_injection(e, taum, depth, literal_cfg)
    expected_literal = e + (
        literal_cfg.etau_frac * e_sfc[..., None]
        * tke_mod._nemo_glibc234_vector_exp(-depth / 10.0))
    np.testing.assert_array_equal(
        np.asarray(literal).view(np.uint64),
        np.asarray(expected_literal).view(np.uint64))


def test_htau_jax_expression_is_legacy_expression_byte_for_byte():
    lat_deg = jnp.asarray([[-69.151, 0.0, 43.7]])
    cfg = TKEConfig(tke_htau_evaluation="jax_expression")
    actual = tke_mod._nemo_etau_htau(lat_deg, cfg, jnp.float64)
    expected = jnp.maximum(
        tke_mod._NEMO_TKE_HTAU_MIN_M,
        jnp.minimum(
            tke_mod._NEMO_TKE_HTAU_MAX_M,
            tke_mod._NEMO_TKE_HTAU_SLOPE_M
            * jnp.abs(jnp.sin(jnp.deg2rad(lat_deg)))))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    with pytest.raises(ValueError, match="tke_htau_evaluation"):
        tke_mod._nemo_etau_htau(
            lat_deg, cfg._replace(tke_htau_evaluation="unknown"), jnp.float64)


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


def test_only_complete_dino_nemo_cards_freeze_step_entry_n2_bundle():
    faithful = {"nemo_dino_kamm", "nemo_dino_kamm_mlf"}
    for name, values in DINO_RECIPES.items():
        resolved = values.get(
            "tke_n2_evaluation_stage", "implicit_solve_state")
        assert resolved == (
            "step_entry" if name in faithful else "implicit_solve_state"), name
        if values.get("vmix_scheme") == "tke":
            built = dino_mod._dino_vertical_mixing_config(
                dino_config_for_recipe(name)).tke
            assert built.tke_n2_evaluation_stage == resolved, name

    unchanged = (TKEConfig(), _nemo_tke_config(), ACC_TKE_CONFIG,
                 ACC_BASIC_TKE_CONFIG)
    assert all(c.tke_n2_evaluation_stage == "implicit_solve_state"
               for c in unchanged)


def test_step_entry_n2_bundle_matches_live_geometry_construction():
    from legoesm.ocean.eos import (
        compute_buoyancy_frequency_nemo_bn2,
        nemo_bn2_depth_ladders,
        nemo_bn2_live_geometry,
        nemo_r3t_stretch,
    )

    tke_cfg = dino_mod._dino_vertical_mixing_config(
        dino_config_for_recipe("nemo_dino_kamm_mlf")).tke
    model, state = _step_entry_helper_fixture(tke_cfg)
    got = LatLonCGridOceanModel._tke_step_entry_n2_bundle(model, state)
    gdept, gdepw, e3w = nemo_bn2_live_geometry(
        model.z_coord, state.eta.data, state.H_bathy.data,
        r3t_evaluation="nemo_reciprocal")
    gdept0 = model.z_coord.nemo_gdept_0
    gdepw0 = model.z_coord.nemo_gdepw_0[..., 1:]
    stretch = nemo_r3t_stretch(
        model.z_coord, state.eta.data, state.H_bathy.data,
        evaluation="nemo_reciprocal")
    literal = dict(
        zrw_evaluation="nemo_literal", zrw_gdept_0=gdept0,
        zrw_gdepw_0=gdepw0, zrw_stretch=stretch)
    expected_now = compute_buoyancy_frequency_nemo_bn2(
        state.T.data, state.S.data, gdept, gdepw, g=constants.g,
        e3w_int=e3w, e3w_source="mesh_reference", **literal)
    expected_before = compute_buoyancy_frequency_nemo_bn2(
        state.T_before.data, state.S_before.data, gdept, gdepw,
        g=constants.g, e3w_int=e3w, e3w_source="mesh_reference", **literal)
    np.testing.assert_array_equal(got.rn2, expected_now)
    np.testing.assert_array_equal(got.rn2b, expected_before)
    raw_e3t = np.asarray(model.z_coord.nemo_e3t_0)
    wet_e3t = raw_e3t * np.asarray(stretch)[..., None]
    expected_e3t = np.where(
        np.asarray(model.z_coord.is_active, dtype=bool), wet_e3t, raw_e3t)
    np.testing.assert_array_equal(got.e3t_Kmm, expected_e3t)
    np.testing.assert_array_equal(got.gdepw_Kmm, gdepw)
    np.testing.assert_array_equal(got.e3w_Kmm, e3w)

    _gd_1d, gw_1d = nemo_bn2_depth_ladders(model.z_coord)
    planted_1d = compute_buoyancy_frequency_nemo_bn2(
        state.T.data, state.S.data, gdept, gdepw, g=constants.g,
        e3w_int=e3w, e3w_source="mesh_reference",
        zrw_evaluation="nemo_literal", zrw_gdept_0=gdept0,
        zrw_gdepw_0=gw_1d, zrw_stretch=stretch)
    assert not np.array_equal(np.asarray(got.rn2), np.asarray(planted_1d))

    legacy_model, legacy_state = _step_entry_helper_fixture(
        tke_cfg._replace(tke_n2_evaluation_stage="implicit_solve_state"))
    assert LatLonCGridOceanModel._tke_step_entry_n2_bundle(
        legacy_model, legacy_state) is None


def test_step_entry_n2_bundle_fails_closed_without_raw_w_mesh():
    tke_cfg = dino_mod._dino_vertical_mixing_config(
        dino_config_for_recipe("nemo_dino_kamm_mlf")).tke
    model, state = _step_entry_helper_fixture(tke_cfg)
    model.z_coord = model.z_coord._replace(nemo_gdepw_0=None)
    with pytest.raises(ValueError, match="requires raw NEMO"):
        LatLonCGridOceanModel._tke_step_entry_n2_bundle(model, state)


def test_base_dino_fe_card_constructs_its_step_entry_squared_shear():
    tke_cfg = dino_mod._dino_vertical_mixing_config(
        dino_config_for_recipe("nemo_dino_kamm")).tke
    assert tke_cfg.tke_shear_production == "squared_centered"
    assert tke_cfg.tke_shear_avm_weighting == "tpoint"
    model, state = _step_entry_helper_fixture(tke_cfg)
    got = LatLonCGridOceanModel._tke_step_entry_p_sh2(model, state)

    from legoesm.ocean.vertical import compute_ocean_jacobian
    J = compute_ocean_jacobian(
        state.eta.data, state.H_bathy.data, model.z_coord)
    dz = model.z_coord.dz_half_ref * J[..., None]
    u_cell = 0.5 * (state.u.data[:, :-1] + state.u.data[:, 1:])
    v_cell = 0.5 * (state.v.data[:-1] + state.v.data[1:])
    expected = state.tke_avm.data * shared_mod.vertical_shear_squared(
        u_cell, v_cell, dz)
    np.testing.assert_array_equal(got, expected)


def test_live_qco_step_entry_helper_is_jittable_and_differentiable():
    tke_cfg = dino_mod._dino_vertical_mixing_config(
        dino_config_for_recipe("nemo_dino_kamm_mlf")).tke
    model, template = _step_entry_helper_fixture(tke_cfg)

    def loss(u_now, v_now, u_before, v_before, eta_now, eta_before):
        state = SimpleNamespace(
            **{**template.__dict__,
               "u": template.u.replace(data=u_now),
               "v": template.v.replace(data=v_now),
               "u_before": template.u_before.replace(data=u_before),
               "v_before": template.v_before.replace(data=v_before),
               "eta": template.eta.replace(data=eta_now),
               "eta_before": template.eta_before.replace(data=eta_before)})
        return jnp.sum(LatLonCGridOceanModel._tke_step_entry_p_sh2(
            model, state))

    args = (template.u.data, template.v.data, template.u_before.data,
            template.v_before.data, template.eta.data,
            template.eta_before.data)
    value, grads = jax.jit(jax.value_and_grad(
        loss, argnums=(0, 1, 2, 3, 4, 5)))(*args)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
    assert all(float(jnp.max(jnp.abs(g))) > 0.0 for g in grads)

    legacy_metric = tke_cfg._replace(tke_shear_metric_source="tpoint_jacobian")
    legacy_model, legacy_state = _step_entry_helper_fixture(legacy_metric)
    legacy_value = LatLonCGridOceanModel._tke_step_entry_p_sh2(
        legacy_model, legacy_state)
    faithful_value = LatLonCGridOceanModel._tke_step_entry_p_sh2(
        model, template)
    assert not np.array_equal(np.asarray(faithful_value),
                              np.asarray(legacy_value))

    implicit = tke_cfg._replace(
        tke_shear_evaluation_stage="implicit_solve_state")
    implicit_model, implicit_state = _step_entry_helper_fixture(implicit)
    assert LatLonCGridOceanModel._tke_step_entry_p_sh2(
        implicit_model, implicit_state) is None


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


def test_step_entry_n2_bundle_feeds_every_registered_tke_consumer(monkeypatch):
    """Hand case: frozen rn2/rn2b/gdepw/e3w survive a poisoned solve state."""
    cfg = TKEConfig(
        prognostic=True, n2_mode="nemo_bn2",
        tke_n2_evaluation_stage="step_entry",
        lc=True, etau_mode="below_ml",
        kappaM_min=0.0, kappaH_min=0.0,
        enable_kappaH_profile=False,
    )
    bundle = tke_mod.TKEEntryN2Bundle(
        rn2=jnp.asarray([[2.0, 4.0]]),
        rn2b=jnp.asarray([[3.0, 5.0]]),
        gdepw_Kmm=jnp.asarray([[1.5, 4.0]]),
        e3w_Kmm=jnp.asarray([[1.5, 2.5]]),
        e3t_Kmm=jnp.asarray([[1.0, 2.0, 3.0]]),
    )
    seen = {"mxl": [], "closure": []}

    def fake_mxl(e, n2, *args, **kwargs):
        seen["mxl"].append(np.asarray(n2))
        return jnp.ones_like(e), jnp.ones_like(e)

    def fake_closure(*args, **kwargs):
        seen["closure"].append((
            np.asarray(kwargs["N2"]), np.asarray(kwargs["N2_prandtl"])))
        return jnp.ones_like(args[0]), jnp.ones_like(args[0])

    def fake_lc(taum, rn2b, depth, e3w, *args, **kwargs):
        seen["lc"] = tuple(np.asarray(x) for x in (rn2b, depth, e3w))
        return jnp.zeros_like(rn2b)

    def fake_solve(**kwargs):
        seen["solve"] = {
            "N2": np.asarray(kwargs["N2"]),
            "e3w": np.asarray(kwargs["dz_half"]),
        }
        return kwargs["e_old"]

    def fake_etau(e, taum, depth, *args, **kwargs):
        seen["etau_depth"] = np.asarray(depth)
        return e

    monkeypatch.setattr(tke_mod, "compute_mixing_lengths", fake_mxl)
    monkeypatch.setattr(tke_mod, "compute_K_from_tke", fake_closure)
    monkeypatch.setattr(tke_mod, "nemo_langmuir_tke_source", fake_lc)
    monkeypatch.setattr(tke_mod, "_solve_tke_backward_euler", fake_solve)
    monkeypatch.setattr(tke_mod, "nemo_etau_injection", fake_etau)

    kw = _column_kwargs(cfg)
    kw["T_cell"] = jnp.asarray([[99.0, -50.0, 7.0]])
    kw["S_cell"] = jnp.asarray([[-20.0, 88.0, 1.0]])
    tke_mod.tke_vertical_mixing(
        **kw, z_interface=jnp.asarray([-7.0, -9.0]),
        lat_deg=jnp.asarray([45.0]), precomputed_n2_bundle=bundle)

    for actual in seen["mxl"]:
        np.testing.assert_array_equal(actual, bundle.rn2)
    for actual, actual_before in seen["closure"]:
        np.testing.assert_array_equal(actual, bundle.rn2)
        np.testing.assert_array_equal(actual_before, bundle.rn2b)
    for actual, expected in zip(
            seen["lc"], (bundle.rn2b, bundle.gdepw_Kmm, bundle.e3w_Kmm)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(seen["solve"]["N2"], bundle.rn2)
    np.testing.assert_array_equal(seen["solve"]["e3w"], bundle.e3w_Kmm)
    np.testing.assert_array_equal(seen["etau_depth"], bundle.gdepw_Kmm)


def test_step_entry_selector_guards_and_legacy_stage_bit_identity():
    base = TKEConfig(prognostic=True)
    explicit = base._replace(
        tke_shear_evaluation_stage="implicit_solve_state")
    old = tke_mod.tke_vertical_mixing(**_column_kwargs(base))
    selected = tke_mod.tke_vertical_mixing(**_column_kwargs(explicit))
    for field in ("K_M", "K_H", "tke_new", "l_eps"):
        np.testing.assert_array_equal(getattr(selected, field), getattr(old, field))

    n2_explicit = base._replace(
        tke_n2_evaluation_stage="implicit_solve_state")
    n2_selected = tke_mod.tke_vertical_mixing(
        **_column_kwargs(n2_explicit))
    for field in ("K_M", "K_H", "tke_new", "l_eps"):
        np.testing.assert_array_equal(
            getattr(n2_selected, field), getattr(old, field))

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

    bundle = tke_mod.TKEEntryN2Bundle(
        *(jnp.ones((1, 2)) for _ in range(4)),
        e3t_Kmm=jnp.ones((1, 3)))
    with pytest.raises(ValueError, match="precomputed_n2_bundle was supplied"):
        tke_mod.tke_vertical_mixing(
            **_column_kwargs(base), precomputed_n2_bundle=bundle)
    bad_n2 = base._replace(tke_n2_evaluation_stage="not-a-stage")
    with pytest.raises(ValueError, match="Unknown TKEConfig.tke_n2"):
        tke_mod.tke_vertical_mixing(**_column_kwargs(bad_n2))
    missing_n2 = base._replace(tke_n2_evaluation_stage="step_entry")
    with pytest.raises(ValueError, match="requires precomputed_n2_bundle"):
        tke_mod.tke_vertical_mixing(**_column_kwargs(missing_n2))
    bad_shape = bundle._replace(rn2=jnp.ones((1, 1)))
    with pytest.raises(ValueError, match="bundle.rn2 must match"):
        tke_mod.tke_vertical_mixing(
            **_column_kwargs(missing_n2), precomputed_n2_bundle=bad_shape)


def test_explicit_legacy_selector_is_bit_identical_to_old_default():
    implicit_legacy = TKEConfig(
        prognostic=True, prandtl_mode="constant",
        kappa_convention="veros_sqrte", enable_kappaH_profile=False)
    explicit_legacy = implicit_legacy._replace(
        tke_preclosure_coeff_source="current_subiteration",
        tke_matrix_evaluation="factored",
        tke_solver_evaluation="shared_thomas")
    old = tke_mod.tke_vertical_mixing(**_column_kwargs(implicit_legacy))
    selected = tke_mod.tke_vertical_mixing(**_column_kwargs(explicit_legacy))
    for field in ("K_M", "K_H", "tke_new", "l_eps"):
        np.testing.assert_array_equal(getattr(selected, field), getattr(old, field))


def test_nemo_literal_solver_hand_case_red_controls_jit_and_grad():
    """Pin zdftke:547-565, including jpkm1 and post-solve mask order."""
    # Surface + two solved W rows + held jpk. The first column is exactly
    # hand-computable: diag=(1/2, 2, 23/8), work=(1, 3/2, 19/8), hence the
    # reverse solution is (22/23, 19/23); the held negative jpk RHS floors
    # first and is then zeroed by wmask.
    a = jnp.asarray([[0.0, -0.25, -0.5, 0.0],
                     [0.0, -0.25, 0.0, 0.0]], dtype=jnp.float64)
    b = jnp.asarray([[1.0, 2.0, 3.0, 1.0],
                     [1.0, 2.0, 1.0, 1.0]], dtype=jnp.float64)
    c = jnp.asarray([[0.0, -0.5, 0.0, 0.0],
                     [0.0, -0.5, 0.0, 0.0]], dtype=jnp.float64)
    rhs = jnp.asarray([[2.0, 1.0, 2.0, -1.0],
                       [4.0, 1.0, 0.4, 7.0]], dtype=jnp.float64)
    surface = jnp.asarray([2.0, 4.0], dtype=jnp.float64)
    wet = jnp.asarray([[1.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0]], dtype=jnp.float64)

    eager = tke_mod._nemo_literal_tke_solve(
        a, b, c, rhs, surface, wet, 0.1)
    np.testing.assert_allclose(
        np.asarray(eager[0]), [22.0 / 23.0, 19.0 / 23.0, 0.0],
        rtol=0.0, atol=2.0e-15)
    # Unequal-depth column: its identity bottom row is masked after the
    # source-ordered solve, while its one wet row remains independently live.
    np.testing.assert_allclose(
        np.asarray(eager[1]), [1.1, 0.0, 0.0], rtol=0.0, atol=2.0e-15)

    compiled = jax.jit(tke_mod._nemo_literal_tke_solve,
                       static_argnums=(6,))(a, b, c, rhs, surface, wet, 0.1)
    np.testing.assert_array_equal(compiled, eager)
    grad = jax.grad(lambda r: jnp.sum(tke_mod._nemo_literal_tke_solve(
        a, b, c, r, surface, wet, 0.1)))(rhs)
    assert np.all(np.isfinite(np.asarray(grad)))

    # Red controls: generic Thomas consumes the held jpk row and a recurrence
    # seeded at that row; it must not accidentally satisfy the literal case.
    shared = tke_mod._tridiag_thomas(a, b, c, rhs)[..., 1:]
    assert not np.allclose(np.asarray(shared), np.asarray(eager),
                           rtol=0.0, atol=1.0e-15)
    source = np.float64(-3.076523269561039e-169)
    lower = np.float64(-9.741726840261526e-158)
    previous_diagonal = np.float64(-3.745647236661062e150)
    previous_rhs = np.float64(-1.1401603475665343e167)
    divide_then_multiply = (
        source - lower / previous_diagonal * previous_rhs)
    multiply_then_divide = (
        source - (lower * previous_rhs) / previous_diagonal)
    assert divide_then_multiply != multiply_then_divide
    association_result = tke_mod._nemo_literal_tke_solve(
        jnp.asarray([[0.0, 0.0, lower, 0.0]], dtype=jnp.float64),
        jnp.asarray([[1.0, previous_diagonal, 1.0, 1.0]],
                    dtype=jnp.float64),
        jnp.zeros((1, 4), dtype=jnp.float64),
        jnp.asarray([[1.0, previous_rhs, source, 0.0]],
                    dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([[1.0, 1.0, 0.0]], dtype=jnp.float64),
        -jnp.inf,
    )
    assert np.asarray(association_result)[0, 1] == divide_then_multiply
    assert np.asarray(association_result)[0, 1] != multiply_then_divide
    planted = rhs.at[0, 2].set(rhs[0, 2] + 0.25)
    poisoned = tke_mod._nemo_literal_tke_solve(
        a, b, c, planted, surface, wet, 0.1)
    assert not np.array_equal(np.asarray(poisoned), np.asarray(eager))


def test_nemo_literal_solver_requires_literal_matrix():
    cfg = TKEConfig(tke_solver_evaluation="nemo_literal")
    with pytest.raises(ValueError, match="requires.*matrix"):
        tke_mod._solve_tke_backward_euler(
            e_old=jnp.ones((1, 2)), K_M_old=jnp.ones((1, 2)),
            K_H_old=jnp.ones((1, 2)), P_s=jnp.zeros((1, 2)),
            N2=jnp.zeros((1, 2)), l_eps=jnp.ones((1, 2)),
            dz_half=jnp.ones((1, 2)), surface_flux=jnp.zeros((1,)),
            dt=1.0, cfg=cfg)


def test_nemo_literal_solver_dispatches_through_production_solve(monkeypatch):
    cfg = TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        tke_solver_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0",
        tke_background=0.0, tke_surface_min=0.0,
    )
    real = tke_mod._nemo_literal_tke_solve
    calls = []

    def capture(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(tke_mod, "_nemo_literal_tke_solve", capture)
    result = tke_mod._solve_tke_backward_euler(
        e_old=jnp.asarray([[1.0, 0.7]]),
        K_M_old=jnp.asarray([[0.2, 0.3]]),
        K_H_old=jnp.asarray([[0.1, 0.15]]),
        P_s=jnp.asarray([[0.01, 0.02]]),
        N2=jnp.asarray([[1.0e-5, -1.0e-5]]),
        l_eps=jnp.asarray([[1.0, 1.2]]),
        dz_half=jnp.asarray([[2.0, 3.0]]),
        surface_flux=jnp.asarray([0.0]), dt=2.0, cfg=cfg,
        dz_surface=jnp.asarray([1.0]),
        surface_dirichlet=jnp.asarray([0.8]),
        surface_bc_level="nemo_z0",
        bottom_dirichlet=jnp.asarray([0.2]),
        K_M_surface=jnp.asarray([0.25]),
        w_active=jnp.asarray([[1.0, 0.0]]),
        nemo_e3t=jnp.asarray([[1.0, 2.0, 3.0]]),
        dissl_old=jnp.asarray([[0.1, 0.2]]),
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(result, calls[0])


def test_nemo_literal_matrix_matches_hand_computed_source_order(monkeypatch):
    """Nonuniform-e3t case pins every zdftke:499-510 operand and can go red."""
    cfg = TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        alpha_tke=1.0, c_eps=0.7,
        tke_background=0.0, tke_surface_min=0.0,
    )
    captured = []

    def capture(a, b, c, rhs):
        captured.append(tuple(np.asarray(x) for x in (a, b, c, rhs)))
        return rhs

    monkeypatch.setattr(tke_mod, "_tridiag_thomas", capture)
    base = dict(
        e_old=jnp.asarray([[1.0, 2.0, 3.0]]),
        K_M_old=jnp.asarray([[4.0, 6.0, 10.0]]),
        K_H_old=jnp.zeros((1, 3)), P_s=jnp.zeros((1, 3)),
        N2=jnp.zeros((1, 3)), l_eps=jnp.ones((1, 3)),
        dz_half=jnp.asarray([[11.0, 13.0, 17.0]]),
        surface_flux=jnp.zeros((1,)), dt=2.0, cfg=cfg,
        dz_surface=jnp.ones((1,)), surface_dirichlet=jnp.asarray([8.0]),
        surface_bc_level="nemo_z0", bottom_dirichlet=jnp.asarray([9.0]),
        K_M_surface=jnp.asarray([2.0]), w_active=jnp.ones((1, 3)),
        nemo_e3t=jnp.asarray([[2.0, 3.0, 5.0, 7.0]]),
        dissl_old=jnp.asarray([[0.1, 0.2, 0.3]]),
    )
    tke_mod._solve_tke_backward_euler(**base)
    a, b, c, rhs = captured[-1]
    lw0, lw1 = -3.0 / 11.0, -10.0 / 39.0
    up0, up1 = -10.0 / 33.0, -16.0 / 65.0
    diag0 = 1.0 - lw0 - up0 + 3.0 * 0.7 * 0.1
    diag1 = 1.0 - lw1 - up1 + 3.0 * 0.7 * 0.2
    np.testing.assert_allclose(a, [[0.0, lw0, lw1, 0.0]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(b, [[1.0, diag0, diag1, 1.0]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(c, [[0.0, up0, up1, 0.0]], rtol=0, atol=1e-15)
    np.testing.assert_allclose(rhs, [[8.0, 1.07, 2.28, 9.0]], rtol=0, atol=1e-15)

    # Planted wrong-slot control: replacing live e3w by e3t must trip the
    # captured coefficient comparison (the fields are deliberately unequal).
    captured.clear()
    tke_mod._solve_tke_backward_euler(
        **{**base, "dz_half": base["nemo_e3t"][..., :3]})
    _, wrong_b, wrong_c, _ = captured[-1]
    assert not np.array_equal(wrong_b, b)
    assert not np.array_equal(wrong_c, c)


def test_nemo_literal_rhs_applies_langmuir_before_budget(monkeypatch):
    """Matched-step arithmetic pins zdftke's two-statement RHS association."""
    cfg = TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        alpha_tke=1.0, c_eps=0.7,
        tke_background=0.0, tke_surface_min=0.0,
    )
    captured = []

    def capture(a, b, c, rhs):
        captured.append(np.asarray(rhs))
        return rhs

    monkeypatch.setattr(tke_mod, "_tridiag_thomas", capture)
    e_old = jnp.asarray([[1.0, 1.0e16, 3.0]], dtype=jnp.float64)
    p_s = jnp.asarray([[0.0, -1.5, 0.0]], dtype=jnp.float64)
    source = jnp.asarray([[0.0, 1.0, 0.0]], dtype=jnp.float64)
    tke_mod._solve_tke_backward_euler(
        e_old=e_old,
        K_M_old=jnp.zeros((1, 3), dtype=jnp.float64),
        K_H_old=jnp.zeros((1, 3), dtype=jnp.float64),
        P_s=p_s, N2=jnp.zeros((1, 3), dtype=jnp.float64),
        l_eps=jnp.ones((1, 3), dtype=jnp.float64),
        dz_half=jnp.ones((1, 3), dtype=jnp.float64),
        surface_flux=jnp.zeros((1,), dtype=jnp.float64),
        dt=2.0, cfg=cfg,
        dz_surface=jnp.ones((1,), dtype=jnp.float64),
        surface_dirichlet=jnp.asarray([8.0], dtype=jnp.float64),
        surface_bc_level="nemo_z0",
        bottom_dirichlet=jnp.asarray([9.0], dtype=jnp.float64),
        K_M_surface=jnp.zeros((1,), dtype=jnp.float64),
        w_active=jnp.ones((1, 3), dtype=bool),
        nemo_e3t=jnp.ones((1, 4), dtype=jnp.float64),
        dissl_old=jnp.zeros((1, 3), dtype=jnp.float64),
        external_source=source,
    )
    rhs = captured[-1]
    expected = ((np.float64(1.0e16) + np.float64(2.0))
                + np.float64(2.0) * np.float64(-1.5))
    legacy_wrong = ((np.float64(1.0e16)
                     + np.float64(2.0) * np.float64(-1.5))
                    + np.float64(2.0))
    assert rhs[0, 2] == expected
    assert rhs[0, 2] != legacy_wrong  # planted association control


def test_nemo_literal_dissipation_uses_post_langmuir_energy(monkeypatch):
    """zdftke:463 then :513-516 reuses the Langmuir-updated en operand."""
    cfg = TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        alpha_tke=1.0, c_eps=0.5,
        tke_background=0.0, tke_surface_min=0.0,
    )
    captured = []

    def capture(a, b, c, rhs):
        captured.append(np.asarray(rhs))
        return rhs

    monkeypatch.setattr(tke_mod, "_tridiag_thomas", capture)
    tke_mod._solve_tke_backward_euler(
        e_old=jnp.ones((1, 3), dtype=jnp.float64),
        K_M_old=jnp.zeros((1, 3), dtype=jnp.float64),
        K_H_old=jnp.zeros((1, 3), dtype=jnp.float64),
        P_s=jnp.zeros((1, 3), dtype=jnp.float64),
        N2=jnp.zeros((1, 3), dtype=jnp.float64),
        l_eps=jnp.ones((1, 3), dtype=jnp.float64),
        dz_half=jnp.ones((1, 3), dtype=jnp.float64),
        surface_flux=jnp.zeros((1,), dtype=jnp.float64),
        dt=3.0, cfg=cfg,
        dz_surface=jnp.ones((1,), dtype=jnp.float64),
        surface_dirichlet=jnp.asarray([8.0], dtype=jnp.float64),
        surface_bc_level="nemo_z0",
        bottom_dirichlet=jnp.asarray([9.0], dtype=jnp.float64),
        K_M_surface=jnp.zeros((1,), dtype=jnp.float64),
        w_active=jnp.ones((1, 3), dtype=bool),
        nemo_e3t=jnp.ones((1, 4), dtype=jnp.float64),
        dissl_old=jnp.full((1, 3), 0.4, dtype=jnp.float64),
        external_source=jnp.asarray([[0.0, 2.0, 0.0]], dtype=jnp.float64),
    )
    # post-LC en = 1 + 3*2 = 7; diss add-back = 3*(.5*.5*.4*7)=2.1.
    assert captured[-1][0, 2] == np.float64(9.1)
    assert captured[-1][0, 2] != np.float64(7.3)  # old e_old operand


def test_nemo_literal_matrix_is_jittable_and_differentiable():
    """Exercise the faithful matrix branch itself under JIT and reverse AD."""
    cfg = TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        alpha_tke=1.0, c_eps=0.7,
        tke_background=1.0e-12, tke_surface_min=0.0,
    )

    def loss(e_old, avm_old, dissl_old):
        out = tke_mod._solve_tke_backward_euler(
            e_old=e_old,
            K_M_old=avm_old,
            K_H_old=jnp.zeros_like(e_old),
            P_s=jnp.asarray([[0.2, 0.1, 0.0]], dtype=jnp.float64),
            N2=jnp.zeros_like(e_old),
            l_eps=jnp.ones_like(e_old),
            dz_half=jnp.asarray([[2.0, 3.0, 5.0]], dtype=jnp.float64),
            surface_flux=jnp.zeros((1,), dtype=jnp.float64),
            dt=2.0, cfg=cfg,
            dz_surface=jnp.ones((1,), dtype=jnp.float64),
            surface_dirichlet=jnp.asarray([0.8], dtype=jnp.float64),
            surface_bc_level="nemo_z0",
            bottom_dirichlet=jnp.asarray([0.4], dtype=jnp.float64),
            K_M_surface=jnp.asarray([0.03], dtype=jnp.float64),
            bottom_level=jnp.asarray([2]),
            w_active=jnp.ones((1, 3), dtype=bool),
            nemo_e3t=jnp.asarray([[1.0, 2.0, 4.0, 7.0]], dtype=jnp.float64),
            dissl_old=dissl_old,
            external_source=jnp.asarray([[0.0, 0.01, 0.0]], dtype=jnp.float64),
        )
        return jnp.sum(out)

    operands = (
        jnp.asarray([[0.5, 0.4, 0.3]], dtype=jnp.float64),
        jnp.asarray([[0.02, 0.03, 0.04]], dtype=jnp.float64),
        jnp.asarray([[0.1, 0.2, 0.3]], dtype=jnp.float64),
    )
    value, grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(
        *operands)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in grads)


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


def _carry_seed_fixture(surface_bc_level="interior_pinned", mxl_choice=3):
    cfg = TKEConfig(
        prognostic=True,
        tke_preclosure_coeff_source="carried_previous_step",
        tke_surface_bc_level=surface_bc_level,
        # nemo_z0 has no meaning without a held surface value, so the fixture
        # describes a configuration that could actually run (codex [MEDIUM]).
        surface_bc=("nemo_dirichlet" if surface_bc_level == "nemo_z0"
                    else "veros_flux"),
        tke_mxl_choice=mxl_choice,
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
    return dummy, state


def test_cold_start_carry_uses_nemo_wmask_on_partial_depth_columns():
    dummy, state = _carry_seed_fixture("nemo_z0")
    out = LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, state)
    np.testing.assert_array_equal(out.tke_avm.data, [[[3.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]]])
    np.testing.assert_array_equal(out.tke_avt.data, [[[5.0, 0.0, 0.0],
                                                       [0.0, 0.0, 0.0]]])
    np.testing.assert_array_equal(out.tke_avm_surface.data, [[3.0, 0.0]])

    partial = state._replace(tke_avm=out.tke_avm)
    with pytest.raises(ValueError, match="partially populated"):
        LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, partial)


def test_nemo_z0_without_the_mxl0_anchor_also_leaves_the_surface_slot_alone():
    """codex 9693003 [HIGH]: the closure builds _K_M_surface only when the
    ln_mxl0 anchor exists, and the anchor needs tke_mxl_choice 3 or 4. Keying
    the guard on the boundary alone left nemo_z0 + choice 2 crashing on its
    second step exactly as before."""
    dummy, state = _carry_seed_fixture("nemo_z0", mxl_choice=2)
    out = LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, state)
    assert out.tke_avm is not None and out.tke_avt is not None
    assert out.tke_avm_surface is None
    # and the state the writeback produces must be accepted on the next step
    again = LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, out)
    assert again.tke_avm is out.tke_avm


def test_interior_pinned_does_not_seed_a_surface_avm_it_never_consumes():
    """The closure builds _K_M_surface only under the nemo_z0 face assembly,
    and the post-solve writeback stores None for it otherwise. Seeding it here
    therefore produced a state the model could never reproduce."""
    dummy, state = _carry_seed_fixture("interior_pinned")
    out = LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, state)
    assert out.tke_avm is not None and out.tke_avt is not None
    assert out.tke_avm_surface is None


def test_a_second_step_accepts_the_state_the_first_step_produced():
    """REGRESSION, job 9692852: the arm died on step 2 with "partially
    populated". Step 1 seeded all three fields, the writeback stored None for
    the surface one (correctly -- interior_pinned produces none), and the
    guard then rejected the model's own output. carried_previous_step was
    therefore unusable on every card except the nemo_z0 ones.

    Reverting either half of the fix makes this raise."""
    dummy, state = _carry_seed_fixture("interior_pinned")
    after_step1 = LatLonCGridOceanModel._seed_tke_preclosure_carry(dummy, state)
    # what step 1's post-solve writeback stores: the two coefficients, and
    # None for the surface value the closure did not produce.
    written_back = after_step1._replace(tke_avm_surface=None)
    again = LatLonCGridOceanModel._seed_tke_preclosure_carry(
        dummy, written_back)
    assert again.tke_avm is written_back.tke_avm
    assert again.tke_avt is written_back.tke_avt

    # NON-VACUITY: a genuinely partial state must still raise under this same
    # boundary setting, or the test above would pass on a guard that never fires.
    with pytest.raises(ValueError, match="partially populated"):
        LatLonCGridOceanModel._seed_tke_preclosure_carry(
            dummy, state._replace(tke_avm=after_step1.tke_avm))


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
