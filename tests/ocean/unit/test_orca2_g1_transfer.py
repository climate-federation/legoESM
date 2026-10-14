"""Round-269 controls for the G1 statements transferred to ORCA2."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


class _Fold:
    is_active = True
    vector_sign_v = -1.0

    def __init__(self, size):
        self.perm_v = (size - np.arange(size)) % size


class _Grid:
    def __init__(self, fold):
        self.fold = fold


def test_een_fold_image_row_and_disabled_plant():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _een_fold_image_v_row,
    )

    size = 6
    values = jnp.arange(30.0).reshape(5, size, 1)
    thickness = values + 100.0
    mask = jnp.ones_like(values).at[-1].set(0.0).at[-2, 2].set(0.0)
    sentinel = object()
    off = _een_fold_image_v_row(
        False, _Grid(_Fold(size)), thickness, values,
        sentinel, sentinel, sentinel, mask)
    assert off[0] is thickness and off[1] is values and off[5] is mask

    moved = _een_fold_image_v_row(
        True, _Grid(_Fold(size)), thickness, values,
        sentinel, sentinel, sentinel, mask)
    perm = (size - np.arange(size)) % size
    wet = np.asarray(mask)[-2][perm]
    assert np.array_equal(np.asarray(moved[5])[-1], wet)
    assert np.array_equal(
        np.asarray(moved[1])[-1], -np.asarray(values)[-2][perm] * wet)
    assert np.array_equal(
        np.asarray(moved[0])[-1], np.asarray(thickness)[-2][perm])
    with pytest.raises(ValueError, match="active north fold"):
        _een_fold_image_v_row(
            True, _Grid(None), thickness, values,
            sentinel, sentinel, sentinel, mask)


def test_fold_top_v_operands_and_disabled_plant():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_fold_top_v_operands,
    )

    mask = jnp.arange(12.0).reshape(4, 3)
    same, flag = _nemo_fold_top_v_operands(mask, False, None)
    assert same is mask and flag is False
    moved, flag = _nemo_fold_top_v_operands(
        mask, False, jnp.asarray([1.0, 0.0, 1.0]))
    assert flag is True
    assert np.array_equal(np.asarray(moved)[-1], [1.0, 0.0, 1.0])
    assert np.array_equal(np.asarray(moved)[:-1], np.asarray(mask)[:-1])


def test_literal_flux_update_matches_written_order_and_legacy_plant_moves():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_flux_form_external_velocity_update,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig

    base = LatLonCGridOceanConfig()
    literal = base._replace(barotropic=base.barotropic._replace(
        barotropic_flux_update_nemo_literal=True,
        barotropic_entry_depth_nemo_literal=True,
    ))
    args = tuple(jnp.asarray(value, dtype=jnp.float64) for value in (
        0.125, 7.0, 7.25, 7.5, 8.0, 8.5, 0.03125, -0.0625, 0.015625,
        830.7692307692307, 1.0, 1.0e-6,
    ))
    actual = _nemo_flux_form_external_velocity_update(
        *args, config=literal, substep_index=0)
    legacy = _nemo_flux_form_external_velocity_update(*args)
    r = np.float64
    forced = r(r(r(args[2] * args[6]) + r(args[3] * args[7]))
               + r(args[4] * args[8]))
    expected = r(r(r(args[4] * args[0]) + r(args[9] * forced))
                 * r(args[10] / r(r(args[5] + 1.0) - args[10])))
    assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert not np.array_equal(np.asarray(actual), np.asarray(legacy))


def test_orca2_and_omt_cards_select_executed_transferred_statements():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round209_omt1_ladder_gate as omt1,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round274_g1_full_import_gate as round274,
    )

    assert LatLonCGridOceanConfig._field_defaults["een_fold_image_v_row"] is False
    assert BarotropicConfig._field_defaults[
        "barotropic_fold_top_v_face_ssh"] is False
    assert LatLonCGridOceanConfig._field_defaults[
        "rk3_stage1_ssh_nemo_literal"] is False
    for name in (
        "rk3_stage1_rhs_nemo_literal",
        "rk3_stage_velocity_written_order",
        "een_vorticity_written_order",
        "zdf_barotropic_removal_nemo_literal",
        "zdf_momentum_nemo_face_thickness",
        "rk3_fold_exchange_top_v_row",
        "rk3_stage_r3f_nemo_literal",
        "rk3_stage_r3t_nemo_literal",
        "nemo_eos_gdept_rounded_product",
        "rk3_stage_r3uv_nemo_literal",
    ):
        assert LatLonCGridOceanConfig._field_defaults[name] is False
    for name in (
        "barotropic_flux_update_nemo_literal",
        "barotropic_substep_dt_nemo_literal",
        "barotropic_entry_depth_nemo_literal",
        "barotropic_een_ratio_fold_exchange",
        "barotropic_een_fe3mask_pivot_row",
        "barotropic_een_ratio_written_order",
    ):
        assert BarotropicConfig._field_defaults[name] is False

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round222/acquisition/orca2_omt4_frames_10step_a_np2")
    if not (deck / "namelist_cfg").is_file():
        pytest.skip("admitted OMT-4 deck is not present")
    shipped = recipe.build_nemo_testcase_card("ORCA2-zps", deck_root=deck)
    omt0_card = omt0.build_omt0_card(deck)
    omt1_card = omt1.build_omt1_card(deck)
    omt4_card = omt4.build_omt4_card(deck)
    for card in (shipped, omt0_card, omt1_card, omt4_card):
        config = card.recipe.model_config
        assert config.een_fold_image_v_row is True
        assert config.barotropic.barotropic_fold_top_v_face_ssh is True
        assert config.rk3_stage1_ssh_nemo_literal is True
        # Decision 122 (round 278): ladder-certified statements land on the
        # ORCA2 identity and every card derived from it.
        assert config.barotropic.barotropic_substep_dt_nemo_literal is True
        assert card.recipe.z_coord.nemo_r3t_stored_product is True
        for name in (
            "rk3_stage1_rhs_nemo_literal",
            "rk3_stage_velocity_written_order",
            "zdf_barotropic_removal_nemo_literal",
            "zdf_momentum_nemo_face_thickness",
            "zdf_momentum_stage_r3u_nemo_literal",
            "zdf_tracer_nemo_rounded_thickness",
            "barotropic_rhs_ascending_level_sum",
            "rk3_stage_r3f_nemo_literal",
            "rk3_stage_r3t_nemo_literal",
            "nemo_eos_gdept_rounded_product",
            "rk3_stage_r3uv_nemo_literal",
        ):
            assert getattr(config, name) is True, name
        for name in (
            "barotropic_een_ratio_fold_exchange",
            "barotropic_een_fe3mask_pivot_row",
            "barotropic_een_ratio_written_order",
        ):
            assert getattr(config.barotropic, name) is True, name
        # np_COR-certified (ORCA2 runs np_CRV) and the held top-V exchange.
        assert config.een_vorticity_written_order is False
        assert config.rk3_fold_exchange_top_v_row is False
    assert omt0_card.recipe.model_config.barotropic.barotropic_flux_update_nemo_literal
    assert omt0_card.recipe.model_config.barotropic.barotropic_entry_depth_nemo_literal
    for card in (shipped, omt1_card, omt4_card):
        assert not card.recipe.model_config.barotropic.barotropic_flux_update_nemo_literal
        assert not card.recipe.model_config.barotropic.barotropic_entry_depth_nemo_literal
    selected = round274.selected_card(omt4_card)
    assert selected.recipe.z_coord.nemo_r3t_stored_product
    assert all(getattr(selected.recipe.model_config, name)
               for name in round274.MODEL_SELECTORS)
    assert all(getattr(selected.recipe.model_config.barotropic, name)
               for name in round274.BAROTROPIC_SELECTORS)
    control = round274.legacy_card(selected)
    assert not control.recipe.z_coord.nemo_r3t_stored_product
    assert not any(getattr(control.recipe.model_config, name)
                   for name in round274.MODEL_SELECTORS)
    assert not any(getattr(control.recipe.model_config.barotropic, name)
                   for name in round274.BAROTROPIC_SELECTORS)
    with pytest.raises(round274.GateError, match="retained selectors"):
        round274.legacy_card(selected, plant="retained-selector")


def test_round275_private_arms_select_only_the_registered_fields():
    from pathlib import Path

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round269_g1_decision96_gate as decision96,
        nemo_testcase_l4_orca2_round275_g1_source_bisect_gate as round275,
    )

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round83/acquisition/orca2_rung0_restart_list_10step_a_np2")
    if not (deck / "namelist_cfg").is_file():
        pytest.skip("admitted rung-0 deck is not present")
    card = decision96._selected_card("rung0", deck)
    universe = round275.MODEL | round275.BAROTROPIC | round275.COORDINATE
    for arm, expected in round275.ARMS.items():
        selected = round275.arm_card(card, arm)
        cfg = selected.recipe.model_config
        actual = {name for name in round275.MODEL if getattr(cfg, name)}
        actual |= {name for name in round275.BAROTROPIC
                   if getattr(cfg.barotropic, name)}
        if selected.recipe.z_coord.nemo_r3t_stored_product:
            actual.add("nemo_r3t_stored_product")
        assert actual == set(expected)
        assert actual <= universe
    with pytest.raises(round275.GateError, match="selected fields differ"):
        round275.arm_card(card, "vertical_residual", plant="extra-selector")
