"""Round-268 controls for the two G1 fold-image-row statements."""

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


def test_orca2_and_omt_cards_select_both_transferred_statements():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    )

    assert LatLonCGridOceanConfig._field_defaults["een_fold_image_v_row"] is False
    assert BarotropicConfig._field_defaults[
        "barotropic_fold_top_v_face_ssh"] is False
    assert LatLonCGridOceanConfig._fields[-1] == "een_fold_image_v_row"
    assert BarotropicConfig._fields[-1] == "barotropic_fold_top_v_face_ssh"

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round222/acquisition/orca2_omt4_frames_10step_a_np2")
    if not (deck / "namelist_cfg").is_file():
        pytest.skip("admitted OMT-4 deck is not present")
    shipped = recipe.build_nemo_testcase_card("ORCA2-zps", deck_root=deck)
    omt = omt4.build_omt4_card(deck)
    for card in (shipped, omt):
        config = card.recipe.model_config
        assert config.een_fold_image_v_row is True
        assert config.barotropic.barotropic_fold_top_v_face_ssh is True
