from __future__ import annotations

import copy
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _nemo_associate_stage_v_ratio,
)
from legoesm.ocean.vertical import nemo_qco_live_face_geometry_from_operands
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round262_v_ratio_fold_gate as gate,
)


def _row(unequal=0):
    return {"unequal": unequal, "max_abs": 0.0, "argmax": None}


def _report(label):
    return {
        "status": "PASS_R261_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": {name: True for name in ("T", "S", "u", "v", "eta")},
        "records": {"record.bin": "abc"},
        "operand_rows": {name: _row() for name in ("e1v", "e3v", "vv", "zvb", "vmask")},
        "correction_input_rows": {name: _row() for name in ("vn_adv", "r1_hv", "vv_b")},
        "zFv": _row(181),
        "replay": {"candidate": _row(), "oracle": _row()},
        "one_bit_control": _row(1),
    }


def _boundary(stage, value=0.0):
    fields = {}
    for name in ("T", "S", "u", "v", "ssh"):
        row = {"nonfinite": 0, "max_abs": value if name == "S" else 0.0}
        fields[name] = {"fold_band": row, "interior": row, "complete": row}
    return {"kt": 1, "stage": stage, "fields": fields}


def _prefix_report(label):
    return {
        "status": "PASS_R249_STAGE_TABLE",
        "label": label,
        "unit": True,
        "operand_route": "resolved-candidate",
        "passivity": [{"kt": 1, "fields": {name: True for name in ("T", "S", "u", "v", "eta")}}],
        "boundaries": [_boundary(0), _boundary(1), _boundary(2, 1.0), _boundary(3, 2.0)],
    }


def test_v_ratio_fold_copies_the_mirrored_southern_row():
    perm = jnp.asarray([2, 1, 0])
    fold = SimpleNamespace(
        is_active=True, fold_j=2, perm_v=perm, vector_sign_v=-1.0)
    grid = SimpleNamespace(fold=fold)
    ratio = jnp.asarray([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])
    actual = np.asarray(_nemo_associate_stage_v_ratio(ratio, grid))
    np.testing.assert_array_equal(actual[:-1], np.asarray(ratio[:-1]))
    np.testing.assert_array_equal(actual[-1], np.asarray([6.0, 5.0, 4.0]))


def test_v_ratio_fold_moves_thickness_and_reciprocal_together():
    perm = jnp.asarray([2, 1, 0])
    fold = SimpleNamespace(
        is_active=True, fold_j=2, perm_v=perm, vector_sign_v=-1.0)
    grid = SimpleNamespace(fold=fold)
    eta = jnp.asarray([[0.0, 0.0, 0.0],
                       [1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
    shape3 = eta.shape + (1,)
    geometry = nemo_qco_live_face_geometry_from_operands(
        eta,
        jnp.ones(shape3), jnp.ones(shape3),
        jnp.ones(shape3), jnp.ones(shape3),
        jnp.ones(eta.shape), jnp.ones(eta.shape),
        jnp.ones(eta.shape), jnp.ones(eta.shape), jnp.ones(eta.shape),
        associate_v_ratio=(
            lambda ratio: _nemo_associate_stage_v_ratio(ratio, grid)),
    )
    expected = np.asarray(geometry.r3v[-2])[::-1]
    np.testing.assert_array_equal(np.asarray(geometry.r3v[-1]), expected)
    np.testing.assert_array_equal(
        np.asarray(geometry.e3v[-1, :, 0]), 1.0 + expected)
    np.testing.assert_array_equal(
        np.asarray(geometry.r1_hv[-1]), 1.0 / (1.0 + expected))


def test_round262_classifier_accepts_exact_source_replay():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["status"] == "PASS_R262_V_RATIO_FOLD_D96_RATCHET"


def test_round262_prefix_classifier_names_first_valued_boundary():
    result = gate.classify_prefix([_prefix_report(label) for label in gate.LABELS])
    assert result["first_valued_boundary"] == (1, 2)


@pytest.mark.parametrize("plant", gate.PREFIX_PLANTS[1:])
def test_round262_prefix_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify_prefix(
            [_prefix_report(label) for label in gate.LABELS], plant=plant)


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round262_plants_fire(plant):
    reports = [_report(label) for label in gate.LABELS]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
