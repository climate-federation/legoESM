import copy

import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round241_compact_mask_atomic_gate as gate,
)


def _report():
    row = {
        "support_slots": 1319,
        "changed_slots": 668,
        "off_support_changed": 0,
        "baseline_unequal": 668,
        "baseline_max_abs": 2.4e-6,
        "arm_unequal": 0,
        "arm_max_abs": 0.0,
        "arm_passivity": True,
        "arm_changes_bits": True,
        "dtype": "float64",
    }
    return {"rows": {
        "independent": copy.deepcopy(row),
        "given_nemo_entry": copy.deepcopy(row),
    }}


def _atomic_report():
    row = {
        "last_completed_kt": 10,
        "kt1_stage1": {"S_max_abs": 0.001, "T_max_abs": 0.001},
    }
    return {"ladders": {
        "independent": copy.deepcopy(row),
        "given_nemo_entry": copy.deepcopy(row),
    }}


def test_private_arm_defaults_off():
    assert _NEMOWSRK3TestHooks().preserve_literal_hpg_vmask is False


def test_classifier_requires_exact_rhs_closure():
    assert gate.classify_rhs(_report())["status"] == (
        "PASS_R241_LITERAL_HPG_RHS_CLOSED")


def test_atomic_classifier_requires_kt8_and_preregistered_limits():
    assert gate.classify_atomic(_atomic_report())["status"] == (
        "PASS_R241_ATOMIC_UNIT_THROUGH_KT8")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify_rhs(_report(), plant=plant)


@pytest.mark.parametrize("plant", gate.ATOMIC_PLANTS[1:])
def test_every_atomic_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify_atomic(_atomic_report(), plant=plant)
