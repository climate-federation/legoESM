"""Binding tests for the ORCA2 round-32 read-out gate."""

from __future__ import annotations

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round32_readout_gate as gate,
)


def test_score_is_bit_exact_and_signed_zero_sensitive():
    same = gate.score(np.array([1.0]), np.array([1.0]))
    signed_zero = gate.score(np.array([-0.0]), np.array([0.0]))
    assert same["bit_identical"] and same["unequal"] == 0
    assert not signed_zero["bit_identical"] and signed_zero["unequal"] == 1


def test_registered_tenth_step_values_are_the_measured_arm():
    assert gate.EXPECTED_KT10 == {
        "u": 0.4230544199344075,
        "v": 0.6838675521949865,
    }


def test_hf0_builder_changes_only_when_the_carried_operand_is_selected():
    def original(*args, **kwargs):
        return "parent"

    assert gate._hf0_builder(original, use_carried_hf0=False)(
        None, None, None) == "parent"
