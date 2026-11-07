"""Binding tests for the ORCA2 round-34 F-column depth gates."""

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round34_hf0_operand_gate as operand,
)


def test_registered_replay_values_are_the_preregistered_secondary_owner():
    assert operand.EXPECTED_REPLAY_MAX == {
        "u_momentum": 3.181628207426175e-09,
        "v_momentum": 2.9702048395431957e-09,
    }
