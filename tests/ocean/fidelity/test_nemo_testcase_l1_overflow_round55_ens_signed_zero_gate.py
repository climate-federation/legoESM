"""Controls for the round-55 OVERFLOW ENS signed-zero walk."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round55_ens_signed_zero_gate as gate  # noqa: E402


def _case(after_bits: int = 0) -> tuple[dict, dict, np.ndarray]:
    shape = (2, 2, 1)
    active = np.ones(shape, dtype=bool)
    positive = np.zeros(shape, dtype=np.float64)
    before = positive.copy()
    before[0, 0, 0] = np.copysign(0.0, -1.0)
    product = positive.copy()
    after = np.add(before, product)
    after.view(np.uint64)[0, 0, 0] = after_bits
    operands = {
        "zwz_prediv": positive.copy(),
        "zwz_postdiv": positive.copy(),
        "zuav": positive.copy(),
        "zwz_pair_u": positive.copy(),
        "product_u": product,
        "rhs_before_u": before,
        "rhs_after_u": after,
    }
    parent = {"after_hpg_u": before.copy(), "after_vor_u": after.copy()}
    return operands, parent, active


def test_source_order_chain_reaches_the_bit_bar():
    operands, parent, active = _case()
    report = gate._analyze(
        operands,
        parent,
        active,
        expected_first=(0, 0, 0),
        expected_changed=1,
        plant=None,
    )
    assert report["status"] == "AT_BAR"
    assert report["negative_zero_plus_positive_zero_to_positive_zero"] == 1
    assert report["addition_unequal"] == 0


def test_both_plants_refuse_the_recorded_statement():
    for plant in ("product_sign", "endpoint"):
        operands, parent, active = _case()
        report = gate._analyze(
            operands,
            parent,
            active,
            expected_first=(0, 0, 0),
            expected_changed=1,
            plant=plant,
        )
        assert report["status"] == "PLANTED_REFUSAL"
        assert report["addition_unequal"] == 1


def test_wrong_recorded_endpoint_is_debt():
    operands, parent, active = _case(after_bits=0x0000000000000001)
    report = gate._analyze(
        operands,
        parent,
        active,
        expected_first=(0, 0, 0),
        expected_changed=1,
        plant=None,
    )
    assert report["status"] == "DEBT"
    assert report["R55-P2"] == "REFUTED"
