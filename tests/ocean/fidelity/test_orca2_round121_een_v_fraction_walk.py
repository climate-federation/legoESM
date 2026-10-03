import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round121_een_v_fraction_walk as gate,
)


def test_round121_literal_fraction_registry_is_complete():
    assert gate.PATHS == ("ne", "nw")
    assert gate.COMPONENTS == ("1", "2", "3")
    assert gate.OPERANDS == ("ff", "e3f0", "r3f", "mask", "denom", "frac")
    assert gate.SHIFTS == {
        "ne": ((0, -1), (0, 0), (1, 0)),
        "nw": ((0, 0), (1, 0), (1, -1)),
    }
    assert gate.EXPECTED_FIRST_COMPONENT == {"ne": "1", "nw": "3"}
    assert gate.EXPECTED_FIRST == {"ne": "1_ff", "nw": "3_ff"}
    assert gate.EXPECTED_SCORES["ne"]["1_ff"] == (1431, 1431)
    assert gate.EXPECTED_SCORES["ne"]["1_e3f0"] == (514, 514)
    assert gate.EXPECTED_SCORES["nw"]["3_ff"] == (1431, 1431)
    assert gate.EXPECTED_SCORES["nw"]["3_mask"] == (1154, 1154)
    assert all(
        gate.EXPECTED_SCORES[path][name] == (0, 0)
        for path, names in {"ne": ("2_ff", "3_frac"), "nw": ("1_ff", "partial")}.items()
        for name in names
    )
    assert gate.SOURCE_ORDER.index("partial") > gate.SOURCE_ORDER.index("2_frac")
    assert gate.SOURCE_ORDER[-1] == "sum"


def test_round121_shift_matches_fortran_neighbor_association():
    values = np.arange(4 * 5, dtype=np.float64).reshape(4, 5)
    np.testing.assert_array_equal(gate._shift(values, 0, -1)[0], values[1])
    np.testing.assert_array_equal(gate._shift(values, 1, 0)[:, 0], values[:, -1])
    assert gate._shift(values, 1, -1)[0, 0] == values[1, -1]


def test_round121_plants_cover_oracle_candidate_and_scope():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "rank-seam", "scope-route")
