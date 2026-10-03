import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round128_een_u_fold_walk as gate,
)


def test_round128_northern_u_row_uses_registered_source_and_permutation():
    field = np.arange(4 * 5 * 2).reshape(4, 5, 2)
    perm = np.array([4, 3, 2, 1, 0])
    result = gate.northern_u_row(field, perm)
    np.testing.assert_array_equal(result, field[-2, perm])
    np.testing.assert_raises(
        gate.GateError, gate.northern_u_row, field, np.arange(4))
    wrong_perm = np.array([0, 4, 3, 2, 1])
    wrong = gate.northern_u_row(field, perm, permutation=wrong_perm)
    assert not np.array_equal(wrong, result)


def test_round128_replaces_only_the_named_northern_neighbor():
    field = np.zeros((3, 5, 2))
    row = np.arange(10).reshape(5, 2)
    ne = gate.replace_northern_neighbor(field, row, "ne")
    nw = gate.replace_northern_neighbor(field, row, "nw")
    np.testing.assert_array_equal(ne[-1], row)
    np.testing.assert_array_equal(nw[-1], np.roll(row, 1, axis=0))
    np.testing.assert_array_equal(ne[:-1], field[:-1])
    np.testing.assert_array_equal(nw[:-1], field[:-1])


def test_round128_registry_locks_baseline_and_controls():
    assert gate.EXPECTED_BASELINE["nw"]["e3u"] == (95, 95)
    assert gate.EXPECTED_BASELINE["ne"]["mask"] == (1283, 1283)
    assert gate.ARMS == ("baseline", "thickness-only", "mask-only", "combined")
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "wrong-row",
        "wrong-permutation", "scope-route",
    )
