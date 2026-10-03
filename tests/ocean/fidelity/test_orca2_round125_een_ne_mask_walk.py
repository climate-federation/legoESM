import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round125_een_ne_mask_walk as gate,
)


def test_round125_frozen_registry_is_complete():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "wrong-row",
        "northwest-isolation", "scope-route",
    )
    assert gate.EXPECTED_BEFORE["1_mask"] == (1160, 1160)
    assert gate.EXPECTED_AFTER["1_mask"] == (0, 0)


def test_round125_arm_replaces_only_ne_first_mask_descendants():
    shape = (3, 4, 2)
    fields = {}
    for component in gate.r121.COMPONENTS:
        fields[f"{component}_ff"] = np.full(shape, 8.0 + int(component))
        fields[f"{component}_e3f0"] = np.ones(shape)
        fields[f"{component}_r3f"] = np.zeros(shape)
        fields[f"{component}_mask"] = np.ones(shape)
        fields[f"{component}_denom"] = np.ones(shape)
        fields[f"{component}_frac"] = fields[f"{component}_ff"].copy()
    fields["partial"] = fields["1_frac"] + fields["2_frac"]
    fields["sum"] = fields["partial"] + fields["3_frac"]
    north = np.zeros((4, 2), dtype=np.float64)
    result = gate._replace_ne_north_mask(
        fields, north, np.ones(shape, dtype=bool)
    )
    np.testing.assert_array_equal(result["1_mask"][-1], north)
    np.testing.assert_array_equal(result["1_mask"][:-1], fields["1_mask"][:-1])
    np.testing.assert_array_equal(result["2_mask"], fields["2_mask"])
    np.testing.assert_array_equal(result["3_mask"], fields["3_mask"])
    np.testing.assert_array_equal(result["1_frac"], fields["1_frac"])
    np.testing.assert_array_equal(result["sum"], fields["sum"])


def test_round125_wrong_mask_changes_descendants_when_r3f_is_nonzero():
    shape = (2, 3, 1)
    fields = {}
    for component in gate.r121.COMPONENTS:
        fields[f"{component}_ff"] = np.full(shape, 2.0)
        fields[f"{component}_e3f0"] = np.ones(shape)
        fields[f"{component}_r3f"] = np.zeros(shape)
        fields[f"{component}_mask"] = np.ones(shape)
        fields[f"{component}_denom"] = np.ones(shape)
        fields[f"{component}_frac"] = np.full(shape, 2.0)
    fields["1_r3f"][-1] = 1.0
    fields["partial"] = fields["1_frac"] + fields["2_frac"]
    fields["sum"] = fields["partial"] + fields["3_frac"]
    result = gate._replace_ne_north_mask(
        fields, np.zeros((3, 1)), np.ones(shape, dtype=bool)
    )
    wrong = gate._replace_ne_north_mask(
        fields, np.ones((3, 1)), np.ones(shape, dtype=bool)
    )
    np.testing.assert_array_equal(result["1_denom"][-1], 1.0)
    np.testing.assert_array_equal(result["1_frac"][-1], 2.0)
    np.testing.assert_array_equal(wrong["1_denom"][-1], 2.0)
    np.testing.assert_array_equal(wrong["1_frac"][-1], 1.0)
