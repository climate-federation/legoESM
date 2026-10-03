import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round126_een_nw_mask_walk as gate,
)


def test_round126_frozen_registry_is_complete():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "wrong-row",
        "northeast-isolation", "scope-route",
    )
    assert gate.EXPECTED_BEFORE["3_mask"] == (1154, 1154)
    assert gate.EXPECTED_AFTER["3_mask"] == (0, 0)


def test_round126_arm_replaces_only_nw_third_mask_descendants():
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
    north = np.arange(8, dtype=np.float64).reshape(4, 2)
    result = gate._replace_nw_north_mask(
        fields, north, np.ones(shape, dtype=bool)
    )
    np.testing.assert_array_equal(result["3_mask"][-1], np.roll(north, 1, axis=0))
    np.testing.assert_array_equal(result["3_mask"][:-1], fields["3_mask"][:-1])
    np.testing.assert_array_equal(result["1_mask"], fields["1_mask"])
    np.testing.assert_array_equal(result["2_mask"], fields["2_mask"])
    np.testing.assert_array_equal(result["3_frac"], fields["3_frac"])
    np.testing.assert_array_equal(result["sum"], fields["sum"])


def test_round126_wrong_mask_changes_descendants_when_r3f_is_nonzero():
    shape = (2, 3, 1)
    fields = {}
    for component in gate.r121.COMPONENTS:
        fields[f"{component}_ff"] = np.full(shape, 2.0)
        fields[f"{component}_e3f0"] = np.ones(shape)
        fields[f"{component}_r3f"] = np.zeros(shape)
        fields[f"{component}_mask"] = np.ones(shape)
        fields[f"{component}_denom"] = np.ones(shape)
        fields[f"{component}_frac"] = np.full(shape, 2.0)
    fields["3_r3f"][-1] = 1.0
    fields["partial"] = fields["1_frac"] + fields["2_frac"]
    fields["sum"] = fields["partial"] + fields["3_frac"]
    result = gate._replace_nw_north_mask(
        fields, np.zeros((3, 1)), np.ones(shape, dtype=bool)
    )
    wrong = gate._replace_nw_north_mask(
        fields, np.ones((3, 1)), np.ones(shape, dtype=bool)
    )
    np.testing.assert_array_equal(result["3_denom"][-1], 1.0)
    np.testing.assert_array_equal(result["3_frac"][-1], 2.0)
    np.testing.assert_array_equal(wrong["3_denom"][-1], 2.0)
    np.testing.assert_array_equal(wrong["3_frac"][-1], 1.0)
