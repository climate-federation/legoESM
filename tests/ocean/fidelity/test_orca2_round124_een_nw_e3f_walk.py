import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round124_een_nw_e3f_walk as gate,
)


def test_round124_frozen_registry_is_complete():
    assert gate.PLANTS == (
        "none", "oracle-bit", "candidate-bit", "wrong-row",
        "northeast-isolation", "scope-route",
    )
    assert gate.EXPECTED_AFTER["3_e3f0"] == (0, 0)
    assert gate.EXPECTED_AFTER["3_mask"] == (1154, 1154)


def test_round124_arm_replaces_only_nw_third_fraction_descendants():
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
    north = np.broadcast_to(np.arange(4.0)[:, None] + 2.0, (4, 2))
    result = gate._replace_nw_north_e3f(
        fields, north, np.ones(shape, dtype=bool)
    )
    np.testing.assert_array_equal(result["3_e3f0"][-1], np.roll(north, 1, axis=0))
    np.testing.assert_array_equal(result["3_e3f0"][:-1], fields["3_e3f0"][:-1])
    np.testing.assert_array_equal(result["1_e3f0"], fields["1_e3f0"])
    np.testing.assert_array_equal(result["2_e3f0"], fields["2_e3f0"])
    np.testing.assert_array_equal(
        result["sum"][-1], fields["partial"][-1] + result["3_frac"][-1]
    )
