import numpy as np
from types import SimpleNamespace

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


def test_round124_production_builder_associates_both_northern_f_operands():
    source = gate.Path(
        gate.REPO_ROOT
        / "packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py"
    ).read_text(encoding="utf-8")
    assert "ff_north = _nemo_een_north_f(ff, grid)" in source
    assert "e3f_north = _nemo_een_north_f(e3f, grid)" in source
    assert "q_north = b(ff_north[..., None] / e3f_north)" in source


def test_round124_northern_f_association_handles_level_fields():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _nemo_een_north_f

    field = np.arange(5 * 4 * 2, dtype=np.float64).reshape(5, 4, 2)
    fold = SimpleNamespace(
        is_active=True,
        fold_j=4,
        pivot_row_stored=True,
        perm_f=np.array([3, 2, 1, 0]),
        perm_v=np.array([0, 3, 2, 1]),
    )
    associated = np.asarray(_nemo_een_north_f(field, SimpleNamespace(fold=fold)))
    np.testing.assert_array_equal(associated[:-1], field[1:])
    np.testing.assert_array_equal(associated[-1], field[-3, ::-1])
