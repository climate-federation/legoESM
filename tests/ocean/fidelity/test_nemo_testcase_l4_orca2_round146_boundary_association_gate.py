"""Known-answer controls for the ORCA2 round-146 association gate."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _nemo_ssh_avg_reference_depth_override,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round146_boundary_association_gate as gate,
)


def test_exact_row_detects_signed_zero():
    candidate = np.array([0.0], dtype=np.float64)
    reference = np.array([-0.0], dtype=np.float64)
    row = gate.exact_row(candidate, reference)
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1
    assert row["maximum_absolute"] == 0.0


def test_exact_row_detects_one_ulp_plant():
    reference = np.array([1.0], dtype=np.float64)
    candidate = np.nextafter(reference, np.float64(np.inf))
    row = gate.exact_row(candidate, reference)
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1


def test_seven_array_registry_reorder_plant_fires():
    registry = [row[0] for row in gate.POST_FIELDS]
    registry[0], registry[1] = registry[1], registry[0]
    with pytest.raises(gate.GateError, match="registry reordered"):
        gate.validate_post_registry(registry)


def test_entry_inverse_v_registry_is_complete_and_ordered():
    names = gate.entry_inverse_v_names()
    assert len(names) == 65
    assert names[:3] == ("i000_hvr_e", "j001_hvr_e", "j002_hvr_e")
    assert names[-1] == "j064_hvr_e"


def test_entry_inverse_v_registry_reorder_plant_fires():
    oracle = {
        name: np.ones((3, 4), dtype=np.float64)
        for name in gate.entry_inverse_v_names()
    }
    with pytest.raises(gate.GateError, match="registry reordered"):
        gate.build_entry_inverse_v_override(
            oracle, plant="inverse-v-registry")


def test_wrong_entry_frame_plant_is_nonvacuous():
    oracle = {
        name: np.full((3, 4), index, dtype=np.float64)
        for index, name in enumerate(gate.entry_inverse_v_names())
    }
    override = gate.build_entry_inverse_v_override(
        oracle, plant="wrong-entry-frame")

    assert np.array_equal(
        override[1], gate.r97._to_model_v(oracle["j002_hvr_e"]))
    assert not np.array_equal(
        override[1], gate.r97._to_model_v(oracle["j001_hvr_e"]))


def test_wrong_entry_frame_plant_refuses_vacuous_record():
    oracle = {
        name: np.ones((3, 4), dtype=np.float64)
        for name in gate.entry_inverse_v_names()
    }
    with pytest.raises(gate.GateError, match="source is not distinct"):
        gate.build_entry_inverse_v_override(
            oracle, plant="wrong-entry-frame")


def test_midpoint_v_operand_registry_is_compiled_order():
    assert gate.midpoint_v_operand_names() == (
        "midpoint_ssh",
        "area_t",
        "local_area_ssh",
        "north_area_ssh",
        "reference_depth_v",
        "reciprocal_area_v",
        "ssvmask",
    )


def test_midpoint_v_bit_control_is_nonvacuous():
    reference = np.array([1.0], dtype=np.float64)
    candidate = np.nextafter(reference, np.float64(np.inf))

    assert gate.exact_row(candidate, reference)["differing_cells"] == 1


def test_reference_depth_override_none_returns_original_prep():
    prep = tuple(np.full((2, 3), value, dtype=np.float64) for value in range(5))

    assert _nemo_ssh_avg_reference_depth_override(
        prep, None, np.float64) is prep


def test_reference_depth_override_changes_only_registered_slots():
    prep = tuple(np.full((2, 3), value, dtype=np.float64) for value in range(5))
    raw = (np.full((2, 3), 7.0), np.full((2, 3), 8.0))

    result = _nemo_ssh_avg_reference_depth_override(prep, raw, np.float64)

    assert np.array_equal(result[0], raw[0])
    assert np.array_equal(result[1], raw[1])
    assert all(result[index] is prep[index] for index in range(2, 5))


def test_reference_depth_override_rejects_malformed_shape():
    prep = tuple(np.zeros((2, 3), dtype=np.float64) for _ in range(5))

    with pytest.raises(ValueError, match="shape does not match"):
        _nemo_ssh_avg_reference_depth_override(
            prep, (np.zeros((1, 3)), np.zeros((2, 3))), np.float64)


@pytest.mark.parametrize("face,changed", [("u", (1, 0)), ("v", (0, 2))])
def test_boundary_scope_accepts_only_registered_storage(face, changed):
    pre = np.zeros((4, 5), dtype=np.float64)
    post = pre.copy()
    post[changed] = 1.0
    row = gate.boundary_scope(pre, post, face)
    assert row["changed_cells"] == 1
    assert row["outside_allowed_cells"] == 0


def test_boundary_scope_rejects_interior_change():
    pre = np.zeros((4, 5), dtype=np.float64)
    post = pre.copy()
    post[1, 2] = 1.0
    row = gate.boundary_scope(pre, post, "u")
    assert row["outside_allowed_cells"] == 1
    assert row["first_outside_index"] == [1, 2]


def test_boundary_scope_accepts_u_pivot_half_row():
    pre = np.zeros((4, 9), dtype=np.float64)
    post = pre.copy()
    post[-1, 5:] = 1.0
    row = gate.boundary_scope(pre, post, "u")
    assert row["changed_cells"] == 4
    assert row["outside_allowed_cells"] == 0
