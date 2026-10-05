"""Known-answer controls for the ORCA2 round-146 association gate."""

from __future__ import annotations

import numpy as np
import pytest

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
