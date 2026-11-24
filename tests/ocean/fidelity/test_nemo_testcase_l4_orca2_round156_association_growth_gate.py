"""Controls for the ORCA2 round-156 finite-growth gate."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round156_association_growth_gate as gate,
)


def test_exact_pair_row_detects_one_bit() -> None:
    reference = np.array([1.0], dtype=np.float64)
    candidate = np.nextafter(reference, np.float64(np.inf))
    row = gate.exact_pair_row(candidate, reference)
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1


def test_exact_pair_row_detects_signed_zero() -> None:
    row = gate.exact_pair_row(
        np.array([0.0], dtype=np.float64),
        np.array([-0.0], dtype=np.float64),
    )
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1
    assert row["maximum_absolute"] == 0.0


def test_exact_pair_row_refuses_reference_nonfinite() -> None:
    with pytest.raises(gate.GateError, match="production reference is non-finite"):
        gate.exact_pair_row(np.array([0.0]), np.array([np.nan]))


def test_exact_pair_row_names_candidate_nonfinite_without_reduction() -> None:
    row = gate.exact_pair_row(np.array([np.inf]), np.array([0.0]))
    assert not row["candidate_finite"]
    assert row["nonfinite_cells"] == 1
    assert row["maximum_absolute"] is None


def test_association_registry_is_compiled_call_order() -> None:
    assert gate.ASSOCIATION_FIELDS == (
        "u", "v", "depth_u", "depth_v", "inverse_u", "inverse_v", "eta",
    )


def test_unknown_association_field_refuses() -> None:
    with pytest.raises(gate.GateError, match="unknown association field"):
        gate.validate_association_field("plausible-zero")


@pytest.mark.parametrize(
    "plant", ("comparison-bit", "signed-zero", "nonfinite", "field-selector"))
def test_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError, match="plant fired|unknown association"):
        gate.run_plant(plant)
