"""Controls for the ORCA2 O1 NCAR bulk gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

GATE_PATH = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/nemo_ncar_o1_bulk_gate.py"
SPEC = importlib.util.spec_from_file_location("nemo_ncar_o1_bulk_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_coverage_is_complete_and_waives_only_unowned_slots() -> None:
    gate.validate_coverage()
    assert len(gate.COVERAGE) == 20
    assert {name for name, row in gate.COVERAGE.items() if row[0] == "WAIVED"} == {"cd_du", "qlwn"}
    bad = dict(gate.COVERAGE)
    bad.pop("taum")
    with pytest.raises(gate.GateError, match="incomplete"):
        gate.validate_coverage(bad)


def test_score_reports_bit_bar_and_row_scale_ulp_separately() -> None:
    oracle = np.asarray([1.0, 2.0], dtype=np.float64)
    candidate = oracle.copy()
    candidate[0] = np.nextafter(candidate[0], np.float64(np.inf))
    row = gate.score(candidate, oracle, "plant")
    assert row["bit_unequal_over_n"] == "1 / 2"
    assert row["over_bar_count"] == 0
    assert row["max_row_scale_ulp_error"] == 0.5


def test_missing_stream_raises_named_gate_error(tmp_path: Path) -> None:
    with pytest.raises(gate.GateError, match="missing O1 stream"):
        gate.read_record(tmp_path / "missing.bin")


def test_malformed_header_raises_named_gate_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(gate.MAGIC + b"\0" * 32)
    original = gate.EXPECTED_SHA256
    try:
        gate.EXPECTED_SHA256 = gate.sha256(path)
        with pytest.raises(gate.GateError, match="invalid header"):
            gate.read_record(path)
    finally:
        gate.EXPECTED_SHA256 = original


@pytest.mark.skipif(
    not (gate.DEFAULT_ROOT / gate.RECORD_NAME).exists(),
    reason="retained ORCA2 O1 oracle is not mounted",
)
def test_full_o1_gate_and_row_plants() -> None:
    result = gate.evaluate()
    assert result["status"] == "AT_BAR"
    assert result["bit_status"] == "BIT_IDENTICAL"
    assert result["bit_unequal_over_n"] == "0 / 158292"
    assert len(result["plants"]) == 18
    assert all(row == {"field": row["field"], "status": "PASS_NONZERO", "exit_code": 1} for row in result["plants"])
    planted = gate.evaluate(plant_field="qns")
    assert planted["bit_status"] == "NON_BIT_IDENTICAL"
    assert next(row for row in planted["rows"] if row["field"] == "qns")["non_bit_identical_count"] == 1
    poisoned = gate.evaluate(plant_libm_log=True)
    assert poisoned["bit_status"] == "NON_BIT_IDENTICAL"
    assert any(row["non_bit_identical_count"] for row in poisoned["rows"])


def test_ncar_selector_is_shared_and_jittable() -> None:
    from legoesm.core.bulk_flux import nemo_ncar_ocean_bulk, validate_bulk_scheme
    from legoesm.core.bulk_flux import air_sea_fluxes

    assert validate_bulk_scheme("nemo_ncar") is None
    assert callable(nemo_ncar_ocean_bulk)
    assert callable(air_sea_fluxes)
