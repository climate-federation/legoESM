"""Unit tests for legoesm.ocean.fidelity.tolerances schema validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legoesm.ocean.fidelity import tolerances


def _valid_payload() -> dict:
    return {
        "schema_version": 1,
        "tier": 1,
        "description": "Linear wave dispersion vs analytical (test fixture)",
        "common": {"precision": "x64", "grid": "latlon_cgrid"},
        "cases": {
            "case_a": {
                "metric": "dispersion_rms_error",
                "tolerance_window": [0.0, 0.02],
                "reference": {
                    "kind": "analytical",
                    "fn": "legoesm.ocean.fidelity.references:igw_omega",
                },
                "ci_marker": "fast",
                "expected_runtime_s": 60,
            },
        },
    }


def test_validate_valid_payload_returns_dataclass():
    parsed = tolerances.validate_tier_payload(_valid_payload())
    assert parsed.tier == 1
    assert parsed.schema_version == 1
    assert "case_a" in parsed.cases
    case = parsed.cases["case_a"]
    assert case.tolerance_window == (0.0, 0.02)
    assert case.ci_marker == "fast"
    assert case.reference["kind"] == "analytical"
    assert case.expected_runtime_s == 60.0
    assert case.subcase_filters is None


def test_subcase_filters_passed_through():
    payload = _valid_payload()
    payload["cases"]["case_a"]["subcase_filters"] = {"k_max_dx": 4}
    parsed = tolerances.validate_tier_payload(payload)
    assert parsed.cases["case_a"].subcase_filters == {"k_max_dx": 4}


@pytest.mark.parametrize("missing", ["schema_version", "tier", "description", "common", "cases"])
def test_missing_top_level_key_raises(missing):
    payload = _valid_payload()
    del payload[missing]
    with pytest.raises(tolerances.TolerancesSchemaError, match=missing):
        tolerances.validate_tier_payload(payload)


def test_wrong_schema_version_raises():
    payload = _valid_payload()
    payload["schema_version"] = 99
    with pytest.raises(tolerances.TolerancesSchemaError, match="schema_version"):
        tolerances.validate_tier_payload(payload)


@pytest.mark.parametrize("bad_tier", [-1, 9, "1", 1.0])
def test_tier_out_of_range_or_wrong_type_raises(bad_tier):
    payload = _valid_payload()
    payload["tier"] = bad_tier
    with pytest.raises(tolerances.TolerancesSchemaError, match="tier"):
        tolerances.validate_tier_payload(payload)


def test_empty_cases_raises():
    payload = _valid_payload()
    payload["cases"] = {}
    with pytest.raises(tolerances.TolerancesSchemaError, match="cases"):
        tolerances.validate_tier_payload(payload)


@pytest.mark.parametrize("bad_window", [[0.1, 0.0], [1.0], [1.0, 2.0, 3.0], "not a list"])
def test_bad_tolerance_window_raises(bad_window):
    payload = _valid_payload()
    payload["cases"]["case_a"]["tolerance_window"] = bad_window
    with pytest.raises(tolerances.TolerancesSchemaError, match="tolerance_window"):
        tolerances.validate_tier_payload(payload)


def test_unknown_reference_kind_raises():
    payload = _valid_payload()
    payload["cases"]["case_a"]["reference"]["kind"] = "magic"
    with pytest.raises(tolerances.TolerancesSchemaError, match="reference.kind"):
        tolerances.validate_tier_payload(payload)


def test_unknown_ci_marker_raises():
    payload = _valid_payload()
    payload["cases"]["case_a"]["ci_marker"] = "weekly"
    with pytest.raises(tolerances.TolerancesSchemaError, match="ci_marker"):
        tolerances.validate_tier_payload(payload)


def test_negative_runtime_raises():
    payload = _valid_payload()
    payload["cases"]["case_a"]["expected_runtime_s"] = -5
    with pytest.raises(tolerances.TolerancesSchemaError, match="expected_runtime_s"):
        tolerances.validate_tier_payload(payload)


def test_load_tier_file_round_trip(tmp_path):
    payload = _valid_payload()
    path = tmp_path / "tier1.json"
    path.write_text(json.dumps(payload))
    parsed = tolerances.load_tier_file(path)
    assert parsed.tier == 1
    assert parsed.cases["case_a"].metric == "dispersion_rms_error"


def test_load_tier_file_error_prepends_path(tmp_path):
    payload = _valid_payload()
    payload["tier"] = 99
    path = tmp_path / "tier_bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(tolerances.TolerancesSchemaError, match=str(path.name)):
        tolerances.load_tier_file(path)


def test_validated_dataclass_is_frozen():
    parsed = tolerances.validate_tier_payload(_valid_payload())
    with pytest.raises(Exception):
        parsed.tier = 0  # type: ignore[misc]
