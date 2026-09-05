"""Behavioral checks for the GYRE oracle-record census schema."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    / "nemo_testcase_l2_gyre_round15_oracle_census.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_oracle_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize(
    ("field_index", "expected"),
    enumerate(("uu_b", "vv_b", "un_adv", "vn_adv")),
)
def test_bt_frame_payload_schema_names_the_changed_field(
    tmp_path: Path, field_index: int, expected: str,
) -> None:
    size = 40 + 4 * gate.N2 * 8
    before = bytearray(size)
    after = bytearray(before)
    after[40 + field_index * gate.N2 * 8 + 111 * 8] = 1
    v1 = tmp_path / "oracle_bt_frames_kt00000001.bin"
    v2 = tmp_path / "candidate.bin"
    v1.write_bytes(before)
    v2.write_bytes(after)

    row = gate.compare_record(v1, v2)

    assert row["first_field"] == expected
    assert row["first_byte"] == 40 + field_index * gate.N2 * 8 + 111 * 8
