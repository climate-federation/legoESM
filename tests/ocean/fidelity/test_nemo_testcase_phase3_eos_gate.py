"""Direct controls for the EOS-first NEMO testcase phase-3 gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path


GATE_PATH = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_eos_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_phase3_eos_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_source_parser_reads_the_complete_teos10_density_table():
    coefficients = gate.parse_teos10_density_coefficients()
    assert len([name for name in coefficients if name.startswith("EOS")]) == 52


def test_both_cards_pass_the_eos_first_gate_on_registered_dumps():
    for case in gate.ROOTS:
        report = gate.run(case)
        assert report["card_eos"] == "nemo_teos10"
        assert report["status"] == "AT-BAR"


def test_planted_wet_density_violation_turns_gate_red():
    report = gate.run("LOCK_EXCHANGE-zco", plant=True)
    assert report["status"] == "DEBT"
    assert report["card_rows"][0]["status"] == "DEBT"
