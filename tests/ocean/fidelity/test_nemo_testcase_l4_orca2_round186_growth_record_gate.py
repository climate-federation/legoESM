"""Controls for the round-186 operator-run growth record."""

from __future__ import annotations

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round186_growth_record_gate as gate,
)


def _source() -> str:
    return """&namrun
   nn_itend = 240
   nn_stock = 240
/
&namtra_ldf
   ln_traldf_lap = .true.
/
"""


def test_render_changes_only_run_protocol(tmp_path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.write_text(_source())
    candidate.write_text(gate.render_deck(source.read_text()))
    report = gate.validate_deck(source, candidate)
    assert report["status"] == "RUN_PROTOCOL_ONLY"
    assert tuple(report["steps"]) == gate.STEPS


def test_hidden_deck_plant_fires(tmp_path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.write_text(_source())
    candidate.write_text(gate.render_deck(source.read_text()))
    with pytest.raises(gate.GateError, match="hidden growth-deck delta"):
        gate.validate_deck(source, candidate, "hidden-deck")


def test_render_refuses_missing_controls() -> None:
    with pytest.raises(gate.GateError, match="nn_stock"):
        gate.render_deck("&namrun\n nn_itend=240\n/\n")
