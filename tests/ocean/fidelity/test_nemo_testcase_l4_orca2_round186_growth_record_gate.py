"""Controls for the round-186 operator-run growth record."""

from __future__ import annotations

import re

import numpy as np
import pytest
from netCDF4 import Dataset

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
    assert tuple(report["steps"]) == gate.REPLACEMENT_STEPS
    assert report["restart_mode"] == "bounded-list-step95"
    assert report["terminal_sentinel"]
    assert "ln_rst_list = .true." in candidate.read_text()
    assert "nn_stocklist = 95, 96" in candidate.read_text()


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


def test_render_refuses_odd_terminal_without_sbc_pair() -> None:
    with pytest.raises(gate.GateError, match="unsupported terminal step 95"):
        gate.render_deck(_source(), 95)


def test_compiled_restart_list_capacity_plant_fires(tmp_path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.write_text(_source())
    candidate.write_text(gate.render_deck(source.read_text()))
    with pytest.raises(gate.GateError, match="capacity 10 exceeded"):
        gate.validate_deck(source, candidate, "oversized-list")


def test_disabled_list_plant_fires(tmp_path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.write_text(_source())
    candidate.write_text(gate.render_deck(source.read_text()))
    with pytest.raises(gate.GateError, match="restart-list mode is off"):
        gate.validate_deck(source, candidate, "list-disabled")


def test_terminal_reopen_requires_zero_length_payload(tmp_path) -> None:
    path = tmp_path / "restart.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time_counter", None)
        dataset.createVariable("kt", "f8").assignValue(0.0)
        for name in gate.FIELDS:
            dataset.createVariable(name, "f8", ("time_counter",))
    row = gate._terminal_overwrite(path)
    assert row is not None
    assert row["classification"] == "TERMINAL_REOPEN_TRUNCATED_COMPLETED_RESTART"

    with Dataset(path, "a") as dataset:
        dataset["tn"][0] = 1.0
    assert gate._terminal_overwrite(path) is None


def test_terminal_sentinel_prefix_and_its_plants(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source"
    deck_a = tmp_path / "deck_a"
    deck_b = tmp_path / "deck_b"
    source.write_text(_source())
    rendered = gate.render_deck(source.read_text())
    deck_a.write_text(rendered)
    deck_b.write_text(rendered)

    def fake_read(path, expected):
        match = re.search(r"_(\d{8})_restart_", path.name)
        gate.require(match is not None, f"missing restart: {path}")
        gate.require(int(match.group(1)) == expected,
                     f"{path.name}: kt is not {expected}")
        arrays = {name: np.ones((1,), dtype=np.float64) for name in gate.FIELDS}
        return arrays, {"path": str(path)}

    monkeypatch.setattr(gate, "_read", fake_read)
    monkeypatch.setattr(gate, "_terminal_overwrite", lambda path: None)
    result = gate.admit(
        source, deck_a, deck_b, tmp_path / "a", tmp_path / "b",
        tmp_path / "calibration", prefix_a=tmp_path / "prefix_a",
        prefix_b=tmp_path / "prefix_b")
    assert result["status"] == "PASS_R189_GROWTH_RECORD"
    assert len(result["sentinel_comparisons"]) == 2

    for plant, message in (
        ("missing-rank", "missing restart"),
        ("twin-ulp", "step 95 rank 0: twin payload moved"),
        ("step10-calibration", "step-10 calibration moved rank 0"),
        ("missing-sentinel", "missing restart"),
        ("sentinel-truncation", "payload shape moved"),
        ("sentinel-header", "kt is not 95"),
    ):
        with pytest.raises(gate.GateError, match=message):
            gate.admit(
                source, deck_a, deck_b, tmp_path / "a", tmp_path / "b",
                tmp_path / "calibration", plant,
                tmp_path / "prefix_a", tmp_path / "prefix_b")
