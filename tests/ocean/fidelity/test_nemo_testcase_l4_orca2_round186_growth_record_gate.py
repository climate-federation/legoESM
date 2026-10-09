"""Controls for the round-186 operator-run growth record."""

from __future__ import annotations

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


def test_render_safe_terminal_avoids_post_checkpoint_step(tmp_path) -> None:
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.write_text(_source())
    candidate.write_text(gate.render_deck(
        source.read_text(), gate.SAFE_ITEND))
    report = gate.validate_deck(source, candidate)
    assert report["itend"] == gate.SAFE_ITEND
    assert report["steps"][-1] == gate.SAFE_ITEND


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
