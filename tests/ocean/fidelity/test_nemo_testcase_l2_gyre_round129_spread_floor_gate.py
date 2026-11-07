"""Direct and record-backed tests for the Round-129 spread-floor gate."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

GATE = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
        / "ocean_fidelity" / "testcases"
        / "nemo_testcase_l2_gyre_round129_spread_floor_gate.py")
EVIDENCE = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round129")
RECEIPT = (Path(__file__).resolve().parents[3] / "docs" / "ocean" / "fidelity"
           / "testcases"
           / "nemo_testcases_l2_gyre_round129_spread_floor_receipt.md")


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("round129_spread_gate", GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_self_check_runs_as_a_subprocess():
    result = subprocess.run([sys.executable, str(GATE), "--self-check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SELF-CHECK OK" in result.stdout


def test_verdict_boundary_uses_exact_binary64_comparison(gate):
    bar = gate.ROUND129_RATIO_BAR
    assert gate.classify_ratio(np.nextafter(bar, -np.inf)) == "NOT_MET"
    assert gate.classify_ratio(bar) == "MET"
    assert gate.classify_ratio(np.nextafter(bar, np.inf)) == "MET"


def test_pair_registry_is_all_six_pairs_and_can_fail(gate):
    wet = np.ones((1, 1, 1), dtype=bool)
    states = {seed: {"T": np.full((1, 1, 1), float(seed))}
              for seed in gate.YEAR.SEEDS}
    report = gate.pairwise_t3d(states, wet)
    assert report["pair_count"] == 6
    assert set(report["pairs_K"]) == {
        "0-1", "0-2", "0-3", "1-2", "1-3", "2-3"}
    assert report["maximum_K"] == 3.0
    with pytest.raises(gate.GateError, match="pair registry"):
        gate.pairwise_t3d(states, wet, plant="pair-registry")


def test_known_growth_exponent_is_recovered(gate):
    days = np.asarray((1.0, 2.0, 4.0, 8.0))
    report = gate._loglog_fit(days, days ** 2.3)
    assert report["exponent"] == pytest.approx(2.3, abs=1.0e-12)
    # Non-vacuity: a flat series must not be called a 2.3 exponent.
    assert abs(gate._loglog_fit(days, np.ones_like(days))["exponent"]) < 1e-12


@pytest.mark.skipif(not (EVIDENCE / "lego_seed1_year/manifest.json").is_file(),
                    reason="Round-129 ensemble is not on this machine")
def test_record_backed_gate_passes(tmp_path):
    target = tmp_path / "spread.json"
    result = subprocess.run(
        [sys.executable, str(GATE), "--score", "--json", str(target)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
    assert "STATUS YEAR-BAR-" in result.stdout
    assert target.is_file()


@pytest.mark.skipif(not (EVIDENCE / "lego_seed1_year/manifest.json").is_file(),
                    reason="Round-129 ensemble is not on this machine")
@pytest.mark.parametrize("plant", ["initial-temperature-ulp", "pair-registry"])
def test_record_backed_plants_exit_nonzero(plant):
    result = subprocess.run(
        [sys.executable, str(GATE), "--score", "--plant", plant],
        capture_output=True, text=True)
    assert result.returncode != 0, result.stdout[-3000:]
    assert "STATUS PLANT-FIRED" in result.stdout
    assert "REFUSE Round-129" in result.stderr


@pytest.mark.skipif(not (EVIDENCE / "spread_floor.json").is_file(),
                    reason="Round-129 score is not on this machine")
def test_receipt_table_is_the_authoritative_json():
    report = json.loads((EVIDENCE / "spread_floor.json").read_text())
    observed = {}
    pattern = re.compile(r"^\|\s*(\d+)\s*\|(.+)\|$")
    for line in RECEIPT.read_text().splitlines():
        match = pattern.match(line)
        if not match or int(match.group(1)) not in report["days"]:
            continue
        cells = [cell.strip().strip("`")
                 for cell in match.group(2).split("|")]
        assert len(cells) == 6, line
        observed[int(match.group(1))] = [float(cell) for cell in cells]
    assert set(observed) == set(report["days"])
    for day in report["days"]:
        row = report["rows"][str(day)]
        expected = [row["lego_spread"]["maximum_K"],
                    row["nemo_spread"]["maximum_K"],
                    *(row["matched_seed_gaps_K"][str(seed)]
                      for seed in range(4))]
        assert observed[day] == expected, day
    text = RECEIPT.read_text()
    for value in (report["day240_verdict"]
                  ["ratio_lego_spread_over_seed0_gap"],
                  report["day240_verdict"]["nemo_spread_over_lego_spread"],
                  report["growth"]["spread_fit_through_day240"]["exponent"],
                  report["growth"]["spread_day240_over_day30"]):
        assert repr(value) in text
