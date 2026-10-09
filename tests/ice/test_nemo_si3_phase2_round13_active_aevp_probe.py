"""Harness tests for the Round-13 exact-input active aEVP discriminator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = (
    ROOT
    / "scripts/validate/ocean_fidelity/testcases"
    / "nemo_si3_phase2_round13_active_aevp_probe.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("round13_active_aevp_probe_test", PROBE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operand_and_setup_registries_are_complete_and_unique() -> None:
    probe = _load_probe()
    assert len(probe.OPERAND_REGISTRY) == 48
    assert len(set(probe.OPERAND_REGISTRY)) == 48
    assert len(probe.SETUP_REGISTRY) == 37
    assert len(set(probe.SETUP_REGISTRY)) == 37


def test_named_row_plant_moves_exact_score_to_debt() -> None:
    probe = _load_probe()
    oracle = np.zeros((4, 4), dtype=np.float64)
    candidate = oracle.copy()
    clean = probe._score("subcycle001.force_u", oracle, candidate)
    candidate[0, 0] = np.nextafter(candidate[0, 0], np.inf)
    planted = probe._score("subcycle001.force_u", oracle, candidate)
    assert probe._assert_plant_transition(clean, planted)
    assert clean["bitwise_nonzero_over_n"] == "0 / 16"
    assert planted["bitwise_nonzero_over_n"] == "1 / 16"
    assert planted["max_ulp"] == 1
