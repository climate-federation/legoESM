"""Unit controls for the round-51 OVERFLOW pair gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round51_pair_gate as gate  # noqa: E402


def test_owned_record_mapping_transposes_fortran_xy_axes():
    values = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
    mapped = gate._nemo_owned(values)
    assert mapped.shape == (3, 4, 2)
    assert np.array_equal(mapped[:, :, 1], values[:, :, 1].T)


def test_exact_scorer_one_ulp_plant_fires_once():
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    mask = np.ones_like(values, dtype=bool)
    clean = gate._score("clean", values, values, mask)
    planted = gate._score("plant", values, values, mask, plant=True)
    assert clean["status"] == "BIT_EXACT" and clean["n_unequal"] == 0
    assert planted["status"] == "NON_BIT" and planted["n_unequal"] == 1


def test_sidecar_hash_and_first_moved_boundary(tmp_path):
    fields = {
        name: np.zeros(2, dtype=np.float64)
        for name in gate.SOURCE_ORDER
    }
    base_path = tmp_path / "base.json"
    base = {
        "format": "nemo-testcase-l1-overflow-round51-pair-v1",
        "worktree": {"commit": "base"},
        "given_input_rows": [
            {"name": "given.s1.qco.T", "exact": True},
            {"name": "given.s1.qco.S", "exact": True},
            {"name": "given.s2.eos.rhd", "exact": True},
            {"name": "given.s2.hpg.u", "exact": False},
        ],
    }
    base["sidecar"] = gate._write_sidecar(base_path, fields)
    base_path.write_text(json.dumps(base))

    moved = {name: value.copy() for name, value in fields.items()}
    moved["s1.tracer.Kaa.T"][1] = np.nextafter(0.0, 1.0)
    candidate_path = tmp_path / "candidate.json"
    candidate = {
        "format": base["format"],
        "worktree": {"commit": "candidate"},
        "given_input_rows": base["given_input_rows"],
    }
    candidate["sidecar"] = gate._write_sidecar(candidate_path, moved)
    result = gate.compare(base_path, candidate)
    assert result["first_moved_boundary"]["name"] == "s1.tracer.Kaa.T"
    assert result["R51-P2"] == "CONFIRMED"
    assert result["R51-P3"] == "CONFIRMED"
    assert result["R51-P4"] == "CONFIRMED"

    candidate["sidecar"]["sha256"] = "0" * 64
    try:
        gate.compare(base_path, candidate)
    except gate.GateError as error:
        assert "hash drift" in str(error)
    else:  # pragma: no cover - fail-closed assertion
        raise AssertionError("tampered sidecar was admitted")
