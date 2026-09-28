"""Controls for round 63's source-ordered dyn_zdf walk."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPTS))

import nemo_testcase_l1_overflow_round63_dynzdf_walk_gate as gate  # noqa: E402


def _reports(tmp_path, implicit):
    oracle = np.zeros(3, dtype=np.float64)
    base = {}
    candidate = {}
    for boundary in gate.BOUNDARIES:
        for component in gate.COMPONENTS:
            name = f"s3.zdf.{boundary}.{component}"
            base[f"oracle::{name}"] = oracle
            candidate[f"oracle::{name}"] = oracle
            base[name] = np.array([3.0, 2.0, 1.0])
            candidate[name] = (
                implicit.copy()
                if boundary == "implicit_solve" and component == "u"
                else np.array([2.0, 1.0, 0.5])
            )
    base_path = tmp_path / "base.json"
    base_report = {
        "format": gate.FORMAT,
        "worktree": {"commit": "base"},
        "sidecar": gate._write_sidecar(base_path, base),
    }
    base_path.write_text(json.dumps(base_report))
    candidate_report = {
        "format": gate.FORMAT,
        "worktree": {"commit": "candidate"},
        "sidecar": gate._write_sidecar(tmp_path / "candidate.json", candidate),
    }
    return base_path, candidate_report


def test_strict_implicit_reversal_names_owner(tmp_path):
    base, candidate = _reports(tmp_path, np.array([4.0, 3.0, 2.0]))
    result = gate.compare(base, candidate)
    assert result["first_direction_change"]["name"] == (
        "s3.zdf.implicit_solve.u")
    assert result["compensating_owner"] == "s3.zdf.implicit_solve.u"


def test_mixed_implicit_boundary_does_not_invent_owner(tmp_path):
    base, candidate = _reports(tmp_path, np.array([2.0, 3.0, 0.5]))
    result = gate.compare(base, candidate)
    assert result["first_direction_change"]["name"] == (
        "s3.zdf.implicit_solve.u")
    assert result["first_direction_change"]["direction"] == "MIXED"
    assert result["first_direction_reversal"] is None
    assert result["compensating_owner"] is None


def test_walk_is_complete_and_source_observer_is_write_only():
    assert gate.SOURCE_ORDER == tuple(
        f"s3.zdf.{boundary}.{component}"
        for boundary in gate.BOUNDARIES for component in gate.COMPONENTS
    )
    source = gate.Path(
        gate.__file__).parents[4] / (
            "packages/ocean/legoesm/ocean/dynamics/"
            "ocean_model_latlon_cgrid.py")
    text = source.read_text()
    assert "zdf_momentum_observer: object = None" in text
    assert "jax.debug.callback(\n                    _zdf_momentum_observer" in text
    assert "zdf_momentum_observer" not in gate.build_nemo_testcase_card(
        "OVERFLOW-zps").recipe.model_config._fields


def test_ordinary_reference_is_hash_bound(tmp_path):
    state = tmp_path / "ordinary.state.npz"
    np.savez_compressed(state, u=np.arange(3, dtype=np.float64))
    report = tmp_path / "ordinary.json"
    report.write_text(json.dumps({
        "format": gate.FORMAT,
        "status": "ORDINARY_WRITTEN",
        "ordinary_state": {
            "path": str(state),
            "sha256": gate.R60._sha256(state),
            "fields": ["u"],
        },
    }))
    _, arrays = gate._read_ordinary(report)
    assert np.array_equal(arrays["u"], np.arange(3, dtype=np.float64))
    state.write_bytes(state.read_bytes() + b"x")
    try:
        gate._read_ordinary(report)
    except RuntimeError as error:
        assert "hash drift" in str(error)
    else:  # pragma: no cover - non-vacuity guard
        raise AssertionError("ordinary-state hash plant did not fire")
