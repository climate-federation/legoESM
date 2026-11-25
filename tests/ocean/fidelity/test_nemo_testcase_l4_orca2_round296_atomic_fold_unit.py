"""Known-answer controls for round 296's atomic finite-prefix classifier."""

import json

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round296_atomic_fold_unit as gate,
)


def _region(value):
    return {
        "support": 1,
        "unequal": int(value != 0.0),
        "max_abs": value,
        "rms": value,
        "argmax": [0],
        "first_nonfinite": None,
        "nonfinite": 0,
    }


def _report(label):
    boundaries = []
    for kt in range(1, 8):
        for stage in range(4):
            value = 20.0 if (kt, stage) == (7, 1) else 1.0
            boundaries.append({
                "kt": kt,
                "stage": stage,
                "fields": {
                    field: {
                        "complete": _region(value),
                        "fold_band": _region(value),
                        "interior": _region(value),
                    }
                    for field in gate.FIELDS
                },
            })
    return {
        "status": "PASS_R249_STAGE_TABLE",
        "label": label,
        "unit": True,
        "operand_route": "resolved-candidate",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": {"commit": "abc"},
        "boundaries": boundaries,
        "passivity": [
            {"kt": kt, "fields": {"T": True, "S": True}}
            for kt in range(1, 8)
        ],
    }


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _inputs(tmp_path):
    prefixes = [
        _write(tmp_path / f"{label}.json", _report(label))
        for label in gate.LABELS
    ]
    refusals = []
    for label in gate.LABELS:
        path = tmp_path / f"{label}.log"
        path.write_text(
            f"PROGRESS round249 {label} unit=1 kt=7\n{gate.REFUSAL}\n",
            encoding="utf-8",
        )
        refusals.append(path)
    raw_depth = _write(tmp_path / "depth.json", {
        "direct_arms": {"raw_hv_only": {"face_depth_v_mid": {
            "complete": {"unequal": 0, "max_abs": 0.0}}}}})
    raw_mask = _write(tmp_path / "mask.json", {
        "single_substitution": {"ssvmask": {"complete": {
            "unequal": 0, "max_abs": 0.0}}}})
    return prefixes, refusals, raw_depth, raw_mask


def test_classifier_retains_finite_prefix_and_all_plants_fire(tmp_path):
    args = _inputs(tmp_path)
    result = gate.classify(*args)
    assert result["status"] == "HELD_R296_LIVE_THICKNESS_AT_KT8"
    assert result["terminal"]["last_completed_kt"] == 7
    for label in gate.LABELS:
        first = result["labels"][label]["first_post_floor_over_10x"]
        assert (first["kt"], first["stage"], first["region"]) == (
            7, 1, "fold_band")
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(*args, plant=plant)


def test_missing_terminal_refusal_is_rejected(tmp_path):
    prefixes, refusals, raw_depth, raw_mask = _inputs(tmp_path)
    refusals[1].write_text(
        "PROGRESS round249 given_nemo_entry unit=1 kt=7\n",
        encoding="utf-8",
    )
    with pytest.raises(gate.GateError, match="refusal moved"):
        gate.classify(prefixes, refusals, raw_depth, raw_mask)
