"""Direct controls for the SI3 round-14 short-window diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path


TOOL = (
    Path(__file__).parents[2]
    / "scripts/validate/ocean_fidelity/testcases/nemo_si3_r14_short_window_diagnosis.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_r14_short_window_diagnosis_test", TOOL)
assert SPEC and SPEC.loader
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def test_trajectory_hash_excludes_timing_but_detects_a_scored_row_change():
    left = [{"compiled_step_wall_s": 1.0, "rows": [{"name": "step9.v_i", "max_abs": 0.0}]}]
    retimed = [{"compiled_step_wall_s": 9.0, "rows": [{"name": "step9.v_i", "max_abs": 0.0}]}]
    changed = [{"compiled_step_wall_s": 1.0, "rows": [{"name": "step9.v_i", "max_abs": 1e-16}]}]

    assert diagnosis._trajectory_rows_sha256(left) == diagnosis._trajectory_rows_sha256(retimed)
    assert diagnosis._trajectory_rows_sha256(left) != diagnosis._trajectory_rows_sha256(changed)


def test_libm_counter_reset_and_snapshot_are_independent():
    stats = {
        "calls": 2,
        "elements": 7,
        "wall_s": 0.125,
        "by_name": {"exp": {"calls": 2, "elements": 7, "wall_s": 0.125}},
    }
    snapshot = diagnosis._snapshot_libm(stats)
    diagnosis._reset_libm(stats)

    assert snapshot["by_name"]["exp"]["calls"] == 2
    assert stats == {"calls": 0, "elements": 0, "wall_s": 0.0, "by_name": {}}
