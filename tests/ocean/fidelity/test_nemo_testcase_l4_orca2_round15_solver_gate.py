"""Non-vacuity and private-hook checks for the ORCA2 round-15 gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        / "nemo_testcase_l4_orca2_round15_barotropic_solver_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("_r15_solver_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_drag_rate_override_is_private_and_off_by_default():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    assert _NEMOWSRK3TestHooks().barotropic_drag_rate_override is None


def test_native_rank0_slices_follow_the_record_layout():
    gate = _gate()
    u = np.arange(148 * 181).reshape(148, 181)
    v = np.arange(149 * 180).reshape(149, 180)
    assert np.array_equal(gate._native(u, "u"), u[:, 1:91])
    assert np.array_equal(gate._native(v, "v"), v[1:, :90])


def test_source_walk_has_no_duplicate_boundaries():
    gate = _gate()
    names = [name for name, _, _ in gate.SOURCE_ORDER]
    assert len(names) == len(set(names))
    assert names[:6] == [
        "u_history_b", "v_history_b", "u_history_bb", "v_history_bb",
        "eta_history_b", "eta_history_bb",
    ]
    assert names[6:9] == ["eta_entry", "u_entry", "v_entry"]
    assert names[-2:] == ["u_exit", "v_exit"]


def test_numpy_plant_coordinates_are_made_json_serializable():
    import json

    gate = _gate()
    index = tuple(np.argwhere(np.eye(2, dtype=bool))[0])
    encoded = json.dumps(gate._json_index(index))
    assert encoded == "[0, 0]"
