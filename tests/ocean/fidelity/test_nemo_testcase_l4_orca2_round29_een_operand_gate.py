import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round29_een_operand_gate as gate,
)


def _capture(variant, output="parent"):
    fields = {
        "zeta": "same", "denominator": variant,
        "h_v": "same", "v": "same", "h_u": "same", "u": "same",
        "u_mask": "same", "v_mask": "same", "f_vtx": "same",
        "output_u": output, "output_v": output,
    }
    return {
        "helper_variant": variant,
        "backend": "cpu", "dtype": "float64", "record_root": "/record",
        "worktree": {"clean": True}, "call_count": 1,
        "calls": [{"index": 0, "digests": fields,
                   "is_exposed_stage2": True}],
    }


def _arrays(value):
    return {
        "exposed_u": np.array([value, 0.0]),
        "exposed_v": np.array([value, 0.0]),
        "call0_denominator": np.array([value, 0.0]),
        "stage2_vorticity_u": np.array([value, 0.0]),
        "stage2_vorticity_v": np.array([value, 0.0]),
    }


def test_round29_gate_accepts_denominator_only(monkeypatch):
    scores = iter([
        {"cells": 2, "unequal": 1, "bit_identical": False, "max_abs": 1.0},
        {"cells": 2, "unequal": 0, "bit_identical": True, "max_abs": 0.0},
        {"cells": 2, "unequal": 0, "bit_identical": True, "max_abs": 0.0},
        {"cells": 2, "unequal": 0, "bit_identical": True, "max_abs": 0.0},
        {"cells": 2, "unequal": 0, "bit_identical": True, "max_abs": 0.0},
        {"cells": 2, "unequal": 413554, "bit_identical": False,
         "max_abs": 1.0},
        {"cells": 2, "unequal": 412558, "bit_identical": False,
         "max_abs": 1.0},
    ])
    monkeypatch.setattr(gate.r27, "array_score", lambda a, b: next(scores))
    parent = _capture("parent")
    raw = _capture("raw_f", output="raw")
    result = gate.evaluate(parent, raw, _arrays(0.0), _arrays(1.0),
                           _arrays(0.0), _arrays(1.0))
    assert result["predictions"]["R29-P1"] == "CONFIRMED"


def test_round29_gate_refuses_transport_change(monkeypatch):
    parent = _capture("parent")
    raw = _capture("raw_f", output="raw")
    raw = copy.deepcopy(raw)
    raw["calls"][0]["digests"]["u"] = "changed"
    with pytest.raises(gate.GateError, match="numerator or transport"):
        gate.evaluate(parent, raw, {}, {}, {}, {})
