from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round82_btstep_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round82_btstep_walk", SCRIPT)
assert SPEC and SPEC.loader
WALK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WALK)


def _exact() -> dict:
    return {
        "bit_exact": True,
        "differing_cells": 0,
        "wet_cells": 1,
        "absolute_max": 0.0,
        "reference_max_abs": 1.0,
    }


def test_first_non_bit_uses_compiled_order() -> None:
    row = {"substep": 1, **{name: _exact() for name in WALK.SOURCE_ORDER}}
    row["slow_v"] = {**_exact(), "bit_exact": False, "differing_cells": 1}
    row["u_exit"] = {**_exact(), "bit_exact": False, "differing_cells": 1}
    assert WALK.first_non_bit([row])["boundary"] == "slow_v"


def test_trace_mapping_covers_every_record_array() -> None:
    assert set(WALK.TRACE_KEYS).issubset(set(WALK.round81.ARRAY_FIELDS))
    assert set(WALK.round81.ARRAY_FIELDS).issubset(set(WALK.SOURCE_ORDER))
    assert len(WALK.SOURCE_ORDER) == len(WALK.round81.ARRAY_FIELDS) + 7
    assert WALK._stagger("depth_u_mid") == "u"
    assert WALK._stagger("depth_v_mid") == "v"
