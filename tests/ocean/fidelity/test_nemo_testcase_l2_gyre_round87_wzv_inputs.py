"""Unit controls for the Round-87 WZV input walk."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases"
          / "nemo_testcase_l2_gyre_round87_wzv_inputs.py")
SPEC = importlib.util.spec_from_file_location("round87_wzv_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROUND87 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUND87)


def test_recurrence_has_zero_bottom_and_integrates_bottom_up():
    e3div = np.asarray([[[1.0, 2.0]]], dtype=np.float64)
    e3t0 = np.ones_like(e3div)
    out = ROUND87._recur(e3div, e3t0, 1.0, np.asarray([[3.0]]), np.ones_like(e3div))
    assert np.array_equal(out, np.asarray([[[-9.0, -5.0, 0.0]]]))


def test_active_w_appends_the_unused_dry_bottom_slot():
    masks = {"t": np.ones((2, 3, 4), dtype=bool),
             "u": np.ones((2, 3, 4), dtype=bool),
             "v": np.ones((2, 3, 4), dtype=bool)}
    active = ROUND87._active("ww", masks)
    assert active.shape == (2, 3, 5)
    assert not active[..., -1].any()


def test_operand_order_places_stored_kaa_after_exact_early_chain():
    assert ROUND87.ORDER.index("eta_kaa") > ROUND87.ORDER.index("e3div")
    assert ROUND87.ORDER.index("eta_kaa") < ROUND87.ORDER.index("stretch")
