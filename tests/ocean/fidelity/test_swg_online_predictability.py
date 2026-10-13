"""Direct synthetic tests for the SWG predictability instrument."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[3] / "scripts/validate/"
          "ocean_fidelity/testcases/nemo_testcase_l1_swg_online_predictability.py")
SPEC = importlib.util.spec_from_file_location("swg_online_predictability", SCRIPT)
predictability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predictability)


def test_first_efold_crossing_recovers_exact_exponential_and_partial_curve():
    days = np.arange(6, dtype=np.float64)
    tau = 2.5
    rms = 1.0e-6 * np.exp(days / tau)
    assert predictability.first_efold_crossing(days, rms, rms[0]) == pytest.approx(tau)
    assert predictability.first_efold_crossing(
        days[:2], rms[:2], rms[0]) is None
    planted = rms.copy()
    planted[1] = np.e * rms[0]
    assert predictability.first_efold_crossing(days, planted, rms[0]) == 1.0


def test_first_efold_crossing_refuses_nonpositive_curve():
    with pytest.raises(ValueError, match="positive finite"):
        predictability.first_efold_crossing(
            np.array([0.0, 1.0]), np.array([1.0, 0.0]), 1.0)
