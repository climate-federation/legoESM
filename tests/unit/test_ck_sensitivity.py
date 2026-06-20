"""Test the pure clause-6 feasibility metric in scripts/experiment/ck_sensitivity_vs_era5.

The model-run + real-ERA5 ``main`` is data-dependent (a model integration + a local ERA5
archive); this locks the ``bias_ck_sensitivity`` metric that turns two C_K runs' biases
into the go/no-go an operator reads.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_bias_ck_sensitivity_flags_insensitive_vs_sensitive():
    from scripts.experiment.ck_sensitivity_vs_era5 import bias_ck_sensitivity

    # Idealization-dominated (the REAL iter-412 result: 11.2 vs 11.2, Δ ≈ 0.001) — a
    # bias that barely moves with C_K ⇒ a correction loop can't lower it ⇒ NOT feasible.
    lo = np.full((4, 4), 11.2)
    hi = lo + 0.001
    s = bias_ck_sensitivity(lo, hi)
    assert not s["c_k_feasible"] and s["controllable_fraction"] < 0.01
    assert s["mean_bias"] == pytest.approx(11.2, abs=0.01)   # avg of the two runs
    assert s["mean_abs_delta"] == pytest.approx(0.001, rel=1e-4)

    # C_K-SENSITIVE: mean bias 4, C_K moves it by 2 (50%) ⇒ feasible.
    s2 = bias_ck_sensitivity(np.full((4, 4), 5.0), np.full((4, 4), 3.0))
    assert s2["controllable_fraction"] == pytest.approx(0.5) and s2["c_k_feasible"]

    # NaN cells (e.g. an SST-over-land env tag) are IGNORED, not propagated.
    lo3 = np.array([[5.0, np.nan], [5.0, 5.0]])
    hi3 = np.array([[3.0, np.nan], [3.0, 3.0]])
    assert np.isfinite(bias_ck_sensitivity(lo3, hi3)["controllable_fraction"])

    # Shape mismatch fails loud (a wrong pairing would silently mis-diff).
    with pytest.raises(ValueError, match="share shape"):
        bias_ck_sensitivity(np.zeros((2, 2)), np.zeros((3,)))
