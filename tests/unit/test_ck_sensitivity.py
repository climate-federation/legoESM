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


def test_per_level_ck_bias_sensitivity_localizes_to_the_boundary_layer():
    """The per-level breakdown (iter 416) localizes C_K control: a free-tropospheric bias
    that C_K does NOT move shows a ~0 controllable fraction, while a BL level C_K DOES
    move shows a large one — vindicating that the LES→C_K correction is effective WHERE it
    acts even when the full-column bias is free-trop (radiation) dominated."""
    from scripts.experiment.ck_sensitivity_vs_era5 import per_level_ck_bias_sensitivity

    ncol, nlev = 4, 3
    ref = np.full((ncol, nlev), 250.0)
    # Free-trop (levels 0,1): a big bias UNCHANGED by C_K (both runs +10 K).
    # BL (level 2, near-surface): C_K MOVES it (+2 K at low C_K → +0.5 K at high C_K).
    t_lo = ref.copy()
    t_lo[:, :2] += 10.0
    t_lo[:, 2] += 2.0
    t_hi = ref.copy()
    t_hi[:, :2] += 10.0
    t_hi[:, 2] += 0.5
    out = per_level_ck_bias_sensitivity(t_lo, t_hi, ref, np.ones(ncol))
    frac = out["controllable_fraction_per_level"]
    assert frac[0] < 1e-6 and frac[1] < 1e-6           # free-trop: C_K-INDEPENDENT
    assert frac[2] > 0.5                                # BL: C_K-CONTROLLABLE
    assert out["bias_per_level"][0] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="share shape"):
        per_level_ck_bias_sensitivity(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((3, 3)), np.ones(2))
