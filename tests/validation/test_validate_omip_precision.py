"""Smoke test for the mixed-precision validation harness.

Exercises scripts/validate/validate_omip_precision.py at tiny size: confirms
the fp64-vs-mixed closed-channel harness runs end-to-end and returns a
structured verdict with finite, conserving fp64 + mixed runs. The full
science verdict (PASS at production resolution) is the sbatch campaign; this
just proves the harness itself works. apply_precision is global, so the
module restores fp64 after.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pytest


@pytest.fixture(autouse=True)
def _restore_fp64():
    from legoesm.core.precision import (
        PrecisionPolicy, set_policy, clear_module_overrides)
    yield
    clear_module_overrides()
    set_policy(PrecisionPolicy.fp64())


def test_precision_validation_harness_runs():
    from scripts.validate.validate_omip_precision import run_precision_validation

    result = run_precision_validation(n_lat=4, n_lon=8, n_steps=2)
    assert set(result) == {"passed", "checks", "metrics"}
    assert isinstance(result["passed"], bool)
    # Both modes must integrate without blowing up at any size.
    assert result["checks"]["fp64_finite"] is True
    assert result["checks"]["mixed_finite"] is True
    # Engagement: the mixed policy must genuinely engage (fp32 storage + a
    # non-zero divergence from fp64) so the harness can never pass vacuously.
    assert result["checks"]["mixed_storage_is_fp32"] is True
    assert result["checks"]["fp64_storage_is_fp64"] is True
    assert result["checks"]["mixed_differs_from_fp64"] is True
    # The fp64 reference must conserve its own budgets on a closed basin.
    assert result["checks"]["fp64_heat_conserved"] is True
    assert result["checks"]["fp64_salt_conserved"] is True
    # #1675 review finding: this test used to assert only that ``passed`` was a
    # BOOLEAN, and never looked at the mixed-mode half of the verdict -- so a
    # run where mixed lost heat, lost salt and disagreed with fp64 on
    # temperature still went green. Assert the verdict and the mixed budgets.
    assert result["checks"]["mixed_heat_conserved"] is True, result["metrics"]
    assert result["checks"]["mixed_salt_conserved"] is True, result["metrics"]
    assert result["passed"] is True, (
        f"harness verdict FAILED: "
        f"{[k for k, v in result['checks'].items() if v is not True]}")
    # Metrics are populated (drift + cross-mode divergence numbers).
    for key in ("fp64_heat_drift_rel", "heat_cross_rel", "T_rms_diff_rel"):
        assert key in result["metrics"]
