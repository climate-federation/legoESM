"""#1455: every box-heat-budget driver must pass the fp64 gate.

The `k33` and `vertmix` buckets are realized increments `(T_after-T_before)/dt`
— a catastrophic cancellation of two O(10) degC temperatures. Under the DEFAULT
fp32 storage policy any increment below ~ULP(T)/dt (~2e-9 degC/s) reports as
exactly 0.0. `box_heat_budget.py` measured that failure directly: peak
|dT_k33| = 2.1193e-09 = 1.00 ULP, 95% of cells at zero, bit-identical across two
eos_depth conventions whose K33 fields differ by rel-L2 2e-3.

So an UNGATED driver reporting "k33 is identically 0.0" cannot distinguish
instrument from physics — which is exactly the open question on #1455. This
test makes the gate structural: a new budget driver cannot be added without it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DINO = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
        / "ocean_fidelity" / "dino_1226")
# Drivers that construct the accumulator, i.e. that READ the fragile buckets.
_MARKER = "BoxHeatBudgetAccumulator"


def _budget_drivers():
    if not DINO.is_dir():
        pytest.skip("dino_1226 driver directory not present")
    return sorted(p for p in DINO.glob("*.py")
                  if _MARKER in p.read_text())


def test_at_least_one_driver_is_scanned():
    """Guards the parametrization from silently covering nothing."""
    assert _budget_drivers(), (
        f"no driver in {DINO} constructs {_MARKER} — the scan below would be "
        f"vacuous")


@pytest.mark.parametrize("path", _budget_drivers(), ids=lambda p: p.name)
def test_budget_driver_calls_require_fp64(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    called = {
        n.func.id if isinstance(n.func, ast.Name) else
        getattr(n.func, "attr", None)
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "require_fp64" in called, (
        f"{path.name} builds a {_MARKER} without calling require_fp64. Its "
        f"k33/vertmix readings would be indistinguishable from fp32 "
        f"quantization noise (#1455). JAX_ENABLE_X64=1 is NOT sufficient — "
        f"legoesm.core.precision is an independent axis.")
