"""Direct tests for the hybrid tracer smoothing diagnostic's masked reduction."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_hybrid_tracer_smoothing.py")


@pytest.mark.parametrize("affinity_available", [True, False])
def test_countmin_import_and_masked_column(monkeypatch, affinity_available):
    """Import is inert with or without affinity APIs; count/minimum are exact."""
    before = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None

    def unexpected_affinity(*args):
        raise AssertionError("import must not change CPU affinity")

    if affinity_available:
        monkeypatch.setattr(os, "sched_setaffinity", unexpected_affinity, raising=False)
    else:
        monkeypatch.delattr(os, "sched_setaffinity", raising=False)
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    spec = importlib.util.spec_from_file_location("_hybrid_smoothing_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if affinity_available and before is not None:
        assert os.sched_getaffinity(0) == before

    column = np.array([[-4.0, -2.5, 0.0, 3.75]])
    region = np.array([[False, True, True, True]])
    count, minimum = mod.countmin(column, (column < 0) & region)
    assert count == 1 and type(count) is int
    assert minimum == -2.5 and type(minimum) is float
    assert mod.countmin(column, np.zeros_like(region)) == (0, None)


@pytest.fixture
def inventory_probe():
    spec = importlib.util.spec_from_file_location("_hybrid_inventory_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def inventory_arrays():
    return (np.array([2.0, 3.0, 5.0]), np.array([1.0, 2.0, 4.0]),
            np.array([[.001, .004], [.002, .002], [.004, .001]]))


def test_inventory_controls_and_unequal_weights(inventory_probe, inventory_arrays):
    """Internal transfers close; the planted missing weight must be detected."""
    volume, rho, q = inventory_arrays
    controls = inventory_probe.inventory_controls(volume, rho, q)
    assert [r["verdicts"] for r in controls] == [
        ["CONSERVES", "CONSERVES"], ["CONSERVES", "CONSERVES"], ["LEAKS", "LEAKS"]]
    scale = np.sum(np.abs(volume[:, None] * rho[:, None] * q), axis=0)
    np.testing.assert_array_equal(controls[0]["initial_abs_inventory"], scale)
    assert controls[1]["donor_indices"] == [2, 2]
    assert controls[1]["receiver_indices"] == [1, 1]
    # Deliberately WRONG: equal delta_q transfer ignores the unequal cell mass.
    delta_q = .01 * q[2]
    wrong = q.copy()
    wrong[2] -= delta_q
    wrong[1] += delta_q
    result = inventory_probe.global_inventory_budget(volume, rho, q, rho, wrong, np.zeros(2))
    expected = delta_q * ((volume * rho)[1] - (volume * rho)[2])
    assert np.all(np.abs(np.asarray(result["signed_residual"]) - expected) / scale <= 1e-12)
    assert result["verdicts"] == ["LEAKS", "LEAKS"]


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_inventory_signed_source_and_boundary(inventory_probe, inventory_arrays, sign):
    volume, rho, q = inventory_arrays
    amount = .01 * (volume * rho)[2] * q[2]
    changed = q.copy()
    changed[1] += sign * amount / (volume * rho)[1]
    result = inventory_probe.global_inventory_budget(volume, rho, q, rho, changed, np.zeros(2))
    assert np.all(sign * np.asarray(result["signed_residual"]) > 0)
    assert np.all(np.abs(np.asarray(result["signed_residual"]) - sign * amount)
                  / result["initial_abs_inventory"] <= 1e-12)
    assert result["verdicts"] == ["LEAKS", "LEAKS"]
    closed = inventory_probe.global_inventory_budget(volume, rho, q, rho, changed, sign * amount)
    assert closed["verdicts"] == ["CONSERVES", "CONSERVES"]


@pytest.mark.parametrize("bad", ["fp32", "shape", "boundary_shape", "nan", "zero_inventory",
                                  "volume", "rho_old", "rho_new", "empty"])
def test_inventory_rejects_invalid_inputs(inventory_probe, inventory_arrays, bad):
    volume, rho, q = inventory_arrays
    args = [volume.copy(), rho.copy(), q.copy(), rho.copy(), q.copy(), np.zeros(2)]
    if bad == "fp32":
        args[2] = args[2].astype(np.float32)
    elif bad == "shape":
        args[4] = args[4][:-1]
    elif bad == "boundary_shape":
        args[5] = np.zeros(1)
    elif bad == "nan":
        args[4][0, 0] = np.nan
    elif bad == "zero_inventory":
        args[2][:, 0] = 0.0
    elif bad == "empty":
        args = [a[:0] for a in args]
    else:
        args[{"volume": 0, "rho_old": 1, "rho_new": 3}[bad]][0] = 0.0
    with pytest.raises(ValueError):
        inventory_probe.global_inventory_budget(*args)


@pytest.mark.parametrize("violation", [None, "missing", "path", "scheme", "statistic",
                                      "scope", "leak", "empty_verdicts", "empty_paths"])
def test_full_step_inventory_acceptance_is_nonvacuous(inventory_probe, violation):
    row = dict(path="fixed", scheme="full split step", statistic="global inventory budget",
               scope="full domain", verdicts=["CONSERVES", "CONSERVES"])
    paths = ("fixed",)
    rows = [row]
    if violation in ("path", "scheme", "statistic", "scope"):
        row[violation] = "wrong"
    elif violation == "missing":
        rows = []
    elif violation == "leak":
        row["verdicts"][1] = "LEAKS"
    elif violation == "empty_verdicts":
        row["verdicts"] = []
    elif violation == "empty_paths":
        paths = ()
    if violation is None:
        inventory_probe.assert_height_inventory_conserves(rows, paths)
    else:
        with pytest.raises(AssertionError):
            inventory_probe.assert_height_inventory_conserves(rows, paths)
