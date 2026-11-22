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


def test_paired_column_budget_detects_planted_leak_and_nonfinite():
    """Conservation gate has both a passing and a nonzero failing column."""
    spec = importlib.util.spec_from_file_location("_hybrid_budget_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mass = np.array([[2.0, 3.0]])
    q = np.array([[1.0, 2.0]])
    dm = np.array([[0.5, -0.5]])
    dq = np.array([[-0.25, 1.0 / 3.0]])
    assert mod.paired_column_budget(mass, q, dq, dm)["verdict"] == "CONSERVES"
    dq[0, 0] += 0.01
    assert mod.paired_column_budget(mass, q, dq, dm)["verdict"] == "LEAKS"
    assert mod.paired_column_budget(mass, q, dq * 0, dm * 0)["max_relative"] == 0
    dq[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        mod.paired_column_budget(mass, q, dq, dm)


@pytest.mark.parametrize("census", [False, True])
def test_main_dispatch_preserves_smoothing_and_selects_census(monkeypatch, tmp_path, census):
    """The new mode is explicit; existing invocations retain their route."""
    import sys
    spec = importlib.util.spec_from_file_location("_hybrid_main_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    monkeypatch.setattr(mod, "smoothing_main", lambda: calls.append("smoothing"))
    monkeypatch.setattr(mod, "all_vertical_paths", lambda out: calls.append(out))
    argv = [str(_SCRIPT), "--out", str(tmp_path)]
    if census:
        argv.append("--all-vertical-paths")
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    assert calls == [tmp_path if census else "smoothing"]


def test_live_ordered_callback_under_jitted_vmap():
    """Exercise the census spy mechanism; ordered callbacks must execute live."""
    import jax
    import jax.numpy as jnp
    received = []

    def observed(x):
        jax.debug.callback(lambda value: received.append(float(value)), x, ordered=True)
        return x * x

    actual = jax.jit(jax.vmap(observed))(jnp.array([1.0, 2.0]))
    jax.effects_barrier()
    np.testing.assert_array_equal(actual, [1.0, 4.0])
    assert received == [1.0, 2.0]
