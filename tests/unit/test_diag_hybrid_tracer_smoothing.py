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
