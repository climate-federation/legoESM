"""Unit tests for the Stage-1 IC perturbation helper (scripts/tmp)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_probe_make_perturbed_ics",
    Path(__file__).resolve().parents[2]
    / "scripts" / "tmp" / "_probe_make_perturbed_ics.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


BASE = np.full((2, 4, 8), 250.0, dtype=np.float32)


def test_perturbation_is_additive_not_multiplicative():
    """The repo's in-process path is MULTIPLICATIVE; this must not be.

    An additive 0.01 K perturbation has the same std at 200 K and 300 K; a 1%
    multiplicative one would differ by 1 K between them.
    """
    cold = np.full((200000,), 200.0, dtype=np.float64)
    warm = np.full((200000,), 300.0, dtype=np.float64)
    dc = _mod.perturb_temperature(cold, 1) - cold
    dw = _mod.perturb_temperature(warm, 1) - warm
    assert dc.std() == pytest.approx(0.01, rel=0.05)
    assert dw.std() == pytest.approx(0.01, rel=0.05)
    # identical seed + additive => identical noise regardless of base value
    assert np.allclose(dc, dw, atol=1e-12)


def test_perturbation_magnitude_matches_sigma():
    out = _mod.perturb_temperature(np.full((500000,), 250.0), 3, sigma=0.01)
    d = out - 250.0
    assert d.std() == pytest.approx(0.01, rel=0.05)
    assert abs(d.mean()) < 5e-4          # zero-mean
    assert np.abs(d).max() < 0.1          # ~10 sigma bound


def test_perturbation_is_deterministic_per_seed():
    a = _mod.perturb_temperature(BASE, 7)
    b = _mod.perturb_temperature(BASE, 7)
    assert np.array_equal(a, b)


def test_different_seeds_are_independent():
    n = 200000
    base = np.full((n,), 250.0, dtype=np.float64)
    d1 = _mod.perturb_temperature(base, 1) - base
    d2 = _mod.perturb_temperature(base, 2) - base
    assert not np.array_equal(d1, d2)
    assert abs(float(np.corrcoef(d1, d2)[0, 1])) < 0.01


def test_dtype_is_preserved():
    assert _mod.perturb_temperature(BASE, 1).dtype == np.float32


def test_perturbation_actually_changes_values():
    """0.01 K must be representable in float32 at ~250 K (it is, ~650 ULP)."""
    out = _mod.perturb_temperature(BASE, 1)
    assert not np.array_equal(out, BASE)
    assert (out != BASE).mean() > 0.99


def test_non_finite_input_is_fatal():
    bad = BASE.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(SystemExit, match="non-finite"):
        _mod.perturb_temperature(bad, 1)


def test_non_positive_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        _mod.perturb_temperature(BASE, 1, sigma=0.0)


def test_duplicate_seeds_rejected(tmp_path):
    """Reused seeds would silently collapse the ensemble to fewer members."""
    with pytest.raises(SystemExit, match="distinct"):
        _mod.main([str(tmp_path / "src.zarr"), str(tmp_path / "out"),
                   "--seeds", "1,1,2"])


def test_module_documents_the_multiplicative_correction():
    """The 0.01-vs-1% discrepancy must stay recorded where the choice is made."""
    src = (Path(__file__).resolve().parents[2] / "scripts" / "tmp"
           / "_probe_make_perturbed_ics.py").read_text()
    assert "multiplicative=True" in src and "250x" in src
