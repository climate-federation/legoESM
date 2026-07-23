"""Unit test for the S5 coupled-scale validator's O(1) verdict (pure, fast).

Exercises ``assess_o1`` — the deterministic HLO-size verdict that decides whether
the coupled CLM-ML land step compiles sub-linearly in ncol.  No driver / JAX build
here (that is the slow manual run of the script itself); this locks the pass/fail
LOGIC so a genuine O(ncol) regression (the pre-S3 unrolled loop) would fail the gate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "validate_clm_ml_coupled_scale.py")


def _load():
    spec = importlib.util.spec_from_file_location("_s5_scale", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_assess_o1_passes_sublinear_fails_linear():
    m = _load()
    # Sub-linear (the measured S3-scan case): ncol 24->54 (2.25x), HLO ~constant.
    sublinear = [{"ncol": 24, "hlo_chars": 500_000},
                 {"ncol": 54, "hlo_chars": 560_000}]  # 1.12x << 2.25x
    is_o1, d = m.assess_o1(sublinear)
    assert is_o1, f"sub-linear HLO growth must read as O(1): {d}"

    # Linear (the pre-S3 O(ncol) Python-loop unroll): HLO grows ~ with ncol.
    linear = [{"ncol": 24, "hlo_chars": 500_000},
              {"ncol": 54, "hlo_chars": 1_125_000}]   # 2.25x ~ ncol ratio
    is_o1, d = m.assess_o1(linear)
    assert not is_o1, f"linear HLO growth must NOT read as O(1): {d}"


def test_assess_o1_input_guards():
    m = _load()
    with pytest.raises(ValueError):
        m.assess_o1([{"ncol": 24, "hlo_chars": 1}])          # <2 rows
    with pytest.raises(ValueError):
        m.assess_o1([{"ncol": 54, "hlo_chars": 1},           # not increasing
                     {"ncol": 24, "hlo_chars": 1}])


def test_sublinear_frac_is_a_real_discriminator():
    """Guard the threshold: it must sit strictly between 'constant' and 'linear'."""
    m = _load()
    assert 0.0 < m.SUBLINEAR_FRAC < 1.0, (
        "SUBLINEAR_FRAC must be in (0,1): 0 rejects everything, >=1 would pass a "
        "linear O(ncol) unroll")
