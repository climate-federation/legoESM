"""Direct tests for ``scripts/validate/diag_scheme_blowup.py``.

The probe exists to say WHICH prognostic went non-finite, at WHICH step and
WHICH level. Its two pure helpers are what carry that claim, so they are tested
against arrays with a known answer rather than trusted from one run -- an
instrument whose first output is quoted as a finding is exactly the failure the
"validate the instrument" rule is about.

Dispatch is tested too: an unknown case or scheme must be a hard error, not a
silently different column.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_scheme_blowup.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_diag_scheme_blowup",
                                                  _SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["_diag_scheme_blowup"] = m
    spec.loader.exec_module(m)
    return m


def test_first_bad_returns_none_on_a_clean_state(mod):
    fields = {"T": np.linspace(280.0, 300.0, 5), "qv": np.zeros(5)}
    assert mod._first_bad(fields) is None


def test_first_bad_names_the_field_index_and_count(mod):
    fields = {
        "T": np.array([300.0, 301.0, 302.0]),
        "qv": np.array([0.0, np.nan, np.inf]),
    }
    name, idx, count = mod._first_bad(fields)
    assert name == "qv"
    assert idx == 1, "must report the FIRST bad entry, not the last"
    assert count == 2, "inf counts as non-finite alongside nan"


def test_first_bad_sees_an_infinity_with_no_nan_present(mod):
    """A blown-up column often overflows to +/-inf long before it makes a NaN;
    a check written as `isnan` alone would have called this state healthy."""
    fields = {"T": np.array([300.0, -np.inf])}
    assert mod._first_bad(fields) == ("T", 1, 1)


def _fake_state():
    import types

    def _f(v):
        return types.SimpleNamespace(data=np.full((1, 1, 1, 3), v))

    return types.SimpleNamespace(
        T=_f(300.0), u=_f(1.0), v=_f(0.0),
        tracers={"q_v": _f(0.01)},
    )


def test_profiles_watches_the_prognostics_and_skips_absent_carries(mod):
    """A diagnostic closure carries no turbulent energy; the probe must watch
    four fields then, not raise and not invent a zero one."""
    import types
    phys_bare = types.SimpleNamespace()
    assert set(mod._profiles(_fake_state(), phys_bare)) == {"T", "qv", "u", "v"}


def test_profiles_watches_the_qke_slot_not_only_tke(mod):
    """mynn25's carry is `qke`, a DIFFERENT PhysicsState slot from `tke`.

    Watching `tke` alone printed a flat zero for every mynn25 run and invited
    the conclusion that its prognostic never advances -- the probe was reading
    a slot that scheme does not use. Regression on the instrument, not the
    model.
    """
    import types
    phys = types.SimpleNamespace(
        tke=np.zeros((1, 1, 1, 3)),
        qke=np.full((1, 1, 1, 3), 0.8),
        clubb_moments=np.zeros((1, 15, 4)),
    )
    got = mod._profiles(_fake_state(), phys)
    assert set(got) == {"T", "qv", "u", "v", "tke", "qke", "clubb_mom"}
    assert got["qke"].shape == (3,), "must be flattened for the reduction"
    assert float(got["qke"].max()) == 0.8


def test_unknown_case_is_a_hard_error(mod):
    with pytest.raises(SystemExit) as exc:
        mod.main(["--case", "not_a_case", "--scheme", "louis"])
    assert "not_a_case" in str(exc.value)


def test_unknown_scheme_is_a_hard_error(mod):
    with pytest.raises(SystemExit) as exc:
        mod.main(["--case", "cbl", "--scheme", "not_a_scheme"])
    assert "not_a_scheme" in str(exc.value)
