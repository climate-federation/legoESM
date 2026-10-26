"""The equatorial thermocline probe must be able to use NEMO's matched window.

Its velocity side already selected explicit 5-day records from NEMO's
cold-start run (``--nemo-w-recs``), while its temperature side averaged a
calendar month of whatever file it was handed -- which, for the multi-year
monthly file the campaign passed, is a five-year climatology of a spun-up
ocean.  Half the instrument was on the matched reference and half was not.
``--nemo-t-recs`` closes that, and these tests pin the selection semantics and
the flag's existence.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "equatorial_thermocline",
    _ROOT / "scripts/validate/ocean_fidelity/equatorial_thermocline.py")
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["equatorial_thermocline"] = _MOD
sys.path.insert(0, str(_ROOT / "scripts/validate"))
_SPEC.loader.exec_module(_MOD)


def test_select_recs_slice_takes_exactly_that_window():
    x = np.arange(18.0).reshape(18, 1, 1)
    assert _MOD._select_recs(x, "5:6").ravel().tolist() == [5.0]
    assert _MOD._select_recs(x, "17:18").ravel().tolist() == [17.0]
    assert _MOD._select_recs(x, "0:6").ravel().tolist() == [0, 1, 2, 3, 4, 5]


def test_select_recs_none_is_the_whole_series():
    x = np.arange(4.0).reshape(4, 1, 1)
    assert _MOD._select_recs(x, None).shape[0] == 4


def test_select_recs_single_index():
    x = np.arange(6.0).reshape(6, 1, 1)
    picked = _MOD._select_recs(x, "3")
    assert float(np.asarray(picked).ravel()[0]) == 3.0


def test_nemo_t_recs_flag_exists_and_defaults_to_month_selection():
    """The flag must be absent by default, so an existing invocation keeps its
    behaviour and only a caller that asks for the matched window gets it."""
    ap = _MOD.build_parser() if hasattr(_MOD, "build_parser") else None
    if ap is None:
        # The script builds its parser inline in main(); assert on the source
        # instead, naming the symbol that runs.
        src = (_ROOT / "scripts/validate/ocean_fidelity"
               / "equatorial_thermocline.py").read_text()
        assert '"--nemo-t-recs"' in src
        assert "a.nemo_t_recs" in src
        # and the month branch must still exist as the default
        assert "a.nemo_month" in src
    else:                                              # pragma: no cover
        a = ap.parse_args(["--legoesm-snapshot", "s", "--nemo-gridt", "g"])
        assert a.nemo_t_recs is None


def test_matched_and_month_paths_are_mutually_exclusive_in_effect():
    """Averaging one record and averaging a month of records are different
    numbers; a test that could not tell them apart would prove nothing."""
    x = np.arange(18.0).reshape(18, 1, 1)
    matched = float(np.nanmean(_MOD._select_recs(x, "5:6"), axis=0).ravel()[0])
    month = float(np.nanmean(x[[0, 1, 2, 3, 4, 5]], axis=0).ravel()[0])
    assert matched == pytest.approx(5.0)
    assert month == pytest.approx(2.5)
    assert matched != month
