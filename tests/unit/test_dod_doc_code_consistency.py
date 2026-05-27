"""Cross-doc / code consistency lock for the CRM Definition-of-done.

iter-108: ``CRM_implementation.md`` ``Definition of done`` section
quotes the DOD criterion 2 thresholds (CWV Wing 2018 plateau range,
MSE drift bound, plateau-window length) that ``scripts/summarize_rce_trajectory.py``
``evaluate_rce_quality`` actually gates against. If the two ever drift
apart — for example, the doc is updated but the code is not, or
vice-versa — production runs gate against one set of numbers while
the iteration log claims another. The iter-107 doc fix surfaced exactly
this kind of drift (pre-fix CWV plateau 30 ± 5 mm vs Wing 2018's 45-60
mm).

The tests below load both files and assert that the doc quotes match
the code constants. Failure means *one* of the two is stale and the
fix is to update them in lockstep.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_summarizer_module():
    """Load the summarizer module the same way the integration tests
    do — by path. Lives under ``scripts/`` which is not on sys.path."""
    path = REPO_ROOT / "scripts" / "summarize_rce_trajectory.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_rce_trajectory_for_consistency", path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["summarize_rce_trajectory_for_consistency"] = mod
    spec.loader.exec_module(mod)
    return mod


def _read_dod_section() -> str:
    """Return just the ``Definition of done`` section text (between
    the heading and the next ``## `` heading). Keeps the assertions
    below from accidentally matching unrelated paragraphs that
    happen to mention CWV or MSE in the iteration log."""
    text = (REPO_ROOT / "CRM_implementation.md").read_text()
    start = text.index("## Definition of done")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def test_dod_quotes_summarizer_default_cwv_upper_bound():
    """iter-107 set ``DEFAULT_CWV_RANGE_MM = (35, 65)`` (Wing 2018
    SST=300K multi-model band + 5 mm tolerance). The DOD section
    documents the (35, 65) range. If a future widening / tightening
    drifts only one of them, this test surfaces the drift."""
    mod = _load_summarizer_module()
    dod = _read_dod_section()
    low, high = mod.DEFAULT_CWV_RANGE_MM
    expected_low = int(low) if float(low).is_integer() else low
    expected_high = int(high) if float(high).is_integer() else high
    needle = f"DEFAULT_CWV_RANGE_MM = ({expected_low}, {expected_high})"
    assert needle in dod, (
        f"DOD section in CRM_implementation.md must quote the active "
        f"DEFAULT_CWV_RANGE_MM = ({expected_low}, {expected_high}). "
        f"Got DOD section that does not contain {needle!r}. Either the "
        f"summarizer constant was tightened/widened without updating "
        f"the doc, or the doc was edited without updating the code."
    )


def test_dod_quotes_summarizer_mse_drift_tolerance():
    """``DEFAULT_MSE_RELATIVE_DRIFT`` (currently 0.05) is the
    code-level gate. The DOD doc quotes ``5 %``. Lock the relationship
    so a future tighten to 0.01 must also update the doc to ``1 %``."""
    mod = _load_summarizer_module()
    dod = _read_dod_section()
    pct = mod.DEFAULT_MSE_RELATIVE_DRIFT * 100.0
    pct_str = f"{int(pct)} %" if pct.is_integer() else f"{pct:g} %"
    assert pct_str in dod, (
        f"DOD section in CRM_implementation.md must quote the active "
        f"MSE drift tolerance {pct_str} (from "
        f"DEFAULT_MSE_RELATIVE_DRIFT = {mod.DEFAULT_MSE_RELATIVE_DRIFT}). "
        f"DOD section does not contain {pct_str!r}."
    )


def test_dod_quotes_plateau_window_in_days():
    """``DEFAULT_LAST_N_DAYS_FOR_PLATEAU`` (currently 10) anchors the
    plateau check. The DOD doc must reference the same number of
    days; otherwise users will write criteria against a window that
    doesn't match what the gate actually enforces."""
    mod = _load_summarizer_module()
    dod = _read_dod_section()
    n_days = mod.DEFAULT_LAST_N_DAYS_FOR_PLATEAU
    # Doc may say "last 10 days" or "10-day plateau" — accept either.
    patterns = [
        rf"last\s+{n_days}\s+days",
        rf"{n_days}-day\s+plateau",
    ]
    assert any(re.search(p, dod) for p in patterns), (
        f"DOD section must reference the plateau window "
        f"DEFAULT_LAST_N_DAYS_FOR_PLATEAU = {n_days} days "
        f"(as 'last {n_days} days' or '{n_days}-day plateau'). "
        f"DOD section does not contain either phrase."
    )


def test_dod_mentions_wing_2018():
    """iter-107: DOD criterion 2 must cite Wing 2018 (the canonical
    RCEMIP1 reference for the CWV range). A future revert that drops
    the Wing reference and goes back to a hand-rolled estimate is
    the exact kind of stale-DOD bug iter-107 fixed."""
    dod = _read_dod_section()
    assert re.search(r"Wing\s*(et\s*al\.?\s*)?2018", dod), (
        "DOD criterion 2 must cite ``Wing 2018`` / ``Wing et al. "
        "2018`` for the RCEMIP1 plateau range. Pre-iter-107 the doc "
        "had a stale 30 +/- 5 mm hand-rolled estimate; this test "
        "blocks any future revert."
    )
