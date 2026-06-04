"""Cross-doc / code consistency lock for the CRM Definition-of-done.

iter-108: ``CRM_implementation.md`` ``Definition of done`` section
quotes the DOD criterion 2 thresholds (CWV Wing 2018 plateau range,
MSE drift bound, plateau-window length) that ``scripts/validate/summarize_rce_trajectory.py``
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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_summarizer_module():
    """Load the summarizer module the same way the integration tests
    do — by path. Lives under ``scripts/`` which is not on sys.path."""
    path = REPO_ROOT / "scripts" / "validate" / "summarize_rce_trajectory.py"
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
    text = (REPO_ROOT / "docs" / "specs" / "CRM_implementation.md").read_text()
    start = text.index("## Definition of done")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _read_criterion_2_text() -> str:
    """Extract DOD *criterion 2 only* (between the ``2. **Reach
    radiative-convective equilibrium**:`` heading and the next
    numbered criterion). iter-109 Codex MEDIUM#2 fix: the previous
    whole-section match let the Wing 2018 citation in the DOD
    preamble satisfy the criterion-2-Wing-citation lock, hollowing
    out the test. Scope the match to the criterion's own text."""
    dod = _read_dod_section()
    crit2_start = dod.index("2. **Reach radiative-convective equilibrium**")
    # Look ahead for the next ``\nN. **`` numbered criterion.
    m = re.search(r"\n[3-9]\.\s+\*\*", dod[crit2_start:])
    assert m is not None, (
        "Could not find criterion 3+ after criterion 2 in DOD."
    )
    crit2_end = crit2_start + m.start()
    return dod[crit2_start:crit2_end]


def test_dod_quotes_summarizer_default_cwv_upper_bound():
    """iter-107 set ``DEFAULT_CWV_RANGE_MM = (35, 65)`` (Wing 2018
    SST=300K multi-model band + asymmetric tolerance per iter-109).
    The DOD section documents the (35, 65) range. iter-109 Codex
    LOW#4 fix: parse the tuple via regex so trivial formatting
    differences (`(35.0, 65.0)`, `(35, 65,)`, whitespace) don't
    break the test on innocent reformatting."""
    mod = _load_summarizer_module()
    dod = _read_dod_section()
    low, high = mod.DEFAULT_CWV_RANGE_MM
    # Match ``DEFAULT_CWV_RANGE_MM = (NUM, NUM)`` with flexible
    # whitespace + optional trailing comma + integer-or-float NUM.
    pattern = (
        r"DEFAULT_CWV_RANGE_MM\s*=\s*\(\s*"
        r"(?P<low>-?\d+(?:\.\d+)?)\s*,\s*"
        r"(?P<high>-?\d+(?:\.\d+)?)\s*,?\s*\)"
    )
    m = re.search(pattern, dod)
    assert m is not None, (
        f"DOD section in CRM_implementation.md must quote the active "
        f"DEFAULT_CWV_RANGE_MM = ({low}, {high}). No matching "
        f"``DEFAULT_CWV_RANGE_MM = (..., ...)`` clause found in DOD."
    )
    doc_low = float(m.group("low"))
    doc_high = float(m.group("high"))
    assert doc_low == low and doc_high == high, (
        f"DOD doc says DEFAULT_CWV_RANGE_MM = ({doc_low}, {doc_high}) "
        f"but the active code says ({low}, {high}). One of them is "
        f"stale; update both in lockstep."
    )


def test_dod_quotes_summarizer_mse_drift_tolerance():
    """``DEFAULT_MSE_RELATIVE_DRIFT`` (currently 0.05) is the
    code-level gate. The DOD doc quotes ``5 %`` (or equivalent
    formatting). iter-109 Codex LOW#5 fix: regex tolerates ``5 %``,
    ``5%``, ``5.0 %``, ``5.0%``."""
    mod = _load_summarizer_module()
    dod = _read_dod_section()
    pct = mod.DEFAULT_MSE_RELATIVE_DRIFT * 100.0
    # Build a tight regex around the active value. Examples for
    # pct=5.0: matches `5 %`, `5%`, `5.0 %`, `5.0%`, but NOT `15 %`
    # or `5.5 %` (those would be different drift tolerances).
    if pct.is_integer():
        # e.g. `5` or `5.0` followed by optional space + ``%``.
        pat = rf"\b{int(pct)}(?:\.0+)?\s*%"
    else:
        # Non-integer like 1.5 % — match exact decimal.
        pat = rf"\b{re.escape(f'{pct:g}')}\s*%"
    assert re.search(pat, dod), (
        f"DOD section in CRM_implementation.md must quote the active "
        f"MSE drift tolerance ({pct:g} %, from "
        f"DEFAULT_MSE_RELATIVE_DRIFT = {mod.DEFAULT_MSE_RELATIVE_DRIFT}). "
        f"DOD section does not match pattern {pat!r}."
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


def _scrape_driver_argparse_default(flag: str) -> str | int | float:
    """Scrape ``--<flag>``'s argparse default literal from
    run_rce_mpi_long.py. Helper for the parametrized
    preamble-claim consistency tests below."""
    driver_text = (REPO_ROOT / "scripts" / "run_rce_mpi_long.py").read_text()
    pat = (
        rf'p\.add_argument\("--{flag}",\s*type=(int|float),'
        rf'\s*default=([0-9.eE_+-]+)'
    )
    m = re.search(pat, driver_text)
    assert m, f"driver missing --{flag} argparse default (regex {pat!r})"
    return float(m.group(2)) if m.group(1) == "float" else int(m.group(2))


@pytest.mark.parametrize("claim", [
    # (claim_pattern_template, driver_flag, transform_fn, label)
    # template uses {v} which is filled with the active driver default.
    pytest.param(
        ("dx", lambda v: int(v / 1000.0), r"\bdx\s*=\s*{v}\s*km\b"),
        id="dx_km",
    ),
    pytest.param(
        ("nlev", lambda v: v, r"\bnlev\s*=\s*{v}\b"),
        id="nlev",
    ),
    pytest.param(
        ("H", lambda v: int(v / 1000.0), r"\bH\s*=\s*{v}\s*km\b"),
        id="H_km",
    ),
    pytest.param(
        ("dt", lambda v: int(v) if float(v).is_integer() else v,
         r"\bdt\s*=\s*{v}\s*s\b"),
        id="dt_s",
    ),
    pytest.param(
        ("n-acoustic-substeps", lambda v: v,
         r"\bN_ACOUSTIC\s*=\s*{v}\b"),
        id="n_acoustic",
    ),
])
def test_dod_preamble_quotes_driver_default(claim):
    """iter-175: the DOD production-config preamble (line 31 of
    CRM_implementation.md) packs 5 active-default claims in one
    line: dx=2 km, nlev=30, H=33 km, dt=5 s, N_ACOUSTIC=12. Each
    must match the corresponding driver argparse default.

    iter-176 already locked the 132x132 grid size + iter-175 the
    RANKS=12 wrapper value; this parametrized test closes the
    remaining 5 preamble claims so the entire ``30-day CRM run on
    production target`` line stays in lockstep with the driver.

    A regression in any one default that doesn't update the doc
    will surface here as a per-parameter test failure.
    """
    flag, transform, pat_tmpl = claim
    raw = _scrape_driver_argparse_default(flag)
    value = transform(raw)
    pat = pat_tmpl.replace("{v}", str(value))
    dod = _read_dod_section()
    assert re.search(pat, dod), (
        f"DOD preamble must reference the active driver default "
        f"--{flag}={raw!r} (transformed: {value}). Pattern "
        f"{pat!r} not found in DOD section."
    )


def test_dod_quotes_driver_production_grid_size():
    """iter-174: DOD section's production-config preamble at
    CRM_implementation.md:31 says ``132×132 plane``; the driver
    script defaults ``--nx 132 --ny 132``. Lock the doc <->
    driver agreement here.

    The test scrapes the active ``--nx`` and ``--ny`` defaults
    from the run_rce_mpi_long.py argparse source. Pre-iter-174
    nothing enforced lockstep — a driver revert to 64x64 would
    leave the doc claim stale until a human noticed.
    """
    dod = _read_dod_section()
    driver_text = (REPO_ROOT / "scripts" / "run_rce_mpi_long.py").read_text()
    # Scrape ``--nx`` and ``--ny`` defaults from argparse calls.
    nx_m = re.search(
        r'p\.add_argument\("--nx",\s*type=int,\s*default=(\d+)',
        driver_text,
    )
    ny_m = re.search(
        r'p\.add_argument\("--ny",\s*type=int,\s*default=(\d+)',
        driver_text,
    )
    assert nx_m and ny_m, "driver missing --nx / --ny argparse defaults"
    nx = int(nx_m.group(1))
    ny = int(ny_m.group(1))
    # ``×`` is the unicode multiplication sign used in the doc
    # (``132×132``). Accept either that or ASCII ``x``.
    pat = rf"\b{nx}\s*[×x]\s*{ny}\s+plane\b"
    assert re.search(pat, dod), (
        f"DOD section must reference the active driver grid size "
        f"({nx}x{ny}) as ``{nx}×{ny} plane`` or ``{nx}x{ny} "
        f"plane``. DOD section does not match pattern {pat!r}."
    )


def test_dod_quotes_wrapper_default_ranks():
    """iter-173: DOD section's production-config preamble at
    CRM_implementation.md:31 says ``12 MPI ranks``; the
    run_rce_30day.sh wrapper defaults RANKS=12. They drift
    independently and pre-iter-173 nothing enforced lockstep.

    A wrapper revert to RANKS=4 (or any other value) would
    leave the doc claim stale. This test scrapes the active
    RANKS default from the wrapper's ``RANKS="${RANKS:-N}"``
    line, then asserts criterion 2's text references the same N.
    """
    dod = _read_dod_section()
    wrapper_text = (REPO_ROOT / "scripts" / "run_rce_30day.sh").read_text()
    m = re.search(r'^RANKS="\$\{RANKS:-(\d+)\}"', wrapper_text, re.MULTILINE)
    assert m, "wrapper missing RANKS env default"
    ranks = int(m.group(1))
    pat = rf"\b{ranks}\s+MPI\s+ranks?\b"
    assert re.search(pat, dod), (
        f"DOD section must reference the active wrapper RANKS "
        f"default ({ranks}). DOD section does not match "
        f"pattern {pat!r}."
    )


def test_dod_quotes_final_dod_min_days():
    """iter-171: ``DOD_FINAL_MIN_DAYS = 30`` is the minimum
    trajectory length below which evaluate_rce_final_dod returns
    INSUFFICIENT. The DOD section must reference ``30-day`` (or
    ``30 day``) somewhere in criterion 2's text so the
    production-window length is unambiguous.

    Pre iter-171 only DOD_FINAL_MSE_DRIFT (1 %) was locked; the
    30-day MIN_DAYS could drift silently (e.g. to 14 / 60) without
    a doc update.
    """
    mod = _load_summarizer_module()
    crit2 = _read_criterion_2_text()
    n_days = mod.DOD_FINAL_MIN_DAYS
    # ``30-day`` or ``30 day`` (singular / plural) both accepted.
    pat = rf"\b{n_days}[-\s]days?\b"
    assert re.search(pat, crit2), (
        f"DOD criterion 2 must reference the active minimum "
        f"trajectory length DOD_FINAL_MIN_DAYS = {n_days} days "
        f"(as '{n_days}-day' or '{n_days} day(s)'). criterion 2 "
        f"does not match pattern {pat!r}."
    )


def test_dod_quotes_final_dod_mse_drift():
    """iter-158: ``DOD_FINAL_MSE_DRIFT = 0.01`` is the production
    final-DOD gate (criterion 2's tighter end-of-run check). DOD
    section quotes ``< 1 %`` — must stay in sync with the code
    constant. Pre-iter-158 only DEFAULT_MSE_RELATIVE_DRIFT (5 %)
    was locked; the production-grade 1 % gate could drift silently.

    iter-109's MSE regex is the model — same int-vs-float tolerance.
    """
    mod = _load_summarizer_module()
    dod = _read_dod_section()
    pct = mod.DOD_FINAL_MSE_DRIFT * 100.0
    if pct.is_integer():
        # ``1`` or ``1.0`` followed by optional whitespace + ``%``.
        # Word boundary + lookahead reject ``10`` / ``15`` /``11``.
        pat = rf"\b{int(pct)}(?:\.0+)?\s*%(?!\d)"
    else:
        pat = rf"\b{re.escape(f'{pct:g}')}\s*%"
    assert re.search(pat, dod), (
        f"DOD section in CRM_implementation.md must quote the active "
        f"final-DOD MSE drift tolerance ({pct:g} %, from "
        f"DOD_FINAL_MSE_DRIFT = {mod.DOD_FINAL_MSE_DRIFT}). "
        f"DOD section does not match pattern {pat!r}."
    )


def test_dod_quotes_summarizer_max_w_threshold():
    """iter-157: DOD criterion 1 says ``max|w| < 50 m/s``; the code
    constant is ``DEFAULT_MAX_W_THRESHOLD_MS = 50.0``. They must stay
    in lockstep — pre-iter-157 this consistency was unverified, so
    a code-side change to 20 or 100 m/s would silently disagree with
    the doc.

    Tolerates ``50``, ``50.0``, ``50 m/s``, ``50.0 m/s`` formatting.
    """
    mod = _load_summarizer_module()
    dod = _read_dod_section()
    threshold = mod.DEFAULT_MAX_W_THRESHOLD_MS
    if float(threshold).is_integer():
        # Match e.g. ``50`` or ``50.0`` followed by optional space + ``m/s``,
        # but reject the leading-digit subset that would match 500 / 5000.
        pat = rf"max\|w\|\s*<\s*{int(threshold)}(?:\.0+)?\s*m/s\b"
    else:
        pat = rf"max\|w\|\s*<\s*{re.escape(f'{threshold:g}')}\s*m/s\b"
    assert re.search(pat, dod), (
        f"DOD section in CRM_implementation.md must quote the active "
        f"max|w| threshold ({threshold:g} m/s, from "
        f"DEFAULT_MAX_W_THRESHOLD_MS = {mod.DEFAULT_MAX_W_THRESHOLD_MS}). "
        f"DOD section does not match pattern {pat!r}."
    )


def test_criterion_2_cites_wing_2018():
    """iter-107: DOD criterion 2 must cite Wing 2018 (the canonical
    RCEMIP1 reference for the CWV range). A future revert that drops
    the Wing reference and goes back to a hand-rolled estimate is
    the exact kind of stale-DOD bug iter-107 fixed.

    iter-109 Codex MEDIUM#2 fix: extract criterion 2 specifically
    so the citation in the DOD preamble cannot satisfy this test —
    criterion 2's own text must carry the Wing reference.

    iter-109 (Codex Q5 follow-up): the surname-only regex
    ``Wing\\s*...2018`` would also match ``Wing-Tatang 2018``. Tighten
    to require either ``Wing 2018`` exactly OR
    ``Wing et al. 2018`` — both with a word boundary AFTER ``Wing``
    so a hyphenated compound surname doesn't satisfy the test."""
    crit2 = _read_criterion_2_text()
    # Word boundary after Wing rejects "Wing-Tatang", "Wingerd", etc.
    # Allow whitespace + "et al." optionally, then a 4-digit year.
    assert re.search(
        r"\bWing\b(?:\s+et\s+al\.?)?\s+2018", crit2,
    ), (
        "DOD criterion 2 (text between ``2. **Reach radiative-"
        "convective equilibrium**`` and the next numbered criterion) "
        "must cite ``Wing 2018`` or ``Wing et al. 2018`` for the "
        "RCEMIP1 plateau range. iter-107 fixed the stale 30 +/- 5 "
        "mm estimate; this test blocks any future revert AND blocks "
        "a same-name false positive (``Wing-Tatang 2018`` is "
        "rejected by the word boundary)."
    )


def test_criterion_2_quotes_wing_doi():
    """iter-109 Codex LOW#6 fix: the DOD criterion 2 citation must
    include the Wing 2018 DOI so auditors can verify the cited
    PWV-vs-SST envelope without leaving the repo."""
    crit2 = _read_criterion_2_text()
    assert "10.5194/gmd-11-793-2018" in crit2, (
        "DOD criterion 2 must include the Wing 2018 DOI "
        "``10.5194/gmd-11-793-2018`` so the CWV plateau range "
        "citation is verifiable from the repo alone (iter-109 "
        "Codex LOW#6)."
    )
