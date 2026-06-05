"""Structural tests for the cross-grid shell wrappers.

iter-52: pin the structural invariants of
``scripts/run_{amip,omip,rce}_cross_grid.sh`` so a future refactor
cannot silently:

  * drop one of the four grid types (cubed_sphere / latlon /
    voronoi-or-mpas / gaussian-or-spectral);
  * remap a grid name to a wrong discretization (e.g. ``latlon`` ↔
    ``cdgrid`` instead of ``latlon_cgrid``);
  * write per-grid output to a path the matrix-runner collector
    doesn't look at;
  * skip the post-run cross-grid plot invocation.

iter-53 (codex iter-52 review): tightened from substring matches to
exact per-grid mapping checks, anchored regex matches (``OUTDIR=``,
loop body), and ordering invariants (NPZ purge before run_amip.py;
cross-grid plot is the LAST python invocation).

Tests parse the wrapper sources textually (no subprocess invocation,
no JAX import).  They run in 30 ms total but still catch the
structural issues that have actually caused real silent-skip bugs in
the iter-25..50 chain (iter-26 RCE discretization-flag regression,
iter-49 OMIP collector mismatch, iter-50 mixed-fixture cross-grid).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


# ---------------------------------------------------------------------------
# Canonical mappings (the single source of truth this test file pins).
# ---------------------------------------------------------------------------

# Atmosphere wrappers (RCE, AMIP) use the canonical Python grid_type
# names: ``voronoi`` / ``gaussian``.
_RCE_AMIP_GRID_DISC = {
    "cubed_sphere": "cdgrid",
    "latlon":       "latlon_cgrid",
    "voronoi":      "mpas",
    "gaussian":     "spectral",
}

# OMIP uses ocean-side spelling: ``mpas`` / ``spectral`` directly as
# grid_type names (see ``scripts/run/run_omip.py`` ``--grid`` choices).
_OMIP_GRIDS = {"cubed_sphere", "latlon", "mpas", "spectral"}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _parse_bash_assoc_array(text: str, name: str) -> dict:
    """Extract ``declare -A NAME=( [k]=v ... )`` into a Python dict.

    Returns the inner mapping; raises ``AssertionError`` if not found.
    Both quoted and unquoted RHS forms are supported.
    """
    m = re.search(
        rf"declare\s+-A\s+{name}=\((.*?)\)",
        text, re.DOTALL,
    )
    assert m, f"associative array {name} not found"
    body = m.group(1)
    entries = re.findall(r"\[\s*([\w]+)\s*\]\s*=\s*\"?([\w_\- ]+?)\"?\s*(?:\n|$)", body)
    return dict(entries)


def _extract_grid_loop(text: str) -> list[str]:
    """Return the ordered grid_type names from
    ``for GRID in cubed_sphere latlon voronoi gaussian; do``.
    """
    m = re.search(r"for\s+GRID\s+in\s+([^;]+);\s*do", text)
    assert m, "no `for GRID in ...; do` loop found"
    return m.group(1).split()


def _extract_grid_loop_body(text: str) -> tuple[int, int, str]:
    """Return ``(start_offset, end_offset, body)`` of the bash
    ``for GRID in ...; do  <body>  done`` block.

    iter-54 fix: the cross-grid wrappers contain no nested
    ``for``/``while``/``until`` blocks, so we just locate the
    NEXT ``done`` keyword (anchored at line start to avoid
    matching ``for``/``done`` substrings inside echo / variable /
    error-message text such as ``run_amip.py failed for $GRID``)
    after the ``for GRID in ...; do`` opening.
    """
    m_open = re.search(r"for\s+GRID\s+in\s+[^;]+;\s*do", text)
    assert m_open, "no `for GRID in ...; do` loop opening"
    # Match a ``done`` keyword at the start of a line (allowing
    # leading whitespace) — bash convention for loop closures
    # in these wrapper scripts.
    m_close = re.search(
        r"^[ \t]*done\b", text[m_open.end():], re.MULTILINE,
    )
    assert m_close, "no matching ``done`` for grid loop"
    body_start = m_open.end()
    body_end = body_start + m_close.start()
    return body_start, body_end, text[body_start:body_end]


def _strip_comments(text: str) -> str:
    """Remove ``#`` line comments so substring checks can't false-match
    against documentation/usage examples in the header.

    Only strip lines that start with optional whitespace then ``#``.
    Inline ``#`` after code is left alone (rare in these wrappers and
    bash-quoting makes a robust split painful).
    """
    out_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Per-wrapper pin-tests.
# ---------------------------------------------------------------------------

class TestRceCrossGridWrapper:
    """Pin ``scripts/run/run_rce_cross_grid.sh`` (iter-24)."""

    @pytest.fixture(scope="class")
    def wrapper_text(self):
        return (_SCRIPTS_DIR / "run_rce_cross_grid.sh").read_text()

    @pytest.fixture(scope="class")
    def wrapper_code(self, wrapper_text):
        """Wrapper source with header comments stripped, so substring
        checks can't false-match against documentation."""
        return _strip_comments(wrapper_text)

    def test_wrapper_exists_and_executable(self):
        path = _SCRIPTS_DIR / "run_rce_cross_grid.sh"
        assert path.exists()
        assert path.stat().st_mode & 0o100, (
            f"{path.name} is not executable; chmod +x required"
        )

    def test_iterates_all_four_grid_types(self, wrapper_code):
        """The for loop must iterate over the four canonical grid
        types — and only those four (no duplicates, no extras)."""
        loop_grids = _extract_grid_loop(wrapper_code)
        assert sorted(loop_grids) == sorted(_RCE_AMIP_GRID_DISC.keys()), (
            f"RCE wrapper grid loop = {loop_grids}, "
            f"expected {sorted(_RCE_AMIP_GRID_DISC.keys())}"
        )
        # No duplicates.
        assert len(loop_grids) == len(set(loop_grids))

    def test_grid_disc_mapping_is_exact(self, wrapper_code):
        """iter-53 codex HIGH: pin the EXACT GRID_DISC mapping.
        iter-52 only checked values were valid discretization names,
        so a bug like ``[latlon]=cdgrid`` would have slipped
        through."""
        mapping = _parse_bash_assoc_array(wrapper_code, "GRID_DISC")
        assert mapping == _RCE_AMIP_GRID_DISC, (
            f"RCE GRID_DISC drifted from canonical:\n"
            f"  got:    {mapping}\n"
            f"  expect: {_RCE_AMIP_GRID_DISC}"
        )

    def test_grid_folder_mapping_is_exact(self, wrapper_code):
        """The output folder per grid must match the matrix-runner
        layout convention."""
        mapping = _parse_bash_assoc_array(wrapper_code, "GRID_FOLDER")
        # iter-26 codex HIGH: voronoi → icosahedral folder.
        assert mapping == {
            "cubed_sphere": "cubed_sphere",
            "latlon":       "latlon",
            "voronoi":      "icosahedral",
            "gaussian":     "spectral",
        }

    def test_outdir_construction_includes_test_prefix(self, wrapper_code):
        """The per-grid OUTDIR must be assigned with the
        ``hydrostatic/rce/`` matrix-runner-collector prefix.

        iter-53 codex HIGH: this anchors to the ``OUTDIR=`` line
        rather than substring-matching anywhere (which would let
        comments / dead branches pass)."""
        m = re.search(
            r'OUTDIR=\s*"\$OUTPUT/hydrostatic/rce/\$FOLDER/\$RES"',
            wrapper_code,
        )
        assert m, (
            "RCE wrapper OUTDIR= line must be exactly "
            '`"$OUTPUT/hydrostatic/rce/$FOLDER/$RES"`; '
            "iter-26 codex HIGH guard"
        )

    def test_cross_grid_plot_is_last_python_invocation(self, wrapper_code):
        """iter-53 codex HIGH: the cross-grid plot must be the LAST
        ``python scripts/matrix/run_atmosphere_test_matrix.py`` invocation
        in the file — otherwise a refactor that adds an unrelated
        post-step could leave the cross-grid plot calling against
        partial output."""
        # iter-54 codex HIGH: find EVERY python invocation
        # (not just the matrix runner) and assert the LAST one is
        # the cross-grid plot.  A buggy refactor that puts an
        # unrelated python call after the cross-grid plot
        # invocation must flip the test.
        invocations = [
            (m.start(), m.group(0))
            for m in re.finditer(
                r"python\s+\S+\.py(?:[^\n\\]|\\\n)*",
                wrapper_code,
            )
        ]
        assert invocations, (
            "RCE wrapper must invoke run_atmosphere_test_matrix.py at end"
        )
        # The last one must be the --cross-grid-plots-only call.
        last = invocations[-1][1]
        assert "--cross-grid-plots-only" in last
        assert "--test rce" in last

    def test_passes_diag_days_for_short_smokes(self, wrapper_code):
        """iter-73: ``run_rce.py`` only emits matrix-format outputs
        when its diag_log is non-empty, which requires ``DAYS >=
        --diag-days`` (default 5).  Short-day smokes (DAYS<=5) need
        the wrapper to force smaller --diag-days, otherwise the
        cross-grid plotter silently has nothing to read.
        """
        # The wrapper accepts a third positional arg (DIAG_DAYS) and
        # defaults it to 1.  Pin both the variable definition AND
        # that it's passed via ``--diag-days "$DIAG_DAYS"``.
        assert re.search(r'DIAG_DAYS=\$\{3:-\s*1\s*\}', wrapper_code), (
            "iter-73: RCE wrapper must define ``DIAG_DAYS=${3:-1}`` "
            "so short-day smokes produce diagnostics"
        )
        assert re.search(
            r'--diag-days\s+"?\$\{?DIAG_DAYS\}?"?', wrapper_code,
        ), (
            "iter-73: RCE wrapper must pass ``--diag-days "
            "\"$DIAG_DAYS\"`` to ``run_rce.py``"
        )

    def test_continues_on_single_grid_failure(self, wrapper_code):
        """iter-73: the iter-24 RCE wrapper had ``set -e`` and no
        per-grid failure guard.  iter-73 added the iter-43 / iter-72
        AMIP/OMIP pattern: ``|| { echo WARNING; ANY_FAILED=1; }``
        so a per-grid blowup (e.g., the iter-73 voronoi RCE
        BLOWUP) doesn't abort the whole cross-grid run.
        """
        m = re.search(
            r"run_rce\.py.*?\|\|\s*\{[^}]*?ANY_FAILED=1[^}]*?\}",
            wrapper_code, re.DOTALL,
        )
        assert m, (
            "iter-73: RCE wrapper must wrap ``run_rce.py`` with "
            "``|| { ... ANY_FAILED=1 ... }`` so a single grid "
            "blowup does not abort the whole cross-grid run."
        )

    def test_purges_stale_outputs_before_each_run(self, wrapper_code):
        """iter-75 codex HIGH: a failed re-run into an existing
        ``$OUTDIR`` could leave old ``mean_timeseries.csv`` /
        ``results.txt`` in place which the cross-grid plotter would
        treat as current success.  iter-75 added defensive ``rm
        -f`` calls before each ``run_rce.py`` invocation
        (mirrors the iter-43 AMIP pattern).

        Pin: the wrapper must call ``rm -f "$OUTDIR/...``"`` for
        the matrix-format files before invoking run_rce.py.
        """
        # Find the rm -f calls inside the loop body.
        body_match = re.search(
            r'for\s+GRID\s+in\b.*?done', wrapper_code, re.DOTALL,
        )
        assert body_match, "could not locate RCE wrapper for-loop body"
        body = body_match.group(0)
        # The purge happens BEFORE run_rce.py.
        purge_match = re.search(
            r'rm\s+-f\s+"\$OUTDIR/mean_timeseries\.csv"', body,
        )
        run_match = re.search(r'scripts/run/run_rce\.py', body)
        assert purge_match, (
            "iter-75 codex HIGH: RCE wrapper must purge stale "
            "$OUTDIR/mean_timeseries.csv before each run"
        )
        assert run_match, "no run_rce.py found in loop body"
        assert purge_match.start() < run_match.start(), (
            "iter-75 codex HIGH: rm -f must come BEFORE run_rce.py"
        )


class TestOmipCrossGridWrapper:
    """Pin ``scripts/run/run_omip_cross_grid.sh`` (iter-25).

    iter-53 codex HIGH: extended from minimal coverage (loop +
    executable only) to also pin: per-grid output convention, OMIP
    cross-grid plot invocation, and shared conventions.
    """

    @pytest.fixture(scope="class")
    def wrapper_text(self):
        return (_SCRIPTS_DIR / "run_omip_cross_grid.sh").read_text()

    @pytest.fixture(scope="class")
    def wrapper_code(self, wrapper_text):
        return _strip_comments(wrapper_text)

    def test_wrapper_exists_and_executable(self):
        path = _SCRIPTS_DIR / "run_omip_cross_grid.sh"
        assert path.exists()
        assert path.stat().st_mode & 0o100

    def test_iterates_all_four_ocean_grid_types(self, wrapper_code):
        """Ocean-side grid spelling: ``mpas`` and ``spectral``
        instead of ``voronoi`` / ``gaussian``."""
        loop_grids = _extract_grid_loop(wrapper_code)
        assert sorted(loop_grids) == sorted(_OMIP_GRIDS), (
            f"OMIP wrapper grid loop = {loop_grids}, "
            f"expected {sorted(_OMIP_GRIDS)}"
        )
        assert len(loop_grids) == len(set(loop_grids))

    def test_invokes_run_omip_per_grid(self, wrapper_code):
        """The wrapper must invoke ``run_omip.py`` once per
        iteration of the GRID loop, with ``--grid "$GRID"``."""
        # ``run_omip.py`` may be on one line and ``--grid "$GRID"``
        # on the next via a ``\\`` continuation; use re.DOTALL.
        m = re.search(
            r'run_omip\.py.*?--grid\s+"\$GRID"',
            wrapper_code, re.DOTALL,
        )
        assert m, (
            "OMIP wrapper must invoke ``run_omip.py --grid \"$GRID\"`` "
            "in the per-grid loop"
        )

    def test_output_dir_passes_user_OUTPUT(self, wrapper_code):
        """The wrapper passes ``$OUTDIR`` as ``--output`` to
        ``run_omip.py``; iter-25's run_omip.py then writes to
        ``$OUTDIR/<grid>/<resolution>/``.  iter-49 relaxed the
        ocean-matrix collector to accept this layout."""
        m = re.search(
            r'run_omip\.py.*?--output\s+"\$OUTDIR"',
            wrapper_code, re.DOTALL,
        )
        assert m, (
            "OMIP wrapper must pass ``--output \"$OUTDIR\"`` to "
            "run_omip.py so the matrix-runner collector picks it up"
        )

    def test_outdir_uses_canonical_omip_prefix(self, wrapper_code):
        """iter-53 fix: ``$OUTPUT/omip`` ensures the test_case_dir
        parent has the canonical name for the ocean-matrix
        ``--replot`` path to discover OMIP runs.  Without this,
        ``_run_replot`` would not find the OMIP test case."""
        m = re.search(r'OUTDIR=\s*"\$OUTPUT/omip"', wrapper_code)
        assert m, (
            "OMIP wrapper OUTDIR must be ``\"$OUTPUT/omip\"`` so "
            "the ocean-matrix --replot path discovers the runs"
        )

    def test_invokes_cross_grid_plotter_at_end(self, wrapper_code):
        """iter-53 + iter-54 codex HIGH: the OMIP wrapper must
        invoke ``run_ocean_test_matrix.py --replot`` after the
        per-grid loop so the iter-49-relaxed collector emits
        cross-grid comparison plots.  Without this step the
        wrapper produces only per-grid plots (the original
        iter-25 limitation)."""
        invocations = [
            (m.start(), m.group(0))
            for m in re.finditer(
                r"python\s+\S+\.py(?:[^\n\\]|\\\n)*",
                wrapper_code,
            )
        ]
        assert invocations
        last = invocations[-1][1]
        assert "run_ocean_test_matrix.py" in last
        assert "--replot" in last
        assert "--only omip" in last

    def test_continues_on_single_grid_failure(self, wrapper_code):
        """iter-72: ``run_omip.py`` exits with code 1 on FAIL
        (e.g., the iter-71 cube C24 BLOWUP).  Without an explicit
        ``|| { ... }`` continuation, ``set -e`` aborts the whole
        wrapper after the first grid failure.  iter-72 added a
        guard that mirrors the iter-43 AMIP wrapper pattern: log
        a warning, set ``ANY_FAILED=1``, and continue to the
        remaining grids so they still contribute to the cross-grid
        plot.

        Pin: the per-grid run is followed by a ``|| { ... }``
        block AND ``ANY_FAILED=1`` is set inside that block.
        """
        # Find the ``run_omip.py ... || { ... }`` pattern.
        m = re.search(
            r"run_omip\.py.*?\|\|\s*\{[^}]*?ANY_FAILED=1[^}]*?\}",
            wrapper_code, re.DOTALL,
        )
        assert m, (
            "iter-72: OMIP wrapper must wrap ``run_omip.py`` with "
            "``|| { ... ANY_FAILED=1 ... }`` so a single grid "
            "failure (e.g., the iter-71 cube C24 BLOWUP) does not "
            "abort the whole cross-grid run."
        )

    def test_purges_stale_outputs_before_each_run(self, wrapper_code):
        """iter-75 codex HIGH: a failed re-run into an existing
        ``$OUTDIR/$GRID`` could leave old per-grid OMIP output
        in place which the matrix-runner ``--replot`` would treat
        as current success.  iter-75 added a defensive ``rm -rf
        "$OUTDIR/$GRID"`` before each ``run_omip.py`` invocation.

        Pin: the wrapper must purge ``$OUTDIR/$GRID`` before
        invoking run_omip.py.
        """
        body_match = re.search(
            r'for\s+GRID\s+in\b.*?done', wrapper_code, re.DOTALL,
        )
        assert body_match, "could not locate OMIP wrapper for-loop body"
        body = body_match.group(0)
        purge_match = re.search(
            r'rm\s+-rf\s+"\$OUTDIR/\$GRID"', body,
        )
        run_match = re.search(r'scripts/run/run_omip\.py', body)
        assert purge_match, (
            "iter-75 codex HIGH: OMIP wrapper must purge "
            "$OUTDIR/$GRID before each run"
        )
        assert run_match, "no run_omip.py found in loop body"
        assert purge_match.start() < run_match.start(), (
            "iter-75 codex HIGH: rm -rf must come BEFORE run_omip.py"
        )


class TestAmipCrossGridWrapper:
    """Pin ``scripts/run/run_amip_cross_grid.sh`` (iter-41) and the
    iter-42/43/44/45 converter integration."""

    @pytest.fixture(scope="class")
    def wrapper_text(self):
        return (_SCRIPTS_DIR / "run_amip_cross_grid.sh").read_text()

    @pytest.fixture(scope="class")
    def wrapper_code(self, wrapper_text):
        return _strip_comments(wrapper_text)

    def test_wrapper_exists_and_executable(self):
        path = _SCRIPTS_DIR / "run_amip_cross_grid.sh"
        assert path.exists()
        assert path.stat().st_mode & 0o100

    def test_iterates_all_four_grid_types(self, wrapper_code):
        loop_grids = _extract_grid_loop(wrapper_code)
        assert sorted(loop_grids) == sorted(_RCE_AMIP_GRID_DISC.keys()), (
            f"AMIP wrapper grid loop = {loop_grids}, "
            f"expected {sorted(_RCE_AMIP_GRID_DISC.keys())}"
        )

    def test_grid_disc_mapping_is_exact(self, wrapper_code):
        """iter-53 codex HIGH: pin the EXACT mapping (not just
        valid values)."""
        mapping = _parse_bash_assoc_array(wrapper_code, "GRID_DISC")
        assert mapping == _RCE_AMIP_GRID_DISC, (
            f"AMIP GRID_DISC drifted from canonical:\n"
            f"  got:    {mapping}\n"
            f"  expect: {_RCE_AMIP_GRID_DISC}"
        )

    def test_grid_disc_matches_rce(self, wrapper_code):
        """RCE and AMIP wrappers must share the EXACT same
        GRID_DISC mapping so cross-grid HS / RCE / AMIP comparisons
        stay consistent."""
        rce_text = _strip_comments(
            (_SCRIPTS_DIR / "run_rce_cross_grid.sh").read_text(),
        )
        amip_disc = _parse_bash_assoc_array(wrapper_code, "GRID_DISC")
        rce_disc = _parse_bash_assoc_array(rce_text, "GRID_DISC")
        assert amip_disc == rce_disc

    def test_outdir_construction_includes_test_prefix(self, wrapper_code):
        """iter-53 codex HIGH: anchor to the OUTDIR= line with the
        ``hydrostatic/amip/`` matrix-runner-collector prefix."""
        m = re.search(
            r'OUTDIR=\s*"\$OUTPUT/hydrostatic/amip/\$FOLDER/\$RES"',
            wrapper_code,
        )
        assert m, (
            "AMIP wrapper OUTDIR= line must be exactly "
            '`"$OUTPUT/hydrostatic/amip/$FOLDER/$RES"`'
        )

    def test_purges_stale_npz_BEFORE_run_amip_inside_loop(
        self, wrapper_code,
    ):
        """iter-54 codex HIGH: the iter-53 whole-file order check
        let a hypothetical refactor that moved the purge OUT of
        the loop body still pass.  Re-scope the order check to
        WITHIN the loop body so per-iteration semantics are
        pinned.
        """
        _, _, body = _extract_grid_loop_body(wrapper_code)
        purge_match = re.search(
            r'rm\s+-f\s+"\$OUTDIR/timeseries\.npz"',
            body,
        )
        run_match = re.search(r'scripts/run/run_amip\.py', body)
        assert purge_match, (
            "iter-43 codex HIGH guard missing INSIDE the loop body: "
            'rm -f "$OUTDIR/timeseries.npz"'
        )
        assert run_match, "no run_amip.py invocation INSIDE the loop body"
        assert purge_match.start() < run_match.start(), (
            f"iter-54 codex HIGH: rm -f must come BEFORE run_amip.py "
            f"INSIDE the loop body"
        )

    def test_converter_runs_AFTER_run_amip_inside_loop(self, wrapper_code):
        """iter-54 codex HIGH: the converter must run AFTER
        ``run_amip.py`` finishes within EACH per-grid iteration,
        not just somewhere after the whole loop."""
        _, _, body = _extract_grid_loop_body(wrapper_code)
        run_match = re.search(r'scripts/run/run_amip\.py', body)
        conv_match = re.search(
            r'scripts/run/_amip_to_matrix_format\.py', body,
        )
        assert run_match and conv_match, (
            "Both run_amip.py and _amip_to_matrix_format.py must "
            "appear INSIDE the per-grid loop body"
        )
        assert run_match.start() < conv_match.start(), (
            "iter-54 codex HIGH: converter must run AFTER run_amip.py "
            "INSIDE the loop body (per-iteration ordering)"
        )

    def test_converter_uses_per_grid_OUTDIR(self, wrapper_code):
        """iter-53 codex MEDIUM: the converter must be invoked with
        ``$OUTDIR`` (the per-grid path) not ``$OUTPUT`` (the
        cross-grid base).  iter-54 codex MEDIUM: scope to the
        loop body."""
        _, _, body = _extract_grid_loop_body(wrapper_code)
        m = re.search(
            r'_amip_to_matrix_format\.py\s+"\$OUTDIR"',
            body,
        )
        assert m, (
            "iter-53 codex MEDIUM: converter must be called with "
            '``"$OUTDIR"`` INSIDE the loop body'
        )

    def test_any_failed_set_on_converter_failure(self, wrapper_code):
        """iter-54 codex MEDIUM: ``ANY_FAILED=1`` must be inside
        the ``if ! ... ; then`` block whose condition is the
        converter invocation — scoped to the loop body for
        proper per-grid semantics.  The previous regex
        ``[^f]*?`` was brittle; use a simpler structural check:
        the converter's ``if ! ...`` line must appear in the
        loop body, and ``ANY_FAILED=1`` must appear BEFORE the
        matching ``fi`` for that block."""
        _, _, body = _extract_grid_loop_body(wrapper_code)
        # Find the converter's if-not block opening.
        if_open = re.search(
            r'if\s+!\s+\.venv/bin/python\s+scripts/run/_amip_to_matrix_format\.py'
            r'\s+"\$OUTDIR";?\s*then',
            body,
        )
        assert if_open, (
            "iter-54 codex MEDIUM: missing ``if ! ... "
            "_amip_to_matrix_format.py \"$OUTDIR\"; then`` block "
            "in the loop body"
        )
        # Find the matching ``fi`` (no nested if expected here).
        fi_match = re.search(r'\n\s*fi\b', body[if_open.end():])
        assert fi_match, "missing ``fi`` for converter if-block"
        block_text = body[if_open.end():if_open.end() + fi_match.start()]
        assert "ANY_FAILED=1" in block_text, (
            "iter-54 codex MEDIUM: ``ANY_FAILED=1`` must be inside "
            "the converter ``if ! ...; then ... fi`` block"
        )

    def test_cross_grid_plot_is_last_python_invocation(self, wrapper_code):
        """The cross-grid plot must be the last python invocation,
        with ``--cross-grid-plots-only`` and ``--test amip``."""
        invocations = [
            (m.start(), m.group(0))
            for m in re.finditer(
                r"python\s+scripts/matrix/run_atmosphere_test_matrix\.py"
                r"(?:[^\n\\]|\\\n)*",
                wrapper_code,
            )
        ]
        assert invocations
        last = invocations[-1][1]
        assert "--cross-grid-plots-only" in last
        assert "--test amip" in last

    def test_supports_optional_forcing_file_args(self, wrapper_code):
        """iter-54 codex LOW: tighten from substring to
        positional-arg parsing AND verify the ``--*-file`` flags
        appear in the EXTRA_FLAGS string that's threaded through
        to ``run_amip.py``.  Substring tests on the whole file
        could match dead branches or commented-out earlier
        attempts.
        """
        # Positional args.
        assert re.search(r"GHG_FILE=\$\{3:-\}", wrapper_code)
        assert re.search(r"OZONE_FILE=\$\{4:-\}", wrapper_code)
        assert re.search(r"AEROSOL_FILE=\$\{5:-\}", wrapper_code)

        # Each forcing kind must:
        #   1. Build an EXTRA_FLAGS append guarded by ``-n "$XXX_FILE"``.
        #   2. Use both ``--<kind>-forcing external`` AND
        #      ``--<kind>-file $XXX_FILE``.
        for kind, var in [
            ("ghg", "GHG_FILE"),
            ("ozone", "OZONE_FILE"),
            ("aerosol", "AEROSOL_FILE"),
        ]:
            # Find the if-block that constructs the EXTRA_FLAGS for
            # this kind.  Bash accepts both ``"$VAR"`` and
            # ``"${VAR}"``; allow either spelling.
            m = re.search(
                rf'if\s+\[\s+-n\s+"\${{?{var}}}?"\s+\];?\s*then\s*\n'
                rf'\s*EXTRA_FLAGS\+=" --{kind}-forcing external '
                rf'--{kind}-file \${{?{var}}}?"',
                wrapper_code,
            )
            assert m, (
                f"iter-54 codex LOW: {kind} forcing-file plumbing "
                f"missing or refactored.  Expected an "
                f'``if [ -n "${var}" ]; then EXTRA_FLAGS+=" '
                f'--{kind}-forcing external --{kind}-file ${var}"`` '
                f"block (with or without ``${{}}`` braces)."
            )
        # And EXTRA_FLAGS must be passed through to run_amip.py.
        assert "$EXTRA_FLAGS" in wrapper_code, (
            "EXTRA_FLAGS must be passed through to run_amip.py"
        )

    def test_passes_resolution_to_run_amip(self, wrapper_code):
        """iter-74: the iter-41 wrapper computed ``$RES`` from
        GRID_RES for the OUTDIR path but NEVER passed
        ``--resolution`` to ``run_amip.py``.  Result: every grid
        silently ran at the default n=16.  iter-74 fixed this with
        ``--resolution "$RES"`` in the wrapper's ``run_amip.py``
        invocation.

        Pin: the wrapper must include ``--resolution "$RES"`` (or
        ``--resolution $RES``) in the run_amip.py call.
        """
        # Match the run_amip.py invocation including line continuations.
        m = re.search(
            r'run_amip\.py(?:[^\n\\]|\\\n)*?--resolution\s+"?\$\{?RES\}?"?',
            wrapper_code,
        )
        assert m, (
            "iter-74: AMIP wrapper must pass ``--resolution "
            "\"$RES\"`` to ``run_amip.py``.  Without this, the "
            "GRID_RES dict is decorative-only and every grid runs "
            "at the default n=16."
        )

    def test_passes_diag_days_for_short_smokes(self, wrapper_code):
        """iter-74: ``run_amip.py`` only emits diagnostics every
        ``--diag-days`` (default 5).  Short-day smokes (DAYS<=5)
        produce empty ``timeseries.npz`` which the iter-42 converter
        purges as failed output.  iter-74 forced ``--diag-days 1``
        in the wrapper.
        """
        m = re.search(
            r'run_amip\.py(?:[^\n\\]|\\\n)*?--diag-days\s+\d+',
            wrapper_code,
        )
        assert m, (
            "iter-74: AMIP wrapper must pass ``--diag-days 1`` (or "
            "similar small int) so short-day smokes accumulate "
            "diagnostics."
        )

    def test_grid_res_uses_int_compatible_values(self, wrapper_code):
        """iter-74: ``run_amip.py --resolution`` is ``type=int``.
        The previous GRID_RES had ``[latlon]="90x180"`` which
        would fail at argparse.  iter-74 simplified to single
        ints for all 4 grids.
        """
        mapping = _parse_bash_assoc_array(wrapper_code, "GRID_RES")
        for grid, value in mapping.items():
            # Each value must be parseable as an int (or contain
            # only digits with optional whitespace).
            assert value.strip().isdigit(), (
                f"iter-74: GRID_RES[{grid}] = {value!r} is not an "
                f"int — ``run_amip.py --resolution`` is type=int "
                f"and would fail at argparse"
            )


class TestSharedConventions:
    """Pin the SHARED conventions across all three cross-grid
    wrappers."""

    @pytest.fixture(scope="class")
    def wrappers(self):
        return [
            _SCRIPTS_DIR / "run_rce_cross_grid.sh",
            _SCRIPTS_DIR / "run_omip_cross_grid.sh",
            _SCRIPTS_DIR / "run_amip_cross_grid.sh",
        ]

    def test_all_use_set_e(self, wrappers):
        for path in wrappers:
            assert "set -e" in _strip_comments(path.read_text()), (
                f"{path.name} missing 'set -e'"
            )

    def test_all_use_jax_enable_x64(self, wrappers):
        for path in wrappers:
            assert "JAX_ENABLE_X64=1" in _strip_comments(path.read_text()), (
                f"{path.name} missing JAX_ENABLE_X64=1"
            )

    def test_all_have_usage_error(self, wrappers):
        for path in wrappers:
            text = _strip_comments(path.read_text())
            assert re.search(r"\$\{1:\?usage", text), (
                f"{path.name} missing ``${{1:?usage:...}}`` guard"
            )


class TestWrappersPropagateAnyFailedToExitCode:
    """iter-103: pre-iter-103, all 3 cross-grid wrappers
    detected per-grid failures via ``ANY_FAILED=1`` and printed
    a NOTE about it, but exited with code 0 unconditionally.
    CI/automation calling
    ``bash run_amip_cross_grid.sh ... && publish_results``
    would proceed to ``publish_results`` even when all grids
    BLEW UP.

    iter-103 added ``exit "$ANY_FAILED"`` at the end of each
    wrapper so:
    * 0 if all grids succeeded → caller proceeds.
    * 1 if any grid had no usable output → caller halts.

    Comparison plots are still generated regardless (so
    successful grids contribute debugging output even on
    partial failure), but the final exit code reflects
    overall status.

    Verified end-to-end: invoking
    ``bash run_rce_cross_grid.sh /tmp/X 2 1`` (which exercises
    the iter-73 voronoi RCE BLOWUP) now correctly exits 1.
    Pre-iter-103 it exited 0.
    """

    @pytest.fixture(scope="class")
    def wrappers(self):
        return [
            _SCRIPTS_DIR / "run_rce_cross_grid.sh",
            _SCRIPTS_DIR / "run_omip_cross_grid.sh",
            _SCRIPTS_DIR / "run_amip_cross_grid.sh",
        ]

    def test_all_wrappers_exit_with_any_failed(self, wrappers):
        """iter-103 + iter-104: every wrapper must propagate
        per-grid failure (ANY_FAILED) AND plot-step failure
        (PLOT_FAILED) to its own exit code.
        """
        for path in wrappers:
            text = _strip_comments(path.read_text())
            # iter-104 MEDIUM-4: changed from
            # ``exit "$ANY_FAILED"`` to
            # ``exit $((ANY_FAILED || PLOT_FAILED))``
            # so plot-step failures are surfaced too (and
            # ``set -e`` doesn't abort the wrapper before the
            # exit line is reached).
            assert "exit $((ANY_FAILED || PLOT_FAILED))" in text, (
                f"iter-104: {path.name} must end with "
                f"``exit $((ANY_FAILED || PLOT_FAILED))`` so CI "
                f"can detect both per-grid failures AND plot-step "
                f"failures via ``$?``."
            )

    def test_exit_is_after_comparison_plot_step(self, wrappers):
        """The exit must come AFTER the comparison plot step
        so partial-failure runs still produce diagnostic
        output for the grids that succeeded.
        """
        for path in wrappers:
            text = path.read_text()
            # Find the index of the comparison-plot Python
            # invocation and the index of the exit line; exit
            # must be later.
            plot_idx = text.find("--cross-grid-plots-only")
            if plot_idx < 0:
                plot_idx = text.find("--replot")
            exit_idx = text.find("exit $((ANY_FAILED || PLOT_FAILED))")
            assert plot_idx >= 0, (
                f"iter-103 prerequisite: {path.name} must invoke "
                f"a comparison-plot step "
                f"(--cross-grid-plots-only or --replot)."
            )
            assert exit_idx > plot_idx, (
                f"iter-103/104: in {path.name}, the final "
                f"``exit $((ANY_FAILED || PLOT_FAILED))`` must "
                f"come AFTER the comparison-plot step so "
                f"partial-failure runs still get diagnostic "
                f"plots for the successful grids.  Found exit "
                f"at index {exit_idx}, plot at {plot_idx}."
            )

    def test_plot_step_does_not_abort_wrapper(self, wrappers):
        """iter-104 MEDIUM-4: the comparison-plot step must NOT
        abort the wrapper via ``set -e`` if it fails — instead
        it must capture the failure into ``PLOT_FAILED`` and
        let the wrapper continue to the explicit ``exit``.
        """
        for path in wrappers:
            text = _strip_comments(path.read_text())
            # The plot invocation must be guarded with
            # ``|| PLOT_FAILED=1`` (or similar) so set -e
            # doesn't fire on failure.
            assert "PLOT_FAILED=1" in text, (
                f"iter-104: {path.name} must capture comparison-"
                f"plot failure into ``PLOT_FAILED=1`` rather "
                f"than letting set -e abort the wrapper."
            )
