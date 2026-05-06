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
# grid_type names (see ``scripts/run_omip.py`` ``--grid`` choices).
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
    """Pin ``scripts/run_rce_cross_grid.sh`` (iter-24)."""

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
        ``python scripts/run_atmosphere_test_matrix.py`` invocation
        in the file — otherwise a refactor that adds an unrelated
        post-step could leave the cross-grid plot calling against
        partial output."""
        # Find every python invocation of the matrix runner.
        # Bash line continuations (``\\\n``) are consumed so we
        # capture the FULL command including any flags on the next
        # line (e.g. ``--cross-grid-plots-only``).
        invocations = [
            (m.start(), m.group(0))
            for m in re.finditer(
                r"python\s+scripts/run_atmosphere_test_matrix\.py"
                r"(?:[^\n\\]|\\\n)*",  # consume through line continuations
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


class TestOmipCrossGridWrapper:
    """Pin ``scripts/run_omip_cross_grid.sh`` (iter-25).

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


class TestAmipCrossGridWrapper:
    """Pin ``scripts/run_amip_cross_grid.sh`` (iter-41) and the
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

    def test_purges_stale_npz_BEFORE_run_amip(self, wrapper_code):
        """iter-53 codex HIGH: not only must ``rm -f
        $OUTDIR/timeseries.npz`` appear, but it must come BEFORE
        the ``run_amip.py`` invocation in source order.  Otherwise
        a wrapper that purges AFTER would still fool the substring
        check while reverting the iter-43 fix."""
        purge_match = re.search(
            r'rm\s+-f\s+"\$OUTDIR/timeseries\.npz"',
            wrapper_code,
        )
        run_match = re.search(r'scripts/run_amip\.py', wrapper_code)
        assert purge_match, (
            "iter-43 codex HIGH guard missing: "
            'rm -f "$OUTDIR/timeseries.npz"'
        )
        assert run_match, "no run_amip.py invocation found"
        assert purge_match.start() < run_match.start(), (
            f"iter-53 codex HIGH: the rm -f \"$OUTDIR/timeseries.npz\" "
            f"line (at offset {purge_match.start()}) must come BEFORE "
            f"the run_amip.py invocation (at offset {run_match.start()})"
        )

    def test_converter_runs_AFTER_run_amip(self, wrapper_code):
        """iter-53 codex MEDIUM: the converter must run AFTER
        ``run_amip.py`` finishes (otherwise it would convert
        whatever stale or partial state was on disk)."""
        run_match = re.search(r'scripts/run_amip\.py', wrapper_code)
        conv_match = re.search(
            r'scripts/_amip_to_matrix_format\.py', wrapper_code,
        )
        assert run_match and conv_match
        assert run_match.start() < conv_match.start(), (
            "iter-53 codex MEDIUM: ``_amip_to_matrix_format.py`` "
            "must be invoked AFTER ``run_amip.py``"
        )

    def test_converter_uses_per_grid_OUTDIR(self, wrapper_code):
        """iter-53 codex MEDIUM: the converter must be invoked with
        ``$OUTDIR`` (the per-grid path) not ``$OUTPUT`` (the
        cross-grid base)."""
        m = re.search(
            r'_amip_to_matrix_format\.py\s+"\$OUTDIR"',
            wrapper_code,
        )
        assert m, (
            "iter-53 codex MEDIUM: converter must be called with "
            '``"$OUTDIR"`` (the per-grid path)'
        )

    def test_any_failed_set_on_converter_failure(self, wrapper_code):
        """iter-53 codex MEDIUM: ``ANY_FAILED=1`` must be inside
        the ``if ! ... ; then`` block whose condition is the
        converter invocation — otherwise a stray assignment
        elsewhere would fool the substring check."""
        # Find the if-not-converter block.
        m = re.search(
            r'if\s+!\s+\.venv/bin/python\s+scripts/_amip_to_matrix_format\.py'
            r'\s+"\$OUTDIR";?\s*then[^f]*?ANY_FAILED=1[^f]*?fi',
            wrapper_code, re.DOTALL,
        )
        assert m, (
            "iter-53 codex MEDIUM: ``ANY_FAILED=1`` must be inside "
            "an ``if ! .../_amip_to_matrix_format.py \"$OUTDIR\"; "
            "then ... fi`` block tied to the converter exit status"
        )

    def test_cross_grid_plot_is_last_python_invocation(self, wrapper_code):
        """The cross-grid plot must be the last python invocation,
        with ``--cross-grid-plots-only`` and ``--test amip``."""
        invocations = [
            (m.start(), m.group(0))
            for m in re.finditer(
                r"python\s+scripts/run_atmosphere_test_matrix\.py"
                r"(?:[^\n\\]|\\\n)*",
                wrapper_code,
            )
        ]
        assert invocations
        last = invocations[-1][1]
        assert "--cross-grid-plots-only" in last
        assert "--test amip" in last

    def test_supports_optional_forcing_file_args(self, wrapper_code):
        """iter-53 codex LOW: tighten from substring to
        positional-arg parsing.  The wrapper must read
        ``$3``/``$4``/``$5`` as GHG/OZONE/AEROSOL files AND
        propagate them via ``--*-forcing external --*-file``."""
        # Positional args.
        assert re.search(r"GHG_FILE=\$\{3:-\}", wrapper_code), (
            "GHG_FILE must come from positional arg 3"
        )
        assert re.search(r"OZONE_FILE=\$\{4:-\}", wrapper_code), (
            "OZONE_FILE must come from positional arg 4"
        )
        assert re.search(r"AEROSOL_FILE=\$\{5:-\}", wrapper_code), (
            "AEROSOL_FILE must come from positional arg 5"
        )
        # Propagation through --*-forcing external --*-file flags.
        assert "--ghg-forcing external" in wrapper_code
        assert "--ghg-file $GHG_FILE" in wrapper_code
        assert "--ozone-forcing external" in wrapper_code
        assert "--ozone-file $OZONE_FILE" in wrapper_code
        assert "--aerosol-forcing external" in wrapper_code
        assert "--aerosol-file $AEROSOL_FILE" in wrapper_code


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
