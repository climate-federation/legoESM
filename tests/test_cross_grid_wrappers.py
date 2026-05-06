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

These tests parse the wrapper sources textually rather than running
them, so they are fast (no subprocess invocation, no JAX import) but
still catch the structural issues that have actually caused real
silent-skip bugs in the iter-25..50 chain (iter-26 RCE
discretization-flag regression, iter-49 OMIP collector mismatch,
etc.).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


# ---------------------------------------------------------------------------
# Shared expectations (the "cross-grid wrapper convention").
# ---------------------------------------------------------------------------

# Every cross-grid wrapper iterates over the four canonical grid types.
# The exact grid_type spelling differs between RCE/AMIP (``voronoi``,
# ``gaussian``) and OMIP (``mpas``, ``spectral``) — both are valid
# canonical names in the legoESM grid registry.  See
# ``src/legoesm/grids/`` for the source of truth.
_RCE_AMIP_GRIDS = {"cubed_sphere", "latlon", "voronoi", "gaussian"}
_OMIP_GRIDS = {"cubed_sphere", "latlon", "mpas", "spectral"}

# Discretization names (centered, finite_volume, cdgrid, latlon_cgrid,
# mpas, spectral) come from ``src/legoesm/supported_matrix.py``.  Each
# wrapper's GRID_DISC mapping picks the natural discretization for
# each grid type.
_VALID_DISCRETIZATIONS = {
    "centered", "finite_volume", "cdgrid", "latlon_cgrid",
    "mpas", "spectral",
}


# ---------------------------------------------------------------------------
# Per-wrapper pin-tests.
# ---------------------------------------------------------------------------

class TestRceCrossGridWrapper:
    """Pin ``scripts/run_rce_cross_grid.sh`` (iter-24)."""

    @pytest.fixture(scope="class")
    def wrapper_text(self):
        path = _SCRIPTS_DIR / "run_rce_cross_grid.sh"
        return path.read_text()

    def test_wrapper_exists_and_executable(self):
        path = _SCRIPTS_DIR / "run_rce_cross_grid.sh"
        assert path.exists()
        # Owner-execute bit is set (chmod +x).
        assert path.stat().st_mode & 0o100, (
            f"{path.name} is not executable; chmod +x required for "
            f"the wrapper convention"
        )

    def test_iterates_all_four_grid_types(self, wrapper_text):
        """The for loop must iterate over all four canonical grid
        types so dropping one (e.g. spectral) flips the test."""
        # Find the bash for-loop list.  Format: ``for GRID in cubed_sphere latlon voronoi gaussian; do``
        m = re.search(r"for\s+GRID\s+in\s+([^;]+);\s*do", wrapper_text)
        assert m, "no grid for-loop found in run_rce_cross_grid.sh"
        loop_grids = set(m.group(1).split())
        assert loop_grids == _RCE_AMIP_GRIDS, (
            f"run_rce_cross_grid.sh grid loop = {loop_grids}, "
            f"expected {_RCE_AMIP_GRIDS}"
        )

    def test_grid_disc_mapping_uses_valid_discretizations(
        self, wrapper_text,
    ):
        """The GRID_DISC mapping must use only valid discretization
        names from ``src/legoesm/supported_matrix.py``.  iter-26
        codex HIGH: ``run_rce.py`` rejects an unknown
        ``--discretization`` so the mapping value must be exact."""
        m = re.search(
            r"declare\s+-A\s+GRID_DISC=\((.*?)\)",
            wrapper_text, re.DOTALL,
        )
        assert m, "no GRID_DISC mapping found"
        body = m.group(1)
        # Each entry: [grid_name]="discretization"
        entries = re.findall(r"\[([\w]+)\]=\"([\w_]+)\"", body)
        assert len(entries) == 4, (
            f"GRID_DISC has {len(entries)} entries, expected 4"
        )
        seen_grids = set()
        for grid, disc in entries:
            assert grid in _RCE_AMIP_GRIDS
            assert disc in _VALID_DISCRETIZATIONS, (
                f"GRID_DISC[{grid}] = {disc!r} is not a valid "
                f"discretization"
            )
            seen_grids.add(grid)
        assert seen_grids == _RCE_AMIP_GRIDS

    def test_output_layout_is_hydrostatic_rce(self, wrapper_text):
        """The OUTDIR path must be ``$OUTPUT/hydrostatic/rce/<folder>/<res>``
        so the atmosphere matrix runner picks it up."""
        # iter-26 codex HIGH: dropping the ``hydrostatic/rce/``
        # prefix would silently make the cross-grid plotter miss
        # the runs.
        assert "hydrostatic/rce/" in wrapper_text, (
            "RCE wrapper output path must include hydrostatic/rce/ "
            "prefix for matrix-runner collection"
        )

    def test_invokes_cross_grid_plots_only_at_end(self, wrapper_text):
        """The wrapper must call ``run_atmosphere_test_matrix.py
        --cross-grid-plots-only`` after all per-grid runs finish."""
        assert (
            "run_atmosphere_test_matrix.py" in wrapper_text
            and "--cross-grid-plots-only" in wrapper_text
            and "--test rce" in wrapper_text
        ), (
            "RCE wrapper must end with `run_atmosphere_test_matrix.py "
            "--cross-grid-plots-only --test rce`"
        )


class TestOmipCrossGridWrapper:
    """Pin ``scripts/run_omip_cross_grid.sh`` (iter-25)."""

    @pytest.fixture(scope="class")
    def wrapper_text(self):
        path = _SCRIPTS_DIR / "run_omip_cross_grid.sh"
        return path.read_text()

    def test_wrapper_exists_and_executable(self):
        path = _SCRIPTS_DIR / "run_omip_cross_grid.sh"
        assert path.exists()
        assert path.stat().st_mode & 0o100

    def test_iterates_all_four_grid_types(self, wrapper_text):
        """Ocean-side grid spelling: ``mpas`` and ``spectral``
        instead of ``voronoi`` / ``gaussian`` (the ocean matrix
        runner uses these names directly)."""
        m = re.search(r"for\s+GRID\s+in\s+([^;]+);\s*do", wrapper_text)
        assert m, "no grid for-loop found in run_omip_cross_grid.sh"
        loop_grids = set(m.group(1).split())
        assert loop_grids == _OMIP_GRIDS, (
            f"run_omip_cross_grid.sh grid loop = {loop_grids}, "
            f"expected {_OMIP_GRIDS}"
        )


class TestAmipCrossGridWrapper:
    """Pin ``scripts/run_amip_cross_grid.sh`` (iter-41) and the
    iter-42/43/44/45 converter integration."""

    @pytest.fixture(scope="class")
    def wrapper_text(self):
        path = _SCRIPTS_DIR / "run_amip_cross_grid.sh"
        return path.read_text()

    def test_wrapper_exists_and_executable(self):
        path = _SCRIPTS_DIR / "run_amip_cross_grid.sh"
        assert path.exists()
        assert path.stat().st_mode & 0o100

    def test_iterates_all_four_grid_types(self, wrapper_text):
        m = re.search(r"for\s+GRID\s+in\s+([^;]+);\s*do", wrapper_text)
        assert m
        loop_grids = set(m.group(1).split())
        assert loop_grids == _RCE_AMIP_GRIDS, (
            f"run_amip_cross_grid.sh grid loop = {loop_grids}, "
            f"expected {_RCE_AMIP_GRIDS}"
        )

    def test_grid_disc_mapping_matches_rce(self, wrapper_text):
        """RCE and AMIP wrappers must use the SAME GRID_DISC
        mapping so cross-grid HS / RCE / AMIP comparisons stay
        consistent."""
        m = re.search(
            r"declare\s+-A\s+GRID_DISC=\((.*?)\)",
            wrapper_text, re.DOTALL,
        )
        assert m
        amip_disc = dict(re.findall(r"\[([\w]+)\]=\"([\w_]+)\"", m.group(1)))

        rce_text = (_SCRIPTS_DIR / "run_rce_cross_grid.sh").read_text()
        m_rce = re.search(
            r"declare\s+-A\s+GRID_DISC=\((.*?)\)",
            rce_text, re.DOTALL,
        )
        assert m_rce
        rce_disc = dict(re.findall(r"\[([\w]+)\]=\"([\w_]+)\"", m_rce.group(1)))

        assert amip_disc == rce_disc, (
            f"AMIP GRID_DISC = {amip_disc} but RCE GRID_DISC = "
            f"{rce_disc} — the two wrappers must agree to keep "
            f"cross-grid comparisons consistent"
        )

    def test_invokes_format_converter_after_each_run(self, wrapper_text):
        """iter-42 integration: each per-grid run must be
        followed by the AMIP→matrix format converter so the
        matrix runner's cross-grid plot path picks up the run.
        Without this step, OMIP-style silent-skip would happen
        for AMIP."""
        assert "_amip_to_matrix_format.py" in wrapper_text

    def test_purges_stale_npz_before_each_run(self, wrapper_text):
        """iter-43 codex HIGH: the wrapper must remove
        ``$OUTDIR/timeseries.npz`` BEFORE invoking ``run_amip.py``
        so a failed re-run cannot reconvert old data."""
        assert 'rm -f "$OUTDIR/timeseries.npz"' in wrapper_text, (
            "iter-43 codex HIGH guard missing: stale "
            "timeseries.npz must be removed before run_amip.py"
        )

    def test_propagates_converter_failure(self, wrapper_text):
        """iter-43: ANY_FAILED must be set when the converter
        returns non-zero so the cross-grid step can warn the
        user about partial output."""
        assert "ANY_FAILED=1" in wrapper_text

    def test_invokes_cross_grid_plots_only_at_end(self, wrapper_text):
        assert (
            "run_atmosphere_test_matrix.py" in wrapper_text
            and "--cross-grid-plots-only" in wrapper_text
            and "--test amip" in wrapper_text
        )

    def test_supports_optional_forcing_files(self, wrapper_text):
        """The wrapper must accept GHG/ozone/aerosol file paths as
        positional args (iter-41 design)."""
        assert "--ghg-forcing external" in wrapper_text
        assert "--ozone-forcing external" in wrapper_text
        assert "--aerosol-forcing external" in wrapper_text


class TestSharedConventions:
    """Pin the SHARED conventions across all three cross-grid
    wrappers so a new wrapper added in the future inherits them."""

    @pytest.fixture(scope="class")
    def wrappers(self):
        return [
            _SCRIPTS_DIR / "run_rce_cross_grid.sh",
            _SCRIPTS_DIR / "run_omip_cross_grid.sh",
            _SCRIPTS_DIR / "run_amip_cross_grid.sh",
        ]

    def test_all_use_set_e(self, wrappers):
        """``set -e`` ensures any failed command (other than
        explicitly handled ones) aborts the wrapper."""
        for path in wrappers:
            assert "set -e" in path.read_text(), (
                f"{path.name} missing 'set -e'"
            )

    def test_all_use_jax_enable_x64(self, wrappers):
        """All cross-grid wrappers must use x64 — spectral solvers
        require it and the matrix-runner cross-grid agreement
        baseline assumes x64."""
        for path in wrappers:
            assert "JAX_ENABLE_X64=1" in path.read_text(), (
                f"{path.name} missing JAX_ENABLE_X64=1"
            )

    def test_all_have_usage_error(self, wrappers):
        """The ``${1:?usage:...}`` pattern fires on missing OUTPUT."""
        for path in wrappers:
            text = path.read_text()
            assert re.search(r"\$\{1:\?usage", text), (
                f"{path.name} missing ``${{1:?usage:...}}`` guard"
            )
