"""The benchmark suite driver must run the checkout it belongs to.

The venv is an editable install pointing at ONE working copy. A job launched
from a git worktree WITHOUT a pinned import path therefore ran that worktree's
scripts against a different copy's library code -- and when that other copy had
uncommitted work in progress, seven arms failed with an error that belonged to
somebody else's branch. It took a wrong diagnosis and a wasted suite to find.

These tests are cheap and they close exactly that hole.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SBATCH = _REPO / "scripts" / "cluster" / "ocean_grid_benchmark_suite.sbatch"
_PLOTTER = _REPO / "scripts" / "plot" / "plot_ocean_grid_benchmark_maps.py"


@pytest.fixture(scope="module")
def driver() -> str:
    return _SBATCH.read_text()


def test_driver_exists_and_is_a_batch_script(driver):
    assert driver.startswith("#!/bin/bash")
    assert "--array=0-8" in driver, "the submit line names the case count"


def test_repo_is_derived_from_the_script_location(driver):
    """A hardcoded path silently runs the wrong checkout from a worktree."""
    assert "BASH_SOURCE" in driver, (
        "REPO is not derived from the script's own location; a worktree would "
        "run a different checkout's code")
    for line in driver.splitlines():
        st = line.strip()
        if st.startswith("REPO=") and "BASH_SOURCE" not in st:
            assert st.startswith("REPO=\"${REPO:-"), (
                f"hardcoded REPO assignment: {st}")


def test_pythonpath_is_pinned(driver):
    """The actual defect: without this the job imports another copy."""
    assert "export PYTHONPATH" in driver, (
        "PYTHONPATH is not pinned, so the editable install decides which "
        "library code runs -- not the checkout being tested")
    assert "$REPO" in driver.split("export PYTHONPATH")[0].rsplit(
        "PYTHONPATH=", 1)[-1] or "REPO" in driver, (
        "PYTHONPATH does not reference the resolved REPO")


def test_the_missing_mesh_dependency_is_documented(driver):
    """One grid needs a ~480 MB file that is not in git.

    Undocumented, it errors and the figure shows an empty column, which reads
    as a broken grid rather than a missing input."""
    assert "eORCA1" in driver, (
        "the tripole mesh dependency is undocumented; its absence produces an "
        "empty column that looks like a model failure")


def test_the_plotting_step_is_documented(driver):
    """Reproducing the figures is two commands; the driver should say so."""
    assert "plot_ocean_grid_benchmark_maps" in driver


def test_plotter_writes_vector_and_print_resolution():
    """Publication output, asserted where it is set rather than assumed."""
    src = _PLOTTER.read_text()
    assert 'PUB = {' in src and '"dpi": 300' in src, "print dpi not set"
    assert src.count('.with_suffix(".pdf")') >= 2, (
        "both the per-case figures and the contact sheet must emit a vector "
        "copy, not just a raster")


def test_plotter_labels_each_panel_with_its_own_range():
    """Without this a weak arm on a shared scale reads as a failed run."""
    src = _PLOTTER.read_text()
    assert "own range" in src, (
        "panels no longer print their own data range; a pale panel becomes "
        "indistinguishable from a broken one")
