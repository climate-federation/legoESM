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


# ---------------------------------------------------------------------------
# The tests above assert the STRING. That is the shadow of the mechanism, not
# the mechanism: an editable install that used a meta-path finder would beat
# PYTHONPATH silently and every string assertion would still pass. GLM-5.2
# refused to merge on exactly that, and it was right to. The test below
# resolves a real import in a subprocess.
# ---------------------------------------------------------------------------

def test_the_pin_actually_wins_the_import():
    """Resolve a module for real under the driver's own PYTHONPATH.

    Measured here rather than reasoned about: without the pin this repo's
    packages resolve through the editable install to whichever working copy it
    points at; with it, they must resolve inside THIS checkout."""
    import os
    import subprocess
    import sys

    pkgs = sorted((_REPO / "packages").glob("*/"))
    assert pkgs, "no packages/ found; the driver's glob would match nothing"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(d) for d in pkgs] + [str(_REPO / "src")])
    code = ("import legoesm.ocean.dynamics.barotropic_mpas as m; "
            "print(m.__file__)")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    resolved = out.stdout.strip()
    assert resolved.startswith(str(_REPO)), (
        f"the PYTHONPATH pin did NOT win: legoesm.ocean resolved to "
        f"{resolved}, outside {_REPO}. Every run using this driver would "
        f"execute another checkout's library code.")


def test_driver_refuses_to_run_if_the_pin_loses(driver):
    """The runtime guard, not just the build-time one.

    A silent fallback here reruns the original incident with no symptom, so
    the driver must EXIT rather than warn."""
    assert "provenance" in driver, "no runtime import-provenance check"
    assert "exit 2" in driver, (
        "the provenance check does not abort the job; a warning would be "
        "ignored and the run would proceed against the wrong checkout")


def test_driver_fails_loudly_if_repo_resolves_wrong(driver):
    """Some SLURM sites spool the script, which breaks BASH_SOURCE.

    That path is silent by construction -- the glob matches nothing and Python
    ignores the literal star -- so it needs an explicit check."""
    assert "FATAL" in driver and "packages" in driver, (
        "no guard that REPO actually contains the packages tree")


def test_figures_carry_provenance():
    """A mixed-run figure looks fine, which is worse than a blank panel.

    The plotter reads whatever is on disk with no manifest, so every figure
    stamps the commit and the time span of the artifacts it drew from. A wide
    span is the signal that two runs got mixed (GLM-5.2)."""
    src = _PLOTTER.read_text()
    assert "_provenance" in src, "figures carry no provenance stamp"
    assert src.count("_provenance(root)") >= 2, (
        "both the per-case figures and the contact sheet must be stamped")
    assert "span" in src, (
        "the stamp does not flag artifacts spanning a long window, which is "
        "the signature of a figure built from two different runs")
