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
    # The documented array bound must match the case list. A literal here went
    # stale the moment a tenth case was added, and a bound that is short by one
    # silently never runs the last case -- the array task simply does not
    # exist, so nothing reports a gap.
    import re

    cases = re.search(r"^CASES=\((.*?)\)$", driver, re.S | re.M)
    assert cases, "no CASES array in the driver"
    n = len([c for c in cases.group(1).replace("\\\n", " ").split() if c])
    assert f"--array=0-{n - 1}" in driver, (
        f"the submit line does not cover all {n} cases; an array bound short "
        f"of the list means the last case never runs and nothing says so")


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
    pinned = driver.split("export PYTHONPATH")[0].rsplit("PYTHONPATH=", 1)[-1]
    assert "$REPO" in pinned, (
        "the PYTHONPATH assignment does not reference the resolved REPO, so "
        f"the job would import from somewhere else: {pinned.strip()!r}")


def test_each_array_task_selects_exactly_its_own_case(driver):
    """Substring selection runs a neighbour's case as well.

    ``inertia_gravity_wave`` is a prefix of ``inertia_gravity_wave_channel``,
    so a plain ``--only <case>`` makes one array task run both, and the
    channel case then also runs in its own task -- two arms of the reported
    suite come from a task that was not accounted for, and either can set the
    task's exit status.  The matrix runner's ``=`` prefix means exact match.
    """
    assert '--only "=$CASE"' in driver, (
        "the array task selects its case by substring, so it also runs every "
        "case whose name starts with the same text")


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


def test_the_preflight_runs_with_nothing_inherited_from_the_submitting_shell():
    """Run the driver's own preflight, not a reconstruction of it.

    ``sbatch script.sbatch`` starts the job with whatever the submitting shell
    exported and nothing else.  The preflight block therefore has to stand on
    its own: it shipped once reading ``REPO`` from the environment while the
    script only ever set it as a shell variable, so every submitted job died
    at the guard before running a single case, and every string assertion in
    this file stayed green.  Executing the block is what catches that.
    """
    import os
    import subprocess
    import sys
    import tempfile

    body = _SBATCH.read_text().splitlines(keepends=True)
    start = next(i for i, l in enumerate(body) if l.startswith("set -euo"))
    end = next(i for i, l in enumerate(body) if l.rstrip() == "PYCHK")
    preflight = "".join(body[start:end + 1])
    # The real script resolves REPO from BASH_SOURCE; a temp copy cannot, so
    # hand it the checkout the way an explicit override would -- everything
    # else (export or not, the guard, the import provenance check) is the
    # script's own text.
    preflight = preflight.replace('${BASH_SOURCE[0]}', str(_SBATCH))
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(preflight)
        script = fh.name
    try:
        # PYTHON_BIN is the script's own override: this checkout may be a
        # worktree with no .venv of its own, and the interpreter is not what
        # is under test here -- the environment plumbing is.
        clean = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "HOME": os.environ.get("HOME", "/tmp"),
                 "PYTHON_BIN": sys.executable}
        out = subprocess.run(["bash", script], env=clean, cwd=str(_REPO),
                             capture_output=True, text=True, timeout=600)
    finally:
        os.unlink(script)
    assert out.returncode == 0, (
        "the driver's preflight fails when the submitting shell exported "
        f"nothing, i.e. on the documented `sbatch` invocation:\n"
        f"{out.stdout[-600:]}\n{out.stderr[-800:]}")
    assert "[provenance]" in out.stdout, (
        "the preflight ran but never reported where the library resolved "
        f"from: {out.stdout[-400:]!r}")


# ---------------------------------------------------------------------------
# Codex adversarial pass: the guard above checked the RIGHT thing about the
# WRONG interpreter, and accepted a neighbouring checkout by string prefix.
# ---------------------------------------------------------------------------

def test_the_cases_run_the_interpreter_the_preflight_validated(driver):
    """A green preflight must cover the interpreter that runs the model.

    ``PYTHON_BIN`` is a documented override, so the preflight can validate one
    interpreter while the case launch hardcodes another -- and the import the
    preflight proved is then not the import the benchmark uses."""
    launch = [l for l in driver.splitlines()
              if "run_ocean_test_matrix.py" in l and l.strip().startswith("srun")]
    assert launch, "no srun line launching the matrix runner"
    assert all("python_bin" in l for l in launch), (
        "the case launch hardcodes an interpreter instead of the one the "
        f"provenance preflight validated: {launch}")


def test_the_provenance_guard_uses_containment_not_a_string_prefix(driver):
    """``/work/run/1622`` is a string-prefix of ``/work/run/1622-old``.

    With ``startswith`` the guard blesses that neighbouring checkout, which is
    precisely the incident it was written to prevent."""
    assert "commonpath" in driver, (
        "the import-provenance guard compares paths by string prefix, so a "
        "different checkout whose path merely starts with REPO passes it")
    assert "m.__file__.startswith(" not in driver


def test_the_figure_stamp_does_not_claim_the_run_s_commit():
    """Nothing on disk records which commit produced the arms.

    Stamping a bare ``commit X`` next to the run times asserts a provenance
    the plotter cannot know: replot from another branch and the figures claim
    the arms came from it."""
    src = _PLOTTER.read_text()
    assert "plotted at commit" in src, (
        "the provenance stamp presents the PLOTTING checkout's commit as the "
        "run's own")


# ---------------------------------------------------------------------------
# GLM-5.2: the time span was standing in for a fact nothing recorded. Rerun one
# case hours later against different code and the span stays narrow; copying
# with timestamps preserved, clock skew and archive extraction defeat it from
# the other side. Each arm now records what produced it.
# ---------------------------------------------------------------------------

def test_each_arm_records_the_commit_that_produced_it(tmp_path, monkeypatch):
    import importlib.util
    import sys

    sys.path.insert(0, str(_REPO / "scripts" / "matrix"))
    import run_ocean_test_matrix as m

    out = tmp_path / "arm"
    out.mkdir()
    m._write_results_txt(out, {"status": "PASS"})
    body = (out / "results.txt").read_text()
    assert "provenance_commit:" in body, (
        "an arm records nothing about what produced it, so a figure built "
        "from two runs is indistinguishable from one built from one")
    assert "provenance_python:" in body
    commit = [l for l in body.splitlines()
              if l.startswith("provenance_commit:")][0].split(":", 1)[1].strip()
    assert commit and commit != "unknown", (
        f"the commit stamp resolved to {commit!r} inside a git checkout")


def test_the_figure_says_so_when_the_arms_came_from_different_runs(tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "plot_ocean_grid_benchmark_maps", _PLOTTER)
    plot = importlib.util.module_from_spec(spec)
    sys.modules["plot_ocean_grid_benchmark_maps"] = plot
    spec.loader.exec_module(plot)

    def _arm(name, commit):
        d = tmp_path / name / "latlon" / "res"
        d.mkdir(parents=True)
        (d / "results.txt").write_text(
            f"status: PASS\nprovenance_commit: {commit}\n")
        import numpy as np
        np.savez(d / "snapshots_latlon.npz", T=np.zeros((2, 2)))

    _arm("case_a", "abc123def456")
    _arm("case_b", "abc123def456")
    line = plot._provenance(tmp_path)
    assert "MIXED RUNS" not in line, line
    assert "abc123def456" in line

    _arm("case_c", "999999999999")
    line = plot._provenance(tmp_path)
    assert "MIXED RUNS" in line and "999999999999" in line, line


def test_a_dirty_tree_is_named_on_the_figure(tmp_path):
    """Pinning the import path to this checkout does not make it clean, and an
    uncommitted edit in the pinned copy reproduces the incident the pin exists
    to prevent -- now with the pin lending it credibility."""
    import importlib.util
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "plot_ocean_grid_benchmark_maps", _PLOTTER)
    plot = importlib.util.module_from_spec(spec)
    sys.modules["plot_ocean_grid_benchmark_maps"] = plot
    spec.loader.exec_module(plot)

    d = tmp_path / "case" / "latlon" / "res"
    d.mkdir(parents=True)
    (d / "results.txt").write_text(
        "status: PASS\nprovenance_commit: abc123def456-dirty\n")
    np.savez(d / "snapshots_latlon.npz", T=np.zeros((2, 2)))
    assert "DIRTY" in plot._provenance(tmp_path)
