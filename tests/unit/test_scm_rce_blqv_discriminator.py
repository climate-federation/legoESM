"""The boundary-layer humidity discriminator: its arms, and its launcher.

Three things this file pins, each from a defect codex found on the PR that
added the arms:

1. The committed launcher must actually select the arms the conclusion rests
   on.  It shipped defaulting to the four ORIGINAL arms, so submitting it
   reproduced the old result and produced no measurement of the new one.
2. Every arm must differ from the control in ONE thing.  The no-saturation-
   adjustment arm hardcoded a convection scheme while the control takes it
   from the command line, so with any other scheme selected the arm changed
   two variables and the difference would have been read as a saturation-
   adjustment effect.
3. The review wrapper must exit with the reviewer's own status.  It printed
   the status and exited 0, so an expired login looked like a completed
   review to anything reading the job state instead of the log.

These are static checks on purpose: running an arm is a 100-day column
integration, and what failed here was the wiring, not the physics.
"""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "scm_rce_blqv_discriminator.py"
_LAUNCHER = _ROOT / "scripts" / "cluster" / "scm_rce_paper" / "blqv_discriminator.sbatch"
_VERDICT = _ROOT / "scripts" / "cluster" / "scm_rce_paper" / "blqv_verdict_codex.sbatch"


def _arm_names():
    """The keys of the probe's ``arm_cfgs`` table, read from the source."""
    tree = ast.parse(_PROBE.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "arm_cfgs"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)]
    pytest.fail("arm_cfgs table not found — did the probe get restructured?")


def _launcher_default_arms():
    for line in _LAUNCHER.read_text().splitlines():
        if "--arms" in line:
            spec = line.split("${ARMS:-", 1)[1].split("}", 1)[0]
            return [a for a in spec.split(",") if a]
    pytest.fail("the launcher does not pass --arms")


def test_the_launcher_selects_every_arm_the_probe_defines():
    """A launcher that runs a subset produces a result for the OLD question."""
    defined, selected = set(_arm_names()), set(_launcher_default_arms())
    assert defined, "the probe defines no arms"
    missing = sorted(defined - selected)
    assert not missing, (
        f"the committed launcher never runs {missing}; submitting it produces "
        "no measurement for those arms, so any conclusion resting on them is "
        "unsupported by anything in the repository")
    unknown = sorted(selected - defined)
    assert not unknown, f"the launcher selects arms that do not exist: {unknown}"


def test_every_arm_takes_its_convection_scheme_from_the_command_line():
    """One variable per arm.

    The control's convection scheme is whatever ``--scheme`` selects.  An arm
    that hardcodes one differs from the control in the scheme AS WELL AS in
    the thing it is testing.  The downdraft pair is the deliberate exception:
    the ported unsaturated downdraft exists only in that one scheme, and both
    halves of the pair use it, so the pair is still internally controlled.
    """
    tree = ast.parse(_PROBE.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "make_physics_config"):
            continue
        for kw in node.keywords:
            if kw.arg != "convection":
                continue
            if isinstance(kw.value, ast.Constant) and kw.value.value != "emanuel":
                offenders.append((node.lineno, kw.value.value))
    assert not offenders, (
        f"arms hardcode a convection scheme at {offenders}; the control uses "
        "--scheme, so these arms change two variables at once")


def test_the_review_wrapper_exits_with_the_reviewers_status():
    body = _VERDICT.read_text()
    assert "exit $RC" in body, (
        "the wrapper prints the reviewer's exit status and then exits 0, so a "
        "failed or unauthenticated review is indistinguishable from a clean "
        "one to anything that reads the job state")
