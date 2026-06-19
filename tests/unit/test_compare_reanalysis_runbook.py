"""Doc-code consistency guard for the operator runbook.

``docs/compare_reanalysis_runbook.md`` gives an HPC operator the exact commands to
run the empirical demonstration.  A command that silently drifts out from under the
doc (a script renamed/moved, a flag removed) is worse than no runbook -- the operator
burns cluster time on a broken command.  These tests make the iter-237 manual
verification PERMANENT: every script the runbook cites exists, and every ``--flag`` it
documents is real.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]              # tests/unit/ -> repo root
_RUNBOOK = _REPO / "docs" / "compare_reanalysis_runbook.md"
# The campaign CLI the .sbatch launcher wraps: the runbook documents its knobs
# (--feedback-strategy, --iterations, …) even though it is invoked via sbatch, so its
# source is part of the flag-search set.
_CAMPAIGN_CLI = "scripts/run/run_correction_campaign.py"

_SCRIPT_RE = re.compile(r"scripts/[A-Za-z0-9_/]+\.(?:py|sbatch)")
_FLAG_RE = re.compile(r"--[a-z][a-z0-9-]+")


def _runbook_text() -> str:
    return _RUNBOOK.read_text()


def _referenced_scripts(text: str) -> list[str]:
    return sorted(set(_SCRIPT_RE.findall(text)))


def test_runbook_exists():
    assert _RUNBOOK.is_file(), f"missing operator runbook at {_RUNBOOK}"


def test_runbook_referenced_scripts_all_exist():
    paths = _referenced_scripts(_runbook_text())
    assert paths, "no script paths parsed from the runbook (regex/file broken?)"
    missing = [p for p in paths if not (_REPO / p).is_file()]
    assert not missing, f"runbook cites nonexistent script(s): {missing}"


def test_runbook_documented_flags_are_real():
    text = _runbook_text()
    scripts = _referenced_scripts(text)
    scripts.append(_CAMPAIGN_CLI)                        # the campaign CLI the launcher wraps
    sources = "\n".join(
        (_REPO / s).read_text() for s in scripts if (_REPO / s).is_file())
    flags = sorted(set(_FLAG_RE.findall(text)))
    assert flags, "no --flags parsed from the runbook"
    # Each documented flag must appear (as an argparse option or sbatch arg) in at
    # least one referenced script's source -- else the doc has drifted / has a typo.
    unreal = [f for f in flags if f not in sources]
    assert not unreal, (
        f"runbook documents flag(s) absent from every referenced script: {unreal}")
