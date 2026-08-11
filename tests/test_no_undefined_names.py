"""Undefined names (ruff F821) must not reach a live code path.

WHY THIS EXISTS
---------------
On 2026-08-11 a refactor of the inertia-gravity-wave gate in
``scripts/matrix/run_ocean_test_matrix.py`` moved two metrics into a helper
and, in doing so, deleted

    amp_upper = _IGW_AMP_UPPER
    corr_upper = _IGW_IC_CORR_UPPER

while leaving both names in use three lines below -- a guaranteed
``NameError`` the moment that case ran. It survived hours and several
sbatch submissions, because the driver takes ``--only <case>`` and the
reruns in between selected other cases. On this cluster that failure mode
costs ~30 minutes per wasted submission.

Nothing caught it, and NOT because the tooling is missing: ``pyproject``
already selects ruff's ``F`` family repo-wide and ``F821`` flags exactly
that deletion. The gap is that CI Actions have been disabled repo-wide
since 2026-05-27, so the configured linter never runs. This test makes it
execute as part of the ordinary test suite.

WHAT IT DOES NOT COVER (GLM-5.2 2026-08-11) -- F821 is lexical scope
resolution, not reachability, so it does NOT catch:
  * a name bound only inside an ``if``/``else`` branch but used
    unconditionally after it;
  * a name bound in a loop body that may run zero times;
  * the nastiest one -- a deleted local whose name still resolves to a
    module-level global, which does not raise at all and silently uses the
    WRONG value.
The third is why the ocean matrix's IGW gate now uses its module constants
INLINE rather than through local aliases: that removes the failure mode
structurally instead of trying to detect it. Closing the first two needs
pyright's ``reportPossiblyUnbound``, which is a project, not a guard.

The real complement to this test is a registry smoke test that actually
CALLS each case with a stubbed backend -- see
``tests/ocean/unit/test_ocean_case_registry_smoke.py``.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

#: Files with a KNOWN pre-existing F821, each a latent NameError that this
#: test found and that is not in scope to fix here. SHRINK-ONLY: never add
#: to this list to make a new failure pass -- fix the name instead.
#:
#: scripts/run/run_rcemip_plane.py:148 uses ``WING_T_V0`` and ``WING_GAMMA``,
#: neither of which is defined anywhere in the file (found 2026-08-11).
_KNOWN_F821 = {
    "scripts/run/run_rcemip_plane.py",
}


def _ruff() -> list[str]:
    exe = shutil.which("ruff")
    return [exe] if exe else [sys.executable, "-m", "ruff"]


def _run_f821(targets: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        _ruff() + ["check", "--select", "F821", "--no-cache",
                   "--output-format", "concise", *targets],
        cwd=_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


@pytest.mark.lint
def test_no_new_undefined_names_anywhere():
    """Repo-wide, minus a shrink-only baseline.

    Repo-wide rather than a curated path list, because a curated list rots
    and stops covering the file that matters (GLM-5.2). ``# noqa: F821`` is
    deliberately still honoured -- F821 does have false positives on
    globals()-injected names, plugin attributes and TYPE_CHECKING-only
    symbols, and a guard that cannot be suppressed gets deleted rather
    than fixed.
    """
    proc = _run_f821([str(_REPO)])
    if proc.returncode not in (0, 1):
        pytest.skip(f"ruff unavailable: {proc.stdout[:200]}")
    offenders = {}
    for line in proc.stdout.splitlines():
        if "F821" not in line:
            continue
        path = line.split(":", 1)[0].strip()
        try:
            rel = str(Path(path).resolve().relative_to(_REPO))
        except ValueError:
            rel = path
        offenders.setdefault(rel, []).append(line.strip())
    new = {f: v for f, v in offenders.items() if f not in _KNOWN_F821}
    assert not new, (
        "undefined name(s) on a live code path -- a NameError waiting to "
        "fire:\n" + "\n".join(l for v in new.values() for l in v))


@pytest.mark.lint
def test_the_check_is_not_vacuous(tmp_path):
    """Prove ruff actually reports F821, so a green run above means
    "clean" and not "the linter silently did nothing".

    Uses a SYNTHETIC file rather than reconstructing the historical bug:
    a probe keyed to real variable names breaks the first time someone
    renames them, which is maintenance load for no extra confidence
    (GLM-5.2).
    """
    probe = tmp_path / "probe.py"
    probe.write_text("def f():\n    return _definitely_not_defined_anywhere\n")
    proc = _run_f821([str(probe)])
    if proc.returncode not in (0, 1):
        pytest.skip(f"ruff unavailable: {proc.stdout[:200]}")
    assert proc.returncode == 1 and "F821" in proc.stdout, (
        f"ruff did not flag an obviously undefined name:\n{proc.stdout}")
