"""Enforce the import-boundary contracts — the path to independent packages.

The contracts in ``pyproject.toml`` ([tool.importlinter]) encode the target
layered architecture (core < components < coupler < driver) that lets each
Earth-system component become an independently installable, self-running package
(``pip install legoesm-ocean``).  The current cross-boundary coupling is captured
as a ``ignore_imports`` baseline; this test fails if a NEW violation is added
(ratchet), and every baseline entry removed is a real step toward independence.

It is a real (subprocess) CI gate, but skips cleanly when import-linter isn't
installed (it lives in the ``dev`` extra).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_LINT_IMPORTS = Path(sys.executable).parent / "lint-imports"


@pytest.mark.skipif(
    not _LINT_IMPORTS.exists(),
    reason="import-linter not installed (pip install -e '.[dev]')",
)
def test_import_linter_contracts_hold() -> None:
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    # --no-cache: the gate must test import boundaries only, never depend on the
    # cache directory being writable (read-only / sandboxed checkouts).
    result = subprocess.run(
        [str(_LINT_IMPORTS), "--no-cache"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, (
        "import-boundary contracts broken — a new cross-package import was added "
        "that breaks component independence. Either remove it, or (if it is a "
        "tracked, intentional coupling) add it to the matching contract's "
        "ignore_imports baseline in pyproject.toml.\n\n"
        + result.stdout
        + result.stderr
    )
