"""Shared deprecation machinery for legacy script wrappers.

Each root-level ``scripts/run_*.py`` wrapper calls this before
delegating to ``scripts/legacy/``.  A single ``DeprecationWarning``
is emitted pointing the user to the legacy/ path.
"""

from __future__ import annotations

import runpy
import warnings
from pathlib import Path


def run_legacy(caller_file: str) -> None:
    """Emit a deprecation warning and delegate to the legacy script.

    Parameters
    ----------
    caller_file : str
        ``__file__`` of the wrapper script.
    """
    caller = Path(caller_file).resolve()
    name = caller.name
    target = caller.parent / "legacy" / name

    if not target.exists():
        raise FileNotFoundError(
            f"Legacy script not found: {target}"
        )

    warnings.warn(
        f"scripts/{name} is a deprecated wrapper. "
        f"Run scripts/legacy/{name} directly instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    runpy.run_path(str(target), run_name="__main__")
