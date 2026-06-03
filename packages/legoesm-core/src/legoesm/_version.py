"""Single source of truth for the installed legoESM version.

The version literal lives in exactly one place — ``pyproject.toml``'s
``[project].version`` — and is read back here.  Resolution order is chosen so
the reported version always describes *the code actually executing*:

1. **Source-tree ``pyproject.toml``** adjacent to this very file (walk upward to
   the ``legoesm`` ``pyproject.toml``).  This is the authoritative descriptor of
   the running code and is preferred first because it is correct in two cases
   installed metadata gets wrong: an editable install whose metadata was not
   refreshed after a version bump, and a bare checkout (``PYTHONPATH=src``) where
   ``importlib.metadata`` might resolve a *different*, stale ``legoesm``
   distribution installed elsewhere in the environment.
2. **Installed package metadata** (``importlib.metadata.version``) — used for a
   normal wheel install, where no source-tree ``pyproject.toml`` sits next to the
   package and metadata is the canonical source.
3. **Sentinel** (``0.0.0+unknown``) — only when *neither* source resolves, so the
   value is never silently wrong.

Every other module imports ``__version__`` from here (or from :mod:`legoesm`),
so the string is never duplicated across the codebase.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

_SENTINEL = "0.0.0+unknown"


def _version_from_source_tree() -> str | None:
    """Read ``[project].version`` from the nearest ancestor ``pyproject.toml``.

    Walks upward from this file looking for a ``pyproject.toml`` whose
    ``[project].name`` is ``legoesm``.  Returns ``None`` if none is found or it
    cannot be parsed — callers fall back to the sentinel.
    """
    import tomllib

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        if not candidate.is_file():
            continue
        try:
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        project = data.get("project", {})
        if project.get("name") == "legoesm" and "version" in project:
            return str(project["version"])
    return None


def _resolve_version() -> str:
    """Resolve the version of the code being executed (see module docstring)."""
    # 1. Prefer the source-tree pyproject adjacent to this package — it describes
    #    the running code, unlike installed metadata which may be stale or point
    #    at a different installed distribution.
    from_source = _version_from_source_tree()
    if from_source is not None:
        return from_source
    # 2. Installed wheel (no adjacent pyproject) — metadata is canonical.
    try:
        return _pkg_version("legoesm")
    except PackageNotFoundError:
        # 3. Neither resolves — never guess a number that could drift.
        return _SENTINEL


__version__ = _resolve_version()

__all__ = ["__version__"]
