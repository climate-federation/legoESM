"""Namespace-aware resolution of legoESM source paths for tests.

Source-introspection tests (AST guards, source-grep regressions, gap-marker
checks) need the on-disk path of a module's ``.py`` file.  They historically
hardcoded ``<repo>/src/legoesm/<sub>/<file>.py``, which assumes the whole
package lives under a single ``src/legoesm`` tree.

The federation carve (FEDERATION.md) splits the package into uv-workspace
members — each lives at ``packages/<member>/legoesm/<subpkg>`` (the substrate at
``packages/core/legoesm``, the components at ``packages/atmosphere/legoesm`` …),
while the root meta keeps its loose modules at ``src/legoesm`` — all merged into
one PEP-420 ``legoesm`` namespace.  Resolving through ``legoesm.__path__`` instead
of a hardcoded ``src/legoesm`` makes these tests location-independent: they pass
before the carve (single root) and after it (several roots), with no per-test
edits when a member is moved.
"""

from __future__ import annotations

import pathlib


def legoesm_root_paths(repo_root: pathlib.Path | None = None,
                       ) -> list[pathlib.Path]:
    """Every filesystem root of the ``legoesm`` namespace package.

    One entry today (``.../src/legoesm``); several once the carve lands
    (``.../src/legoesm`` + ``.../packages/<member>/src/legoesm``).

    ``repo_root`` makes this WORKTREE-TOLERANT (#1389).  ``legoesm.__path__``
    reports the paths of the *editable install*, which point at the canonical
    checkout — so a test running from a ``git worktree`` got roots outside its
    own tree and blew up in ``Path.relative_to`` with a bare ``ValueError``
    naming a foreign repo path.  Pass the caller's repo root and every root is
    remapped to the equivalent directory inside it when one exists; roots with
    no counterpart there are returned unchanged.
    """
    import legoesm

    roots = [pathlib.Path(p).resolve() for p in legoesm.__path__]
    if repo_root is None:
        return roots
    repo_root = pathlib.Path(repo_root).resolve()
    out: list[pathlib.Path] = []
    for root in roots:
        if root.is_relative_to(repo_root):
            out.append(root)
            continue
        # `.../src/legoesm` or `.../packages/<member>/legoesm`: try the
        # shortest tail that resolves inside this checkout.
        for depth in (2, 3, 1):
            if len(root.parts) < depth:
                continue
            candidate = repo_root.joinpath(*root.parts[-depth:])
            if candidate.is_dir():
                out.append(candidate)
                break
        else:
            out.append(root)
    return out


def legoesm_source_path(rel: str | pathlib.PurePath,
                        repo_root: pathlib.Path | None = None,
                        ) -> pathlib.Path:
    """Resolve a path *under* the ``legoesm`` package across all namespace roots.

    ``rel`` is the path beneath ``legoesm/`` — e.g. ``"runtime/backend.py"`` or
    ``"parallel/device_config.py"``.  A leading ``legoesm/`` or ``src/legoesm/``
    is accepted and stripped, so existing hardcoded strings can be passed almost
    verbatim.  Raises ``FileNotFoundError`` if no root contains it.
    """
    parts = pathlib.PurePosixPath(str(rel).replace("\\", "/")).parts
    if parts[:2] == ("src", "legoesm"):
        parts = parts[2:]
    elif parts[:1] == ("legoesm",):
        parts = parts[1:]
    sub = pathlib.Path(*parts)
    roots = legoesm_root_paths(repo_root)
    for root in roots:
        candidate = root / sub
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{rel!r} not found under any legoesm namespace root: "
        f"{[str(p) for p in roots]}"
    )


def legoesm_subpackages() -> set[str]:
    """Names of the top-level legoesm subpackages across all namespace roots.

    A subpackage is a non-underscore directory with an ``__init__.py`` in any
    namespace root (so the substrate packages remain visible after the carve
    relocates them out of ``src/legoesm``).
    """
    names: set[str] = set()
    for root in legoesm_root_paths():
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if (
                child.is_dir()
                and not child.name.startswith("_")
                and (child / "__init__.py").is_file()
            ):
                names.add(child.name)
    return names


def legoesm_loose_modules() -> set[str]:
    """Names of the top-level legoesm *modules* (loose ``.py`` files) across roots.

    A loose module is a non-underscore ``*.py`` file (other than ``__init__.py``)
    sitting directly in a namespace root — the orchestration/meta modules (cli,
    config, ...) and the substrate helper modules (constants, thermo, ...) that
    are not part of any subpackage.  The federation plan must still assign each
    of these to exactly one member, so this is the module-level counterpart of
    ``legoesm_subpackages`` (``test_federation_plan`` pins both).  Underscore
    modules (e.g. ``_version``) are private and excluded, mirroring the
    subpackage rule.
    """
    names: set[str] = set()
    for root in legoesm_root_paths():
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if (
                child.is_file()
                and child.suffix == ".py"
                and child.name != "__init__.py"
                and not child.name.startswith("_")
            ):
                names.add(child.stem)
    return names
