"""Guard the single-source-of-truth version invariant (Stage A0).

**Enforced surface — the machine-read provenance files** that determine what a
build, an install, or a running process *reports* as its version:

* ``pyproject.toml`` (``[project].version``) — the canonical build source.
* ``CITATION.cff`` (``version``) — duplicated only because the CFF schema cannot
  reference an external value; this test keeps it in lockstep.
* ``uv.lock`` (the ``legoesm`` root entry) — committed lock metadata.
* ``src/legoesm/**.py`` — must contain *no* version literal; the runtime value is
  read from metadata/pyproject via :mod:`legoesm._version` (no hand-copies).

If any of these drift, a release ships inconsistent provenance, so they fail
fast here (a clear ``AssertionError``, not a confusing downstream mismatch).

**Deliberately out of scope:** *illustrative* version strings inside
documentation (e.g. an example ``pyproject`` snippet in ``docs/specs/`` or the
generated ``docs/*.html``) and throwaway agent-review notes.  Those are
human-readable examples, not machine-read provenance; forcing them through this
guard would require allowlisting essentially every file that contains an example
and would churn generated artifacts.  A doc example showing an old version
misleads nobody about the *shipped* version, which is single-sourced above.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _citation_version() -> str:
    with open(REPO_ROOT / "CITATION.cff") as f:
        data = yaml.safe_load(f)
    # CFF stores version as a string; PyYAML may parse "0.1.0" fine, but a value
    # like 1.2 would become a float — normalise to str for a literal compare.
    return str(data["version"])


def test_pyproject_and_citation_versions_match() -> None:
    """The two declarative version literals must be identical."""
    assert _pyproject_version() == _citation_version(), (
        "pyproject.toml [project].version and CITATION.cff version disagree — "
        "bump both (single source of truth)."
    )


def test_runtime_version_matches_pyproject() -> None:
    """``legoesm.__version__`` (from installed metadata) tracks pyproject.

    Skipped only when running from a bare, un-installed source tree, where
    ``importlib.metadata`` cannot resolve the package and ``_version`` falls back
    to a clearly-fake sentinel.
    """
    # ``legoesm`` is a PEP-420 namespace package (federation carve) with no
    # ``__init__``, so the version lives on the single-source ``_version`` module.
    from legoesm._version import __version__

    if __version__.endswith("+unknown"):
        pytest.skip("legoesm not installed (bare source tree); metadata version unavailable")
    assert __version__ == _pyproject_version(), (
        f"Installed legoesm version {__version__!r} != pyproject "
        f"{_pyproject_version()!r}; reinstall with `pip install -e .` after a bump."
    )


# A user-facing version banner: "legoESM v" immediately followed by a digit means
# a *literal* version was baked in.  The single-sourced form interpolates
# (``legoESM v{__version__}``), so a brace — not a digit — follows the "v".
_BANNER_LITERAL = re.compile(r"legoESM\s+v\d")


def test_no_hardcoded_version_literal_in_source() -> None:
    """No ``.py`` under ``src/legoesm`` may embed the project version literal.

    The version must always be read from installed metadata via
    :mod:`legoesm._version`.  Two failure modes are caught:

    1. A version *banner* with any baked-in number (``legoESM v0.1.0``).  This
       catches a literal even after a bump makes it *stale* (the regex keys on
       the ``legoESM v<digit>`` shape, not on the current number) — the gap an
       earlier version of this guard had.
    2. The *current* project version appearing as a literal anywhere else.
    """
    version = _pyproject_version()
    pkg_root = REPO_ROOT / "src" / "legoesm"
    offenders: list[str] = []
    for path in pkg_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _BANNER_LITERAL.search(line) or version in line:
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Hardcoded version literal found in source (interpolate "
        "legoesm._version.__version__ instead):\n" + "\n".join(offenders)
    )


def test_banner_literal_regex_catches_stale_versions() -> None:
    """Regression proof the guard flags a stale banner even off the current bump."""
    assert _BANNER_LITERAL.search("legoESM v0.1.0")  # current
    assert _BANNER_LITERAL.search("legoESM v0.2.0")  # stale-after-bump (the missed case)
    assert _BANNER_LITERAL.search("legoESM v10.3.1")
    # The single-sourced, interpolated form must NOT trip the guard.
    assert not _BANNER_LITERAL.search("legoESM v{__version__}")
    assert not _BANNER_LITERAL.search('f"legoESM v{__version__} | Benchmark"')


def test_uv_lock_root_package_version_matches() -> None:
    """The ``legoesm`` root entry in ``uv.lock`` must track pyproject.

    The lockfile is committed, so a bump that touches only pyproject/CITATION
    would leave it carrying a stale package version for anyone resolving from the
    lock — another version-skew path this suite closes.
    """
    lock_path = REPO_ROOT / "uv.lock"
    if not lock_path.is_file():
        pytest.skip("uv.lock not present")
    with open(lock_path, "rb") as f:
        lock = tomllib.load(f)
    entries = [p for p in lock.get("package", []) if p.get("name") == "legoesm"]
    assert entries, "no `legoesm` package entry found in uv.lock"
    for entry in entries:
        assert entry.get("version") == _pyproject_version(), (
            f"uv.lock legoesm version {entry.get('version')!r} != pyproject "
            f"{_pyproject_version()!r}; regenerate the lockfile after a bump."
        )


def test_source_tree_fallback_resolves_pyproject_version() -> None:
    """The not-installed fallback reads the real version, not the sentinel.

    Guards the provenance path: running from a bare checkout, restart/checkpoint
    metadata (``model_version=None`` -> ``legoesm._version.__version__``) must
    still record the canonical pyproject version rather than ``0.0.0+unknown``.
    """
    from legoesm import _version

    resolved = _version._version_from_source_tree()
    assert resolved == _pyproject_version(), (
        f"source-tree fallback resolved {resolved!r}, expected pyproject "
        f"{_pyproject_version()!r}"
    )
    assert resolved != "0.0.0+unknown"


def test_source_tree_version_wins_over_stale_installed_metadata(monkeypatch) -> None:
    """Public version must not go stale when installed metadata disagrees.

    Simulates a checkout running while ``importlib.metadata`` resolves an older
    ``legoesm`` distribution elsewhere in the environment.  Resolution must
    prefer the source-tree pyproject (the code actually executing).
    """
    from importlib.metadata import PackageNotFoundError

    from legoesm import _version

    monkeypatch.setattr(_version, "_pkg_version", lambda _name: "0.0.1-stale")
    assert _version._resolve_version() == _pyproject_version()

    # No source tree (real wheel) -> metadata is canonical.
    monkeypatch.setattr(_version, "_version_from_source_tree", lambda: None)
    monkeypatch.setattr(_version, "_pkg_version", lambda _name: "9.9.9")
    assert _version._resolve_version() == "9.9.9"

    # Neither resolves -> sentinel, never a guessed number.
    def _raise(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(_version, "_pkg_version", _raise)
    assert _version._resolve_version() == "0.0.0+unknown"


def test_citation_cff_is_valid_if_cffconvert_available() -> None:
    """Validate CITATION.cff against the CFF schema when cffconvert is present.

    cffconvert is not a hard dependency; this is a best-effort gate that runs in
    environments (and CI) where it is installed.
    """
    cffconvert = pytest.importorskip("cffconvert")  # noqa: F841
    from cffconvert.cli.create_citation import create_citation

    citation = create_citation(str(REPO_ROOT / "CITATION.cff"), None)
    # Raises on schema violations; returns a (valid?, message) on some versions.
    citation.validate()
