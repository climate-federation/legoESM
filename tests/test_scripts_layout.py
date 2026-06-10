"""CI ratchet: ``scripts/`` keeps its sanctioned bucket layout (CLAUDE.md).

Doctrine (File Layout): every script lives in one of the sanctioned buckets —
``run/ matrix/ bench/ plot/ validate/ data/ experiment/ cluster/`` (+ ``tmp/``
for throwaway probes). Nothing else at ``scripts/`` root except ``README.md``
and ``__init__.py``. This went unenforced and 11 unsanctioned campaign dirs
accumulated; they are seeded into the shrink-only ``LEGACY_DIRS_TODO``
allowlist below and re-homed incrementally (audit-eng-practices branch).

Ratchet is two-way, mirroring ``tests/_ratchet_audit.py`` doctrine:
  * an unsanctioned entry NOT in the allowlist fails (no new drift), and
  * an allowlist entry that no longer exists fails (forces the shrink to be
    recorded — budgets only go down, never silently linger).
Anti-vacuity: the checker is a pure function exercised against synthetic
listings with known violations, so a refactor that makes the scan vacuous
goes red here, not silently green.
"""

from __future__ import annotations

import pathlib

from tests._ratchet_audit import repo_root

# CLAUDE.md File Layout buckets. Adding a bucket here requires a real category
# (and a CLAUDE.md + scripts/README.md update) — not a dumping ground.
SANCTIONED_BUCKETS = frozenset(
    {
        "run",
        "matrix",
        "bench",
        "plot",
        "validate",
        "data",
        "experiment",
        "cluster",
        "tmp",
    }
)

# Non-bucket files permitted at scripts/ root.
SANCTIONED_ROOT_FILES = frozenset({"README.md", "__init__.py"})

# Shrink-only: legacy dirs predating this ratchet, queued for re-homing.
# Remove an entry the moment its dir is moved/deleted — a stale entry fails.
# NEVER add to this set; a new unsanctioned dir must go straight to a bucket.
LEGACY_DIRS_TODO = frozenset(
    {
        "ocean_long_runs",
        "ocean_test_matrix",
        "pgf_validation",
        "s2s",
        "scm",
    }
)

# Shrink-only: legacy root files queued for disposal (currently none).
LEGACY_ROOT_FILES_TODO = frozenset()

_IGNORED = {"__pycache__"}


def _scripts_dir() -> pathlib.Path:
    return repo_root() / "scripts"


def list_scripts_entries(scripts_dir: pathlib.Path) -> list[tuple[str, bool]]:
    """``(name, is_dir)`` for every non-hidden, non-ignored scripts/ entry."""
    out: list[tuple[str, bool]] = []
    for p in sorted(scripts_dir.iterdir()):
        if p.name.startswith(".") or p.name in _IGNORED:
            continue
        out.append((p.name, p.is_dir()))
    return out


def layout_violations(entries: list[tuple[str, bool]]) -> list[str]:
    """Pure checker (synthetically testable): violations for one listing.

    Two-way: flags (a) unsanctioned entries not in the legacy allowlists and
    (b) allowlist entries that no longer exist (ratchet down).
    """
    names = {name for name, _ in entries}
    errors: list[str] = []
    for name, is_dir in entries:
        if is_dir:
            if name in SANCTIONED_BUCKETS:
                continue
            if name in LEGACY_DIRS_TODO:
                continue
            errors.append(
                f"unsanctioned scripts/ dir: {name}/ — new scripts go in a "
                f"sanctioned bucket {sorted(SANCTIONED_BUCKETS)} (CLAUDE.md "
                f"File Layout); do NOT extend LEGACY_DIRS_TODO"
            )
        else:
            if name in SANCTIONED_ROOT_FILES or name in LEGACY_ROOT_FILES_TODO:
                continue
            errors.append(
                f"stray file at scripts/ root: {name} — only "
                f"{sorted(SANCTIONED_ROOT_FILES)} allowed; debug/one-off "
                f"scripts go in scripts/tmp/"
            )
    for name in sorted(LEGACY_DIRS_TODO - names):
        errors.append(
            f"stale LEGACY_DIRS_TODO entry: {name} — dir is gone, remove the "
            f"allowlist entry (shrink-only ratchet)"
        )
    for name in sorted(LEGACY_ROOT_FILES_TODO - names):
        errors.append(
            f"stale LEGACY_ROOT_FILES_TODO entry: {name} — file is gone, "
            f"remove the allowlist entry (shrink-only ratchet)"
        )
    return errors


def test_scripts_layout_ratchet() -> None:
    scripts_dir = _scripts_dir()
    assert scripts_dir.is_dir(), f"scripts/ missing at {scripts_dir} (vacuous scan)"
    entries = list_scripts_entries(scripts_dir)
    # Anti-vacuity: a healthy tree has the core buckets present.
    names = {name for name, is_dir in entries if is_dir}
    core = {"run", "matrix", "bench", "plot", "validate", "data"}
    assert core <= names, (
        f"scripts/ listing missing core buckets {sorted(core - names)} — "
        f"scan mis-resolved, ratchet would be vacuous"
    )
    errors = layout_violations(entries)
    assert not errors, "scripts/ layout drift:\n" + "\n".join(errors)


def test_layout_checker_flags_synthetic_violations() -> None:
    """Self-test (non-vacuous): known violations must trip the checker."""
    entries = [
        ("run", True),  # sanctioned bucket — ok
        ("scm", True),  # legacy allowlisted — ok
        ("my_new_campaign", True),  # unsanctioned dir — must flag
        ("_scratch_debug.sbatch", False),  # stray root file — must flag
        ("README.md", False),  # sanctioned root file — ok
    ]
    errors = layout_violations(entries)
    assert any("my_new_campaign" in e for e in errors), errors
    assert any("_scratch_debug.sbatch" in e for e in errors), errors
    # And the sanctioned/allowlisted entries must NOT be flagged.
    assert not any("scm" in e and "stale" not in e for e in errors), errors
    # Missing allowlist entries (everything else in LEGACY_DIRS_TODO) are
    # reported stale on this synthetic listing — that IS the two-way ratchet.
    stale = [e for e in errors if "stale LEGACY_DIRS_TODO" in e]
    assert len(stale) == len(LEGACY_DIRS_TODO) - 1, errors


def test_layout_checker_passes_sanctioned_listing() -> None:
    """Self-test: a fully sanctioned listing (all legacy present) is clean."""
    entries = [(b, True) for b in sorted(SANCTIONED_BUCKETS)]
    entries += [(d, True) for d in sorted(LEGACY_DIRS_TODO)]
    entries += [(f, False) for f in sorted(SANCTIONED_ROOT_FILES)]
    assert layout_violations(entries) == []
