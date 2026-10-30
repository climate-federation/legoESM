"""Prospective provenance gate for receipt and preregistration SHA citations."""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
CUTOFF = dt.datetime(2026, 10, 30, tzinfo=dt.timezone.utc).isoformat()
HEX_RE = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{64}|[0-9A-Fa-f]{40})(?![0-9A-Fa-f])")
DOC_RE = re.compile(r"(?i)(?:receipt|prereg).*\.md$")
PATH_RE = re.compile(r"`([^`]+)`|(/[A-Za-z0-9_@%+=:,./~-]+)")
SUPERSESSION_RE = re.compile(
    r"^PROVENANCE-SUPERSESSION: "
    r"(?P<old>[0-9A-Fa-f]{40}) -> blob (?P<blob>[0-9A-Fa-f]{40}) "
    r"`(?P<path>[^`\n]+)`$"
)
MINIMUM_DOCUMENTS = 1389
POST_CUTOFF_ORIGINS: dict[Path, dict[int, tuple[str, int]]] = {}
_MERGE_PARENT_LINES: dict[tuple[str, Path], list[str] | None] = {}


def run(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True)


def documents() -> tuple[Path, ...]:
    return tuple(sorted(path for path in (ROOT / "docs").rglob("*.md") if DOC_RE.search(path.name)))


def post_cutoff_changes() -> dict[Path, set[int]]:
    """Map first-parent prospective lines to their current documents.

    A catch-up merge imports an independently authored history.  A plain
    ``base..HEAD`` tree diff would relabel every imported second-parent line as
    a new lane citation, even though those lines were not authored or changed
    on the guarded lane.  Follow the first-parent chain, inspect non-merge
    commits plus combined merge resolutions, then use blame to retain only
    lines actually introduced by those prospective first-parent commits.
    """
    base = run(
        "git", "rev-list", "--first-parent", "-1", f"--before={CUTOFF}",
        "HEAD").stdout.strip()
    if not base:
        raise AssertionError(f"no repository commit exists before {CUTOFF}")

    prospective = set(run(
        "git", "rev-list", "--first-parent", f"{base}..HEAD").stdout.splitlines())
    candidate_names = set(run(
        "git", "log", "--first-parent", "--no-merges", "--format=",
        "--name-only", f"{base}..HEAD", "--", "docs").stdout.splitlines())
    merges = run(
        "git", "rev-list", "--first-parent", "--merges",
        f"{base}..HEAD").stdout.splitlines()
    for merge in merges:
        candidate_names.update(run(
            "git", "diff-tree", "--cc", "--name-only", "--format=", merge,
            "--", "docs").stdout.splitlines())

    changed: dict[Path, set[int]] = {}
    header = re.compile(
        r"^(?P<sha>[0-9a-f]{40}) (?P<origin>\d+) "
        r"(?P<line>\d+)(?: \d+)?$")
    for name in sorted(candidate_names):
        if not name:
            continue
        path = ROOT / name
        if not path.is_file() or not DOC_RE.search(path.name):
            continue
        blame = run(
            "git", "blame", "--line-porcelain", "--root", "HEAD", "--",
            name).stdout
        selected: set[int] = set()
        origins: dict[int, tuple[str, int]] = {}
        for line in blame.splitlines():
            match = header.fullmatch(line)
            if match is None or match.group("sha") not in prospective:
                continue
            line_number = int(match.group("line"))
            selected.add(line_number)
            origins[line_number] = (
                match.group("sha"), int(match.group("origin")))
        if selected:
            changed[path] = selected
            POST_CUTOFF_ORIGINS[path] = origins
    return changed


POST_CUTOFF_CHANGES = post_cutoff_changes()


def inherited_merge_sha(path: Path, line_number: int, value: str) -> bool:
    """Whether a merge re-anchor carried this SHA token from parent one.

    Only a token on the same original line of a merge's first parent is
    carried forward.  New tokens, copied tokens, and ordinary post-cutoff
    commits remain subject to the normal resolution/digest checks.
    """
    origin = POST_CUTOFF_ORIGINS.get(path, {}).get(line_number)
    if origin is None:
        return False
    commit, original_line = origin
    parents = run(
        "git", "rev-list", "--parents", "-n", "1", commit).stdout.split()
    if len(parents) < 3:
        return False
    relative = path.relative_to(ROOT)
    key = (commit, relative)
    if key not in _MERGE_PARENT_LINES:
        shown = run(
            "git", "show", f"{commit}^1:{relative}", check=False)
        _MERGE_PARENT_LINES[key] = (
            shown.stdout.splitlines() if shown.returncode == 0 else None)
    lines = _MERGE_PARENT_LINES[key]
    return (
        lines is not None
        and 1 <= original_line <= len(lines)
        and value in HEX_RE.findall(lines[original_line - 1])
    )


def cited_path(path: Path, lines: list[str], index: int, start: int) -> Path | None:
    candidates: list[tuple[int, int, Path]] = []
    for offset in (0, -1, 1):
        other = index + offset
        if not 0 <= other < len(lines):
            continue
        for match in PATH_RE.finditer(lines[other]):
            raw_text = (match.group(1) or match.group(2)).rstrip(".,;:)]}")
            if HEX_RE.fullmatch(raw_text) or ("/" not in raw_text and "." not in Path(raw_text).name):
                continue
            raw = Path(raw_text)
            options = [raw] if raw.is_absolute() else [ROOT / raw, path.parent / raw]
            resolved = {option.resolve() for option in options if option.is_file()}
            if len(resolved) == 1:
                gap = abs(start - match.end()) if offset == 0 else 10000
                candidates.append((abs(offset), gap, resolved.pop()))
    if not candidates:
        return None
    best = min((distance, gap) for distance, gap, _ in candidates)
    paths = {candidate for distance, gap, candidate in candidates if (distance, gap) == best}
    return paths.pop() if len(paths) == 1 else None


def superseded_git_shas(lines: list[str]) -> tuple[set[str], list[str]]:
    """Validate append-only commit-to-file-content supersessions.

    A receipt cannot rewrite its historical paragraph after a rebase.  This
    record form retires that paragraph's round-local commit identity in favour
    of a blob that is reachable from ``HEAD`` at the named path.  Merely naming
    an arbitrary object, an unreachable blob, or a different path cannot make
    an unresolved SHA pass.
    """
    superseded: set[str] = set()
    failures: list[str] = []
    for line_number, line in enumerate(lines, 1):
        if not line.startswith("PROVENANCE-SUPERSESSION:"):
            continue
        match = SUPERSESSION_RE.fullmatch(line)
        if match is None:
            failures.append(f"line {line_number}: malformed provenance supersession")
            continue
        old = match.group("old").lower()
        blob = match.group("blob").lower()
        rel = Path(match.group("path"))
        if rel.is_absolute() or ".." in rel.parts:
            failures.append(f"line {line_number}: supersession path must be repository-relative")
            continue
        kind = run("git", "cat-file", "-t", blob, check=False)
        if kind.returncode != 0 or kind.stdout.strip() != "blob":
            failures.append(f"line {line_number}: supersession target is not a reachable Git blob")
            continue
        objects = run("git", "rev-list", "--objects", "HEAD", "--", str(rel)).stdout.splitlines()
        if not any(row.split(" ", 1)[0].lower() == blob for row in objects):
            failures.append(
                f"line {line_number}: blob {blob} is not reachable from HEAD at {rel}")
            continue
        superseded.add(old)
    return superseded, failures


def validate_document(path: Path, selected: set[int] | None = None) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    selected = POST_CUTOFF_CHANGES.get(path, set()) if selected is None else selected
    superseded, failures = superseded_git_shas(lines)
    for line_number in sorted(selected):
        line = lines[line_number - 1]
        for match in HEX_RE.finditer(line):
            value = match.group().lower()
            if inherited_merge_sha(path, line_number, value):
                continue
            if len(value) == 40:
                if value in superseded:
                    continue
                found = run("git", "cat-file", "-e", value, check=False).returncode == 0
                if not found:
                    failures.append(f"{path}:{line_number}: unresolved Git SHA {value}")
                continue
            target = cited_path(path, lines, line_number - 1, match.start())
            if target is None:
                failures.append(f"{path}:{line_number}: SHA-256 has no unambiguous existing cited file")
                continue
            actual = run("sha256sum", "--", str(target)).stdout.split()[0].lower()
            if actual != value:
                failures.append(f"{path}:{line_number}: SHA-256 mismatch for {target}")
    return failures


def test_receipt_and_preregistration_discovery_does_not_shrink() -> None:
    assert len(documents()) >= MINIMUM_DOCUMENTS


@pytest.mark.parametrize("document", documents(), ids=lambda path: str(path.relative_to(ROOT)))
def test_post_cutoff_sha_citations_resolve_per_document(document: Path) -> None:
    assert validate_document(document) == []


def test_planted_unresolved_git_sha_fails(tmp_path: Path) -> None:
    document = tmp_path / "plant_receipt.md"
    document.write_text("commit `0000000000000000000000000000000000000000`\n")
    assert "unresolved Git SHA" in validate_document(document, {1})[0]


def test_planted_wrong_file_sha256_fails(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"real bytes")
    document = tmp_path / "plant_preregister.md"
    document.write_text(f"`{artifact}` SHA-256 `{'0' * 64}`\n")
    assert "SHA-256 mismatch" in validate_document(document, {1})[0]


def test_real_file_sha256_plant_control_passes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"real bytes")
    digest = run("sha256sum", "--", str(artifact)).stdout.split()[0]
    document = tmp_path / "control_receipt.md"
    document.write_text(f"`{artifact}` SHA-256 `{digest}`\n")
    assert validate_document(document, {1}) == []


def test_reachable_blob_supersedes_an_unresolved_commit_append_only(tmp_path: Path) -> None:
    document = tmp_path / "amended_receipt.md"
    blob = run("git", "rev-parse", "HEAD:CLAUDE.md").stdout.strip()
    old = "0" * 40
    document.write_text(
        f"historical commit `{old}`\n"
        f"PROVENANCE-SUPERSESSION: {old} -> blob {blob} `CLAUDE.md`\n"
    )
    assert validate_document(document, {1, 2}) == []


def test_wrong_path_cannot_supersede_an_unresolved_commit(tmp_path: Path) -> None:
    document = tmp_path / "bad_amendment_receipt.md"
    blob = run("git", "rev-parse", "HEAD:CLAUDE.md").stdout.strip()
    old = "0" * 40
    document.write_text(
        f"historical commit `{old}`\n"
        f"PROVENANCE-SUPERSESSION: {old} -> blob {blob} `README.md`\n"
    )
    failures = validate_document(document, {1, 2})
    assert any("not reachable from HEAD at README.md" in failure for failure in failures)
    assert any("unresolved Git SHA" in failure for failure in failures)
