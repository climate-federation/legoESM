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
MINIMUM_DOCUMENTS = 1389


def run(*args: str, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True)


def documents() -> tuple[Path, ...]:
    return tuple(sorted(path for path in (ROOT / "docs").rglob("*.md") if DOC_RE.search(path.name)))


def post_cutoff_changes() -> dict[Path, set[int]]:
    """Map current lines added or changed after the cutoff to their documents."""
    base = run("git", "rev-list", "-1", f"--before={CUTOFF}", "HEAD").stdout.strip()
    if not base:
        raise AssertionError(f"no repository commit exists before {CUTOFF}")
    diff = run("git", "diff", "--unified=0", base, "HEAD", "--", "docs").stdout
    changed: dict[Path, set[int]] = {}
    path: Path | None = None
    current_line = 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = ROOT / line[6:]
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            assert match
            current_line = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            if path is not None and DOC_RE.search(path.name):
                changed.setdefault(path, set()).add(current_line)
            current_line += 1
        elif not line.startswith("-"):
            current_line += 1
    return changed


POST_CUTOFF_CHANGES = post_cutoff_changes()


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


def validate_document(path: Path, selected: set[int] | None = None) -> list[str]:
    lines = path.read_text(errors="replace").splitlines()
    selected = POST_CUTOFF_CHANGES.get(path, set()) if selected is None else selected
    failures: list[str] = []
    for line_number in sorted(selected):
        line = lines[line_number - 1]
        for match in HEX_RE.finditer(line):
            value = match.group().lower()
            if len(value) == 40:
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
