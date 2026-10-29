"""Fail closed on provenance SHAs in receipts added after the 2026-10-29 audit."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).parents[3]
CUTOFF = dt.date(2026, 10, 29)
HEX_RE = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{64}|[0-9A-Fa-f]{40})(?![0-9A-Fa-f])")
DOC_RE = re.compile(r"^docs/.*(?:receipt|prereg).*\.md$", re.IGNORECASE)
PATH_RE = re.compile(r"`([^`]+)`|(/[A-Za-z0-9_@%+=:,./~-]+)")


def _run(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def _added_after(repo: Path, cutoff: dt.date = CUTOFF) -> list[Path]:
    """Return still-present receipt/prereg Markdown added strictly after cutoff."""
    output = _run(
        "git",
        "log",
        "--diff-filter=A",
        "--format=@@%aI",
        "--name-only",
        "--",
        "docs",
        cwd=repo,
    ).stdout
    added: dict[str, dt.date] = {}
    stamp: dt.date | None = None
    for line in output.splitlines():
        if line.startswith("@@"):
            stamp = dt.datetime.fromisoformat(line[2:]).date()
        elif stamp is not None and DOC_RE.match(line):
            added[line] = stamp
    return sorted(
        repo / name for name, date in added.items() if date > cutoff and (repo / name).is_file()
    )


def _candidate_paths(
    repo: Path, doc: Path, lines: list[str], index: int, digest_column: int
) -> list[Path]:
    candidates: list[tuple[int, int, Path]] = []
    for distance in (0, 1, 2):
        for other in {index - distance, index + distance}:
            if not 0 <= other < len(lines):
                continue
            for match in PATH_RE.finditer(lines[other]):
                raw = (match.group(1) or match.group(2)).rstrip(".,;:)]}")
                if HEX_RE.fullmatch(raw) or not ("/" in raw or "." in Path(raw).name):
                    continue
                path = Path(raw)
                options = [path] if path.is_absolute() else [repo / path, doc.parent / path]
                for option in options:
                    if option.is_file():
                        gap = abs(match.start() - digest_column) if distance == 0 else 10_000
                        candidates.append((distance, gap, option.resolve()))
    if not candidates:
        return []
    best = min((distance, gap) for distance, gap, _ in candidates)
    return sorted({path for distance, gap, path in candidates if (distance, gap) == best})


def _failures(repo: Path, doc: Path) -> list[str]:
    lines = doc.read_text(errors="replace").splitlines()
    failures: list[str] = []
    for index, line in enumerate(lines):
        for match in HEX_RE.finditer(line):
            value = match.group().lower()
            label = f"{doc.relative_to(repo)}:{index + 1}"
            if len(value) == 40:
                result = _run("git", "cat-file", "-e", value, cwd=repo, check=False)
                if result.returncode:
                    failures.append(f"{label}: unresolved Git object {value}")
                continue
            paths = _candidate_paths(repo, doc, lines, index, match.start())
            if len(paths) != 1:
                failures.append(f"{label}: SHA-256 has {len(paths)} resolving cited files")
                continue
            actual = _run("sha256sum", "--", str(paths[0]), cwd=repo).stdout.split()[0]
            if actual != value:
                failures.append(f"{label}: SHA-256 mismatch for {paths[0]}: {value} != {actual}")
    return failures


def test_default_gate_checks_every_post_audit_receipt_and_preregistration():
    failures = [failure for doc in _added_after(REPO) for failure in _failures(REPO, doc)]
    assert failures == []


def _init_plant_repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "plant@example.invalid", cwd=repo)
    _run("git", "config", "user.name", "provenance plant", cwd=repo)
    payload = repo / "evidence.bin"
    payload.write_bytes(b"measured evidence\n")
    _run("git", "add", "evidence.bin", cwd=repo)
    _run("git", "commit", "-q", "-m", "plant evidence", cwd=repo)
    commit = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    doc = repo / "docs/ocean/fidelity/testcases/plant_receipt.md"
    doc.parent.mkdir(parents=True)
    doc.write_text(f"Commit `{commit}`. File `evidence.bin` SHA-256 `{digest}`.\n")
    return repo, doc, commit, digest


def test_planted_valid_git_object_and_file_digest_pass(tmp_path: Path):
    repo, doc, _, _ = _init_plant_repo(tmp_path)
    assert _failures(repo, doc) == []


def test_planted_unresolved_git_object_fails(tmp_path: Path):
    repo, doc, commit, _ = _init_plant_repo(tmp_path)
    bad = ("0" if commit[0] != "0" else "1") + commit[1:]
    doc.write_text(doc.read_text().replace(commit, bad))
    assert "unresolved Git object" in "\n".join(_failures(repo, doc))


def test_planted_wrong_file_digest_fails(tmp_path: Path):
    repo, doc, _, digest = _init_plant_repo(tmp_path)
    bad = ("0" if digest[0] != "0" else "1") + digest[1:]
    doc.write_text(doc.read_text().replace(digest, bad))
    assert "SHA-256 mismatch" in "\n".join(_failures(repo, doc))


def test_planted_digest_without_an_existing_cited_file_fails(
    tmp_path: Path,
):
    repo, doc, _, _ = _init_plant_repo(tmp_path)
    doc.write_text(doc.read_text().replace("evidence.bin", "missing.bin"))
    assert "0 resolving cited files" in "\n".join(_failures(repo, doc))
