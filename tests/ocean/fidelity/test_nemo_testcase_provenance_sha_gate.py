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


def _git_commit_failure(repo: Path, value: str, label: str) -> str | None:
    resolved = _run("git", "cat-file", "-e", f"{value}^{{commit}}", cwd=repo, check=False)
    if resolved.returncode:
        return f"{label}: unresolved Git commit {value}"

    ancestor = _run(
        "git", "merge-base", "--is-ancestor", value, "HEAD", cwd=repo, check=False
    )
    if ancestor.returncode == 0:
        return None

    keep_refs = _run(
        "git", "for-each-ref", "--format=%(objectname)", "refs/keep/", cwd=repo
    ).stdout.splitlines()
    if value in keep_refs:
        return None
    return (
        f"{label}: Git commit {value} is neither an ancestor of HEAD "
        "nor pinned under refs/keep/*"
    )


def _failures(repo: Path, doc: Path) -> list[str]:
    lines = doc.read_text(errors="replace").splitlines()
    failures: list[str] = []
    for index, line in enumerate(lines):
        for match in HEX_RE.finditer(line):
            value = match.group().lower()
            label = f"{doc.relative_to(repo)}:{index + 1}"
            if len(value) == 40:
                failure = _git_commit_failure(repo, value, label)
                if failure is not None:
                    failures.append(failure)
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


def test_planted_ancestor_and_file_digest_pass(tmp_path: Path):
    repo, doc, _, _ = _init_plant_repo(tmp_path)
    assert _failures(repo, doc) == []


def _plant_side_branch_commit(repo: Path) -> str:
    main_branch = _run("git", "branch", "--show-current", cwd=repo).stdout.strip()
    _run("git", "checkout", "-q", "-b", "pre-rebase-side", cwd=repo)
    side_payload = repo / "side.txt"
    side_payload.write_text("pre-rebase side commit\n")
    _run("git", "add", "side.txt", cwd=repo)
    _run("git", "commit", "-q", "-m", "plant pre-rebase side commit", cwd=repo)
    side_commit = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    _run("git", "checkout", "-q", main_branch, cwd=repo)
    return side_commit


def test_planted_dangling_pre_rebase_side_branch_commit_fails(tmp_path: Path):
    repo, doc, ancestor, _ = _init_plant_repo(tmp_path)
    side_commit = _plant_side_branch_commit(repo)
    doc.write_text(doc.read_text().replace(ancestor, side_commit))
    assert "neither an ancestor of HEAD nor pinned" in "\n".join(_failures(repo, doc))


def test_planted_refs_keep_commit_passes(tmp_path: Path):
    repo, doc, ancestor, _ = _init_plant_repo(tmp_path)
    side_commit = _plant_side_branch_commit(repo)
    _run("git", "update-ref", "refs/keep/orca2_r345_plant", side_commit, cwd=repo)
    doc.write_text(doc.read_text().replace(ancestor, side_commit))
    assert _failures(repo, doc) == []


def test_planted_unresolved_git_commit_fails(tmp_path: Path):
    repo, doc, commit, _ = _init_plant_repo(tmp_path)
    bad = ("0" if commit[0] != "0" else "1") + commit[1:]
    doc.write_text(doc.read_text().replace(commit, bad))
    assert "unresolved Git commit" in "\n".join(_failures(repo, doc))


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
