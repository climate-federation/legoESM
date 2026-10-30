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
# Exact round-341 correction citations: historical identities only, not producers
# for admitted round-345 evidence. Every other 40-hex citation requires ancestry.
GIT_COMMIT_EXEMPTIONS = {
    (
        "docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round341_rescore_walk_receipt.md",
        "5fae40dddf6b09178bd9008464227adb76b9ebf7",
    ),
    (
        "docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round341_rescore_walk_receipt.md",
        "ca77c284e0a5b9c5d804d9e25bc61581350e87bd",
    ),
}


def _run(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def _author_date(timestamp: int, offset: str) -> dt.date:
    sign = 1 if offset[0] == "+" else -1
    delta = dt.timedelta(hours=int(offset[1:3]), minutes=int(offset[3:5]))
    return dt.datetime.fromtimestamp(timestamp, dt.timezone(sign * delta)).date()


def _added_after(repo: Path, cutoff: dt.date = CUTOFF) -> dict[Path, set[int]]:
    """Return present lines blamed to author dates on or after the cutoff."""
    audited: dict[Path, set[int]] = {}
    history = _run(
        "git", "log", "--format=@@%aI", "--name-only", "--", "docs", cwd=repo
    ).stdout
    candidates: set[str] = set()
    stamp: dt.date | None = None
    for line in history.splitlines():
        if line.startswith("@@"):
            stamp = dt.datetime.fromisoformat(line[2:]).date()
        elif stamp is not None and stamp >= cutoff and DOC_RE.match(line):
            candidates.add(line)
    for name in sorted(candidates):
        doc = repo / name
        if not doc.is_file():
            continue
        output = _run("git", "blame", "--line-porcelain", "--", name, cwd=repo).stdout
        line_number: int | None = None
        timestamp: int | None = None
        offset: str | None = None
        included: set[int] = set()
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 3 and re.fullmatch(r"[0-9a-f^]{40}", fields[0]):
                line_number = int(fields[2])
                timestamp = None
                offset = None
            elif line.startswith("author-time "):
                timestamp = int(line.removeprefix("author-time "))
            elif line.startswith("author-tz "):
                offset = line.removeprefix("author-tz ")
            elif line.startswith("\t"):
                if line_number is None or timestamp is None or offset is None:
                    raise AssertionError(f"incomplete git blame metadata for {name}")
                if _author_date(timestamp, offset) >= cutoff:
                    included.add(line_number)
        if included:
            audited[doc] = included
    return audited


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

    return f"{label}: Git commit {value} is not an ancestor of HEAD"


def _failures(repo: Path, doc: Path, audited_lines: set[int] | None = None) -> list[str]:
    lines = doc.read_text(errors="replace").splitlines()
    failures: list[str] = []
    for index, line in enumerate(lines):
        if audited_lines is not None and index + 1 not in audited_lines:
            continue
        for match in HEX_RE.finditer(line):
            value = match.group().lower()
            label = f"{doc.relative_to(repo)}:{index + 1}"
            if len(value) == 40:
                if (str(doc.relative_to(repo)), value) in GIT_COMMIT_EXEMPTIONS:
                    continue
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
    failures = [
        failure
        for doc, audited_lines in _added_after(REPO).items()
        for failure in _failures(REPO, doc, audited_lines)
    ]
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
    assert "not an ancestor of HEAD" in "\n".join(_failures(repo, doc))


def test_planted_bad_digest_authored_on_cutoff_date_fails(tmp_path: Path):
    repo, doc, _, digest = _init_plant_repo(tmp_path)
    bad = ("0" if digest[0] != "0" else "1") + digest[1:]
    doc.write_text(doc.read_text().replace(digest, bad))
    _run("git", "add", str(doc.relative_to(repo)), cwd=repo)
    _run(
        "git",
        "commit",
        "-q",
        "--date=2026-10-29T00:00:00+00:00",
        "-m",
        "plant receipt on inclusive audit date",
        cwd=repo,
    )
    audited = _added_after(repo)
    assert doc in audited
    assert "SHA-256 mismatch" in "\n".join(_failures(repo, doc, audited[doc]))


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
