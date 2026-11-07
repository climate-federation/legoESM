#!/usr/bin/env python3
"""Locate the ORCA-side ocean file that moves GYRE's merged 17-day state.

The production stepping loop remains the committed GYRE year harness.  This
probe changes only an explicit set of ocean-package files from round 100's
combined tree to the live GYRE source versions, records every content hash,
and compares full saved arrays with ``np.array_equal``.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


SOURCE = "d7de69d51f6392e77085aabd879ab9e0da23ebf5"
COMBINED = "23a9911ae48f80ef4c40e4639373b06d5bd42ad8"
DEFAULT_REFERENCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest/"
    "lego_seed0_r6carried"
)
DEFAULT_OUTPUT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round101"
)
OCEAN_PREFIX = "packages/ocean/legoesm/ocean/"
EXPECTED_DIFF_FILES = (
    "dynamics/barotropic_latlon_cgrid.py",
    "dynamics/latlon_cgrid_operators.py",
    "dynamics/ocean_model_latlon_cgrid.py",
    "dynamics/ocean_pe_latlon_cgrid.py",
    "dynamics/ocean_tendency_common.py",
    "eos.py",
    "fidelity/nemo_recipe.py",
    "fidelity/nemo_testcase_recipe.py",
    "freshwater.py",
    "physics/combined.py",
    "physics/convection/enhanced_diffusion.py",
    "physics/vertical_mixing/internal_wave_mixing.py",
    "physics/vertical_mixing/k_profiles.py",
    "state.py",
    "vertical.py",
)
FIELDS = ("T", "S", "u", "v", "ssh")


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(repo), *args))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def differing_files(repo: Path) -> tuple[str, ...]:
    output = _git(
        repo, "diff", "--name-only", f"{SOURCE}..{COMBINED}", "--",
        "packages/ocean/legoesm/ocean",
    ).decode().splitlines()
    relative = tuple(
        line.removeprefix(OCEAN_PREFIX) for line in output if line.strip()
    )
    require(
        relative == EXPECTED_DIFF_FILES,
        "combined/source ocean-file census moved:\n"
        f"expected={EXPECTED_DIFF_FILES}\nactual={relative}",
    )
    return relative


def parse_source_files(repo: Path, value: str) -> tuple[str, ...]:
    available = differing_files(repo)
    if value == "ALL":
        return available
    if value == "NONE":
        return ()
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    require(bool(selected), "--source-files resolved to an empty list")
    unknown = sorted(set(selected) - set(available))
    require(not unknown, f"source files are outside the frozen census: {unknown}")
    require(len(selected) == len(set(selected)), "duplicate --source-files entry")
    return tuple(path for path in available if path in selected)


def parse_reverse_commits(repo: Path, value: str) -> tuple[str, ...]:
    if not value:
        return ()
    commits = tuple(part.strip() for part in value.split(",") if part.strip())
    require(len(commits) == len(set(commits)), "duplicate --reverse-commits entry")
    for commit in commits:
        try:
            kind = _git(repo, "cat-file", "-t", commit).decode().strip()
        except subprocess.CalledProcessError as error:
            raise GateError(f"unknown reverse commit {commit}") from error
        require(kind == "commit", f"reverse object {commit} is {kind}, not commit")
    return commits


def parse_source_opcodes(repo: Path, value: str) -> tuple[int, ...]:
    if not value:
        return ()
    selected: list[int] = []
    for part in value.split(","):
        bounds = part.strip().split("-")
        require(len(bounds) in (1, 2), f"bad source-opcode range {part!r}")
        try:
            start = int(bounds[0])
            stop = int(bounds[-1])
        except ValueError as error:
            raise GateError(f"bad source-opcode range {part!r}") from error
        require(0 <= start <= stop, f"bad source-opcode range {part!r}")
        selected.extend(range(start, stop + 1))
    require(len(selected) == len(set(selected)), "duplicate --source-opcodes entry")
    path = f"{OCEAN_PREFIX}dynamics/ocean_model_latlon_cgrid.py"
    source = _git(repo, "show", f"{SOURCE}:{path}").decode().splitlines(True)
    combined = _git(repo, "show", f"{COMBINED}:{path}").decode().splitlines(True)
    count = sum(
        tag != "equal" for tag, *_ in
        difflib.SequenceMatcher(None, source, combined, autojunk=False).get_opcodes()
    )
    require(max(selected, default=-1) < count,
            f"source opcode exceeds frozen count {count}")
    return tuple(selected)


def source_opcode_overlay(repo: Path, selected: tuple[int, ...]) -> tuple[bytes, list[dict]]:
    path = f"{OCEAN_PREFIX}dynamics/ocean_model_latlon_cgrid.py"
    source = _git(repo, "show", f"{SOURCE}:{path}").decode().splitlines(True)
    combined = _git(repo, "show", f"{COMBINED}:{path}").decode().splitlines(True)
    output: list[str] = []
    records: list[dict] = []
    diff_index = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, source, combined, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            output.extend(combined[j1:j2])
            continue
        use_source = diff_index in selected
        output.extend(source[i1:i2] if use_source else combined[j1:j2])
        records.append({
            "index": diff_index,
            "tag": tag,
            "source_lines": [i1 + 1, i2],
            "combined_lines": [j1 + 1, j2],
            "selected": use_source,
        })
        diff_index += 1
    return "".join(output).encode(), records


def build_overlay(
    repo: Path,
    target: Path,
    selected: tuple[str, ...],
    reverse_commits: tuple[str, ...] = (),
    source_opcodes: tuple[int, ...] = (),
) -> dict:
    require(not target.exists(), f"refusing existing overlay {target}")
    unchanged = subprocess.run(
        ("git", "-C", str(repo), "diff", "--quiet", COMBINED, "--",
         "packages/ocean"),
        check=False,
    )
    require(
        unchanged.returncode == 0,
        "current ocean package differs from round-100 combined tree",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ("git", "clone", "-q", "--shared", str(repo), str(target)),
        check=True,
    )
    for relative in differing_files(repo):
        path = f"{OCEAN_PREFIX}{relative}"
        destination = target / path
        if relative in selected:
            source_payload = _git(repo, "show", f"{SOURCE}:{path}")
            destination.write_bytes(source_payload)
    opcode_records: list[dict] = []
    if source_opcodes:
        payload, opcode_records = source_opcode_overlay(repo, source_opcodes)
        destination = target / (
            OCEAN_PREFIX + "dynamics/ocean_model_latlon_cgrid.py"
        )
        destination.write_bytes(payload)
    for commit in reverse_commits:
        patch = _git(
            repo, "show", "--format=", "--binary", commit, "--",
            "packages/ocean/legoesm/ocean",
        )
        require(bool(patch), f"reverse commit {commit} has no ocean-package diff")
        applied = subprocess.run(
            ("git", "-C", str(target), "apply", "--3way", "-R"),
            input=patch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        require(
            applied.returncode == 0,
            f"reverse commit {commit} did not apply:\n"
            f"{applied.stdout.decode(errors='replace')}",
        )
    records = []
    for relative in differing_files(repo):
        path = f"{OCEAN_PREFIX}{relative}"
        combined_payload = _git(repo, "show", f"{COMBINED}:{path}")
        source_payload = _git(repo, "show", f"{SOURCE}:{path}")
        destination = target / path
        records.append({
            "path": path,
            "selected": relative in selected,
            "combined_sha256": sha256_bytes(combined_payload),
            "source_sha256": sha256_bytes(source_payload),
            "candidate_sha256": sha256(destination),
        })
    manifest = {
        "format": "orca2-round101-gyre-year-file-overlay-v1",
        "source_commit": SOURCE,
        "combined_commit": COMBINED,
        "selected_source_files": list(selected),
        "reversed_commits": list(reverse_commits),
        "source_opcodes": list(source_opcodes),
        "opcode_records": opcode_records,
        "files": records,
    }
    (target / "overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def compare_snapshots(
    reference: Path,
    candidate: Path,
    *,
    days: int = 17,
    plant: str | None = None,
) -> dict:
    require(days >= 1, "days must be positive")
    first = None
    differing_days = 0
    compared_arrays = 0
    for day in range(1, days + 1):
        left_path = reference / f"day{day:03d}.npz"
        right_path = candidate / f"day{day:03d}.npz"
        require(left_path.is_file(), f"missing reference {left_path}")
        require(right_path.is_file(), f"missing candidate {right_path}")
        day_differs = False
        with np.load(left_path) as left, np.load(right_path) as right:
            require(set(left.files) == set(right.files),
                    f"day {day}: field names differ")
            require(set(FIELDS).issubset(left.files),
                    f"day {day}: missing scored fields")
            for field in left.files:
                a = np.asarray(left[field])
                b = np.asarray(right[field])
                require(a.shape == b.shape, f"day {day} {field}: shape differs")
                require(a.dtype == b.dtype, f"day {day} {field}: dtype differs")
                require(a.dtype.kind == "f" and a.dtype.itemsize in (4, 8),
                        f"day {day} {field}: unsupported dtype {a.dtype}")
                compared_arrays += 1
                if plant == "flip-first-bit" and day == 1 and field == left.files[0]:
                    b = b.copy()
                    bits = b.view(np.uint64 if b.dtype == np.float64 else np.uint32)
                    bits.flat[0] ^= 1
                bit_dtype = np.uint64 if a.dtype.itemsize == 8 else np.uint32
                equal = (
                    np.array_equal(a, b)
                    and np.array_equal(a.view(bit_dtype), b.view(bit_dtype))
                )
                if equal:
                    continue
                day_differs = True
                if first is None:
                    unequal = np.flatnonzero(a.view(np.uint8) != b.view(np.uint8))
                    bit_unequal = np.flatnonzero(
                        a.view(bit_dtype).ravel() != b.view(bit_dtype).ravel()
                    )
                    value_unequal = np.flatnonzero(a.ravel() != b.ravel())
                    require(bit_unequal.size > 0,
                            f"day {day} {field}: array differs without bit delta")
                    index = int(bit_unequal[0])
                    location = tuple(int(v) for v in np.unravel_index(index, a.shape))
                    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    first = {
                        "day": day,
                        "field": field,
                        "index": location,
                        "reference": float(a[location]),
                        "candidate": float(b[location]),
                        "max_abs": float(np.max(delta)),
                        "byte_unequal": int(unequal.size),
                        "value_unequal": int(value_unequal.size),
                    }
        differing_days += int(day_differs)
    return {
        "status": "EXACT" if first is None else "DIFFERENT",
        "days": days,
        "differing_days": differing_days,
        "compared_arrays": compared_arrays,
        "first_difference": first,
        "reference": str(reference),
        "candidate": str(candidate),
    }


def run_candidate(args, repo: Path) -> dict:
    selected = parse_source_files(repo, args.source_files)
    reverse_commits = parse_reverse_commits(repo, args.reverse_commits)
    source_opcodes = parse_source_opcodes(repo, args.source_opcodes)
    arm_count = sum(bool(arm) for arm in (selected, reverse_commits, source_opcodes))
    require(arm_count <= 1,
            "source-file, source-opcode, and reverse-commit arms are mutually exclusive")
    root = args.output / "candidates" / args.tag
    overlay = args.output / "overlays" / args.tag
    require(not root.exists(), f"refusing existing candidate output {root}")
    manifest = build_overlay(
        repo, overlay, selected, reverse_commits, source_opcodes
    )
    harness = overlay / (
        "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l2_gyre_year_fromrest.py"
    )
    env = dict(os.environ)
    env["LEGOESM_GATE_ALLOW_DIRTY"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(overlay / path) for path in (
        "packages/core", "packages/ocean", "packages/atmosphere",
        "packages/coupler", "packages/ice", "packages/land", "packages/ml",
        "packages/tools", "src",
    ))
    command = (
        sys.executable, str(harness), "--member", "0", "--days", "17",
        "--snap-steps", "6", "--tag", args.tag, "--root", str(root),
    )
    log_path = args.output / f"candidate_{args.tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command, cwd=overlay, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    require(completed.returncode == 0,
            f"candidate {args.tag} failed ({completed.returncode}); see {log_path}")
    member = root / f"lego_seed0_{args.tag}"
    report = compare_snapshots(
        args.reference, member, days=17, plant=args.plant
    )
    report["tag"] = args.tag
    report["overlay_manifest"] = str(overlay / "overlay_manifest.json")
    report["overlay_manifest_sha256"] = sha256(
        overlay / "overlay_manifest.json"
    )
    report["selected_source_files"] = list(selected)
    report["reversed_commits"] = list(reverse_commits)
    report["source_opcodes"] = list(source_opcodes)
    report["run_log"] = str(log_path)
    report["run_command"] = list(command)
    report["overlay"] = manifest
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-candidate", action="store_true")
    parser.add_argument("--tag")
    parser.add_argument("--source-files", default="NONE")
    parser.add_argument("--reverse-commits", default="")
    parser.add_argument("--source-opcodes", default="")
    parser.add_argument("--compare", nargs=2, type=Path,
                        metavar=("REFERENCE", "CANDIDATE"))
    parser.add_argument("--days", type=int, default=17)
    parser.add_argument("--expect", choices=("EXACT", "DIFFERENT"))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--plant", choices=("flip-first-bit",))
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    if args.run_candidate:
        require(args.tag is not None, "--run-candidate requires --tag")
        require(args.compare is None, "choose run-candidate or compare")
        report = run_candidate(args, repo)
    else:
        require(args.compare is not None, "--compare is required")
        report = compare_snapshots(
            args.compare[0], args.compare[1], days=args.days, plant=args.plant
        )
    if args.expect is not None:
        require(report["status"] == args.expect,
                f"expected {args.expect}, got {report['status']}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"STATUS {report['status']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        raise SystemExit(2)
