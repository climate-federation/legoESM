#!/usr/bin/env python3
"""Fail-closed identity bridge from round-46 stage entries to year owners."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp

STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/"
    "oracle_kt2_stage"
)
ENTRY_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/nemo_seed0"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(stage_root: Path, entry_root: Path, plant: bool) -> dict:
    rows = []
    for kt in (2, 3):
        name = f"oracle_step_entry_kt{kt:08d}.bin"
        stage_path = stage_root / name
        entry_path = entry_root / name
        if not stage_path.is_file() or not entry_path.is_file():
            raise RuntimeError(f"missing identity input for kt={kt}")
        stage_sha = sha256(stage_path)
        entry_sha = sha256(entry_path)
        if plant and kt == 2:
            entry_sha = "0" * 64
        equal = stage_sha == entry_sha
        rows.append(
            {
                "kt": kt,
                "stage_path": str(stage_path),
                "entry_path": str(entry_path),
                "stage_sha256": stage_sha,
                "entry_sha256": entry_sha,
                "bit_identical": equal,
            }
        )
    if not all(row["bit_identical"] for row in rows):
        raise RuntimeError(f"entry identity bridge failed: {rows}")
    return {
        "format": "nemo-testcase-l2-gyre-round53-entry-identity-v1",
        "worktree": worktree_stamp(),
        "stage_root": str(stage_root),
        "entry_root": str(entry_root),
        "rows": rows,
        "status": "PASS",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=STAGE_ROOT)
    parser.add_argument("--entry-root", type=Path, default=ENTRY_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.stage_root, args.entry_root, args.plant)
    except (OSError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
