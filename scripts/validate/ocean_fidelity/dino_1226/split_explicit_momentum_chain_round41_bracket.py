#!/usr/bin/env python3
"""Require byte-identical duplicate row-5 production captures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-a", type=Path, required=True)
    parser.add_argument("--capture-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True).strip():
        raise SystemExit("tracked-clean checkout required")
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    records = []
    for directory in (args.capture_a.resolve(), args.capture_b.resolve()):
        meta_path = directory / "capture.json"
        meta = json.loads(meta_path.read_text())
        info = meta["files"]["wzv_call2_production.bin"]
        stream = directory / "wzv_call2_production.bin"
        if (meta.get("session_id") != session
                or meta.get("raw_capture_count", 0) < 1
                or meta.get("post_dyn_zdf_matching_raw_count") != 1
                or meta.get("post_dyn_zdf_boundary_count") != 1
                or not meta.get("post_dyn_zdf_w_carried_exact")
                or not meta.get("hook_restored")
                or info.get("shape") != [199, 52, 37]
                or info.get("size_bytes") != 199 * 52 * 37 * 8
                or _sha(stream) != info.get("sha256")):
            raise SystemExit(f"capture admission failed: {directory}")
        records.append({
            "directory": str(directory),
            "metadata_sha256": _sha(meta_path),
            "stream_sha256": _sha(stream),
        })
    exact = records[0]["stream_sha256"] == records[1]["stream_sha256"]
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round41-bracket-v1",
        "session_id": session,
        "captures": records,
        "duplicate_byte_exact": exact,
        "scorer": _sha(Path(__file__).resolve()),
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"duplicate_byte_exact={exact}")
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
