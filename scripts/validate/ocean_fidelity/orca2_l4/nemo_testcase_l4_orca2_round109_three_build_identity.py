#!/usr/bin/env python3
"""Gate rung-0 base, accumulator, and per-level recorder identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PLANTS = ("none", "restart-byte", "deck-byte")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(base: Path, accumulator: Path, per_level: Path, plant: str) -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    rows = []
    for rank in range(2):
        name = f"ORCA2_00000010_restart_{rank:04d}.nc"
        base_bytes = (base / name).read_bytes()
        accumulator_bytes = (accumulator / name).read_bytes()
        per_level_bytes = (per_level / name).read_bytes()
        if plant == "restart-byte" and rank == 0:
            per_level_bytes = bytes([per_level_bytes[0] ^ 1]) + per_level_bytes[1:]
        require(base_bytes == accumulator_bytes == per_level_bytes,
                f"three-build restart identity moved: {name}")
        rows.append({"rank": rank, "bytes": len(base_bytes),
                     "sha256": digest(base_bytes)})

    deck_rows = []
    for name in ("namelist_cfg", "deck_files.sha256", "input_files.sha256"):
        accumulator_bytes = (accumulator / name).read_bytes()
        per_level_bytes = (per_level / name).read_bytes()
        if plant == "deck-byte" and name == "namelist_cfg":
            per_level_bytes = bytes([per_level_bytes[0] ^ 1]) + per_level_bytes[1:]
        require(accumulator_bytes == per_level_bytes,
                f"accumulator/per-level deck identity moved: {name}")
        base_bytes = (base / name).read_bytes()
        deck_rows.append({
            "name": name,
            "base_sha256": digest(base_bytes),
            "accumulator_sha256": digest(accumulator_bytes),
            "per_level_sha256": digest(per_level_bytes),
            "base_equal": base_bytes == accumulator_bytes,
            "recorder_builds_equal": True,
        })
    return {"status": "PASS_R109_THREE_BUILD_IDENTITY",
            "kt10_restarts": rows, "deck_identity": deck_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--accumulator-root", type=Path, required=True)
    parser.add_argument("--per-level-root", type=Path, required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.base_root, args.accumulator_root,
                     args.per_level_root, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
