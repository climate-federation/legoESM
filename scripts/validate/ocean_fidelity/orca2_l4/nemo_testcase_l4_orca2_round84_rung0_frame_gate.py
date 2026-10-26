#!/usr/bin/env python3
"""Fail-closed admission for round-84 ORCA2 rung-0 entry/stage frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class GateError(RuntimeError):
    """The frame acquisition cannot be admitted."""


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "nemo_testcase_l4_orca2_round84_rung0_frames"
MODULE = ARTIFACTS / "l4_r84_frames.F90"
PATCH = ARTIFACTS / "stprk3_round84.patch"
SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
    "ORCA2_OMIP_L4/MY_SRC/stprk3.F90"
)
MAGIC = "NEMO_L4_R84FRM1"
HEADER = struct.Struct("=17i")
FIELD_HEADER = struct.Struct("=4i")
FIELDS = ("T", "S", "u", "v", "ssh")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _take(raw: bytes, offset: int, size: int, what: str) -> tuple[bytes, int]:
    end = offset + size
    require(end <= len(raw), f"truncated {what}")
    return raw[offset:end], end


def _mutate(raw: bytes, plant: str | None) -> bytes:
    planted = bytearray(raw)
    if plant == "header":
        planted[0] = ord("X")
    elif plant == "field-name":
        planted[16 + HEADER.size] = ord("X")
    elif plant == "truncation":
        planted.pop()
    elif plant == "nonfinite":
        first_payload = 16 + HEADER.size + 16 + FIELD_HEADER.size
        struct.pack_into("=d", planted, first_payload, float("nan"))
    return bytes(planted)


def read_frame(path: Path, plant: str | None = None) -> dict:
    raw = _mutate(path.read_bytes(), plant)
    chunk, offset = _take(raw, 0, 16, "magic")
    require(chunk.decode("ascii").rstrip() == MAGIC, f"{path.name}: bad magic")
    chunk, offset = _take(raw, offset, HEADER.size, "header")
    header = HEADER.unpack(chunk)
    (version, kt, stage, level, rank, nx, ny, nz, ntr, nimpp, njmpp,
     ntsi, ntsj, ntei, ntej, bits, nfields) = header
    require(version == 1, f"{path.name}: bad version {version}")
    require(1 <= kt <= 10 and 0 <= stage <= 3,
            f"{path.name}: bad step/stage {(kt, stage)}")
    require(rank in (0, 1), f"{path.name}: bad rank {rank}")
    require(nx > 0 and ny > 0 and nz > 1 and ntr == 2 and bits == 64,
            f"{path.name}: invalid dimensions/precision {header}")
    require(nfields == len(FIELDS), f"{path.name}: bad field count {nfields}")
    require(1 <= ntsi <= ntei <= nx and 1 <= ntsj <= ntej <= ny,
            f"{path.name}: invalid owned bounds")
    require(nimpp > 0 and njmpp > 0, f"{path.name}: invalid global origin")
    values = {}
    offsets = {}
    for expected_name in FIELDS:
        chunk, offset = _take(raw, offset, 16, f"{expected_name} name")
        name = chunk.decode("ascii").rstrip()
        require(name == expected_name,
                f"{path.name}: expected field {expected_name!r}, got {name!r}")
        chunk, offset = _take(raw, offset, FIELD_HEADER.size,
                              f"{name} dimensions")
        ndim, dx, dy, dz = FIELD_HEADER.unpack(chunk)
        wanted = (2, nx, ny, 1) if name == "ssh" else (3, nx, ny, nz)
        require((ndim, dx, dy, dz) == wanted,
                f"{path.name}: {name} dimensions {(ndim, dx, dy, dz)} != {wanted}")
        count = math.prod((dx, dy, dz))
        offsets[name] = offset
        chunk, offset = _take(raw, offset, count * 8, f"{name} payload")
        array = np.frombuffer(chunk, dtype="=f8")
        require(bool(np.isfinite(array).all()), f"{path.name}: {name} non-finite")
        values[name] = {
            "count": count,
            "min": float(array.min()),
            "max": float(array.max()),
        }
    require(offset == len(raw), f"{path.name}: trailing bytes after physical EOF")
    return {
        "path": str(path),
        "header": {
            "kt": kt, "stage": stage, "level": level, "rank": rank,
            "shape": [nx, ny, nz], "origin": [nimpp, njmpp],
            "owned": [ntsi, ntsj, ntei, ntej], "bits": bits,
        },
        "fields": values,
        "offsets": offsets,
        "sha256": sha256_bytes(path.read_bytes()),
    }


def preflight() -> dict:
    for path in (MODULE, PATCH, SOURCE):
        require(path.is_file(), f"missing committed acquisition artifact {path}")
    removed = [line for line in PATCH.read_text().splitlines()
               if line.startswith("-") and not line.startswith("---")]
    require(not removed, f"writer patch is not additions-only: {removed[:1]}")
    with tempfile.TemporaryDirectory(prefix="orca2-r84-frame-") as tmp:
        target = Path(tmp) / "stprk3.F90"
        shutil.copyfile(SOURCE, target)
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", str(target), str(PATCH)],
            text=True, capture_output=True, check=False,
        )
        require(result.returncode == 0,
                f"patch does not apply: {result.stderr or result.stdout}")
        patched = target.read_text()
    require(patched.count("CALL r84_dump_frame") == 4,
            "patched source does not carry exactly four frame calls")
    require(MODULE.read_text().count("CALL put3") == 4,
            "writer does not carry exactly four 3-D fields")
    require(MODULE.read_text().count("CALL put2") == 1,
            "writer does not carry exactly one 2-D field")
    return {
        "status": "PREFLIGHT_PASS",
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "module_sha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        "removed_source_lines": len(removed),
        "frame_calls": 4,
        "fields": list(FIELDS),
    }


def _stamp(path: Path, expected_commit: str, digest: str) -> None:
    stamp = Path(f"{path}.stamp")
    require(stamp.is_file(), f"missing {stamp}")
    parts = stamp.read_text().strip().split()
    require(len(parts) == 3, f"{stamp}: malformed stamp")
    require(parts == [digest, expected_commit, path.name],
            f"{stamp}: provenance mismatch")


def admit(root: Path, expected_commit: str, plant: str | None) -> dict:
    require(root.is_dir(), f"missing record root {root}")
    commit_file = root / "producer_commit.txt"
    require(commit_file.is_file(), "missing producer_commit.txt")
    require(commit_file.read_text().strip() == expected_commit,
            "producer commit does not match")
    wanted = {
        f"oracle_r84_frame_rank{rank:04d}_kt{kt:08d}_s{stage}.bin"
        for rank in (0, 1) for kt in range(1, 11) for stage in range(4)
    }
    actual = {path.name for path in root.glob("oracle_r84_frame_*.bin")}
    require(actual == wanted,
            f"frame inventory mismatch missing={sorted(wanted-actual)} "
            f"extra={sorted(actual-wanted)}")
    records = []
    layouts: dict[int, tuple] = {}
    for name in sorted(wanted):
        path = root / name
        one_plant = plant if name == sorted(wanted)[0] and plant != "stamp" else None
        record = read_frame(path, one_plant)
        header = record["header"]
        expected_name = (
            f"oracle_r84_frame_rank{header['rank']:04d}_"
            f"kt{header['kt']:08d}_s{header['stage']}.bin"
        )
        require(name == expected_name, f"{name}: header/name mismatch")
        expected_level = ({0: 1, 1: 3, 2: 2, 3: 3}
                          if header["kt"] % 2 else
                          {0: 3, 1: 1, 2: 2, 3: 1})[header["stage"]]
        require(header["level"] == expected_level,
                f"{name}: level {header['level']} != {expected_level}")
        layout = (tuple(header["shape"]), tuple(header["origin"]),
                  tuple(header["owned"]))
        layouts.setdefault(header["rank"], layout)
        require(layouts[header["rank"]] == layout,
                f"rank {header['rank']} layout changed within record")
        digest = record["sha256"]
        if plant == "stamp" and name == sorted(wanted)[0]:
            digest = "0" * 64
        _stamp(path, expected_commit, digest)
        records.append({key: value for key, value in record.items()
                        if key != "offsets"})
    require(set(layouts) == {0, 1}, "record does not cover both MPI ranks")
    require(layouts[0] != layouts[1], "two rank records claim one layout")
    return {
        "status": "PASS_RUNG0_FRAMES",
        "producer_commit": expected_commit,
        "record_count": len(records),
        "layouts": {str(key): value for key, value in layouts.items()},
        "records": records,
        "preflight": preflight(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=("header", "field-name", "truncation",
                                             "nonfinite", "stamp"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight:
            require(args.root is None and args.expect_commit is None,
                    "preflight does not accept record arguments")
            report = preflight()
        else:
            require(args.root is not None and args.expect_commit is not None,
                    "--root and --expect-commit are required")
            report = admit(args.root, args.expect_commit, args.plant)
            require(args.plant is None, f"{args.plant} plant stayed green")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"STATUS {report['status']}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as error:
        if args.plant:
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
