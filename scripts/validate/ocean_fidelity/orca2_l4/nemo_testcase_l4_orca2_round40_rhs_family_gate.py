#!/usr/bin/env python3
"""Admit ORCA2 round 40's passive per-rank momentum-family stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round38_ranked_slow_forcing_gate as round38,
)

RECORD_TEMPLATE = "oracle_rhs_families_ranked_kt00000001_r{rank:04d}.bin"
PARENT_TEMPLATE = "oracle_slow_forcing_ranked_kt00000001_r{rank:04d}.bin"
MAGIC = "NEMO_L4_R40FAM"
FAMILIES = ("hpg", "ldf", "vor", "keg", "zad")
FIELDS = tuple(
    f"after_{family}_{face}"
    for family in FAMILIES
    for face in ("u", "v")
)
NX, NY, NZ = round38.DIMS
COUNT = NX * NY * NZ
EXPECTED_SIZE = 16 + 9 * 4 + len(FIELDS) * 4 + len(FIELDS) * COUNT * 8
RESTARTS = (
    "ORCA2_00000010_restart_0000.nc",
    "ORCA2_00000010_restart_0001.nc",
    "ORCA2_00000010_restart_ice_0000.nc",
    "ORCA2_00000010_restart_ice_0001.nc",
)
CITATIONS = {
    "stage1_momentum_order": (
        "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/"
        "stp2d.f90:139-166"
    ),
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _take(payload: bytes, offset: int, count: int, label: str):
    end = offset + count
    require(end <= len(payload), f"truncated {label}")
    return payload[offset:end], end


def read_record_bytes(payload: bytes, expected_rank: int) -> dict[str, object]:
    """Decode every registered field and fail closed on layout drift."""
    require(
        len(payload) == EXPECTED_SIZE,
        f"record is {len(payload)} bytes, expected {EXPECTED_SIZE}",
    )
    raw, offset = _take(payload, 0, 16, "magic")
    try:
        magic = raw.decode("ascii").rstrip()
    except UnicodeDecodeError as error:
        raise GateError("magic is not ASCII") from error
    raw, offset = _take(payload, offset, 9 * 4, "header")
    version, kt, kbb, krhs, rank, nx, ny, nz, bits = struct.unpack("=9i", raw)
    raw, offset = _take(payload, offset, len(FIELDS) * 4, "sizes")
    sizes = struct.unpack(f"={len(FIELDS)}i", raw)
    require(
        (magic, version, kt, kbb, krhs, rank, nx, ny, nz, bits)
        == (MAGIC, 1, 1, 1, 3, expected_rank, NX, NY, NZ, 64),
        "bad header "
        + repr((magic, version, kt, kbb, krhs, rank, nx, ny, nz, bits)),
    )
    require(sizes == (COUNT,) * len(FIELDS), "bad field sizes")
    fields = {}
    for name in FIELDS:
        raw, offset = _take(payload, offset, COUNT * 8, name)
        values = np.frombuffer(raw, dtype=np.float64).copy()
        require(
            values.size == COUNT and bool(np.all(np.isfinite(values))),
            f"bad values in {name}",
        )
        fields[name] = values.reshape((NX, NY, NZ), order="F").transpose(1, 0, 2)
    require(offset == len(payload), "trailing payload")
    require(tuple(fields) == FIELDS, "field registry changed")
    return {
        "header": {
            "magic": magic,
            "version": version,
            "kt": kt,
            "Kbb": kbb,
            "Krhs": krhs,
            "rank": rank,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "bits": bits,
        },
        "fields": fields,
    }


def _identity(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    require(left.shape == right.shape, f"shape changed: {left.shape} != {right.shape}")
    unequal = left.view(np.uint64) != right.view(np.uint64)
    return {
        "bit_exact": not bool(np.any(unequal)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "absolute_max": (
            float(np.max(np.abs(left - right))) if left.size else 0.0
        ),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_stamp(repo: Path, expected_commit: str) -> dict[str, object]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        text=True,
    )
    require(head == expected_commit, f"commit {head} != expected {expected_commit}")
    require(not status, "worktree is dirty")
    return {"commit": head, "clean": True}


def run(
    root: Path,
    baseline_root: Path,
    repo: Path,
    expected_commit: str,
    *,
    plant: str = "none",
) -> dict[str, object]:
    family_records = []
    parent_records = []
    for rank in range(2):
        payload = (root / RECORD_TEMPLATE.format(rank=rank)).read_bytes()
        if plant == "header" and rank == 0:
            changed = bytearray(payload)
            changed[16:20] = struct.pack("=i", 2)
            payload = bytes(changed)
        elif plant == "truncation" and rank == 0:
            payload = payload[:-1]
        family_records.append(read_record_bytes(payload, rank))
        parent_records.append(
            round38.read_ranked(root / PARENT_TEMPLATE.format(rank=rank), rank)
        )
    if plant == "swapped-rank":
        read_record_bytes(
            (root / RECORD_TEMPLATE.format(rank=0)).read_bytes(), 1
        )

    parent_streams = {}
    final_closure = {}
    for rank in range(2):
        parent_name = PARENT_TEMPLATE.format(rank=rank)
        current_parent = (root / parent_name).read_bytes()
        baseline_parent = (baseline_root / parent_name).read_bytes()
        if plant == "parent-byte" and rank == 0:
            current_parent = current_parent[:-1] + bytes((current_parent[-1] ^ 1,))
        parent_streams[str(rank)] = {
            "bit_exact": current_parent == baseline_parent,
            "candidate_sha256": hashlib.sha256(current_parent).hexdigest(),
            "baseline_sha256": hashlib.sha256(baseline_parent).hexdigest(),
        }
        require(parent_streams[str(rank)]["bit_exact"], "inherited ranked stream moved")

        fields = family_records[rank]["fields"]
        assert isinstance(fields, dict)
        if plant == "final-ulp" and rank == 0:
            fields["after_zad_u"][2, 2, 0] = np.nextafter(
                fields["after_zad_u"][2, 2, 0], np.inf
            )
        final_closure[str(rank)] = {}
        for face in ("u", "v"):
            final = fields[f"after_zad_{face}"][2:-2, 2:-2, : NZ - 1]
            parent = parent_records[rank][f"krhs_{face}"]
            row = _identity(final, parent)
            final_closure[str(rank)][face] = row
            require(row["bit_exact"], f"rank-{rank} final {face} RHS moved")

    restart_identity = {}
    for name in RESTARTS:
        current = (root / name).read_bytes()
        baseline = (baseline_root / name).read_bytes()
        if plant == "restart-byte" and name == RESTARTS[0]:
            current = current[:-1] + bytes((current[-1] ^ 1,))
        restart_identity[name] = {
            "bit_exact": current == baseline,
            "candidate_sha256": hashlib.sha256(current).hexdigest(),
            "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        }
        require(restart_identity[name]["bit_exact"], f"restart moved: {name}")

    time_text = (root / "run.user.time.log").read_text()
    stdout_text = (root / "run.user.stdout.log").read_text()
    ocean_text = (root / "ocean.output").read_text()
    require("MPIRUN_RC=0" in time_text and "RUN DONE" in time_text,
            "completion log changed")
    require(any(line.strip() == "STOP 0" for line in stdout_text.splitlines()),
            "stdout lacks STOP 0")
    for text, rank in ((ocean_text, 0), (stdout_text, 1)):
        marker = RECORD_TEMPLATE.format(rank=rank)
        require(sum(marker in line for line in text.splitlines()) == 1,
                f"rank-{rank} marker is not unique")

    stamp = _git_stamp(repo, expected_commit)
    return {
        "format": "nemo-testcase-l4-orca2-round40-rhs-family-v1",
        "status": "PASS_ACQUISITION",
        "label": "given NEMO's entry",
        "worktree": stamp,
        "expected_size": EXPECTED_SIZE,
        "fields": list(FIELDS),
        "record_sha256": {
            str(rank): _digest(root / RECORD_TEMPLATE.format(rank=rank))
            for rank in range(2)
        },
        "inherited_ranked_streams": parent_streams,
        "final_vs_same_run_completed_rhs": final_closure,
        "restart_identity": restart_identity,
        "citations": CITATIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plant",
        choices=("none", "header", "truncation", "swapped-rank", "final-ulp",
                 "parent-byte", "restart-byte"),
        default="none",
    )
    args = parser.parse_args()
    try:
        result = run(
            args.root,
            args.baseline_root,
            args.repo,
            args.expect_commit,
            plant=args.plant,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("ORCA2 ROUND40 RHS FAMILY ACQUISITION PASS")
        return 0
    except (GateError, round38.GateError, OSError, ValueError) as error:
        result = {
            "format": "nemo-testcase-l4-orca2-round40-rhs-family-v1",
            "status": "PLANT-FIRED" if args.plant != "none" else "FAIL",
            "plant": args.plant,
            "error": str(error),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        marker = "STATUS PLANT-FIRED" if args.plant != "none" else "STATUS FAIL"
        print(f"ORCA2 ROUND40 RHS FAMILY {args.plant.upper()} {marker}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
