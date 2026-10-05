#!/usr/bin/env python3
"""Admit ORCA2 round 41's passive per-rank stage-1 HPG stream."""

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
    nemo_testcase_l4_orca2_round40_rhs_family_gate as round40,
)

RECORD_TEMPLATE = "oracle_stage1_hpg_ranked_kt00000001_r{rank:04d}.bin"
PARENT_TEMPLATE = round40.RECORD_TEMPLATE
MAGIC = "NEMO_L4_R41HPG1"
NX, NY, NZ = round40.NX, round40.NY, round40.NZ
N2, N3 = NX * NY, NX * NY * NZ
FIELDS_3D = (
    "rhd", "e3w", "gdept_z0", "zhpi_u", "zhpi_v",
    "zuap_u", "zuap_v", "sum_u", "sum_v",
)
FIELDS_2D = ("r1_e1u", "r1_e2v")
EXPECTED_SIZE = 16 + 9 * 4 + (len(FIELDS_3D) * N3 + len(FIELDS_2D) * N2) * 8
RESTARTS = round40.RESTARTS
CITATIONS = {
    "hpg_sco": (
        "ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/"
        "dynhpg.f90:340-451"
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


def _field3(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float64).copy().reshape(
        (NX, NY, NZ), order="F").transpose(1, 0, 2)


def _field2(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float64).copy().reshape(
        (NX, NY), order="F").T


def read_record_bytes(payload: bytes, expected_rank: int) -> dict[str, object]:
    require(len(payload) == EXPECTED_SIZE,
            f"record is {len(payload)} bytes, expected {EXPECTED_SIZE}")
    raw, offset = _take(payload, 0, 16, "magic")
    try:
        magic = raw.decode("ascii").rstrip()
    except UnicodeDecodeError as error:
        raise GateError("magic is not ASCII") from error
    raw, offset = _take(payload, offset, 9 * 4, "header")
    header = struct.unpack("=9i", raw)
    require((magic, *header) == (
        MAGIC, 1, 1, 1, 3, expected_rank, NX, NY, NZ, 64),
        f"bad header {(magic, *header)!r}")
    fields = {}
    for name in FIELDS_3D:
        raw, offset = _take(payload, offset, N3 * 8, name)
        fields[name] = _field3(raw)
    for name in FIELDS_2D:
        raw, offset = _take(payload, offset, N2 * 8, name)
        fields[name] = _field2(raw)
    require(offset == len(payload), "trailing payload")
    require(all(np.isfinite(value).all() for value in fields.values()),
            "record has non-finite values")
    return {"header": header, "fields": fields}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_stamp(repo: Path, expected_commit: str) -> dict[str, object]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo, text=True)
    require(head == expected_commit, f"commit {head} != expected {expected_commit}")
    require(not status, "worktree is dirty")
    return {"commit": head, "clean": True}


def run(root: Path, baseline_root: Path, repo: Path, expected_commit: str,
        *, plant: str = "none") -> dict[str, object]:
    records = []
    for rank in range(2):
        payload = (root / RECORD_TEMPLATE.format(rank=rank)).read_bytes()
        if plant == "header" and rank == 0:
            changed = bytearray(payload)
            changed[16:20] = struct.pack("=i", 2)
            payload = bytes(changed)
        elif plant == "truncation" and rank == 0:
            payload = payload[:-1]
        records.append(read_record_bytes(payload, rank))
    if plant == "swapped-rank":
        read_record_bytes(
            (root / RECORD_TEMPLATE.format(rank=0)).read_bytes(), 1)

    parent_identity = {}
    for rank in range(2):
        name = PARENT_TEMPLATE.format(rank=rank)
        current = (root / name).read_bytes()
        baseline = (baseline_root / name).read_bytes()
        if plant == "parent-byte" and rank == 0:
            current = current[:-1] + bytes((current[-1] ^ 1,))
        parent_identity[str(rank)] = {
            "bit_exact": current == baseline,
            "candidate_sha256": hashlib.sha256(current).hexdigest(),
            "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        }
        require(parent_identity[str(rank)]["bit_exact"],
                f"rank-{rank} inherited RHS-family stream moved")

    stage2_name = "oracle_rkstage2_hpg_literal_kt00000001.bin"
    stage2_identity = {
        "bit_exact": ((root / stage2_name).read_bytes()
                      == (baseline_root / stage2_name).read_bytes()),
        "candidate_sha256": _sha256(root / stage2_name),
        "baseline_sha256": _sha256(baseline_root / stage2_name),
    }
    require(stage2_identity["bit_exact"],
            "inherited stage-2 HPG literal stream moved")

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

    return {
        "format": "nemo-testcase-l4-orca2-round41-stage1-hpg-v1",
        "status": "PASS_ACQUISITION",
        "label": "given NEMO's entry",
        "worktree": _git_stamp(repo, expected_commit),
        "expected_size": EXPECTED_SIZE,
        "fields_3d": list(FIELDS_3D),
        "fields_2d": list(FIELDS_2D),
        "record_sha256": {
            str(rank): _sha256(root / RECORD_TEMPLATE.format(rank=rank))
            for rank in range(2)
        },
        "inherited_rhs_family_streams": parent_identity,
        "inherited_stage2_hpg_literal": stage2_identity,
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
        choices=("none", "header", "truncation", "swapped-rank",
                 "parent-byte", "restart-byte"),
        default="none")
    args = parser.parse_args()
    try:
        result = run(args.root, args.baseline_root, args.repo,
                     args.expect_commit, plant=args.plant)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("ORCA2 ROUND41 STAGE1 HPG ACQUISITION PASS")
        return 0
    except (GateError, round40.GateError, OSError, ValueError) as error:
        result = {
            "format": "nemo-testcase-l4-orca2-round41-stage1-hpg-v1",
            "status": "PLANT-FIRED" if args.plant != "none" else "FAIL",
            "plant": args.plant,
            "error": str(error),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        marker = "STATUS PLANT-FIRED" if args.plant != "none" else "STATUS FAIL"
        print(f"ORCA2 ROUND41 STAGE1 HPG {args.plant.upper()} {marker}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
