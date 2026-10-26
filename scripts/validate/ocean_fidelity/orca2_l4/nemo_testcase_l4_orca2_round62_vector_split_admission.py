#!/usr/bin/env python3
"""Fail-closed admission for the ORCA2 round-62 KEG/ZAD split records."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp


MAGIC = "NEMO_L4_VADV_1"
HEADER_NAMES = (
    "version", "kt", "stage", "Kmm", "Krhs", "rank", "nimpp", "njmpp",
    "ntsi", "ntei", "ntsj", "ntej", "jpi", "jpj", "jpk", "jpkm1",
    "bits", "nn_dynkeg", "ln_vortex_force", "ln_zad_Aimp",
)
FIELDS = (
    ("before_keg_u", 3), ("before_keg_v", 3),
    ("after_keg_u", 3), ("after_keg_v", 3),
    ("after_zad_u", 3), ("after_zad_v", 3),
    ("uu_Kmm", 3), ("vv_Kmm", 3), ("ww", 3),
    ("wsd_effective", 3), ("e3u_Kmm", 3), ("e3v_Kmm", 3),
    ("e1e2t", 2), ("e1e2u", 2), ("e1e2v", 2),
    ("r1_e1u", 2), ("r1_e2v", 2),
    ("r1_e1e2u", 2), ("r1_e1e2v", 2),
    ("umask", 3), ("vmask", 3),
)
NAME_RE = re.compile(r"oracle_vector_adv_split_kt00000001_s2_r(\d{4})\.bin$")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(path: Path, *, plant: str | None = None) -> dict[str, object]:
    match = NAME_RE.fullmatch(path.name)
    require(match is not None, f"unexpected record name: {path.name}")
    filename_rank = int(match.group(1))
    with path.open("rb") as handle:
        magic_raw = handle.read(16)
        require(len(magic_raw) == 16, f"{path}: short magic")
        magic = magic_raw.decode("ascii").rstrip()
        raw_header = handle.read(4 * len(HEADER_NAMES))
        require(len(raw_header) == 4 * len(HEADER_NAMES), f"{path}: short header")
        values = list(struct.unpack(f"={len(HEADER_NAMES)}i", raw_header))
        if plant == "header" and filename_rank == 0:
            values[2] = 3
        header = dict(zip(HEADER_NAMES, values, strict=True))
        require(magic == MAGIC, f"{path}: bad magic {magic!r}")
        require((header["version"], header["kt"], header["stage"],
                 header["Kmm"], header["Krhs"], header["bits"],
                 header["nn_dynkeg"], header["ln_vortex_force"],
                 header["ln_zad_Aimp"]) == (1, 1, 2, 3, 2, 64, 0, 0, 0),
                f"{path}: wrong executed branch header {header}")
        require(header["rank"] == filename_rank,
                f"{path}: filename/header rank disagreement")
        require((header["jpi"], header["jpj"], header["jpk"], header["jpkm1"])
                == (94, 152, 31, 30), f"{path}: wrong local extents")
        expected = list(FIELDS)
        if plant == "field-order" and filename_rank == 0:
            expected[0], expected[1] = expected[1], expected[0]
        arrays: dict[str, np.ndarray] = {}
        observed = []
        for wanted_name, wanted_rank in expected:
            raw_name = handle.read(16)
            require(len(raw_name) == 16, f"{path}: short name before {wanted_name}")
            name = raw_name.decode("ascii").rstrip()
            raw_shape = handle.read(16)
            require(len(raw_shape) == 16, f"{path}: short shape for {name}")
            rank, n1, n2, n3 = struct.unpack("=4i", raw_shape)
            require((name, rank) == (wanted_name, wanted_rank),
                    f"{path}: got {(name, rank)}, expected {(wanted_name, wanted_rank)}")
            require((n1, n2, n3) == (94, 152, 31 if rank == 3 else 1),
                    f"{path}: bad shape for {name}: {(n1, n2, n3)}")
            count = n1 * n2 * (n3 if rank == 3 else 1)
            raw = handle.read(8 * count)
            require(len(raw) == 8 * count, f"{path}: short payload for {name}")
            array = np.frombuffer(raw, dtype=np.float64).copy()
            require(bool(np.isfinite(array).all()), f"{path}: non-finite {name}")
            arrays[name] = array.reshape((n1, n2, n3), order="F")
            observed.append((name, rank))
        if plant == "truncation" and filename_rank == 0:
            require(False, "truncation plant fired")
        require(handle.read(1) == b"", f"{path}: trailing payload")
    require(tuple(observed) == FIELDS, f"{path}: field contract changed")
    require(np.count_nonzero(arrays["wsd_effective"]) == 0,
            f"{path}: inactive wsd is nonzero")
    return {"header": header, "arrays": arrays, "sha256": sha256(path)}


def run(root: Path, *, baseline: Path, expect_commit: str,
        plant: str | None = None) -> dict[str, object]:
    stamp = worktree_stamp()
    expected_commit = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(stamp["commit"].lower() == expected_commit,
            f"commit stamp {stamp['commit']} != {expected_commit}")
    paths = sorted(root.glob("oracle_vector_adv_split_kt00000001_s2_r*.bin"))
    require([path.name for path in paths] == [
        "oracle_vector_adv_split_kt00000001_s2_r0000.bin",
        "oracle_vector_adv_split_kt00000001_s2_r0001.bin",
    ], f"rank stream census differs: {[path.name for path in paths]}")
    records = [read_record(path, plant=plant) for path in paths]
    require([record["header"]["rank"] for record in records] == [0, 1],
            "rank headers differ")
    require(records[0]["header"]["nimpp"] != records[1]["header"]["nimpp"],
            "rank global x origins are not distinct")
    for name in (
        "ORCA2_00000010_restart_0000.nc", "ORCA2_00000010_restart_0001.nc",
        "ORCA2_00000010_restart_ice_0000.nc", "ORCA2_00000010_restart_ice_0001.nc",
    ):
        require((root / name).is_file() and (baseline / name).is_file(),
                f"missing restart calibration file {name}")
        left, right = sha256(root / name), sha256(baseline / name)
        if plant == "restart" and name.endswith("_0000.nc"):
            left = "0" * 64
        require(left == right, f"instrument changed restart {name}")
    return {
        "format": "nemo-testcase-l4-orca2-round62-vector-split-admission-v1",
        "claim_label": "given NEMO's recorded operands",
        "worktree": stamp,
        "records": [
            {"path": str(path), "sha256": record["sha256"],
             "header": record["header"], "field_count": len(record["arrays"])}
            for path, record in zip(paths, records, strict=True)
        ],
        "physical_eof_parsed": True,
        "restart_identity": True,
        "status": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("header", "field-order", "truncation", "restart", "stamp"))
    args = parser.parse_args(argv)
    try:
        report = run(args.root, baseline=args.baseline,
                     expect_commit=args.expect_commit, plant=args.plant)
    except GateError as exc:
        if args.plant:
            print(f"STATUS PLANT-FIRED: {args.plant}: {exc}")
            return 1
        raise
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
