#!/usr/bin/env python3
"""Diagnose the incomplete round-62 ORCA2 vector-advection streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp

MAGIC = "NEMO_L4_VADV_1"
RECORD_COMMIT = "990a34547bc1ca1f074a580f5ff9801324400a0d"
HEADER_NAMES = (
    "version",
    "kt",
    "stage",
    "Kmm",
    "Krhs",
    "rank",
    "nimpp",
    "njmpp",
    "ntsi",
    "ntei",
    "ntsj",
    "ntej",
    "jpi",
    "jpj",
    "jpk",
    "jpkm1",
    "bits",
    "nn_dynkeg",
    "ln_vortex_force",
    "ln_zad_Aimp",
)
PREFIX = (
    ("before_keg_u", 3),
    ("before_keg_v", 3),
    ("after_keg_u", 3),
    ("after_keg_v", 3),
    ("after_zad_u", 3),
    ("after_zad_v", 3),
    ("uu_Kmm", 3),
    ("vv_Kmm", 3),
    ("ww", 3),
    ("wsd_effective", 3),
)
SUFFIX = (
    ("e1e2t", 2),
    ("e1e2u", 2),
    ("e1e2v", 2),
    ("r1_e1u", 2),
    ("r1_e2v", 2),
    ("r1_e1e2u", 2),
    ("r1_e1e2v", 2),
    ("umask", 3),
    ("vmask", 3),
)


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


def _header(handle, *, path: Path) -> tuple[str, int, int, int, int]:
    raw_name = handle.read(16)
    require(len(raw_name) == 16, f"{path}: short field name")
    try:
        name = raw_name.decode("ascii").rstrip()
    except UnicodeDecodeError as exc:
        raise GateError(f"{path}: non-ASCII field name") from exc
    raw_shape = handle.read(16)
    require(len(raw_shape) == 16, f"{path}: short shape for {name}")
    return (name, *struct.unpack("=4i", raw_shape))


def _consume(
    handle,
    *,
    path: Path,
    wanted: tuple[str, int],
    dims: tuple[int, int, int],
    offset_plant: bool = False,
) -> None:
    name, rank, n1, n2, n3 = _header(handle, path=path)
    require((name, rank) == wanted, f"{path}: got {(name, rank)}, expected {wanted}")
    expected_shape = (dims[0], dims[1], dims[2] if rank == 3 else 1)
    require((n1, n2, n3) == expected_shape, f"{path}: bad shape for {name}: {(n1, n2, n3)}")
    count = n1 * n2 * (n3 if rank == 3 else 1)
    handle.seek(8 * count + (8 if offset_plant else 0), 1)
    require(handle.tell() <= path.stat().st_size, f"{path}: short payload for {name}")


def diagnose_record(
    path: Path,
    *,
    plant: str | None = None,
    expected_dims: tuple[int, int, int, int] | None = None,
) -> dict[str, object]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        require(magic == MAGIC, f"{path}: bad magic {magic!r}")
        raw_header = handle.read(4 * len(HEADER_NAMES))
        require(len(raw_header) == 4 * len(HEADER_NAMES), f"{path}: short header")
        values = struct.unpack(f"={len(HEADER_NAMES)}i", raw_header)
        header = dict(zip(HEADER_NAMES, values, strict=True))
        dims = (header["jpi"], header["jpj"], header["jpk"])
        if expected_dims is not None:
            require((*dims, header["jpkm1"]) == expected_dims, f"{path}: wrong local extents")
        for index, wanted in enumerate(PREFIX):
            _consume(
                handle,
                path=path,
                wanted=wanted,
                dims=dims,
                offset_plant=(plant == "offset" and index == 0),
            )
        missing = []
        for wanted in (("e3u_Kmm", 3), ("e3v_Kmm", 3)):
            name, rank, n1, n2, n3 = _header(handle, path=path)
            require((name, rank) == wanted, f"{path}: got {(name, rank)}, expected {wanted}")
            require((n1, n2, n3) == dims, f"{path}: bad shape for {name}: {(n1, n2, n3)}")
            missing.append(name)
        if plant == "signature":
            missing.pop()
        require(
            missing == ["e3u_Kmm", "e3v_Kmm"],
            f"{path}: missing-payload signature changed: {missing}",
        )
        for wanted in SUFFIX:
            _consume(handle, path=path, wanted=wanted, dims=dims)
        require(handle.read(1) == b"", f"{path}: trailing payload")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "header": header,
        "missing_payloads": missing,
        "later_fields_recovered": len(SUFFIX),
        "physical_eof_parsed": True,
    }


def run(root: Path, *, plant: str | None = None) -> dict[str, object]:
    producer = root / "producer_commit.txt"
    require(producer.is_file(), "missing producer commit")
    require(producer.read_text().strip() == RECORD_COMMIT, "failed record producer commit changed")
    paths = sorted(root.glob("oracle_vector_adv_split_kt00000001_s2_r*.bin"))
    require(
        [path.name for path in paths]
        == [
            "oracle_vector_adv_split_kt00000001_s2_r0000.bin",
            "oracle_vector_adv_split_kt00000001_s2_r0001.bin",
        ],
        "rank stream census changed",
    )
    records = [
        diagnose_record(path, plant=plant, expected_dims=(94, 152, 31, 30)) for path in paths
    ]
    require([record["header"]["rank"] for record in records] == [0, 1], "rank headers changed")
    return {
        "format": "nemo-testcase-l4-orca2-round63-vector-record-diagnosis-v1",
        "claim_label": "given NEMO's recorded operands",
        "worktree": worktree_stamp(),
        "producer_commit": RECORD_COMMIT,
        "records": records,
        "status": "CONFIRMED_INCOMPLETE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("offset", "signature"))
    args = parser.parse_args(argv)
    try:
        report = run(args.root, plant=args.plant)
    except GateError as exc:
        if args.plant:
            print(f"STATUS PLANT-FIRED: {args.plant}: {exc}")
            return 1
        raise
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print("STATUS CONFIRMED_INCOMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
