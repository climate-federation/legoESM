#!/usr/bin/env python3
"""Structural admission for the passive ORCA2 round-328 dyn_zdf record."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = "NEMO_L4_R328ZDF"
HEADER_NAMES = (
    "version", "kt", "stage", "kbb", "kmm", "krhs", "kaa", "rank",
    "nx", "ny", "nz", "ntsi", "ntsj", "real_bits", "n_fields",
)
STATE_PHASES = ("explicit", "baro_sub", "bottom_rhs", "top_rhs")
MATRIX_PHASES = ("base_zdf", "after_zad", "after_ldf", "bottom_diag", "top_diag")
MATRIX_TERMS = ("low", "diag", "up")
EXPECTED_ORDER = (
    "before_u", "before_v", "rhs_u", "rhs_v", "avm_input", "wi_input",
    "barotropic_u", "barotropic_v", "rCdU_bot", "rCdU_top", "utauU", "vtauV",
    *(f"{phase}_{component}" for phase in STATE_PHASES for component in "uv"),
    *(f"{component}_{phase}_{term}" for component in "uv"
      for phase in MATRIX_PHASES for term in MATRIX_TERMS),
    "solve_u", "solve_v", "corrected_u", "corrected_v",
)
PLANTS = ("header", "field-order", "truncation", "stamp", "avm", "bottom", "top", "solve", "endpoint")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _take(stream: io.BytesIO, count: int, what: str) -> bytes:
    value = stream.read(count)
    require(len(value) == count, f"truncated {what}")
    return value


def record_name(rank: int) -> str:
    return f"oracle_dynzdf_kt00000001_s3_r{rank:04d}.bin"


def read_record(path: Path, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(f"={len(HEADER_NAMES)}i", _take(stream, 4 * len(HEADER_NAMES), "header"))
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant == "header":
        header["stage"] = 2
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")
    require((header["version"], header["kt"], header["stage"]) == (1, 1, 3), "wrong version/kt/stage")
    require((header["kbb"], header["kmm"], header["krhs"], header["kaa"]) == (1, 2, 3, 3),
            "wrong RK indices")
    require(header["real_bits"] == 64, "record is not fp64")
    require(header["n_fields"] == len(EXPECTED_ORDER), "wrong field count")
    fields: dict[str, np.ndarray] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = _take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack("=7i", _take(stream, 28, f"{name} metadata"))
        require(rank in (2, 3), f"{name}: bad array rank {rank}")
        expected = (header["nx"], header["ny"], header["nz"] if rank == 3 else 1)
        require((n1, n2, n3) == expected, f"{name}: extent {(n1, n2, n3)} != {expected}")
        require((i0, j0, k0) == (header["ntsi"], header["ntsj"], 1), f"{name}: wrong origin")
        array = np.frombuffer(_take(stream, n1 * n2 * n3 * 8, f"{name} payload"), dtype="=f8").copy()
        require(bool(np.all(np.isfinite(array))), f"{name}: non-finite payload")
        require(name not in fields, f"duplicate field {name}")
        fields[name] = array.reshape((n1, n2, n3), order="F")
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    observed = tuple(reversed(order)) if plant == "field-order" else tuple(order)
    require(observed == EXPECTED_ORDER, "field order differs from NEMO source order")
    return {"header": header, "fields": fields, "sha256": hashlib.sha256(raw).hexdigest()}


def _changed(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.any(a.view(np.uint64) != b.view(np.uint64)))


def run_gate(root: Path, expect_commit: str, plant: str | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), "missing producer_commit.txt")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit, f"producer commit {produced} != {expect_commit}")
    records = {}
    for rank in (0, 1):
        path = root / record_name(rank)
        require(path.is_file(), f"missing {path.name}")
        record = read_record(path, plant=plant if rank == 0 else None)
        require(record["header"]["rank"] == rank, f"{path.name}: wrong rank")
        records[rank] = record
    fields = records[0]["fields"]
    if plant == "avm":
        fields["avm_input"].fill(0.0)
    if plant == "bottom":
        fields["u_bottom_diag_diag"] = fields["u_after_ldf_diag"].copy()
        fields["v_bottom_diag_diag"] = fields["v_after_ldf_diag"].copy()
    if plant == "top":
        fields["u_top_diag_diag"] = fields["u_bottom_diag_diag"].copy()
        fields["v_top_diag_diag"] = fields["v_bottom_diag_diag"].copy()
    if plant == "solve":
        fields["solve_u"] = fields["top_rhs_u"].copy()
        fields["solve_v"] = fields["top_rhs_v"].copy()
    if plant == "endpoint":
        fields["corrected_u"] = fields["solve_u"].copy()
        fields["corrected_v"] = fields["solve_v"].copy()
    for rank, record in records.items():
        f = record["fields"]
        require(bool(np.any(f["avm_input"] != 0.0)), f"rank {rank}: inactive avm channel")
        require(_changed(f["u_after_ldf_diag"], f["u_bottom_diag_diag"]) or
                _changed(f["v_after_ldf_diag"], f["v_bottom_diag_diag"]),
                f"rank {rank}: bottom drag changes no matrix diagonal")
        require(_changed(f["u_bottom_diag_diag"], f["u_top_diag_diag"]) or
                _changed(f["v_bottom_diag_diag"], f["v_top_diag_diag"]),
                f"rank {rank}: top drag changes no matrix diagonal")
        require(_changed(f["top_rhs_u"], f["solve_u"]) or _changed(f["top_rhs_v"], f["solve_v"]),
                f"rank {rank}: implicit solve is inactive")
        require(_changed(f["solve_u"], f["corrected_u"]) or _changed(f["solve_v"], f["corrected_v"]),
                f"rank {rank}: post-dyn_zdf barotropic correction is inactive")
    return {
        "format": "nemo-testcase-l4-orca2-round328-dynzdf-v1",
        "status": "PASS_R328_DYNZDF_STRUCTURE",
        "producer_commit": produced,
        "records": {record_name(rank): record["sha256"] for rank, record in records.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_gate(args.record_dir, args.expect_commit, args.plant)
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as error:
        print(f"REFUSE: {error}")
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
