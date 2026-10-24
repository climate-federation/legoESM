#!/usr/bin/env python3
"""Admit the passive ORCA2 round-328 zdf_phy boundary record."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_dynzdf_gate import (
    GateError,
    _checksum,
    _changed,
    _one_ulp,
    _take,
    require,
    same_program_observers,
)

MAGIC = "NEMO_L4_R328ZP"
HEADER_NAMES = (
    "version", "kt", "kbb", "kmm", "krhs", "rank", "nx", "ny", "nz",
    "ntsi", "ntsj", "real_bits", "n_fields",
)
EXPECTED_ORDER = (
    "b1_avm", "b1_avt", "b1_avs", "tke_en", "tke_avm_k", "tke_avt_k",
    "tke_rn2", "tke_r3t_kmm", "tke_ht_0", "tke_mbkt", "tke_wmask",
    "b2_avm", "b2_avt", "b2_avs", "ddm_heat_inc", "ddm_salt_inc",
    "ddm_max_inc", "b3_avm", "b3_avt", "b3_avs", "iwm_zav_wave",
    "iwm_zemx_iwm", "iwm_zreb",
)
TWO_D_FIELDS = {"tke_r3t_kmm", "tke_ht_0", "tke_mbkt"}
PLANTS = ("header", "field-order", "truncation", "stamp", "boundary1",
          "boundary2", "boundary3")
OBSERVER_UNITS = ("dynzdf", "stprk3_stg", "zdfphy", "zdfddm", "zdfiwm")
WRITER_UNITS = ("dynzdf_round328_writer", "zdfphy_round328_writer")


def record_name(rank: int) -> str:
    return f"oracle_zdfphy_kt00000001_r{rank:04d}.bin"


def _extent(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    return (header["nx"], header["ny"],
            1 if name in TWO_D_FIELDS else header["nz"])


def read_record(path: Path, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(HEADER_NAMES)}i",
        _take(stream, 4 * len(HEADER_NAMES), "header"),
    )
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant == "header":
        header["kmm"] = 2
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")
    require((header["version"], header["kt"], header["kbb"],
             header["kmm"], header["krhs"]) == (1, 1, 1, 1, 3),
            "wrong version/kt/RK indices")
    require(header["real_bits"] == 64, "record is not fp64")
    require(header["n_fields"] == len(EXPECTED_ORDER), "wrong field count")
    require(min(header["nx"], header["ny"], header["nz"]) > 0,
            "non-positive record extent")
    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = _take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", _take(stream, 28, f"{name} metadata"))
        require(rank in (2, 3), f"{name}: bad array rank {rank}")
        require((n1, n2, n3) == _extent(name, header),
                f"{name}: extent {(n1, n2, n3)} != {_extent(name, header)}")
        require((i0, j0, k0) == (header["ntsi"], header["ntsj"], 2),
                f"{name}: wrong origin")
        checksum, = struct.unpack("=Q", _take(stream, 8, f"{name} checksum"))
        payload = _take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(
            (n1, n2, n3), order="F")
        require(bool(np.all(np.isfinite(array))), f"{name}: non-finite payload")
        require(name not in fields, f"duplicate field {name}")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    observed = tuple(reversed(order)) if plant == "field-order" else tuple(order)
    require(observed == EXPECTED_ORDER, "field order differs from NEMO source order")
    return {"header": header, "fields": fields, "checksums": checksums,
            "sha256": hashlib.sha256(raw).hexdigest()}


def _verify_checksums(record: dict) -> None:
    for name, values in record["fields"].items():
        require(_checksum(values) == record["checksums"][name],
                f"{name}: payload checksum differs from writer")


def _bits_equal(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    require(not _changed(actual, expected), message)


def _verify_source_order(record: dict) -> None:
    f = record["fields"]
    wet = f["tke_wmask"] != 0.0
    _bits_equal(f["b1_avm"][wet], f["tke_avm_k"][wet],
                "post-TKE avm differs from copied avm_k")
    _bits_equal(f["b1_avt"][wet], f["tke_avt_k"][wet],
                "post-TKE avt differs from copied avt_k")
    # River-mouth and EVD terms occur between boundary 1 and zdf_ddm in
    # zdf_phy.  The DDM tracer operands are therefore evidence channels, not
    # a reconstruction from the earlier boundary.
    _bits_equal((f["b1_avm"] + f["ddm_max_inc"])[wet], f["b2_avm"][wet],
                "post-DDM avm does not reproduce its recorded increment")
    _bits_equal((f["b2_avt"] + f["iwm_zav_wave"])[wet], f["b3_avt"][wet],
                "post-IWM avt does not reproduce zav_wave")
    _bits_equal((f["b2_avm"] + f["iwm_zav_wave"])[wet], f["b3_avm"][wet],
                "post-IWM avm does not reproduce zav_wave")
    require(bool(np.any(f["ddm_max_inc"][wet] != 0.0)), "DDM operand channel is inactive")
    require(bool(np.any(f["iwm_zav_wave"][wet] != 0.0)), "IWM operand channel is inactive")
    require(bool(np.all(f["tke_mbkt"] == np.floor(f["tke_mbkt"]))),
            "mbkt payload is not integer-valued")


def read_admitted_final(path: Path) -> dict[str, np.ndarray]:
    stream = io.BytesIO(path.read_bytes())
    magic = _take(stream, 16, "admitted ZDF magic").decode("ascii").rstrip()
    version, kt, level, nx, ny, nz, real_bits = struct.unpack(
        "=7i", _take(stream, 28, "admitted ZDF header"))
    require(magic == "NEMO_L4_ZDF___2", f"admitted ZDF magic {magic!r}")
    require((version, kt, level, real_bits) == (2, 1, 1, 64),
            "wrong admitted ZDF header")
    halo = 2
    full = nx * ny * nz
    interior = (nx - 2 * halo) * (ny - 2 * halo) * nz
    payload = np.frombuffer(_take(
        stream, (full + 3 * interior) * 8, "admitted ZDF payload"),
        dtype="=f8")
    require(stream.read(1) == b"", "admitted ZDF has trailing bytes")
    require(bool(np.all(np.isfinite(payload))), "admitted ZDF is non-finite")
    fields = {
        "avm": payload[:full].reshape((nx, ny, nz), order="F")[halo:-halo, halo:-halo],
    }
    offset = full
    for name in ("avt", "avs", "en"):
        fields[name] = payload[offset:offset + interior].reshape(
            (nx - 2 * halo, ny - 2 * halo, nz), order="F")
        offset += interior
    return fields


def _verify_final(record: dict, admitted: dict[str, np.ndarray], plant: bool) -> None:
    f, h = record["fields"], record["header"]
    wet = f["tke_wmask"] != 0.0
    require(bool(np.any(wet)), "record has no wet interfaces")
    for name in ("avm", "avt", "avs"):
        expected = admitted[name][:, :, 1:1 + h["nz"]]
        require(expected.shape == f[f"b3_{name}"].shape,
                f"admitted {name} shape {expected.shape} differs from record")
        if plant and name == "avm":
            expected = expected.copy()
            index = np.flatnonzero(wet)[0]
            expected.flat[index] = np.nextafter(expected.flat[index], np.float64(np.inf))
        _bits_equal(f[f"b3_{name}"][wet], expected[wet],
                    f"post-IWM {name} differs from admitted final wet-cell bits")


def run_gate(root: Path, expect_commit: str, plant: str | None = None) -> dict:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), "missing producer_commit.txt")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit, f"producer commit {produced} != {expect_commit}")
    admitted_path = root / "oracle_zdf_entry_kt00000001.bin"
    require(admitted_path.is_file(), f"missing {admitted_path.name}")
    admitted = read_admitted_final(admitted_path)
    records = {}
    for rank in (0, 1):
        path = root / record_name(rank)
        require(path.is_file(), f"missing {path.name}")
        record = read_record(path, plant=plant if rank == 0 else None)
        require(record["header"]["rank"] == rank, f"{path.name}: wrong rank")
        if rank == 0 and plant in {"boundary1", "boundary2", "boundary3"}:
            field = {"boundary1": "b1_avm", "boundary2": "b2_avm",
                     "boundary3": "b3_avm"}[plant]
            record["fields"][field] = _one_ulp(record["fields"][field])
        _verify_checksums(record)
        _verify_source_order(record)
        _verify_final(record, admitted, plant == "boundary3")
        records[rank] = record
    return {
        "format": "nemo-testcase-l4-orca2-round328-zdfphy-v1",
        "status": "PASS_R328_ZDFPHY_ADMISSION",
        "producer_commit": produced,
        "records": {record_name(rank): record["sha256"]
                    for rank, record in records.items()},
    }


def same_program(new_bld: Path, ref_bld: Path, patches: dict[str, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    require(tuple(patches) == OBSERVER_UNITS, "wrong zdf_phy observer unit order")
    return same_program_observers(
        new_bld, ref_bld, patches, WRITER_UNITS,
        new_cfg=new_cfg, ref_cfg=ref_cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--same-program", type=Path, metavar="NEW_BLD")
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--observer-patch", action="append", default=[])
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.same_program is not None:
            require(None not in (args.reference_bld, args.new_cfg, args.reference_cfg),
                    "--same-program needs reference BLD and config names")
            entries = [item.split("=", 1) for item in args.observer_patch]
            require(all(len(item) == 2 for item in entries),
                    "observer patches must be UNIT=PATH")
            patches = {unit: Path(path) for unit, path in entries}
            result = same_program(
                args.same_program, args.reference_bld, patches,
                new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(args.record_dir is not None and args.expect_commit is not None,
                    "admission needs --record-dir and --expect-commit")
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
