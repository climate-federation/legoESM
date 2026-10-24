#!/usr/bin/env python3
"""Admit the passive ORCA2 round-331 source-order TKE record."""

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
    _changed,
    _take,
    _verify_checksums,
    read_record as read_dynzdf_record,
    record_name as dynzdf_record_name,
    require,
    same_program_observers,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round328_zdf_terms_gate import (
    read_record as read_zdfphy_record,
    record_name as zdfphy_record_name,
)

MAGIC = "NEMO_L4_R331TK"
HEADER_NAMES = (
    "version", "kt", "kbb", "kmm", "krhs", "rank", "nx", "ny", "nz",
    "ntsi", "ntsj", "real_bits", "n_fields",
)
EXPECTED_ORDER = (
    "sh2_output", "drag_bottom_output", "drag_top_output",
    "mxl_nmln_output", "mxl_hmlp_output", "tke_sh2_input",
    "tke_drag_bottom_input", "tke_drag_top_input", "tke_nmln_input",
    "tke_hmlp_input", "tke_avm_input", "tke_dissl_input", "after_bc_en",
    "matrix_rhs_en", "matrix_diag", "matrix_upper", "matrix_lower",
    "after_solve_en", "after_tke_en", "after_tke_avm", "after_tke_avt",
    "tke_wmask",
)
TWO_D_FIELDS = {
    "drag_bottom_output", "drag_top_output", "mxl_nmln_output",
    "mxl_hmlp_output", "tke_drag_bottom_input", "tke_drag_top_input",
    "tke_nmln_input", "tke_hmlp_input",
}
FULL_DEPTH_FIELDS = {
    "after_bc_en", "matrix_rhs_en", "matrix_diag", "matrix_upper",
    "matrix_lower", "after_solve_en", "after_tke_en", "tke_wmask",
}
PLANTS = (
    "header", "field-order", "truncation", "stamp", "sh2", "drag", "mxl",
    "boundary", "matrix", "solve", "final",
)
OBSERVER_UNITS = ("dynzdf", "stprk3_stg", "zdfphy", "zdftke")
WRITER_UNITS = ("dynzdf_round328_writer", "tke_round331_writer")


def record_name(rank: int) -> str:
    return f"oracle_tke331_kt00000001_r{rank:04d}.bin"


def _extent(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    nz = header["nz"] + (name in FULL_DEPTH_FIELDS)
    return header["nx"], header["ny"], 1 if name in TWO_D_FIELDS else nz


def _origin(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    k0 = 1 if name in TWO_D_FIELDS or name in FULL_DEPTH_FIELDS else 2
    return header["ntsi"], header["ntsj"], k0


def read_record(path: Path, plant: str | None = None) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(HEADER_NAMES)}i", _take(stream, 4 * len(HEADER_NAMES), "header"))
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
                f"{name}: wrong extent {(n1, n2, n3)}")
        require((i0, j0, k0) == _origin(name, header), f"{name}: wrong origin")
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


def _bits_equal(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    require(not _changed(actual, expected), message)


def _verify_source_order(record: dict) -> None:
    f = record["fields"]
    for output, consumed in (
        ("sh2_output", "tke_sh2_input"),
        ("drag_bottom_output", "tke_drag_bottom_input"),
        ("drag_top_output", "tke_drag_top_input"),
        ("mxl_nmln_output", "tke_nmln_input"),
        ("mxl_hmlp_output", "tke_hmlp_input"),
    ):
        _bits_equal(f[output], f[consumed], f"{output} differs at TKE entry")
    # A zero shear field is valid at nit000 from rest.  Execution is proved
    # by the writer's lsh2 flag and complete input-row census; the value gate
    # therefore checks the producer-to-consumer handoff, not nonzeroness.
    _bits_equal(f["after_bc_en"][:, :, :1], f["matrix_rhs_en"][:, :, :1],
                "surface TKE changed between boundary and matrix records")
    _bits_equal(f["matrix_rhs_en"][:, :, :1], f["after_solve_en"][:, :, :1],
                "surface TKE changed during the interior solve")
    _bits_equal(f["after_solve_en"][:, :, :1], f["after_tke_en"][:, :, :1],
                "surface TKE changed after the solve")
    require(bool(np.any(f["drag_bottom_output"] != 0.0)), "bottom drag channel is inactive")
    require(bool(np.all(f["mxl_nmln_output"] == np.floor(f["mxl_nmln_output"]))),
            "mixed-layer level is not integer-valued")
    require(bool(np.any(f["matrix_diag"] != 0.0)), "matrix channel is inactive")


def _verify_references(record: dict, dynzdf: dict, zdfphy: dict) -> None:
    f, h = record["fields"], record["header"]
    df = dynzdf["fields"]
    for name, source in (("drag_bottom_output", "rCdU_bot"),
                         ("drag_top_output", "rCdU_top")):
        expected = df[source][:h["nx"], :h["ny"], :]
        _bits_equal(f[name], expected, f"{name} differs from admitted dyn_zdf input")
    zf, zh = zdfphy["fields"], zdfphy["header"]
    require((h["nx"], h["ny"], h["nz"]) == (zh["nx"], zh["ny"], zh["nz"]),
            "round-331 and admitted zdf_phy extents differ")
    wet = zf["tke_wmask"] != 0.0
    require(bool(np.any(wet)), "admitted zdf_phy record has no wet interfaces")
    for current, admitted in (("after_tke_en", "tke_en"),
                              ("after_tke_avm", "tke_avm_k"),
                              ("after_tke_avt", "tke_avt_k")):
        values = f[current][:, :, 1:] if current == "after_tke_en" else f[current]
        _bits_equal(values[wet], zf[admitted][wet],
                    f"{current} differs from admitted zdf_phy wet-cell bits")


def _plant(record: dict, plant: str) -> None:
    name = {
        "sh2": "sh2_output", "drag": "drag_bottom_output",
        "mxl": "mxl_nmln_output", "boundary": "after_bc_en",
        "matrix": "matrix_rhs_en", "solve": "after_solve_en",
        "final": "after_tke_avm",
    }[plant]
    values = record["fields"][name]
    wmask = record["fields"]["tke_wmask"] != 0.0
    if name in TWO_D_FIELDS:
        wet = wmask[:, :, :1]
    elif name in {"after_bc_en", "matrix_rhs_en", "after_solve_en"}:
        wet = np.zeros_like(values, dtype=bool)
        wet[:, :, :1] = wmask[:, :, :1]
    else:
        wet = wmask[:, :, 1:]
    require(wet.shape == values.shape, f"{name}: wet-mask shape differs")
    wet_index = np.flatnonzero(wet)
    require(bool(wet_index.size), f"cannot plant {name} without wet cells")
    values.flat[wet_index[0]] = np.nextafter(
        values.flat[wet_index[0]], np.float64(np.inf))


def run_gate(root: Path, zdf_reference_dir: Path, expect_commit: str,
             plant: str | None = None) -> dict:
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
        _verify_checksums(record)
        records[rank] = record
    if plant in {"sh2", "drag", "mxl", "boundary", "matrix", "solve", "final"}:
        _plant(records[0], plant)
    references = {}
    for rank in (0, 1):
        dyn_path = root / dynzdf_record_name(rank)
        zdf_path = zdf_reference_dir / zdfphy_record_name(rank)
        require(dyn_path.is_file(), f"missing {dyn_path.name}")
        require(zdf_path.is_file(), f"missing reference {zdf_path.name}")
        dynzdf = read_dynzdf_record(dyn_path)
        zdfphy = read_zdfphy_record(zdf_path)
        _verify_checksums(dynzdf); _verify_checksums(zdfphy)
        _verify_source_order(records[rank])
        _verify_references(records[rank], dynzdf, zdfphy)
        references[rank] = {"dynzdf": dyn_path.name, "zdfphy": str(zdf_path)}
    return {
        "format": "nemo-testcase-l4-orca2-round331-tke-v1",
        "status": "PASS_R331_TKE_ADMISSION",
        "producer_commit": produced,
        "records": {record_name(rank): records[rank]["sha256"] for rank in records},
        "references": references,
    }


def same_program(new_bld: Path, ref_bld: Path, patches: dict[str, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    require(tuple(patches) == OBSERVER_UNITS, "wrong TKE observer unit order")
    return same_program_observers(
        new_bld, ref_bld, patches, WRITER_UNITS,
        new_cfg=new_cfg, ref_cfg=ref_cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--zdf-reference-dir", type=Path)
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
            result = same_program(
                args.same_program, args.reference_bld,
                {unit: Path(path) for unit, path in entries},
                new_cfg=args.new_cfg, ref_cfg=args.reference_cfg)
        else:
            require(None not in (args.record_dir, args.zdf_reference_dir,
                                 args.expect_commit),
                    "admission needs record/reference dirs and expected commit")
            result = run_gate(args.record_dir, args.zdf_reference_dir,
                              args.expect_commit, args.plant)
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
