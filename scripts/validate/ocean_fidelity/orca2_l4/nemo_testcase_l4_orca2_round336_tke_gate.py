#!/usr/bin/env python3
"""Admit the passive, production-arithmetic ORCA2 round-336 kt=2 TKE walk."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
    nemo_testcase_l4_orca2_round336_trazdf_gate as trazdf,
)

MAGIC = "NEMO_L4_R336TKE"
HEADER_NAMES = (
    "version", "kt", "kbb", "kmm", "rank", "jpi", "jpj", "jpk",
    "ntsi", "ntei", "ntsj", "ntej", "real_bits", "n_fields",
)
FIELDS = (
    "entry_en", "entry_avt", "entry_avm", "entry_sh2", "entry_dissl",
    "entry_rn2", "entry_rn2b", "entry_drag_bottom", "entry_drag_top",
    "entry_nmln", "entry_hmlp", "entry_fr_i", "entry_hm_i", "entry_taum",
    "after_bc_en", "post_langmuir_en", "matrix_rhs_en", "matrix_diag",
    "matrix_upper", "matrix_lower", "after_solve_en", "after_tke_en",
    "tke_pdlr", "pre_sweep_mxl", "final_mxl", "final_mxld",
    "pre_pdlr_zav", "pre_pdlr_avt", "final_avm", "final_avt", "tke_wmask",
)
FULL_3D = {"entry_avm", "entry_rn2", "entry_rn2b", "final_avm", "tke_wmask"}
INTERIOR_2D = {
    "entry_drag_bottom", "entry_drag_top", "entry_nmln", "entry_hmlp",
    "entry_fr_i", "entry_hm_i", "entry_taum",
}
NEGATIVE_ZERO_EDGE = {
    "matrix_diag", "matrix_upper", "matrix_lower", "after_solve_en",
    "tke_pdlr", "pre_pdlr_zav", "pre_pdlr_avt",
}
REAL_LAYOUT = (94, 152, 31, 3, 92, 3, 150)
PLANTS = (
    "header", "field-order", "truncation", "checksum", "stamp", "extent",
    "entry", "rn2", "langmuir", "pre-sweep", "zav", "pdlr", "final",
    "signed-zero", "time-level",
)
OBSERVER_UNITS = ("dynzdf", "stprk3_stg", "zdftke")
WRITER_UNITS = ("dynzdf_round328_writer", "tke_round336_writer")

GateError = r331.GateError
require = r331.require


def record_name(rank: int) -> str:
    return f"oracle_r336_tke_kt00000002_r{rank:04d}.bin"


def _layout(header: dict[str, int]) -> tuple[int, ...]:
    return tuple(header[name] for name in (
        "jpi", "jpj", "jpk", "ntsi", "ntei", "ntsj", "ntej"))


def _extent(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    if name in FULL_3D:
        return header["jpi"], header["jpj"], header["jpk"]
    nx = header["ntei"] - header["ntsi"] + 1
    ny = header["ntej"] - header["ntsj"] + 1
    return nx, ny, 1 if name in INTERIOR_2D else header["jpk"]


def _origin(name: str, header: dict[str, int]) -> tuple[int, int, int]:
    if name in FULL_3D:
        return 1, 1, 1
    return header["ntsi"], header["ntsj"], 1


def read_record(path: Path, plant: str | None = None,
                expected_layout: tuple[int, ...] = REAL_LAYOUT) -> dict:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = r331._take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(HEADER_NAMES)}i",
        r331._take(stream, 4 * len(HEADER_NAMES), "header"),
    )
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant in {"header", "time-level"}:
        header["kmm"] = -1
    if plant == "extent":
        header["ntei"] -= 1
    require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")
    require((header["version"], header["kt"], header["kbb"], header["kmm"],
             header["real_bits"], header["n_fields"]) == (2, 2, 3, 3, 64, len(FIELDS)),
            "wrong version/kt/RK/field-count header")
    require(_layout(header) == expected_layout,
            f"wrong declared TKE layout {_layout(header)} != {expected_layout}")
    require(1 <= header["ntsi"] <= header["ntei"] <= header["jpi"],
            "invalid i bounds")
    require(1 <= header["ntsj"] <= header["ntej"] <= header["jpj"],
            "invalid j bounds")
    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = r331._take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", r331._take(stream, 28, f"{name} metadata"))
        require(rank == (2 if name in INTERIOR_2D else 3),
                f"{name}: bad array rank {rank}")
        require((n1, n2, n3) == _extent(name, header),
                f"{name}: wrong extent {(n1, n2, n3)}")
        require((i0, j0, k0) == _origin(name, header),
                f"{name}: wrong origin {(i0, j0, k0)}")
        checksum, = struct.unpack("=Q", r331._take(stream, 8, f"{name} checksum"))
        payload = r331._take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(
            (n1, n2, n3), order="F")
        require(bool(np.all(np.isfinite(array))), f"{name}: non-finite payload")
        require(name not in fields, f"duplicate field {name}")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    observed = tuple(reversed(order)) if plant == "field-order" else tuple(order)
    require(observed == FIELDS, "field order differs from TKE source order")
    record = {"header": header, "fields": fields, "checksums": checksums,
              "sha256": hashlib.sha256(raw).hexdigest()}
    if plant == "checksum":
        record["fields"][FIELDS[0]].flat[0] = np.nextafter(
            record["fields"][FIELDS[0]].flat[0], np.inf)
    r331._verify_checksums(record)
    return record


def _bits_equal(left: np.ndarray, right: np.ndarray, message: str) -> None:
    require(left.shape == right.shape, f"{message}: shape differs")
    require(np.array_equal(left.view(np.uint64), right.view(np.uint64)), message)


def _wet_index(wet: np.ndarray) -> tuple[int, ...]:
    indices = np.argwhere(wet)
    require(bool(indices.size), "record has no wet interface")
    return tuple(int(v) for v in indices[0])


def verify_record(record: dict, phase: dict | None = None,
                  plant: str | None = None) -> dict[str, int]:
    require(plant is None or plant in {
        "entry", "rn2", "langmuir", "pre-sweep", "zav", "pdlr", "final",
        "signed-zero"}, f"unsupported semantic plant {plant}")
    fields = record["fields"]
    header = record["header"]
    i0, i1 = header["ntsi"] - 1, header["ntei"]
    j0, j1 = header["ntsj"] - 1, header["ntej"]
    wet_full = fields["tke_wmask"] != 0.0
    wet = wet_full[i0:i1, j0:j1, :]
    index = _wet_index(wet)
    field_for_plant = {
        "entry": "entry_en", "rn2": "entry_rn2", "langmuir": "post_langmuir_en",
        "pre-sweep": "pre_sweep_mxl", "zav": "pre_pdlr_zav",
        "pdlr": "tke_pdlr", "final": "final_avt",
    }
    if plant in field_for_plant:
        name = field_for_plant[plant]
        target = fields[name]
        if name in FULL_3D:
            global_index = (index[0] + i0, index[1] + j0, index[2])
            target[global_index] = np.nan
        else:
            target[index] = np.nan
    for name, values in fields.items():
        require(bool(np.all(np.isfinite(values))),
                f"{name}: non-finite value after semantic checks")
    for name in NEGATIVE_ZERO_EDGE:
        edge = fields[name][..., -1]
        if plant == "signed-zero" and name == "matrix_diag":
            edge.flat[0] = 0.0
        require(bool(np.all(edge.view(np.uint64) == np.uint64(1 << 63))),
                f"{name}: unfilled bottom edge lost -0 sentinel")
    require(bool(np.all(fields["entry_nmln"] == np.floor(fields["entry_nmln"]))),
            "mixed-layer level is not integer-valued")
    interior = wet[..., 1:-1]
    require(bool(np.any(interior)), "record has no wet process-range interface")
    for name in ("pre_sweep_mxl", "final_mxl", "final_mxld", "pre_pdlr_zav",
                 "pre_pdlr_avt", "final_avt"):
        require(bool(np.all(fields[name][..., 1:-1][interior] >= 0.0)),
                f"{name}: negative wet process-range value")
    require(bool(np.all(fields["final_mxld"][..., 1:-1][interior] >=
                        fields["final_mxl"][..., 1:-1][interior])),
            "final wet dissipative length is below mixing length")
    pdlr = fields["tke_pdlr"][..., 1:-1][interior]
    require(bool(np.all((pdlr >= 0.1) & (pdlr <= 1.0))),
            "wet inverse Prandtl ratio is outside [0.1,1]")
    if phase is not None:
        _bits_equal(fields["final_avt"], phase["fields"]["avt_after_tke"],
                    "kt=2 final avt differs from admitted zdf_phy handoff")
    return {"wet_interfaces": int(np.count_nonzero(wet)),
            "wet_process_interfaces": int(np.count_nonzero(interior))}


def run_gate(root: Path, phase_root: Path, expect_commit: str,
             plant: str | None = None) -> dict[str, object]:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), "missing producer_commit.txt")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit,
            f"producer commit {produced} != {expect_commit}")
    phase_stamp = phase_root / "producer_commit.txt"
    require(phase_stamp.is_file(), "phase reference lacks producer_commit.txt")
    phase_producer = phase_stamp.read_text().strip()
    phase_admission = trazdf.run_gate(phase_root, expect_commit=phase_producer)
    require(phase_admission["status"] == "PASS_R336_TRAZDF_ACQUISITION",
            "round-336 phase reference is not admitted")
    counts: dict[str, object] = {}
    hashes: dict[str, object] = {}
    for rank in (0, 1):
        path = root / record_name(rank)
        phase_path = phase_root / trazdf.record_name(rank)
        require(path.is_file(), f"missing {path.name}")
        require(phase_path.is_file(), f"missing phase reference {phase_path.name}")
        parser_plant = plant if rank == 0 and plant in {
            "header", "field-order", "truncation", "checksum", "extent", "time-level"
        } else None
        record = read_record(path, plant=parser_plant)
        require(record["header"]["rank"] == rank, f"{path.name}: wrong rank")
        phase = trazdf.read_record(phase_path)
        trazdf._verify_checksums(phase)
        semantic_plant = plant if rank == 0 and plant in {
            "entry", "rn2", "langmuir", "pre-sweep", "zav", "pdlr", "final",
            "signed-zero"} else None
        counts[str(rank)] = verify_record(record, phase, semantic_plant)
        hashes[str(rank)] = {"tke": record["sha256"], "phase": phase["sha256"]}
    return {
        "format": "nemo-testcase-l4-orca2-round336-tke-v2",
        "status": "PASS_R336_TKE_KT2_ADMISSION",
        "producer_commit": produced,
        "phase_reference_producer": phase_producer,
        "counts": counts,
        "sha256": hashes,
    }


def same_program(new_bld: Path, ref_bld: Path, patches: dict[str, Path], *,
                 new_cfg: str, ref_cfg: str) -> dict:
    require(tuple(patches) == OBSERVER_UNITS, "wrong round-336 TKE observer unit order")
    return r331.same_program_observers(
        new_bld, ref_bld, patches, WRITER_UNITS,
        new_cfg=new_cfg, ref_cfg=ref_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--phase-reference-dir", type=Path)
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
            require(None not in (args.record_dir, args.phase_reference_dir,
                                 args.expect_commit),
                    "admission needs record/reference dirs and expected commit")
            result = run_gate(args.record_dir, args.phase_reference_dir,
                              args.expect_commit, args.plant)
            require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, OSError, UnicodeError, ValueError, KeyError, struct.error) as error:
        print(f"REFUSE: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_R336_TKE_KT2_ADMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
