#!/usr/bin/env python3
"""Admit the canonical kt=2 TKE-walk twins and their inherited V2 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)

MAGIC = "NEMO_L4_TKEW_1"
RECORD = "oracle_tke_walk_kt00000002.bin"
HEADER_INTS = 15
FIELDS_3D = (
    "dissl_pre", "zpelc", "en_post_lc", "pdlr", "zdiag_pre_solve", "zd_lw_pre_solve",
    "zd_up_pre_solve", "en_rhs_pre_solve", "zdiag_after_forward",
    "zd_lw_after_forward", "en_post_solve", "en_post_etau", "mxlm",
    "mxld", "avm_post", "avt_post", "dissl_post",
)
FIELDS_2D = (
    "zice_fra", "zWlc2", "imlc_real", "zhlc", "zus3", "htau", "hm_i",
)
FIELDS = FIELDS_3D + FIELDS_2D
REJECTED_FIELDS_3D = FIELDS_3D[1:]
REJECTED_FIELDS_2D = FIELDS_2D[:5]
REJECTED_FIELDS = REJECTED_FIELDS_3D + REJECTED_FIELDS_2D


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


def _canonical_masks(mesh: Path, jpi: int, jpj: int, jpk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(mesh) as dataset:
        native_t = np.asarray(dataset["tmask"][0], dtype=bool).transpose(2, 1, 0)
    require(native_t.shape == (jpi - 4, jpj - 4, jpk),
            f"unexpected rank-zero tmask shape {native_t.shape}")
    matrix = np.zeros((jpi, jpj, jpk), dtype=bool)
    water_column = np.zeros_like(matrix)
    bottom = np.count_nonzero(native_t, axis=2)
    for ii in range(native_t.shape[0]):
        for jj in range(native_t.shape[1]):
            if bottom[ii, jj]:
                water_column[ii + 2, jj + 2, :min(bottom[ii, jj] + 1, jpk)] = True
                # The writer's matrix-workspace ownership is its explicit
                # surface-wet + jk<=mbkt copy loop.  Use that allocation
                # contract, rather than a post-LBC mesh mask whose fold row
                # can intentionally differ from the local loop domain.
                matrix[ii + 2, jj + 2, :min(bottom[ii, jj], jpk - 1)] = True
    surface = matrix[..., :1]
    return matrix, water_column, surface


def read_tke(path: Path, mesh: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, "truncated magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(4 * HEADER_INTS)
        require(len(raw_header) == 4 * HEADER_INTS, "truncated base header")
        header = struct.unpack(f"={HEADER_INTS}i", raw_header)
        (version, kt, kbb, kmm, jpi, jpj, jpk, real_bits, n3, n2, payload,
         nn_eice, nn_etau, nn_mxl, nn_pdl) = header
        require(magic == MAGIC, f"magic {magic!r}")
        require((version, kt, kbb, kmm) == (1, 2, 3, 3),
                f"version/levels {version}/{kt}/{kbb}/{kmm}")
        require((real_bits, n3, n2) == (64, len(FIELDS_3D), len(FIELDS_2D)),
                f"precision/counts {real_bits}/{n3}/{n2}")
        require((nn_eice, nn_etau, nn_mxl, nn_pdl) == (1, 1, 3, 1),
                f"resolved selectors {nn_eice}/{nn_etau}/{nn_mxl}/{nn_pdl}")
        raw_extents = handle.read(4 * 3 * len(FIELDS))
        require(len(raw_extents) == 4 * 3 * len(FIELDS), "truncated extents")
        extents = np.frombuffer(raw_extents, dtype=np.int32).reshape(
            (len(FIELDS), 3))
        expected = [(jpi, jpj, jpk)] * len(FIELDS_3D) + [
            (jpi, jpj, 1)] * len(FIELDS_2D)
        require(tuple(map(tuple, extents.tolist())) == tuple(expected),
                "extent table differs from writer SHAPE expressions")
        derived = sum(int(np.prod(shape)) for shape in expected)
        require(payload == derived, f"payload {payload} != derived {derived}")

        matrix_mask, water_column_mask, surface_mask = _canonical_masks(
            mesh, jpi, jpj, jpk)
        matrix_fields = {
            "pdlr", "zdiag_pre_solve", "zd_lw_pre_solve", "zd_up_pre_solve",
            "zdiag_after_forward", "zd_lw_after_forward",
        }
        rows = []
        arrays: dict[str, np.ndarray] = {}
        for name, shape in zip(FIELDS, expected, strict=True):
            count = int(np.prod(shape))
            raw = handle.read(8 * count)
            require(len(raw) == 8 * count, f"{name}: truncated payload")
            value = np.frombuffer(raw, dtype=np.float64).reshape(shape, order="F")
            require(np.isfinite(value).all(), f"{name}: non-finite")
            if name in FIELDS_2D:
                allowed = surface_mask
            elif name in matrix_fields:
                allowed = matrix_mask
            else:
                allowed = water_column_mask
            require(np.all(value[~allowed] == 0.0),
                    f"{name}: noncanonical unowned/undefined slot")
            arrays[name] = value
            rows.append({"name": name, "extent": list(shape), "count": count,
                         "sha256": hashlib.sha256(raw).hexdigest()})
        require(handle.read(1) == b"", "trailing bytes")

    expected_bytes = 16 + 4 * HEADER_INTS + 4 * 3 * len(FIELDS) + 8 * payload
    require(path.stat().st_size == expected_bytes, "exact-EOF byte count")
    require(np.count_nonzero(arrays["zWlc2"]) > 0, "vacuous Langmuir operand")
    require(np.count_nonzero(arrays["en_post_solve"]) > 0, "vacuous solve target")
    require(np.count_nonzero(arrays["avm_post"]) > 0, "vacuous closure target")
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "magic": magic, "header": list(header), "extent_table": extents.tolist(),
        "extent_table_sha256": hashlib.sha256(raw_extents).hexdigest(),
        "extent_derived_payload_f64": derived, "fields": rows,
        "nonzero": {name: int(np.count_nonzero(arrays[name])) for name in FIELDS},
        "exact_eof": True,
    }


def audit_rejected_twins(run_a: Path, run_b: Path) -> dict[str, object]:
    """Map every differing byte through the frozen self-describing schema."""
    path_a, path_b = run_a / RECORD, run_b / RECORD
    raw_a, raw_b = path_a.read_bytes(), path_b.read_bytes()
    require(len(raw_a) == len(raw_b), "rejected records have unequal sizes")
    require(raw_a[:16 + 4 * HEADER_INTS] == raw_b[:16 + 4 * HEADER_INTS],
            "rejected records have unequal base headers")
    header = struct.unpack(f"={HEADER_INTS}i", raw_a[16:16 + 4 * HEADER_INTS])
    jpi, jpj, jpk = header[4:7]
    extent_offset = 16 + 4 * HEADER_INTS
    payload_offset = extent_offset + 4 * 3 * len(REJECTED_FIELDS)
    require(raw_a[extent_offset:payload_offset] == raw_b[extent_offset:payload_offset],
            "rejected records have unequal extent headers")
    extents = np.frombuffer(
        raw_a[extent_offset:payload_offset], dtype=np.int32).reshape(
            len(REJECTED_FIELDS), 3)

    offset = payload_offset
    affected = []
    all_byte_indices = []
    for name, shape_raw in zip(REJECTED_FIELDS, extents, strict=True):
        shape = tuple(int(value) for value in shape_raw)
        nbytes = 8 * int(np.prod(shape))
        part_a = np.frombuffer(raw_a[offset:offset + nbytes], dtype=np.uint8)
        part_b = np.frombuffer(raw_b[offset:offset + nbytes], dtype=np.uint8)
        byte_indices = np.flatnonzero(part_a != part_b)
        if byte_indices.size:
            values_a = np.frombuffer(raw_a[offset:offset + nbytes], dtype=np.float64)
            values_b = np.frombuffer(raw_b[offset:offset + nbytes], dtype=np.float64)
            element_indices = np.flatnonzero(
                values_a.view(np.uint64) != values_b.view(np.uint64))
            levels = sorted({
                int(np.unravel_index(int(index), shape, order="F")[2]) + 1
                for index in element_indices
            })
            file_indices = offset + byte_indices
            all_byte_indices.extend(int(value) for value in file_indices)
            affected.append({
                "field": name,
                "byte_differences": int(byte_indices.size),
                "element_differences": int(element_indices.size),
                "levels_1_based": levels,
                "first_file_offset_0_based": int(file_indices[0]),
                "last_file_offset_0_based": int(file_indices[-1]),
                "first_file_offset_1_based": int(file_indices[0]) + 1,
                "last_file_offset_1_based": int(file_indices[-1]) + 1,
            })
        offset += nbytes
    require(offset == len(raw_a), "schema does not reach exact EOF")
    require({row["field"] for row in affected} == {
        "zdiag_pre_solve", "zd_lw_pre_solve", "zd_up_pre_solve",
        "zdiag_after_forward", "zd_lw_after_forward",
    }, "difference escaped the five tridiagonal workspace fields")
    require(all(row["levels_1_based"] == [jpk] for row in affected),
            "difference escaped the undefined jpk workspace level")
    require(len(all_byte_indices) == 54889,
            f"unexpected differing-byte census {len(all_byte_indices)}")
    return {
        "status": "REJECTED_NONREPRODUCIBLE_TKE_RECORD",
        "record": RECORD,
        "record_a_sha256": sha256(path_a),
        "record_b_sha256": sha256(path_b),
        "bytes_each": len(raw_a),
        "differing_bytes": len(all_byte_indices),
        "first_file_offset_0_based": min(all_byte_indices),
        "last_file_offset_0_based": max(all_byte_indices),
        "first_file_offset_1_based": min(all_byte_indices) + 1,
        "last_file_offset_1_based": max(all_byte_indices) + 1,
        "jpk": jpk,
        "jpkm1": jpk - 1,
        "affected_fields": affected,
        "localization": (
            "persistent l4_tke_* arrays were zeroed, but the writer copied "
            "local zdiag/zd_lw/zd_up workspace level jpk=31 over those zeros; "
            "NEMO defines these workspaces only at the surface and jk=2:jpkm1=30"
        ),
        "physics_disposition": "UNOWNED_UNDEFINED_WORKSPACE_SLOTS_ONLY",
    }


def validate(run_a: Path, run_b: Path, baseline: Path,
             identity_control: Path) -> dict[str, object]:
    names_a = {p.name for p in run_a.glob("oracle_*.bin")}
    names_b = {p.name for p in run_b.glob("oracle_*.bin")}
    names_old = {p.name for p in baseline.glob("oracle_*.bin")}
    require(names_a == names_b == names_old | {RECORD},
            "inventory is not the 100-record V2 root plus one TKE record")
    twins = []
    for name in sorted(names_a):
        require((run_a / name).read_bytes() == (run_b / name).read_bytes(),
                f"twin differs: {name}")
        twins.append({"file": name, "sha256": sha256(run_a / name)})
    inherited = []
    for name in sorted(names_old):
        require((run_a / name).read_bytes() == (baseline / name).read_bytes(),
                f"inherited record differs: {name}")
        inherited.append({"file": name, "sha256": sha256(run_a / name)})
    return {
        "status": "PASS", "schema": read_tke(
            run_a / RECORD, baseline / "mesh_mask_0000.nc"),
        "twin_raw_exact": len(twins), "twin_total": len(twins),
        "inherited_raw_exact": len(inherited), "inherited_total": len(inherited),
        "records": twins,
        "ordinary_output_identity": phase1.validate_identity(identity_control, run_a),
    }


def _overlay(root: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="orca2-l4-p2u-tke-plant-"))
    for item in root.iterdir():
        os.symlink(item, target / item.name)
    target_record = target / RECORD
    target_record.unlink()
    shutil.copyfile(root / RECORD, target_record)
    return target


def _patch(path: Path, offset: int, raw: bytes) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(raw)


def run_plant(name: str, run_a: Path, run_b: Path, baseline: Path,
              identity_control: Path) -> None:
    planted = _overlay(run_a)
    record = planted / RECORD
    try:
        if name == "magic":
            _patch(record, 0, b"X")
        elif name == "extent":
            _patch(record, 16 + 4 * HEADER_INTS, struct.pack("=i", 93))
        elif name == "count":
            _patch(record, 16 + 4 * 10, struct.pack("=i", 1))
        elif name == "truncated":
            record.write_bytes(record.read_bytes()[:-8])
        elif name == "trailing":
            with record.open("ab") as handle:
                handle.write(b"X")
        elif name == "canonical":
            payload_offset = 16 + 4 * HEADER_INTS + 4 * 3 * len(FIELDS)
            _patch(record, payload_offset, struct.pack("=d", 1.0))
        elif name == "workspace_jpk":
            payload_offset = 16 + 4 * HEADER_INTS + 4 * 3 * len(FIELDS)
            field_values = 94 * 152 * 31
            # Plant an interior jk=jpk slot in zdiag_pre_solve; derive the
            # field index from the frozen inventory so schema growth cannot
            # silently move this control into another payload.
            # which the pre-fix writer copied from an undefined local workspace.
            field_index = FIELDS_3D.index("zdiag_pre_solve")
            element = field_index * field_values + 2 + 2 * 94 + 30 * 94 * 152
            _patch(record, payload_offset + 8 * element, struct.pack("=d", 1.0))
        elif name == "twin":
            _patch(record, 16 + 4 * HEADER_INTS + 4 * 3 * len(FIELDS),
                   struct.pack("=d", 1.0))
            validate(planted, run_b, baseline, identity_control)
            raise GateError("twin plant escaped")
        else:
            raise GateError(f"unknown plant {name}")
        read_tke(record, baseline / "mesh_mask_0000.nc")
        raise GateError(f"{name} plant escaped")
    except GateError as exc:
        if "escaped" in str(exc):
            raise
        print(f"FAIL: binding {name} plant rejected: {exc}")
        raise SystemExit(1)
    finally:
        shutil.rmtree(planted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--identity-control", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--audit-rejected", action="store_true")
    parser.add_argument("--plant", choices=("magic", "extent", "count", "truncated",
                                             "trailing", "canonical",
                                             "workspace_jpk", "twin"))
    args = parser.parse_args()
    if args.audit_rejected:
        report = audit_rejected_twins(args.run_a, args.run_b)
        text = json.dumps(report, indent=2) + "\n"
        if args.json_out:
            args.json_out.write_text(text)
        print(text, end="")
        return 0
    if args.plant:
        run_plant(args.plant, args.run_a, args.run_b, args.baseline,
                  args.identity_control)
    report = validate(args.run_a, args.run_b, args.baseline, args.identity_control)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {report['twin_raw_exact']}/{report['twin_total']} twins; "
          f"{report['inherited_raw_exact']}/{report['inherited_total']} inherited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
