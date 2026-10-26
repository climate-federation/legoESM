#!/usr/bin/env python3
"""Admission gate for the ORCA2 Lane-4 EEN discriminator records."""

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


NX, NY, NZ = 94, 152, 31
HALO = 2
HEADER_BYTES = 16 + 10 * 4
NEW_STREAMS = {
    "oracle_een_e3f0vor_kt00000001.bin": ("NEMO_L4_E3F0_1", 1),
    "oracle_een_e3fvor_kt00000001.bin": ("NEMO_L4_E3FV_1", 1),
    "oracle_een_q_kt00000001.bin": ("NEMO_L4_QEEN_1", 1),
    "oracle_een_zpvo_kt00000001.bin": ("NEMO_L4_ZPVO_1", 8),
}
TRANSPORT_S1 = "oracle_transport_kt00000001_s1.bin"
TRANSPORT_HEADER_BYTES = 16 + 8 * 4


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


def _arrays(values: np.ndarray, nfields: int) -> list[np.ndarray]:
    n3 = NX * NY * NZ
    return [values[i * n3:(i + 1) * n3].reshape((NX, NY, NZ), order="F")
            for i in range(nfields)]


def _halo_mask() -> np.ndarray:
    mask = np.ones((NX, NY, NZ), dtype=bool)
    mask[HALO:-HALO, HALO:-HALO, :] = False
    return mask


def _face_masks(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(mesh_path) as dataset:
        # mesh_mask has no halo: (time,z,y,x).  Embed it into the rank-zero
        # local (i,j,k) writer shape so the mask and payload share indexing.
        um = np.asarray(dataset["umask"][0], dtype=bool).transpose(2, 1, 0)
        vm = np.asarray(dataset["vmask"][0], dtype=bool).transpose(2, 1, 0)
    require(um.shape == (NX - 2 * HALO, NY - 2 * HALO, NZ),
            f"unexpected U mask shape {um.shape}")
    require(vm.shape == um.shape, f"unexpected V mask shape {vm.shape}")
    ufull = np.zeros((NX, NY, NZ), dtype=bool)
    vfull = np.zeros_like(ufull)
    ufull[HALO:-HALO, HALO:-HALO, :] = um
    vfull[HALO:-HALO, HALO:-HALO, :] = vm
    return ufull, vfull


def read_stream(path: Path, magic_expected: str, nfields_expected: int,
                mesh_path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=10i", handle.read(40))
        values = np.fromfile(handle, dtype=np.float64)
    n3 = NX * NY * NZ
    expected_header = (
        1, 1, 1, 3, NX, NY, NZ, nfields_expected, 64,
        nfields_expected * n3,
    )
    require(magic == magic_expected, f"{path.name}: magic {magic!r}")
    require(header == expected_header, f"{path.name}: header {header}")
    require(values.size == nfields_expected * n3,
            f"{path.name}: derived payload {values.size} != {nfields_expected*n3}")
    require(path.stat().st_size == HEADER_BYTES + 8 * values.size,
            f"{path.name}: derived byte size mismatch")
    require(np.isfinite(values).all(), f"{path.name}: non-finite payload")
    arrays = _arrays(values, nfields_expected)
    halo = _halo_mask()
    for index, array in enumerate(arrays):
        require(np.all(array[halo] == 0.0),
                f"{path.name}: field {index} nonzero canonical halo")
    if "e3fvor" in path.name or "_q_" in path.name:
        require(np.all(arrays[0][:, :, -1] == 0.0),
                f"{path.name}: inactive jpk slot is not zero")
    if "zpvo" in path.name:
        umask, vmask = _face_masks(mesh_path)
        for index, array in enumerate(arrays):
            live = umask if index < 4 else vmask
            # dyn_cor_2D_init evaluates the primitive for jk=1..mbku/mbkv,
            # including jk=1 on dry columns (NEMO bottom indices are at
            # least one).  The subsequent coefficient accumulation applies
            # the neighbouring face mask.  Thus source-owned slots are the
            # loop extent, not merely live face cells.
            bottom = np.maximum(np.count_nonzero(live, axis=2), 1)
            source_owned = np.arange(NZ)[None, None, :] < bottom[:, :, None]
            require(np.all(array[~source_owned] == 0.0),
                    f"{path.name}: field {index} below-loop slot nonzero")
    return {
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "magic": magic, "header": list(header),
        "payload_f64": int(values.size), "fields": nfields_expected,
        "finite": True, "canonical_slots": "ZERO",
    }


def validate_new(root: Path) -> dict[str, object]:
    mesh = root / "mesh_mask_0000.nc"
    require(mesh.is_file(), f"missing {mesh}")
    rows = {}
    for name, (magic, nfields) in NEW_STREAMS.items():
        path = root / name
        require(path.is_file(), f"missing {path}")
        rows[name] = read_stream(path, magic, nfields, mesh)
    return {"status": "PASS", "count": len(rows), "rows": rows}


def _record_inventory(root: Path) -> set[str]:
    return {path.name for path in root.glob("oracle_*.bin")}


def _owned_bytes(array: np.ndarray) -> bytes:
    """Return rank-zero owned cells in a stable byte order."""
    return np.ascontiguousarray(
        array[HALO:-HALO, HALO:-HALO, :]
    ).tobytes()


def _transport_arrays(path: Path) -> tuple[bytes, list[np.ndarray]]:
    raw = path.read_bytes()
    n3 = NX * NY * NZ
    require(len(raw) == TRANSPORT_HEADER_BYTES + 3 * n3 * 8,
            f"{path.name}: unexpected transport byte size {len(raw)}")
    values = np.frombuffer(raw, dtype=np.float64, offset=TRANSPORT_HEADER_BYTES)
    return raw[:TRANSPORT_HEADER_BYTES], _arrays(values, 3)


def compare_transport_consumed(reference: Path, candidate: Path) -> dict[str, object]:
    """Admit only the pre-consumer zFw slot; zFu/zFv are consumed fields."""
    ref_header, ref_arrays = _transport_arrays(reference)
    got_header, got_arrays = _transport_arrays(candidate)
    require(ref_header == got_header,
            f"{candidate.name}: transport header differs")
    rows = []
    for index, field in enumerate(("zFu", "zFv")):
        ref_owned = _owned_bytes(ref_arrays[index])
        got_owned = _owned_bytes(got_arrays[index])
        require(ref_owned == got_owned,
                f"{candidate.name}: consumed {field} owned cells differ")
        rows.append({
            "field": field,
            "status": "EXACT_BYTES_OWNED_CONSUMED",
            "owned_cells": (NX - 2 * HALO) * (NY - 2 * HALO) * NZ,
        })
    ref_w, got_w = ref_arrays[2], got_arrays[2]
    xor = np.frombuffer(_owned_bytes(ref_w), dtype=np.uint8) != np.frombuffer(
        _owned_bytes(got_w), dtype=np.uint8)
    return {
        "status": "PASS_CONSUMED_FIELDS_EXACT",
        "header": "EXACT_BYTES",
        "consumed": rows,
        "unconsumed": {
            "field": "zFw",
            "status": "UNINFORMATIVE_PRE_CONSUMER_WORKSPACE",
            "owned_differing_bytes": int(np.count_nonzero(xor)),
            "raw_differing_elements": int(np.count_nonzero(
                ref_w.view(np.uint64) != got_w.view(np.uint64))),
        },
    }


def validate_twins(a: Path, b: Path, inherited: Path | None) -> dict[str, object]:
    names_a, names_b = _record_inventory(a), _record_inventory(b)
    require(names_a == names_b,
            f"twin inventory differs: only_a={sorted(names_a-names_b)} "
            f"only_b={sorted(names_b-names_a)}")
    require(NEW_STREAMS.keys() <= names_a, "twin inventory lacks EEN streams")
    rows = []
    for name in sorted(names_a):
        require((a / name).read_bytes() == (b / name).read_bytes(),
                f"twin record mismatch: {name}")
        rows.append({"file": name, "status": "EXACT_BYTES",
                     "sha256": sha256(a / name)})
    result: dict[str, object] = {
        "status": "PASS", "raw_exact": len(rows), "total": len(rows),
        "rows": rows,
    }
    if inherited is not None:
        inherited_names = _record_inventory(inherited)
        require(inherited_names == names_a - set(NEW_STREAMS),
                "inherited V2 inventory is not exactly candidate minus four new streams")
        inherited_rows = []
        consumed_exceptions = []
        for name in sorted(inherited_names):
            if name == TRANSPORT_S1:
                consumed_exceptions.append({
                    "file": name,
                    **compare_transport_consumed(inherited / name, a / name),
                })
            else:
                require((inherited / name).read_bytes() == (a / name).read_bytes(),
                        f"inherited raw-identity mismatch: {name}")
                inherited_rows.append({"file": name, "status": "EXACT_BYTES",
                                       "sha256": sha256(a / name)})
        result["inherited_raw_identity"] = {
            "status": "PASS", "raw_exact": len(inherited_rows),
            "consumed_exact": len(consumed_exceptions),
            "total": len(inherited_names), "rows": inherited_rows,
            "consumed_field_exceptions": consumed_exceptions,
        }
    return result


def _materialized_overlay(root: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="orca2-l4-een-plant-"))
    for name in NEW_STREAMS:
        os.symlink(root / name, target / name)
    os.symlink(root / "mesh_mask_0000.nc", target / "mesh_mask_0000.nc")
    return target


def _materialize(overlay: Path, root: Path, name: str) -> Path:
    path = overlay / name
    path.unlink()
    shutil.copyfile(root / name, path)
    return path


def _patch(path: Path, offset: int, data: bytes) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(data)


def planted_controls(root: Path) -> dict[str, str]:
    results: dict[str, str] = {}

    def expect(name: str, mutate) -> None:
        overlay = _materialized_overlay(root)
        try:
            mutate(overlay)
            try:
                validate_new(overlay)
            except GateError:
                results[name] = "PASS_NONZERO"
            else:
                raise GateError(f"plant did not fail through production checker: {name}")
        finally:
            shutil.rmtree(overlay)

    first = "oracle_een_e3f0vor_kt00000001.bin"
    expect("header_integer", lambda overlay: _patch(
        _materialize(overlay, root, first), 16 + 4, struct.pack("=i", 2)))

    # A defined e3f_0vor interior value is necessarily nonzero; advance it by
    # one ULP, then demand that twin raw identity rejects the altered record.
    with tempfile.TemporaryDirectory(prefix="orca2-l4-een-twin-plant-") as td:
        altered = Path(td)
        for entry in root.iterdir():
            if entry.name.startswith("oracle_") or entry.name == "mesh_mask_0000.nc":
                os.symlink(entry, altered / entry.name)
        changed = altered / first
        changed.unlink()
        shutil.copyfile(root / first, changed)
        offset = HEADER_BYTES + 8 * (HALO + NX * (HALO + NY * 0))
        with changed.open("r+b") as handle:
            handle.seek(offset)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(offset)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))
        try:
            validate_twins(root, altered, None)
        except GateError:
            results["defined_payload_bit"] = "PASS_NONZERO"
        else:
            raise GateError("defined-payload plant passed twin checker")

    # First element is a halo slot in every stream and must remain zero.
    expect("canonical_zero_slot", lambda overlay: _patch(
        _materialize(overlay, root, first), HEADER_BYTES,
        struct.pack("=d", 1.0)))

    # The exception itself must remain binding.  Change an owned zFu cell in a
    # materialized copy and call the same consumed-field comparator used by
    # baseline admission.
    with tempfile.TemporaryDirectory(prefix="orca2-l4-transport-plant-") as td:
        changed = Path(td) / TRANSPORT_S1
        shutil.copyfile(root / TRANSPORT_S1, changed)
        linear = HALO + NX * (HALO + NY * 0)
        offset = TRANSPORT_HEADER_BYTES + 8 * linear
        with changed.open("r+b") as handle:
            handle.seek(offset)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(offset)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))
        try:
            compare_transport_consumed(root / TRANSPORT_S1, changed)
        except GateError:
            results["consumed_transport_bit"] = "PASS_NONZERO"
        else:
            raise GateError("consumed-transport plant passed production checker")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path)
    parser.add_argument("--inherited-v2", type=Path)
    parser.add_argument("--identity-control", type=Path)
    parser.add_argument("--plants", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result: dict[str, object] = {"run_a": validate_new(args.run_a)}
        if args.run_b is not None:
            result["run_b"] = validate_new(args.run_b)
            result["twins"] = validate_twins(
                args.run_a, args.run_b, args.inherited_v2)
        if args.identity_control is not None:
            # Reuse the Phase-1 production identity gate, including raw restart
            # bytes, raw NetCDF data variables, and dynamic result counts.
            from scripts.validate.ocean_fidelity.testcases import (
                nemo_testcase_l4_orca2_phase1_gate as phase1,
            )
            result["ordinary_output_identity"] = phase1.validate_identity(
                args.identity_control, args.run_a)
        if args.plants:
            result["plants"] = planted_controls(args.run_a)
        result["status"] = "PASS"
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
