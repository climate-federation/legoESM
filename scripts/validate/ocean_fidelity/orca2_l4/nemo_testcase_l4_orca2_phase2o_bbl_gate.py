#!/usr/bin/env python3
"""Admission and schema gate for the ORCA2 diffusive-BBL discriminator."""

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
NAME = "oracle_bbl_diffusive_kt00000001.bin"
MAGIC = "NEMO_L4_BBLDF_1"
HEADER_BYTES = 16 + 13 * 4
FIELD_NAMES = (
    "temperature_Kbb", "salinity_Kbb",
    "temperature_Krhs_pre", "salinity_Krhs_pre",
    "temperature_Krhs_post", "salinity_Krhs_post",
    "ahu_bbl", "ahv_bbl",
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


def _mesh_masks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        t = np.asarray(dataset["tmask"][0], dtype=bool).transpose(2, 1, 0)
        u = np.asarray(dataset["umask"][0, 0], dtype=bool).T
        v = np.asarray(dataset["vmask"][0, 0], dtype=bool).T
    expected3 = (NX - 2 * HALO, NY - 2 * HALO, NZ)
    require(t.shape == expected3, f"unexpected tmask shape {t.shape}")
    require(u.shape == expected3[:2], f"unexpected umask shape {u.shape}")
    require(v.shape == expected3[:2], f"unexpected vmask shape {v.shape}")
    tf = np.zeros((NX, NY, NZ), dtype=bool)
    uf = np.zeros((NX, NY), dtype=bool)
    vf = np.zeros((NX, NY), dtype=bool)
    tf[HALO:-HALO, HALO:-HALO, :] = t
    uf[HALO:-HALO, HALO:-HALO] = u
    vf[HALO:-HALO, HALO:-HALO] = v
    return tf, uf, vf


def read_bbl(path: Path, mesh_path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=13i", handle.read(13 * 4))
        values = np.fromfile(handle, dtype=np.float64)
    n2, n3 = NX * NY, NX * NY * NZ
    payload = 6 * n3 + 2 * n2
    expected = (1, 1, 3, 1, 2, 3, 1, 0, NX, NY, NZ, 64, payload)
    require(magic == MAGIC, f"{path.name}: bad magic {magic!r}")
    require(header == expected, f"{path.name}: bad header {header}")
    require(values.size == payload,
            f"{path.name}: derived payload {values.size} != {payload}")
    require(path.stat().st_size == HEADER_BYTES + 8 * payload,
            f"{path.name}: derived EOF size mismatch")
    require(np.isfinite(values).all(), f"{path.name}: non-finite payload")
    arrays: dict[str, np.ndarray] = {}
    cursor = 0
    for name in FIELD_NAMES[:6]:
        arrays[name] = values[cursor:cursor + n3].reshape((NX, NY, NZ), order="F")
        cursor += n3
    for name in FIELD_NAMES[6:]:
        arrays[name] = values[cursor:cursor + n2].reshape((NX, NY), order="F")
        cursor += n2
    require(cursor == values.size, "BBL parser did not consume payload")
    tmask, umask, vmask = _mesh_masks(mesh_path)
    for name in FIELD_NAMES[:6]:
        require(np.all(arrays[name][~tmask] == 0.0),
                f"{path.name}: {name} nonzero canonical T slot")
    for name, mask in (("ahu_bbl", umask), ("ahv_bbl", vmask)):
        require(np.all(arrays[name][~mask] == 0.0),
                f"{path.name}: {name} nonzero canonical face slot")
    return {
        "path": str(path), "sha256": sha256(path),
        "bytes": path.stat().st_size, "magic": magic,
        "header": list(header), "derived_payload_f64": int(payload),
        "field_count": len(arrays), "fields": list(arrays),
        "finite": True, "canonical_slots": "ZERO",
    }


def _inventory(root: Path) -> set[str]:
    return {path.name for path in root.glob("oracle_*.bin")}


def validate_twins(a: Path, b: Path, inherited: Path) -> dict[str, object]:
    ia, ib, old = _inventory(a), _inventory(b), _inventory(inherited)
    require(ia == ib, f"twin inventories differ: {sorted(ia ^ ib)}")
    require(ia == old | {NAME}, "candidate inventory is not V2 plus BBL")
    twin_rows = []
    for name in sorted(ia):
        require((a / name).read_bytes() == (b / name).read_bytes(),
                f"twin record differs: {name}")
        twin_rows.append({"file": name, "sha256": sha256(a / name),
                          "status": "EXACT_BYTES"})
    inherited_rows = []
    for name in sorted(old):
        require((a / name).read_bytes() == (inherited / name).read_bytes(),
                f"inherited record differs: {name}")
        inherited_rows.append({"file": name, "sha256": sha256(a / name),
                               "status": "EXACT_BYTES"})
    return {
        "status": "PASS", "twin_raw_exact": len(twin_rows),
        "twin_total": len(twin_rows), "inherited_raw_exact": len(inherited_rows),
        "inherited_total": len(inherited_rows), "rows": twin_rows,
    }


def _overlay(root: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="orca2-l4-bbl-plant-"))
    for entry in root.iterdir():
        os.symlink(entry, target / entry.name)
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


def planted_controls(root: Path, witness: Path, inherited: Path,
                     control: Path) -> dict[str, str]:
    from netCDF4 import Dataset
    from scripts.validate.ocean_fidelity.testcases import (
        nemo_testcase_l4_orca2_phase1_gate as phase1,
    )

    results: dict[str, str] = {}

    def expect_stream(label: str, offset: int, payload: bytes) -> None:
        overlay = _overlay(root)
        try:
            changed = _materialize(overlay, root, NAME)
            _patch(changed, offset, payload)
            try:
                read_bbl(changed, overlay / "mesh_mask_0000.nc")
            except GateError:
                results[label] = "PASS_NONZERO"
            else:
                raise GateError(f"plant did not fail schema path: {label}")
        finally:
            shutil.rmtree(overlay)

    expect_stream("derived_header_count", 16 + 12 * 4,
                  struct.pack("=i", 1))
    expect_stream("canonical_zero", HEADER_BYTES, struct.pack("=d", 1.0))

    overlay = _overlay(root)
    try:
        changed = _materialize(overlay, root, NAME)
        linear = HALO + NX * (HALO + NY * 0)
        offset = HEADER_BYTES + 8 * linear
        with changed.open("r+b") as handle:
            handle.seek(offset)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(offset)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))
        try:
            validate_twins(overlay, witness, inherited)
        except GateError:
            results["owned_twin_payload"] = "PASS_NONZERO"
        else:
            raise GateError("owned twin payload plant passed")
    finally:
        shutil.rmtree(overlay)

    overlay = _overlay(root)
    try:
        restart = next(path.name for path in root.glob("*_restart_0000.nc"))
        changed = _materialize(overlay, root, restart)
        _patch(changed, changed.stat().st_size - 1, b"X")
        try:
            phase1.validate_identity(control, overlay)
        except (phase1.GateError, OSError, ValueError):
            results["ordinary_restart_identity"] = "PASS_NONZERO"
        else:
            raise GateError("restart identity plant passed")
    finally:
        shutil.rmtree(overlay)

    overlay = _overlay(root)
    try:
        history = phase1.HISTORY[0]
        changed = _materialize(overlay, root, history)
        with Dataset(changed, "r+") as dataset:
            variable = next(v for v in dataset.variables.values()
                            if v.size and np.issubdtype(v.dtype, np.floating))
            variable.set_auto_maskandscale(False)
            index = (0,) * variable.ndim
            dtype = np.dtype(variable.dtype)
            value = np.asarray(variable[index], dtype=dtype)
            variable[index] = np.nextafter(
                value, np.asarray(np.inf, dtype=dtype), dtype=dtype)
        try:
            phase1.validate_identity(control, overlay)
        except (phase1.GateError, OSError, ValueError):
            results["history_raw_payload"] = "PASS_NONZERO"
        else:
            raise GateError("history payload plant passed")
    finally:
        shutil.rmtree(overlay)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--inherited-v2", type=Path, required=True)
    parser.add_argument("--identity-control", type=Path, required=True)
    parser.add_argument("--plants", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        from scripts.validate.ocean_fidelity.testcases import (
            nemo_testcase_l4_orca2_phase1_gate as phase1,
        )
        result = {
            "run_a_schema": read_bbl(args.run_a / NAME, args.run_a / "mesh_mask_0000.nc"),
            "run_b_schema": read_bbl(args.run_b / NAME, args.run_b / "mesh_mask_0000.nc"),
            "records": validate_twins(args.run_a, args.run_b, args.inherited_v2),
            "ordinary_output_identity": phase1.validate_identity(
                args.identity_control, args.run_a),
        }
        if args.plants:
            result["plants"] = planted_controls(
                args.run_a, args.run_b, args.inherited_v2,
                args.identity_control)
        result["status"] = "PASS"
    except (GateError, OSError, UnicodeError, struct.error, ValueError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
