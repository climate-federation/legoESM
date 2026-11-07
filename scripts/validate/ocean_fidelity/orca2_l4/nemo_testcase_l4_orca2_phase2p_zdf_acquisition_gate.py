#!/usr/bin/env python3
"""Schema, canonical-slot, twin, and inherited-record gate for Phase 2p ZDF."""

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
NAME = "oracle_zdf_sh2_operands_kt00000001.bin"
ZDF_ENTRY = "oracle_zdf_entry_kt00000001.bin"
SI3_ZDF = "oracle_si3_zdf_inputs.bin"
EEN_FILES = {
    "oracle_een_e3f0vor_kt00000001.bin",
    "oracle_een_e3fvor_kt00000001.bin",
    "oracle_een_q_kt00000001.bin",
    "oracle_een_zpvo_kt00000001.bin",
}
MAGIC = "NEMO_L4_ZSH2_1"
HEADER_INTS = 13
HEADER_BYTES = 16 + 4 * HEADER_INTS
FIELDS_3D = (
    "sh2", "avm_k_pre", "avt_k_pre", "en_pre", "rn2", "rn2b",
    "u_Kbb", "u_Kmm", "v_Kbb", "v_Kmm",
    "e3uw_Kbb", "e3uw_Kmm", "e3vw_Kbb", "e3vw_Kmm",
    "umask", "vmask", "wumask", "wvmask", "gdepw_Kmm",
    "e3t_Kmm", "e3w_Kmm",
)
FIELDS_2D = ("taum", "fr_i", "rCdU_bot", "mbkt_real")
GRID_3D = (
    "W", "W", "W", "W", "W", "W", "U", "U", "V", "V",
    "X", "X", "Y", "Y", "U", "V", "X", "Y", "W", "T", "W",
)
ALLOCATION_3D = (
    "reduced", "full", "reduced", "reduced", *(["full"] * 17),
)
ALLOCATION_2D = ("reduced", "full", "full", "full")


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


def _rank0_masks(mesh: Path) -> dict[tuple[str, str], np.ndarray]:
    from netCDF4 import Dataset

    # The retained rank-zero mesh contains NEMO's post-lbc T/U/V masks,
    # including its T-pivot fold row; deriving from bottom_level alone would
    # silently miss those owned fold values.
    with Dataset(mesh) as dataset:
        native = {
            grid: np.asarray(dataset[name][0], dtype=bool).transpose(2, 1, 0)
            for grid, name in {"T": "tmask", "U": "umask", "V": "vmask"}.items()
        }
    require(all(value.shape == (NX - 2 * HALO, NY - 2 * HALO, NZ)
                for value in native.values()), "unexpected rank-zero mesh mask shape")
    w_native = np.zeros_like(native["T"])
    wu_native = np.zeros_like(native["U"])
    wv_native = np.zeros_like(native["V"])
    w_native[..., 0] = native["T"][..., 0]
    wu_native[..., 0] = native["U"][..., 0]
    wv_native[..., 0] = native["V"][..., 0]
    w_native[..., 1:] = native["T"][..., 1:] * native["T"][..., :-1]
    wu_native[..., 1:] = native["U"][..., 1:] * native["U"][..., :-1]
    wv_native[..., 1:] = native["V"][..., 1:] * native["V"][..., :-1]
    native.update({"W": w_native, "X": wu_native, "Y": wv_native})

    masks: dict[tuple[str, str], np.ndarray] = {}
    for grid, source in native.items():
        local = np.zeros((NX, NY, NZ), dtype=bool)
        local[HALO:-HALO, HALO:-HALO] = source
        masks[(grid, "full")] = local
        masks[(grid, "reduced")] = source
    return masks


def read_zdf(path: Path, mesh: Path) -> dict[str, object]:
    full2, full3 = NX * NY, NX * NY * NZ
    reduced2 = (NX - 2 * HALO) * (NY - 2 * HALO)
    reduced3 = reduced2 * NZ
    sizes3 = {"full": full3, "reduced": reduced3}
    sizes2 = {"full": full2, "reduced": reduced2}
    payload = sum(sizes3[kind] for kind in ALLOCATION_3D)
    payload += sum(sizes2[kind] for kind in ALLOCATION_2D)
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack(f"={HEADER_INTS}i", handle.read(4 * HEADER_INTS))
        values = np.fromfile(handle, dtype=np.float64)
    expected = (1, 1, 1, 1, 3, NX, NY, NZ, 64,
                len(FIELDS_3D), len(FIELDS_2D), payload, 1)
    require(magic == MAGIC, f"bad magic {magic!r}")
    require(header == expected, f"bad header {header!r}")
    require(values.size == payload, f"derived payload {values.size} != {payload}")
    require(path.stat().st_size == HEADER_BYTES + 8 * payload,
            "derived EOF size mismatch")
    require(np.isfinite(values).all(), "non-finite payload")

    masks = _rank0_masks(mesh)
    cursor = 0
    arrays: dict[str, np.ndarray] = {}
    canonical: dict[str, str] = {}
    for name, grid, allocation in zip(
            FIELDS_3D, GRID_3D, ALLOCATION_3D, strict=True):
        shape = ((NX, NY, NZ) if allocation == "full"
                 else (NX - 2 * HALO, NY - 2 * HALO, NZ))
        count = sizes3[allocation]
        field = values[cursor:cursor + count].reshape(shape, order="F")
        cursor += count
        require(np.all(field[~masks[(grid, allocation)]] == 0.0),
                f"{name}: nonzero unowned/halo/land slot")
        arrays[name] = field
        canonical[name] = f"ZERO_OUTSIDE_{grid}_{allocation.upper()}"
    for name, allocation in zip(FIELDS_2D, ALLOCATION_2D, strict=True):
        shape = ((NX, NY) if allocation == "full"
                 else (NX - 2 * HALO, NY - 2 * HALO))
        count = sizes2[allocation]
        field = values[cursor:cursor + count].reshape(shape, order="F")
        cursor += count
        t2 = masks[("T", allocation)][..., 0]
        require(np.all(field[~t2] == 0.0), f"{name}: nonzero inactive T slot")
        arrays[name] = field
        canonical[name] = f"ZERO_OUTSIDE_T_SURFACE_{allocation.upper()}"
    require(cursor == values.size, "parser did not consume payload")
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "magic": magic, "header": list(header), "derived_payload_f64": payload,
        "fields_3d": [
            {"name": name, "allocation": allocation}
            for name, allocation in zip(FIELDS_3D, ALLOCATION_3D, strict=True)
        ],
        "fields_2d": [
            {"name": name, "allocation": allocation}
            for name, allocation in zip(FIELDS_2D, ALLOCATION_2D, strict=True)
        ],
        "canonical_slots": canonical,
    }


def read_zdf_entry(path: Path) -> dict[str, object]:
    n3 = NX * NY * NZ
    ni3 = (NX - 2 * HALO) * (NY - 2 * HALO) * NZ
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    payload = n3 + 3 * ni3
    require(magic == "NEMO_L4_ZDF___2", f"{path.name}: bad magic {magic!r}")
    require(header == (2, 1, 1, NX, NY, NZ, 64),
            f"{path.name}: bad header {header}")
    require(values.size == payload,
            f"{path.name}: derived payload {values.size} != {payload}")
    require(path.stat().st_size == 16 + 28 + 8 * payload,
            f"{path.name}: exact EOF mismatch")
    require(np.isfinite(values).all(), f"{path.name}: non-finite payload")
    fields = (
        ("avm", n3, "full jpi*jpj*jpk"),
        ("avt", ni3, "rank0 interior"),
        ("avs", ni3, "rank0 interior"),
        ("en", ni3, "rank0 interior"),
    )
    cursor = 0
    rows = []
    for field, count, allocation in fields:
        rows.append({"field": field, "count": count,
                     "allocation": allocation})
        cursor += count
    require(cursor == values.size, f"{path.name}: schema did not reach EOF")
    return {"path": str(path), "sha256": sha256(path),
            "bytes": path.stat().st_size, "magic": magic,
            "header": list(header), "derived_payload_f64": payload,
            "fields": rows, "finite": True}


def read_si3_zdf(path: Path) -> dict[str, object]:
    frames = []
    last_kt = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            magic_raw = handle.read(16)
            if not magic_raw:
                break
            require(len(magic_raw) == 16, f"{path.name}: truncated magic")
            magic = magic_raw.decode("ascii").rstrip()
            require(magic == "NEMO_L3ZIN_002",
                    f"{path.name}: bad magic {magic!r}")
            raw = handle.read(24)
            require(len(raw) == 24, f"{path.name}: truncated header")
            version, kt, kl, npti, real_bits, count = struct.unpack("=6i", raw)
            require(version == 2 and real_bits == 64,
                    f"{path.name}: bad version/storage {(version, real_bits)}")
            require(npti > 0 and count == (13 + 5) * npti,
                    f"{path.name}: derived payload {count} != 18*{npti}")
            require(kt >= last_kt, f"{path.name}: non-monotone kt {last_kt}->{kt}")
            last_kt = kt
            raw_values = handle.read(8 * count)
            require(len(raw_values) == 8 * count,
                    f"{path.name}: truncated payload")
            values = np.frombuffer(raw_values, dtype=np.float64)
            require(np.isfinite(values).all(), f"{path.name}: non-finite payload")
            frames.append({"kt": kt, "kl": kl, "npti": npti,
                           "derived_payload_f64": count})
    require(len(frames) == 25, f"{path.name}: frame count {len(frames)} != 25")
    expected_kt_kl = [(kt, kl) for kt in (1, 3, 5, 7, 9)
                      for kl in (1, 2, 3, 4, 5)]
    require([(row["kt"], row["kl"]) for row in frames] == expected_kt_kl,
            f"{path.name}: frame cadence is not nn_fsbc=2 x five categories")
    return {"path": str(path), "sha256": sha256(path),
            "bytes": path.stat().st_size, "magic": "NEMO_L3ZIN_002",
            "frames": len(frames), "first": frames[0], "last": frames[-1],
            "derived_count_expression": "(13+nlay_s)*npti; nlay_s=5",
            "finite": True, "exact_eof": True}


def _artifact_rows(root: Path) -> list[dict[str, object]]:
    names = {"ocean.output", "time.step"}
    names.update(path.name for path in root.glob("*restart*.nc"))
    names.update(path.name for path in root.glob("ORCA2_1*.nc"))
    return [
        {"file": name, "bytes": (root / name).stat().st_size,
         "sha256": sha256(root / name)}
        for name in sorted(names) if (root / name).is_file()
    ]


def validate(a: Path, b: Path, inherited: Path, mesh: Path,
             control: Path, rejected: Path | None = None) -> dict[str, object]:
    from scripts.validate.ocean_fidelity.testcases import (
        nemo_testcase_l4_orca2_phase1_gate as phase1,
    )

    ia = {p.name for p in a.glob("oracle_*.bin")}
    ib = {p.name for p in b.glob("oracle_*.bin")}
    old = {p.name for p in inherited.glob("oracle_*.bin")}
    require(ia == ib, f"twin inventories differ: {sorted(ia ^ ib)}")
    expected = (old - EEN_FILES) | {NAME}
    require(ia == expected,
            "candidate inventory is not (Phase-2n V2 minus EEN) plus ZDF")
    twin_rows = []
    for name in sorted(ia):
        require((a / name).read_bytes() == (b / name).read_bytes(),
                f"twin record differs: {name}")
        twin_rows.append({"file": name, "sha256": sha256(a / name),
                          "status": "EXACT_BYTES"})
    inherited_rows = []
    shared = old & ia
    require(len(shared) == 94, f"shared inherited inventory {len(shared)} != 94")
    for name in sorted(shared):
        require((a / name).read_bytes() == (inherited / name).read_bytes(),
                f"inherited record differs: {name}")
        inherited_rows.append({"file": name, "sha256": sha256(a / name),
                               "status": "EXACT_BYTES"})
    rejected_row = None
    if rejected is not None:
        rejected_inventory = {p.name for p in rejected.glob("oracle_*.bin")}
        require(rejected_inventory == ia,
                "rejected Phase-2p inventory differs from replacement")
        unchanged = rejected_inventory - {NAME}
        for name in sorted(unchanged):
            require((a / name).read_bytes() == (rejected / name).read_bytes(),
                    f"replacement changed non-ZDF record: {name}")
        require((a / NAME).read_bytes() != (rejected / NAME).read_bytes(),
                "replacement ZDF record did not change its rejected header")
        rejected_row = {
            "status": "SUPERSEDED_REJECTED_MALFORMED_ZDF_HEADER",
            "raw_exact_excluding_zdf": len(unchanged),
            "total_excluding_zdf": len(unchanged),
            "zdf_old_sha256": sha256(rejected / NAME),
            "zdf_new_sha256": sha256(a / NAME),
        }
    return {
        "status": "PASS",
        "schemas": {
            "zdf_sh2": read_zdf(a / NAME, mesh),
            "zdf_entry": read_zdf_entry(a / ZDF_ENTRY),
            "si3_zdf": read_si3_zdf(a / SI3_ZDF),
        },
        "twin_raw_exact": len(twin_rows), "twin_total": len(twin_rows),
        "inherited_raw_exact": len(inherited_rows),
        "inherited_total": len(inherited_rows), "records": twin_rows,
        "ordinary_output_identity": phase1.validate_identity(control, a),
        "run_a_artifacts": _artifact_rows(a),
        "run_b_artifacts": _artifact_rows(b),
        "omitted_een_streams": {
            "status": "REFUTED_PHASE2P_99_RECORD_PREDICTION",
            "files": sorted(EEN_FILES),
            "disposition": "remain pinned on Phase-2m twin A",
        },
        "rejected_phase2p_comparison": rejected_row,
    }


def _overlay(root: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="orca2-l4-zdf-plant-"))
    for entry in root.iterdir():
        os.symlink(entry, target / entry.name)
    return target


def _materialize(overlay: Path, root: Path, name: str) -> Path:
    target = overlay / name
    target.unlink()
    shutil.copyfile(root / name, target)
    return target


def _patch(path: Path, offset: int, value: bytes) -> None:
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(value)


def planted_controls(a: Path, b: Path, inherited: Path, mesh: Path,
                     control: Path) -> dict[str, str]:
    from netCDF4 import Dataset
    from scripts.validate.ocean_fidelity.testcases import (
        nemo_testcase_l4_orca2_phase1_gate as phase1,
    )

    results: dict[str, str] = {}

    def expect_schema(label: str, name: str, offset: int, value: bytes,
                      reader) -> None:
        overlay = _overlay(a)
        try:
            changed = _materialize(overlay, a, name)
            _patch(changed, offset, value)
            try:
                reader(changed)
            except (GateError, OSError, UnicodeError, struct.error, ValueError):
                results[label] = "PASS_NONZERO"
            else:
                raise GateError(f"plant did not fail real schema path: {label}")
        finally:
            shutil.rmtree(overlay)

    expect_schema("zdf_sh2_header_count", NAME, 16 + 11 * 4,
                  struct.pack("=i", 1), lambda p: read_zdf(p, mesh))
    masks = _rank0_masks(mesh)
    first_canonical = int(np.flatnonzero(
        ~masks[("W", "reduced")].ravel(order="F"))[0])
    expect_schema("zdf_sh2_canonical_zero", NAME,
                  HEADER_BYTES + 8 * first_canonical,
                  struct.pack("=d", 1.0), lambda p: read_zdf(p, mesh))
    expect_schema("zdf_entry_header", ZDF_ENTRY, 16,
                  struct.pack("=i", 99), read_zdf_entry)
    expect_schema("si3_zdf_header", SI3_ZDF, 16,
                  struct.pack("=i", 99), read_si3_zdf)

    overlay = _overlay(a)
    try:
        changed = _materialize(overlay, a, NAME)
        # First owned T value in the first 3-D field; the twin check binds it.
        linear = int(np.flatnonzero(
            masks[("W", "reduced")].ravel(order="F"))[0])
        with changed.open("r+b") as handle:
            handle.seek(HEADER_BYTES + 8 * linear)
            value = struct.unpack("=d", handle.read(8))[0]
            handle.seek(HEADER_BYTES + 8 * linear)
            handle.write(struct.pack("=d", np.nextafter(value, np.inf)))
        try:
            validate(overlay, b, inherited, mesh, control)
        except GateError:
            results["owned_twin_payload"] = "PASS_NONZERO"
        else:
            raise GateError("owned ZDF payload plant passed")
    finally:
        shutil.rmtree(overlay)

    overlay = _overlay(a)
    try:
        restart = next(path.name for path in a.glob("*_restart_0000.nc"))
        changed = _materialize(overlay, a, restart)
        _patch(changed, changed.stat().st_size - 1, b"X")
        try:
            phase1.validate_identity(control, overlay)
        except (phase1.GateError, OSError, ValueError):
            results["restart_identity"] = "PASS_NONZERO"
        else:
            raise GateError("restart identity plant passed")
    finally:
        shutil.rmtree(overlay)

    overlay = _overlay(a)
    try:
        history = phase1.HISTORY[0]
        changed = _materialize(overlay, a, history)
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
            results["history_payload"] = "PASS_NONZERO"
        else:
            raise GateError("history payload plant passed")
    finally:
        shutil.rmtree(overlay)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--inherited", type=Path, required=True)
    parser.add_argument("--identity-control", type=Path, required=True)
    parser.add_argument("--rejected-phase2p", type=Path)
    parser.add_argument("--mesh", type=Path, required=True,
                        help="rank-zero mesh_mask_0000.nc from either twin")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plants", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate(args.run_a, args.run_b, args.inherited, args.mesh,
                          args.identity_control, args.rejected_phase2p)
        if args.plants:
            result["plants"] = planted_controls(
                args.run_a, args.run_b, args.inherited, args.mesh,
                args.identity_control)
    except GateError as error:
        print(f"FAIL: {error}")
        return 1
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
