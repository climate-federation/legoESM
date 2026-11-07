#!/usr/bin/env python3
"""Admit the self-describing kt=1/2 ORCA2 ZDF acquisition twins."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_een_discriminator_gate as een,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2p_zdf_acquisition_gate as p2p,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)

NX, NY, NZ = p2p.NX, p2p.NY, p2p.NZ
HALO = p2p.HALO
MAGIC = "NEMO_L4_ZSH2_2"
HEADER_INTS = 13
FIELD_COUNT = len(p2p.FIELDS_3D) + len(p2p.FIELDS_2D)
EXTENT_INTS = 3 * FIELD_COUNT
PAYLOAD_OFFSET = 16 + 4 * (HEADER_INTS + EXTENT_INTS)
ZDF_NAMES = (
    "oracle_zdf_sh2_operands_kt00000001.bin",
    "oracle_zdf_sh2_operands_kt00000002.bin",
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


def _expected_extents() -> tuple[tuple[int, int, int], ...]:
    full3 = (NX, NY, NZ)
    reduced3 = (NX - 2 * HALO, NY - 2 * HALO, NZ)
    full2 = (NX, NY, 1)
    reduced2 = (NX - 2 * HALO, NY - 2 * HALO, 1)
    return tuple(
        reduced3 if allocation == "reduced" else full3
        for allocation in p2p.ALLOCATION_3D
    ) + tuple(
        reduced2 if allocation == "reduced" else full2
        for allocation in p2p.ALLOCATION_2D
    )


def read_zdf_v2(path: Path, mesh: Path, *, require_shear: bool) -> dict[str, object]:
    expected_kt = int(path.stem.rsplit("kt", 1)[1])
    with path.open("rb") as handle:
        raw_magic = handle.read(16)
        require(len(raw_magic) == 16, f"{path.name}: truncated magic")
        magic = raw_magic.decode("ascii").rstrip()
        raw_header = handle.read(4 * HEADER_INTS)
        require(len(raw_header) == 4 * HEADER_INTS,
                f"{path.name}: truncated base header")
        header = struct.unpack(f"={HEADER_INTS}i", raw_header)
        raw_extents = handle.read(4 * EXTENT_INTS)
        require(len(raw_extents) == 4 * EXTENT_INTS,
                f"{path.name}: truncated allocation table")
        extents = np.frombuffer(raw_extents, dtype=np.int32).reshape(
            (FIELD_COUNT, 3))

        version, kt, kbb, kmm, krhs, jpi, jpj, jpk, real_bits, n3, n2, payload, armed = header
        require(magic == MAGIC, f"{path.name}: magic {magic!r}")
        require(version == 2, f"{path.name}: version {version}")
        require(kt == expected_kt, f"{path.name}: kt {kt} != {expected_kt}")
        require(kbb == kmm and kbb in (1, 2, 3) and krhs in (1, 2, 3),
                f"{path.name}: invalid RK3 levels Kbb/Kmm/Krhs={kbb}/{kmm}/{krhs}")
        require((jpi, jpj, jpk, real_bits, n3, n2, armed) ==
                (NX, NY, NZ, 64, len(p2p.FIELDS_3D), len(p2p.FIELDS_2D), 1),
                f"{path.name}: base header {header}")
        expected_extents = _expected_extents()
        require(tuple(map(tuple, extents.tolist())) == expected_extents,
                f"{path.name}: allocation table does not match writer expressions")
        derived_payload = sum(int(np.prod(shape)) for shape in expected_extents)
        require(payload == derived_payload,
                f"{path.name}: payload {payload} != extent-derived {derived_payload}")

        masks = p2p._rank0_masks(mesh)
        arrays: dict[str, np.ndarray] = {}
        rows = []
        field_specs = list(zip(
            p2p.FIELDS_3D, p2p.GRID_3D, p2p.ALLOCATION_3D, strict=True
        )) + [
            (name, "T", allocation)
            for name, allocation in zip(
                p2p.FIELDS_2D, p2p.ALLOCATION_2D, strict=True)
        ]
        for index, ((name, grid, allocation), shape) in enumerate(
                zip(field_specs, expected_extents, strict=True)):
            count = int(np.prod(shape))
            raw = handle.read(8 * count)
            require(len(raw) == 8 * count,
                    f"{path.name}: field {index} {name} truncated")
            field = np.frombuffer(raw, dtype=np.float64).reshape(shape, order="F")
            require(np.isfinite(field).all(), f"{path.name}: {name} non-finite")
            live = masks[(grid, allocation)]
            if shape[2] == 1:
                live = live[..., :1]
            require(np.all(field[~live] == 0.0),
                    f"{path.name}: {name} nonzero canonical slot")
            arrays[name] = field
            rows.append({"index": index, "name": name, "extent": list(shape),
                         "count": count, "grid": grid,
                         "allocation": allocation})
        require(handle.read(1) == b"", f"{path.name}: trailing bytes after payload")

    gradient_counts = {}
    for name, grid, allocation in (
        ("u_Kbb", "U", "full"), ("u_Kmm", "U", "full"),
        ("v_Kbb", "V", "full"), ("v_Kmm", "V", "full"),
    ):
        field = arrays[name]
        live = masks[(grid, allocation)]
        valid = live[..., 1:] & live[..., :-1]
        gradient_counts[name] = int(np.count_nonzero(
            (field[..., :-1] - field[..., 1:])[valid]))
    if require_shear:
        require(gradient_counts["u_Kbb"] + gradient_counts["v_Kbb"] > 0,
                f"{path.name}: Kbb velocity-gradient operand is vacuous")
        require(gradient_counts["u_Kmm"] + gradient_counts["v_Kmm"] > 0,
                f"{path.name}: Kmm velocity-gradient operand is vacuous")

    require(path.stat().st_size == PAYLOAD_OFFSET + 8 * payload,
            f"{path.name}: exact EOF size mismatch")
    return {
        "path": str(path), "sha256": sha256(path),
        "bytes": path.stat().st_size, "magic": magic,
        "header": list(header), "extent_table": extents.tolist(),
        "extent_table_sha256": hashlib.sha256(raw_extents).hexdigest(),
        "extent_derived_payload_f64": payload, "fields": rows,
        "velocity_gradient_nonzero": gradient_counts,
        "non_vacuous_required": require_shear, "exact_eof": True,
    }


def _inventory(root: Path) -> set[str]:
    return {path.name for path in root.glob("oracle_*.bin")}


def validate(run_a: Path, run_b: Path, inherited: Path, een_witness: Path,
             mesh: Path, identity_control: Path) -> dict[str, object]:
    names_a, names_b, old = _inventory(run_a), _inventory(run_b), _inventory(inherited)
    expected = old | set(een.NEW_STREAMS) | {ZDF_NAMES[1]}
    require(names_a == names_b == expected,
            "candidate inventory is not Phase-2q plus kt=2 ZDF plus four EEN streams")
    twin_rows = []
    for name in sorted(names_a):
        require((run_a / name).read_bytes() == (run_b / name).read_bytes(),
                f"twin record differs: {name}")
        twin_rows.append({"file": name, "sha256": sha256(run_a / name),
                          "status": "EXACT_BYTES"})

    inherited_rows = []
    for name in sorted(old - {ZDF_NAMES[0]}):
        require((run_a / name).read_bytes() == (inherited / name).read_bytes(),
                f"pre-existing record differs: {name}")
        inherited_rows.append({"file": name, "sha256": sha256(run_a / name),
                               "status": "EXACT_BYTES"})
    require((run_a / ZDF_NAMES[0]).read_bytes() !=
            (inherited / ZDF_NAMES[0]).read_bytes(),
            "versioned kt=1 ZDF record did not change")

    een_rows = een.validate_new(run_a)
    witness = een.validate_new(een_witness)
    for name in een.NEW_STREAMS:
        require((run_a / name).read_bytes() == (een_witness / name).read_bytes(),
                f"restored EEN stream differs from Phase-2m witness: {name}")

    return {
        "status": "PASS",
        "schemas": {
            ZDF_NAMES[0]: read_zdf_v2(run_a / ZDF_NAMES[0], mesh,
                                      require_shear=False),
            ZDF_NAMES[1]: read_zdf_v2(run_a / ZDF_NAMES[1], mesh,
                                      require_shear=True),
            p2p.ZDF_ENTRY: p2p.read_zdf_entry(run_a / p2p.ZDF_ENTRY),
            p2p.SI3_ZDF: p2p.read_si3_zdf(run_a / p2p.SI3_ZDF),
        },
        "twin_raw_exact": len(twin_rows), "twin_total": len(twin_rows),
        "records": twin_rows,
        "inherited_raw_exact_excluding_versioned_zdf": len(inherited_rows),
        "inherited_total_excluding_versioned_zdf": len(old) - 1,
        "versioned_zdf_kt1": {
            "old_sha256": sha256(inherited / ZDF_NAMES[0]),
            "new_sha256": sha256(run_a / ZDF_NAMES[0]),
            "status": "SUPERSEDED_BY_SELF_DESCRIBING_V2",
        },
        "een": een_rows, "een_phase2m_witness": witness,
        "ordinary_output_identity": phase1.validate_identity(identity_control, run_a),
    }


def _overlay(root: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="orca2-l4-p2s-plant-"))
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


def _truncate(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 8)


def _append(path: Path) -> None:
    with path.open("ab") as handle:
        handle.write(b"X")


def planted_controls(run_a: Path, run_b: Path, inherited: Path,
                     een_witness: Path, mesh: Path,
                     identity_control: Path) -> dict[str, str]:
    results: dict[str, str] = {}

    def expect(label: str, name: str, mutate) -> None:
        overlay = _overlay(run_a)
        try:
            changed = _materialize(overlay, run_a, name)
            mutate(changed)
            try:
                validate(overlay, run_b, inherited, een_witness, mesh,
                         identity_control)
            except (GateError, p2p.GateError, een.GateError, phase1.GateError,
                    OSError, UnicodeError, ValueError, struct.error):
                results[label] = "PASS_NONZERO"
            else:
                raise GateError(f"plant did not fail real admission: {label}")
        finally:
            shutil.rmtree(overlay)

    kt2 = ZDF_NAMES[1]
    expect("zdf_magic", kt2,
           lambda path: _patch(path, 0, b"X"))
    expect("zdf_extent", kt2,
           lambda path: _patch(path, 16 + 4 * HEADER_INTS,
                               struct.pack("=i", 1)))
    expect("zdf_payload_count", kt2,
           lambda path: _patch(path, 16 + 11 * 4,
                               struct.pack("=i", 1)))
    expect("zdf_truncated", kt2, _truncate)
    expect("zdf_trailing", kt2, _append)

    masks = p2p._rank0_masks(mesh)
    first_dead = int(np.flatnonzero(
        ~masks[("W", "reduced")].ravel(order="F"))[0])
    expect("zdf_canonical_zero", kt2,
           lambda path: _patch(path, PAYLOAD_OFFSET + 8 * first_dead,
                               struct.pack("=d", 1.0)))

    # Zero every velocity payload.  The same real parser must reject kt=2 as
    # vacuous, independently of twin identity.
    def zero_velocities(path: Path) -> None:
        extents = _expected_extents()
        starts = []
        cursor = PAYLOAD_OFFSET
        for shape in extents:
            starts.append(cursor)
            cursor += 8 * int(np.prod(shape))
        with path.open("r+b") as handle:
            for index in (6, 7, 8, 9):
                count = int(np.prod(extents[index]))
                handle.seek(starts[index])
                handle.write(bytes(8 * count))

    expect("zdf_nonvacuous", kt2, zero_velocities)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--inherited", type=Path, required=True)
    parser.add_argument("--een-witness", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--identity-control", type=Path, required=True)
    parser.add_argument("--plants", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.run_a, args.run_b, args.inherited,
                          args.een_witness, args.mesh, args.identity_control)
        if args.plants:
            result["plants"] = planted_controls(
                args.run_a, args.run_b, args.inherited, args.een_witness,
                args.mesh, args.identity_control)
    except (GateError, p2p.GateError, een.GateError, phase1.GateError,
            OSError, UnicodeError, ValueError, struct.error) as error:
        print(f"FAIL: {error}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
