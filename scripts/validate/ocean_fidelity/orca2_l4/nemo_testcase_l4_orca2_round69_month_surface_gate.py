#!/usr/bin/env python3
"""Preflight and admit the round-69 ORCA2 month surface record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l4_orca2_phase1_gate as phase1,
)


HERE = Path(__file__).resolve().parent
INSTRUMENT = HERE / "nemo_testcase_l4_orca2_round69_month_surface_acquisition"
MODULE = INSTRUMENT / "l4_r69_surface.F90"
PATCH = INSTRUMENT / "stprk3_round69.patch"
BASE_STPRK3 = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/stprk3.F90"
)
EXPECTED_BASE_SHA256 = (
    "d12b246db6b77b122ef1c53a485a3d742ce4acf113a58684a639c9031f0a9e1a"
)
EXPECTED_DECK_SHA256 = (
    "09a350860ff6eaef17d1f0e18aa8e16c4d929e994d9d6804d0976f531b06f66e"
)
EXPECTED_INPUT_SHA256 = (
    "3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5"
)
MAGIC = "NEMO_L4_R69SFC1"
FIELDS = (
    "utau", "vtau", "taum", "qsr", "qns", "emp", "sfx", "rnf",
    "fr_i", "rnf_tsc",
)
HEADER = struct.Struct("=8i")
FIELD_HEADER = struct.Struct("=4i")
PLANTS = (
    "none", "field-name", "truncated", "calibration-ulp", "missing-frame",
    "extra-stream", "restart-ulp",
)


class GateError(RuntimeError):
    """The surface acquisition violated a frozen round-69 predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _take(raw: bytes, offset: int, size: int, label: str) -> tuple[bytes, int]:
    end = offset + size
    require(end <= len(raw), f"truncated {label}")
    return raw[offset:end], end


def read_surface(path: Path, *, kt: int, rank: int,
                 plant: str = "none") -> dict[str, object]:
    """Parse one self-describing frame through physical EOF."""
    raw = path.read_bytes()
    if plant == "truncated":
        require(len(raw) > 8, f"{path}: cannot truncate empty record")
        raw = raw[:-8]
    chunk, offset = _take(raw, 0, 16, "magic")
    magic = chunk.decode("ascii").rstrip()
    require(magic == MAGIC, f"{path}: bad magic {magic!r}")
    chunk, offset = _take(raw, offset, HEADER.size, "header")
    version, got_kt, level, got_rank, nfields, nx, ny, bits = HEADER.unpack(chunk)
    require(version == 1, f"{path}: bad version {version}")
    require(got_kt == kt, f"{path}: step {got_kt} != {kt}")
    require(level in (1, 2, 3), f"{path}: bad time level {level}")
    require(got_rank == rank, f"{path}: rank {got_rank} != {rank}")
    require(nfields == len(FIELDS), f"{path}: field count {nfields}")
    require(nx > 0 and ny > 0 and bits == 64,
            f"{path}: invalid shape/precision {(nx, ny, bits)}")
    fields: dict[str, np.ndarray] = {}
    payload_offsets: dict[str, int] = {}
    for index, expected_name in enumerate(FIELDS):
        chunk, offset = _take(raw, offset, 16, f"{expected_name} name")
        name = chunk.decode("ascii").rstrip()
        if plant == "field-name" and index == 0:
            name = "planted_name"
        require(name == expected_name,
                f"{path}: expected {expected_name!r}, got {name!r}")
        require(name not in fields, f"{path}: duplicate field {name}")
        chunk, offset = _take(raw, offset, FIELD_HEADER.size, f"{name} shape")
        ndim, n1, n2, n3 = FIELD_HEADER.unpack(chunk)
        wanted = (3, nx, ny, 2) if name == "rnf_tsc" else (2, nx, ny, 1)
        require((ndim, n1, n2, n3) == wanted,
                f"{path}: {name} shape header {(ndim, n1, n2, n3)} != {wanted}")
        count = math.prod((n1, n2, n3))
        payload_offsets[name] = offset
        chunk, offset = _take(raw, offset, count * 8, f"{name} payload")
        values = np.frombuffer(chunk, dtype="=f8").reshape(
            (n1, n2, n3), order="F"
        )
        require(bool(np.isfinite(values).all()), f"{path}: {name} non-finite")
        transposed = values.transpose(1, 0, 2)
        fields[name] = transposed if ndim == 3 else transposed[..., 0]
    require(offset == len(raw), f"{path}: trailing bytes after physical EOF")
    require(tuple(fields) == FIELDS, f"{path}: missing/reordered fields")
    return {
        "path": str(path),
        "kt": kt,
        "rank": rank,
        "level": level,
        "shape": [ny, nx],
        "fields": fields,
        "payload_offsets": payload_offsets,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _record_name(kt: int, rank: int) -> str:
    return f"oracle_r69_surface_rank{rank:04d}_kt{kt:08d}.bin"


def validate_inventory(root: Path, *, steps=range(1, 241),
                       plant: str = "none") -> list[dict[str, object]]:
    expected = {_record_name(kt, rank) for kt in steps for rank in (0, 1)}
    observed = {path.name for path in root.glob("oracle_*.bin")}
    if plant == "missing-frame":
        observed.discard(sorted(expected)[0])
    elif plant == "extra-stream":
        observed.add("oracle_unregistered.bin")
    require(observed == expected,
            f"oracle inventory mismatch: missing={sorted(expected-observed)[:3]} "
            f"extra={sorted(observed-expected)[:3]}")
    rows = []
    first = True
    for kt in steps:
        for rank in (0, 1):
            local_plant = plant if first and plant in ("field-name", "truncated") else "none"
            record = read_surface(root / _record_name(kt, rank), kt=kt,
                                  rank=rank, plant=local_plant)
            rows.append({key: record[key] for key in
                         ("path", "kt", "rank", "level", "shape", "sha256")})
            first = False
    return rows


def _raw_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


def _old_surface(root: Path, kt: int, rank: int) -> dict[str, np.ndarray]:
    name = (
        f"oracle_ocean_surface_input_kt{kt:08d}.bin" if rank == 0 else
        f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin"
    )
    return ladder.read_surface_fields(root / name, kt, rank)


def validate_calibration(root: Path, old_root: Path, *, steps=range(1, 11),
                         plant: str = "none") -> dict[str, object]:
    comparisons = 0
    planted = False
    for kt in steps:
        for rank in (0, 1):
            new = read_surface(root / _record_name(kt, rank), kt=kt, rank=rank)
            old = _old_surface(old_root, kt, rank)
            for name in FIELDS:
                values = np.asarray(new["fields"][name])
                if plant == "calibration-ulp" and not planted:
                    values = values.copy()
                    values.flat[0] = np.nextafter(values.flat[0], np.inf)
                    planted = True
                require(_raw_equal(values, old[name]),
                        f"calibration differs: kt={kt} rank={rank} field={name}")
                comparisons += 1
    return {"status": "BIT_EXACT", "field_comparisons": comparisons}


def _mutated_copy(path: Path, target: Path) -> None:
    shutil.copy2(path, target)
    with Dataset(target, "r+") as dataset:
        for variable in dataset.variables.values():
            if variable.dtype == np.dtype("float64") and variable.size:
                variable.set_auto_maskandscale(False)
                values = np.asarray(variable[:]).copy()
                values.reshape(-1)[0] = np.nextafter(values.reshape(-1)[0], np.inf)
                variable[:] = values
                return
    raise GateError(f"no mutable float64 payload in {path}")


def validate_terminal_restarts(root: Path, month_root: Path, *,
                               plant: str = "none") -> dict[str, object]:
    names = [
        f"ORCA2_00000240_restart{suffix}_{rank:04d}.nc"
        for suffix in ("", "_ice") for rank in (0, 1)
    ]
    for index, name in enumerate(names):
        expected = month_root / name
        observed = root / name
        require(expected.is_file() and observed.is_file(), f"missing restart pair {name}")
        if plant == "restart-ulp" and index == 0:
            with tempfile.TemporaryDirectory(prefix="orca2-r69-restart-") as temporary:
                changed = Path(temporary) / name
                _mutated_copy(observed, changed)
                phase1._netcdf_equal_except_timestamp(expected, changed)
        else:
            phase1._netcdf_equal_except_timestamp(expected, observed)
    return {"status": "BIT_EXACT", "restart_shards": names}


def _manifest(root: Path, name: str, expected_sha: str) -> None:
    path = root / name
    require(path.is_file() and sha256(path) == expected_sha,
            f"manifest pin mismatch: {path}")
    for line in path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        target = root / filename.lstrip("* ")
        require(target.is_file() and sha256(target) == digest,
                f"manifest target mismatch: {target}")


def _sha_rows(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing SHA-256 ledger: {path}")
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        require(len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]) is not None,
                f"malformed SHA-256 ledger row: {line!r}")
        name = Path(parts[1].lstrip("* ")).name
        require(name not in rows, f"duplicate SHA-256 ledger name: {name}")
        rows[name] = parts[0]
    return rows


def validate_producer(root: Path) -> dict[str, object]:
    binary = _sha_rows(root / "binary.sha256")
    compiled = _sha_rows(root / "compiled_source.sha256")
    sources = _sha_rows(root / "acquisition_sources.sha256")
    expected = {
        "nemo": root / "nemo",
        "compiled_stprk3.f90": root / "compiled_stprk3.f90",
        "compiled_l4_r69_surface.f90": root / "compiled_l4_r69_surface.f90",
    }
    require(set(binary) == {"nemo"}, "binary ledger inventory changed")
    require(set(compiled) == set(expected) - {"nemo"},
            "compiled-source ledger inventory changed")
    for name, target in expected.items():
        ledger = binary if name == "nemo" else compiled
        require(target.is_file() and sha256(target) == ledger[name],
                f"producer artifact digest mismatch: {target}")
    require(sources == {
        MODULE.name: sha256(MODULE),
        PATCH.name: sha256(PATCH),
    }, "committed acquisition source digest mismatch")
    compiled_module = (root / "compiled_l4_r69_surface.f90").read_text()
    compiled_stprk3 = (root / "compiled_stprk3.f90").read_text()
    require("STATUS='NEW'" in compiled_module and
            "CALL l4_r69_dump( kstp, Nbb )" in compiled_stprk3,
            "compiled writer/call-site markers are absent")
    return {
        "binary_sha256": binary["nemo"],
        "compiled_source_sha256": compiled,
        "acquisition_source_sha256": sources,
    }


def preflight() -> dict[str, object]:
    require(BASE_STPRK3.is_file() and sha256(BASE_STPRK3) == EXPECTED_BASE_SHA256,
            "pinned base stprk3 changed")
    require(MODULE.is_file() and PATCH.is_file(), "committed writer artifact missing")
    patch_lines = PATCH.read_text().splitlines()
    removed = [line for line in patch_lines if line.startswith("-") and not line.startswith("---")]
    require(not removed, f"round-69 patch is not additions-only: {removed[:1]}")
    source = MODULE.read_text()
    operands = "utau|vtau|taum|qsr|qns|emp|sfx|rnf|fr_i|rnf_tsc"
    require(re.search(rf"(?mi)^\s*(?:{operands})\s*=", source) is None,
            "writer assigns a recorded NEMO operand")
    with tempfile.TemporaryDirectory(prefix="orca2-r69-patch-") as temporary:
        target = Path(temporary) / "stprk3.F90"
        shutil.copy2(BASE_STPRK3, target)
        completed = subprocess.run(
            ["patch", "-p0", "-i", str(PATCH)], cwd=temporary,
            text=True, capture_output=True, check=False,
        )
        require(completed.returncode == 0,
                f"round-69 patch no longer applies: {completed.stderr.strip()}")
        patched = target.read_text()
        require("USE l4_r69_surface, ONLY : l4_r69_dump" in patched and
                "CALL l4_r69_dump( kstp, Nbb )" in patched,
                "patched call site is incomplete")
    return {
        "status": "PREFLIGHT_PASS",
        "base_stprk3_sha256": sha256(BASE_STPRK3),
        "module_sha256": sha256(MODULE),
        "patch_sha256": sha256(PATCH),
        "patch_removed_lines": 0,
    }


def validate_record(root: Path, old_root: Path, month_root: Path, *,
                    expect_commit: str, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    require((root / "producer_commit.txt").read_text().strip() == expect_commit,
            "producer commit mismatch")
    _manifest(root, "deck_files.sha256", EXPECTED_DECK_SHA256)
    _manifest(root, "input_files.sha256", EXPECTED_INPUT_SHA256)
    producer = validate_producer(root)
    stdout = (root / "run.user.stdout.log").read_text()
    timing = (root / "run.user.time.log").read_text()
    require("STOP 0" in stdout and "RUN_DONE" in timing,
            "NEMO run did not complete")
    require((root / "time.step").read_text().strip() == "240",
            "NEMO run did not reach step 240")
    inventory = validate_inventory(root, plant=plant)
    calibration = validate_calibration(root, old_root, plant=plant)
    terminal = validate_terminal_restarts(root, month_root, plant=plant)
    return {
        "format": "nemo-testcase-l4-orca2-round69-surface-v1",
        "status": "PASS_MONTH_SURFACE_RECORD",
        "claim_label": "independent",
        "producer_commit": expect_commit,
        "producer": producer,
        "surface_frames": len(inventory),
        "surface_fields": list(FIELDS),
        "surface_inventory": inventory,
        "calibration": calibration,
        "terminal_restarts": terminal,
        "next": "RUN_INDEPENDENT_ORCA2_MONTH_RANKING",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--month-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            report = preflight()
        else:
            require(all((args.record, args.calibration_root, args.month_root,
                         args.expect_commit)), "full admission arguments are required")
            report = validate_record(
                args.record, args.calibration_root, args.month_root,
                expect_commit=args.expect_commit, plant=args.plant,
            )
        report["worktree"] = worktree_stamp()
    except (GateError, phase1.GateError, KeyError, OSError, TypeError,
            ValueError, struct.error) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
