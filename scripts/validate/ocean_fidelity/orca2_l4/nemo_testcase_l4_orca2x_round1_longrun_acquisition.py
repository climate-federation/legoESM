#!/usr/bin/env python3
"""Admission and sizing helpers for the ORCA2X long-run record.

The reader deliberately imports the reviewed round-1 surface schema, round-326
hm_i reader/ownership convention, and round-328 same-program proof.  It does
not provide a second interpretation of any binary layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round326_hmi_gate as hmi,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round328_dynzdf_gate as r328,
)

STEPS_PER_DAY = 8
PILOT_DAYS = 30
PILOT_STEPS = PILOT_DAYS * STEPS_PER_DAY
YEAR_DAYS = 365
YEAR_STEPS = YEAR_DAYS * STEPS_PER_DAY
RANKS = (0, 1)
SURFACE_BYTES_PER_RANK = 4_101_508
HMI_BYTES_PER_RANK = 457_456
ENTRY_BYTES_PER_RANK = 14_288_048
RESTART_BYTES_PER_DAY = 74_182_288
ENTRY_FRAME_COUNT = 60
STAGE_BYTES_PER_FRAME = 14_288_052
STAGE_FRAME_COUNT = 30
BT_BYTES_PER_FRAME = 457_256
BT_FRAME_COUNT = 10
# Measured round-326 bytes from observers capped after kt 10 by this tool:
# exchange + thermodynamics + ZDF input streams + ten two-stage Prather frames.
CAPPED_SI3_BYTES = 39_097_360 + 35_715_040 + 1_194_184 + 10 * 6_286_780
LEGACY_HMI_BYTES = 10 * len(RANKS) * HMI_BYTES_PER_RANK
DISK_HEADROOM_BYTES = 2 * 2**30
PLANTS = (
    "missing-surface", "missing-hmi", "hmi-content", "hm-zero",
    "even-step", "missing-restart", "stamp", "corrupt-byte",
)
SHA256_LINE_RE = re.compile(r"^([0-9a-f]{64})\s+\*?([^\s/]+)\s*$")


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256_file(path: Path, *, flip_last_byte: bool = False) -> str:
    require(path.is_file(), f"missing file for SHA-256: {path.name}")
    if flip_last_byte:
        data = bytearray(path.read_bytes())
        require(bool(data), f"empty file cannot flip byte: {path.name}")
        data[-1] ^= 0x01
        return hashlib.sha256(data).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_manifest(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing SHA-256 manifest: {path.name}")
    entries: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        match = SHA256_LINE_RE.match(raw)
        require(match is not None,
                f"{path.name}:{line_no}: malformed SHA-256 line")
        digest, name = match.group(1), match.group(2)
        require(name not in ("", ".", ".."),
                f"{path.name}:{line_no}: invalid filename {name!r}")
        require(name not in entries,
                f"{path.name}:{line_no}: duplicate filename {name}")
        entries[name] = digest
    require(bool(entries), f"empty SHA-256 manifest: {path.name}")
    return entries


def surface_name(kt: int, rank: int) -> str:
    return (f"oracle_ocean_surface_input_kt{kt:08d}.bin" if rank == 0 else
            f"oracle_ocean_surface_input_rank{rank:04d}_kt{kt:08d}.bin")


def hmi_name(kt: int, rank: int) -> str:
    return f"oracle_orca2x_hmi_kt{kt:08d}_r{rank:04d}.bin"


def estimate_bytes(days: int) -> dict[str, int | float]:
    require(days > 0, "days must be positive")
    steps = days * STEPS_PER_DAY
    replay = steps * len(RANKS) * (SURFACE_BYTES_PER_RANK + HMI_BYTES_PER_RANK)
    restarts = days * RESTART_BYTES_PER_DAY
    inherited = ENTRY_FRAME_COUNT * len(RANKS) * ENTRY_BYTES_PER_RANK
    capped = (STAGE_FRAME_COUNT * STAGE_BYTES_PER_FRAME
              + BT_FRAME_COUNT * BT_BYTES_PER_FRAME
              + CAPPED_SI3_BYTES + LEGACY_HMI_BYTES)
    total = replay + restarts + inherited + capped
    return {
        "days": days,
        "steps": steps,
        "replay_bytes": replay,
        "daily_restart_bytes": restarts,
        "fixed_entry_frame_bytes": inherited,
        "fixed_capped_observer_bytes": capped,
        "total_bytes": total,
        "total_gib": total / 2**30,
        "disk_guard_bytes": total + DISK_HEADROOM_BYTES,
        "disk_guard_gib": (total + DISK_HEADROOM_BYTES) / 2**30,
    }


def same_program(new_bld: Path, reference_bld: Path,
                 observer_patches: dict[str, Path],
                 *, new_cfg: str, reference_cfg: str) -> dict:
    """Round-326 plus capped observers and one write-only writer unit."""
    return r328.same_program_observers(
        new_bld,
        reference_bld,
        observer_patches,
        ("orca2x_longrun_hmi_writer",),
        new_cfg=new_cfg,
        ref_cfg=reference_cfg,
    )


def _restart_files(root: Path, day: int) -> list[Path]:
    step = day * STEPS_PER_DAY
    return sorted(root.glob(f"*_{step:08d}_restart*.nc"))


def read_hmi_pair(
    root: Path, kt: int, *, plant_missing_rank0: bool = False
) -> dict[str, np.ndarray]:
    slabs = []
    for rank in RANKS:
        path = root / hmi_name(kt, rank)
        if plant_missing_rank0 and rank == 0:
            path = root / "planted_missing_hmi.bin"
        require(path.is_file(), f"missing hm_i record: {path.name}")
        record = hmi.read_record(path)
        require(record["header"]["kt"] == kt and record["header"]["rank"] == rank,
                f"{path.name}: wrong kt/rank")
        slabs.append({name: hmi.owned(record["fields"][name])
                      for name in hmi.EXPECTED_ORDER})
    return {name: np.concatenate([slabs[0][name], slabs[1][name]], axis=1)
            for name in hmi.EXPECTED_ORDER}


def _first_wet(mask: np.ndarray) -> tuple[int, ...]:
    indices = np.argwhere(mask)
    require(bool(indices.size), "hm_i record has no wet cells")
    return tuple(int(value) for value in indices[0])


def _validate_hmi_content(pair: dict[str, np.ndarray],
                          fields: dict[str, np.ndarray], kt: int) -> None:
    wet = pair["tmask_1"] > 0.0
    for name in ("fr_i", "taum"):
        require(pair[name].shape == fields[name].shape,
                f"kt {kt}: hm_i {name} shape differs from surface frame")
        require(np.array_equal(pair[name][wet].view(np.uint64),
                               fields[name][wet].view(np.uint64)),
                f"kt {kt}: hm_i {name} differs from surface frame on wet cells")
    ice_wet = wet & (pair["fr_i"] > 0.0)
    require(bool(np.any(pair["hm_i"][ice_wet] > 0.0)),
            f"kt {kt}: hm_i is zero on every ice-covered wet point")
    require(bool(np.all(pair["hm_i"] >= 0.0)
                 and np.all(pair["hm_i"] < hmi.HM_I_MAX_M)),
            f"kt {kt}: hm_i outside admitted range")


def validate_step(root: Path, kt: int, *, plant: str | None = None) -> dict:
    surface_paths = [root / surface_name(kt, rank) for rank in RANKS]
    if plant == "missing-surface":
        surface_paths[0] = root / "planted_missing_surface.bin"
    for rank, path in zip(RANKS, surface_paths, strict=True):
        require(path.is_file(), f"missing surface record: {path.name}")
        ladder.validate_surface_frame(path, kt, rank)
    fields = ladder.assemble_surface_fields(root, kt)
    hmi_paths = [root / hmi_name(kt, rank) for rank in RANKS]
    pair = read_hmi_pair(root, kt, plant_missing_rank0=plant == "missing-hmi")
    if plant == "hmi-content":
        pair["fr_i"] = pair["fr_i"].copy()
        index = _first_wet(pair["tmask_1"] > 0.0)
        pair["fr_i"][index] = np.nextafter(pair["fr_i"][index], np.inf)
    elif plant == "hm-zero":
        pair["hm_i"] = np.zeros_like(pair["hm_i"])
    _validate_hmi_content(pair, fields, kt)
    surface_digests = {path.name: sha256_file(path) for path in surface_paths}
    hmi_digests = {path.name: sha256_file(path) for path in hmi_paths}
    return {
        "kt": kt,
        "surface": [p.name for p in surface_paths],
        "surface_sha256": surface_digests,
        "hmi": [p.name for p in hmi_paths],
        "hmi_sha256": hmi_digests,
        "hm_i_max_m": float(pair["hm_i"].max()),
        "hmi_fields": pair,
    }


def run_gate(root: Path, expect_commit: str, *, days: int = PILOT_DAYS,
             plant: str | None = None,
             fixture_steps: int | None = None) -> dict:
    require(plant in (None, *PLANTS), f"unknown plant {plant}")
    stamp = root / "producer_commit.txt"
    require(stamp.is_file(), "missing producer_commit.txt")
    produced = stamp.read_text().strip()
    if plant == "stamp":
        produced += "-planted"
    require(produced == expect_commit, f"producer commit {produced} != {expect_commit}")
    require(fixture_steps in (None, 2), "preflight fixture must have two steps")
    steps = days * STEPS_PER_DAY if fixture_steps is None else fixture_steps
    rows = []
    file_sha256: dict[str, str] = {}
    previous = None
    for kt in range(1, steps + 1):
        step_plant = (
            plant
            if kt == 1 and plant in (
                "missing-surface", "missing-hmi", "hmi-content", "hm-zero")
            else None
        )
        row = validate_step(root, kt, plant=step_plant)
        if kt % 2 == 0:
            current = row["hmi_fields"]["hm_i"]
            if plant == "even-step" and kt == 2:
                current = current.copy()
                current[_first_wet(row["hmi_fields"]["tmask_1"] > 0.0)] += 1.0e-3
            require(np.array_equal(current.view(np.uint64),
                                   previous.view(np.uint64)),
                    f"kt {kt}: even-step hm_i differs from previous odd step")
        previous = row["hmi_fields"]["hm_i"]
        del row["hmi_fields"]
        file_sha256.update(row["surface_sha256"])
        file_sha256.update(row["hmi_sha256"])
        rows.append(row)
    restart_counts = {}
    restart_days = days if fixture_steps is None else 1
    for day in range(1, restart_days + 1):
        files = _restart_files(root, day)
        if plant == "missing-restart" and day == 1:
            files = []
        require(
            len(files) == 4,
            f"day {day}: expected four ocean/ice restart slabs, got {len(files)}",
        )
        restart_counts[str(day)] = [p.name for p in files]
        for path in files:
            file_sha256[path.name] = sha256_file(path)
    ocean_output = root / "ocean.output"
    require(ocean_output.is_file(), "missing ocean.output")
    file_sha256["ocean.output"] = sha256_file(ocean_output)
    result = {
        "format": "orca2x-longrun-record-v1",
        "status": "PASS_ORCA2X_LONGRUN_ADMISSION",
        "producer_commit": produced,
        "days": days,
        "steps": steps,
        "size_estimate": estimate_bytes(days) if fixture_steps is None else None,
        "step_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "daily_restarts": restart_counts,
        "file_sha256": file_sha256,
    }
    if plant == "corrupt-byte":
        verify_admitted_outputs(
            root,
            expect_commit,
            days=days,
            fixture_steps=fixture_steps,
            plant=plant,
            admitted_override=result,
        )
    return result


def verify_admitted_outputs(
    root: Path,
    expect_commit: str,
    *,
    days: int = PILOT_DAYS,
    fixture_steps: int | None = None,
    plant: str | None = None,
    admitted_override: dict | None = None,
) -> dict:
    require(plant in (None, "corrupt-byte"), f"unsupported verify plant {plant}")
    require(fixture_steps in (None, 2), "preflight fixture must have two steps")
    steps = days * STEPS_PER_DAY if fixture_steps is None else fixture_steps
    restart_days = days if fixture_steps is None else 1
    admission_path = root / "orca2x_longrun_admission.json"
    if admitted_override is not None and not admission_path.is_file():
        admitted = admitted_override
    else:
        require(admission_path.is_file(), "long-run record is not ADMITTED")
        admitted = json.loads(admission_path.read_text())
    require(admitted.get("status") == "PASS_ORCA2X_LONGRUN_ADMISSION",
            "long-run admission status is not PASS")
    require(admitted.get("producer_commit") == expect_commit,
            f"admitted producer commit {admitted.get('producer_commit')} != {expect_commit}")
    require(admitted.get("days") == days and admitted.get("steps") == steps,
            "admitted days/steps mismatch")
    expected_records = {"ocean.output"}
    for kt in range(1, steps + 1):
        for rank in RANKS:
            expected_records.add(surface_name(kt, rank))
            expected_records.add(hmi_name(kt, rank))
    daily_restarts = admitted.get("daily_restarts")
    require(isinstance(daily_restarts, dict), "admission lacks daily_restarts")
    for day in range(1, restart_days + 1):
        files = _restart_files(root, day)
        require(len(files) == 4,
                f"day {day}: expected four ocean/ice restart slabs, got {len(files)}")
        names = [p.name for p in files]
        require(daily_restarts.get(str(day)) == names,
                f"day {day}: restart file list differs from admission")
        expected_records.update(names)
    admitted_hashes = admitted.get("file_sha256")
    require(isinstance(admitted_hashes, dict),
            "admission lacks file_sha256 manifest")
    require(set(admitted_hashes.keys()) == expected_records,
            "admission file_sha256 keys do not match expected record set")
    outputs_sha_path = root / "orca2x_longrun_outputs.sha256"
    if admitted_override is not None and not outputs_sha_path.is_file():
        manifest = dict(admitted_hashes)
    else:
        manifest = parse_sha256_manifest(outputs_sha_path)
        require(
            set(manifest.keys())
            == expected_records | {"orca2x_longrun_admission.json"},
            "orca2x_longrun_outputs.sha256 keys do not match admitted record set",
        )
        for name in expected_records:
            require(
                manifest[name] == admitted_hashes[name],
                f"orca2x_longrun_outputs.sha256 disagrees with admission for {name}",
            )
    target_corrupt = surface_name(1, 0)
    for name, expected_digest in sorted(manifest.items()):
        path = root / name
        require(path.is_file(), f"missing admitted record file: {name}")
        observed = sha256_file(
            path,
            flip_last_byte=(plant == "corrupt-byte" and name == target_corrupt),
        )
        require(observed == expected_digest,
                f"admitted SHA-256 mismatch for {name}")
    input_sha_path = root / "input_files.sha256"
    if fixture_steps is None or input_sha_path.is_file():
        require(input_sha_path.is_file(), "missing input_files.sha256")
        for name, expected_digest in sorted(
            parse_sha256_manifest(input_sha_path).items()
        ):
            path = root / name
            require(path.is_file(), f"missing admitted deck input: {name}")
            require(sha256_file(path) == expected_digest,
                    f"admitted deck input SHA-256 mismatch for {name}")
    return admitted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--days", type=int, default=PILOT_DAYS)
    parser.add_argument("--fixture-steps", type=int, choices=(2,))
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--verify-outputs", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--estimate-days", type=int)
    parser.add_argument("--required-kib", action="store_true")
    parser.add_argument("--same-program", type=Path)
    parser.add_argument("--reference-bld", type=Path)
    parser.add_argument("--observer-patch", action="append", default=[])
    parser.add_argument("--new-cfg")
    parser.add_argument("--reference-cfg")
    args = parser.parse_args()
    try:
        if args.estimate_days is not None:
            result = estimate_bytes(args.estimate_days)
        elif args.same_program is not None:
            require(None not in (args.reference_bld, args.new_cfg,
                                 args.reference_cfg) and args.observer_patch,
                    "same-program arguments incomplete")
            patches = {}
            for item in args.observer_patch:
                unit, separator, path = item.partition("=")
                require(bool(separator and unit and path),
                        f"bad --observer-patch {item!r}; need unit=path")
                require(unit not in patches, f"duplicate observer patch {unit}")
                patches[unit] = Path(path)
            result = same_program(args.same_program, args.reference_bld,
                                  patches, new_cfg=args.new_cfg,
                                  reference_cfg=args.reference_cfg)
        elif args.verify_outputs:
            require(args.root is not None and args.expect_commit,
                    "--root and --expect-commit are required")
            result = verify_admitted_outputs(
                args.root,
                args.expect_commit,
                days=args.days,
                fixture_steps=args.fixture_steps,
                plant=args.plant,
            )
        else:
            require(args.root is not None and args.expect_commit,
                    "--root and --expect-commit are required")
            result = run_gate(args.root, args.expect_commit, days=args.days,
                              plant=args.plant, fixture_steps=args.fixture_steps)
        if args.required_kib:
            require(args.estimate_days is not None,
                    "--required-kib needs --estimate-days")
            print((int(result["disk_guard_bytes"]) + 1023) // 1024)
            return 0
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(text)
        print(text, end="")
        return 0
    except (GateError, hmi.GateError, ladder.GateError, r328.GateError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
