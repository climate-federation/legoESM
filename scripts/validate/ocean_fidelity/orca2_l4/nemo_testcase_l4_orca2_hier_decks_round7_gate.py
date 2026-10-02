#!/usr/bin/env python3
"""Preflight and admit the repaired ORCA2 hierarchy rung-5 record."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_hier_decks_round6_gate as rung5,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "nemo_testcase_l4_orca2_hier_decks_round7_acquisition"
PATCH = ARTIFACTS / "l4_r69_surface_round7_absent.patch"
MANIFEST = ARTIFACTS / "rung5_absent_manifest.json"
MODULE = HERE / "nemo_testcase_l4_orca2_round69_month_surface_acquisition" / "l4_r69_surface.F90"
RUNG6_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung6"
)
RUNG6_RECORD = (
    RUNG6_ROOT / "record_havtb1_superseded"
    if (RUNG6_ROOT / "record_havtb1_superseded").exists()
    else RUNG6_ROOT / "record"
)
MAGIC = "NEMO_L4_R69SFC1"
FIELDS = ("utau", "vtau", "taum", "qsr", "qns", "emp", "sfx", "rnf", "fr_i", "rnf_tsc")
RUNOFF_FIELDS = frozenset(("rnf", "rnf_tsc"))
HEADER = struct.Struct("=8i")
FIELD_HEADER = struct.Struct("=4i")
PLANTS = (
    "none",
    "field-name",
    "truncated",
    "frame-nonfinite",
    "absent-as-zero",
    "owner-on",
    "missing-frame",
    "terminal-nonfinite",
    "terminal-step",
    "ice-sentinel-read",
    "tke-sentinel-read",
    "resolved-consequence",
    "runoff-group-unread",
    "active-runoff-print",
    "sha-inventory",
    "calibration-frame",
    "calibration-restart",
)


class GateError(RuntimeError):
    """The repaired recorder or resulting record is inadmissible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight() -> dict[str, object]:
    """Prove the committed patch is additions-only and applies at fuzz zero."""
    rung5.preflight()
    require(
        PATCH.is_file() and MANIFEST.is_file() and MODULE.is_file(),
        "recorder repair artifact is missing",
    )
    manifest = json.loads(MANIFEST.read_text())
    require(manifest["rung"] == 5, "repair manifest rung changed")
    require(
        manifest["build"]["repair_patch_sha256"] == sha256(PATCH),
        "repair manifest patch pin changed",
    )
    require(
        manifest["build"]["source_writer_sha256"] == sha256(MODULE),
        "repair manifest source pin changed",
    )
    require(
        manifest["namelists"]["rung5_deck_sha256"] == rung5.RUNG5_CFG_SHA,
        "repair manifest deck pin changed",
    )
    removed = [
        line
        for line in PATCH.read_text().splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    require(not removed, "recorder repair is not additions-only")
    with tempfile.TemporaryDirectory(prefix="orca2-hier-r7-") as temporary:
        target = Path(temporary) / "l4_r69_surface.F90"
        shutil.copyfile(MODULE, target)
        result = subprocess.run(
            ["patch", "-s", "--fuzz=0", str(target), str(PATCH)],
            text=True,
            capture_output=True,
            check=False,
        )
        require(result.returncode == 0, f"repair patch does not apply: {result.stderr}")
        patched = target.read_text()
    require(patched.count("CALL put_absent") == 2, "repair must mark two runoff fields ABSENT")
    require(patched.count("CALL l4_r7_dump_absent") == 1, "repair dispatch count changed")
    require("IF(.NOT.ln_rnf) THEN" in patched, "repair is not owned by ln_rnf")
    return {
        "status": "PREFLIGHT_PASS_RUNG5_ABSENT_REPAIR",
        "patch_additions_only": True,
        "source_sha256": sha256(MODULE),
        "patch_sha256": sha256(PATCH),
        "fields": list(FIELDS),
        "owners": {
            "surface_core": sorted(set(FIELDS) - RUNOFF_FIELDS),
            "ln_rnf": sorted(RUNOFF_FIELDS),
        },
    }


def _take(raw: bytes, offset: int, size: int, label: str) -> tuple[bytes, int]:
    end = offset + size
    require(end <= len(raw), f"truncated {label}")
    return raw[offset:end], end


def _resolved_runoff(root: Path) -> bool:
    execution = namelist_values(root / "namelist_cfg")
    deck = namelist_values(root / "namelist_cfg.deck")
    execution_value = rung5.rung6.rung10._token(execution["namsbc.ln_rnf"])
    deck_value = rung5.rung6.rung10._token(deck["namsbc.ln_rnf"])
    require(execution_value == deck_value == ".false.", "rung-5 runoff owner is not false")
    output = (root / "ocean.output").read_text(errors="strict")
    require(
        re.search(r"runoff / runoff mouths\s+ln_rnf\s*=\s*F\b", output) is not None,
        "ocean.output does not resolve runoff false",
    )
    return False


def read_surface(
    path: Path, *, kt: int, rank: int, runoff_on: bool, plant: str = "none"
) -> dict[str, object]:
    """Parse names, header-derived payload lengths, and physical EOF."""
    raw = path.read_bytes()
    if plant == "truncated":
        require(raw, "cannot truncate an empty frame")
        raw = raw[:-1]
    chunk, offset = _take(raw, 0, 16, "magic")
    require(chunk.decode("ascii").rstrip() == MAGIC, "bad frame magic")
    chunk, offset = _take(raw, offset, HEADER.size, "frame header")
    version, got_kt, level, got_rank, nfields, nx, ny, bits = HEADER.unpack(chunk)
    require(version == 2, f"rung-5 frame version is {version}, expected 2")
    require((got_kt, got_rank) == (kt, rank), "frame step/rank mismatch")
    require(level in (1, 2, 3) and nfields == len(FIELDS), "invalid level/field count")
    require(nx > 0 and ny > 0 and bits == 64, "invalid frame shape/precision")
    if plant == "owner-on":
        runoff_on = True
    fields: dict[str, dict[str, object]] = {}
    for index, expected in enumerate(FIELDS):
        chunk, offset = _take(raw, offset, 16, f"{expected} name")
        name = chunk.decode("ascii").rstrip()
        if plant == "field-name" and index == 0:
            name = "planted"
        require(name == expected, f"expected {expected!r}, got {name!r}")
        chunk, offset = _take(raw, offset, FIELD_HEADER.size, f"{name} header")
        ndim, n1, n2, n3 = FIELD_HEADER.unpack(chunk)
        owner_on = runoff_on if name in RUNOFF_FIELDS else True
        if not owner_on:
            if plant == "absent-as-zero" and name == "rnf":
                ndim, n1, n2, n3 = 2, nx, ny, 1
            require((ndim, n1, n2, n3) == (0, 0, 0, 0), f"{name}: off owner is not ABSENT")
            fields[name] = {"status": "ABSENT"}
            continue
        wanted = (3, nx, ny, 2) if name == "rnf_tsc" else (2, nx, ny, 1)
        require((ndim, n1, n2, n3) == wanted, f"{name}: shape {ndim, n1, n2, n3} != {wanted}")
        count = math.prod((n1, n2, n3))
        chunk, offset = _take(raw, offset, count * 8, f"{name} payload")
        values = np.frombuffer(chunk, dtype="=f8")
        if plant == "frame-nonfinite" and index == 0:
            values = values.copy()
            values[0] = np.nan
        require(bool(np.isfinite(values).all()), f"{name}: non-finite payload")
        fields[name] = {"status": "PRESENT", "count": count}
    require(offset == len(raw), "trailing bytes after physical EOF")
    absent = [name for name, row in fields.items() if row["status"] == "ABSENT"]
    require(absent == ["rnf", "rnf_tsc"], f"ABSENT census changed: {absent}")
    return {"version": version, "kt": kt, "rank": rank, "fields": fields, "absent": absent}


def validate_frames(root: Path, *, plant: str = "none") -> dict[str, object]:
    runoff_on = _resolved_runoff(root)
    expected = {
        rung5.rung6.rung10.surface._record_name(kt, rank) for kt in range(1, 241) for rank in (0, 1)
    }
    observed = {path.name for path in root.glob("oracle_*.bin")}
    if plant == "missing-frame":
        observed.discard(sorted(expected)[0])
    missing = sorted(expected - observed)[:2]
    extra = sorted(observed - expected)[:2]
    require(observed == expected, f"frame inventory differs: missing={missing} extra={extra}")
    present = absent = 0
    first = True
    parser_plants = {"field-name", "truncated", "frame-nonfinite", "absent-as-zero", "owner-on"}
    for kt in range(1, 241):
        for rank in (0, 1):
            local = plant if first and plant in parser_plants else "none"
            frame = read_surface(
                root / rung5.rung6.rung10.surface._record_name(kt, rank),
                kt=kt,
                rank=rank,
                runoff_on=runoff_on,
                plant=local,
            )
            present += sum(row["status"] == "PRESENT" for row in frame["fields"].values())
            absent += len(frame["absent"])
            first = False
    require((present, absent) == (3840, 960), f"field census changed: {present}/{absent}")
    return {
        "status": "SELF_DESCRIBING_8_PRESENT_2_ABSENT",
        "frames": 480,
        "present_fields": present,
        "absent_fields": absent,
    }


def validate_calibration(root: Path, *, plant: str = "none") -> dict[str, object]:
    """Require the repaired active-runoff arm to reproduce admitted rung 6."""
    expected_frames = sorted(RUNG6_RECORD.glob("oracle_r69_surface_*.bin"))
    require(len(expected_frames) == 480, "admitted rung-6 frame inventory changed")
    for index, reference in enumerate(expected_frames):
        actual = root / reference.name
        require(actual.is_file(), f"calibration frame missing: {reference.name}")
        equal = actual.read_bytes() == reference.read_bytes()
        if plant == "calibration-frame" and index == 0:
            equal = False
        require(equal, f"calibration frame differs: {reference.name}")
    restart_names = sorted(
        path.name for path in RUNG6_RECORD.glob("ORCA2_00000240_restart_????.nc")
    )
    require(len(restart_names) == 2, "admitted rung-6 terminal inventory changed")
    for index, name in enumerate(restart_names):
        equal = (root / name).read_bytes() == (RUNG6_RECORD / name).read_bytes()
        if plant == "calibration-restart" and index == 0:
            equal = False
        require(equal, f"calibration restart differs: {name}")
    resolved = rung5.rung6.validate_resolved(root)
    return {
        "status": "BIT_IDENTICAL_RUNG6",
        "frames": 480,
        "restart_shards": 2,
        "resolved": resolved,
    }


def validate_record(
    root: Path, calibration: Path, *, expect_commit: str, plant: str = "none"
) -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant: {plant}")
    preflight()
    require(
        (root / "producer_commit.txt").read_text().strip() == expect_commit,
        "producer commit differs",
    )
    require(
        (calibration / "producer_commit.txt").read_text().strip() == expect_commit,
        "calibration producer differs",
    )
    require(
        (root / "namelist_cfg").read_bytes() == rung5.execution_cfg_bytes(),
        "rung-5 execution namelist differs",
    )
    require(
        (root / "namelist_cfg.deck").read_bytes() == rung5.rung5_cfg_bytes(),
        "rung-5 exact deck differs",
    )
    require(
        (root / "recorder_repair_manifest.json").read_bytes() == MANIFEST.read_bytes(),
        "repair manifest differs",
    )
    require(
        json.loads((root / "hierarchy_manifest.json").read_text())
        == json.loads(rung5.MANIFEST.read_text()),
        "hierarchy manifest differs",
    )
    calibration_report = validate_calibration(calibration, plant=plant)
    parser_plant = (
        plant
        if plant
        in {
            "field-name",
            "truncated",
            "frame-nonfinite",
            "absent-as-zero",
            "owner-on",
            "missing-frame",
        }
        else "none"
    )
    frames = validate_frames(root, plant=parser_plant)
    inherited_plant = plant if plant in set(rung5.PLANTS) else "none"
    terminal = rung5.rung6.rung7.rung8.rung9.validate_terminal(root, plant=inherited_plant)
    month = rung5.rung6.rung10.validate_month_products(root)
    resolved = rung5.validate_resolved(root, plant=inherited_plant)
    inventory = rung5.rung6.rung10.validate_sha_inventory(root, plant=inherited_plant)
    require(
        (root / "nemo").read_bytes() == (calibration / "nemo").read_bytes(),
        "rung-5 and calibration binaries differ",
    )
    return {
        "format": "nemo-testcase-l4-orca2-hierarchy-rung5-record-v2",
        "status": "PASS_RUNG5_ABSENT_RECORD",
        "claim_label": "independent",
        "producer_commit": expect_commit,
        "calibration": calibration_report,
        "frames": frames,
        "terminal": terminal,
        "month_products": month,
        "resolved": resolved,
        "sha_inventory": inventory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight_only:
            report = preflight()
            require(args.plant == "none", f"{args.plant} plant stayed green")
        else:
            require(
                args.record and args.calibration and args.expect_commit,
                "record, calibration, and expected commit are required",
            )
            report = validate_record(
                args.record, args.calibration, expect_commit=args.expect_commit, plant=args.plant
            )
            require(args.plant == "none", f"{args.plant} plant stayed green")
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"STATUS {report['status']}")
        return 0
    except (RuntimeError, OSError, UnicodeError, ValueError, struct.error) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
