#!/usr/bin/env python3
"""Admit the passive Round-146 developed RHS-family record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np


RECORD = "oracle_developed_rhs_families_kt00001081.bin"
MAGIC = "NEMO_L2_R146FAM"
FAMILIES = ("hpg", "ldf", "vor", "keg", "zad")
FIELDS = tuple(f"after_{family}_{face}" for family in FAMILIES for face in ("u", "v"))
NX, NY, NZ = 36, 26, 31
COUNT = NX * NY * NZ
EXPECTED_SIZE = 16 + 8 * 4 + len(FIELDS) * 4 + len(FIELDS) * COUNT * 8
ROUND140_RECORD = "oracle_developed_rhs_kt00001081.bin"
ROUND140_EXPECTED_SHA = "3523ad8eba0b53f4462e85237d85a066195437e6ad1c867e4cb0179d1764a8f5"
ROUND140_EXPECTED_SIZE = 1_486_548
ROUND140_FIELDS = (
    "e3u", "rhs_u", "umask", "e3v", "rhs_v", "vmask",
    "depth_mean_u", "depth_mean_v", "r1_hu0", "r1_hv0",
    "post_drag_u", "post_drag_v", "cd_u", "cd_v", "r1_rho0",
    "wind_tau_u", "wind_tau_v", "wind_r1_hu", "wind_r1_hv",
    "post_wind_u", "post_wind_v",
)
EXACT_INHERITED = (
    "GYRE_OMIP_L2_P3_00001080_restart.nc",
    "GYRE_OMIP_L2_P3_00001081_restart.nc",
    "oracle_process_budget_kt00001081.bin",
    "oracle_bt_step_operands_kt00001081.bin",
    "oracle_stage1_qco_operands_kt00001081.bin",
    "oracle_slow_forcing_split_kt00001081.bin",
)
INTERIOR_OWNED_FIELDS = ("cd_u", "cd_v", "wind_tau_u", "wind_tau_v")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _take(payload: bytes, offset: int, count: int, label: str):
    end = offset + count
    require(end <= len(payload), f"truncated {label}")
    return payload[offset:end], end


def read_record_bytes(payload: bytes) -> dict:
    require(len(payload) == EXPECTED_SIZE,
            f"record is {len(payload)} bytes, expected {EXPECTED_SIZE}")
    raw, offset = _take(payload, 0, 16, "magic")
    try:
        magic = raw.decode("ascii").rstrip()
    except UnicodeDecodeError as error:
        raise GateError("magic is not ASCII") from error
    raw, offset = _take(payload, offset, 8 * 4, "header")
    version, kt, kbb, krhs, nx, ny, nz, bits = struct.unpack("=8i", raw)
    raw, offset = _take(payload, offset, len(FIELDS) * 4, "sizes")
    sizes = struct.unpack(f"={len(FIELDS)}i", raw)
    require((magic, version, kt, kbb, krhs, nx, ny, nz, bits)
            == (MAGIC, 1, 1081, 1, 3, NX, NY, NZ, 64),
            "bad header " + repr((magic, version, kt, kbb, krhs, nx, ny, nz, bits)))
    require(sizes == (COUNT,) * len(FIELDS), "bad field sizes")
    fields = {}
    for name in FIELDS:
        raw, offset = _take(payload, offset, COUNT * 8, name)
        values = np.frombuffer(raw, dtype=np.float64).copy()
        require(values.size == COUNT and np.all(np.isfinite(values)),
                f"bad values in {name}")
        fields[name] = values.reshape((NX, NY, NZ), order="F").transpose(1, 0, 2)
    require(offset == len(payload), "trailing payload")
    require(tuple(fields) == FIELDS, "field registry changed")
    return {"header": {"magic": magic, "version": version, "kt": kt,
                       "Kbb": kbb, "Krhs": krhs, "nx": nx, "ny": ny,
                       "nz": nz, "bits": bits}, "fields": fields}


def read_round140_bytes(payload: bytes) -> dict:
    """Decode every Round-140 field without discarding halos or the jpk slot."""
    require(len(payload) == ROUND140_EXPECTED_SIZE,
            f"Round-140 record is {len(payload)} bytes, expected "
            f"{ROUND140_EXPECTED_SIZE}")
    raw, offset = _take(payload, 0, 16, "Round-140 magic")
    try:
        magic = raw.decode("ascii").rstrip()
    except UnicodeDecodeError as error:
        raise GateError("Round-140 magic is not ASCII") from error
    raw, offset = _take(payload, offset, 8 * 4, "Round-140 header")
    version, kt, kbb, krhs, nx, ny, nz, bits = struct.unpack("=8i", raw)
    raw, offset = _take(payload, offset, 7 * 4, "Round-140 sizes")
    sizes = struct.unpack("=7i", raw)
    interior_count = (NX - 4) * (NY - 4)
    expected_sizes = (COUNT,) * 6 + (interior_count,)
    require((magic, version, kt, kbb, krhs, nx, ny, nz, bits, sizes)
            == ("NEMO_L2_R140RHS", 3, 1081, 1, 3, NX, NY, NZ, 64,
                expected_sizes),
            "bad Round-140 header "
            + repr((magic, version, kt, kbb, krhs, nx, ny, nz, bits, sizes)))

    def values(count: int, label: str) -> np.ndarray:
        nonlocal offset
        raw_values, offset = _take(payload, offset, count * 8, label)
        result = np.frombuffer(raw_values, dtype=np.float64).copy()
        require(result.size == count and np.all(np.isfinite(result)),
                f"bad Round-140 values in {label}")
        return result

    def field3(label: str) -> np.ndarray:
        return values(COUNT, label).reshape(
            (NX, NY, NZ), order="F").transpose(1, 0, 2)

    def full2(label: str) -> np.ndarray:
        return values(NX * NY, label).reshape((NX, NY), order="F").T

    def interior2(label: str) -> np.ndarray:
        return values(interior_count, label).reshape(
            (NX - 4, NY - 4), order="F").T

    fields = {
        "e3u": field3("e3u"),
        "rhs_u": field3("rhs_u"),
        "umask": field3("umask"),
        "e3v": field3("e3v"),
        "rhs_v": field3("rhs_v"),
        "vmask": field3("vmask"),
        "depth_mean_u": interior2("depth_mean_u"),
        "depth_mean_v": interior2("depth_mean_v"),
        "r1_hu0": full2("r1_hu0"),
        "r1_hv0": full2("r1_hv0"),
        "post_drag_u": interior2("post_drag_u"),
        "post_drag_v": interior2("post_drag_v"),
        "cd_u": full2("cd_u"),
        "cd_v": full2("cd_v"),
        "r1_rho0": values(1, "r1_rho0"),
        "wind_tau_u": full2("wind_tau_u"),
        "wind_tau_v": full2("wind_tau_v"),
        "wind_r1_hu": full2("wind_r1_hu"),
        "wind_r1_hv": full2("wind_r1_hv"),
        "post_wind_u": interior2("post_wind_u"),
        "post_wind_v": interior2("post_wind_v"),
    }
    require(offset == len(payload), "trailing Round-140 payload")
    require(tuple(fields) == ROUND140_FIELDS,
            "Round-140 field registry changed")
    return {
        "header": {"magic": magic, "version": version, "kt": kt,
                   "Kbb": kbb, "Krhs": krhs, "nx": nx, "ny": ny,
                   "nz": nz, "bits": bits, "sizes": sizes},
        "fields": fields,
    }


def _identity(a: np.ndarray, b: np.ndarray) -> dict:
    unequal = a.view(np.uint64) != b.view(np.uint64)
    return {"bit_exact": not bool(np.any(unequal)),
            "differing_cells": int(np.count_nonzero(unequal)),
            "max_abs": float(np.max(np.abs(a - b))) if a.size else 0.0}


def _parent_comparison(candidate: dict, baseline: dict) -> dict:
    require(candidate["header"] == baseline["header"],
            "Round-140 parent header moved")
    require(tuple(candidate["fields"]) == ROUND140_FIELDS
            and tuple(baseline["fields"]) == ROUND140_FIELDS,
            "Round-140 parent field registry moved")
    rows = {}
    for name in ROUND140_FIELDS:
        candidate_field = candidate["fields"][name]
        baseline_field = baseline["fields"][name]
        require(candidate_field.shape == baseline_field.shape,
                f"Round-140 parent shape moved for {name}")
        row = {"all": _identity(candidate_field, baseline_field)}
        if name in ("rhs_u", "rhs_v"):
            mask_name = "umask" if name == "rhs_u" else "vmask"
            mask = baseline["fields"][mask_name] != 0.0
        elif name in INTERIOR_OWNED_FIELDS:
            mask = np.zeros(candidate_field.shape, dtype=bool)
            mask[2:-2, 2:-2] = True
        else:
            mask = None
        if mask is not None:
            row["owned"] = _identity(candidate_field[mask], baseline_field[mask])
            row["excluded"] = _identity(
                candidate_field[~mask], baseline_field[~mask])
            row["owned_cells"] = int(np.count_nonzero(mask))
            row["excluded_cells"] = int(np.count_nonzero(~mask))
        rows[name] = row
    owned_fields = ("rhs_u", "rhs_v") + INTERIOR_OWNED_FIELDS
    exact_fields = tuple(name for name in ROUND140_FIELDS
                         if name not in owned_fields)
    passive = (
        all(rows[name]["all"]["bit_exact"] for name in exact_fields)
        and all(rows[name]["owned"]["bit_exact"] for name in owned_fields)
    )
    return {"passive": passive, "rows": rows}


def _exact_inherited(root: Path, baseline_root: Path,
                     plant_restart: bool = False) -> dict:
    rows = {}
    for name in EXACT_INHERITED:
        candidate = (root / name).read_bytes()
        baseline = (baseline_root / name).read_bytes()
        if plant_restart and name == "GYRE_OMIP_L2_P3_00001081_restart.nc":
            candidate = candidate[:-1] + bytes((candidate[-1] ^ 1,))
        rows[name] = {
            "bit_exact": candidate == baseline,
            "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
            "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        }
    require(all(row["bit_exact"] for row in rows.values()),
            "an exact inherited artifact moved")
    return rows


def _git_stamp(repo: Path, expected: str) -> dict:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo, text=True)
    require(head == expected, f"commit {head} != expected {expected}")
    require(not status, "worktree is dirty")
    return {"commit": head, "clean": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round140-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=("none", "header", "truncation",
                                             "final-ulp", "missing-field",
                                             "parent-wet-ulp",
                                             "parent-dry-ulp",
                                             "parent-interior-ulp",
                                             "parent-halo-ulp",
                                             "restart-byte"),
                        default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    details = {}
    try:
        payload = (args.root / RECORD).read_bytes()
        if args.plant == "header":
            changed = bytearray(payload)
            changed[16:20] = struct.pack("=i", 2)
            payload = bytes(changed)
        elif args.plant == "truncation":
            payload = payload[:-1]
        record = read_record_bytes(payload)
        if args.plant == "missing-field":
            del record["fields"][FIELDS[-1]]
            require(tuple(record["fields"]) == FIELDS, "field registry changed")
        baseline_payload = (args.round140_root / ROUND140_RECORD).read_bytes()
        require(hashlib.sha256(baseline_payload).hexdigest()
                == ROUND140_EXPECTED_SHA,
                "Round-140 completed-RHS ancestry changed")
        baseline = read_round140_bytes(baseline_payload)
        candidate_parent = read_round140_bytes(
            (args.root / ROUND140_RECORD).read_bytes())
        parent_plants = ("parent-wet-ulp", "parent-dry-ulp",
                         "parent-interior-ulp", "parent-halo-ulp")
        if args.plant in parent_plants:
            if args.plant in ("parent-wet-ulp", "parent-dry-ulp"):
                field_name = "rhs_u"
                mask = candidate_parent["fields"]["umask"] != 0.0
                target_owned = args.plant == "parent-wet-ulp"
            else:
                field_name = "cd_u"
                mask = np.zeros((NY, NX), dtype=bool)
                mask[2:-2, 2:-2] = True
                target_owned = args.plant == "parent-interior-ulp"
            locations = np.argwhere(mask if target_owned else ~mask)
            require(locations.size > 0, "parent plant has no target cell")
            index = tuple(int(value) for value in locations[0])
            field = candidate_parent["fields"][field_name]
            field[index] = np.nextafter(field[index], np.inf)
        parent = _parent_comparison(candidate_parent, baseline)
        details["parent_vs_round140"] = parent
        require(parent["passive"],
                "Round-140 parent moved on a model-owned cell")
        if args.plant in ("parent-dry-ulp", "parent-halo-ulp"):
            field_name = "rhs_u" if args.plant == "parent-dry-ulp" else "cd_u"
            excluded = parent["rows"][field_name]["excluded"]
            require(not excluded["bit_exact"]
                    and parent["rows"][field_name]["owned"]["bit_exact"],
                    "exclusion plant did not isolate an excluded cell")
            raise GateError("excluded-cell change classified without admitting "
                            "an owned-cell change")
        inherited = _exact_inherited(
            args.root, args.round140_root,
            plant_restart=args.plant == "restart-byte")
        details["exact_inherited"] = inherited
        if args.plant == "final-ulp":
            record["fields"]["after_zad_u"][2, 2, 0] = np.nextafter(
                record["fields"]["after_zad_u"][2, 2, 0], np.inf)
        final_vs_same_run = {
            "u": _identity(record["fields"]["after_zad_u"],
                           candidate_parent["fields"]["rhs_u"]),
            "v": _identity(record["fields"]["after_zad_v"],
                           candidate_parent["fields"]["rhs_v"]),
        }
        final_vs_baseline_owned = {}
        for face in ("u", "v"):
            mask = baseline["fields"][f"{face}mask"] != 0.0
            final_vs_baseline_owned[face] = _identity(
                record["fields"][f"after_zad_{face}"][mask],
                baseline["fields"][f"rhs_{face}"][mask])
        require(all(row["bit_exact"] for row in final_vs_same_run.values()),
                "final ZAD boundary differs from same-run completed RHS")
        require(all(row["bit_exact"]
                    for row in final_vs_baseline_owned.values()),
                "final ZAD boundary differs from admitted owned RHS")
        stamp = _git_stamp(args.repo, args.expect_commit)
        result = {"format": "nemo-testcase-l2-gyre-round146-rhs-family-v2",
                  "status": "PASS", "worktree": stamp,
                  "record_sha256": hashlib.sha256(
                      (args.root / RECORD).read_bytes()).hexdigest(),
                  "expected_size": EXPECTED_SIZE,
                  "fields": list(FIELDS),
                  "exact_inherited": inherited,
                  "parent_vs_round140": parent,
                  "final_vs_same_run": final_vs_same_run,
                  "final_vs_round140_owned": final_vs_baseline_owned}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("ROUND146 RHS FAMILY RECORD PASS")
        return 0
    except (GateError, OSError, ValueError) as error:
        result = {"format": "nemo-testcase-l2-gyre-round146-rhs-family-v2",
                  "status": "PLANT-FIRED" if args.plant != "none" else "FAIL",
                  "plant": args.plant, "error": str(error), **details}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        marker = "STATUS PLANT-FIRED" if args.plant != "none" else "STATUS FAIL"
        print(f"ROUND146 RHS FAMILY {args.plant.upper()} {marker}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
