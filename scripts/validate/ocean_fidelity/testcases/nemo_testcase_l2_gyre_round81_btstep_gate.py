#!/usr/bin/env python3
"""Fail-closed gate for the Round-81 GYRE kt=2 external-step record."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.time_levels import time_level_for_dump

RECORD = "oracle_bt_step_operands_kt00000002.bin"
UAMID_RECORD = "oracle_bt_uamid_operands_kt00000002.bin"
MAGIC = "NEMO_L2_BTSTP_1"
N_CYCLE = 50
JPI, JPJ = 36, 26
NTSI, NTEI, NTSJ, NTEJ = 3, 34, 3, 24
NX, NY = NTEI - NTSI + 1, NTEJ - NTSJ + 1
ARRAY_FIELDS = (
    "u_entry", "v_entry", "eta_entry",
    "u_b", "v_b", "eta_b",
    "u_bb", "v_bb", "eta_bb",
    "u_mid", "v_mid", "eta_mid",
    "depth_u_mid", "depth_v_mid",
    "transport_u", "transport_v",
    "ssh_forcing", "continuity_div", "eta_continuity",
    "eta_pgf", "pgf_u", "pgf_v", "cor_u", "cor_v",
    "drag_coefficient_u", "drag_coefficient_v",
    "inverse_depth_u", "inverse_depth_v", "trd_u", "trd_v",
    "slow_u", "slow_v", "u_exit", "v_exit",
    "swap_u", "swap_v", "swap_eta",
)
MASK_FIELDS = ("t_mask", "u_mask", "v_mask")
N_ARRAYS = len(ARRAY_FIELDS)
COUNT = NX * NY
PREFIX_SIZE = 16 + 11 * 4 + 8 + len(MASK_FIELDS) * COUNT * 8
SUBSTEP_SIZE = 4 + 3 * 8 + N_ARRAYS * COUNT * 8 + 4 * 8
EXPECTED_SIZE = PREFIX_SIZE + N_CYCLE * SUBSTEP_SIZE


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"REFUSE: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_array(handle, name: str, substep: int | None = None) -> np.ndarray:
    values = np.fromfile(handle, dtype=np.float64, count=COUNT)
    where = f" at substep {substep}" if substep is not None else ""
    require(values.size == COUNT, f"truncated {name}{where}")
    result = values.reshape((NX, NY), order="F").T
    require(np.all(np.isfinite(result)), f"non-finite {name}{where}")
    return result


def read_record(path: Path, *, expected_kt: int = 2) -> dict:
    require(time_level_for_dump(path.name) == "now", "record level is not NOW")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        raw_header = handle.read(44)
        require(len(raw_header) == 44, "truncated header")
        header = struct.unpack("=11i", raw_header)
        (version, kt, ncycle, jpi, jpj, bits, ntsi, ntei, ntsj, ntej,
         nfields) = header
        require(
            (magic, version, kt, ncycle, jpi, jpj, bits, ntsi, ntei,
             ntsj, ntej, nfields)
            == (MAGIC, 1, expected_kt, N_CYCLE, JPI, JPJ, 64, NTSI,
                NTEI, NTSJ, NTEJ, N_ARRAYS),
            f"bad header {(magic, *header)}",
        )
        raw_dt = handle.read(8)
        require(len(raw_dt) == 8, "truncated external timestep")
        (dt_s,) = struct.unpack("=d", raw_dt)
        require(np.isfinite(dt_s) and dt_s > 0.0, "invalid external timestep")
        masks = {name: _read_array(handle, name) for name in MASK_FIELDS}
        rows = {name: [] for name in ARRAY_FIELDS}
        mid_coefficients = []
        back_coefficients = []
        for expected in range(1, N_CYCLE + 1):
            raw_jn = handle.read(4)
            require(len(raw_jn) == 4, f"truncated substep {expected}")
            (jn,) = struct.unpack("=i", raw_jn)
            require(jn == expected, f"substep {jn} != {expected}")
            mid = np.fromfile(handle, dtype=np.float64, count=3)
            require(mid.size == 3 and np.all(np.isfinite(mid)),
                    f"bad midpoint coefficients {expected}")
            mid_coefficients.append(mid)
            for name in ARRAY_FIELDS[:19]:
                rows[name].append(_read_array(handle, name, expected))
            back = np.fromfile(handle, dtype=np.float64, count=4)
            require(back.size == 4 and np.all(np.isfinite(back)),
                    f"bad backward coefficients {expected}")
            back_coefficients.append(back)
            for name in ARRAY_FIELDS[19:]:
                rows[name].append(_read_array(handle, name, expected))
        require(handle.read(1) == b"", "trailing payload")
    return {
        "header": {
            "version": version, "kt": kt, "ncycle": ncycle,
            "jpi": jpi, "jpj": jpj, "bits": bits,
            "ntsi": ntsi, "ntei": ntei, "ntsj": ntsj,
            "ntej": ntej, "nfields": nfields,
            "registry_level": "now",
        },
        "dt_s": np.float64(dt_s),
        **masks,
        "mid_coefficients": np.stack(mid_coefficients),
        "back_coefficients": np.stack(back_coefficients),
        **{name: np.stack(values) for name, values in rows.items()},
    }


def _bits_equal(left, right) -> bool:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _row(candidate, oracle) -> dict:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(candidate.shape == oracle.shape, "replay shape mismatch")
    require(np.all(np.isfinite(candidate)) and np.all(np.isfinite(oracle)),
            "non-finite replay row")
    delta = candidate - oracle
    return {
        "bit_exact": _bits_equal(candidate, oracle),
        "differing_cells": int(np.count_nonzero(
            candidate.view(np.uint64) != oracle.view(np.uint64))),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
    }


def _midpoint(coefficients, now, before, before_before):
    first = coefficients[0] * now
    second = coefficients[1] * before
    third = coefficients[2] * before_before
    return (first + second) + third


def validate_fields(fields: dict, *, replay_ulp: bool = False,
                    swap_ulp: bool = False) -> list[dict]:
    rows = []
    dt_s = fields["dt_s"]
    for index in range(N_CYCLE):
        mid = fields["mid_coefficients"][index]
        back = fields["back_coefficients"][index]
        targets = {
            "u_mid": np.array(fields["u_mid"][index], copy=True),
            "v_mid": fields["v_mid"][index],
            "eta_mid": fields["eta_mid"][index],
            "eta_continuity": fields["eta_continuity"][index],
            "eta_pgf": fields["eta_pgf"][index],
            "trd_u": fields["trd_u"][index],
            "trd_v": fields["trd_v"][index],
            "u_exit": fields["u_exit"][index],
            "v_exit": fields["v_exit"][index],
            "swap_u": np.array(fields["swap_u"][index], copy=True),
            "swap_v": fields["swap_v"][index],
            "swap_eta": fields["swap_eta"][index],
        }
        if replay_ulp and index == 0:
            targets["u_mid"].flat[0] = np.nextafter(
                targets["u_mid"].flat[0], np.inf)
        if swap_ulp and index == 0:
            targets["swap_u"].flat[0] = np.nextafter(
                targets["swap_u"].flat[0], np.inf)
        candidate = {
            "u_mid": _midpoint(mid, fields["u_entry"][index],
                               fields["u_b"][index], fields["u_bb"][index]),
            "v_mid": _midpoint(mid, fields["v_entry"][index],
                               fields["v_b"][index], fields["v_bb"][index]),
            "eta_mid": _midpoint(mid, fields["eta_entry"][index],
                                 fields["eta_b"][index], fields["eta_bb"][index]),
            "eta_continuity": (
                fields["eta_entry"][index]
                - dt_s * (fields["ssh_forcing"][index]
                          + fields["continuity_div"][index])
            ) * fields["t_mask"],
            "eta_pgf": (
                back[0] * fields["eta_continuity"][index]
                + back[1] * fields["eta_entry"][index]
                + back[2] * fields["eta_b"][index]
                + back[3] * fields["eta_bb"][index]
            ),
            "trd_u": fields["cor_u"][index] + (
                fields["drag_coefficient_u"][index]
                * fields["u_entry"][index]
                * fields["inverse_depth_u"][index]),
            "trd_v": fields["cor_v"][index] + (
                fields["drag_coefficient_v"][index]
                * fields["v_entry"][index]
                * fields["inverse_depth_v"][index]),
            "u_exit": (
                fields["u_entry"][index]
                + dt_s * (fields["pgf_u"][index] + fields["trd_u"][index]
                          + fields["slow_u"][index])
            ) * fields["u_mask"],
            "v_exit": (
                fields["v_entry"][index]
                + dt_s * (fields["pgf_v"][index] + fields["trd_v"][index]
                          + fields["slow_v"][index])
            ) * fields["v_mask"],
            "swap_u": fields["u_exit"][index],
            "swap_v": fields["v_exit"][index],
            "swap_eta": fields["eta_continuity"][index],
        }
        row = {"substep": index + 1}
        row.update({name: _row(value, targets[name])
                    for name, value in candidate.items()})
        rows.append(row)
    require(all(result["bit_exact"] for row in rows
                for name, result in row.items() if name != "substep"),
            "compiled external-step replay is not bit-exact")
    return rows


def compare_uamid(fields: dict, path: Path, *, plant: bool = False) -> dict:
    import importlib.util

    gate_path = Path(__file__).with_name(
        "nemo_testcase_l2_gyre_round76_uamid_gate.py")
    spec = importlib.util.spec_from_file_location("round76_uamid_gate", gate_path)
    require(spec is not None and spec.loader is not None,
            "cannot load admitted U-midpoint reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    admitted = module.read_record(path)
    pairs = {
        "coefficients": fields["mid_coefficients"],
        "un_e": fields["u_entry"],
        "ub_e": fields["u_b"],
        "ubb_e": fields["u_bb"],
        "ua_e": np.array(fields["u_mid"], copy=True),
    }
    if plant:
        pairs["ua_e"].flat[0] = np.nextafter(pairs["ua_e"].flat[0], np.inf)
    rows = {name: _row(value, admitted[name]) for name, value in pairs.items()}
    require(all(row["bit_exact"] for row in rows.values()),
            "Round-81 U rows differ from admitted Round-77 record")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--uamid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant",
        choices=("none", "stamp", "header", "truncation", "replay-ulp",
                 "swap-ulp", "uamid-ulp"),
        default="none",
    )
    args = parser.parse_args()
    path = args.root / RECORD
    stamp = args.root / f"{RECORD}.stamp"
    require(path.is_file() and stamp.is_file(), "record or stamp missing")
    producer = (args.root / "producer_commit.txt").read_text().strip()
    require(producer == args.expect_commit, "producer_commit.txt mismatch")
    expected_commit = "planted-wrong-commit" if args.plant == "stamp" else producer
    require(stamp.read_text().split() == [sha256(path), expected_commit, RECORD],
            "record stamp mismatch")
    observed_size = path.stat().st_size - (1 if args.plant == "truncation" else 0)
    require(observed_size == EXPECTED_SIZE, "record byte size is not exact")
    fields = read_record(path, expected_kt=(3 if args.plant == "header" else 2))
    rows = validate_fields(
        fields, replay_ulp=args.plant == "replay-ulp",
        swap_ulp=args.plant == "swap-ulp")
    uamid = compare_uamid(
        fields, args.uamid_root / UAMID_RECORD,
        plant=args.plant == "uamid-ulp")
    report = {
        "format": "nemo-testcase-l2-gyre-round81-btstep-v1",
        "producer_commit": producer,
        "record_sha256": sha256(path),
        "record_size": path.stat().st_size,
        "header": fields["header"],
        "dt_s": float(fields["dt_s"]),
        "replay_rows": rows,
        "round77_uamid_identity": uamid,
        "plant": args.plant,
        "status": "AT-BAR",
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("ROUND81 BTSTEP RECORD AT BAR")


if __name__ == "__main__":
    main()
