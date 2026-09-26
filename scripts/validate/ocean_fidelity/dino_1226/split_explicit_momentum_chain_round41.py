#!/usr/bin/env python3
"""Score row-5 production W against the retained NEMO call-2 stream."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import netCDF4
import numpy as np

import split_explicit_momentum_chain_round38 as r38


ROUND40_SHA = "723fe0e74724febab71537ef31cde16e32388c9da01c9058ef2c37629067258b"
ORACLE_SHA = {
    "wzv_dump_ww_call1.bin": "defad5014cc7210dd52d6373dafd46856471afedb892267cef30232fdd9baeb2",
    "wzv_dump_ww_call2.bin": "895a141385f33775c4df7fc127a4beadf2e8e496f68a46624931cf8eb1487634",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_capture(directory: Path) -> tuple[np.ndarray, dict]:
    meta_path = directory / "capture.json"
    meta = json.loads(meta_path.read_text())
    info = meta["files"]["wzv_call2_production.bin"]
    stream = directory / "wzv_call2_production.bin"
    if (info.get("shape") != [199, 52, 37]
            or info.get("size_bytes") != 199 * 52 * 37 * 8
            or _sha(stream) != info.get("sha256")):
        raise SystemExit("production capture admission failed")
    return np.fromfile(stream, dtype="<f8").reshape(199, 52, 37), meta


def _two_bar_plant(oracle: np.ndarray, mask: np.ndarray) -> bool:
    planted = np.array(oracle, copy=True)
    wet = np.argwhere(mask)
    point = tuple(wet[int(np.argmax(np.abs(oracle[mask])))])
    rms = float(np.sqrt(np.mean(oracle[mask] ** 2)))
    planted[point] += 2.0e-12 * rms
    return r38._metric(
        planted, oracle, mask,
        "wzv (vertical velocity)")["gate_status"] != "AT BAR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bracket", type=Path, required=True)
    parser.add_argument("--bracket-sha", required=True)
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--round40", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True).strip():
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round40.resolve()) != ROUND40_SHA:
        raise SystemExit("official round-40 receipt changed")
    prior = json.loads(args.round40.read_text())
    if prior.get("disposition") != "ROW4_ZAD_AT_BAR":
        raise SystemExit("round 40 does not release row 5")
    if _sha(args.bracket.resolve()) != args.bracket_sha:
        raise SystemExit("bracket SLOT/hash mismatch")
    bracket = json.loads(args.bracket.read_text())
    if (bracket.get("session_id") != session
            or not bracket.get("duplicate_byte_exact")):
        raise SystemExit("duplicate bracket failed")

    run = args.run_stepdump.resolve()
    for name, expected in ORACLE_SHA.items():
        if _sha(run / name) != expected:
            raise SystemExit(f"retained oracle changed: {name}")
    candidate_full, capture = _load_capture(args.capture.resolve())
    admitted_meta = _sha(args.capture.resolve() / "capture.json")
    if (capture.get("session_id") != session
            or capture.get("raw_capture_count", 0) < 1
            or capture.get("post_dyn_zdf_matching_raw_count") != 1
            or not capture.get("hook_restored")
            or admitted_meta != bracket["captures"][0]["metadata_sha256"]):
        raise SystemExit("capture/bracket provenance mismatch")

    call1 = r38._load_ww(run / "wzv_dump_ww_call1.bin")
    call2 = r38._load_ww(run / "wzv_dump_ww_call2.bin")
    candidate = candidate_full[..., :36]
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        full_mask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    # This retained mesh_mask is already the bridge/interior 199x52 domain;
    # only the raw STREAM writer is full-halo.  Do not strip it twice.
    if full_mask.shape != (199, 52, 36):
        raise SystemExit(
            f"mesh_mask cited-interior shape changed: {full_mask.shape}")
    mask = full_mask
    if candidate.shape != call2.shape or candidate.shape != mask.shape:
        raise SystemExit(
            f"interior shape mismatch: {candidate.shape}, {call2.shape}, {mask.shape}")
    if int(np.count_nonzero(mask[..., 0])) != 9_920:
        raise SystemExit("active T-column population drift")

    row = r38._metric(candidate, call2, mask, "wzv (vertical velocity)")
    wrong_order = r38._metric(candidate, call1, mask, "wzv (vertical velocity)")
    identity = r38._metric(call2, call2, mask, "wzv (vertical velocity)")
    controls = {
        "identity_at_bar": identity["gate_status"] == "AT BAR",
        "sign_plant_fires": r38._metric(
            -call2, call2, mask, "wzv (vertical velocity)")["gate_status"] != "AT BAR",
        "roll_plant_fires": r38._metric(
            np.roll(call2, 1, axis=1), call2, mask,
            "wzv (vertical velocity)")["gate_status"] != "AT BAR",
        "two_bar_point_plant_fires": _two_bar_plant(call2, mask),
        "call_order_control_has_power": (
            wrong_order["normalized_rms_error"] > row["normalized_rms_error"]),
        "duplicate_byte_exact": bool(bracket["duplicate_byte_exact"]),
    }
    valid = all(controls.values())
    if not valid:
        disposition = "INVALID"
    elif row["gate_status"] == "AT BAR":
        disposition = "ROW5_WZV_CALL2_AT_BAR"
    else:
        disposition = "ROW5_WZV_CALL2_DIVERGED"

    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round41-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 5,
        "nemo_lines": {
            "call_order": "cfgs/DINO/MY_SRC/stpmlf.F90:396,411-412,578",
            "recurrence": "cfgs/DINO/MY_SRC/sshwzv.F90:198-228",
            "writer": "cfgs/DINO/MY_SRC/sshwzv.F90:276-300",
        },
        "row5": row,
        "wrong_order_call1": wrong_order,
        "controls": controls,
        "shape_receipt": {
            "legoesm_interior": [199, 52, 37],
            "nemo_full_halo": [36, 203, 56],
            "mesh_mask_cited_interior": [199, 52, 36],
            "compared": [199, 52, 36],
            "active_surface_columns": int(np.count_nonzero(mask[..., 0])),
        },
        "bindings": {
            "round40": _sha(args.round40.resolve()),
            "capture_metadata": admitted_meta,
            "bracket": _sha(args.bracket.resolve()),
            **{name: _sha(run / name) for name in ORACLE_SHA},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round41.md"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "nemo_sshwzv": _sha(nemo / "cfgs/DINO/MY_SRC/sshwzv.F90"),
        },
        "disposition": disposition,
        "ordered_next": 6 if disposition == "ROW5_WZV_CALL2_AT_BAR" else 5,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    print("row5", row["gate_status"], row["normalized_rms_error"])
    print("call1_control", wrong_order["gate_status"],
          wrong_order["normalized_rms_error"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
