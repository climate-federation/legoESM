#!/usr/bin/env python3
"""Offline executed-live QCO association peel for mlf_baro_corr."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import netCDF4
import numpy as np

import split_explicit_momentum_chain_round47 as r47
import split_explicit_momentum_chain_round48 as r48


ROUND48_SHA = "8055e1138c7784942167ae06212e2801d78f6690d3646351b6040d557a70c06b"
R3_SHA = {
    "seq_dump_r3u_aaa_kt00005761.bin": "5b60d354f052af78762d5c8b178d27c3a62507364e675cbbf9aa787d85088b97",
    "seq_dump_r3v_aaa_kt00005761.bin": "c6ab4be3184a02539e9de8e53d7b4c496a9307faf496e3ef8a7154da4d471023",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load2(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != 203 * 56:
        raise SystemExit(f"{path}: expected full-halo (203,56) stream")
    return raw.reshape(203, 56)[2:-2, 2:-2]


def _evaluate(before, target, r3, e3ref, h0, mask, live):
    wet2 = mask[..., 0].astype(np.float64)
    r1h0 = wet2 / (h0 + 1.0 - wet2)
    scale = (1.0 + r3[..., None] * mask) if live else np.ones_like(mask)
    e3 = e3ref * scale
    acc = e3[..., 0] * before[..., 0] * mask[..., 0]
    for k in range(1, 35):
        acc = acc + e3[..., k] * before[..., k] * mask[..., k]
    r1 = r1h0 / (1.0 + r3) if live else r1h0
    mean = acc * r1
    return (before - mean[..., None] + target[..., None]) * mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round48", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r47.r46.r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round48.resolve()) != ROUND48_SHA:
        raise SystemExit("official round-48 receipt changed")
    prior = json.loads(args.round48.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "OPEN_UNRESOLVED"):
        raise SystemExit("round 48 does not open the live association peel")
    run = args.run.resolve()
    bound = {**r47.EXPECTED, **r48.TARGET_SHA, **R3_SHA}
    for name, expected in bound.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"retained round-49 input changed: {name}")

    before = {c: r47._load3(run / f"baro_dump_{c}_before_kt00005761.bin")
              for c in ("u", "v")}
    after = {c: r47._load3(run / f"baro_dump_{c}_after_kt00005761.bin")
             for c in ("u", "v")}
    target = {"u": _load2(run / "spg_dump_puu_b_final.bin"),
              "v": _load2(run / "spg_dump_pvv_b_final.bin")}
    r3 = {c: _load2(run / f"seq_dump_r3{c}_aaa_kt00005761.bin")
          for c in ("u", "v")}
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        data = {}
        for c in ("u", "v"):
            e3 = np.moveaxis(
                np.asarray(ds[f"e3{c}_0"][0], dtype=np.float64), 0, -1)[..., :35]
            mask = np.moveaxis(
                np.asarray(ds[f"{c}mask"][0]), 0, -1)[..., :35]
            h0 = e3[..., 0] * mask[..., 0]
            for k in range(1, 35):
                h0 = h0 + e3[..., k] * mask[..., k]
            data[c] = {"e3": e3, "h0": h0, "mask": mask}
    arms = {}
    arrays = {}
    for live, name in ((False, "C0_cancelled"), (True, "C1_executed_live")):
        arrays[name] = {
            c: _evaluate(before[c], target[c], r3[c], data[c]["e3"],
                         data[c]["h0"], data[c]["mask"], live)
            for c in ("u", "v")}
        arms[name] = {
            c: r47._metric(arrays[name][c], after[c], data[c]["mask"] > 0.5)
            for c in ("u", "v")}

    controls = {
        "c0_reproduces_round48_u": arms["C0_cancelled"]["u"]
            == prior["arms"]["T1A1"]["u"],
        "c0_reproduces_round48_v": arms["C0_cancelled"]["v"]
            == prior["arms"]["T1A1"]["v"],
        "live_factor_u_nonzero": bool(np.any(r3["u"][data["u"]["mask"][..., 0] > .5] != 0)),
        "live_factor_v_nonzero": bool(np.any(r3["v"][data["v"]["mask"][..., 0] > .5] != 0)),
        "identity_u_at_bar": r47._metric(after["u"], after["u"], data["u"]["mask"] > .5)["gate_status"] == "AT BAR",
        "identity_v_at_bar": r47._metric(after["v"], after["v"], data["v"]["mask"] > .5)["gate_status"] == "AT BAR",
        "u_roll_fires": r47._metric(np.roll(after["u"], 1, axis=1), after["u"], data["u"]["mask"] > .5)["gate_status"] != "AT BAR",
        "v_roll_fires": r47._metric(np.roll(after["v"], 1, axis=1), after["v"], data["v"]["mask"] > .5)["gate_status"] != "AT BAR",
        "u_point_fires": r47._plant(after["u"], data["u"]["mask"] > .5),
        "v_point_fires": r47._plant(after["v"], data["v"]["mask"] > .5),
    }
    valid = all(controls.values())
    live_at_bar = all(arms["C1_executed_live"][c]["gate_status"] == "AT BAR"
                      for c in ("u", "v"))
    disposition = (
        "ROW6_LOCALIZED_TO_LIVE_QCO_ASSOCIATION_GIVEN_ORACLE_TARGET"
        if valid and live_at_bar else "INVALID" if not valid else "OPEN_UNRESOLVED")
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round49-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 6,
        "arms": arms,
        "controls": controls,
        "bindings": {
            "round48": _sha(args.round48.resolve()),
            **{name: _sha(run / name) for name in bound},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round49.md"),
        },
        "disposition": disposition,
        "ordered_next": 6,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name in arms:
        print(name, arms[name]["u"]["per_element_max_error_over_nemo_rms"],
              arms[name]["v"]["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
