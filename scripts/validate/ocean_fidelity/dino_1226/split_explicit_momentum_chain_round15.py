#!/usr/bin/env python3
"""Bind hardened QCO acceptance to the ordered full-recurrence replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from zdf_stream_bracket import sha256


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance", type=Path, required=True)
    p.add_argument("--recurrence", type=Path, required=True)
    p.add_argument("--package-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != args.package_commit or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True):
        raise SystemExit("clean package commit required")
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID required")

    acceptance = _load(args.acceptance)
    recurrence = _load(args.recurrence)
    if acceptance.get("git_commit") != commit \
            or recurrence.get("git", {}).get("commit") != commit:
        raise SystemExit("both inputs must be measured at package commit")
    if acceptance.get("session_id") != session \
            or recurrence.get("session_id") != session:
        raise SystemExit("session mismatch")
    if acceptance.get("backend") != "cpu" \
            or not acceptance.get("jax_enable_x64"):
        raise SystemExit("acceptance is not CPU/fp64")
    if recurrence.get("backend") != "cpu" \
            or not recurrence.get("jax_enable_x64") \
            or recurrence.get("lane") != "d180" \
            or recurrence.get("e3t_mode") != "both":
        raise SystemExit("recurrence runtime mismatch")
    bindings = acceptance.get("bindings", {})
    if set(bindings.get("input_sha256", {})) != {
            "mesh_mask.nc", "DINO_00005760_restart.nc"}:
        raise SystemExit("hardened mesh/restart binding missing")
    recurrence_inputs = recurrence.get("input_sha256", {})
    if any(recurrence_inputs.get(name) != value
           for name, value in bindings["input_sha256"].items()):
        raise SystemExit(
            "acceptance mesh/restart hashes do not match recurrence inputs")
    if set(bindings.get("production_sha256", {})) != {
            "core_grid", "barotropic", "dino_card", "nemo_io", "state_bridge",
            "inherited_probe", "round9_scorer"}:
        raise SystemExit("hardened production binding missing")
    if acceptance.get("disposition") \
            != "ROW_1.3_ALL_OPERANDS_AT_BAR_RELEASE_1.4" \
            or acceptance.get("first_diverged_subrow") is not None \
            or any(row.get("gate_status") != "AT BAR"
                   for row in acceptance.get("measurements", [])):
        raise SystemExit("continuity acceptance did not release recurrence")
    if not all(acceptance.get("controls", {}).values()):
        raise SystemExit("acceptance control failed")
    required_controls = (
        "identical_array_zero", "planted_identity_flips_campaign_gate",
        "actual_binding_perturbation_changes_score", "zero_shift_is_best")
    if not all(recurrence.get("controls", {}).get(k) for k in required_controls):
        raise SystemExit("recurrence control failed")
    if not all(all(v.values()) for v in
               recurrence.get("restoration_receipts", {}).values()):
        raise SystemExit("recurrence monkeypatch restoration failed")
    if any(m.get("normalized_rms_error") != 0.0 for m in
           recurrence.get("held_forcing_metrics", {}).values()):
        raise SystemExit("held forcing is not exact")

    stop = recurrence.get("literal_ordered_stop")
    disposition = ("CHAIN_ROWS_1_2_TO_1_4_CLEAR_RELEASE_2"
                   if stop is None else f"ORDERED_STOP_{stop}")
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round15-v1",
        "session_id": session,
        "git_commit": commit,
        "input_sha256": {
            "hardened_acceptance": sha256(args.acceptance),
            "full_recurrence": sha256(args.recurrence),
        },
        "hardened_bindings": bindings,
        "continuity_disposition": acceptance["disposition"],
        "literal_ordered_stop": stop,
        "disposition": disposition,
        "measurements": recurrence["measurements"],
        "later_rows": recurrence["later_rows"],
        "controls": {
            "acceptance": acceptance["controls"],
            "recurrence": recurrence["controls"],
            "restoration": recurrence["restoration_receipts"],
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND15 disposition={disposition} artifact={args.output} "
          f"sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
