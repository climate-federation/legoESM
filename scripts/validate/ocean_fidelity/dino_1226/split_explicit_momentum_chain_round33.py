#!/usr/bin/env python3
"""Ordered row-4 scorer using the committed S17 oracle-input bracket."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import numpy as np

import split_explicit_momentum_chain_round29 as r29
import s17_dynzdf_bracket as s17
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND32_SHA = "0b284d7c8880646850daee86b2b85fb13fb65540402e2e9a02760b0ed60ba86f"
ALIASES = {
    "stp_dump_07_dynspg_u.bin": "stp_dump_07_dynspg_kt00005761_u.bin",
    "stp_dump_07_dynspg_v.bin": "stp_dump_07_dynspg_kt00005761_v.bin",
    "stp_dump_07_dynspg_ub.bin": "stp_dump_07_dynspg_kt00005761_ub.bin",
    "stp_dump_07_dynspg_vb.bin": "stp_dump_07_dynspg_kt00005761_vb.bin",
    "stp_dump_08_dynzdf_u.bin": "stp_dump_08_dynzdf_kt00005761_u.bin",
    "stp_dump_08_dynzdf_v.bin": "stp_dump_08_dynzdf_kt00005761_v.bin",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tracked_status(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def _link_view(run: Path, view: Path) -> None:
    for source in run.iterdir():
        if source.is_file():
            (view / source.name).symlink_to(source)
    for alias, source_name in ALIASES.items():
        source = run / source_name
        if not source.is_file():
            raise SystemExit(f"missing required stream {source}")
        (view / alias).symlink_to(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--entry-restart", type=Path, required=True)
    parser.add_argument("--round32", type=Path, required=True)
    parser.add_argument("--manifest-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        "RETRACTED: round 33 injected NEMO's already barotropic-free naa_B at "
        "the public mixing-method entry, so production stripped the mean a "
        "second time and the scorer compared a post-splice field to NEMO's "
        "pre-splice stage-8 field. Use the round-35 raw-dispatch scorer."
    )
    set_policy(PrecisionPolicy.fp64())

    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked_status(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round32) != ROUND32_SHA:
        raise SystemExit("round-32 certification changed")
    r32 = json.loads(args.round32.read_text())
    if (r32.get("disposition") != "NO_STRICT_SSH_FAILURE"
            or any(value is not None for value in
                   r32["first_strict_failure_substep"].values())):
        raise SystemExit("round-32 receipt does not release row 4")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")

    run = args.run.resolve()
    entry_restart = args.entry_restart.resolve()
    if not entry_restart.is_file():
        raise SystemExit(f"missing day-180 entry restart {entry_restart}")
    manifest_artifact = json.loads(args.manifest_artifact.read_text())
    manifest = manifest_artifact["bracket"]["off_stream_manifest"]
    for source_name in (*ALIASES.values(), "zdf_dump_u1_prestress.bin",
                        "zdf_dump_u1_poststress.bin",
                        "zdf_dump_v1_prestress.bin",
                        "zdf_dump_v1_poststress.bin"):
        path = run / source_name
        expected = manifest.get(source_name)
        if expected is None or _sha(path) != expected["sha256"]:
            raise SystemExit(f"retained stream hash admission failed: {source_name}")

    reports: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    calls: list[dict[str, np.ndarray]] = []
    real_report = s17._report
    real_run = s17._run_with_hook
    real_bridge = s17.bridge_nemo_to_legoesm_topo

    def capture_report(tag, lego, nemo, mask, weights):
        reports[tag] = (np.asarray(lego), np.asarray(nemo),
                        np.asarray(mask, dtype=bool))
        return real_report(tag, lego, nemo, mask, weights)

    def capture_run(*call_args, **call_kwargs):
        result = real_run(*call_args, **call_kwargs)
        calls.append(result)
        return result

    def current_bridge(*call_args, **call_kwargs):
        call_kwargs.setdefault("carry_native_lat_deg", True)
        return real_bridge(*call_args, **call_kwargs)

    with tempfile.TemporaryDirectory(prefix="dino-row33-view.") as tmp:
        view = Path(tmp)
        _link_view(run, view)
        restart_view = view / "DINO_00005760_restart.nc"
        if restart_view.exists():
            if _sha(restart_view) != _sha(entry_restart):
                raise SystemExit("retained entry-restart link disagrees with explicit binding")
        else:
            restart_view.symlink_to(entry_restart)
        s17.RUN_DIR = str(view)
        s17.RESTART = "DINO_00005760_restart.nc"
        s17._report = capture_report
        s17._run_with_hook = capture_run
        s17.bridge_nemo_to_legoesm_topo = current_bridge
        try:
            code = s17.main()
        finally:
            s17._report = real_report
            s17._run_with_hook = real_run
            s17.bridge_nemo_to_legoesm_topo = real_bridge
            s17._SUB["u"] = s17._SUB["v"] = None
    if code != 0:
        raise SystemExit(f"S17 bracket failed with exit {code}")
    if len(calls) != 4:
        raise SystemExit(f"expected four S17 arms, got {len(calls)}")
    required = ("fed_ws post_u", "fed_ws post_v")
    if any(name not in reports for name in required):
        raise SystemExit(f"missing reports: {sorted(set(required)-set(reports))}")

    rows = {}
    controls = {}
    for component, name, expected_n in (
            ("u", "fed_ws post_u", 336_338),
            ("v", "fed_ws post_v", 340_271)):
        candidate, oracle, mask = reports[name]
        population = int(mask.sum())
        if population != expected_n:
            raise SystemExit(
                f"{component} 3-D wet population drift: {population} != {expected_n}")
        rows[component] = r29._score(
            candidate, oracle, mask,
            "dyn_zdf (momentum implicit vertical solve)")
        controls[component] = r29._strict_controls(candidate, oracle, mask)
        bar_plant = np.array(oracle, copy=True)
        wet_points = np.argwhere(mask)
        wet_values = np.abs(oracle[mask])
        point = tuple(wet_points[int(np.argmax(wet_values))])
        reference_rms = float(np.sqrt(np.mean(oracle[mask] ** 2)))
        bar_plant[point] += 2.0e-12 * reference_rms
        controls[component]["two_bar_point_plant"] = r29._score(
            bar_plant, oracle, mask,
            "dyn_zdf (momentum implicit vertical solve)")
        controls[component]["two_bar_point_plant_fires"] = (
            controls[component]["two_bar_point_plant"]["gate_status"] != "AT BAR")

    # S17's null arm is call 2 and self is call 1.
    null_exact = all(np.array_equal(calls[0][name], calls[1][name])
                     for name in ("post_u", "post_v"))
    restored = (s17._report is real_report
                and s17._run_with_hook is real_run
                and s17.bridge_nemo_to_legoesm_topo is real_bridge)
    strict_controls = all(
        item["identity_at_bar"] and item["zero_shift_best"]
        and item["two_bar_point_plant_fires"]
        for item in controls.values())
    valid = null_exact and restored and strict_controls
    if not valid:
        disposition = "INVALID"
    elif all(row["gate_status"] == "AT BAR" for row in rows.values()):
        disposition = "ROW4_DYN_ZDF_VERIFIED"
    else:
        disposition = "ROW4_DYN_ZDF_DIVERGED"

    bindings = {
        "round32": _sha(args.round32),
        "manifest_artifact": _sha(args.manifest_artifact),
        "scorer": _sha(Path(__file__).resolve()),
        "preregistration": _sha(
            root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round33.md"),
        "entry_restart": _sha(entry_restart),
    }
    for source_name in (*ALIASES.values(), "zdf_dump_u1_prestress.bin",
                        "zdf_dump_u1_poststress.bin",
                        "zdf_dump_v1_prestress.bin",
                        "zdf_dump_v1_poststress.bin", "dump_avm.bin",
                        "mesh_mask.nc"):
        bindings[source_name] = _sha(run / source_name)
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round33-row4-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "row": 4,
        "nemo_lines": {
            "call": "stpmlf.F90:396-409",
            "state_and_removal": "dynzdf.F90:137-178",
            "matrix_and_recurrence": "dynzdf.F90:180-337",
        },
        "rows": rows,
        "controls": {
            "component_controls": controls,
            "null_substitution_byte_exact": null_exact,
            "hooks_restored": restored,
            "strict_controls_pass": strict_controls,
        },
        "bindings": bindings,
        "disposition": disposition,
        "ordered_next": 5 if disposition == "ROW4_DYN_ZDF_VERIFIED" else 4,
    }
    if _tracked_status(root):
        raise SystemExit("tracked worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for component, row in rows.items():
        print(component, row["gate_status"],
              row["normalized_rms_error"],
              row["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
