#!/usr/bin/env python3
"""Bind and score the held row-1.3 bottom-stress/vector-update instrument."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import spg_substep_chain as inherited
from zdf_stream_bracket import (
    files_byte_identical,
    manifest_sha256,
    one_bit_file_control,
    sha256,
    stream_manifest,
)

ROUND25_SHA = "a9db28c458bb6e409ec23b74ea371b2d2e4aeaf1d84de322074ab72b46fca16e"
POINTWISE = 1.0e-15
NEW_STREAMS = (
    "row13_dump_zCdU_u_substep1.bin",
    "row13_dump_zCdU_v_substep1.bin",
    "row13_dump_hu_e_substep1.bin",
    "row13_dump_hv_e_substep1.bin",
    "row13_dump_hur_e_substep1.bin",
    "row13_dump_hvr_e_substep1.bin",
    "row13_dump_zu_trd_after_bottom_substep1.bin",
    "row13_dump_zv_trd_after_bottom_substep1.bin",
    "row13_dump_zu_spg_substep1.bin",
    "row13_dump_zv_spg_substep1.bin",
    "row13_dump_ua_e_after_update_substep1.bin",
    "row13_dump_va_e_after_update_substep1.bin",
)


def _metric(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    wet = np.asarray(mask, dtype=bool)
    a = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64)[wet])
    b = np.ascontiguousarray(np.asarray(oracle, dtype=np.float64)[wet])
    if not a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise SystemExit("empty or non-finite registered population")
    rms = float(np.sqrt(np.mean(b * b)))
    if rms == 0.0:
        raise SystemExit("zero oracle RMS on registered population")
    delta = a - b
    error = float(np.sqrt(np.mean(delta * delta)) / rms)
    maximum = float(np.max(np.abs(delta)) / rms)
    return {
        "n": int(a.size),
        "normalized_rms_error": error,
        "max_error_over_nemo_rms": maximum,
        "bit_mismatch_count": int(np.count_nonzero(
            a.view(np.uint64) != b.view(np.uint64))),
        "gate_status": (
            "AT BAR" if error <= POINTWISE and maximum <= POINTWISE else "DEBT"
        ),
    }


def _rdt(run: Path) -> float:
    text = (run / "ocean.output").read_text(errors="replace")
    match = re.search(
        r"Barotropic time steps => in seconds\s*=\s*([0-9.eEdD+-]+)", text)
    if match is None:
        raise SystemExit("missing rDt_e runtime receipt")
    value = float(match.group(1).replace("D", "E").replace("d", "e"))
    if not np.isfinite(value) or value <= 0.0:
        raise SystemExit(f"invalid rDt_e {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--on-run", type=Path, required=True)
    parser.add_argument("--off-run", type=Path, required=True)
    parser.add_argument("--bracket", type=Path, required=True)
    parser.add_argument("--bracket-sha", required=True)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--round25", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Shared campaign worktrees intentionally carry unrelated untracked run
    # artifacts. Ignore those, but still stop on staged or unstaged tracked
    # edits to any committed scorer/model input.
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True).strip():
        raise SystemExit("tracked-clean scorer checkout required")
    subprocess.run(
        ["git", "cat-file", "-e", f"{args.producer}^{{commit}}"],
        cwd=root, check=True)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", args.producer, head, "--",
         "packages/core", "packages/ocean", "src"],
        cwd=root, text=True).strip()
    if changed:
        raise SystemExit(
            f"model differs from SHA-pinned producer {args.producer}: {changed}")
    if sha256(args.round25) != ROUND25_SHA:
        raise SystemExit("round-25 artifact SHA changed")
    round25 = json.loads(args.round25.read_text())
    if not (
        round25.get("disposition")
        == "NEMO_SOURCE_ASSOCIATION_OWNS_FINAL_COEFFICIENT_DEBT"
        and round25.get("full_literal_reduction") == 1.0
    ):
        raise SystemExit("round-25 exact association verdict changed")

    if sha256(args.bracket) != args.bracket_sha:
        raise SystemExit("bracket receipt SHA changed")
    bracket = json.loads(args.bracket.read_text())
    on, off = args.on_run.resolve(), args.off_run.resolve()
    expected = set(NEW_STREAMS)
    if not (
        bracket.get("schema") == "dino-spg-row13-bottom-update-bracket-v1"
        and Path(bracket.get("on_dir", "")).resolve() == on
        and Path(bracket.get("off_dir", "")).resolve() == off
        and bracket.get("shared_count") == 211
        and bracket.get("shared_exact") is True
        and set(bracket.get("new_streams", [])) == expected
        and all(bracket.get("controls", {}).values())
    ):
        raise SystemExit("invalid bottom/update bracket receipt")
    on_manifest, off_manifest = stream_manifest(on), stream_manifest(off)
    if not (
        len(on_manifest) == 223
        and len(off_manifest) == 211
        and set(on_manifest) - set(off_manifest) == expected
        and all(files_byte_identical(on / name, off / name)
                for name in off_manifest)
        and manifest_sha256({name: on_manifest[name] for name in sorted(off_manifest)})
            == bracket["shared_manifest_sha256"]
    ):
        raise SystemExit("run directories changed after exact bracket")
    if not one_bit_file_control(on / sorted(off_manifest)[0]):
        raise SystemExit("one-bit file control failed")

    jpi, jpj, _jpk, hls, icycle, nn_e = inherited._read_dims(str(on))
    if (jpi, jpj, hls, icycle, nn_e) != (56, 203, 2, 68, 23):
        raise SystemExit(
            f"unexpected DINO runtime dimensions {(jpi, jpj, hls, icycle, nn_e)}")

    def load(name: str) -> np.ndarray:
        return inherited._load_full(str(on / name), jpi, jpj, hls)

    def load_interior(name: str) -> np.ndarray:
        """Load a NEMO A2D(0) stream, whose declaration excludes halos."""
        ni, nj = jpi - 2 * hls, jpj - 2 * hls
        path = on / name
        expected_bytes = ni * nj * np.dtype("<f8").itemsize
        if path.stat().st_size != expected_bytes:
            raise SystemExit(
                f"interior A2D(0) stream {name} has {path.stat().st_size} "
                f"bytes, expected {expected_bytes}")
        return inherited._load_interior(str(path), ni, nj)

    # The mesh's first level is the exact active 2-D U/V population.  The
    # deterministic streams are already halo-cropped by inherited._load_full.
    import xarray as xr
    with xr.open_dataset(on / "mesh_mask.nc", decode_times=False) as ds:
        umask = np.asarray(ds["umask"].squeeze())[0] > 0.5
        vmask = np.asarray(ds["vmask"].squeeze())[0] > 0.5
    # mesh_mask.nc is the canonical halo-free 199x52 mesh, while every raw
    # stream is a full-halo 203x56 writer product cropped by load() above.
    expected_shape = (jpj - 2 * hls, jpi - 2 * hls)
    if umask.shape != expected_shape or vmask.shape != expected_shape:
        raise SystemExit(f"unexpected halo-free mask shapes {umask.shape}/{vmask.shape}")
    if (int(umask.sum()), int(vmask.sum())) != (9758, 9868):
        raise SystemExit("registered U/V populations changed")
    uf, vf = umask.astype(np.float64), vmask.astype(np.float64)

    un, vn = load("cor2d_dump_ua_e_in_substep1.bin"), load("cor2d_dump_va_e_in_substep1.bin")
    cor_u, cor_v = load("cor2d_dump_zu_trd_substep1.bin"), load("cor2d_dump_zv_trd_substep1.bin")
    # dynspg_ts.F90:168 declares these DIMENSION(A2D(0)); their writers emit
    # the genuine 199x52 interior grid. The new jpi,jpj operands remain full
    # halo 203x56 products and continue through load() above.
    frc_u = load_interior("spg_dump_zu_frc.bin")
    frc_v = load_interior("spg_dump_zv_frc.bin")
    zcdu, zcdv = load(NEW_STREAMS[0]), load(NEW_STREAMS[1])
    hu, hv = load(NEW_STREAMS[2]), load(NEW_STREAMS[3])
    hur, hvr = load(NEW_STREAMS[4]), load(NEW_STREAMS[5])
    after_u, after_v = load(NEW_STREAMS[6]), load(NEW_STREAMS[7])
    pgf_u, pgf_v = load(NEW_STREAMS[8]), load(NEW_STREAMS[9])
    out_u, out_v = load(NEW_STREAMS[10]), load(NEW_STREAMS[11])

    reciprocal = {
        "u": _metric(uf / (hu + 1.0 - uf), hur, umask),
        "v": _metric(vf / (hv + 1.0 - vf), hvr, vmask),
    }
    bottom = {
        "u": _metric(cor_u + ((zcdu * un) * hur), after_u, umask),
        "v": _metric(cor_v + ((zcdv * vn) * hvr), after_v, vmask),
    }
    rdt = _rdt(on)
    update = {
        "u": _metric((un + rdt * ((pgf_u + after_u) + frc_u)) * uf,
                     out_u, umask),
        "v": _metric((vn + rdt * ((pgf_v + after_v) + frc_v)) * vf,
                     out_v, vmask),
    }

    identity = _metric(out_u, out_u, umask)["gate_status"] == "AT BAR"
    plant = np.array(out_u, copy=True)
    wet_points = np.argwhere(umask)
    wet_values = np.abs(out_u[umask])
    point = tuple(wet_points[int(np.argmax(wet_values))])
    for _ in range(4):
        plant[point] = np.nextafter(plant[point], np.inf)
    plant_red = _metric(plant, out_u, umask)["gate_status"] == "DEBT"
    controls = {
        "bracket_controls": all(bracket["controls"].values()),
        "identity_at_bar": identity,
        "four_step_nextafter_debt": plant_red,
    }
    if not all(controls.values()):
        raise SystemExit(f"control failed: {controls}")
    all_rows = [*reciprocal.values(), *bottom.values(), *update.values()]
    valid = all(row["gate_status"] == "AT BAR" for row in all_rows)
    disposition = (
        "NEMO_BOTTOM_UPDATE_IDENTITIES_AT_BAR_PRODUCTION_REPLAY_HELD"
        if valid else "INVALID_HELD_INSTRUMENT"
    )
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round26-v1",
        "session_id": session,
        "git_commit": head,
        "physics_producer": args.producer,
        "producer_model_diff_zero": True,
        "bars": {"pointwise": POINTWISE},
        "runtime": {"rDt_e_seconds": rdt, "icycle": icycle, "nn_e": nn_e},
        "stream_layout": {
            "new_row26_operands": "full_halo_jpi_jpj_203x56",
            "spg_dump_zu_frc": "interior_A2D(0)_199x52",
            "spg_dump_zv_frc": "interior_A2D(0)_199x52",
            "nemo_declaration": (
                "dynspg_ts.F90:168 REAL(wp), DIMENSION(A2D(0)) :: "
                "zu_frc, zv_frc"),
        },
        "nemo_source_identities": {
            "reciprocal_depth": reciprocal,
            "bottom_stress_commit": bottom,
            "vector_update": update,
        },
        "controls": controls,
        "disposition": disposition,
        "ordered_rows": {
            "1.3_production_replay": "HELD",
            "1.4": "ORDERED_BLOCKED",
            **{str(row): "ORDERED_BLOCKED" for row in range(2, 7)},
            "free_surface_filter": "ORDERED_BLOCKED",
            "momentum_rhs": "ORDERED_BLOCKED",
            "tracer_tail": "ORDERED_BLOCKED",
        },
        "bindings": {
            "round25_sha256": ROUND25_SHA,
            "bracket_sha256": args.bracket_sha,
            "on_manifest_sha256": manifest_sha256(on_manifest),
            "off_manifest_sha256": manifest_sha256(off_manifest),
            "new_stream_sha256": {name: sha256(on / name) for name in NEW_STREAMS},
            "source_sha256": sha256(on / ".dynspg_source_sha256"),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    print("production_replay=HELD")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
