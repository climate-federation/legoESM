#!/usr/bin/env python3
"""Existing-dump ordered scorer for post-split rows 2 and 3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import numpy as np

import fidelity_bar_gate as bar_gate
import split_explicit_momentum_chain_round1 as r1
import split_explicit_momentum_chain_round6 as r6
import spg_substep_chain as inherited
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
from legoesm.ocean.experiments.dino import dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from zu_frc_term_walk import _load_full_3d


ROUND28_PRODUCTION_SHA = "1b4e83ec06416711cfc2b866c878cccb2a23988e8953c9b440a9e94c22bc2be9"
ROUND28_LITERAL_SHA = "0e8757a2c10cd99cbb8394f639e15beb84c05b244f4eeb349718785763e9106e"


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


def _score(candidate, oracle, mask, gate_name=None):
    n = int(np.asarray(mask, dtype=bool).sum())
    metric = r1._metric(candidate, oracle, mask, n)
    metric["gate_name"] = gate_name
    metric["class_bar"] = bar_gate.class_bar_for(gate_name)
    metric["gate_status"] = r1._classify_metric(metric, gate_name)
    return metric


def _strict_controls(candidate, oracle, mask) -> dict[str, Any]:
    wet = np.asarray(mask, dtype=bool)
    identity = _score(oracle, oracle, wet)
    plant = np.array(oracle, copy=True)
    wet_points = np.argwhere(wet)
    values = np.abs(np.asarray(oracle)[wet])
    point = tuple(wet_points[int(np.argmax(values))])
    for _ in range(4):
        plant[point] = np.nextafter(plant[point], np.inf)
    planted = _score(plant, oracle, wet)
    alignment = []
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            shifted = np.roll(np.asarray(candidate), (dj, di), axis=(0, 1))
            alignment.append({
                "dj": dj, "di": di,
                "normalized_rms_error": _score(
                    shifted, oracle, wet)["normalized_rms_error"],
            })
    best = min(alignment, key=lambda row: row["normalized_rms_error"])
    return {
        "identity": identity,
        "four_nextafter": planted,
        "four_nextafter_location": [int(index) for index in point],
        "alignment_scan": alignment,
        "identity_at_bar": identity["gate_status"] == "AT BAR",
        "four_nextafter_fires": planted["gate_status"] != "AT BAR",
        "zero_shift_best": (best["dj"], best["di"]) == (0, 0),
    }


def _capture_production_pssh(output: Path) -> np.ndarray:
    """Reuse round 6 and retain its forcing-only production pssh binding."""
    captured: dict[str, np.ndarray] = {}
    real_record = r6._metric_record
    old_argv = list(sys.argv)

    def record(arm, subrow, field, gate_name, expected_n, sample):
        if arm == "forcing_only" and subrow == "1.4" and field == "pssh_final":
            captured["pssh"] = np.asarray(sample[0])
        return real_record(arm, subrow, field, gate_name, expected_n, sample)

    r6._metric_record = record
    sys.argv = [r6.__file__, "--output", str(output)]
    try:
        code = r6.main()
    finally:
        r6._metric_record = real_record
        sys.argv = old_argv
    if code != 0 or "pssh" not in captured:
        raise SystemExit(
            f"round-6 production pssh capture failed: exit={code}, "
            f"captured={sorted(captured)}")
    return captured["pssh"]


def _literal_qco(eta: np.ndarray, mesh) -> dict[str, np.ndarray]:
    """Transcribe domqco.F90:153-185 with its source association."""
    one = np.float64(1.0)
    half = np.float64(0.5)
    quarter = np.float64(0.25)
    tmask = np.asarray(mesh.tmask, dtype=np.float64)
    umask = np.asarray(mesh.umask, dtype=np.float64)
    vmask = np.asarray(mesh.vmask, dtype=np.float64)
    fmask = np.asarray(mesh.fmask, dtype=np.float64)
    ht0 = (np.asarray(mesh.e3t_0) * tmask).sum(axis=-1)
    hf0 = (np.asarray(mesh.e3f_0) * fmask).sum(axis=-1)

    def recip(depth, wet3):
        wet = wet3.max(axis=-1)
        return wet / (depth + one - wet)

    r1_ht0 = recip(ht0, tmask)
    r1_hu0 = recip(np.asarray(mesh.hu_0), umask)
    r1_hv0 = recip(np.asarray(mesh.hv_0), vmask)
    r1_hf0 = recip(hf0, fmask)
    area = np.asarray(mesh.e1t) * np.asarray(mesh.e2t)
    weighted = area * eta
    east = np.roll(weighted, -1, axis=1)
    north = np.roll(weighted, -1, axis=0)
    northeast = np.roll(north, -1, axis=1)
    return {
        "r3t": eta * r1_ht0,
        "r3u": (half * (weighted + east)) * r1_hu0 / (
            np.asarray(mesh.e1u) * np.asarray(mesh.e2u)),
        "r3v": (half * (weighted + north)) * r1_hv0 / (
            np.asarray(mesh.e1v) * np.asarray(mesh.e2v)),
        "r3f": (quarter * ((weighted + east) + (north + northeast)))
        * r1_hf0 / (np.asarray(mesh.e1f) * np.asarray(mesh.e2f)),
    }


def _literal_hdiv(state, mesh) -> np.ndarray:
    """Transcribe divhor.F90:172-181 on native NEMO Nnn arrays."""
    qco = _literal_qco(np.asarray(state.ssh), mesh)
    e3u = (np.asarray(mesh.e3u_0)
           * (1.0 + qco["r3u"][..., None] * np.asarray(mesh.umask))
           * np.asarray(mesh.umask))
    e3v = (np.asarray(mesh.e3v_0)
           * (1.0 + qco["r3v"][..., None] * np.asarray(mesh.vmask))
           * np.asarray(mesh.vmask))
    e3t = (np.asarray(mesh.e3t_0)
           * (1.0 + qco["r3t"][..., None] * np.asarray(mesh.tmask))
           * np.asarray(mesh.tmask))
    flux_u = ((np.asarray(mesh.e2u)[..., None] * e3u)
              * np.asarray(state.u))
    flux_v = ((np.asarray(mesh.e1v)[..., None] * e3v)
              * np.asarray(state.v))
    west_u = np.roll(flux_u, 1, axis=1)
    south_v = np.concatenate((np.zeros_like(flux_v[:1]), flux_v[:-1]), axis=0)
    numerator = ((flux_u - west_u) + (flux_v - south_v))
    safe_e3t = np.where(np.asarray(mesh.tmask) > 0.5, e3t, 1.0)
    return (numerator * (1.0 / (
        np.asarray(mesh.e1t) * np.asarray(mesh.e2t)))[..., None]) / safe_e3t


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round28-production", type=Path, required=True)
    parser.add_argument("--round28-literal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if (os.environ.get("DINO_1226_LANE") != "d180"
            or os.environ.get("LEGOESM_NEMO_E3T") != "both"):
        raise SystemExit("DINO_1226_LANE=d180 and LEGOESM_NEMO_E3T=both required")
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    root = Path(__file__).resolve().parents[4]
    if _tracked_status(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round28_production) != ROUND28_PRODUCTION_SHA:
        raise SystemExit("round-28 production replay changed")
    if _sha(args.round28_literal) != ROUND28_LITERAL_SHA:
        raise SystemExit("round-28 literal receipt changed")
    bound = json.loads(args.round28_production.read_text())
    primary = [row for row in bound["measurements"]
               if row["arm"] == "forcing_only"
               and row["subrow"] in ("1.3", "1.4")]
    if not primary or any(row["gate_status"] != "AT BAR" for row in primary):
        raise SystemExit("bound rows 1.3/1.4 no longer release row 2")

    run = Path(inherited.RUN_DIR).resolve()
    jpi, jpj, jpk, hls, _icycle, _nn_e = inherited._read_dims(str(run))
    mesh = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(str(run / inherited.RESTART_FILE), nn_hls=0)
    cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    bridge = bridge_nemo_to_legoesm_topo(
        mesh, now, periodic_i=True, full_step=True, omega=cfg.omega,
        carry_native_lat_deg=True)

    oracle_hdiv = _load_full_3d(
        str(run / "seq_dump_hdiv_nnn_kt00005761.bin"),
        jpi, jpj, jpk - 1, hls)
    production_hdiv = np.asarray(divergence_cgrid(
        bridge.state.u.data, bridge.state.v.data, bridge.geometry,
        u_mask=bridge.state.u_mask.data, v_mask=bridge.state.v_mask.data))
    # NEMO's div_hor loops jpkm1 active levels; the bridged lego state keeps
    # the jpk dummy bottom level as an exactly dry padded plane.
    nlev_hdiv = oracle_hdiv.shape[-1]
    production_hdiv = production_hdiv[..., :nlev_hdiv]
    hmask = np.asarray(mesh.tmask)[..., :nlev_hdiv] > 0.5
    row2 = _score(
        production_hdiv, oracle_hdiv, hmask, "ssh_nxt / div_hor")
    literal_hdiv = _literal_hdiv(now, mesh)[..., :nlev_hdiv]
    row2_literal = _score(literal_hdiv, oracle_hdiv, hmask)
    row2_controls = _strict_controls(production_hdiv, oracle_hdiv, hmask)
    if not all(row2_controls[name] for name in (
            "identity_at_bar", "four_nextafter_fires", "zero_shift_best")):
        raise SystemExit(f"row-2 control failed: {row2_controls}")
    row2_released = row2["gate_status"] in ("AT BAR", "CEILING")

    row3 = None
    row3_controls = None
    capture_path = Path("/tmp/dino_split_explicit_momentum_chain_round29_capture.json")
    if row2_released:
        pssh = _capture_production_pssh(capture_path)
        qco = _literal_qco(pssh, mesh)
        oracle_qco = {
            "r3t": inherited._load_full(
                str(run / "seq_dump_r3t_aaa_kt00005761.bin"), jpi, jpj, hls),
            "r3u": inherited._load_full(
                str(run / "seq_dump_r3u_aaa_kt00005761.bin"), jpi, jpj, hls),
            "r3v": inherited._load_full(
                str(run / "seq_dump_r3v_aaa_kt00005761.bin"), jpi, jpj, hls),
            "r3f": inherited._load_full(
                str(run / "seq_dump_r3f_kt00005761.bin"), jpi, jpj, hls),
        }
        masks = {
            "r3t": np.asarray(mesh.tmask[..., 0]) > 0.5,
            "r3u": np.asarray(mesh.umask[..., 0]) > 0.5,
            "r3v": np.asarray(mesh.vmask[..., 0]) > 0.5,
            "r3f": np.asarray(mesh.fmask[..., 0]) > 0.5,
        }
        names = {
            "r3t": "dom_qco_r3c r3t",
            "r3u": "dom_qco_r3c r3u/r3v",
            "r3v": "dom_qco_r3c r3u/r3v",
            "r3f": None,
        }
        row3 = {name: _score(qco[name], oracle_qco[name], masks[name], names[name])
                for name in ("r3t", "r3u", "r3v", "r3f")}
        row3_controls = _strict_controls(
            qco["r3t"], oracle_qco["r3t"], masks["r3t"])
        if not all(row3_controls[name] for name in (
                "identity_at_bar", "four_nextafter_fires", "zero_shift_best")):
            raise SystemExit(f"row-3 control failed: {row3_controls}")

    row3_released = bool(
        row3 is not None
        and all(value["gate_status"] == "AT BAR" for value in row3.values()))
    if not row2_released:
        disposition = "STOPPED_AT_ROW_2_DIV_HOR"
    elif not row3_released:
        disposition = "STOPPED_AT_ROW_3_DOM_QCO_R3C"
    else:
        disposition = "ROWS_2_AND_3_RELEASED"

    dumps = [
        "seq_dump_hdiv_nnn_kt00005761.bin",
        "seq_dump_r3t_aaa_kt00005761.bin",
        "seq_dump_r3u_aaa_kt00005761.bin",
        "seq_dump_r3v_aaa_kt00005761.bin",
        "seq_dump_r3f_kt00005761.bin",
    ]
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round29-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": "cpu",
        "jax_enable_x64": True,
        "row2": row2,
        "row2_literal_source_association": row2_literal,
        "row2_controls": row2_controls,
        "row3": row3,
        "row3_controls": row3_controls,
        "disposition": disposition,
        "ordered_rows": {
            "1.3": "AT BAR",
            "1.4": "AT BAR",
            "2": row2["gate_status"],
            "3": ("ORDERED_BLOCKED" if row3 is None else
                  "AT BAR" if row3_released else "DEBT"),
            **{str(row): "NEXT" if row == 4 and row3_released
               else "ORDERED_BLOCKED" for row in range(4, 7)},
            "free_surface_filter": "ORDERED_BLOCKED",
            "momentum_rhs": "ORDERED_BLOCKED",
            "tracer_tail": "ORDERED_BLOCKED",
        },
        "nemo_source": {
            "row2": "stpmlf.F90:349-376; divhor.F90:172-181",
            "row3": "stpmlf.F90:378-394; domqco.F90:153-185",
        },
        "bindings": {
            "round28_production_sha256": ROUND28_PRODUCTION_SHA,
            "round28_literal_sha256": ROUND28_LITERAL_SHA,
            "capture_sha256": _sha(capture_path) if capture_path.exists() else None,
            "dump_sha256": {name: _sha(run / name) for name in dumps},
            "mesh_sha256": _sha(run / "mesh_mask.nc"),
            "restart_sha256": _sha(run / inherited.RESTART_FILE),
            "script_sha256": _sha(Path(__file__)),
        },
    }
    if _tracked_status(root):
        raise SystemExit("tracked tree changed during measurement")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"row2={row2['gate_status']} rms={row2['normalized_rms_error']:.17g} "
          f"literal_rms={row2_literal['normalized_rms_error']:.17g}")
    if row3 is not None:
        for name, metric in row3.items():
            print(f"row3_{name}={metric['gate_status']} "
                  f"rms={metric['normalized_rms_error']:.17g} "
                  f"max={metric['per_element_max_error_over_nemo_rms']:.17g}")
    print(f"disposition={disposition}")
    print(f"artifact={args.output} sha256={_sha(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
