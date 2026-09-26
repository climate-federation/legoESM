#!/usr/bin/env python3
"""Offline 2^3 operand factorial for the row-3 ``dom_qco_r3c`` residue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import numpy as np

import split_explicit_momentum_chain_round29 as r29
import spg_substep_chain as inherited
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask


ROUND29_SHA = "3cd09d6bf3f7f5bb9ee51705a1e5e8cbf3671ff19cec6ebf407efd9e883d60d5"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _left_depth(e3: np.ndarray, mask: np.ndarray, nlev: int) -> np.ndarray:
    """Transcribe ``depth=depth+e3(:,:,jk)*mask(:,:,jk)`` through jpkm1."""
    depth = np.zeros(e3.shape[:2], dtype=np.float64)
    for jk in range(nlev):
        depth = depth + e3[..., jk] * mask[..., jk]
    return depth


def _depth_operands(mesh, *, literal: bool, nlev: int) -> dict[str, np.ndarray]:
    one = np.float64(1.0)
    tmask = np.asarray(mesh.tmask, dtype=np.float64)
    umask = np.asarray(mesh.umask, dtype=np.float64)
    vmask = np.asarray(mesh.vmask, dtype=np.float64)
    fmask = np.asarray(mesh.fmask, dtype=np.float64)
    if literal:
        ht = _left_depth(np.asarray(mesh.e3t_0), tmask, nlev)
        hu = _left_depth(np.asarray(mesh.e3u_0), umask, nlev)
        hv = _left_depth(np.asarray(mesh.e3v_0), vmask, nlev)
        f_depth_mask = vmask * np.roll(vmask, -1, axis=1)
        hf = _left_depth(np.asarray(mesh.e3f_0), f_depth_mask, nlev)
    else:
        ht = (np.asarray(mesh.e3t_0) * tmask).sum(axis=-1)
        hu = np.asarray(mesh.hu_0)
        hv = np.asarray(mesh.hv_0)
        hf = (np.asarray(mesh.e3f_0) * fmask).sum(axis=-1)

    def reciprocal(depth: np.ndarray, mask3: np.ndarray) -> np.ndarray:
        wet = mask3.max(axis=-1)
        return wet / (depth + one - wet)

    return {
        "t": reciprocal(ht, tmask),
        "u": reciprocal(hu, umask),
        "v": reciprocal(hv, vmask),
        "f": reciprocal(hf, fmask),
    }


def _qco(
    eta: np.ndarray,
    mesh,
    depths: dict[str, np.ndarray],
    *,
    literal_metric: bool,
) -> dict[str, np.ndarray]:
    half = np.float64(0.5)
    quarter = np.float64(0.25)
    e1e2t = np.asarray(mesh.e1t) * np.asarray(mesh.e2t)
    weighted = e1e2t * eta
    east = np.roll(weighted, -1, axis=1)
    north = np.roll(weighted, -1, axis=0)
    northeast = np.roll(north, -1, axis=1)
    u_sum = half * (weighted + east)
    v_sum = half * (weighted + north)
    f_sum = quarter * ((weighted + east) + (north + northeast))
    if literal_metric:
        # domhgr.F90:144-160 materializes these reciprocals before dom_qco;
        # domqco.F90:165-181 then applies both post factors by multiplication.
        r1_e1e2u = np.float64(1.0) / (
            np.asarray(mesh.e1u) * np.asarray(mesh.e2u))
        r1_e1e2v = np.float64(1.0) / (
            np.asarray(mesh.e1v) * np.asarray(mesh.e2v))
        r1_e1e2f = np.float64(1.0) / (
            np.asarray(mesh.e1f) * np.asarray(mesh.e2f))
        r3u = (u_sum * depths["u"]) * r1_e1e2u
        r3v = (v_sum * depths["v"]) * r1_e1e2v
        r3f = (f_sum * depths["f"]) * r1_e1e2f
    else:
        r3u = (u_sum * depths["u"]) / (
            np.asarray(mesh.e1u) * np.asarray(mesh.e2u))
        r3v = (v_sum * depths["v"]) / (
            np.asarray(mesh.e1v) * np.asarray(mesh.e2v))
        r3f = (f_sum * depths["f"]) / (
            np.asarray(mesh.e1f) * np.asarray(mesh.e2f))
    return {"r3t": eta * depths["t"], "r3u": r3u, "r3v": r3v, "r3f": r3f}


def _effect_table(errors: dict[str, dict[str, float]], field: str) -> dict[str, float]:
    def e(p: int, d: int, m: int) -> float:
        return errors[f"P{p}D{d}M{m}"][field]

    main_p = np.mean([e(0, d, m) - e(1, d, m) for d in (0, 1) for m in (0, 1)])
    main_d = np.mean([e(p, 0, m) - e(p, 1, m) for p in (0, 1) for m in (0, 1)])
    main_m = np.mean([e(p, d, 0) - e(p, d, 1) for p in (0, 1) for d in (0, 1)])
    pd = np.mean([
        (e(0, 0, m) - e(1, 0, m)) - (e(0, 1, m) - e(1, 1, m))
        for m in (0, 1)])
    pm = np.mean([
        (e(0, d, 0) - e(1, d, 0)) - (e(0, d, 1) - e(1, d, 1))
        for d in (0, 1)])
    dm = np.mean([
        (e(p, 0, 0) - e(p, 1, 0)) - (e(p, 0, 1) - e(p, 1, 1))
        for p in (0, 1)])
    three = (
        (e(0, 0, 0) - e(1, 0, 0)) - (e(0, 1, 0) - e(1, 1, 0))
        - (e(0, 0, 1) - e(1, 0, 1)) + (e(0, 1, 1) - e(1, 1, 1)))
    baseline = e(0, 0, 0)
    return {
        "baseline_normalized_rms": baseline,
        "main_P": float(main_p),
        "main_D": float(main_d),
        "main_M": float(main_m),
        "interaction_PD": float(pd),
        "interaction_PM": float(pm),
        "interaction_DM": float(dm),
        "interaction_PDM": float(three),
        "P0D1M1_fractional_removal": float(
            (baseline - e(0, 1, 1)) / baseline),
        "P1D1M1_fractional_removal": float(
            (baseline - e(1, 1, 1)) / baseline),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round29", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    root = Path(__file__).resolve().parents[4]
    if r29._tracked_status(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round29) != ROUND29_SHA:
        raise SystemExit("round-29 artifact changed")
    if (jax.default_backend() != "cpu" or not jax.config.jax_enable_x64
            or os.environ.get("DINO_1226_LANE") != "d180"):
        raise SystemExit("day-180 CPU/fp64 execution required")

    bound = json.loads(args.round29.read_text())
    if bound["disposition"] != "STOPPED_AT_ROW_3_DOM_QCO_R3C":
        raise SystemExit("round-29 stop changed")
    run = Path(inherited.RUN_DIR).resolve()
    jpi, jpj, jpk, hls, _icycle, _nn_e = inherited._read_dims(str(run))
    mesh = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    production_eta = r29._capture_production_pssh(
        Path("/tmp/dino_split_explicit_momentum_chain_round30_capture.json"))
    oracle_eta = inherited._load_full(
        str(run / "spg_dump_pssh_final.bin"), jpi, jpj, hls)
    oracle = {
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
    gate_names = {
        "r3t": "dom_qco_r3c r3t",
        "r3u": "dom_qco_r3c r3u/r3v",
        "r3v": "dom_qco_r3c r3u/r3v",
        "r3f": None,
    }
    depths = {
        d: _depth_operands(mesh, literal=bool(d), nlev=jpk - 1)
        for d in (0, 1)
    }
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    errors: dict[str, dict[str, float]] = {}
    for p in (0, 1):
        eta = (production_eta, oracle_eta)[p]
        for d in (0, 1):
            for m in (0, 1):
                arm = f"P{p}D{d}M{m}"
                values = _qco(eta, mesh, depths[d], literal_metric=bool(m))
                metrics[arm] = {
                    field: r29._score(values[field], oracle[field], masks[field], gate_names[field])
                    for field in ("r3t", "r3u", "r3v", "r3f")
                }
                errors[arm] = {
                    field: row["normalized_rms_error"]
                    for field, row in metrics[arm].items()
                }

    controls = r29._strict_controls(
        _qco(production_eta, mesh, depths[0], literal_metric=False)["r3t"],
        oracle["r3t"], masks["r3t"])
    if not all(controls[name] for name in (
            "identity_at_bar", "four_nextafter_fires", "zero_shift_best")):
        raise SystemExit(f"control failed: {controls}")

    def all_at_bar(arm: str) -> bool:
        return all(row["gate_status"] == "AT BAR" for row in metrics[arm].values())

    p0_literal = all_at_bar("P0D1M1")
    p1_literal = all_at_bar("P1D1M1")
    inherited_owner = all(
        errors["P0D1M1"][field] > 0.0
        and ((errors["P0D1M1"][field] - errors["P1D1M1"][field])
             / errors["P0D1M1"][field]) >= 0.9
        for field in errors["P0D1M1"]
        if metrics["P0D1M1"][field]["gate_status"] != "AT BAR")
    if p0_literal:
        d_only = all_at_bar("P0D1M0")
        m_only = all_at_bar("P0D0M1")
        disposition = (
            "LOCALIZED_TO_ADMISSIBLE_QCO_ASSOCIATION"
            if d_only or m_only else "COMPOSED")
    elif p1_literal and inherited_owner:
        disposition = "INHERITED_FROM_ROW_1_4"
    else:
        disposition = "OPEN_UNRESOLVED"

    paths = {
        "round29": args.round29,
        "mesh": run / "mesh_mask.nc",
        "pssh": run / "spg_dump_pssh_final.bin",
        "r3t": run / "seq_dump_r3t_aaa_kt00005761.bin",
        "r3u": run / "seq_dump_r3u_aaa_kt00005761.bin",
        "r3v": run / "seq_dump_r3v_aaa_kt00005761.bin",
        "r3f": run / "seq_dump_r3f_kt00005761.bin",
        "script": Path(__file__).resolve(),
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round30-v1",
        "session_id": session,
        "git_commit": r29.subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "source": {
            "depths": "domain.F90:139-160",
            "metrics": "domhgr.F90:144-160",
            "qco": "domqco.F90:153-185",
        },
        "shapes": {
            "mesh": list(np.asarray(mesh.tmask).shape),
            "eta": list(production_eta.shape),
            "oracle_fields": {name: list(value.shape) for name, value in oracle.items()},
        },
        "populations": {name: int(mask.sum()) for name, mask in masks.items()},
        "bindings": {name: _sha(path) for name, path in paths.items()},
        "metrics": metrics,
        "effects": {field: _effect_table(errors, field) for field in oracle},
        "controls": controls,
        "disposition": disposition,
    }
    if r29._tracked_status(root):
        raise SystemExit("tracked worktree changed during measurement")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for arm in sorted(metrics):
        print(arm, " ".join(
            f"{field}={metrics[arm][field]['gate_status']}:"
            f"{metrics[arm][field]['normalized_rms_error']:.3e}/"
            f"{metrics[arm][field]['per_element_max_error_over_nemo_rms']:.3e}"
            for field in ("r3t", "r3u", "r3v", "r3f")))
    return 0 if disposition != "OPEN_UNRESOLVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
