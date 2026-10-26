#!/usr/bin/env python3
"""Score preregistered row-1.3 arithmetic-association arms from existing dumps."""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
from legoesm.ocean.experiments.dino import dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo

import spg_substep_chain as inherited
import split_explicit_momentum_chain_round1 as base
import split_explicit_momentum_chain_round9 as r9
from zdf_stream_bracket import sha256


ROUND8_SHA = "16883e8e140f3e25de24866d9deedf9dd6b18189ca4f280c1448144620b48a92"
ROUND10_RAW_SHA = "d2f2b8da84dffb6948f51a08f03327831ae3d5db9e5e812631c9f48e89b8822a"
ROUND10_SHA = "b15e942a7e69b6e352207146b6da06ffe25a8ee3e67395b4fb8ce86ff6e1e845"
POINTWISE = 1.0e-15
ACCUMULATING = 1.0e-12


def _bound(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return json.loads(path.read_text())


def _bits(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a)[mask], dtype=np.float64).view(np.uint64)


def _classes(candidates: dict[str, np.ndarray], mask: np.ndarray) -> list[dict[str, Any]]:
    classes: list[dict[str, Any]] = []
    for name, value in candidates.items():
        bits = _bits(value, mask)
        for group in classes:
            if np.array_equal(bits, group["_bits"]):
                group["members"].append(name)
                break
        else:
            classes.append({"members": [name], "_bits": bits})
    return [{"members": group["members"]} for group in classes]


def _owned(records: dict[str, dict[str, Any]], classes: list[dict[str, Any]], literal: str) -> str:
    literal_group = next(group for group in classes if literal in group["members"])
    at_bar_groups = [
        group for group in classes
        if records[group["members"][0]]["gate_status"] == "AT BAR"
    ]
    if records[literal_group["members"][0]]["gate_status"] != "AT BAR":
        return "OPEN_UNRESOLVED"
    if len(at_bar_groups) != 1:
        return "AMBIGUOUS"
    return "LITERAL_OWNED"


def _records(subrow: str, field: str, candidates: dict[str, np.ndarray],
             oracle: np.ndarray, mask: np.ndarray, n: int, bar: float) -> dict[str, dict[str, Any]]:
    klass = "POINTWISE" if bar == POINTWISE else "ACCUMULATING"
    return {
        name: r9._raw_record(subrow, field, value, oracle, mask, n, klass, bar)
        for name, value in candidates.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round8-artifact", type=Path, required=True)
    parser.add_argument("--round10-raw", type=Path, required=True)
    parser.add_argument("--round10-artifact", type=Path, required=True)
    parser.add_argument("--package-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + JAX fp64 are required")
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID is required")
    root = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != args.package_commit:
        raise SystemExit(f"checkout {commit} != package commit {args.package_commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True):
        raise SystemExit("clean worktree required")

    round8 = _bound(args.round8_artifact, ROUND8_SHA, "round-8 artifact")
    raw = _bound(args.round10_raw, ROUND10_RAW_SHA, "round-10 fixed rescore")
    round10 = _bound(args.round10_artifact, ROUND10_SHA, "round-10 adjudication")
    if raw["session_id"] != session or round10["session_id"] != session:
        raise SystemExit("artifact/current session mismatch")
    if raw["git"]["commit"] != "7b10cdcd08ea7031fede782b5bc104c0ed039454":
        raise SystemExit("unexpected fixed-tree scorer commit")
    if round10["disposition"] != "VFACE_WIDTH_DOMINANT_FLUX_OWNER_CONFIRMED":
        raise SystemExit("round-10 V-face owner is not confirmed")
    if round10["post_first_diverged_subrow"] != "9.4":
        raise SystemExit("round-10 ordered stop changed")
    if not all(raw["controls"].values()) or not all(round10["controls"].values()):
        raise SystemExit("inherited control failed")

    run = args.run.resolve()
    if sha256(run / "nemo") != raw["source_bindings"]["binary_sha256"]:
        raise SystemExit("NEMO binary binding changed")
    dump_hashes = dict(raw["dump_sha256"])
    for name, expected in dump_hashes.items():
        if sha256(run / name) != expected:
            raise SystemExit(f"{name}: SHA changed")
    translated = raw["deterministic_writer_translation"]
    for name in ("cor2d_dump_ua_e_in_substep1.bin", "cor2d_dump_va_e_in_substep1.bin"):
        if sha256(run / name) != translated[name]["deterministic_full_file_sha256"]:
            raise SystemExit(f"{name}: translated dump changed")
    for name in ("mesh_mask.nc", inherited.RESTART_FILE):
        if sha256(run / name) != round8["input_sha256"][name]:
            raise SystemExit(f"{name}: frozen input changed")

    jpi, jpj, _jpk, hls, _icycle, _nn_e = inherited._read_dims(str(run))
    if (jpi, jpj, hls) != (56, 203, 2):
        raise SystemExit(f"unexpected dimensions {(jpi, jpj, hls)}")
    load = lambda name: inherited._load_full(str(run / name), jpi, jpj, hls)
    ua, va = load("cor2d_dump_ua_e_in_substep1.bin"), load("cor2d_dump_va_e_in_substep1.bin")
    hu, hv = load("qco_dump_zhup2_substep1.bin"), load("qco_dump_zhvp2_substep1.bin")
    zh_u, zh_v = load("qco_dump_zhU_substep1.bin"), load("qco_dump_zhV_substep1.bin")
    zhdiv = load("qco_dump_zhdiv_substep1.bin")

    g = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(str(run / inherited.RESTART_FILE), nn_hls=0)
    cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                              lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    bridge_kw: dict[str, Any] = {"periodic_i": True, "full_step": True, "omega": cfg.omega}
    if "carry_native_lat_deg" in inspect.signature(bridge_nemo_to_legoesm_topo).parameters:
        bridge_kw["carry_native_lat_deg"] = (
            getattr(cfg, "tke_htau_evaluation", None) == "nemo_literal"
            or getattr(cfg, "gm_treguier_final_evaluation", None) == "nemo_literal")
    br = bridge_nemo_to_legoesm_topo(g, s, **bridge_kw)
    tmask = np.asarray(g.tmask)[..., 0] > 0.5
    umask = np.asarray(g.umask)[..., 0] > 0.5
    vmask = np.asarray(g.vmask)[..., 0] > 0.5
    if (int(tmask.sum()), int(umask.sum()), int(vmask.sum())) != (9920, 9758, 9868):
        raise SystemExit("registered populations changed")

    e2u = np.broadcast_to((np.asarray(br.geometry.dy) * 0.5)[:, None], zh_u.shape)
    e1v = np.asarray(br.geometry.dx_v)[1:, :]
    u_candidates = {
        "nemo_literal_left": np.asarray((jnp.asarray(e2u) * jnp.asarray(ua)) * jnp.asarray(hu)),
        "metric_depth_first": np.asarray((jnp.asarray(e2u) * jnp.asarray(hu)) * jnp.asarray(ua)),
        "production_current": np.asarray((jnp.asarray(hu) * jnp.asarray(ua)) * jnp.asarray(e2u)),
    }
    u_records = _records("9.4", "zhU", u_candidates, zh_u, umask, 9758, POINTWISE)
    u_classes = _classes(u_candidates, umask)
    u_verdict = _owned(u_records, u_classes, "nemo_literal_left")

    v_candidates: dict[str, np.ndarray] = {}
    v_records: dict[str, dict[str, Any]] = {}
    v_classes: list[dict[str, Any]] = []
    v_verdict = "ORDERED_BLOCKED"
    if u_verdict == "LITERAL_OWNED":
        v_candidates = {
            "nemo_literal_left": np.asarray((jnp.asarray(e1v) * jnp.asarray(va)) * jnp.asarray(hv)),
            "metric_depth_first": np.asarray((jnp.asarray(e1v) * jnp.asarray(hv)) * jnp.asarray(va)),
            "production_current": np.asarray((jnp.asarray(hv) * jnp.asarray(va)) * jnp.asarray(e1v)),
        }
        v_records = _records("9.5", "zhV", v_candidates, zh_v, vmask, 9868, POINTWISE)
        v_classes = _classes(v_candidates, vmask)
        v_verdict = _owned(v_records, v_classes, "nemo_literal_left")

    d_candidates: dict[str, np.ndarray] = {}
    d_records: dict[str, dict[str, Any]] = {}
    d_classes: list[dict[str, Any]] = []
    d_verdict = "ORDERED_BLOCKED"
    if v_verdict == "LITERAL_OWNED":
        ju, jv = jnp.asarray(zh_u), jnp.asarray(zh_v)
        du = ju - jnp.roll(ju, 1, axis=1)
        dv = jv - jnp.concatenate([jnp.zeros_like(jv[:1]), jv[:-1]], axis=0)
        area = jnp.asarray(br.geometry.area)
        r1 = 1.0 / area
        f_u = np.divide(zh_u, e2u, out=np.zeros_like(zh_u), where=e2u != 0.0)
        f_v = np.divide(zh_v, e1v, out=np.zeros_like(zh_v), where=e1v != 0.0)
        generic = divergence_cgrid(jnp.asarray(r9._u_face(f_u)), jnp.asarray(r9._v_face(f_v)),
                                   br.geometry,
                                   u_mask=jnp.asarray(r9._u_face(umask.astype(np.float64))),
                                   v_mask=jnp.asarray(r9._v_face(vmask.astype(np.float64))))
        d_candidates = {
            "nemo_literal_left": np.asarray((du + dv) * r1),
            "divide_after_sum": np.asarray((du + dv) / area),
            "normalize_axes_separately": np.asarray(du * r1 + dv * r1),
            "production_generic": np.asarray(generic),
        }
        d_records = _records("9.6", "zhdiv", d_candidates, zhdiv, tmask, 9920, ACCUMULATING)
        d_classes = _classes(d_candidates, tmask)
        d_verdict = _owned(d_records, d_classes, "nemo_literal_left")

    identity = r9._raw_record("control", "identity", zh_u, zh_u, umask, 9758,
                              "POINTWISE", POINTWISE)
    planted = np.array(zh_u, copy=True)
    target = tuple(np.argwhere(umask)[np.argmax(np.abs(zh_u[umask]))])
    planted[target] += 1.0e-6 * float(np.sqrt(np.mean(zh_u[umask] ** 2)))
    plant = r9._raw_record("control", "plant", planted, zh_u, umask, 9758,
                           "POINTWISE", POINTWISE)
    baseline = {row["subrow"]: row for row in raw["measurements"]}
    controls = {
        "identity_at_bar": identity["gate_status"] == "AT BAR",
        "rms_plant_debt": plant["gate_status"] == "DEBT",
        "fixed_baseline_9.4_reproduced": base._metric(
            u_candidates["production_current"], zh_u, umask, 9758)
            ["per_element_max_error_over_nemo_rms"]
            == baseline["9.4"]["per_element_max_error_over_nemo_rms"],
        "literal_classes_recorded": bool(u_classes),
    }
    if not all(controls.values()):
        raise SystemExit(f"control failed: {controls}")

    if u_verdict != "LITERAL_OWNED":
        disposition = "STOP_9.4_" + u_verdict
    elif v_verdict != "LITERAL_OWNED":
        disposition = "STOP_9.5_" + v_verdict
    elif d_verdict != "LITERAL_OWNED":
        disposition = "STOP_9.6_" + d_verdict
    else:
        disposition = "ALL_ASSOCIATIONS_LITERAL_OWNED_FIX_AUTHORIZED"
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round11-v1",
        "session_id": session,
        "git": {"commit": commit, "clean_before": True, "clean_after": True},
        "backend": jax.default_backend(), "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                    "numpy": np.__version__},
        "bindings": {"round8_sha256": ROUND8_SHA, "round10_raw_sha256": ROUND10_RAW_SHA,
                     "round10_sha256": ROUND10_SHA, "run": str(run),
                     "dump_sha256": dump_hashes},
        "rows": {
            "9.4": {"verdict": u_verdict, "records": u_records, "bit_classes": u_classes},
            "9.5": {"verdict": v_verdict, "records": v_records, "bit_classes": v_classes},
            "9.6": {"verdict": d_verdict, "records": d_records, "bit_classes": d_classes},
        },
        "controls": controls,
        "disposition": disposition,
        "later_rows": {"1.4": "ORDERED_BLOCKED", **{str(i): "ORDERED_BLOCKED" for i in range(2, 7)}},
    }
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True):
        raise SystemExit("worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND11 disposition={disposition}")
    for row in ("9.4", "9.5", "9.6"):
        print(f"ROW {row} verdict={receipt['rows'][row]['verdict']}")
        for name, record in receipt["rows"][row]["records"].items():
            print(f"  {name} gate={record['gate_status']} E={record['normalized_rms_error']:.17g} "
                  f"max={record['per_element_max_error_over_nemo_rms']:.17g}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
