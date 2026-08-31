#!/usr/bin/env python3
"""Exact-bit tie-break for the two round-11 AT-BAR divergence arms."""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.experiments.dino import dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo

import spg_substep_chain as inherited
from zdf_stream_bracket import sha256


ROUND11_SHA = "fffe3bfb1daa97ef6d9d85c1bb4eaa7541e6a71eb9f3376b835187a322b07e09"


def _bits(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a)[mask], dtype=np.float64).view(np.uint64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--round11-artifact", type=Path, required=True)
    p.add_argument("--package-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + fp64 required")
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID required")
    root = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != args.package_commit or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True):
        raise SystemExit("clean package commit required")
    if sha256(args.round11_artifact) != ROUND11_SHA:
        raise SystemExit("round-11 SHA mismatch")
    r11 = json.loads(args.round11_artifact.read_text())
    if r11["session_id"] != session or r11["rows"]["9.4"]["verdict"] != "LITERAL_OWNED" \
            or r11["rows"]["9.5"]["verdict"] != "LITERAL_OWNED" \
            or r11["rows"]["9.6"]["verdict"] != "AMBIGUOUS":
        raise SystemExit("unexpected round-11 state")
    passing = {name for name, rec in r11["rows"]["9.6"]["records"].items()
               if rec["gate_status"] == "AT BAR"}
    if passing != {"nemo_literal_left", "divide_after_sum"}:
        raise SystemExit(f"round-11 AT-BAR candidate set changed: {passing}")

    run = args.run.resolve()
    for name, expected in r11["bindings"]["dump_sha256"].items():
        if sha256(run / name) != expected:
            raise SystemExit(f"{name}: SHA changed")
    jpi, jpj, _jpk, hls, _icycle, _nn_e = inherited._read_dims(str(run))
    load = lambda name: inherited._load_full(str(run / name), jpi, jpj, hls)
    zu, zv, oracle = (load("qco_dump_zhU_substep1.bin"),
                      load("qco_dump_zhV_substep1.bin"),
                      load("qco_dump_zhdiv_substep1.bin"))
    g = read_nemo_mesh_mask(str(run / "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(str(run / inherited.RESTART_FILE), nn_hls=0)
    cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                              lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    kw = {"periodic_i": True, "full_step": True, "omega": cfg.omega}
    if "carry_native_lat_deg" in inspect.signature(bridge_nemo_to_legoesm_topo).parameters:
        kw["carry_native_lat_deg"] = True
    br = bridge_nemo_to_legoesm_topo(g, s, **kw)
    wet = np.asarray(g.tmask)[..., 0] > 0.5
    if int(wet.sum()) != 9920:
        raise SystemExit("wet T population changed")

    ju, jv = jnp.asarray(zu), jnp.asarray(zv)
    du = ju - jnp.roll(ju, 1, axis=1)
    dv = jv - jnp.concatenate([jnp.zeros_like(jv[:1]), jv[:-1]], axis=0)
    area = jnp.asarray(br.geometry.area)
    candidates = {
        "nemo_literal_left": np.asarray((du + dv) * (1.0 / area)),
        "divide_after_sum": np.asarray((du + dv) / area),
    }
    obits = _bits(oracle, wet)
    mismatches = {name: int(np.count_nonzero(_bits(value, wet) != obits))
                  for name, value in candidates.items()}
    planted = np.array(oracle, copy=True)
    ij = tuple(np.argwhere(wet)[0])
    planted.view(np.uint64)[ij] ^= np.uint64(1)
    plant_mismatch = int(np.count_nonzero(_bits(candidates["nemo_literal_left"], wet)
                                          != _bits(planted, wet)))
    zero = [name for name, count in mismatches.items() if count == 0]
    if zero == ["nemo_literal_left"]:
        disposition = "DIVERGENCE_LITERAL_EXACT_OWNED_FIX_AUTHORIZED"
    elif "nemo_literal_left" not in zero:
        disposition = "DIVERGENCE_LITERAL_OPEN_UNRESOLVED"
    else:
        disposition = "DIVERGENCE_EXACT_AMBIGUOUS"
    controls = {"one_bit_oracle_plant_fires": plant_mismatch > 0,
                "round11_two_at_bar_candidates_bound": passing == set(candidates)}
    if not all(controls.values()):
        raise SystemExit(f"control failed: {controls}")
    receipt = {"schema": "dino-split-explicit-momentum-chain-round12-v1",
               "session_id": session, "git_commit": commit,
               "round11_sha256": ROUND11_SHA, "backend": "cpu", "jax_enable_x64": True,
               "wet_population": 9920, "bit_mismatches": mismatches,
               "controls": controls, "disposition": disposition,
               "later_rows": {"1.4": "ORDERED_BLOCKED", **{str(i): "ORDERED_BLOCKED" for i in range(2, 7)}}}
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True):
        raise SystemExit("worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND12 disposition={disposition} mismatches={mismatches}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
