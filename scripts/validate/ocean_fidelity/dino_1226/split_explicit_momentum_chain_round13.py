#!/usr/bin/env python3
"""Production acceptance score for NEMO-order QCO continuity."""
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
import legoesm.grids.latlon as grid_module
import legoesm.ocean.dynamics.barotropic_latlon_cgrid as barotropic_module
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_literal_continuity_divergence,
    nemo_literal_metric_transports,
    nemo_ssh_avg_face_depth,
)
import legoesm.ocean.experiments.dino as dino_module
from legoesm.ocean.experiments.dino import dino_config_for_recipe
import legoesm.ocean.fidelity.nemo_io as nemo_io_module
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
import legoesm.ocean.fidelity.nemo_state_bridge as bridge_module
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo

import spg_substep_chain as inherited
import split_explicit_momentum_chain_round9 as r9
from zdf_stream_bracket import sha256


R10_RAW_SHA = "d2f2b8da84dffb6948f51a08f03327831ae3d5db9e5e812631c9f48e89b8822a"
R11_SHA = "fffe3bfb1daa97ef6d9d85c1bb4eaa7541e6a71eb9f3376b835187a322b07e09"
R12_SHA = "cc6dd626cdc37950164c9bdbe2cbf3678b3c3b31b5c2f4f53d12d9b030f54a0b"
EXPECTED_INPUT_SHA256 = {
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "DINO_00005760_restart.nc":
        "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
}


def _load_json(path: Path, expected: str, label: str):
    if sha256(path) != expected:
        raise SystemExit(f"{label} SHA mismatch")
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--round10-raw", type=Path, required=True)
    p.add_argument("--round11-artifact", type=Path, required=True)
    p.add_argument("--round12-artifact", type=Path, required=True)
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
    production_paths = {
        "core_grid": Path(grid_module.__file__).resolve(),
        "barotropic": Path(barotropic_module.__file__).resolve(),
        "dino_card": Path(dino_module.__file__).resolve(),
        "nemo_io": Path(nemo_io_module.__file__).resolve(),
        "state_bridge": Path(bridge_module.__file__).resolve(),
        "inherited_probe": Path(inherited.__file__).resolve(),
        "round9_scorer": Path(r9.__file__).resolve(),
    }
    escaped = {name: str(path) for name, path in production_paths.items()
               if not path.is_relative_to(root)}
    if escaped:
        raise SystemExit(f"production imports escaped measured checkout: {escaped}")
    if os.environ.get("DINO_1226_LANE") != "d180" \
            or os.environ.get("LEGOESM_NEMO_E3T") != "both":
        raise SystemExit("DINO_1226_LANE=d180 and LEGOESM_NEMO_E3T=both required")
    raw = _load_json(args.round10_raw, R10_RAW_SHA, "round10 raw")
    r11 = _load_json(args.round11_artifact, R11_SHA, "round11")
    r12 = _load_json(args.round12_artifact, R12_SHA, "round12")
    if not (raw["session_id"] == r11["session_id"] == r12["session_id"] == session):
        raise SystemExit("session mismatch")
    if r11["rows"]["9.4"]["verdict"] != "LITERAL_OWNED" \
            or r11["rows"]["9.5"]["verdict"] != "LITERAL_OWNED" \
            or r12["disposition"] != "DIVERGENCE_LITERAL_EXACT_OWNED_FIX_AUTHORIZED":
        raise SystemExit("association fix is not authorized")

    run = args.run.resolve()
    input_sha256 = {
        "mesh_mask.nc": sha256(run / "mesh_mask.nc"),
        inherited.RESTART_FILE: sha256(run / inherited.RESTART_FILE),
    }
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            f"mesh/restart differ from frozen day-180 inputs: {input_sha256}")
    for name, expected in raw["dump_sha256"].items():
        if sha256(run / name) != expected:
            raise SystemExit(f"{name}: SHA changed")
    trans = raw["deterministic_writer_translation"]
    for name in ("spg_dump_sshn_e_init.bin", "cor2d_dump_ua_e_in_substep1.bin",
                 "cor2d_dump_va_e_in_substep1.bin", "spg_dump_ssh_substep1.bin",
                 "spg_dump_ssh_frc.bin"):
        if sha256(run / name) != trans[name]["deterministic_full_file_sha256"]:
            raise SystemExit(f"{name}: translated SHA changed")
    jpi, jpj, _jpk, hls, _icycle, _nn_e = inherited._read_dims(str(run))
    load = lambda name: inherited._load_full(str(run / name), jpi, jpj, hls)
    eta = load("spg_dump_sshn_e_init.bin")
    ua, va = load("cor2d_dump_ua_e_in_substep1.bin"), load("cor2d_dump_va_e_in_substep1.bin")
    ssh_out, ssh_frc = load("spg_dump_ssh_substep1.bin"), load("spg_dump_ssh_frc.bin")
    nz, nhu, nhv = (load("qco_dump_zsshp2_substep1.bin"),
                     load("qco_dump_zhup2_substep1.bin"),
                     load("qco_dump_zhvp2_substep1.bin"))
    nzu, nzv, ndiv = (load("qco_dump_zhU_substep1.bin"),
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
    tm = np.asarray(g.tmask)[..., 0] > 0.5
    um = np.asarray(g.umask)[..., 0] > 0.5
    vm = np.asarray(g.vmask)[..., 0] > 0.5
    ufm, vfm = r9._u_face(um.astype(np.float64)), r9._v_face(vm.astype(np.float64))
    hu, hv = nemo_ssh_avg_face_depth(jnp.asarray(eta), br.state.H_bathy.data,
                                     jnp.asarray(tm), jnp.asarray(ufm), jnp.asarray(vfm),
                                     br.geometry, br.geometry.area, jnp.float64)
    zu, zv = nemo_literal_metric_transports(hu, hv, jnp.asarray(r9._u_face(ua)),
                                            jnp.asarray(r9._v_face(va)),
                                            jnp.asarray(ufm), jnp.asarray(vfm), br.geometry)
    div = nemo_literal_continuity_divergence(hu, hv, jnp.asarray(r9._u_face(ua)),
                                             jnp.asarray(r9._v_face(va)),
                                             jnp.asarray(ufm), jnp.asarray(vfm), br.geometry)
    hu, hv, zu, zv, div = map(np.asarray, (hu, hv, zu, zv, div))
    dt = float(raw["dt_e_s"])
    rows = [
        r9._raw_record("9.1", "zsshp2_e", eta, nz, tm, 9920, "POINTWISE", 1e-15),
        r9._raw_record("9.2", "zhup2_e", r9._u_nemo(hu), nhu, um, 9758, "POINTWISE", 1e-15),
        r9._raw_record("9.3", "zhvp2_e", r9._v_nemo(hv), nhv, vm, 9868, "POINTWISE", 1e-15),
        r9._raw_record("9.4", "zhU", r9._u_nemo(zu), nzu, um, 9758, "POINTWISE", 1e-15),
        r9._raw_record("9.5", "zhV", r9._v_nemo(zv), nzv, vm, 9868, "POINTWISE", 1e-15),
        r9._raw_record("9.6", "zhdiv", div, ndiv, tm, 9920, "ACCUMULATING", 1e-12),
        r9._raw_record("9.7", "ssha_e", eta - dt * (ssh_frc + div), ssh_out, tm, 9920,
                       "ACCUMULATING", 1e-12),
    ]
    first = next((row["subrow"] for row in rows if row["gate_status"] != "AT BAR"), None)
    planted = np.array(nzu, copy=True)
    ij = tuple(np.argwhere(um)[0])
    planted[ij] += 1e-6 * float(np.sqrt(np.mean(nzu[um] ** 2)))
    controls = {
        "all_inherited_controls": all(raw["controls"].values()),
        "identity_at_bar": r9._raw_record("c", "id", nzu, nzu, um, 9758,
                                           "POINTWISE", 1e-15)["gate_status"] == "AT BAR",
        "plant_debt": r9._raw_record("c", "plant", planted, nzu, um, 9758,
                                      "POINTWISE", 1e-15)["gate_status"] == "DEBT",
    }
    if not all(controls.values()):
        raise SystemExit(f"control failed: {controls}")
    disposition = "ROW_1.3_ALL_OPERANDS_AT_BAR_RELEASE_1.4" if first is None else f"STOP_{first}"
    receipt = {"schema": "dino-split-explicit-momentum-chain-round13-v1",
               "session_id": session, "git_commit": commit, "backend": "cpu",
               "jax_enable_x64": True,
               "runtime": {"python": sys.version, "platform": platform.platform(),
                           "jax": jax.__version__, "jaxlib": jaxlib.__version__,
                           "numpy": np.__version__},
               "bindings": {"round10_raw_sha256": R10_RAW_SHA, "round11_sha256": R11_SHA,
                            "round12_sha256": R12_SHA, "dump_sha256": raw["dump_sha256"],
                            "input_sha256": input_sha256,
                            "production_sha256": {
                                name: sha256(path) for name, path in production_paths.items()
                            }},
               "production_paths": {
                   name: str(path) for name, path in production_paths.items()
               },
               "measurements": rows, "first_diverged_subrow": first,
               "controls": controls, "disposition": disposition,
               "later_rows": {"1.4": "RELEASED" if first is None else "ORDERED_BLOCKED",
                              **{str(i): "ORDERED_BLOCKED" for i in range(2, 7)}}}
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True):
        raise SystemExit("worktree changed during score")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND13 disposition={disposition} first_diverged_subrow={first}")
    for row in rows:
        print(f"ROW {row['subrow']} {row['field']} {row['gate_status']} "
              f"E={row['normalized_rms_error']:.17g} max={row['per_element_max_error_over_nemo_rms']:.17g}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
