#!/usr/bin/env python3
"""Capture production W at the matched post-dyn_zdf/call-2 boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax
import numpy as np

import kamm_twin_90d as twin
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND40_SHA = "723fe0e74724febab71537ef31cde16e32388c9da01c9058ef2c37629067258b"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked(root: Path) -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round40", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if _tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round40.resolve()) != ROUND40_SHA:
        raise SystemExit("official round-40 receipt changed")
    prior = json.loads(args.round40.read_text())
    if prior.get("disposition") != "ROW4_ZAD_AT_BAR":
        raise SystemExit("round 40 does not release row 5")
    output = args.output_dir.resolve()
    if not output.is_dir() or any(output.iterdir()):
        raise SystemExit("output directory must already exist and be empty")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, cfg, model_cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()),
        str(args.run_stepdump.resolve()), bridge_before=True,
        restart_file="DINO_00005760_restart.nc")
    if (model_cfg.outer_integrator != "leapfrog"
            or model_cfg.zad_qco_evaluation != "nemo_literal"):
        raise SystemExit("faithful DINO production selectors changed")
    if any(getattr(state, name) is None for name in
           ("T_before", "S_before", "u_before", "v_before", "eta_before")):
        raise SystemExit("matched MLF BEFORE state is incomplete")
    # step() normally performs this eager carry admission before dispatch.
    # The instrument calls the single-pass verification method directly, so
    # reproduce that dispatcher-owned, idempotent seed explicitly.
    state = model._seed_tke_preclosure_carry(state)

    captured = []
    post_zdf = []
    original = model_module.diagnose_w_from_flux_div
    model_class = type(model)
    original_reconcile = model_class._apply_after_level_reconcile

    def capture_w(*call_args, **call_kwargs):
        value = original(*call_args, **call_kwargs)
        captured.append(value)
        return value

    def capture_reconcile(self, naa, *call_args, **call_kwargs):
        # This is the exact production boundary immediately after the
        # implicit dyn_zdf analogue and before mlf_baro_corr.
        post_zdf.append(naa.w.data)
        return original_reconcile(self, naa, *call_args, **call_kwargs)

    model_module.diagnose_w_from_flux_div = capture_w
    model_class._apply_after_level_reconcile = capture_reconcile
    try:
        with jax.disable_jit():
            model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
    finally:
        model_module.diagnose_w_from_flux_div = original
        model_class._apply_after_level_reconcile = original_reconcile
    restored = (model_module.diagnose_w_from_flux_div is original
                and model_class._apply_after_level_reconcile is original_reconcile)
    if len(post_zdf) != 1:
        raise SystemExit(
            f"expected one post-dyn_zdf boundary capture, got {len(post_zdf)}")
    raw = [np.asarray(value, dtype="<f8") for value in captured]
    if not raw or any(value.shape != (199, 52, 37) for value in raw):
        raise SystemExit(
            "production W is genuinely interior (n_lat,n_lon,nlev+1); "
            f"expected every capture (199,52,37), got {[x.shape for x in raw]}")
    post = np.asarray(post_zdf[0], dtype="<f8")
    matching = [index for index, value in enumerate(raw)
                if np.array_equal(
                    post, 0.5 * (value[..., :-1] + value[..., 1:]))]
    if len(matching) != 1:
        raise SystemExit(
            f"expected one W carried across dyn_zdf, matching indices={matching}")
    selected_index = matching[0]
    ww = raw[selected_index]
    carried_exact = True
    if (post.shape != (199, 52, 36) or not carried_exact
            or not restored or not np.isfinite(ww).all()):
        raise SystemExit("capture hook restoration/finiteness failed")

    stream = output / "wzv_call2_production.bin"
    stream.write_bytes(ww.tobytes(order="C"))
    metadata = {
        "schema": "dino-split-explicit-momentum-chain-round41-capture-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "recipe": "nemo_dino_kamm_mlf",
        "backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "raw_capture_count": len(captured),
        "selected_raw_capture_index": selected_index,
        "post_dyn_zdf_matching_raw_count": len(matching),
        "post_dyn_zdf_boundary_count": len(post_zdf),
        "post_dyn_zdf_w_carried_exact": carried_exact,
        "hook_restored": restored,
        "shape_contract": {
            "kind": "genuine_interior",
            "shape": list(ww.shape),
            "source": "ocean_model_latlon_cgrid.py diagnose_w_from_flux_div; "
                      "(n_lat,n_lon,nlev+1)",
        },
        "files": {
            stream.name: {
                "dtype": "<f8",
                "shape": list(ww.shape),
                "size_bytes": stream.stat().st_size,
                "sha256": _sha(stream),
            }
        },
        "bindings": {
            "round40": _sha(args.round40.resolve()),
            "entry_restart": _sha(
                args.run_stepdump.resolve() / "DINO_00005760_restart.nc"),
            "mesh_mask": _sha(args.run_stepdump.resolve() / "mesh_mask.nc"),
            "capture": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round41.md"),
            "model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
        },
        "config": {
            "outer_integrator": str(model_cfg.outer_integrator),
            "zad_qco_evaluation": str(model_cfg.zad_qco_evaluation),
            "executed_verification_path": "_nemo_mlf_step",
        },
    }
    meta = output / "capture.json"
    meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if _tracked(root):
        raise SystemExit("tracked worktree changed during capture")
    print(f"capture={output} metadata_sha256={_sha(meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
