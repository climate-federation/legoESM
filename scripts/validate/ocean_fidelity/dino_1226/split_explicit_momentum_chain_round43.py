#!/usr/bin/env python3
"""Certify the coupled production WZV call-2 implementation at day 180."""
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
import netCDF4
import numpy as np

import kamm_twin_90d as twin
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import split_explicit_momentum_chain_round38 as r38
from legoesm.core.precision import PrecisionPolicy, set_policy


ROUND42_SHA = "d9dec7def95b9dbe4389452f988ea2ea05087b4830b4af7dd1b90d3a8a441d29"
EXPECTED = {
    "wzv_dump_ww_call2.bin": "895a141385f33775c4df7fc127a4beadf2e8e496f68a46624931cf8eb1487634",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "DINO_00005760_restart.nc": "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(candidate, oracle, mask):
    return r38._metric(candidate, oracle, mask, "wzv (vertical velocity)")


def _plant(oracle, mask) -> bool:
    planted = np.array(oracle, copy=True)
    wet = np.argwhere(mask)
    point = tuple(wet[int(np.argmax(np.abs(oracle[mask])))])
    rms = float(np.sqrt(np.mean(oracle[mask] ** 2)))
    planted[point] += 2.0e-12 * rms
    return _metric(planted, oracle, mask)["gate_status"] != "AT BAR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round42", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True).strip():
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round42.resolve()) != ROUND42_SHA:
        raise SystemExit("official round-42 receipt changed")
    prior = json.loads(args.round42.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "CALL2_HDIV_X_KAA_COMPOSITION"):
        raise SystemExit("round 42 does not admit the coupled implementation")
    run = args.run_stepdump.resolve()
    for name, expected in EXPECTED.items():
        if _sha(run / name) != expected:
            raise SystemExit(f"retained input changed: {name}")

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise SystemExit("CPU/fp64 required")
    _, _, model_cfg, model, _, sf, state = twin._build_twin_state(
        "nemo_dino_kamm_mlf", str(args.run_traj.resolve()), str(run),
        bridge_before=True, restart_file="DINO_00005760_restart.nc")
    if (model_cfg.zad_qco_evaluation != "nemo_literal"
            or model_cfg.wzv_call2_evaluation != "nemo_literal"):
        raise SystemExit("coupled call-1/call-2 selectors changed")
    state = model._seed_tke_preclosure_carry(state)

    captures = []
    boundary = []
    original_wzv = model_module.nemo_qco_wzv_operands
    model_class = type(model)
    original_reconcile = model_class._apply_after_level_reconcile

    def capture_wzv(*call_args, **call_kwargs):
        result = original_wzv(*call_args, **call_kwargs)
        if call_kwargs.get("eta_after_override") is not None:
            captures.append(result[0])
        return result

    def capture_reconcile(self, naa, *call_args, **call_kwargs):
        boundary.append(naa.w.data)
        return original_reconcile(self, naa, *call_args, **call_kwargs)

    runs = []
    model_module.nemo_qco_wzv_operands = capture_wzv
    model_class._apply_after_level_reconcile = capture_reconcile
    try:
        for _ in range(2):
            captures.clear()
            boundary.clear()
            with jax.disable_jit():
                model._nemo_mlf_step(state, twin.DT, surface_forcing=sf)
            if len(captures) != 1 or len(boundary) != 1:
                raise SystemExit(
                    f"expected one call2/boundary capture, got "
                    f"{len(captures)}/{len(boundary)}")
            raw = np.asarray(captures[0], dtype=np.float64)
            carried = np.asarray(boundary[0], dtype=np.float64)
            if (raw.shape != (199, 52, 37)
                    or carried.shape != (199, 52, 36)
                    or not np.array_equal(
                        carried, 0.5 * (raw[..., :-1] + raw[..., 1:]))):
                raise SystemExit("post-dyn_zdf call-2 carry receipt failed")
            runs.append(raw.copy())
    finally:
        model_module.nemo_qco_wzv_operands = original_wzv
        model_class._apply_after_level_reconcile = original_reconcile
    restored = (model_module.nemo_qco_wzv_operands is original_wzv
                and model_class._apply_after_level_reconcile is original_reconcile)
    duplicate = np.array_equal(runs[0], runs[1])

    oracle = r38._load_ww(run / "wzv_dump_ww_call2.bin")
    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        mask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    if mask.shape != (199, 52, 36) or int(mask.sum()) != 342_134:
        raise SystemExit("cited-interior W mask changed")
    candidate = runs[0][..., :36]
    row = _metric(candidate, oracle, mask)
    controls = {
        "duplicate_byte_exact": duplicate,
        "hooks_restored": restored,
        "identity_at_bar": _metric(oracle, oracle, mask)["gate_status"] == "AT BAR",
        "sign_plant_fires": _metric(-oracle, oracle, mask)["gate_status"] != "AT BAR",
        "roll_plant_fires": _metric(
            np.roll(oracle, 1, axis=1), oracle, mask)["gate_status"] != "AT BAR",
        "two_bar_plant_fires": _plant(oracle, mask),
        "round42_h_only_worsens": (
            prior["squared_energy_removal"]["H1Q0"] < 0.0),
        "round42_q_only_worsens": (
            prior["squared_energy_removal"]["H0Q1"] < 0.0),
    }
    valid = all(controls.values())
    if not valid:
        disposition = "INVALID"
    elif row["gate_status"] == "AT BAR":
        disposition = "ROW5_WZV_CALL2_AT_BAR"
    else:
        disposition = "ROW5_WZV_CALL2_LITERAL_DIVERGED"

    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round43-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 5,
        "row5": row,
        "controls": controls,
        "selectors": {
            "zad_qco_evaluation": model_cfg.zad_qco_evaluation,
            "wzv_call2_evaluation": model_cfg.wzv_call2_evaluation,
        },
        "bindings": {
            "round42": _sha(args.round42.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round43.md"),
            "model": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
            "tendency_kernel": _sha(root / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"),
            "nemo_sshwzv": _sha(nemo / "cfgs/DINO/MY_SRC/sshwzv.F90"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
        },
        "disposition": disposition,
        "ordered_next": 6 if disposition == "ROW5_WZV_CALL2_AT_BAR" else 5,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    print("row5", row["gate_status"], row["normalized_rms_error"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
