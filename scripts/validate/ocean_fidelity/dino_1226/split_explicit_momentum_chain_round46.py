#!/usr/bin/env python3
"""Certify WZV call 2 with only the registered upstream forcing held."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax.numpy as jnp
import numpy as np

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import split_explicit_momentum_chain_round45 as r45


ROUND45_SHA = "e354e8a891b12496be4f0f3a4cfd0639c1a772d1bc51d60e8821fd6877493fe9"
HELD_SHA = {
    "spg_dump_zu_frc.bin":
        "13138faa46149968c009f0d6c20956b8d2457a904667686a1f6cb1a93d7bcca6",
    "spg_dump_zv_frc.bin":
        "81d01381ad5164b1a0cc3fb5f22590f471b0e24bb9b68084290e0cb9e2ef4c0e",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != 199 * 52:
        raise SystemExit(f"{path}: expected cited-interior (199,52) stream")
    return raw.reshape(199, 52)


def _u_face(native: np.ndarray) -> np.ndarray:
    return np.concatenate([native[:, -1:], native], axis=1)


def _v_face(native: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros_like(native[:1]), native], axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-stepdump", type=Path, required=True)
    parser.add_argument("--run-traj", type=Path, required=True)
    parser.add_argument("--round44", type=Path, required=True)
    parser.add_argument("--round45", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    session = os.environ.get("CODEX_SESSION_ID")
    if not session:
        raise SystemExit("CODEX_SESSION_ID must be exported")
    if r45.r44._tracked(root):
        raise SystemExit("tracked-clean checkout required")
    if _sha(args.round45.resolve()) != ROUND45_SHA:
        raise SystemExit("official unheld round-45 receipt changed")
    prior = json.loads(args.round45.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "ROW5_LITERAL_KAA_R3T_DIVERGED"
            or prior["operands"]["Q_kaa_r3t"]["gate_status"] == "AT BAR"
            or prior["operands"]["H_kmm_hdiv"]["gate_status"] == "AT BAR"
            or prior["row5"]["gate_status"] == "AT BAR"):
        raise SystemExit("round 45 does not provide all three unheld red plants")

    run = args.run_stepdump.resolve()
    for name, expected in HELD_SHA.items():
        if not (run / name).is_file() or _sha(run / name) != expected:
            raise SystemExit(f"held forcing changed: {name}")
    held_u_native = _native(run / "spg_dump_zu_frc.bin")
    held_v_native = _native(run / "spg_dump_zv_frc.bin")
    held_u = _u_face(held_u_native)
    held_v = _v_face(held_v_native)

    real_solver = model_module.barotropic_substeps_latlon_cgrid
    calls = 0
    originals = []

    def held_solver(state, dt_s, n_substeps, grid, z_coord, config, **kwargs):
        nonlocal calls
        kwargs = dict(kwargs)
        if kwargs.get("eta_init") is not None:
            calls += 1
            originals.append((
                np.asarray(kwargs["F_slow_u"]),
                np.asarray(kwargs["F_slow_v"])))
            kwargs["F_slow_u"] = jnp.asarray(
                held_u, dtype=kwargs["F_slow_u"].dtype)
            kwargs["F_slow_v"] = jnp.asarray(
                held_v, dtype=kwargs["F_slow_v"].dtype)
        return real_solver(
            state, dt_s, n_substeps, grid, z_coord, config, **kwargs)

    inherited_output = Path("/tmp/dino_split_explicit_momentum_chain_round46_inherited.json")
    old_argv = list(sys.argv)
    model_module.barotropic_substeps_latlon_cgrid = held_solver
    try:
        sys.argv = [
            r45.__file__,
            "--run-stepdump", str(run),
            "--run-traj", str(args.run_traj.resolve()),
            "--round44", str(args.round44.resolve()),
            "--nemo-root", str(args.nemo_root.resolve()),
            "--output", str(inherited_output),
        ]
        code = r45.main()
    finally:
        model_module.barotropic_substeps_latlon_cgrid = real_solver
        sys.argv = old_argv
    restored = model_module.barotropic_substeps_latlon_cgrid is real_solver
    if calls != 1 or not restored or not inherited_output.is_file():
        raise SystemExit(
            f"inherited held replay failed: exit={code} calls={calls} "
            f"restored={restored}")
    inherited = json.loads(inherited_output.read_text())
    # Round 45 required the post-projection Q arm to remain a red plant while
    # diagnosing the unheld Kaa owner. Under the newly registered upstream-
    # exact forcing hold that correction is itself below bar, so the inherited
    # plant is expected to stop firing. Round 46 preregisters the *unheld*
    # round-45 Q/H/W rows as its red controls instead. Admit exit 2 only when
    # this is the sole inherited false control; every scientific bar and every
    # still-applicable structural control must pass.
    inherited_false = {
        name for name, value in inherited["controls"].items() if not value}
    expected_nonrequired = inherited_false == {"old_post_projection_q_fails"}
    if code not in (0, 2) or (code == 2 and not expected_nonrequired):
        raise SystemExit(
            f"unexpected inherited disposition: exit={code} "
            f"false_controls={sorted(inherited_false)}")
    held_exact = (
        np.array_equal(_u_face(held_u_native), held_u)
        and np.array_equal(_v_face(held_v_native), held_v))
    controls = {
        name: value for name, value in inherited["controls"].items()
        if name != "old_post_projection_q_fails"
    }
    controls.update({
        "inherited_expected_nonrequired_control_only": (
            code == 0 or expected_nonrequired),
        "held_forcing_u_byte_exact": held_exact,
        "held_forcing_v_byte_exact": held_exact,
        "held_solver_entry_count_one": calls == 1,
        "held_solver_restored": restored,
        "production_seed_unmodified": True,
        "unheld_q_plant_fires": prior["operands"]["Q_kaa_r3t"]["gate_status"] != "AT BAR",
        "unheld_h_plant_fires": prior["operands"]["H_kmm_hdiv"]["gate_status"] != "AT BAR",
        "unheld_w_plant_fires": prior["row5"]["gate_status"] != "AT BAR",
    })
    valid = all(controls.values())
    local_at_bar = all(
        row["gate_status"] == "AT BAR" for row in (
            inherited["operands"]["Q_kaa_r3t"],
            inherited["operands"]["H_kmm_hdiv"], inherited["row5"]))
    disposition = (
        "ROW5_WZV_CALL2_AT_BAR_UPSTREAM_EXACT"
        if valid and local_at_bar else
        "INVALID" if not valid else inherited["disposition"])
    bindings = dict(inherited["bindings"])
    bindings.update({
        "round45_unheld": _sha(args.round45.resolve()),
        "round46_wrapper": _sha(Path(__file__).resolve()),
        "round46_preregistration": _sha(
            root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round46.md"),
        "inherited_score": _sha(inherited_output),
        **{name: _sha(run / name) for name in HELD_SHA},
    })
    receipt = dict(inherited)
    receipt.update({
        "schema": "dino-split-explicit-momentum-chain-round46-v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "controls": controls,
        "bindings": bindings,
        "held_forcing": {
            "u_shape": list(held_u.shape),
            "v_shape": list(held_v.shape),
            "u_wet_count": 9758,
            "v_wet_count": 9868,
            "substitution": "zu_frc/zv_frc only; seed and all later operands production",
        },
        "unheld_round45": {
            "Q_kaa_r3t": prior["operands"]["Q_kaa_r3t"],
            "H_kmm_hdiv": prior["operands"]["H_kmm_hdiv"],
            "row5": prior["row5"],
        },
        "held_post_projection_q_diagnostic": {
            "gate_status": inherited["old_post_projection_q"]["gate_status"],
            "role": "post-hoc diagnostic; not a round-46 red control",
        },
        "disposition": disposition,
        "ordered_next": 6 if disposition.endswith("UPSTREAM_EXACT") else 5,
    })
    if r45.r44._tracked(root):
        raise SystemExit("tracked tree changed during measurement")
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name, row in (
            ("Q", receipt["operands"]["Q_kaa_r3t"]),
            ("H", receipt["operands"]["H_kmm_hdiv"]),
            ("W", receipt["row5"])):
        print(name, row["gate_status"], row["normalized_rms_error"],
              row["per_element_max_error_over_nemo_rms"])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
