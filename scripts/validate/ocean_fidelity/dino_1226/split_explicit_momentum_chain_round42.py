#!/usr/bin/env python3
"""Offline 2x2 peel of NEMO WZV call1-to-call2 operands."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import netCDF4
import numpy as np

import split_explicit_momentum_chain_round38 as r38


ROUND41_SHA = "e4b8b814746476d4b19cf31d05a65e2ee145632c07822a56591550cf157c410d"
EXPECTED = {
    "wzv_dump_ww_call1.bin": "defad5014cc7210dd52d6373dafd46856471afedb892267cef30232fdd9baeb2",
    "wzv_dump_ww_call2.bin": "895a141385f33775c4df7fc127a4beadf2e8e496f68a46624931cf8eb1487634",
    "mesh_mask.nc": "3285fc4af36854a38b4e6f7985ab0372b95424398750a23b628935da02f72622",
    "DINO_00005760_restart.nc": "0cc00f9945606d1dea52592280e363b45476103de96f5cef471d70b1b881ff3e",
}
JPI, JPJ, JPKM1, HLS = 56, 203, 35, 2


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load3(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != JPI * JPJ * JPKM1:
        raise SystemExit(f"{path}: expected genuine full-halo 3-D stream")
    full = raw.reshape(JPKM1, JPJ, JPI)
    return np.moveaxis(full[:, HLS:-HLS, HLS:-HLS], 0, -1)


def _load2(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    if raw.size != JPI * JPJ:
        raise SystemExit(f"{path}: expected genuine full-halo 2-D stream")
    return raw.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]


def _recur(hdiv, r3aa, r3bb, e3tmm, e3t0, tmask) -> np.ndarray:
    levels = [None] * JPKM1
    carry = np.zeros_like(r3aa)
    r1_dt = np.float64(1.0) / np.float64(5400.0)
    for k in range(JPKM1 - 1, -1, -1):
        bracket = (e3tmm[..., k] * hdiv[..., k]
                   + (r1_dt * e3t0[..., k]) * (r3aa - r3bb))
        carry = carry - bracket * tmask[..., k]
        levels[k] = carry.copy()
    return np.stack(levels + [np.zeros_like(carry)], axis=-1)


def _metric(candidate, oracle, mask):
    return r38._metric(candidate, oracle, mask, "wzv (vertical velocity)")


def _energy(candidate, oracle, mask) -> float:
    delta = np.asarray(candidate)[mask] - np.asarray(oracle)[mask]
    return float(np.sum(delta * delta, dtype=np.float64))


def _plant(oracle, mask) -> bool:
    planted = np.array(oracle, copy=True)
    wet = np.argwhere(mask)
    point = tuple(wet[int(np.argmax(np.abs(oracle[mask])))])
    rms = float(np.sqrt(np.mean(oracle[mask] ** 2)))
    planted[point] += 2.0e-12 * rms
    return _metric(planted, oracle, mask)["gate_status"] != "AT BAR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round41", type=Path, required=True)
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
    if _sha(args.round41.resolve()) != ROUND41_SHA:
        raise SystemExit("official round-41 receipt changed")
    prior = json.loads(args.round41.read_text())
    if (prior.get("session_id") != session
            or prior.get("disposition") != "ROW5_WZV_CALL2_DIVERGED"):
        raise SystemExit("round 41 does not open the call-2 operand peel")

    run = args.run.resolve()
    for name, expected in EXPECTED.items():
        if _sha(run / name) != expected:
            raise SystemExit(f"retained input changed: {name}")
    names = {
        "h0": "sshnxt_dump_hdiv.bin",
        "h1": "seq_dump_hdiv_nnn_kt00005761.bin",
        "q0": "r3c_dump_r3t_kt00005761.bin",
        "q1": "seq_dump_r3t_aaa_kt00005761.bin",
    }
    for name in names.values():
        if not (run / name).is_file():
            raise SystemExit(f"missing existing operand {name}")

    with netCDF4.Dataset(run / "mesh_mask.nc") as ds:
        e3t0 = np.moveaxis(np.asarray(ds["e3t_0"][0], dtype=np.float64), 0, -1)
        tmask = np.moveaxis(np.asarray(ds["tmask"][0]), 0, -1) > 0.5
    if e3t0.shape != (199, 52, 36) or tmask.shape != e3t0.shape:
        raise SystemExit("mesh fields must be cited-interior (199,52,36)")
    with netCDF4.Dataset(run / "DINO_00005760_restart.nc") as ds:
        sshb = np.asarray(ds["sshb"][0], dtype=np.float64)
        sshn = np.asarray(ds["sshn"][0], dtype=np.float64)
    if sshb.shape != (199, 52) or sshn.shape != sshb.shape:
        raise SystemExit("restart SSH cited-interior shape changed")

    h0_depth = np.zeros_like(sshb)
    for k in range(JPKM1):
        h0_depth = h0_depth + e3t0[..., k] * tmask[..., k]
    wet = h0_depth > 0.0
    r1_h0 = np.where(wet, np.float64(1.0) / h0_depth, 0.0)
    r3bb = sshb * r1_h0
    r3mm = sshn * r1_h0
    e3tmm = e3t0[..., :JPKM1] * (
        np.float64(1.0) + r3mm[..., None] * tmask[..., :JPKM1])

    hdiv = {key: _load3(run / names[key]) for key in ("h0", "h1")}
    r3aa = {key: _load2(run / names[key]) for key in ("q0", "q1")}
    call1 = r38._load_ww(run / "wzv_dump_ww_call1.bin")
    call2 = r38._load_ww(run / "wzv_dump_ww_call2.bin")
    mask = tmask
    if int(mask.sum()) != 342_134:
        raise SystemExit("3-D active W population drift")

    arrays = {}
    arms = {}
    for hi in (0, 1):
        for qi in (0, 1):
            name = f"H{hi}Q{qi}"
            arrays[name] = _recur(
                hdiv[f"h{hi}"], r3aa[f"q{qi}"], r3bb,
                e3tmm, e3t0[..., :JPKM1], tmask[..., :JPKM1])
            target = call1 if name == "H0Q0" else call2
            arms[name] = _metric(arrays[name], target, mask)

    base_energy = _energy(arrays["H0Q0"], call2, mask)
    if base_energy <= 0.0:
        raise SystemExit("call1-to-call2 delta unexpectedly vanished")
    residual_energy = {
        name: _energy(value, call2, mask) for name, value in arrays.items()
    }
    removals = {
        name: 1.0 - residual_energy[name] / base_energy
        for name in arrays
    }
    interaction = ((arrays["H1Q1"] - arrays["H1Q0"])
                   - (arrays["H0Q1"] - arrays["H0Q0"]))
    interaction_norm = float(
        np.sqrt(np.mean(interaction[mask] ** 2))
        / np.sqrt(np.mean(call2[mask] ** 2)))
    controls = {
        "call1_reconstruction_at_bar": arms["H0Q0"]["gate_status"] == "AT BAR",
        "call2_reconstruction_at_bar": arms["H1Q1"]["gate_status"] == "AT BAR",
        "identity_at_bar": _metric(call2, call2, mask)["gate_status"] == "AT BAR",
        "sign_plant_fires": _metric(-call2, call2, mask)["gate_status"] != "AT BAR",
        "roll_plant_fires": _metric(
            np.roll(call2, 1, axis=1), call2, mask)["gate_status"] != "AT BAR",
        "two_bar_plant_fires": _plant(call2, mask),
        "factorial_interaction_closes": interaction_norm <= 1.0e-12,
    }
    valid = all(controls.values())
    h_remove = removals["H1Q0"]
    q_remove = removals["H0Q1"]
    if not valid:
        disposition = "INVALID"
    elif h_remove >= 0.95 and q_remove < 0.05:
        disposition = "CALL2_DELTA_LOCALIZED_TO_HDIV_UPDATE"
    elif q_remove >= 0.95 and h_remove < 0.05:
        disposition = "CALL2_DELTA_LOCALIZED_TO_KAA_R3T_UPDATE"
    elif ((0.05 <= h_remove < 0.95 and 0.05 <= q_remove < 0.95)
          or h_remove < 0.0 or q_remove < 0.0):
        disposition = "CALL2_HDIV_X_KAA_COMPOSITION"
    else:
        disposition = "CALL2_OPERANDS_BOUNDED"

    nemo = args.nemo_root.resolve()
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round42-v1",
        "session_id": session,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "row": 5,
        "arms": arms,
        "residual_energy": residual_energy,
        "squared_energy_removal": removals,
        "factorial_interaction_normalized_rms": interaction_norm,
        "call1_to_call2": _metric(call1, call2, mask),
        "controls": controls,
        "bindings": {
            "round41": _sha(args.round41.resolve()),
            **{name: _sha(run / name) for name in EXPECTED},
            **{key: _sha(run / name) for key, name in names.items()},
            "scorer": _sha(Path(__file__).resolve()),
            "preregistration": _sha(
                root / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round42.md"),
            "nemo_sshwzv": _sha(nemo / "cfgs/DINO/MY_SRC/sshwzv.F90"),
            "nemo_stpmlf": _sha(nemo / "cfgs/DINO/MY_SRC/stpmlf.F90"),
            "nemo_dynspg_ts": _sha(nemo / "cfgs/DINO/MY_SRC/dynspg_ts.F90"),
        },
        "disposition": disposition,
        "ordered_next": 5,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={disposition}")
    for name in ("H0Q0", "H1Q0", "H0Q1", "H1Q1"):
        print(name, arms[name]["gate_status"],
              arms[name]["normalized_rms_error"], removals[name])
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
