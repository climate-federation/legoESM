#!/usr/bin/env python
"""Hash-bound current-epoch floor for the corrected-T basin re-verdict.

The legoESM side is the registered NEMO-Omega/EEN-on legacy-carry control plus
three 1e-14 temperature-perturbation seeds.  The NEMO side is re-derived from
the unchanged four RUN_VERDICT360 members used by basin_seasonal_decomp.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_DIR.parent))
import acceptance_gate_90d as G  # noqa: E402, N812
import basin_seasonal_decomp as B  # noqa: E402, N812
import kamm_twin_90d as K  # noqa: E402, N812
import tcarry_basin_reverdict as R  # noqa: E402, N812

DAY = 90
CONTROL_PRODUCER = "9e339ad1b2032bc47ec132fb2ad6f00ea071bbb9"
SEED_PRODUCER = "820e3500bf3317c1cc19ce61484497391e4b1f6b"
CONTROL_ARTIFACT_SHA256 = (
    "862debedf76b5e4ff686655f89ee9d33eec56c87a8a05c3f7cffff1502dc53d6")
CONTROL_LOG_SHA256 = (
    "4081b4950d3e551937ee492541573f48b0fcc19ee8c4d3eee90f0e45254beaf4")
SEED_ARTIFACT_SHA256 = {
    1: "67f04b2e04a7e7d13f4e9e5cfa4f4ab128dcdf17ccd0eed27bb554d2efd694ee",
    2: "7420d121f8a3f0e631f563ac869daff9d242e1a8ce14de62e4d35e41f182e595",
    3: "c4688539ee33042a1dc41fbff9605fd43bd30b0a880e8327d6ff1e010d046fa9",
}
SEED_LOG_SHA256 = {
    1: "11b96452254544fb9c882590526ec133a93121c26ba3ec62c70852fb5abf6138",
    2: "f628c4b44cbaed686df1ccab2f619bcee359fd56a70d4318ace1e30d1b887982",
    3: "47838be749fb9e572277c77307a9b6327386b3c25d11d73d9b3f142e41ce884c",
}
OLD_RECEIPT_SHA256 = (
    "63d4e60dd68281bc6101a35f86cb3f4406cb2ddbc27848b74473876226e85849")
OLD_FLOOR90 = 0.00014499002386242506


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: str | Path, expected: str) -> None:
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"STOP input hash {path}: {got} != {expected}")
    print(f"[PROVENANCE] input={Path(path).resolve()} sha256={got}")


def _spread(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise SystemExit(f"STOP floor requires four finite members, got {array}")
    return float(np.std(array, ddof=1))


def _rss(lego_std: float, nemo_std: float) -> float:
    values = np.asarray((lego_std, nemo_std), dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise SystemExit("STOP invalid floor-side spread")
    return float(np.hypot(lego_std, nemo_std))


def _require_log(path: str | Path, expected_hash: str, *, seed: int | None) -> None:
    _require_hash(path, expected_hash)
    text = Path(path).read_text(errors="replace")
    producer = CONTROL_PRODUCER if seed is None else SEED_PRODUCER
    required = (
        f"PROVENANCE: HEAD={producer} dirty_tracked_files=0",
        "vertical ladder: LEGOESM_NEMO_E3T=both",
        "twin start: bridged",
        "ARM: bridge_omega=nemo",
        "resolved EEN metric weighting=nemo",
        "BEFORE-STRESS RECEIPT: stagger=U_AS_T_LEGACY",
        "PRECISION: materialized state dtype = float64",
        "seasonal clock: t_seconds = 15552000s",
        "DONE nsteps=2880 STABLE=True",
    )
    if seed is not None:
        required += (f"PERTURB seed={seed} eps=1.000e-14",)
    missing = [stamp for stamp in required if stamp not in text]
    if missing:
        raise SystemExit(f"STOP floor member log missing receipts: {missing}")
    if seed is None and "PERTURB seed=" in text:
        raise SystemExit("STOP floor control log contains a perturbation")
    print(f"[CONTROL PASS] {'control' if seed is None else f'seed {seed}'} log")


def _artifact_errors(control, members: dict[int, object]) -> list[str]:
    errors: list[str] = []
    common_keys = (
        "control_dtype", "nemo_ladder_mode", "seasonal_t0_reference_seconds",
        "seasonal_t0_seconds", "surface_stress_implicit", "twin_start_mode",
        "vertical_ladder_sha256", "rn_Uv", "producer_dirty_tracked_files",
        "bridge_before_stress_stagger", "bridge_omega_mode",
        "bridge_omega_rad_s", "config_omega_rad_s", "bridge_f_T_sha256",
        "bridge_f_u_sha256", "bridge_f_v_sha256", "een_metric_weighting",
    )
    expected_control = {
        "producer_git_sha": CONTROL_PRODUCER,
        "producer_dirty_tracked_files": 0,
        "stable": True,
        "bridge_omega_mode": "nemo",
        "een_metric_weighting": "nemo",
        "bridge_before_stress_stagger": "U_AS_T_LEGACY",
    }
    for key, want in expected_control.items():
        try:
            got = R._scalar(control, key)
        except SystemExit as exc:
            errors.append(str(exc))
            continue
        if got != want:
            errors.append(f"control.{key}: got {got!r}, expected {want!r}")
    try:
        control_config = json.loads(str(R._scalar(control, "run_config")))
    except (TypeError, ValueError, SystemExit) as exc:
        errors.append(f"control run_config unreadable: {exc}")
        control_config = {}
    if control_config.get("perturb_seed") is not None:
        errors.append("control perturb_seed is not null")

    for seed, member in members.items():
        expected = {
            "producer_git_sha": SEED_PRODUCER,
            "producer_dirty_tracked_files": 0,
            "stable": True,
            "bridge_omega_mode": "nemo",
            "een_metric_weighting": "nemo",
            "bridge_before_stress_stagger": "U_AS_T_LEGACY",
        }
        for key, want in expected.items():
            try:
                got = R._scalar(member, key)
            except SystemExit as exc:
                errors.append(str(exc))
                continue
            if got != want:
                errors.append(f"seed{seed}.{key}: got {got!r}, expected {want!r}")
        for key in common_keys:
            try:
                left, right = R._scalar(control, key), R._scalar(member, key)
            except SystemExit as exc:
                errors.append(str(exc))
                continue
            if left != right:
                errors.append(f"seed{seed} off-axis stamp {key} differs")
        try:
            config = json.loads(str(R._scalar(member, "run_config")))
            if config.get("perturb_seed") != seed:
                errors.append(f"seed{seed} run_config perturb_seed mismatch")
            normalized = dict(config)
            normalized["perturb_seed"] = None
            if normalized != control_config:
                errors.append(f"seed{seed} ordinary run_config leaves differ")
        except (TypeError, ValueError, SystemExit) as exc:
            errors.append(f"seed{seed} run_config unreadable: {exc}")
        ok, reasons, _ladder, _dtype = K.certifiable_grid_and_precision(member)
        if not ok:
            errors.extend(f"seed{seed} off-claim: {reason}" for reason in reasons)
    return errors


def _check_artifacts(control, members: dict[int, object]) -> None:
    errors = _artifact_errors(control, members)
    if errors:
        raise SystemExit("STOP floor receipt control failed:\n  " + "\n  ".join(errors))
    planted = R._Swap(members[1], {"producer_git_sha": CONTROL_PRODUCER})
    planted_errors = _artifact_errors(control, {**members, 1: planted})
    if not any("seed1.producer_git_sha" in error for error in planted_errors):
        raise SystemExit("STOP planted seed producer violation did not fire")
    print("[CONTROL PASS] planted seed producer violation rejected")
    planted_config = json.loads(str(R._scalar(members[2], "run_config")))
    planted_config["perturb_seed"] = 3
    planted = R._Swap(members[2], {"run_config": json.dumps(planted_config)})
    planted_errors = _artifact_errors(control, {**members, 2: planted})
    if not any("seed2 run_config perturb_seed" in error for error in planted_errors):
        raise SystemExit("STOP planted perturb-seed violation did not fire")
    print("[CONTROL PASS] planted perturb-seed violation rejected")


def _both_reductions(state: dict[str, np.ndarray]) -> float:
    row_value, _rows = R._reduce(state)
    driver_value = float(R.D.rowset(state, R.A.tmask, with_density=False)["g_south"])
    if abs(row_value - driver_value) > 1.0e-12:
        raise SystemExit("STOP independent basin reducers disagree")
    return row_value


def _self_test() -> int:
    values = [1.0, 2.0, 4.0, 8.0]
    expected = float(np.std(values, ddof=1))
    if _spread(values) != expected:
        raise SystemExit("STOP ddof=1 spread self-test failed")
    if _rss(3.0, 4.0) != 5.0:
        raise SystemExit("STOP RSS self-test failed")
    for bad in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, float("nan")]):
        try:
            _spread(bad)
        except SystemExit:
            pass
        else:
            raise SystemExit("STOP invalid-member plant did not fire")

    common_config = {
        "recipe": "nemo_dino_kamm_mlf", "n_days": DAY,
        "bridge_before": True, "bridge_before_stress_tpoint": False,
        "bridge_omega": "nemo", "bridge_tke": False, "daily_acc": False,
        "perturb_baro": None, "perturb_baro_key": "dU_avg",
        "perturb_baro_scale": 1.0, "perturb_baro_sha256": None,
        "perturb_eps": 1e-14, "perturb_seed": None,
        "save_step_eta": False, "surface_stress_implicit": False,
        "surface_tendency_placement": None, "u_m": None,
        "use_gm_redi": None, "vmix_scheme": None,
    }
    common = {
        "control_dtype": "float64", "nemo_ladder_mode": "both",
        "seasonal_t0_reference_seconds": 15552000.0,
        "seasonal_t0_seconds": 15552000.0, "surface_stress_implicit": False,
        "twin_start_mode": "bridged", "vertical_ladder_sha256": "a" * 64,
        "rn_Uv": 0.27, "producer_dirty_tracked_files": 0,
        "stable": True, "bridge_before_stress_stagger": "U_AS_T_LEGACY",
        "bridge_omega_mode": "nemo",
        "bridge_omega_rad_s": K.NEMO_CONSTANTS_CONFIG.Omega,
        "config_omega_rad_s": K.NEMO_CONSTANTS_CONFIG.Omega,
        "bridge_f_T_sha256": "b" * 64, "bridge_f_u_sha256": "c" * 64,
        "bridge_f_v_sha256": "d" * 64, "een_metric_weighting": "nemo",
    }
    with tempfile.TemporaryDirectory(prefix="tcarry-floor-selftest-") as tmp:
        paths = [Path(tmp) / f"m{i}.npz" for i in range(4)]
        np.savez(paths[0], **common, producer_git_sha=CONTROL_PRODUCER,
                 run_config=json.dumps(common_config, sort_keys=True))
        for seed in (1, 2, 3):
            config = dict(common_config, perturb_seed=seed)
            np.savez(paths[seed], **common, producer_git_sha=SEED_PRODUCER,
                     run_config=json.dumps(config, sort_keys=True))
        with (np.load(paths[0]) as control, np.load(paths[1]) as one,
              np.load(paths[2]) as two, np.load(paths[3]) as three):
            _check_artifacts(control, {1: one, 2: two, 3: three})
    print("SELF-TEST PASS: ddof=1/RSS and real-NPZ receipt plants fired")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", default=(
        "results/dino_1455/tcarry_omega90_nemo.npz"))
    parser.add_argument("--control-log", default=(
        "results/dino_1455/tcarry_omega90_nemo.log"))
    for seed in (1, 2, 3):
        parser.add_argument(f"--seed{seed}", default=(
            f"results/dino_1455/tcarry_floor90_seed{seed}.npz"))
        parser.add_argument(f"--seed{seed}-log", default=(
            f"results/dino_1455/tcarry_floor90_seed{seed}.log"))
    parser.add_argument("--old-receipt", default="/tmp/dino_basin_seasonal_decomp.json")
    parser.add_argument("--out", default="/tmp/tcarry_basin_floor90.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    git_sha = subprocess.run(
        ["git", "-C", str(_DIR), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(_DIR), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True).stdout.strip()
    print(f"[PROVENANCE] git_sha={git_sha} dirty_tracked_files="
          f"{len(dirty.splitlines()) if dirty else 0}")
    print(f"[PROVENANCE] scorer={Path(__file__).resolve()} sha256={sha256(__file__)}")
    print("[PROVENANCE] flags=" + json.dumps(vars(args), sort_keys=True))
    if dirty:
        raise SystemExit("STOP floor scorer checkout has dirty tracked files")

    _require_hash(args.control, CONTROL_ARTIFACT_SHA256)
    _require_log(args.control_log, CONTROL_LOG_SHA256, seed=None)
    for seed in (1, 2, 3):
        _require_hash(getattr(args, f"seed{seed}"), SEED_ARTIFACT_SHA256[seed])
        _require_log(getattr(args, f"seed{seed}_log"), SEED_LOG_SHA256[seed], seed=seed)
    _require_hash(args.old_receipt, OLD_RECEIPT_SHA256)
    command = ["git", "-C", str(_DIR), "diff", "--quiet",
               f"{CONTROL_PRODUCER}..{SEED_PRODUCER}", "--", "packages/", "src/",
               "scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py"]
    if subprocess.run(command, check=False).returncode != 0:
        raise SystemExit("STOP model/harness changed between control and seeds")
    print("[CONTROL PASS] control-to-seed model/harness diff is empty")

    seed_paths = {seed: getattr(args, f"seed{seed}") for seed in (1, 2, 3)}
    with (np.load(args.control) as control, np.load(seed_paths[1]) as one,
          np.load(seed_paths[2]) as two, np.load(seed_paths[3]) as three):
        members = {1: one, 2: two, 3: three}
        _check_artifacts(control, members)
        land = np.asarray(control["land_mask"], dtype=np.float64)
        for seed, member in members.items():
            if not R._bit_identical(control["land_mask"], member["land_mask"]):
                raise SystemExit(f"STOP seed{seed} land mask differs")
    if not np.array_equal(land > 0.5, R.A.tmask[:, :, 0]):
        raise SystemExit("STOP artifact land mask differs from NEMO tmask")

    lego_states = [G.load_candidate(args.control, day=DAY)] + [
        G.load_candidate(seed_paths[seed], day=DAY) for seed in (1, 2, 3)]
    nemo_states = [B.nemo_state(member, DAY) for member in range(4)]
    wet = R.A.tmask & (land[:, :, None] > 0.5)
    for label, state in [
            *[(f"lego{member}", state) for member, state in enumerate(lego_states)],
            *[(f"nemo{member}", state) for member, state in enumerate(nemo_states)]]:
        R._check_state_finite(label, state, wet)
    R._reducer_plants(nemo_states[0])

    lego_values = [_both_reductions(state) for state in lego_states]
    nemo_values = [_both_reductions(state) for state in nemo_states]
    if len(set(lego_values)) < 2:
        raise SystemExit("STOP all legoESM floor members are identical")
    if len(set(nemo_values)) < 2:
        raise SystemExit("STOP all NEMO floor members are identical")
    lego_std, nemo_std = _spread(lego_values), _spread(nemo_values)
    floor = _rss(lego_std, nemo_std)

    nemo_metrics = G.metrics(nemo_states[0], wet)
    gates = {}
    for member, state in enumerate(lego_states):
        rows = G.classify(G.metrics(state, wet), nemo_metrics, 5)
        failures = G.print_gate(rows, 5, tag=f"[basin-floor lego{member}] ")
        gates[f"lego{member}"] = {"pass": len(rows) - failures, "fail": failures}
        if failures:
            raise SystemExit(f"STOP floor member {member} acceptance gate failed")

    result = {
        "measurement_label": "CONFIRMED measurement",
        "member_set_lego": ["control", "seed1", "seed2", "seed3"],
        "member_set_nemo": ["RUN_VERDICT360_M0", "RUN_VERDICT360_M1",
                            "RUN_VERDICT360_M2", "RUN_VERDICT360_M3"],
        "lego_absolute_sv": lego_values,
        "nemo_absolute_sv": nemo_values,
        "lego_sample_std_sv": lego_std,
        "nemo_sample_std_sv": nemo_std,
        "F90_current_sv": floor,
        "formula": "hypot(std_lego_ddof1, std_nemo_ddof1)",
        "old_F90_sv": OLD_FLOOR90,
        "acceptance": gates,
        "git_sha": git_sha,
        "scorer_sha256": sha256(__file__),
        "control_artifact_sha256": sha256(args.control),
        "seed_artifact_sha256": {
            str(seed): sha256(seed_paths[seed]) for seed in (1, 2, 3)},
        "old_receipt_sha256": sha256(args.old_receipt),
    }
    print("[CONFIRMED measurement] [TCARRY-BASIN-FLOOR] "
          + json.dumps(result, sort_keys=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[PROVENANCE] wrote={Path(args.out).resolve()} sha256={sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
