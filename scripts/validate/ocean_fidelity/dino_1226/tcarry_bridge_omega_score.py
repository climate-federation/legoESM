#!/usr/bin/env python
"""Fail-closed paired scorer for the bridge-Omega baseline discriminator.

Artifacts remain deliberately unbound until their SHA-256s and clean common
producer are committed below. See ``PREREG_tcarry_bridge_omega_discriminator.md``.
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
import kamm_twin_90d as K  # noqa: E402, N812
import tcarry_basin_reverdict as R  # noqa: E402, N812

HISTORICAL_BASELINE = -0.4257848785815366
CURRENT_BASELINE = -0.43908550999203477
TARGET_DELTA = 0.01330063141049817
TWO_F = 0.0002899800477248501
DAY = 90

# Bound 2026-08-28 in a committed amendment before either NPZ was loaded.
NEMO_ARTIFACT_SHA256 = "862debedf76b5e4ff686655f89ee9d33eec56c87a8a05c3f7cffff1502dc53d6"
LEGACY_ARTIFACT_SHA256 = "dc11a5f831f4dec573b3c04d1b2f3b73ea8e74bfd1a30f3c20029757c5928a94"
NEMO_LOG_SHA256 = "4081b4950d3e551937ee492541573f48b0fcc19ee8c4d3eee90f0e45254beaf4"
LEGACY_LOG_SHA256 = "52045b52fee947dd8fdbd682773726d6e88ae6152f82f9a42446f99949d14b7d"
BOUND_PRODUCER_SHA = "9e339ad1b2032bc47ec132fb2ad6f00ea071bbb9"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(g_new: float, g_old: float) -> str:
    values = np.asarray((g_new, g_old), dtype=np.float64)
    if not np.isfinite(values).all():
        raise SystemExit("STOP non-finite bridge-Omega classifier input")
    delta = g_old - g_new
    if abs(g_new - CURRENT_BASELINE) > TWO_F:
        return "INVALID_STOP_NEW_OMEGA_REPRODUCTION"
    if (abs(g_old - HISTORICAL_BASELINE) <= TWO_F
            and abs(delta - TARGET_DELTA) <= TWO_F):
        return "CONFIRMED_BRIDGE_OMEGA_OWNERSHIP"
    return "REFUTED_BRIDGE_OMEGA_OWNERSHIP"


def _valid_hash(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _require_bound() -> None:
    bindings = {
        "NEMO_ARTIFACT_SHA256": NEMO_ARTIFACT_SHA256,
        "LEGACY_ARTIFACT_SHA256": LEGACY_ARTIFACT_SHA256,
        "NEMO_LOG_SHA256": NEMO_LOG_SHA256,
        "LEGACY_LOG_SHA256": LEGACY_LOG_SHA256,
    }
    bad = [name for name, value in bindings.items() if not _valid_hash(value)]
    if (not isinstance(BOUND_PRODUCER_SHA, str)
            or len(BOUND_PRODUCER_SHA) != 40
            or any(c not in "0123456789abcdef" for c in BOUND_PRODUCER_SHA)):
        bad.append("BOUND_PRODUCER_SHA")
    if bad:
        raise SystemExit(
            "STOP bridge-Omega artifacts are unbound; commit exact artifact/log "
            f"hashes and common producer before scoring: {bad}")


def _require_hash(path: str | Path, expected: str) -> None:
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"STOP input hash {path}: {got} != {expected}")
    print(f"[PROVENANCE] input={Path(path).resolve()} sha256={got}")


def _require_log(path: str | Path, mode: str, expected_hash: str) -> None:
    _require_hash(path, expected_hash)
    text = Path(path).read_text(errors="replace")
    required = (
        f"PROVENANCE: HEAD={BOUND_PRODUCER_SHA} dirty_tracked_files=0",
        f"ARM: bridge_omega={mode}",
        "vertical ladder: LEGOESM_NEMO_E3T=both",
        "twin start: bridged",
        "BEFORE-STRESS RECEIPT: stagger=U_AS_T_LEGACY",
        "resolved EEN metric weighting=nemo",
        "PRECISION: materialized state dtype = float64",
        "seasonal clock: t_seconds = 15552000s",
        "DONE nsteps=2880 STABLE=True",
    )
    missing = [stamp for stamp in required if stamp not in text]
    if missing:
        raise SystemExit(f"STOP {mode} log missing receipts: {missing}")
    if "ARM: een_metric_weighting=" in text:
        raise SystemExit(f"STOP {mode} log contains forbidden EEN override")
    print(f"[CONTROL PASS] {mode} log receipts and default EEN selection")


def _receipt_errors(new, old, expected_producer: str) -> list[str]:
    errors: list[str] = []
    common = (*R.IDENTICAL_STAMPS, "config_omega_rad_s", "een_metric_weighting")
    for key in common:
        try:
            left, right = R._scalar(new, key), R._scalar(old, key)
        except SystemExit as exc:
            errors.append(str(exc))
            continue
        if left != right:
            errors.append(f"common stamp {key} differs: {left!r} != {right!r}")
    expected = {
        "new.mode": (R._scalar(new, "bridge_omega_mode"), "nemo"),
        "old.mode": (R._scalar(old, "bridge_omega_mode"), "legacy-rounded"),
        "new.bridge_omega": (
            float(R._scalar(new, "bridge_omega_rad_s")),
            float(K.NEMO_CONSTANTS_CONFIG.Omega)),
        "old.bridge_omega": (
            float(R._scalar(old, "bridge_omega_rad_s")), float(K.constants.Omega)),
        "new.config_omega": (
            float(R._scalar(new, "config_omega_rad_s")),
            float(K.NEMO_CONSTANTS_CONFIG.Omega)),
        "old.config_omega": (
            float(R._scalar(old, "config_omega_rad_s")),
            float(K.NEMO_CONSTANTS_CONFIG.Omega)),
        "new.producer": (str(R._scalar(new, "producer_git_sha")), expected_producer),
        "old.producer": (str(R._scalar(old, "producer_git_sha")), expected_producer),
        "new.dirty": (int(R._scalar(new, "producer_dirty_tracked_files")), 0),
        "old.dirty": (int(R._scalar(old, "producer_dirty_tracked_files")), 0),
        "new.stable": (bool(R._scalar(new, "stable")), True),
        "old.stable": (bool(R._scalar(old, "stable")), True),
        "new.stress_stagger": (
            str(R._scalar(new, "bridge_before_stress_stagger")), "U_AS_T_LEGACY"),
        "old.stress_stagger": (
            str(R._scalar(old, "bridge_before_stress_stagger")), "U_AS_T_LEGACY"),
        "new.een": (str(R._scalar(new, "een_metric_weighting")), "nemo"),
        "old.een": (str(R._scalar(old, "een_metric_weighting")), "nemo"),
    }
    errors.extend(f"{name}: got {got!r}, expected {want!r}"
                  for name, (got, want) in expected.items() if got != want)

    hashes = {}
    for tag, artifact in (("new", new), ("old", old)):
        for key in ("bridge_f_T_sha256", "bridge_f_u_sha256",
                    "bridge_f_v_sha256"):
            value = str(R._scalar(artifact, key))
            hashes[(tag, key)] = value
            if not _valid_hash(value):
                errors.append(f"{tag}.{key} is not a SHA-256: {value!r}")
        ok, reasons, _ladder, _dtype = K.certifiable_grid_and_precision(artifact)
        if not ok:
            errors.extend(f"{tag} off-claim: {reason}" for reason in reasons)
    for key in ("bridge_f_T_sha256", "bridge_f_u_sha256", "bridge_f_v_sha256"):
        if hashes.get(("new", key)) == hashes.get(("old", key)):
            errors.append(f"registered Coriolis content hash did not change: {key}")

    try:
        new_config = json.loads(str(R._scalar(new, "run_config")))
        old_config = json.loads(str(R._scalar(old, "run_config")))
        differing = {key for key in new_config.keys() | old_config.keys()
                     if new_config.get(key) != old_config.get(key)}
        if differing != {"bridge_omega"}:
            errors.append(f"run_config differences {sorted(differing)!r}, expected only "
                          "bridge_omega")
        if new_config.get("bridge_omega") != "nemo":
            errors.append("new run_config bridge_omega is not nemo")
        if old_config.get("bridge_omega") != "legacy-rounded":
            errors.append("old run_config bridge_omega is not legacy-rounded")
        errors.extend(R._registered_config_errors(new_config, "new", DAY))
        errors.extend(R._registered_config_errors(old_config, "old", DAY))
    except (KeyError, TypeError, ValueError, SystemExit) as exc:
        errors.append(f"run_config unreadable: {exc}")
    return errors


def _check_receipts(new, old, expected_producer: str) -> None:
    errors = _receipt_errors(new, old, expected_producer)
    if errors:
        raise SystemExit("STOP bridge-Omega receipt control failed:\n  "
                         + "\n  ".join(errors))
    plants = (
        ("mode", R._Swap(old, {"bridge_omega_mode": "nemo"}), "old.mode"),
        ("rate", R._Swap(old, {"bridge_omega_rad_s": K.NEMO_CONSTANTS_CONFIG.Omega}),
         "old.bridge_omega"),
        ("config rate", R._Swap(old, {"config_omega_rad_s": K.constants.Omega}),
         "old.config_omega"),
        ("f_T hash", R._Swap(old, {"bridge_f_T_sha256":
                                    R._scalar(new, "bridge_f_T_sha256")}),
         "registered Coriolis content hash"),
    )
    for name, planted, token in plants:
        planted_errors = _receipt_errors(new, planted, expected_producer)
        if not any(token in error for error in planted_errors):
            raise SystemExit(f"STOP planted bridge-Omega {name} violation did not fire")
        print(f"[CONTROL PASS] planted bridge-Omega {name} violation rejected")


def _both_reductions(state: dict[str, np.ndarray]) -> tuple[float, np.ndarray, float]:
    row_value, rows = R._reduce(state)
    driver_value = float(R.D.rowset(state, R.A.tmask, with_density=False)["g_south"])
    difference = abs(row_value - driver_value)
    if difference > 1.0e-12:
        raise SystemExit(f"STOP independent basin reducers differ by {difference:.17g} Sv")
    return row_value, rows, driver_value


def _self_test() -> int:
    if classify(CURRENT_BASELINE, HISTORICAL_BASELINE) != (
            "CONFIRMED_BRIDGE_OMEGA_OWNERSHIP"):
        raise SystemExit("STOP CONFIRM classifier plant did not fire")
    if classify(CURRENT_BASELINE, HISTORICAL_BASELINE + 2.0 * TWO_F) != (
            "REFUTED_BRIDGE_OMEGA_OWNERSHIP"):
        raise SystemExit("STOP REFUTE classifier plant did not fire")
    if classify(CURRENT_BASELINE + 2.0 * TWO_F, HISTORICAL_BASELINE) != (
            "INVALID_STOP_NEW_OMEGA_REPRODUCTION"):
        raise SystemExit("STOP INVALID classifier plant did not fire")
    try:
        classify(float("nan"), HISTORICAL_BASELINE)
    except SystemExit:
        pass
    else:
        raise SystemExit("STOP non-finite classifier plant did not fire")

    producer = "a" * 40
    common_config = {
        "recipe": "nemo_dino_kamm_mlf", "n_days": DAY,
        "bridge_tke": False, "bridge_before": True,
        "bridge_before_stress_tpoint": False, "vmix_scheme": None,
        "use_gm_redi": None, "surface_stress_implicit": False,
        "surface_tendency_placement": None, "save_step_eta": False,
        "perturb_seed": None, "perturb_eps": 1e-14, "perturb_baro": None,
        "perturb_baro_sha256": None, "perturb_baro_key": "dU_avg",
        "perturb_baro_scale": 1.0, "daily_acc": False, "u_m": None,
    }
    common_stamps = {
        "control_dtype": "float64", "nemo_ladder_mode": "both",
        "seasonal_t0_reference_seconds": 15552000.0,
        "seasonal_t0_seconds": 15552000.0, "surface_stress_implicit": False,
        "twin_start_mode": "bridged", "vertical_ladder_sha256": "b" * 64,
        "rn_Uv": 0.27, "producer_git_sha": producer,
        "producer_dirty_tracked_files": 0, "stable": True,
        "bridge_before_stress_stagger": "U_AS_T_LEGACY",
        "config_omega_rad_s": K.NEMO_CONSTANTS_CONFIG.Omega,
        "een_metric_weighting": "nemo",
    }
    with tempfile.TemporaryDirectory(prefix="tcarry-omega-selftest-") as tmp:
        new_path, old_path = Path(tmp) / "new.npz", Path(tmp) / "old.npz"
        np.savez(
            new_path, **common_stamps, bridge_omega_mode="nemo",
            bridge_omega_rad_s=K.NEMO_CONSTANTS_CONFIG.Omega,
            bridge_f_T_sha256="c" * 64, bridge_f_u_sha256="d" * 64,
            bridge_f_v_sha256="e" * 64,
            run_config=json.dumps(dict(common_config, bridge_omega="nemo"),
                                  sort_keys=True))
        np.savez(
            old_path, **common_stamps, bridge_omega_mode="legacy-rounded",
            bridge_omega_rad_s=K.constants.Omega,
            bridge_f_T_sha256="f" * 64, bridge_f_u_sha256="1" * 64,
            bridge_f_v_sha256="2" * 64,
            run_config=json.dumps(
                dict(common_config, bridge_omega="legacy-rounded"), sort_keys=True))
        with np.load(new_path) as new, np.load(old_path) as old:
            _check_receipts(new, old, producer)
    print("SELF-TEST PASS: CONFIRM/REFUTE/INVALID and every real-NPZ receipt "
          "plant fired")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", nargs="?", default=(
        "results/dino_1455/tcarry_omega90_nemo.npz"))
    parser.add_argument("old", nargs="?", default=(
        "results/dino_1455/tcarry_omega90_legacy_rounded.npz"))
    parser.add_argument("--new-log", default=(
        "results/dino_1455/tcarry_omega90_nemo.log"))
    parser.add_argument("--old-log", default=(
        "results/dino_1455/tcarry_omega90_legacy_rounded.log"))
    parser.add_argument("--out", default="/tmp/tcarry_bridge_omega_score.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()

    _require_bound()
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
        raise SystemExit("STOP bridge-Omega scorer checkout has dirty tracked files")

    _require_hash(args.new, NEMO_ARTIFACT_SHA256)
    _require_hash(args.old, LEGACY_ARTIFACT_SHA256)
    _require_log(args.new_log, "nemo", NEMO_LOG_SHA256)
    _require_log(args.old_log, "legacy-rounded", LEGACY_LOG_SHA256)
    with np.load(args.new) as new_npz, np.load(args.old) as old_npz:
        _check_receipts(new_npz, old_npz, BOUND_PRODUCER_SHA)
        R._check_day0(new_npz, old_npz)
        if not R._bit_identical(new_npz["land_mask"], old_npz["land_mask"]):
            raise SystemExit("STOP paired land_mask differs")
        land = np.asarray(new_npz["land_mask"], dtype=np.float64)
    if not np.array_equal(land > 0.5, R.A.tmask[:, :, 0]):
        raise SystemExit("STOP artifact land_mask differs from NEMO surface tmask")

    new = G.load_candidate(args.new, day=DAY)
    old = G.load_candidate(args.old, day=DAY)
    nemo = R._nemo(DAY)
    wet = R.A.tmask & (land[:, :, None] > 0.5)
    for label, state in (("new", new), ("old", old), ("NEMO", nemo)):
        R._check_state_finite(label, state, wet)
    R._reducer_plants(nemo)

    new_abs, new_rows, new_driver = _both_reductions(new)
    old_abs, old_rows, old_driver = _both_reductions(old)
    nemo_abs, nemo_rows, nemo_driver = _both_reductions(nemo)
    g_new, g_old = new_abs - nemo_abs, old_abs - nemo_abs
    delta = g_old - g_new
    verdict = classify(g_new, g_old)

    nemo_metrics = G.metrics(nemo, wet)
    gate = {}
    for label, state in (("new", new), ("old", old)):
        rows = G.classify(G.metrics(state, wet), nemo_metrics, 5)
        failures = G.print_gate(rows, 5, tag=f"[bridge-Omega {label}] ")
        gate[label] = {"pass": len(rows) - failures, "fail": failures}
        if failures:
            raise SystemExit(f"STOP bridge-Omega {label} acceptance gate failed")

    result = {
        "measurement_label": "CONFIRMED measurement",
        "verdict": verdict,
        "GnewOmega90_sv": g_new,
        "GoldOmega90_sv": g_old,
        "Domega90_sv": delta,
        "current_baseline_sv": CURRENT_BASELINE,
        "historical_baseline_sv": HISTORICAL_BASELINE,
        "target_delta_sv": TARGET_DELTA,
        "two_F_sv": TWO_F,
        "new_reproduction_miss_sv": abs(g_new - CURRENT_BASELINE),
        "old_reproduction_miss_sv": abs(g_old - HISTORICAL_BASELINE),
        "delta_miss_sv": abs(delta - TARGET_DELTA),
        "new_driver_absdiff_sv": abs(new_abs - new_driver),
        "old_driver_absdiff_sv": abs(old_abs - old_driver),
        "nemo_driver_absdiff_sv": abs(nemo_abs - nemo_driver),
        "rows_new_sv": (new_rows - nemo_rows).tolist(),
        "rows_old_sv": (old_rows - nemo_rows).tolist(),
        "acceptance": gate,
        "git_sha": git_sha,
        "scorer_sha256": sha256(__file__),
        "new_artifact_sha256": sha256(args.new),
        "old_artifact_sha256": sha256(args.old),
        "producer_git_sha": BOUND_PRODUCER_SHA,
    }
    print("[CONFIRMED measurement] [BRIDGE-OMEGA] "
          + json.dumps(result, sort_keys=True))
    print(f"[CONFIRMED measurement] VERDICT={verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[PROVENANCE] wrote={Path(args.out).resolve()} sha256={sha256(args.out)}")
    if verdict == "INVALID_STOP_NEW_OMEGA_REPRODUCTION":
        raise SystemExit("STOP new-Omega arm did not reproduce the retained baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
