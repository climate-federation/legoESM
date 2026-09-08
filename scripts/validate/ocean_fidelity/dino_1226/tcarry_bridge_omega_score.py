#!/usr/bin/env python
"""Fail-closed four-corner scorer for the EEN x bridge-Omega discriminator.

The original paired Omega result and its three locked corners remain visible;
the bound fourth corner adds the registered interaction/combined-ownership
decision.  See ``PREREG_tcarry_een_omega_interaction.md``.
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
import tcarry_een_off_discriminator as E  # noqa: E402, N812

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

# Fourth corner bound 2026-08-28 in commit 19acf16b2 before its NPZ was opened.
FOURTH_ARTIFACT_SHA256 = (
    "678a6a914367561596a259cc27a50f9294d14e7c0ed32c68aefe9ee6fff2a6f8")
FOURTH_LOG_SHA256 = (
    "9145e6100b8b4eaea38741209190ba08bfa71578406c21ae8fcee39ce7f83639")
FOURTH_PRODUCER_SHA = "b14a17dd6f14592594daacc4b64c6a1de2a0a004"

LOCKED_A = -0.4390855099920348
LOCKED_B = -0.4326316717808396
LOCKED_C = -0.3484425774226274
LOCKED_TOL = 1.0e-12
D_ADDITIVE = LOCKED_B + LOCKED_C - LOCKED_A


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


def classify_interaction(a: float, b: float, c: float, d: float,
                         *, receipts_valid: bool = True) -> str:
    """Frozen combined-ownership decision with locked-corner validity first."""
    values = np.asarray((a, b, c, d), dtype=np.float64)
    if not np.isfinite(values).all():
        raise SystemExit("STOP non-finite EEN x Omega classifier input")
    if (not receipts_valid
            or abs(a - LOCKED_A) > LOCKED_TOL
            or abs(b - LOCKED_B) > LOCKED_TOL
            or abs(c - LOCKED_C) > LOCKED_TOL):
        return "INVALID_STOP_LOCKED_CORNER"
    if abs(d - HISTORICAL_BASELINE) <= TWO_F:
        return "CONFIRMED_COMBINED_EEN_OMEGA_OWNERSHIP"
    return "REFUTED_COMBINED_EEN_OMEGA_OWNERSHIP"


def classify_additivity(interaction: float) -> str:
    if not np.isfinite(interaction):
        raise SystemExit("STOP non-finite EEN x Omega interaction")
    return "NON_ADDITIVE" if abs(interaction) > TWO_F else "ADDITIVE_BELOW_BAND"


def _valid_hash(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _require_bound() -> None:
    bindings = {
        "NEMO_ARTIFACT_SHA256": NEMO_ARTIFACT_SHA256,
        "LEGACY_ARTIFACT_SHA256": LEGACY_ARTIFACT_SHA256,
        "NEMO_LOG_SHA256": NEMO_LOG_SHA256,
        "LEGACY_LOG_SHA256": LEGACY_LOG_SHA256,
        "FOURTH_ARTIFACT_SHA256": FOURTH_ARTIFACT_SHA256,
        "FOURTH_LOG_SHA256": FOURTH_LOG_SHA256,
    }
    bad = [name for name, value in bindings.items() if not _valid_hash(value)]
    if (not isinstance(BOUND_PRODUCER_SHA, str)
            or len(BOUND_PRODUCER_SHA) != 40
            or any(c not in "0123456789abcdef" for c in BOUND_PRODUCER_SHA)):
        bad.append("BOUND_PRODUCER_SHA")
    if (not isinstance(FOURTH_PRODUCER_SHA, str)
            or len(FOURTH_PRODUCER_SHA) != 40
            or any(c not in "0123456789abcdef" for c in FOURTH_PRODUCER_SHA)):
        bad.append("FOURTH_PRODUCER_SHA")
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


def _require_fourth_log(path: str | Path) -> None:
    _require_hash(path, FOURTH_LOG_SHA256)
    text = Path(path).read_text(errors="replace")
    required = (
        f"PROVENANCE: HEAD={FOURTH_PRODUCER_SHA} dirty_tracked_files=0",
        "ARM: bridge_omega=legacy-rounded omega=7.292e-05 ",
        "f_reference_mode=selected_omega",
        "ARM: een_metric_weighting=off",
        "resolved EEN metric weighting=off",
        "vertical ladder: LEGOESM_NEMO_E3T=both",
        "twin start: bridged",
        "BEFORE-STRESS RECEIPT: stagger=U_AS_T_LEGACY",
        "PRECISION: materialized state dtype = float64",
        "seasonal clock: t_seconds = 15552000s",
        "DONE nsteps=2880 STABLE=True",
    )
    missing = [stamp for stamp in required if stamp not in text]
    if missing:
        raise SystemExit(f"STOP fourth-corner log missing receipts: {missing}")
    print("[CONTROL PASS] fourth-corner log receipts: legacy-rounded, EEN=off, "
          "U_AS_T_LEGACY, fp64, both, bridged, seasonal, stable")


def _require_no_model_diff(left: str, right: str) -> None:
    command = ["git", "-C", str(_DIR), "diff", "--quiet", f"{left}..{right}",
               "--", "packages/", "src/"]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"STOP packages/src changed between {left} and {right}")
    print(f"[CONTROL PASS] git diff --stat {left[:10]}..{right[:10]} -- "
          "packages/ src/: (empty)")


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


def _fourth_receipt_errors(old, een_off_new, een_off_old) -> list[str]:
    """Validate B/C/D while respecting C's older, hash-bound stamp schema."""
    errors: list[str] = []
    expected = {
        "old.een": (str(R._scalar(old, "een_metric_weighting")), "nemo"),
        "fourth.een": (
            str(R._scalar(een_off_old, "een_metric_weighting")), "off"),
        "fourth.mode": (
            str(R._scalar(een_off_old, "bridge_omega_mode")), "legacy-rounded"),
        "fourth.bridge_omega": (
            float(R._scalar(een_off_old, "bridge_omega_rad_s")),
            float(K.constants.Omega)),
        "fourth.config_omega": (
            float(R._scalar(een_off_old, "config_omega_rad_s")),
            float(K.NEMO_CONSTANTS_CONFIG.Omega)),
        "fourth.producer": (
            str(R._scalar(een_off_old, "producer_git_sha")),
            FOURTH_PRODUCER_SHA),
        "fourth.dirty": (
            int(R._scalar(een_off_old, "producer_dirty_tracked_files")), 0),
        "fourth.stable": (bool(R._scalar(een_off_old, "stable")), True),
        "fourth.stress_stagger": (
            str(R._scalar(een_off_old, "bridge_before_stress_stagger")),
            "U_AS_T_LEGACY"),
    }
    errors.extend(f"{name}: got {got!r}, expected {want!r}"
                  for name, (got, want) in expected.items() if got != want)

    # B and D differ only by registered EEN selection and producer provenance.
    common = tuple(key for key in R.IDENTICAL_STAMPS
                   if key not in {"producer_git_sha"})
    for key in (*common, "config_omega_rad_s", "bridge_omega_mode",
                "bridge_omega_rad_s", "bridge_f_T_sha256",
                "bridge_f_u_sha256", "bridge_f_v_sha256"):
        try:
            left = R._scalar(old, key)
            right = R._scalar(een_off_old, key)
        except SystemExit as exc:
            errors.append(str(exc))
            continue
        if left != right:
            errors.append(f"B/D off-axis stamp {key} differs: {left!r} != {right!r}")

    try:
        old_config = json.loads(str(R._scalar(old, "run_config")))
        fourth_config = json.loads(str(R._scalar(een_off_old, "run_config")))
        c_config = json.loads(str(R._scalar(een_off_new, "run_config")))
        if old_config != fourth_config:
            errors.append("B/D run_config differs despite EEN being environment-only")
        # C predates the explicit default selector stamp. Normalize that one
        # registered field, then require every ordinary config leaf identical.
        c_config = dict(c_config)
        c_config.setdefault("bridge_omega", "nemo")
        c_without_axis = dict(c_config)
        d_without_axis = dict(fourth_config)
        c_without_axis.pop("bridge_omega", None)
        d_without_axis.pop("bridge_omega", None)
        if c_without_axis != d_without_axis:
            errors.append("C/D ordinary run_config leaves differ")
        if c_config.get("bridge_omega") != "nemo":
            errors.append("C normalized bridge_omega is not nemo")
        if fourth_config.get("bridge_omega") != "legacy-rounded":
            errors.append("D run_config bridge_omega is not legacy-rounded")
    except (KeyError, TypeError, ValueError, SystemExit) as exc:
        errors.append(f"fourth-corner run_config unreadable: {exc}")

    ok, reasons, _ladder, _dtype = K.certifiable_grid_and_precision(een_off_old)
    if not ok:
        errors.extend(f"fourth off-claim: {reason}" for reason in reasons)
    return errors


def _check_fourth_receipts(old, een_off_new, een_off_old) -> None:
    # The hash-bound C consumer is authoritative for its pre-selector schema.
    E._require_artifact_receipts(een_off_new)
    errors = _fourth_receipt_errors(old, een_off_new, een_off_old)
    if errors:
        raise SystemExit("STOP EEN x Omega receipt control failed:\n  "
                         + "\n  ".join(errors))
    planted = R._Swap(een_off_old, {"een_metric_weighting": "nemo"})
    planted_errors = _fourth_receipt_errors(old, een_off_new, planted)
    if not any("fourth.een" in error for error in planted_errors):
        raise SystemExit("STOP planted fourth-corner EEN receipt swap did not fire")
    print("[CONTROL PASS] planted fourth-corner EEN receipt swap rejected")


def _check_interaction_arithmetic(d: float, reduced_d: float,
                                  interaction: float, d_additive: float) -> None:
    errors = []
    if abs(d - reduced_d) > LOCKED_TOL:
        errors.append(f"reported D differs from reduction by {abs(d-reduced_d):.17g}")
    expected_i = LOCKED_A - LOCKED_B - LOCKED_C + d
    if abs(interaction - expected_i) > LOCKED_TOL:
        errors.append("I differs from A-B-C+D")
    if abs((d - d_additive) - interaction) > LOCKED_TOL:
        errors.append("D-Dadd differs from I")
    if errors:
        raise SystemExit("STOP interaction arithmetic control failed: "
                         + "; ".join(errors))


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

    if classify_interaction(LOCKED_A, LOCKED_B, LOCKED_C,
                            HISTORICAL_BASELINE) != (
            "CONFIRMED_COMBINED_EEN_OMEGA_OWNERSHIP"):
        raise SystemExit("STOP interaction CONFIRM classifier plant did not fire")
    if classify_interaction(LOCKED_A, LOCKED_B, LOCKED_C,
                            HISTORICAL_BASELINE + 2.0 * TWO_F) != (
            "REFUTED_COMBINED_EEN_OMEGA_OWNERSHIP"):
        raise SystemExit("STOP interaction REFUTE classifier plant did not fire")
    if classify_interaction(LOCKED_A + 2.0 * LOCKED_TOL, LOCKED_B, LOCKED_C,
                            HISTORICAL_BASELINE) != "INVALID_STOP_LOCKED_CORNER":
        raise SystemExit("STOP interaction INVALID classifier plant did not fire")
    if classify_additivity(0.0) != "ADDITIVE_BELOW_BAND":
        raise SystemExit("STOP additive classifier plant did not fire")
    if classify_additivity(2.0 * TWO_F) != "NON_ADDITIVE":
        raise SystemExit("STOP non-additive classifier plant did not fire")

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
        c_path, d_path = Path(tmp) / "c.npz", Path(tmp) / "d.npz"
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
        c_stamps = dict(common_stamps)
        c_stamps["producer_git_sha"] = E.ARM_PRODUCER
        np.savez(
            c_path, **c_stamps,
            run_config=json.dumps(common_config, sort_keys=True),
            land_mask=np.ones((2, 2), dtype=np.float64))
        d_stamps = dict(common_stamps)
        d_stamps["producer_git_sha"] = FOURTH_PRODUCER_SHA
        d_stamps["een_metric_weighting"] = "off"
        np.savez(
            d_path, **d_stamps, bridge_omega_mode="legacy-rounded",
            bridge_omega_rad_s=K.constants.Omega,
            bridge_f_T_sha256="f" * 64, bridge_f_u_sha256="1" * 64,
            bridge_f_v_sha256="2" * 64,
            run_config=json.dumps(
                dict(common_config, bridge_omega="legacy-rounded"), sort_keys=True))
        with (np.load(new_path) as new, np.load(old_path) as old,
              np.load(c_path) as een_off_new, np.load(d_path) as een_off_old):
            _check_receipts(new, old, producer)
            _check_fourth_receipts(old, een_off_new, een_off_old)
        test_d = HISTORICAL_BASELINE
        test_i = LOCKED_A - LOCKED_B - LOCKED_C + test_d
        _check_interaction_arithmetic(test_d, test_d, test_i, D_ADDITIVE)
        try:
            _check_interaction_arithmetic(D_ADDITIVE, test_d, test_i, D_ADDITIVE)
        except SystemExit:
            print("[CONTROL PASS] planted D -> Dadd substitution rejected")
        else:
            raise SystemExit("STOP planted D -> Dadd substitution did not fire")
    print("SELF-TEST PASS: paired and interaction CONFIRM/REFUTE/INVALID, "
          "real-NPZ receipts, and Dadd plant fired")
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
    parser.add_argument("--een-off-new", default=(
        "results/dino_1455/tcarry_basin90_legacy_een_off.npz"))
    parser.add_argument("--een-off-new-log", default=(
        "results/dino_1455/tcarry_basin90_legacy_een_off.log"))
    parser.add_argument("--een-off-old", default=(
        "results/dino_1455/tcarry_omega90_legacy_een_off.npz"))
    parser.add_argument("--een-off-old-log", default=(
        "results/dino_1455/tcarry_omega90_legacy_een_off.log"))
    parser.add_argument("--out", default=(
        "/tmp/tcarry_een_omega_interaction_score.json"))
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
    _require_hash(args.een_off_new, E.ARTIFACT_SHA256)
    _require_hash(args.een_off_old, FOURTH_ARTIFACT_SHA256)
    _require_log(args.new_log, "nemo", NEMO_LOG_SHA256)
    _require_log(args.old_log, "legacy-rounded", LEGACY_LOG_SHA256)
    _require_hash(args.een_off_new_log, E.LOG_SHA256)
    E._require_log(args.een_off_new_log)
    _require_fourth_log(args.een_off_old_log)
    E._require_no_model_diff()
    _require_no_model_diff(BOUND_PRODUCER_SHA, FOURTH_PRODUCER_SHA)
    with (np.load(args.new) as new_npz, np.load(args.old) as old_npz,
          np.load(args.een_off_new) as een_off_new_npz,
          np.load(args.een_off_old) as een_off_old_npz):
        _check_receipts(new_npz, old_npz, BOUND_PRODUCER_SHA)
        _check_fourth_receipts(old_npz, een_off_new_npz, een_off_old_npz)
        R._check_day0(new_npz, old_npz)
        R._check_day0(new_npz, een_off_new_npz)
        R._check_day0(new_npz, een_off_old_npz)
        for label, artifact in (("B", old_npz), ("C", een_off_new_npz),
                                ("D", een_off_old_npz)):
            if not R._bit_identical(new_npz["land_mask"], artifact["land_mask"]):
                raise SystemExit(f"STOP A/{label} land_mask differs")
        land = np.asarray(new_npz["land_mask"], dtype=np.float64)
    if not np.array_equal(land > 0.5, R.A.tmask[:, :, 0]):
        raise SystemExit("STOP artifact land_mask differs from NEMO surface tmask")

    new = G.load_candidate(args.new, day=DAY)
    old = G.load_candidate(args.old, day=DAY)
    een_off_new = G.load_candidate(args.een_off_new, day=DAY)
    een_off_old = G.load_candidate(args.een_off_old, day=DAY)
    nemo = R._nemo(DAY)
    wet = R.A.tmask & (land[:, :, None] > 0.5)
    for label, state in (("A", new), ("B", old), ("C", een_off_new),
                         ("D", een_off_old), ("NEMO", nemo)):
        R._check_state_finite(label, state, wet)
    R._reducer_plants(nemo)

    new_abs, new_rows, new_driver = _both_reductions(new)
    old_abs, old_rows, old_driver = _both_reductions(old)
    een_off_new_abs, een_off_new_rows, een_off_new_driver = _both_reductions(
        een_off_new)
    een_off_old_abs, een_off_old_rows, een_off_old_driver = _both_reductions(
        een_off_old)
    nemo_abs, nemo_rows, nemo_driver = _both_reductions(nemo)
    g_new, g_old = new_abs - nemo_abs, old_abs - nemo_abs
    g_een_off_new = een_off_new_abs - nemo_abs
    g_een_off_old = een_off_old_abs - nemo_abs
    delta = g_old - g_new
    omega_verdict = classify(g_new, g_old)
    verdict = classify_interaction(g_new, g_old, g_een_off_new, g_een_off_old)
    interaction = g_new - g_old - g_een_off_new + g_een_off_old
    d_minus_additive = g_een_off_old - D_ADDITIVE
    additivity = classify_additivity(interaction)
    _check_interaction_arithmetic(
        g_een_off_old, een_off_old_abs - nemo_abs, interaction, D_ADDITIVE)
    try:
        _check_interaction_arithmetic(
            D_ADDITIVE, een_off_old_abs - nemo_abs, interaction, D_ADDITIVE)
    except SystemExit:
        print("[CONTROL PASS] planted D -> Dadd substitution rejected")
    else:
        raise SystemExit("STOP planted D -> Dadd substitution did not fire")
    if verdict == "INVALID_STOP_LOCKED_CORNER":
        raise SystemExit("STOP locked A/B/C corner failed 1e-12 reproduction")

    nemo_metrics = G.metrics(nemo, wet)
    gate = {}
    for label, state in (("A-newOmega-EENnemo", new),
                         ("B-oldOmega-EENnemo", old),
                         ("C-newOmega-EENoff", een_off_new),
                         ("D-oldOmega-EENoff", een_off_old)):
        rows = G.classify(G.metrics(state, wet), nemo_metrics, 5)
        failures = G.print_gate(rows, 5, tag=f"[bridge-Omega {label}] ")
        gate[label] = {"pass": len(rows) - failures, "fail": failures}
        if failures:
            raise SystemExit(f"STOP bridge-Omega {label} acceptance gate failed")

    result = {
        "measurement_label": "CONFIRMED measurement",
        "verdict": verdict,
        "additivity_label": additivity,
        "prior_omega_verdict": omega_verdict,
        "GnewOmega90_sv": g_new,
        "GoldOmega90_sv": g_old,
        "GnewOmega_EENoff90_sv": g_een_off_new,
        "GoldOmega_EENoff90_sv": g_een_off_old,
        "Domega90_sv": delta,
        "interaction_I_sv": interaction,
        "Dadd_sv": D_ADDITIVE,
        "D_minus_Dadd_sv": d_minus_additive,
        "current_baseline_sv": CURRENT_BASELINE,
        "historical_baseline_sv": HISTORICAL_BASELINE,
        "target_delta_sv": TARGET_DELTA,
        "two_F_sv": TWO_F,
        "new_reproduction_miss_sv": abs(g_new - CURRENT_BASELINE),
        "old_reproduction_miss_sv": abs(g_old - HISTORICAL_BASELINE),
        "delta_miss_sv": abs(delta - TARGET_DELTA),
        "combined_ownership_miss_sv": abs(g_een_off_old - HISTORICAL_BASELINE),
        "locked_corner_miss_A_sv": abs(g_new - LOCKED_A),
        "locked_corner_miss_B_sv": abs(g_old - LOCKED_B),
        "locked_corner_miss_C_sv": abs(g_een_off_new - LOCKED_C),
        "new_driver_absdiff_sv": abs(new_abs - new_driver),
        "old_driver_absdiff_sv": abs(old_abs - old_driver),
        "een_off_new_driver_absdiff_sv": abs(
            een_off_new_abs - een_off_new_driver),
        "een_off_old_driver_absdiff_sv": abs(
            een_off_old_abs - een_off_old_driver),
        "nemo_driver_absdiff_sv": abs(nemo_abs - nemo_driver),
        "rows_new_sv": (new_rows - nemo_rows).tolist(),
        "rows_old_sv": (old_rows - nemo_rows).tolist(),
        "rows_een_off_new_sv": (een_off_new_rows - nemo_rows).tolist(),
        "rows_een_off_old_sv": (een_off_old_rows - nemo_rows).tolist(),
        "acceptance": gate,
        "git_sha": git_sha,
        "scorer_sha256": sha256(__file__),
        "new_artifact_sha256": sha256(args.new),
        "old_artifact_sha256": sha256(args.old),
        "een_off_new_artifact_sha256": sha256(args.een_off_new),
        "een_off_old_artifact_sha256": sha256(args.een_off_old),
        "producer_git_sha_AB": BOUND_PRODUCER_SHA,
        "producer_git_sha_C": E.ARM_PRODUCER,
        "producer_git_sha_D": FOURTH_PRODUCER_SHA,
    }
    print("[CONFIRMED measurement] [EEN-X-BRIDGE-OMEGA] "
          + json.dumps(result, sort_keys=True))
    print(f"[CONFIRMED measurement] VERDICT={verdict}")
    print(f"[CONFIRMED measurement] ADDITIVITY={additivity}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[PROVENANCE] wrote={Path(args.out).resolve()} sha256={sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
