#!/usr/bin/env python
"""Hash-bound current-SHA EEN-off basin baseline discriminator.

Pre-registration: ``PREREG_tcarry_basin_reverdict.md``, 2026-08-28 binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_DIR.parent))
import acceptance_gate_90d as G  # noqa: E402, N812
import tcarry_basin_reverdict as R  # noqa: E402, N812

ARTIFACT_SHA256 = "a7f3bf5555ec1f6a7ed58792ede8b9f10188c93c21ff81f0553b2fb3cc400dd4"
LOG_SHA256 = "465c01c104dfb8e8f82dc5f6bd0ae70eb50d009ec125457ac259a5f0168e0d02"
ARM_PRODUCER = "6c64f261aa33b43c372b81b30d4569481116766d"
LEGACY_PRODUCER = "d6dc89e91c9ae6b07d146991d2cb6c850f261bb0"
HISTORICAL_BASELINE = -0.4257848785815366
CURRENT_LEGACY_BASELINE = -0.43908550999203477
OWNERSHIP_BAND = 0.0002899800477248501
DAY = 90


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


def classify_ownership(gap: float) -> str:
    if not np.isfinite(gap):
        raise SystemExit("STOP non-finite Goff_current90")
    miss = abs(gap - HISTORICAL_BASELINE)
    return ("CONFIRMED_FULL_EEN_OWNERSHIP" if miss <= OWNERSHIP_BAND
            else "REFUTED_FULL_EEN_OWNERSHIP")


def _self_test() -> int:
    if classify_ownership(HISTORICAL_BASELINE) != "CONFIRMED_FULL_EEN_OWNERSHIP":
        raise SystemExit("STOP exact-baseline CONFIRM plant did not classify")
    inside = HISTORICAL_BASELINE + 0.5 * OWNERSHIP_BAND
    if classify_ownership(inside) != "CONFIRMED_FULL_EEN_OWNERSHIP":
        raise SystemExit("STOP inside-band CONFIRM plant did not classify")
    outside = HISTORICAL_BASELINE + 2.0 * OWNERSHIP_BAND
    if classify_ownership(outside) != "REFUTED_FULL_EEN_OWNERSHIP":
        raise SystemExit("STOP outside-band REFUTE plant did not classify")
    try:
        classify_ownership(float("nan"))
    except SystemExit:
        pass
    else:
        raise SystemExit("STOP non-finite classifier plant did not fire")
    print("SELF-TEST PASS: CONFIRM, REFUTE, and non-finite controls fired")
    return 0


def _require_log(path: str | Path) -> None:
    text = Path(path).read_text(errors="replace")
    required = (
        f"PROVENANCE: HEAD={ARM_PRODUCER} dirty_tracked_files=0",
        "ARM: een_metric_weighting=off",
        "vertical ladder: LEGOESM_NEMO_E3T=both",
        "twin start: bridged",
        "BEFORE-STRESS RECEIPT: stagger=U_AS_T_LEGACY",
        "PRECISION: materialized state dtype = float64",
        "seasonal clock: t_seconds = 15552000s",
        "DONE nsteps=2880 STABLE=True",
    )
    missing = [stamp for stamp in required if stamp not in text]
    if missing:
        raise SystemExit(f"STOP log is missing registered stamps: {missing}")
    print("[CONTROL PASS] log stamps: EEN=off, U_AS_T_LEGACY, fp64, both, "
          "bridged, 15552000s, 2880 steps, STABLE=True, clean producer")


def _require_no_model_diff() -> None:
    command = ["git", "-C", str(_DIR), "diff", "--quiet",
               f"{LEGACY_PRODUCER}..{ARM_PRODUCER}", "--", "packages/", "src/"]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit("STOP packages/src changed between legacy and EEN-off producers")
    print(f"[CONTROL PASS] git diff --stat {LEGACY_PRODUCER[:10]}.."
          f"{ARM_PRODUCER[:10]} -- packages/ src/: (empty)")


def _require_artifact_receipts(artifact) -> np.ndarray:
    expected = {
        "producer_git_sha": ARM_PRODUCER,
        "producer_dirty_tracked_files": 0,
        "control_dtype": "float64",
        "nemo_ladder_mode": "both",
        "twin_start_mode": "bridged",
        "seasonal_t0_reference_seconds": 15552000.0,
        "seasonal_t0_seconds": 15552000.0,
        "bridge_before_stress_stagger": "U_AS_T_LEGACY",
        "surface_stress_implicit": False,
        "rn_Uv": 0.27,
        "stable": True,
    }
    for key, want in expected.items():
        got = R._scalar(artifact, key)
        if got != want:
            raise SystemExit(f"STOP artifact stamp {key}={got!r} != {want!r}")
    ok, reasons, ladder, dtype = R.K.certifiable_grid_and_precision(artifact)
    if not ok:
        raise SystemExit(f"STOP off-claim artifact: {reasons}")
    config = json.loads(str(R._scalar(artifact, "run_config")))
    config_expected = {
        "recipe": "nemo_dino_kamm_mlf",
        "n_days": DAY,
        "bridge_before": True,
        "bridge_before_stress_tpoint": False,
        "surface_stress_implicit": False,
        "perturb_seed": None,
        "perturb_baro": None,
        "daily_acc": False,
    }
    for key, want in config_expected.items():
        if config.get(key, object()) != want:
            raise SystemExit(f"STOP run_config {key}={config.get(key)!r} != {want!r}")
    land = np.asarray(artifact["land_mask"], dtype=np.float64)
    print("[PROVENANCE] artifact receipts=" + json.dumps(
        {**expected, "certified_ladder": ladder, "certified_dtype": dtype,
         "run_config": config}, sort_keys=True))
    return land


def _both_reductions(state: dict[str, np.ndarray]) -> tuple[float, np.ndarray, float]:
    row_value, rows = R._reduce(state)
    driver_value = float(R.D.rowset(state, R.A.tmask, with_density=False)["g_south"])
    difference = abs(row_value - driver_value)
    if difference > 1.0e-12:
        raise SystemExit(f"STOP independent reducers differ by {difference:.17g} Sv")
    return row_value, rows, driver_value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", default=(
        "results/dino_1455/tcarry_basin90_legacy_een_off.npz"))
    parser.add_argument("--log", default=(
        "results/dino_1455/tcarry_basin90_legacy_een_off.log"))
    parser.add_argument("--out", default="/tmp/tcarry_een_off_discriminator.json")
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
        raise SystemExit("STOP discriminator checkout has dirty tracked files")

    _require_hash(args.artifact, ARTIFACT_SHA256)
    _require_hash(args.log, LOG_SHA256)
    _require_log(args.log)
    _require_no_model_diff()

    with np.load(args.artifact) as artifact:
        land = _require_artifact_receipts(artifact)
    if not np.array_equal(land > 0.5, R.A.tmask[:, :, 0]):
        raise SystemExit("STOP artifact land_mask differs from NEMO surface tmask")

    candidate = G.load_candidate(args.artifact, day=DAY)
    nemo = R._nemo(DAY)
    wet = R.A.tmask & (land[:, :, None] > 0.5)
    R._check_state_finite("EEN-off", candidate, wet)
    R._check_state_finite("NEMO", nemo, wet)
    R._reducer_plants(nemo)

    candidate_abs, candidate_rows, candidate_driver = _both_reductions(candidate)
    nemo_abs, nemo_rows, nemo_driver = _both_reductions(nemo)
    gap = candidate_abs - nemo_abs
    miss = abs(gap - HISTORICAL_BASELINE)
    verdict = classify_ownership(gap)

    nemo_metrics = G.metrics(nemo, wet)
    candidate_metrics = G.metrics(candidate, wet)
    gate_rows = G.classify(candidate_metrics, nemo_metrics, 5)
    gate_fail = G.print_gate(gate_rows, 5, tag="[EEN-off discriminator] ")
    if gate_fail:
        raise SystemExit(f"STOP EEN-off acceptance gate has {gate_fail} failures")

    result = {
        "measurement_label": "CONFIRMED measurement",
        "verdict": verdict,
        "Goff_current90_sv": gap,
        "historical_baseline_sv": HISTORICAL_BASELINE,
        "current_legacy_baseline_sv": CURRENT_LEGACY_BASELINE,
        "absolute_miss_sv": miss,
        "ownership_band_sv": OWNERSHIP_BAND,
        "within_band": bool(miss <= OWNERSHIP_BAND),
        "candidate_absolute_sv": candidate_abs,
        "nemo_absolute_sv": nemo_abs,
        "candidate_driver_sv": candidate_driver,
        "nemo_driver_sv": nemo_driver,
        "candidate_reducer_absdiff_sv": abs(candidate_abs - candidate_driver),
        "nemo_reducer_absdiff_sv": abs(nemo_abs - nemo_driver),
        "rows_gap_sv": (candidate_rows - nemo_rows).tolist(),
        "acceptance_pass": len(gate_rows) - gate_fail,
        "acceptance_fail": gate_fail,
        "git_sha": git_sha,
        "scorer_sha256": sha256(__file__),
        "artifact_sha256": sha256(args.artifact),
        "log_sha256": sha256(args.log),
        "producer_git_sha": ARM_PRODUCER,
    }
    print("[CONFIRMED measurement] [EEN-OFF] " + json.dumps(result, sort_keys=True))
    print(f"[CONFIRMED measurement] VERDICT={verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[PROVENANCE] wrote={Path(args.out).resolve()} sha256={sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
