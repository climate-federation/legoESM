#!/usr/bin/env python
"""Fail-closed paired scorer for the corrected-T-carry basin re-verdict.

This composes the campaign's existing state loader, acceptance metrics,
``g_south`` reduction, and per-row reduction. It intentionally owns no new
ocean metric. See ``PREREG_tcarry_basin_reverdict.md``.
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
import acc_driver_decomp as D  # noqa: E402, N812
import acc_thermal_wind as A  # noqa: E402, N812
import acceptance_gate_90d as G  # noqa: E402, N812
import basin_seasonal_decomp as B  # noqa: E402, N812
import kamm_twin_90d as K  # noqa: E402, N812

BASELINE = {90: -0.4257848785815366, 360: -0.9519122331848315}
FLOOR = {90: 0.00014499002386242506, 360: 0.06173656216045926}
EXPECTED_T_HASH = "b6a08b8395017c8e3f8df0b8b13eefa75fdfe7be3770d788beaaf1ca514127ae"
FLOOR_RECEIPT = "/tmp/dino_basin_seasonal_decomp.json"
FLOOR_RECEIPT_SHA256 = "63d4e60dd68281bc6101a35f86cb3f4406cb2ddbc27848b74473876226e85849"
IDENTICAL_STAMPS = (
    "control_dtype",
    "nemo_ladder_mode",
    "seasonal_t0_reference_seconds",
    "seasonal_t0_seconds",
    "surface_stress_implicit",
    "twin_start_mode",
    "vertical_ladder_sha256",
    "rn_Uv",
    "producer_git_sha",
    "producer_dirty_tracked_files",
)
DAY0_KEYS = ("T3d_day0", "S3d_day0", "eta3d_day0", "u3d_day0", "v3d_day0")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_finite(label: str, *values) -> None:
    for value in values:
        if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
            raise SystemExit(f"STOP non-finite {label}")


def classify(delta: float, legacy: float, floor: float, safe: bool) -> str:
    """Registered, mutually exclusive decision tree."""
    _require_finite("classifier input", delta, legacy, floor)
    if legacy == 0.0 or floor < 0.0:
        raise SystemExit("STOP invalid zero legacy or negative floor")
    if abs(delta) <= 2.0 * floor:
        return "UNRESOLVED/FLOOR"
    ratio = delta / abs(legacy)
    if not safe:
        return "UNRESOLVED/COMPENSATION"
    if ratio >= 0.10:
        return "CONFIRMED"
    if ratio <= 0.02:
        return "REFUTED"
    return "UNRESOLVED"


def _scalar(z: np.lib.npyio.NpzFile, key: str):
    if key not in z.files:
        raise SystemExit(f"STOP missing artifact stamp {key}")
    return np.asarray(z[key]).item()


def _registered_config_errors(config: dict, tag: str, day: int) -> list[str]:
    expected = {
        "recipe": "nemo_dino_kamm_mlf",
        "n_days": day,
        "bridge_tke": False,
        "bridge_before": True,
        "vmix_scheme": None,
        "use_gm_redi": None,
        "surface_stress_implicit": False,
        "surface_tendency_placement": None,
        "save_step_eta": False,
        "perturb_seed": None,
        "perturb_eps": 1e-14,
        "perturb_baro": None,
        "perturb_baro_sha256": None,
        "perturb_baro_key": "dU_avg",
        "perturb_baro_scale": 1.0,
        "daily_acc": False,
        "u_m": None,
    }
    return [f"{tag} run_config {key}={config.get(key)!r}, expected {want!r}"
            for key, want in expected.items() if config.get(key, object()) != want]


def _receipt_errors(legacy: np.lib.npyio.NpzFile,
                    corrected: np.lib.npyio.NpzFile,
                    day: int, expected_sha: str | None = None) -> list[str]:
    errors = []
    for key in IDENTICAL_STAMPS:
        try:
            left, right = _scalar(legacy, key), _scalar(corrected, key)
        except SystemExit as exc:
            errors.append(str(exc))
            continue
        if left != right:
            errors.append(f"stamp {key} differs: {left!r} != {right!r}")
    expected = {
        "legacy.bridge_before_stress_stagger": (
            _scalar(legacy, "bridge_before_stress_stagger"), "U_AS_T_LEGACY"),
        "corrected.bridge_before_stress_stagger": (
            _scalar(corrected, "bridge_before_stress_stagger"), "T"),
        "corrected.bridge_before_stress_reconstruction_seconds": (
            float(_scalar(corrected, "bridge_before_stress_reconstruction_seconds")),
            15552000.0),
        "corrected.bridge_before_stress_sha256": (
            _scalar(corrected, "bridge_before_stress_sha256"), EXPECTED_T_HASH),
        "legacy.surface_stress_implicit": (
            bool(_scalar(legacy, "surface_stress_implicit")), False),
        "corrected.surface_stress_implicit": (
            bool(_scalar(corrected, "surface_stress_implicit")), False),
        "legacy.stable": (bool(_scalar(legacy, "stable")), True),
        "corrected.stable": (bool(_scalar(corrected, "stable")), True),
        "legacy.rn_Uv": (float(_scalar(legacy, "rn_Uv")), 0.27),
        "corrected.rn_Uv": (float(_scalar(corrected, "rn_Uv")), 0.27),
        "legacy.seasonal_t0_seconds": (
            float(_scalar(legacy, "seasonal_t0_seconds")), 15552000.0),
        "corrected.seasonal_t0_seconds": (
            float(_scalar(corrected, "seasonal_t0_seconds")), 15552000.0),
        "legacy.seasonal_t0_reference_seconds": (
            float(_scalar(legacy, "seasonal_t0_reference_seconds")), 15552000.0),
        "corrected.seasonal_t0_reference_seconds": (
            float(_scalar(corrected, "seasonal_t0_reference_seconds")), 15552000.0),
        "legacy.producer_dirty_tracked_files": (
            int(_scalar(legacy, "producer_dirty_tracked_files")), 0),
        "corrected.producer_dirty_tracked_files": (
            int(_scalar(corrected, "producer_dirty_tracked_files")), 0),
    }
    errors.extend(f"{name}: got {got!r}, expected {want!r}"
                  for name, (got, want) in expected.items() if got != want)
    for tag, artifact in (("legacy", legacy), ("corrected", corrected)):
        producer_sha = str(_scalar(artifact, "producer_git_sha"))
        if len(producer_sha) != 40 or any(c not in "0123456789abcdef" for c in producer_sha):
            errors.append(f"{tag} producer_git_sha is not a full SHA-1: {producer_sha!r}")
        if expected_sha is not None and producer_sha != expected_sha:
            errors.append(f"{tag} producer_git_sha {producer_sha} != scorer HEAD "
                          f"{expected_sha}")

    try:
        lc = json.loads(str(_scalar(legacy, "run_config")))
        cc = json.loads(str(_scalar(corrected, "run_config")))
        differing = {key for key in lc.keys() | cc.keys() if lc.get(key) != cc.get(key)}
        if differing != {"bridge_before_stress_tpoint"}:
            errors.append(f"run_config differences {sorted(differing)!r}, expected only "
                          "bridge_before_stress_tpoint")
        if lc.get("bridge_before_stress_tpoint") is not False:
            errors.append("legacy run_config did not select bridge_before_stress_tpoint=false")
        if cc.get("bridge_before_stress_tpoint") is not True:
            errors.append("corrected run_config did not select bridge_before_stress_tpoint=true")
        if lc.get("n_days") != day or cc.get("n_days") != day:
            errors.append(f"run_config n_days is not the scored endpoint {day}")
        errors.extend(_registered_config_errors(lc, "legacy", day))
        errors.extend(_registered_config_errors(cc, "corrected", day))
    except (KeyError, TypeError, ValueError, SystemExit) as exc:
        errors.append(f"run_config receipt unreadable: {exc}")
    for tag, artifact in (("legacy", legacy), ("corrected", corrected)):
        ok, reasons, _ladder, _dtype = K.certifiable_grid_and_precision(artifact)
        if not ok:
            errors.extend(f"{tag} off-claim: {reason}" for reason in reasons)
        for key in ("perturb_baro", "perturb_baro_key", "injected_sv"):
            if key in artifact.files:
                errors.append(f"{tag} carries forbidden perturbation stamp {key}")
    return errors


def _check_receipts(legacy: np.lib.npyio.NpzFile,
                    corrected: np.lib.npyio.NpzFile, day: int,
                    expected_sha: str) -> None:
    errors = _receipt_errors(legacy, corrected, day, expected_sha)
    if errors:
        raise SystemExit("STOP receipt control failed:\n  " + "\n  ".join(errors))

    # Planted stagger swap: a receipt that labels the corrected arm as legacy
    # must be rejected. This proves the selector control can fail.
    class _Swap:
        files = corrected.files

        def __init__(self, overrides):
            self.overrides = overrides

        def __getitem__(self, key):
            if key in self.overrides:
                return np.asarray(self.overrides[key])
            return corrected[key]

    planted_errors = _receipt_errors(
        legacy, _Swap({"bridge_before_stress_stagger": "U_AS_T_LEGACY"}), day,
        expected_sha)
    if not any("corrected.bridge_before_stress_stagger" in x for x in planted_errors):
        raise SystemExit("STOP planted selector swap did not fire")
    print("[CONTROL PASS] planted corrected-T -> U_AS_T_LEGACY stamp swap rejected")
    offclaim_errors = _receipt_errors(
        legacy, _Swap({"control_dtype": "float32"}), day, expected_sha)
    if not any("control dtype" in x for x in offclaim_errors):
        raise SystemExit("STOP planted off-claim precision did not fire")
    print("[CONTROL PASS] planted corrected float64 -> float32 claim violation rejected")
    viscosity_errors = _receipt_errors(
        legacy, _Swap({"rn_Uv": 0.54}), day, expected_sha)
    if not any("corrected.rn_Uv" in x or "stamp rn_Uv differs" in x
               for x in viscosity_errors):
        raise SystemExit("STOP planted rn_Uv confound did not fire")
    print("[CONTROL PASS] planted corrected rn_Uv 0.27 -> 0.54 confound rejected")
    bad_config = json.loads(str(_scalar(corrected, "run_config")))
    bad_config["perturb_baro"] = "/tmp/planted.npz"
    bad_config["perturb_baro_sha256"] = "0" * 64
    perturb_errors = _receipt_errors(
        legacy, _Swap({"run_config": json.dumps(bad_config, sort_keys=True)}),
        day, expected_sha)
    if not any("perturb_baro" in x for x in perturb_errors):
        raise SystemExit("STOP planted barotropic perturbation confound did not fire")
    print("[CONTROL PASS] planted unregistered perturb_baro config rejected")
    receipt = {
        key: {"legacy": _scalar(legacy, key), "corrected": _scalar(corrected, key)}
        for key in IDENTICAL_STAMPS
    }
    receipt["bridge_before_stress_stagger"] = {
        "legacy": _scalar(legacy, "bridge_before_stress_stagger"),
        "corrected": _scalar(corrected, "bridge_before_stress_stagger"),
    }
    receipt["bridge_before_stress_reconstruction_seconds"] = {
        "legacy": float(_scalar(legacy, "bridge_before_stress_reconstruction_seconds")),
        "corrected": float(_scalar(corrected, "bridge_before_stress_reconstruction_seconds")),
    }
    receipt["bridge_before_stress_sha256"] = {
        "legacy": _scalar(legacy, "bridge_before_stress_sha256"),
        "corrected": _scalar(corrected, "bridge_before_stress_sha256"),
    }
    print("[PROVENANCE] stamps=" + json.dumps(receipt, sort_keys=True))


def _bit_identical(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return (a.shape == b.shape and a.dtype == b.dtype
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def _check_day0(legacy: np.lib.npyio.NpzFile,
                corrected: np.lib.npyio.NpzFile) -> None:
    for key in DAY0_KEYS:
        if key not in legacy.files or key not in corrected.files:
            raise SystemExit(f"STOP missing paired day-0 field {key}")
        if not _bit_identical(legacy[key], corrected[key]):
            raise SystemExit(f"STOP paired day-0 field is nonfinite or not bit-identical: {key}")
    print(f"[CONTROL PASS] paired day-0 fields bit-identical: {', '.join(DAY0_KEYS)}")


def _reduce(st: dict[str, np.ndarray]) -> tuple[float, np.ndarray]:
    south = slice(0, A.J0)
    rows = B.row_transport(st["u"], A.umask, south)
    g_rows = float(rows.sum())
    g_driver = float(D.rowset(st, A.tmask, with_density=False)["g_south"])
    _require_finite("basin row/reducer", rows, g_rows, g_driver)
    if abs(g_rows - g_driver) > 1e-12:
        raise SystemExit(f"STOP independent basin reductions differ: "
                         f"rows={g_rows:.17g}, g_south={g_driver:.17g}, "
                         f"absdiff={abs(g_rows-g_driver):.17g}")
    return g_rows, rows


def _reducer_plants(reference: dict[str, np.ndarray]) -> None:
    u = np.asarray(reference["u"], dtype=np.float64)
    base, _ = _reduce(reference)

    dry = u.copy()
    dry[~A.umask] += 123.0
    dry_score, _ = _reduce({**reference, "u": dry})
    if dry_score != base:
        raise SystemExit(f"STOP dry-face plant moved basin score by {dry_score-base:.17g} Sv")

    wet = u.copy()
    support = A.umask[:A.J0, 2:-2]
    j, ii, k = np.argwhere(support)[0]
    i = int(ii) + 2
    wet[int(j), i, int(k)] += 1.0
    wet_score, _ = _reduce({**reference, "u": wet})
    if not wet_score > base:
        raise SystemExit("STOP wet-face plant did not move basin score positively")
    print(f"[CONTROL PASS] dry plant delta={dry_score-base:.17g} Sv; "
          f"wet plant at ({int(j)},{i},{int(k)}) delta={wet_score-base:.17g} Sv")


def _check_state_finite(label: str, st: dict[str, np.ndarray], wet_t: np.ndarray) -> None:
    for key in ("T", "S"):
        _require_finite(f"{label} {key} on wet T support", np.asarray(st[key])[wet_t])
    _require_finite(f"{label} u on wet U support", np.asarray(st["u"])[A.umask])


def _nemo(day: int) -> dict[str, np.ndarray]:
    # Member zero is the exact retained verdict member that supplied the
    # registered baseline and floor at both endpoints.
    return B.nemo_state(0, day)


def _check_floor_receipt(path: str, day: int) -> None:
    got_hash = sha256(path)
    if got_hash != FLOOR_RECEIPT_SHA256:
        raise SystemExit(f"STOP floor receipt hash {got_hash} != {FLOOR_RECEIPT_SHA256}")
    data = json.loads(Path(path).read_text())
    try:
        pos = data["days"].index(day)
        got_gap = float(data["gap"][pos])
        got_floor = float(data["floor"][pos])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"STOP floor receipt cannot supply day {day}: {exc}") from exc
    if got_gap != BASELINE[day] or got_floor != FLOOR[day]:
        raise SystemExit(f"STOP floor receipt day {day}: gap/floor "
                         f"{got_gap:.17g}/{got_floor:.17g} != registered "
                         f"{BASELINE[day]:.17g}/{FLOOR[day]:.17g}")
    print(f"[CONTROL PASS] floor receipt={Path(path).resolve()} sha256={got_hash} "
          f"day={day} gap={got_gap:.17g} floor={got_floor:.17g}")


def _check_baseline(value: float, day: int) -> None:
    _require_finite("legacy baseline", value)
    if abs(value - BASELINE[day]) > 2.0 * FLOOR[day]:
        raise SystemExit(f"STOP legacy baseline {value:.17g} misses registered "
                         f"{BASELINE[day]:.17g} by more than 2F")


def _self_test() -> int:
    cases = (
        (0.0, -1.0, 0.1, True, "UNRESOLVED/FLOOR"),
        (0.25, -1.0, 0.01, True, "CONFIRMED"),
        (0.01, -1.0, 0.001, True, "REFUTED"),
        (0.05, -1.0, 0.001, True, "UNRESOLVED"),
        (0.25, -1.0, 0.01, False, "UNRESOLVED/COMPENSATION"),
    )
    for delta, legacy, floor, safe, want in cases:
        got = classify(delta, legacy, floor, safe)
        if got != want:
            raise SystemExit(f"classifier self-test failed: {got} != {want}")
    for bad in (float("nan"), float("inf")):
        try:
            classify(bad, -1.0, 0.1, True)
        except SystemExit:
            pass
        else:
            raise SystemExit("non-finite classifier plant did not fire")
    if _bit_identical(np.array([1.0]), np.array([np.nan])):
        raise SystemExit("non-finite day-0 plant did not fire")
    print("SELF-TEST PASS: floor/confirm/refute/intermediate/compensation are exclusive")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("legacy", nargs="?")
    ap.add_argument("corrected", nargs="?")
    ap.add_argument("--day", type=int, choices=(90, 360), default=90)
    ap.add_argument("--out")
    ap.add_argument("--floor-receipt", default=FLOOR_RECEIPT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.legacy or not args.corrected or not args.out:
        ap.error("legacy, corrected, and --out are required unless --self-test")

    git_sha = subprocess.run(
        ["git", "-C", str(_DIR), "rev-parse", "HEAD"],
        capture_output=True, check=False, text=True).stdout.strip()
    print(f"[PROVENANCE] git_sha={git_sha}")
    print(f"[PROVENANCE] scorer={Path(__file__).resolve()} sha256={sha256(__file__)}")
    print(f"[PROVENANCE] legacy={Path(args.legacy).resolve()} sha256={sha256(args.legacy)}")
    print(f"[PROVENANCE] corrected={Path(args.corrected).resolve()} "
          f"sha256={sha256(args.corrected)}")
    print(f"[PROVENANCE] day={args.day} baseline={BASELINE[args.day]:.17g} "
          f"floor={FLOOR[args.day]:.17g}")
    print("[PROVENANCE] flags=" + json.dumps(vars(args), sort_keys=True))
    dirty = subprocess.run(
        ["git", "-C", str(_DIR), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, check=False, text=True).stdout.strip()
    print(f"[PROVENANCE] dirty_tracked_files={len(dirty.splitlines()) if dirty else 0}")
    if dirty:
        raise SystemExit("STOP scorer checkout has dirty tracked files")
    _check_floor_receipt(args.floor_receipt, args.day)

    with np.load(args.legacy) as legacy_npz, np.load(args.corrected) as corrected_npz:
        _check_receipts(legacy_npz, corrected_npz, args.day, git_sha)
        _check_day0(legacy_npz, corrected_npz)
        for key in ("land_mask",):
            if not _bit_identical(legacy_npz[key], corrected_npz[key]):
                raise SystemExit(f"STOP paired {key} differs")
        land = np.asarray(legacy_npz["land_mask"], dtype=np.float64)

    if not np.array_equal(land > 0.5, A.tmask[:, :, 0]):
        raise SystemExit("STOP artifact land_mask differs from NEMO surface tmask")

    legacy = G.load_candidate(args.legacy, day=args.day)
    corrected = G.load_candidate(args.corrected, day=args.day)
    nemo = _nemo(args.day)
    wet = A.tmask & (land[:, :, None] > 0.5)
    _check_state_finite("legacy", legacy, wet)
    _check_state_finite("corrected", corrected, wet)
    _check_state_finite("NEMO", nemo, wet)
    _reducer_plants(nemo)

    nemo_g, nemo_rows = _reduce(nemo)
    legacy_g_abs, legacy_rows_abs = _reduce(legacy)
    corrected_g_abs, corrected_rows_abs = _reduce(corrected)
    legacy_gap = legacy_g_abs - nemo_g
    corrected_gap = corrected_g_abs - nemo_g
    legacy_rows = legacy_rows_abs - nemo_rows
    corrected_rows = corrected_rows_abs - nemo_rows
    delta = corrected_gap - legacy_gap
    ratio = delta / abs(legacy_gap)
    _require_finite("transport verdict", legacy_gap, corrected_gap, delta, ratio)
    _check_baseline(legacy_gap, args.day)

    nemo_metrics = G.metrics(nemo, wet)
    legacy_metrics = G.metrics(legacy, wet)
    corrected_metrics = G.metrics(corrected, wet)
    metric_rows = {}
    safe = True
    for key in G.KEYS:
        lgap = abs(legacy_metrics[key] - nemo_metrics[key])
        cgap = abs(corrected_metrics[key] - nemo_metrics[key])
        regression = cgap - lgap
        _require_finite(f"acceptance metric {key}", nemo_metrics[key],
                        legacy_metrics[key], corrected_metrics[key], lgap, cgap,
                        regression)
        passed = regression <= G.FLOORS[key]
        safe &= passed
        metric_rows[key] = {
            "nemo": nemo_metrics[key], "legacy": legacy_metrics[key],
            "corrected": corrected_metrics[key], "legacy_abs_gap": lgap,
            "corrected_abs_gap": cgap, "gap_regression": regression,
            "allowed_regression": G.FLOORS[key], "safe": bool(passed),
            "legacy_gate_5x": bool(lgap <= 5.0 * G.FLOORS[key]),
            "corrected_gate_5x": bool(cgap <= 5.0 * G.FLOORS[key]),
        }

    verdict = classify(delta, legacy_gap, FLOOR[args.day], bool(safe))
    result = {
        "label": verdict,
        "measurement_label": "CONFIRMED measurement",
        "day": args.day,
        "legacy_gap_sv": legacy_gap,
        "corrected_gap_sv": corrected_gap,
        "delta_sv": delta,
        "ratio": ratio,
        "floor_sv": FLOOR[args.day],
        "two_floor_sv": 2.0 * FLOOR[args.day],
        "baseline_sv": BASELINE[args.day],
        "G4_legacy_sv": float(legacy_rows[1:5].sum()),
        "G4_corrected_sv": float(corrected_rows[1:5].sum()),
        "rows_legacy_sv": legacy_rows.tolist(),
        "rows_corrected_sv": corrected_rows.tolist(),
        "metrics": metric_rows,
        "acceptance_5x": {
            "legacy": all(row["legacy_gate_5x"] for row in metric_rows.values()),
            "corrected": all(row["corrected_gate_5x"] for row in metric_rows.values()),
        },
        "git_sha": git_sha,
        "scorer_sha256": sha256(__file__),
        "legacy_sha256": sha256(args.legacy),
        "corrected_sha256": sha256(args.corrected),
    }
    evidence_label = result["measurement_label"]
    print(f"[{evidence_label}] [BASIN] " + json.dumps({k: result[k] for k in (
        "day", "legacy_gap_sv", "corrected_gap_sv", "delta_sv", "ratio",
        "floor_sv", "G4_legacy_sv", "G4_corrected_sv")}, sort_keys=True))
    print(f"[{evidence_label}] [ROWS legacy] " + json.dumps(result["rows_legacy_sv"]))
    print(f"[{evidence_label}] [ROWS corrected] " + json.dumps(result["rows_corrected_sv"]))
    print(f"[{evidence_label}] [METRICS] " + json.dumps(metric_rows, sort_keys=True))
    print(f"[{evidence_label}] [ACCEPTANCE_5X] "
          + json.dumps(result["acceptance_5x"], sort_keys=True))
    print(f"[{evidence_label}] VERDICT={verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[PROVENANCE] wrote={Path(args.out).resolve()} sha256={sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
