#!/usr/bin/env python
"""Rule-1e reconciliation of the old and current T-carry basin baselines.

Pre-registration: ``PREREG_tcarry_baseline_reconciliation.md``.
This composes the campaign's existing loaders and reducers; it defines no new
transport convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
import basin_seasonal_decomp as B  # noqa: E402, N812
import tcarry_basin_reverdict as R  # noqa: E402, N812

DAYS = (30, 60, 90)
OLD_RECEIPT_SHA = "63d4e60dd68281bc6101a35f86cb3f4406cb2ddbc27848b74473876226e85849"
OLD_ARTIFACT_SHA = "f2031397ec6230aa07c40408947821e5c67a2156c5099ead6a66182e3e3f4ec9"
CURRENT_ARTIFACT_SHA = "ae114e7c71f530da253083e4f07f83e94bb66ed1d89c06c27909da9b36fc1e37"
EEN_OFF_SHA = "7677ef28c8733673117576b73f000574e1737f008d62c7686fdc45997f86a6ac"
EEN_NEMO_SHA = "5c164e41a8a1f8259a712263d281391dc7dff04bbabf72c5d6e2cc72e02830ac"
OLD_PRODUCER = "a7b940f75c04d824b478de4e1728220e3a71989e"
CURRENT_PRODUCER = "d6dc89e91c9ae6b07d146991d2cb6c850f261bb0"
EEN_PRODUCER = "a6a07a9e2b4c31201691bd829bceb4984f374d62"
OMEGA_ROUNDED = 7.292e-5  # const-ok: historical legoESM bridge receipt
OMEGA_NEMO = 7.292115083046e-5  # const-ok: NEMO ff_f inversion receipt
OLD_REPRO_TOL = 1.0e-12


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: str | Path, expected: str) -> None:
    got = _sha256(path)
    if got != expected:
        raise SystemExit(f"STOP input hash {path}: {got} != {expected}")
    print(f"[PROVENANCE] input={Path(path).resolve()} sha256={got}")


def _log_producer(path: str | Path, expected: str) -> None:
    text = Path(path).read_text(errors="replace")
    match = re.search(r"PROVENANCE: HEAD=([0-9a-f]{40}) dirty_tracked_files=(\d+)", text)
    if not match or match.group(1) != expected or match.group(2) != "0":
        raise SystemExit(f"STOP producer receipt in {path} is absent/dirty/wrong: {match}")
    print(f"[CONTROL PASS] log={Path(path).resolve()} producer={expected} dirty=0")


def _stamp(npz_path: str | Path, producer: str, stagger: str | None = None) -> None:
    with np.load(npz_path) as artifact:
        if "producer_git_sha" in artifact.files:
            got = str(np.asarray(artifact["producer_git_sha"]).item())
            dirty = int(np.asarray(artifact["producer_dirty_tracked_files"]).item())
            if got != producer or dirty != 0:
                raise SystemExit(f"STOP artifact producer {got}/{dirty} != {producer}/0")
        for key, want in (("control_dtype", "float64"),
                          ("nemo_ladder_mode", "both"),
                          ("seasonal_t0_seconds", 15552000.0)):
            got = np.asarray(artifact[key]).item()
            if got != want:
                raise SystemExit(f"STOP artifact {key}={got!r} != {want!r}")
        if stagger is not None:
            got = str(np.asarray(artifact["bridge_before_stress_stagger"]).item())
            if got != stagger:
                raise SystemExit(f"STOP stress stagger {got!r} != {stagger!r}")


def _bit_identical_day0(old_path: str | Path, current_path: str | Path) -> None:
    with np.load(old_path) as old, np.load(current_path) as current:
        for key in R.DAY0_KEYS:
            if not R._bit_identical(old[key], current[key]):
                raise SystemExit(f"STOP old/current day-0 field differs: {key}")
    print("[CONTROL PASS] old/current day-0 prognostic fields are bit-identical")


def _old_gap_matches(value: float, expected: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(expected)
                and abs(value - expected) <= OLD_REPRO_TOL)


def _gap(path: str | Path, day: int, nemo: dict[str, np.ndarray],
         *, historical: bool = False):
    prior = os.environ.get("DINO_GATE_ALLOW_LEGACY_CLOCK")
    if historical:
        print(f"[PROVENANCE] historical clock-stamp escape active for {path}; "
              "hash/log/15552000s receipts passed")
        os.environ["DINO_GATE_ALLOW_LEGACY_CLOCK"] = "1"
    else:
        os.environ.pop("DINO_GATE_ALLOW_LEGACY_CLOCK", None)
    try:
        state = G.load_candidate(str(path), day=day)
    finally:
        if prior is None:
            os.environ.pop("DINO_GATE_ALLOW_LEGACY_CLOCK", None)
        else:
            os.environ["DINO_GATE_ALLOW_LEGACY_CLOCK"] = prior
    value, rows = R._reduce(state)
    nemo_value, nemo_rows = R._reduce(nemo)
    return value - nemo_value, rows - nemo_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old-receipt", default="/tmp/dino_basin_seasonal_decomp.json")
    parser.add_argument("--old-artifact", default="/tmp/dino_verdict360/m0_control.npz")
    parser.add_argument("--old-log", default="/tmp/dino_verdict360/m0_control.log")
    parser.add_argument("--old-launch", default="/tmp/dino_verdict360/.launch_sha")
    parser.add_argument("--current", default="results/dino_1455/tcarry_basin90_legacy.npz")
    parser.add_argument("--current-log", default="results/dino_1455/tcarry_basin90_legacy.log")
    een_dir = "/home/dbalwada/legoESM/results/dino_1455_een_metric"
    parser.add_argument("--een-off", default=f"{een_dir}/twin90_armA_off.npz")
    parser.add_argument("--een-off-log", default=f"{een_dir}/twin90_armA_off.log")
    parser.add_argument("--een-nemo", default=f"{een_dir}/twin90_armB_nemo.npz")
    parser.add_argument("--een-nemo-log", default=f"{een_dir}/twin90_armB_nemo.log")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    git_sha = subprocess.run(["git", "-C", str(_DIR), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(_DIR), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True).stdout.strip()
    print(f"[PROVENANCE] git_sha={git_sha} dirty_tracked_files="
          f"{len(dirty.splitlines()) if dirty else 0}")
    print("[PROVENANCE] flags=" + json.dumps(vars(args), sort_keys=True))
    print(f"[PROVENANCE] probe={Path(__file__).resolve()} sha256={_sha256(__file__)}")
    if dirty:
        raise SystemExit("STOP reconciliation probe checkout has dirty tracked files")

    for path, expected in ((args.old_receipt, OLD_RECEIPT_SHA),
                           (args.old_artifact, OLD_ARTIFACT_SHA),
                           (args.current, CURRENT_ARTIFACT_SHA),
                           (args.een_off, EEN_OFF_SHA),
                           (args.een_nemo, EEN_NEMO_SHA)):
        _require_hash(path, expected)
    launch = Path(args.old_launch).read_text().strip()
    if launch != OLD_PRODUCER:
        raise SystemExit(f"STOP old launch SHA {launch} != {OLD_PRODUCER}")
    _log_producer(args.old_log, OLD_PRODUCER)
    _log_producer(args.current_log, CURRENT_PRODUCER)
    _log_producer(args.een_off_log, EEN_PRODUCER)
    _log_producer(args.een_nemo_log, EEN_PRODUCER)
    _stamp(args.old_artifact, OLD_PRODUCER)
    _stamp(args.current, CURRENT_PRODUCER, "U_AS_T_LEGACY")
    _stamp(args.een_off, EEN_PRODUCER)
    _stamp(args.een_nemo, EEN_PRODUCER)
    _bit_identical_day0(args.old_artifact, args.current)

    receipt = json.loads(Path(args.old_receipt).read_text())
    if receipt.get("git_head") != "1d68fb289d6457e74ced8a1c71ab7eaceb8a29b1":
        raise SystemExit("STOP old analysis receipt HEAD changed")
    old_by_day = {int(day): (float(receipt["gap"][i]), float(receipt["floor"][i]))
                  for i, day in enumerate(receipt["days"])}
    if _old_gap_matches(old_by_day[90][0] + 1.0e-6, old_by_day[90][0]):
        raise SystemExit("STOP planted 1e-6 Sv old-gap receipt violation did not fire")
    print("[CONTROL PASS] planted +1e-6 Sv old-gap receipt violation rejected")

    results = []
    for day in DAYS:
        nemo = B.nemo_state(0, day)
        R._reducer_plants(nemo)
        old_gap, old_rows = _gap(args.old_artifact, day, nemo, historical=True)
        current_gap, current_rows = _gap(args.current, day, nemo)
        een_off_gap, een_off_rows = _gap(args.een_off, day, nemo, historical=True)
        een_nemo_gap, een_nemo_rows = _gap(args.een_nemo, day, nemo, historical=True)
        expected_old, floor = old_by_day[day]
        if not _old_gap_matches(old_gap, expected_old):
            raise SystemExit(f"STOP day-{day} old gap {old_gap:.17g} differs from JSON "
                             f"{expected_old:.17g} by more than {OLD_REPRO_TOL:.1e} Sv")
        print(f"[CONTROL PASS] day-{day} old gap reproduces JSON: "
              f"difference={old_gap - expected_old:.17g} Sv")
        epoch = current_gap - old_gap
        een = een_nemo_gap - een_off_gap
        remainder = epoch - een
        omega_scale = abs(old_gap) * abs(OMEGA_NEMO / OMEGA_ROUNDED - 1.0)
        item = {
            "day": day, "old_gap_sv": old_gap, "current_gap_sv": current_gap,
            "epoch_shift_sv": epoch, "een_off_gap_sv": een_off_gap,
            "een_nemo_gap_sv": een_nemo_gap, "een_shift_sv": een,
            "remainder_sv": remainder,
            "een_fraction_of_epoch": een / epoch if epoch else float("nan"),
            "old_floor_sv": floor, "two_old_floor_sv": 2.0 * floor,
            "omega_direct_scale_sv": omega_scale,
            "remainder_over_omega_direct": abs(remainder) / omega_scale,
            "epoch_rows_sv": (current_rows - old_rows).tolist(),
            "een_rows_sv": (een_nemo_rows - een_off_rows).tolist(),
            "remainder_rows_sv": ((current_rows - old_rows)
                                  - (een_nemo_rows - een_off_rows)).tolist(),
            "epoch_G4_sv": float((current_rows - old_rows)[1:5].sum()),
            "een_G4_sv": float((een_nemo_rows - een_off_rows)[1:5].sum()),
        }
        results.append(item)
        print("[CONFIRMED measurement] " + json.dumps(item, sort_keys=True))

    day90 = results[-1]
    fraction = day90["een_fraction_of_epoch"]
    same_sign = np.sign(day90["een_shift_sv"]) == np.sign(day90["epoch_shift_sv"])
    if same_sign and 0.8 <= fraction <= 1.2:
        een_label = "CONFIRMED_DOMINANT"
    elif not same_sign or abs(fraction) < 0.2:
        een_label = "REFUTED_DOMINANT"
    else:
        een_label = "PARTIAL"
    floor_owned = abs(day90["remainder_sv"]) <= day90["two_old_floor_sv"]
    rotation_linear = ("REFUTED_DIRECT_LINEAR" if
                       day90["remainder_over_omega_direct"] > 100.0 else
                       "PLAUSIBLE_DIRECT_LINEAR")
    verdict = {"een_epoch_ownership": een_label,
               "owned_to_old_floor": bool(floor_owned),
               "rotation_direct_linear": rotation_linear,
               "nonlinear_rotation": "UNRESOLVED_OFFLINE"}
    print("[VERDICT] " + json.dumps(verdict, sort_keys=True))
    output = {"measurement_label": "CONFIRMED", "results": results,
              "verdict": verdict, "git_sha": git_sha,
              "probe_sha256": _sha256(__file__), "inputs": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"[PROVENANCE] wrote={Path(args.out).resolve()} sha256={_sha256(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
