#!/usr/bin/env python3
"""Round-151 GYRE amplification-threshold member runner and scorer.

This file never copies the year integrator.  Member mode imports and calls the
certified ``year_fromrest.run_member`` after changing only its instrument's
absolute perturbation amplitude in the current process.  Score mode admits the
resulting manifests and compares their T3D snapshots with the landed control.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


AMPLITUDES_K = (1.0e-8, 1.0e-6, 1.0e-4)
AMPLITUDE_TAGS = {
    1.0e-8: "amp1e-8",
    1.0e-6: "amp1e-6",
    1.0e-4: "amp1e-4",
}
SCORED_DAYS = (30, 60, 90, 120, 180, 240, 300, 360)
CONTROL_GAP_DAY240_K = 1.644671864406711e-2
AMPLIFICATION_FACTOR = 0.3
AMPLIFICATION_THRESHOLD_K = AMPLIFICATION_FACTOR * CONTROL_GAP_DAY240_K
ROUND129_FLOOR_DAY240_K = 2.0891293703252062e-10
SYSTEMATIC_CEILING_K = 1.0e-6
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round151")
DEFAULT_CONTROL = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round149/"
    "candidate_year/lego_seed0_year")
DEFAULT_NEMO = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest")


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def _year_module():
    path = Path(__file__).with_name(
        "nemo_testcase_l2_gyre_year_fromrest.py")
    spec = importlib.util.spec_from_file_location("round151_year", path)
    require(spec is not None and spec.loader is not None,
            f"cannot import certified year harness {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tag(amplitude_K: float) -> str:
    for registered, tag in AMPLITUDE_TAGS.items():
        if amplitude_K == registered:
            return tag
    raise GateError(
        f"amplitude {amplitude_K!r} is not registered; expected "
        f"{list(AMPLITUDES_K)}")


def run_member(amplitude_K: float, root: Path, expect_commit: str) -> int:
    year = _year_module()
    tag = _tag(amplitude_K)
    out = Path(root) / "members" / f"lego_seed1_{tag}"
    require(not out.exists() or not any(out.iterdir()),
            f"REFUSE stale amplitude-member output {out}")
    # The compiled Round-129 source uses 1e-10 here
    # (GYRE_OMIP_L2_P3_SM_YRPERT/.../usrdef_istate.f90:101-105).  The user
    # ordered these three amplitudes with every other operand held fixed.
    year.PERT_AMPLITUDE_K = amplitude_K
    return year.run_member(
        1, Path(root) / "members", days=year.YEAR_DAYS, tag=tag,
        snap_steps=year.STEPS_PER_DAY, expect_commit=expect_commit)


def _load_manifest(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    require(manifest_path.is_file(), f"missing {manifest_path}")
    return json.loads(manifest_path.read_text())


def _admit_member(path: Path, amplitude_K: float, year, *,
                  plant: str | None = None) -> dict:
    manifest = _load_manifest(path)
    tag = _tag(amplitude_K)
    observed_amplitude = float(
        manifest.get("perturbation", {}).get("absolute_amplitude_K", np.nan))
    if plant == "member-amplitude" and amplitude_K == AMPLITUDES_K[0]:
        observed_amplitude = np.nextafter(observed_amplitude, np.inf)
    require(manifest.get("format") ==
            "nemo-testcase-l2-gyre-year-fromrest-member-v1",
            f"{path}: wrong member format")
    require(manifest.get("case") == year.CASE and manifest.get("seed") == 1,
            f"{path}: not the registered GYRE seed-1 member")
    require(manifest.get("tag") == tag,
            f"{path}: tag {manifest.get('tag')!r} != {tag!r}")
    require(manifest.get("days") == 360 and manifest.get("steps") == 2160,
            f"{path}: not a complete 360-day/2160-step member")
    require(manifest.get("snapshot_step_interval") == 6,
            f"{path}: snapshots are not daily")
    require(observed_amplitude == amplitude_K,
            f"{path}: amplitude {observed_amplitude!r} != {amplitude_K!r}")
    require(manifest.get("worktree", {}).get("clean") is True,
            f"{path}: producer worktree was not clean")
    snapshots = sorted(path.glob("day*.npz"))
    require(len(snapshots) == 360,
            f"{path}: {len(snapshots)} daily snapshots != 360")
    return {
        "path": str(path),
        "manifest_sha256": year.sha256(path / "manifest.json"),
        "producer_commit": manifest["worktree"]["commit"],
        "phase3_gate_sha256": manifest["phase3_gate_sha256"],
        "requested_amplitude_K": amplitude_K,
        "measured_peak_K": manifest["perturbation"]["measured_peak_K"],
        "daily_snapshot_count": len(snapshots),
    }


def _load_temperature(path: Path, day: int) -> np.ndarray:
    snapshot = path / f"day{day:03d}.npz"
    require(snapshot.is_file(), f"missing {snapshot}")
    with np.load(snapshot) as handle:
        values = np.asarray(handle["T"], dtype=np.float64)
    require(np.all(np.isfinite(values)), f"{snapshot}: non-finite T")
    return values


def _initial_delta(amplitude_K: float, year, card, wet: np.ndarray) -> dict:
    depth = np.asarray(card.recipe.z_coord.nemo_gdept_0, dtype=np.float64)
    latitude = np.broadcast_to(
        np.asarray(card.recipe.grid.native_lat_T_deg,
                   dtype=np.float64)[..., None], depth.shape)
    saved_amplitude = year.PERT_AMPLITUDE_K
    year.PERT_AMPLITUDE_K = amplitude_K
    try:
        # Use the exact statement member mode used; post-hoc rescaling can
        # round differently from multiplying SIN by the requested amplitude.
        perturbation = year.nemo_istate_perturbation(
            depth, latitude, wet.astype(np.float64), 1)
    finally:
        year.PERT_AMPLITUDE_K = saved_amplitude
    base = np.asarray(card.recipe.initial_state.T.data, dtype=np.float64)
    applied = (base + perturbation) - base
    require(int(np.count_nonzero(applied[~wet])) == 0,
            "initial perturbation changed a dry cell")
    require(int(np.count_nonzero(applied[wet])) > 0,
            "initial perturbation changed no wet temperature cell")
    return {
        "pattern": ("sin(NINT(depth)*73 + NINT(latitude*1000)*179 + "
                    "seed1*997) * tmask"),
        "requested_amplitude_K": amplitude_K,
        "wet_temperature_cells": int(np.count_nonzero(wet)),
        "wet_temperature_cells_unequal": int(np.count_nonzero(applied[wet])),
        "applied_peak_K": float(np.max(np.abs(applied[wet]))),
        "other_initial_fields_unequal": {"S": 0, "u": 0, "v": 0,
                                           "ssh": 0},
    }


def _classify(day240: dict[str, float]) -> str:
    amplified = any(day240[tag] >= AMPLIFICATION_THRESHOLD_K
                    for tag in (AMPLITUDE_TAGS[1.0e-6],
                                AMPLITUDE_TAGS[1.0e-4]))
    small_below = day240[AMPLITUDE_TAGS[1.0e-8]] < AMPLIFICATION_THRESHOLD_K
    if amplified and small_below:
        return "THRESHOLD_AMPLIFICATION"
    if all(value < SYSTEMATIC_CEILING_K for value in day240.values()):
        return "SYSTEMATIC_TRACER_STEP_OWNER"
    return "INCONCLUSIVE_UNDER_REGISTERED_DISCRIMINATOR"


def score(root: Path, control: Path, nemo_root: Path, *,
          plant: str | None = None) -> dict:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    import jax

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"] is True,
            "scorer worktree is dirty; evidence requires a committed gate")
    year = _year_module()
    card = build_nemo_testcase_card(year.CASE)
    _, wet3, _, _, _, _, _ = year._geometry(card, year.DEFAULT_NEMO_MESH)
    require(int(np.count_nonzero(wet3)) == 18000,
            "registered wet-cell population moved")

    control_manifest = _load_manifest(control)
    require(control_manifest.get("seed") == 0
            and control_manifest.get("days") == 360
            and control_manifest.get("snapshot_step_interval") == 6,
            f"{control}: not the admitted daily member-0 year")
    require(control_manifest.get("worktree", {}).get("clean") is True,
            f"{control}: producer worktree was not clean")

    admissions, initial = {}, {}
    for amplitude in AMPLITUDES_K:
        tag = _tag(amplitude)
        path = Path(root) / "members" / f"lego_seed1_{tag}"
        admissions[tag] = _admit_member(
            path, amplitude, year, plant=plant)
        initial[tag] = _initial_delta(amplitude, year, card, wet3)
        require(admissions[tag]["phase3_gate_sha256"] ==
                control_manifest["phase3_gate_sha256"],
                f"{tag}: stepping-gate hash differs from control")

    registered_days = list(SCORED_DAYS)
    if plant == "day-registry":
        registered_days.remove(240)
    require(tuple(registered_days) == SCORED_DAYS,
            f"day registry {registered_days} != {list(SCORED_DAYS)}")

    rows = {}
    for day in registered_days:
        baseline = _load_temperature(control, day)
        require(baseline.shape == wet3.shape,
                f"day {day}: control shape {baseline.shape} != {wet3.shape}")
        rows[str(day)] = {}
        for amplitude in AMPLITUDES_K:
            tag = _tag(amplitude)
            path = Path(root) / "members" / f"lego_seed1_{tag}"
            candidate = _load_temperature(path, day)
            require(candidate.shape == wet3.shape,
                    f"day {day} {tag}: shape {candidate.shape} != {wet3.shape}")
            rows[str(day)][tag] = year._rms(candidate - baseline, wet3)

    nemo = year._load_nemo(
        nemo_root, 0, 240, card.recipe.z_coord.n_levels)
    control_240 = _load_temperature(control, 240)
    measured_control_gap = year._rms(control_240 - nemo["T"], wet3)
    require(measured_control_gap == CONTROL_GAP_DAY240_K,
            f"control day-240 gap {measured_control_gap!r} != "
            f"{CONTROL_GAP_DAY240_K!r}")

    day240 = rows["240"]
    verdict = _classify(day240)
    return {
        "schema": "nemo-testcase-l2-gyre-round151-amplification-v1",
        "case": year.CASE,
        "metric": "unweighted fp64 T3D RMS over NEMO tmask",
        "precision": {"policy": str(get_policy()),
                      "platform": jax.default_backend()},
        "control": {
            "path": str(control),
            "manifest_sha256": year.sha256(control / "manifest.json"),
            "producer_commit": control_manifest["worktree"]["commit"],
            "day240_gap_to_nemo_K": measured_control_gap,
            "day240_gap_control_reproduced_bit_for_bit": True,
        },
        "amplitudes_K": list(AMPLITUDES_K),
        "days": list(SCORED_DAYS),
        "wet_cells": int(np.count_nonzero(wet3)),
        "member_admission": admissions,
        "initial_state_delta": initial,
        "rows": rows,
        "day240_verdict": {
            "spreads_K": day240,
            "amplification_factor": AMPLIFICATION_FACTOR,
            "amplification_threshold_K": AMPLIFICATION_THRESHOLD_K,
            "small_arm_over_round129_floor": (
                day240[AMPLITUDE_TAGS[1.0e-8]] /
                ROUND129_FLOOR_DAY240_K),
            "systematic_alternative_ceiling_K": SYSTEMATIC_CEILING_K,
            "verdict": verdict,
        },
        "worktree": stamp,
    }


def self_check() -> int:
    below = np.nextafter(AMPLIFICATION_THRESHOLD_K, -np.inf)
    above = np.nextafter(AMPLIFICATION_THRESHOLD_K, np.inf)
    base = {tag: 0.0 for tag in AMPLITUDE_TAGS.values()}
    exact = dict(base, **{AMPLITUDE_TAGS[1.0e-4]:
                          AMPLIFICATION_THRESHOLD_K})
    low = dict(base, **{AMPLITUDE_TAGS[1.0e-4]: below})
    high = dict(base, **{AMPLITUDE_TAGS[1.0e-4]: above})
    require(_classify(exact) == "THRESHOLD_AMPLIFICATION",
            "threshold equality did not pass")
    require(_classify(low) != "THRESHOLD_AMPLIFICATION",
            "adjacent value below threshold did not fail")
    require(_classify(high) == "THRESHOLD_AMPLIFICATION",
            "adjacent value above threshold did not pass")
    try:
        _tag(1.0e-5)
    except GateError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unregistered amplitude did not fail")
    print("SELF-CHECK OK: amplitude registry and exact binary64 threshold")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-amplitude", type=float)
    parser.add_argument("--expect-commit")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--nemo-root", type=Path, default=DEFAULT_NEMO)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--plant", choices=("member-amplitude",
                                             "day-registry"))
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            return self_check()
        if args.member_amplitude is not None:
            require(args.expect_commit is not None,
                    "--member-amplitude requires --expect-commit")
            require(not args.score and args.plant is None,
                    "member mode cannot be combined with score or plant")
            return run_member(args.member_amplitude, args.root,
                              args.expect_commit)
        require(args.score, "one of --member-amplitude/--score/--self-check is required")
        report = score(args.root, args.control, args.nemo_root,
                       plant=args.plant)
    except GateError as error:
        if args.plant is not None:
            print(f"REFUSE Round-151 {args.plant}: {error}", file=sys.stderr)
            print(f"STATUS PLANT-FIRED: {args.plant}")
            return 1
        raise
    if args.plant is not None:
        print(f"REFUSE Round-151 plant {args.plant} did not fire",
              file=sys.stderr)
        return 3
    target = args.json or (args.root / "amplification.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print("day  amp1e-8_K  amp1e-6_K  amp1e-4_K")
    for day in SCORED_DAYS:
        row = report["rows"][str(day)]
        print(f"{day:3d}  " + "  ".join(
            f"{row[AMPLITUDE_TAGS[amplitude]]:.12e}"
            for amplitude in AMPLITUDES_K))
    verdict = report["day240_verdict"]
    print(f"DAY240 threshold={verdict['amplification_threshold_K']:.12e} "
          f"verdict={verdict['verdict']}")
    print(f"STATUS {verdict['verdict']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as error:
        print(f"GATE ERROR: {error}", file=sys.stderr)
        sys.exit(2)
