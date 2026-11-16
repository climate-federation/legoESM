#!/usr/bin/env python3
"""Compare two legoESM ORCA2 year arms on daily temperature metrics.

The loaders and area weights are the existing round-342/Decision-160 metric
card.  Surface temperature uses its area-weighted RMS reduction.  The 3-D wet
temperature RMS is the existing round-343 temperature scorer.  Optional raw
snapshot comparison covers every common array and treats ``-0.0`` and ``+0.0``
as different, as required by a bitwise comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in reversed(
    (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")
):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round342_s4_year_metrics as r342_metrics,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round343_speed_benchmark as r343_benchmark,
)

pilot_metrics = r342_metrics.pilot_metrics
require = pilot_metrics.require
STEPS_PER_DAY = pilot_metrics.STEPS_PER_DAY


def parse_steps(value: str) -> tuple[int, ...]:
    """Parse a comma-separated, strictly increasing positive step list."""
    try:
        steps = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from exc
    if not steps or any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("steps must be positive")
    if tuple(sorted(set(steps))) != steps:
        raise argparse.ArgumentTypeError("steps must be unique and increasing")
    return steps


def compare_snapshot_data(
    reference_root: Path,
    candidate_root: Path,
    steps: tuple[int, ...],
    *,
    plant: str | None = None,
) -> dict[str, Any]:
    """Compare every NPZ field by dtype, shape, and raw C-order data bytes."""
    first_difference: dict[str, Any] | None = None
    differing_snapshots = 0
    compared_arrays = 0
    inventories: list[list[str]] = []
    for step in steps:
        name = f"candidate_step{step:08d}.npz"
        reference_path = reference_root / name
        candidate_path = candidate_root / name
        require(reference_path.is_file(), f"missing reference snapshot {reference_path}")
        require(candidate_path.is_file(), f"missing candidate snapshot {candidate_path}")
        snapshot_differs = False
        with np.load(reference_path, allow_pickle=False) as reference, np.load(
            candidate_path, allow_pickle=False
        ) as candidate:
            require(
                set(reference.files) == set(candidate.files),
                f"step {step}: snapshot field inventories differ",
            )
            fields = sorted(reference.files)
            inventories.append(fields)
            for field in fields:
                left = np.asarray(reference[field])
                right = np.asarray(candidate[field])
                require(left.shape == right.shape, f"step {step} {field}: shape differs")
                require(left.dtype == right.dtype, f"step {step} {field}: dtype differs")
                require(left.dtype.kind != "O", f"step {step} {field}: object dtype refused")
                compared_arrays += 1
                left_bytes = np.ascontiguousarray(left).view(np.uint8).reshape(-1)
                right_bytes = np.ascontiguousarray(right).view(np.uint8).reshape(-1)
                if plant == "flip-first-bit" and step == steps[0] and field == fields[0]:
                    right_bytes = right_bytes.copy()
                    right_bytes[0] ^= np.uint8(1)
                unequal_bytes = np.flatnonzero(left_bytes != right_bytes)
                if unequal_bytes.size == 0:
                    continue
                snapshot_differs = True
                if first_difference is None:
                    first_difference = {
                        "step": step,
                        "field": field,
                        "first_unequal_byte": int(unequal_bytes[0]),
                        "unequal_bytes": int(unequal_bytes.size),
                    }
        differing_snapshots += int(snapshot_differs)
    require(
        all(inventory == inventories[0] for inventory in inventories),
        "snapshot field inventory changed across saved steps",
    )
    return {
        "status": "EXACT" if first_difference is None else "DIFFERENT",
        "signed_zero_policy": "DISTINCT_RAW_BITS",
        "steps": list(steps),
        "snapshot_count": len(steps),
        "compared_arrays": compared_arrays,
        "fields": inventories[0],
        "differing_snapshots": differing_snapshots,
        "first_difference": first_difference,
    }


def planted_metric_controls() -> dict[str, float]:
    """Prove that area weighting and the wet mask both affect their scores."""
    difference = np.zeros((148, 180, 30), dtype=np.float64)
    wet = np.ones_like(difference, dtype=bool)
    difference[..., 0] = 0.01
    area = np.ones((148, 180), dtype=np.float64)
    area[0, 0] = 3.0
    difference[0, 0, 0] = 0.02
    area_rms = pilot_metrics.bias_rms(difference[..., 0], area)["rms"]
    unweighted_rms = float(np.sqrt(np.mean(np.square(difference[..., 0]))))
    require(area_rms != unweighted_rms, "area-weight plant did not move SST RMS")

    wet[0, 0, :] = False
    wet_score = r343_benchmark.score_daily_temperature_against_reference(
        difference, np.zeros_like(difference), wet, kt=STEPS_PER_DAY
    )["t3d_rms_wet_c"]
    all_wet_score = r343_benchmark.score_daily_temperature_against_reference(
        difference,
        np.zeros_like(difference),
        np.ones_like(wet),
        kt=STEPS_PER_DAY,
    )["t3d_rms_wet_c"]
    require(wet_score != all_wet_score, "wet-mask plant did not move 3-D T RMS")
    return {
        "area_weighted_sst_rms_plant": area_rms,
        "unweighted_sst_rms_control": unweighted_rms,
        "wet_t3d_rms_plant": wet_score,
        "all_cell_t3d_rms_control": all_wet_score,
    }


def compare_daily_temperature(
    reference_root: Path,
    candidate_root: Path,
    geometry_root: Path,
    days: int,
    *,
    plant: str | None = None,
) -> list[dict[str, Any]]:
    """Compute daily area-weighted SST RMS and unweighted 3-D wet T RMS."""
    require(1 <= days <= 365, f"days must be in 1..365, got {days}")
    require(
        plant in (None, "first-wet-temperature"),
        f"unknown daily-temperature plant {plant!r}",
    )
    geometry = pilot_metrics.static_geometry(geometry_root)
    wet3 = np.asarray(geometry["tmask"], dtype=bool)
    surface_weight = np.asarray(geometry["area_t"]) * wet3[..., 0]
    rows: list[dict[str, Any]] = []
    for day in range(1, days + 1):
        reference = pilot_metrics.read_candidate_day(reference_root, day)
        candidate = pilot_metrics.read_candidate_day(candidate_root, day)
        candidate_t = np.asarray(candidate["T"])
        reference_t = np.asarray(reference["T"])
        if plant == "first-wet-temperature" and day == 1:
            candidate_t = candidate_t.copy()
            surface_wet = np.argwhere(wet3[..., 0])
            require(surface_wet.size > 0, "daily-temperature plant has no wet surface cell")
            j, i = (int(value) for value in surface_wet[0])
            candidate_t[j, i, 0] = reference_t[j, i, 0] + 1.0e6
        score = r343_benchmark.score_daily_temperature_against_reference(
            candidate_t, reference_t, wet3, kt=day * STEPS_PER_DAY
        )
        sst = pilot_metrics.bias_rms(
            candidate_t[..., 0] - reference_t[..., 0],
            surface_weight,
        )
        rows.append(
            {
                "day": day,
                "kt": day * STEPS_PER_DAY,
                "sst_area_weighted_rms_k": float(sst["rms"]),
                "t3d_wet_rms_k": float(score["t3d_rms_wet_c"]),
            }
        )
    return rows


def planted_daily_temperature_control(
    reference_root: Path,
    candidate_root: Path,
    geometry_root: Path,
    baseline: dict[str, Any],
) -> dict[str, float]:
    """Prove the production daily-temperature comparison detects a plant."""
    planted = compare_daily_temperature(
        reference_root,
        candidate_root,
        geometry_root,
        1,
        plant="first-wet-temperature",
    )[0]
    changed = {
        name: float(planted[name]) != float(baseline[name])
        for name in ("sst_area_weighted_rms_k", "t3d_wet_rms_k")
    }
    require(all(changed.values()), "daily-temperature plant did not move both scores")
    return {
        "daily_sst_rms_plant": float(planted["sst_area_weighted_rms_k"]),
        "daily_t3d_rms_plant": float(planted["t3d_wet_rms_k"]),
    }


def write_curve(rows: list[dict[str, Any]], output_path: Path) -> str:
    """Write the requested two-metric daily curve and return its SHA-256."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = [row["day"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(days, [row["sst_area_weighted_rms_k"] for row in rows])
    axes[0].set_ylabel("SST area-weighted RMS [K]")
    axes[1].plot(days, [row["t3d_wet_rms_k"] for row in rows])
    axes[1].set_ylabel("3-D wet T RMS [K]")
    axes[1].set_xlabel("day")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def compare_manifest_timing(
    reference_path: Path,
    candidate_path: Path,
    *,
    reference_root: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    """Compare timing, accepting certified pre-r348 reference as implicit libm.

    The round-342 certified reference predates the ``transcendentals`` field.
    That lane unconditionally selected libm, so only that reference manifest
    may omit the field; candidate manifests must state their mode explicitly.
    """
    require(
        reference_path.resolve() != candidate_path.resolve(),
        "reference and candidate manifest paths must differ",
    )
    manifests = []
    for index, (path, root) in enumerate(
        ((reference_path, reference_root), (candidate_path, candidate_root))
    ):
        require(path.is_file(), f"missing run manifest {path}")
        require(
            path.resolve().parent == root.resolve(),
            f"{path}: manifest must belong to run root {root}",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(int(payload["steps"]) > 0, f"{path}: steps must be positive")
        require(float(payload["wall_seconds"]) > 0.0, f"{path}: wall_seconds must be positive")
        require(
            payload.get("jax_default_backend") == "cpu",
            f"{path}: manifest backend must be 'cpu'",
        )
        mode = payload.get("transcendentals")
        if mode is None:
            require(
                index == 0
                and payload.get("round") == "orca2_r342"
                and payload.get("status") == "PASS_R342_S4_K0_LEGOESM_RUN",
                f"{path}: only the certified pre-r348 round-342 reference may "
                "omit transcendentals",
            )
            mode = "libm"
            print(
                "R348_REFERENCE_MANIFEST_LEGACY_MODE transcendentals=libm "
                "rule=certified-pre-r348-round342-lane-used-libm"
            )
        require(
            mode in ("libm", "native"),
            f"{path}: manifest transcendentals must be 'libm' or 'native'",
        )
        payload["transcendentals"] = mode
        manifests.append(payload)
    reference_s = float(manifests[0]["wall_seconds"]) / int(manifests[0]["steps"])
    candidate_s = float(manifests[1]["wall_seconds"]) / int(manifests[1]["steps"])
    return {
        "reference_seconds_per_step": reference_s,
        "candidate_seconds_per_step": candidate_s,
        "candidate_over_reference": candidate_s / reference_s,
        "speedup": reference_s / candidate_s,
        "reference_backend": manifests[0]["jax_default_backend"],
        "candidate_backend": manifests[1]["jax_default_backend"],
        "reference_transcendentals": manifests[0]["transcendentals"],
        "candidate_transcendentals": manifests[1]["transcendentals"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--bitwise-steps", type=parse_steps)
    parser.add_argument("--require-bitwise-exact", action="store_true")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    args = parser.parse_args(argv)
    require(
        (args.reference_manifest is None) == (args.candidate_manifest is None),
        "reference and candidate manifests must be provided together",
    )
    require(
        args.reference_root.resolve() != args.candidate_root.resolve(),
        "reference_root and candidate_root must differ",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = compare_daily_temperature(
        args.reference_root, args.candidate_root, args.geometry_root, args.days
    )
    figure = args.output_dir / "transcendentals_arm_temperature_rms.png"
    figure_sha = write_curve(rows, figure)
    bitwise = None
    if args.bitwise_steps is not None:
        bitwise = compare_snapshot_data(
            args.reference_root, args.candidate_root, args.bitwise_steps
        )
    timing = (
        None
        if args.reference_manifest is None
        else compare_manifest_timing(
            args.reference_manifest,
            args.candidate_manifest,
            reference_root=args.reference_root,
            candidate_root=args.candidate_root,
        )
    )
    report = {
        "format": "orca2-r348-transcendentals-arm-comparison-v1",
        "reference_root": str(args.reference_root),
        "candidate_root": str(args.candidate_root),
        "geometry_root": str(args.geometry_root),
        "days": args.days,
        "metrics": {
            "sst": "Decision-160 area_t * surface tmask weighted RMS",
            "t3d": "Round-343 unweighted RMS over wet 3-D tmask cells",
        },
        "planted_controls": {
            **planted_metric_controls(),
            **planted_daily_temperature_control(
                args.reference_root,
                args.candidate_root,
                args.geometry_root,
                rows[0],
            ),
        },
        "daily": rows,
        "figure": {"path": str(figure), "sha256": figure_sha},
        "bitwise": bitwise,
        "timing": timing,
        "worktree": worktree_stamp(allow_dirty=False),
    }
    report_path = args.output_dir / "transcendentals_arm_temperature_rms.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.require_bitwise_exact and (bitwise is None or bitwise["status"] != "EXACT"):
        print("REFUSE_R348_BITWISE_NOT_EXACT", file=sys.stderr)
        return 1
    if args.require_bitwise_exact:
        print(
            f"PASS_R348_BITWISE_EXACT snapshots={bitwise['snapshot_count']} "
            f"arrays={bitwise['compared_arrays']}"
        )
    if timing is not None:
        print(
            "R348_MANIFEST_MODES "
            f"reference_backend={timing['reference_backend']} "
            f"reference_transcendentals={timing['reference_transcendentals']} "
            f"candidate_backend={timing['candidate_backend']} "
            f"candidate_transcendentals={timing['candidate_transcendentals']}"
        )
    print(f"PASS_R348_ARM_COMPARISON days={args.days} output={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
