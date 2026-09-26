#!/usr/bin/env python3
"""Dump-only four-axis oracle-substitution matrix for row-1.1 V forcing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import split_explicit_momentum_chain_round2 as round2  # noqa: E402

AXES = (
    "lateral_friction",
    "vorticity",
    "cor_removal",
    "pressure_gradient",
)
ORDERED = (
    "kinetic_energy_gradient",
    "vertical_advection",
    "vorticity",
    "lateral_friction",
    "pressure_gradient",
    "cor_removal",
    "drag",
    "wind",
)
BITS = tuple(f"{value:04b}" for value in range(16))
LOCKED_TOL = 1.0e-12
LINEARITY_BAR = 1.0e-12
ENERGY_IDENTITY_BAR = 1.0e-12
ENERGY_BOUND = 0.01
TERM_GAIN_BOUND = 0.10
ROUND4_SHA256 = "4539d9a7ec4e2f20a41226a5707268ebcd268f97f03f0e2d59901906775fc154"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _names(bits: str) -> list[str]:
    return [name for name, bit in zip(AXES, bits) if bit == "1"]


def _subsets(mask: int):
    subset = mask
    while subset:
        yield subset
        subset = (subset - 1) & mask


def _label(value: float) -> str:
    if value >= ENERGY_BOUND:
        return "OWNED_POSITIVE_CONTRIBUTOR"
    if value <= -ENERGY_BOUND:
        return "OWNED_CANCELLER"
    return "BOUNDED_BELOW_OWNERSHIP_REMAINDER"


def _find_total_call(calls, target):
    target = {key: value for key, value in target.items() if key != "campaign_gate"}
    matches = [
        call
        for call in calls
        if {
            key: value
            for key, value in call["metric"].items()
            if key != "campaign_gate"
        }
        == target
    ]
    if len(matches) != 1:
        raise SystemExit(f"V total-field interception changed: {len(matches)}")
    return matches[0]


def _matrix(
    term_calls: dict[str, dict[str, Any]],
    total_call: dict[str, Any],
    round4: dict[str, Any],
    round2_artifact: dict[str, Any],
) -> dict[str, Any]:
    mask = term_calls[AXES[0]]["mask"]
    expected_n = term_calls[AXES[0]]["expected_n"]
    r0_full = term_calls[AXES[0]]["residual"]
    r0 = np.asarray(r0_full)[mask]
    if expected_n != round2.EXPECTED_V or r0.size != round2.EXPECTED_V:
        raise SystemExit("V population changed")
    if not np.array_equal(
        np.asarray(total_call["lego"]) - np.asarray(total_call["nemo"]),
        r0_full,
    ):
        raise SystemExit("V total field and captured residual differ")
    full_errors = {
        name: np.asarray(call["lego"]) - np.asarray(call["nemo"])
        for name, call in term_calls.items()
    }
    errors = {name: value[mask] for name, value in full_errors.items()}
    r0_rms = _rms(r0)
    arms: dict[str, dict[str, Any]] = {}
    residuals: dict[int, np.ndarray] = {}
    benefits: dict[int, float] = {}
    for bits in BITS:
        integer = int(bits, 2)
        names = _names(bits)
        correction = sum((errors[name] for name in names), np.zeros_like(r0))
        residual = r0 - correction
        residuals[integer] = residual
        score = (
            {
                "correlation_with_total_residual": None,
                "rms_gain": 0.0,
                "residual_removal": 0.0,
                "verdict": "BASELINE",
            }
            if integer == 0
            else round2._attribute_error(correction, r0)
        )
        q = _rms(residual) ** 2 / r0_rms**2
        benefits[integer] = 1.0 - q
        model_arm = np.asarray(total_call["lego"]).copy()
        for name in names:
            model_arm -= full_errors[name]
        campaign = round2.round1._metric(
            model_arm, total_call["nemo"], mask, expected_n
        )
        campaign["campaign_gate"] = round2.round1._classify_metric(campaign, None)
        arms[bits] = {
            "substitutions": names,
            "normalized_residual_rms": _rms(residual) / r0_rms,
            "normalized_squared_residual_energy": q,
            "squared_energy_benefit": benefits[integer],
            "attribution": score,
            "campaign_metric": campaign,
        }

    locked = {}
    for bits3 in round4["bits_order"]:
        bits4 = bits3 + "0"
        prior = round4["components"]["v"]["arms"][bits3]["attribution"][
            "residual_removal"
        ]
        actual = arms[bits4]["attribution"]["residual_removal"]
        locked[bits4] = {"prior": prior, "actual": actual, "delta": actual - prior}
    pressure_prior = round2_artifact["faithful_term_metrics"]["v"][
        "pressure_gradient"
    ]["attribution"]["residual_removal"]
    pressure_actual = arms["0001"]["attribution"]["residual_removal"]
    locked["pressure_singleton"] = {
        "prior": pressure_prior,
        "actual": pressure_actual,
        "delta": pressure_actual - pressure_prior,
    }
    if any(abs(item["delta"]) > LOCKED_TOL for item in locked.values()):
        raise SystemExit(f"locked V corner changed: {locked}")

    mobius: dict[int, float] = {}
    for size in range(1, len(AXES) + 1):
        for integer in range(1, 16):
            if integer.bit_count() != size:
                continue
            lower = sum(
                mobius[subset]
                for subset in _subsets(integer)
                if subset != integer
            )
            mobius[integer] = benefits[integer] - lower
    shapley = {}
    for axis_index, name in enumerate(AXES):
        bit = 1 << (len(AXES) - 1 - axis_index)
        shapley[name] = sum(
            value / integer.bit_count()
            for integer, value in mobius.items()
            if integer & bit
        )
    full_benefit = benefits[15]
    identities = {
        "mobius_sum_minus_full_benefit": sum(mobius.values()) - full_benefit,
        "shapley_sum_minus_full_benefit": sum(shapley.values()) - full_benefit,
    }
    if any(abs(value) > ENERGY_IDENTITY_BAR for value in identities.values()):
        raise SystemExit(f"V energy identity failed: {identities}")

    interactions = {}
    field_linearity = {}
    for integer, value in mobius.items():
        bits = f"{integer:04b}"
        size = integer.bit_count()
        if size < 2:
            continue
        contrast = np.zeros_like(r0)
        subset = integer
        while True:
            sign = -1.0 if (size - subset.bit_count()) % 2 else 1.0
            contrast += sign * residuals[subset]
            if subset == 0:
                break
            subset = (subset - 1) & integer
        normalized = _rms(contrast) / r0_rms
        field_linearity[bits] = normalized
        interactions[bits] = {
            "terms": _names(bits),
            "order": size,
            "value": value,
            "label": "MATERIAL" if abs(value) >= ENERGY_BOUND else "BOUNDED",
        }
    if any(value > LINEARITY_BAR for value in field_linearity.values()):
        raise SystemExit(f"V field interaction breached linearity: {field_linearity}")

    full_score = arms["1111"]["attribution"]
    group_confirms = (
        full_score["verdict"] == "CONFIRMS_CARRY"
        and arms["1111"]["normalized_squared_residual_energy"] <= ENERGY_BOUND
    )
    term_dispositions = {
        name: (
            _label(value)
            if group_confirms
            else "UNRESOLVED_IN_COMPOSITE"
        )
        for name, value in shapley.items()
    }
    noncandidate = {}
    for name in ORDERED:
        if name in AXES:
            continue
        gain = _rms(errors[name]) / r0_rms
        bounded = gain <= TERM_GAIN_BOUND
        noncandidate[name] = {
            "rms_gain": gain,
            "bound": TERM_GAIN_BOUND,
            "bounded": bounded,
            "disposition": (
                "BOUNDED_MAGNITUDE" if bounded else "UNBOUNDED_REMAINDER"
            ),
        }
        term_dispositions[name] = noncandidate[name]["disposition"]
    all_error = sum((errors[name] for name in ORDERED), np.zeros_like(r0))
    assembly_remainder = r0 - all_error
    remainder_gain = _rms(assembly_remainder) / r0_rms
    remainder_bounded = remainder_gain <= TERM_GAIN_BOUND
    noncandidate["assembly_remainder"] = {
        "rms_gain": remainder_gain,
        "bound": TERM_GAIN_BOUND,
        "bounded": remainder_bounded,
        "disposition": (
            "BOUNDED_MAGNITUDE"
            if remainder_bounded
            else "UNBOUNDED_REMAINDER"
        ),
    }
    term_dispositions["assembly_remainder"] = noncandidate[
        "assembly_remainder"
    ]["disposition"]
    fully_disposed = group_confirms and all(
        item["bounded"] for item in noncandidate.values()
    )
    return {
        "component": "v",
        "sample_count": int(r0.size),
        "arms": arms,
        "locked_corners": locked,
        "mobius": {
            f"{integer:04b}": {
                "terms": _names(f"{integer:04b}"),
                "order": integer.bit_count(),
                "value": value,
                "label": (
                    "MATERIAL" if abs(value) >= ENERGY_BOUND else "BOUNDED"
                ),
            }
            for integer, value in mobius.items()
        },
        "interactions": interactions,
        "shapley": {
            name: {
                "value": value,
                "label": (
                    _label(value)
                    if group_confirms
                    else "DESCRIPTIVE_UNADMITTED_GROUP_UNRESOLVED"
                ),
            }
            for name, value in shapley.items()
        },
        "energy_identities": identities,
        "field_linearity_normalized_rms": field_linearity,
        "group_confirms": group_confirms,
        "noncandidate_bounds": noncandidate,
        "term_dispositions": term_dispositions,
        "assembly_remainder": {
            "normalized_rms": remainder_gain,
            "max_abs": float(np.max(np.abs(assembly_remainder))),
            "bounded": remainder_bounded,
        },
        "fully_disposed": fully_disposed,
    }


def _controls() -> dict[str, bool]:
    phase = 2.0 * np.pi * np.arange(1024, dtype=np.float64) / 1024.0
    residual = np.sin(phase)
    confirm = round2._attribute_error(residual, residual)
    refute = round2._attribute_error(np.cos(phase), residual)
    values = {
        "locked_corner_plant_rejected": 2.0 * LOCKED_TOL > LOCKED_TOL,
        "field_interaction_plant_rejected": 2.0 * LINEARITY_BAR > LINEARITY_BAR,
        "positive_label_reached": _label(0.02) == "OWNED_POSITIVE_CONTRIBUTOR",
        "canceller_label_reached": _label(-0.02) == "OWNED_CANCELLER",
        "bounded_label_reached": _label(0.005)
        == "BOUNDED_BELOW_OWNERSHIP_REMAINDER",
        "unadmitted_label_distinct": (
            "DESCRIPTIVE_UNADMITTED_GROUP_UNRESOLVED" != _label(0.02)
        ),
        "group_confirm_reached": confirm["verdict"] == "CONFIRMS_CARRY",
        "group_refute_reached": refute["verdict"] == "REFUTES_CARRY",
        "remainder_0p11_rejected": not (0.11 <= TERM_GAIN_BOUND),
    }
    if not all(values.values()):
        raise SystemExit(f"round5 control failed: {values}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty_before = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    if dirty_before:
        raise SystemExit("clean worktree required:\n" + dirty_before)
    if jax.default_backend() != "cpu" or not bool(jax.config.jax_enable_x64):
        raise SystemExit("CPU + JAX x64 are required")

    round4_path = repo / (
        "docs/ocean/fidelity/dino_split_explicit_momentum_chain_round4_artifact.json"
    )
    if _sha256(round4_path) != ROUND4_SHA256:
        raise SystemExit("accepted round4 artifact hash changed")
    round4 = json.loads(round4_path.read_text())
    if not round4["components"]["u"]["fully_disposed"]:
        raise SystemExit("accepted round4 U disposition is not closed")
    round2_path = repo / (
        "docs/ocean/fidelity/dino_split_explicit_momentum_chain_round2_artifact.json"
    )
    round2_artifact = json.loads(round2_path.read_text())

    metric_calls: list[dict[str, Any]] = []
    total_calls: list[dict[str, Any]] = []
    real_term_metric = round2._term_metric
    real_total_metric = round2.round1._metric

    def capture_term(lego, nemo, mask, expected_n, total_residual):
        metric = real_term_metric(lego, nemo, mask, expected_n, total_residual)
        metric_calls.append({
            "lego": np.asarray(lego).copy(),
            "nemo": np.asarray(nemo).copy(),
            "mask": np.asarray(mask, dtype=bool).copy(),
            "expected_n": int(expected_n),
            "residual": np.asarray(total_residual).copy(),
            "metric": metric,
        })
        return metric

    def capture_total(lego, nemo, mask, expected_n):
        metric = real_total_metric(lego, nemo, mask, expected_n)
        if np.asarray(lego).ndim == 2 and int(np.asarray(mask).sum()) in (
            round2.EXPECTED_U,
            round2.EXPECTED_V,
        ):
            total_calls.append({
                "lego": np.asarray(lego).copy(),
                "nemo": np.asarray(nemo).copy(),
                "mask": np.asarray(mask, dtype=bool).copy(),
                "expected_n": int(expected_n),
                "metric": dict(metric),
            })
        return metric

    round2._term_metric = capture_term
    round2.round1._metric = capture_total
    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory(prefix="dino-split-r5-") as tmpdir:
            base_path = Path(tmpdir) / "round2.json"
            sys.argv = [str(Path(round2.__file__).resolve()), "--output", str(base_path)]
            status = round2.main()
            if status != 0:
                raise SystemExit(f"round-2 base probe returned {status}")
            base = json.loads(base_path.read_text())
            base_sha256 = _sha256(base_path)
    finally:
        sys.argv = old_argv
        round2._term_metric = real_term_metric
        round2.round1._metric = real_total_metric
    if len(metric_calls) != 34:
        raise SystemExit(f"round-2 term capture sequence changed: {len(metric_calls)}")
    calls = {
        name: metric_calls[19 + 2 * index]
        for index, name in enumerate(ORDERED)
    }
    for name, call in calls.items():
        if call["metric"] != base["faithful_term_metrics"]["v"][name]:
            raise SystemExit(f"captured V metric changed at {name}")
    total = _find_total_call(total_calls, base["total_metrics"]["faithful_v"])
    matrix = _matrix(calls, total, round4, round2_artifact)
    controls = _controls()

    dirty_after = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    if dirty_after:
        raise SystemExit("worktree changed during measurement:\n" + dirty_after)
    prereg = repo / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round5.md"
    provenance = {
        "probe": Path(__file__).resolve(),
        "preregistration": prereg,
        "round2_probe": Path(round2.__file__).resolve(),
        "round2_artifact": round2_path,
        "round4_artifact": round4_path,
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round5-v1",
        "session_id": os.environ.get("CODEX_SESSION_ID", "unset"),
        "git": {"commit": head, "clean_before": True, "clean_after": True},
        "runtime": base["runtime"],
        "lane": base["lane"],
        "kt": base["kt"],
        "matrix_axes": list(AXES),
        "bits_order": list(BITS),
        "v": matrix,
        "round4_u_fully_disposed": True,
        "row_1_1_fully_disposed": bool(matrix["fully_disposed"]),
        "controls": {"matrix": controls, "round2": base["controls"]},
        "base_round2_artifact_sha256": base_sha256,
        "dump_sha256": base["dump_sha256"],
        "run_input_sha256": base["run_input_sha256"],
        "provenance_sha256": {
            name: _sha256(path) for name, path in provenance.items()
        },
        "slot": "NOT_ALLOCATED_EXISTING_DUMPS_ONLY",
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    full = matrix["arms"]["1111"]
    print(
        "V 1111: removal="
        f"{full['attribution']['residual_removal']:.12f} "
        f"Q={full['normalized_squared_residual_energy']:.12e} "
        f"{full['attribution']['verdict']} "
        f"fully_disposed={matrix['fully_disposed']}"
    )
    for name, item in matrix["shapley"].items():
        print(f"  {name}: phi={item['value']:+.12f} {item['label']}")
    print(f"row_1_1_fully_disposed={receipt['row_1_1_fully_disposed']}")
    print(f"artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
