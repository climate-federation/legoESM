#!/usr/bin/env python3
"""Dump-only 2^3 oracle-substitution matrix for row-1.1 forcing terms.

The matrix and ownership algebra are frozen in
``PREREG_split_explicit_momentum_chain_round4.md``.  This probe reruns the
committed round-2 loader, intercepts its validated raw operands, and never
builds or executes NEMO.
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
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import split_explicit_momentum_chain_round2 as round2  # noqa: E402

AXES = ("lateral_friction", "vorticity", "cor_removal")
BITS = ("000", "100", "010", "001", "110", "101", "011", "111")
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
LOCKED_U_REMOVAL = {
    "000": 0.0,
    "100": 0.6612415045308856,
    "010": -3.9651538510810838,
    "001": -3.8528100525195788,
    "110": -3.8067621779483583,
    "101": -3.8351151874225913,
    "011": 0.08652382878366294,
}
LOCKED_TOL = 1.0e-12
LINEARITY_BAR = 1.0e-12
ENERGY_IDENTITY_BAR = 1.0e-12
ENERGY_OWNERSHIP_BOUND = 0.01
TERM_GAIN_BOUND = 0.10
PRECEDENT_PREREG = "19acf16b2f0"
PRECEDENT_PROBE = "820e3500bf3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rms(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def _axis_names(bits: str) -> list[str]:
    return [name for name, bit in zip(AXES, bits) if bit == "1"]


def _ownership_label(value: float) -> str:
    if value >= ENERGY_OWNERSHIP_BOUND:
        return "OWNED_POSITIVE_CONTRIBUTOR"
    if value <= -ENERGY_OWNERSHIP_BOUND:
        return "OWNED_CANCELLER"
    return "BOUNDED_BELOW_OWNERSHIP_REMAINDER"


def _interaction_label(value: float) -> str:
    return (
        "MATERIAL"
        if abs(value) >= ENERGY_OWNERSHIP_BOUND
        else "BOUNDED"
    )


def _find_total_call(calls, target):
    target_without_gate = {
        key: value for key, value in target.items() if key != "campaign_gate"
    }
    matches = [
        call
        for call in calls
        if {
            key: value
            for key, value in call["metric"].items()
            if key != "campaign_gate"
        }
        == target_without_gate
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"total-field interception changed: found {len(matches)} matches"
        )
    return matches[0]


def _locked_valid(removals: dict[str, float]) -> bool:
    return all(
        key in removals and abs(removals[key] - value) <= LOCKED_TOL
        for key, value in LOCKED_U_REMOVAL.items()
    )


def _component_matrix(
    component: str,
    term_calls: dict[str, dict[str, Any]],
    total_call: dict[str, Any],
) -> dict[str, Any]:
    mask = term_calls[AXES[0]]["mask"]
    expected_n = term_calls[AXES[0]]["expected_n"]
    r0_full = term_calls[AXES[0]]["residual"]
    r0 = np.asarray(r0_full)[mask]
    if expected_n != r0.size or r0.size == 0:
        raise SystemExit(f"{component} residual population changed")
    if not np.array_equal(
        np.asarray(total_call["lego"]) - np.asarray(total_call["nemo"]),
        r0_full,
    ):
        raise SystemExit(f"{component} total interception disagrees with residual")

    full_errors = {
        name: np.asarray(call["lego"]) - np.asarray(call["nemo"])
        for name, call in term_calls.items()
    }
    errors = {name: values[mask] for name, values in full_errors.items()}
    r0_rms = _rms(r0)
    arms: dict[str, dict[str, Any]] = {}
    arm_residuals: dict[str, np.ndarray] = {}
    benefits: dict[str, float] = {}
    for bits in BITS:
        names = _axis_names(bits)
        correction = sum((errors[name] for name in names), np.zeros_like(r0))
        residual = r0 - correction
        arm_residuals[bits] = residual
        score = (
            {
                "correlation_with_total_residual": None,
                "rms_gain": 0.0,
                "residual_removal": 0.0,
                "verdict": "BASELINE",
            }
            if bits == "000"
            else round2._attribute_error(correction, r0)
        )
        q = _rms(residual) ** 2 / r0_rms**2
        benefit = 1.0 - q
        benefits[bits] = benefit
        model_arm = np.asarray(total_call["lego"]).copy()
        for name in names:
            model_arm -= full_errors[name]
        campaign_metric = round2.round1._metric(
            model_arm,
            total_call["nemo"],
            mask,
            expected_n,
        )
        campaign_metric["campaign_gate"] = round2.round1._classify_metric(
            campaign_metric, None
        )
        arms[bits] = {
            "substitutions": names,
            "normalized_residual_rms": _rms(residual) / r0_rms,
            "normalized_squared_residual_energy": q,
            "squared_energy_benefit": benefit,
            "attribution": score,
            "campaign_metric": campaign_metric,
        }

    main = {
        "L": benefits["100"],
        "V": benefits["010"],
        "C": benefits["001"],
    }
    pair = {
        "LxV": benefits["110"] - benefits["100"] - benefits["010"],
        "LxC": benefits["101"] - benefits["100"] - benefits["001"],
        "VxC": benefits["011"] - benefits["010"] - benefits["001"],
    }
    triple = (
        benefits["111"]
        - benefits["110"]
        - benefits["101"]
        - benefits["011"]
        + benefits["100"]
        + benefits["010"]
        + benefits["001"]
    )
    shapley = {
        "lateral_friction": (
            main["L"] + 0.5 * (pair["LxV"] + pair["LxC"]) + triple / 3.0
        ),
        "vorticity": (
            main["V"] + 0.5 * (pair["LxV"] + pair["VxC"]) + triple / 3.0
        ),
        "cor_removal": (
            main["C"] + 0.5 * (pair["LxC"] + pair["VxC"]) + triple / 3.0
        ),
    }
    energy_sum = sum(main.values()) + sum(pair.values()) + triple
    energy_checks = {
        "mobius_sum_minus_full_benefit": energy_sum - benefits["111"],
        "shapley_sum_minus_full_benefit": sum(shapley.values()) - benefits["111"],
    }
    if any(abs(value) > ENERGY_IDENTITY_BAR for value in energy_checks.values()):
        raise SystemExit(f"{component} energy identity failed: {energy_checks}")

    pair_field_contrasts = {
        "LxV": arm_residuals["110"] - arm_residuals["100"]
        - arm_residuals["010"] + arm_residuals["000"],
        "LxC": arm_residuals["101"] - arm_residuals["100"]
        - arm_residuals["001"] + arm_residuals["000"],
        "VxC": arm_residuals["011"] - arm_residuals["010"]
        - arm_residuals["001"] + arm_residuals["000"],
    }
    triple_field_contrast = (
        arm_residuals["111"]
        - arm_residuals["110"]
        - arm_residuals["101"]
        - arm_residuals["011"]
        + arm_residuals["100"]
        + arm_residuals["010"]
        + arm_residuals["001"]
        - arm_residuals["000"]
    )
    field_linearity = {
        **{
            name: _rms(value) / r0_rms
            for name, value in pair_field_contrasts.items()
        },
        "LxVxC": _rms(triple_field_contrast) / r0_rms,
    }
    if any(value > LINEARITY_BAR for value in field_linearity.values()):
        raise SystemExit(f"{component} field substitution is not linear: {field_linearity}")

    group_confirms = (
        arms["111"]["attribution"]["verdict"] == "CONFIRMS_CARRY"
        and arms["111"]["normalized_squared_residual_energy"]
        <= ENERGY_OWNERSHIP_BOUND
    )
    term_dispositions = {}
    for name in AXES:
        term_dispositions[name] = (
            _ownership_label(shapley[name])
            if group_confirms
            else "UNRESOLVED_IN_COMPOSITE"
        )
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
    closure = _rms(assembly_remainder) / r0_rms
    remainder_bounded = closure <= TERM_GAIN_BOUND
    noncandidate["assembly_remainder"] = {
        "rms_gain": closure,
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
    all_model = np.asarray(total_call["lego"]).copy()
    for name in ORDERED:
        all_model -= full_errors[name]
    all_metric = round2.round1._metric(
        all_model, total_call["nemo"], mask, expected_n
    )
    all_metric["campaign_gate"] = round2.round1._classify_metric(all_metric, None)
    fully_disposed = group_confirms and all(
        item["bounded"] for item in noncandidate.values()
    )
    return {
        "component": component,
        "sample_count": int(r0.size),
        "base_residual_rms": r0_rms,
        "arms": arms,
        "energy_decomposition": {
            "main": main,
            "pair_interactions": {
                name: {"value": value, "label": _interaction_label(value)}
                for name, value in pair.items()
            },
            "triple_interaction": {
                "value": triple,
                "label": _interaction_label(triple),
            },
            "shapley": {
                name: {
                    "value": value,
                    "label": (
                        _ownership_label(value)
                        if group_confirms
                        else "DESCRIPTIVE_UNADMITTED_GROUP_UNRESOLVED"
                    ),
                }
                for name, value in shapley.items()
            },
            "identities": energy_checks,
        },
        "field_linearity_normalized_rms": field_linearity,
        "group_confirms": group_confirms,
        "noncandidate_bounds": noncandidate,
        "term_dispositions": term_dispositions,
        "all_eight_term_closure_normalized_rms": closure,
        "assembly_remainder": {
            "normalized_rms": closure,
            "max_abs": float(np.max(np.abs(assembly_remainder))),
            "bound": TERM_GAIN_BOUND,
            "bounded": remainder_bounded,
            "disposition": noncandidate["assembly_remainder"]["disposition"],
        },
        "all_eight_oracle_endpoint": all_metric,
        "fully_disposed": fully_disposed,
    }


def _controls() -> dict[str, Any]:
    locked = dict(LOCKED_U_REMOVAL)
    planted_locked = dict(locked)
    planted_locked["110"] += 2.0 * LOCKED_TOL
    phase = 2.0 * np.pi * np.arange(1024, dtype=np.float64) / 1024.0
    residual = np.sin(phase)
    confirming = round2._attribute_error(residual.copy(), residual)
    refuting = round2._attribute_error(np.cos(phase), residual)
    controls = {
        "locked_corners_accept_exact": _locked_valid(locked),
        "planted_locked_corner_rejected": not _locked_valid(planted_locked),
        "planted_linearity_breach_rejected": 2.0 * LINEARITY_BAR > LINEARITY_BAR,
        "positive_label_reached": _ownership_label(0.02)
        == "OWNED_POSITIVE_CONTRIBUTOR",
        "canceller_label_reached": _ownership_label(-0.02)
        == "OWNED_CANCELLER",
        "bounded_label_reached": _ownership_label(0.005)
        == "BOUNDED_BELOW_OWNERSHIP_REMAINDER",
        "group_confirm_reached": confirming["verdict"] == "CONFIRMS_CARRY",
        "group_refute_reached": refuting["verdict"] == "REFUTES_CARRY",
        "bounded_term_accepts_0p09": 0.09 <= TERM_GAIN_BOUND,
        "planted_0p11_term_rejected": not (0.11 <= TERM_GAIN_BOUND),
    }
    if not all(controls.values()):
        raise SystemExit(f"matrix control failed: {controls}")
    return controls


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
        with tempfile.TemporaryDirectory(prefix="dino-split-r4-") as tmpdir:
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
    components = {}
    for component, offset in (("u", 18), ("v", 19)):
        calls = {
            name: metric_calls[offset + 2 * index]
            for index, name in enumerate(ORDERED)
        }
        for name, call in calls.items():
            if call["metric"] != base["faithful_term_metrics"][component][name]:
                raise SystemExit(f"{component} captured metric changed at {name}")
        total = _find_total_call(
            total_calls, base["total_metrics"][f"faithful_{component}"]
        )
        components[component] = _component_matrix(component, calls, total)

    measured_locked = {
        bits: components["u"]["arms"][bits]["attribution"]["residual_removal"]
        for bits in LOCKED_U_REMOVAL
    }
    if not _locked_valid(measured_locked):
        raise SystemExit(f"locked U matrix corners changed: {measured_locked}")
    controls = _controls()

    dirty_after = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    if dirty_after:
        raise SystemExit("worktree changed during measurement:\n" + dirty_after)

    prereg = repo / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round4.md"
    precedent_paths = {
        "preregistration": (
            PRECEDENT_PREREG,
            "scripts/validate/ocean_fidelity/dino_1226/"
            "PREREG_tcarry_een_omega_interaction.md",
        ),
        "probe": (
            PRECEDENT_PROBE,
            "scripts/validate/ocean_fidelity/dino_1226/"
            "tcarry_bridge_omega_score.py",
        ),
    }
    precedent = {}
    for name, (commit, path) in precedent_paths.items():
        value = subprocess.check_output(
            ["git", "show", f"{commit}:{path}"], cwd=repo
        )
        precedent[name] = {
            "commit": commit,
            "path": path,
            "sha256": _sha256_bytes(value),
        }
    provenance_paths = {
        "probe": Path(__file__).resolve(),
        "preregistration": prereg,
        "round2_probe": Path(round2.__file__).resolve(),
        "round2_preregistration": (
            repo / "docs/ocean/fidelity/PREREG_split_explicit_momentum_chain_round2.md"
        ),
        "round3_artifact": (
            repo / "docs/ocean/fidelity/"
            "dino_split_explicit_momentum_chain_round3_artifact.json"
        ),
    }
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round4-v1",
        "session_id": os.environ.get("CODEX_SESSION_ID", "unset"),
        "git": {"commit": head, "clean_before": True, "clean_after": True},
        "runtime": base["runtime"],
        "lane": base["lane"],
        "kt": base["kt"],
        "matrix_axes": list(AXES),
        "bits_order": list(BITS),
        "locked_u_removal": measured_locked,
        "components": components,
        "row_1_1_fully_disposed": all(
            item["fully_disposed"] for item in components.values()
        ),
        "controls": {"matrix": controls, "round2": base["controls"]},
        "precedent": precedent,
        "base_round2_artifact_sha256": base_sha256,
        "base_round2_term_error_closure": base["term_error_sum_closure"],
        "dump_sha256": base["dump_sha256"],
        "run_input_sha256": base["run_input_sha256"],
        "provenance_sha256": {
            name: _sha256(path) for name, path in provenance_paths.items()
        },
        "slot": "NOT_ALLOCATED_EXISTING_DUMPS_ONLY",
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for component, result in components.items():
        full = result["arms"]["111"]
        print(
            f"{component.upper()} 111: removal="
            f"{full['attribution']['residual_removal']:.12f} "
            f"Q={full['normalized_squared_residual_energy']:.12e} "
            f"{full['attribution']['verdict']} "
            f"fully_disposed={result['fully_disposed']}"
        )
        for name, item in result["energy_decomposition"]["shapley"].items():
            print(f"  {name}: phi={item['value']:+.12f} {item['label']}")
    print(f"row_1_1_fully_disposed={receipt['row_1_1_fully_disposed']}")
    print(f"artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
