#!/usr/bin/env python3
"""Adjudicate the preregistered V-face metric production counterfactual."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PRE_SHA = "ed680b0a340b96efbc974099603e1699181a868b5c1fb0242be9c3f966511d40"
CONFIRM_REDUCTION = 0.99
REFUTE_REDUCTION = 0.10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_bound(path: Path, expected: str, label: str) -> dict[str, Any]:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return json.loads(path.read_text())


def _rows(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["subrow"]: row for row in artifact["measurements"]}
    if set(rows) != {f"9.{i}" for i in range(1, 8)}:
        raise SystemExit(f"unexpected ordered row set: {sorted(rows)}")
    return rows


def _reduction(before: float, after: float) -> float:
    if not (math.isfinite(before) and math.isfinite(after) and before > 0.0):
        raise SystemExit(f"invalid reduction operands: before={before}, after={after}")
    return 1.0 - after / before


def _wall_rms(artifact: dict[str, Any], row: str) -> float:
    values = artifact["wall_pointwise_deltas"]["B_nemo_divergence"][row]["residual"]
    if not values:
        raise SystemExit(f"empty wall residual at {row}")
    return math.sqrt(sum(float(x) ** 2 for x in values) / len(values))


def _classify(reductions: tuple[float, ...]) -> str:
    if not reductions or not all(math.isfinite(value) for value in reductions):
        raise SystemExit(f"invalid ownership reductions: {reductions}")
    if all(value >= CONFIRM_REDUCTION for value in reductions):
        return "VFACE_WIDTH_DOMINANT_FLUX_OWNER_CONFIRMED"
    if all(value <= REFUTE_REDUCTION for value in reductions):
        return "VFACE_WIDTH_DOMINANT_FLUX_OWNER_REFUTED"
    return "VFACE_WIDTH_DOMINANT_FLUX_OWNER_OPEN_UNRESOLVED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-artifact", type=Path, required=True)
    parser.add_argument("--post-artifact", type=Path, required=True)
    parser.add_argument("--post-sha256", required=True)
    parser.add_argument("--expected-post-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pre = _load_bound(args.pre_artifact, PRE_SHA, "round-9 artifact")
    post = _load_bound(args.post_artifact, args.post_sha256, "fixed-tree rescore")
    if pre.get("schema") != "dino-split-explicit-momentum-chain-round9-v1":
        raise SystemExit("unexpected pre-fix schema")
    if post.get("schema") != pre["schema"]:
        raise SystemExit("fixed-tree scorer schema changed")
    if post.get("git", {}).get("commit") != args.expected_post_commit:
        raise SystemExit("fixed-tree package commit mismatch")
    if post.get("source_bindings", {}).get("package_commit") != args.expected_post_commit:
        raise SystemExit("fixed-tree source binding commit mismatch")
    pre_session = pre.get("session_id")
    post_session = post.get("session_id")
    if not pre_session or pre_session != post_session:
        raise SystemExit(
            f"same nonempty session required: pre={pre_session!r}, post={post_session!r}")
    active_session = os.environ.get("CODEX_SESSION_ID")
    if not active_session or active_session != post_session:
        raise SystemExit(
            f"active CODEX_SESSION_ID must match artifacts: active={active_session!r}, "
            f"artifact={post_session!r}")
    if pre.get("dt_e_s") != post.get("dt_e_s"):
        raise SystemExit("fixed-tree rescore changed dt_e_s")
    if pre.get("runtime") != post.get("runtime"):
        raise SystemExit("fixed-tree rescore changed runtime versions/platform")
    if not (pre.get("git", {}).get("clean_before") and pre.get("git", {}).get("clean_after")):
        raise SystemExit("round-9 scorer was not clean")
    if not (post.get("git", {}).get("clean_before") and post.get("git", {}).get("clean_after")):
        raise SystemExit("fixed-tree scorer was not clean")
    if not all(pre.get("controls", {}).values()) or not all(post.get("controls", {}).values()):
        raise SystemExit("a scorer planted control failed")
    if pre.get("backend") != "cpu" or post.get("backend") != "cpu":
        raise SystemExit("both scores must use CPU")
    if pre.get("jax_enable_x64") is not True or post.get("jax_enable_x64") is not True:
        raise SystemExit("both scores must use JAX fp64")
    if pre.get("dump_sha256") != post.get("dump_sha256"):
        raise SystemExit("fixed-tree rescore did not use the same six dumps")
    immutable = ("binary_sha256", "off_binary_sha256", "dynspg_source_sha256",
                 "round8_artifact_sha256", "bracket_receipt_sha256")
    for key in immutable:
        if pre["source_bindings"].get(key) != post["source_bindings"].get(key):
            raise SystemExit(f"fixed-tree rescore changed source binding {key}")

    before, after = _rows(pre), _rows(post)
    reductions = {
        "subrow_9.5_zhV_normalized_rms": _reduction(
            before["9.5"]["normalized_rms_error"], after["9.5"]["normalized_rms_error"]),
        "subrow_9.6_zhdiv_normalized_rms": _reduction(
            before["9.6"]["normalized_rms_error"], after["9.6"]["normalized_rms_error"]),
        "wall_j=1_residual_rms": _reduction(_wall_rms(pre, "j=1"), _wall_rms(post, "j=1")),
        "wall_j=197_residual_rms": _reduction(_wall_rms(pre, "j=197"), _wall_rms(post, "j=197")),
    }
    values = tuple(reductions.values())
    disposition = _classify(values)

    # Data-derived non-vacuity controls exercise the same classifier as the
    # measured result. Treating the pre artifact as its own post artifact gives
    # zero reduction on every extracted axis and must REFUTE. Replacing one
    # measured post axis with its pre value gives one zero plus the other three
    # measured reductions and must be OPEN, proving the all-four conjunction.
    same_artifact_plant = tuple(
        _reduction(value, value)
        for value in (
            before["9.5"]["normalized_rms_error"],
            before["9.6"]["normalized_rms_error"],
            _wall_rms(pre, "j=1"),
            _wall_rms(pre, "j=197"),
        )
    )
    mixed_axis_plant = (same_artifact_plant[0], *values[1:])
    controls = {
        "both_scorers_passed_all_controls": True,
        "pre_as_post_classifies_refuted": _classify(same_artifact_plant)
            == "VFACE_WIDTH_DOMINANT_FLUX_OWNER_REFUTED",
        "one_axis_unfixed_classifies_open": _classify(mixed_axis_plant)
            == "VFACE_WIDTH_DOMINANT_FLUX_OWNER_OPEN_UNRESOLVED",
        "same_nonempty_session": bool(pre_session and pre_session == post_session),
        "active_session_matches": active_session == post_session,
        "same_timestep": pre.get("dt_e_s") == post.get("dt_e_s"),
        "same_runtime": pre.get("runtime") == post.get("runtime"),
    }
    if not all(controls.values()):
        raise SystemExit(f"adjudicator planted control failed: {controls}")

    ordered = [after[f"9.{i}"] for i in range(1, 8)]
    first_debt = next((row["subrow"] for row in ordered if row["gate_status"] != "AT BAR"), None)
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round10-v1",
        "session_id": post["session_id"],
        "pre_artifact": {"path": str(args.pre_artifact.resolve()), "sha256": PRE_SHA,
                         "package_commit": pre["git"]["commit"]},
        "post_artifact": {"path": str(args.post_artifact.resolve()),
                          "sha256": args.post_sha256,
                          "package_commit": args.expected_post_commit},
        "fixed_production_sha256": post["source_bindings"]["production_sha256"],
        "same_six_dumps": True,
        "backend": "cpu",
        "jax_enable_x64": True,
        "registered_bars": {"confirm_min_reduction_each_axis": CONFIRM_REDUCTION,
                            "refute_max_reduction_each_axis": REFUTE_REDUCTION},
        "reductions": reductions,
        "pre_wall_residual_rms": {row: _wall_rms(pre, row) for row in ("j=1", "j=197")},
        "post_wall_residual_rms": {row: _wall_rms(post, row) for row in ("j=1", "j=197")},
        "post_measurements": ordered,
        "post_first_diverged_subrow": first_debt,
        "frozen_round9": {
            "disposition": pre["disposition"],
            "Q_depth_held": pre["substitution_arms"]["Q_depth_held"],
            "F_flux_held": pre["substitution_arms"]["F_flux_held"],
            "B_nemo_divergence": pre["substitution_arms"]["B_nemo_divergence"],
        },
        "post_substitution_arms": post["substitution_arms"],
        "controls": controls,
        "disposition": disposition,
        "later_chain": post["later_rows"],
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"ROUND10 disposition={disposition} post_first_diverged_subrow={first_debt}")
    for key, value in reductions.items():
        print(f"REDUCTION {key}={value:.17g}")
    print(f"artifact={args.output} sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
