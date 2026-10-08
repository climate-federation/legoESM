#!/usr/bin/env python3
"""Split ORCA2 kt=8 substep-2 exit depth into raw-depth and SSH operands."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round129_substep2_walk as r129,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round167_exit_depth_walk as r167,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round95_spgts_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

PLANTS = ("none", "registry", "source-order", "depth-replay", "ssh-replay")
SOURCE_ORDER = (
    "raw_depth_u",
    "face_ssh_u",
    "exit_depth_u",
    "exit_inverse_u",
)
UPSTREAM_ORDER = (
    "entry_u", "entry_v", "entry_ssh", "entry_inverse_u", "entry_inverse_v",
    "mid_u", "mid_v", "mid_ssh", "mid_depth_u", "mid_depth_v",
    "transport_u", "transport_v", "continuity_du", "continuity_dv",
    "continuity_divergence", "after_ssh",
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float64).view(np.uint64)


def _score(candidate: np.ndarray, oracle: np.ndarray, active: np.ndarray) -> dict:
    return r167._score(candidate, oracle, active)


def _registered_row(candidate: np.ndarray, oracle: np.ndarray,
                    registered: np.ndarray) -> dict:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    registered = np.asarray(registered, dtype=bool)
    require(candidate.shape == oracle.shape == registered.shape,
            "registered operand shapes moved")
    require(bool(np.any(registered)), "registered operand set is empty")
    unequal = (_bits(candidate) != _bits(oracle)) & registered
    delta = np.abs(candidate - oracle)
    finite = np.isfinite(delta) & registered
    return {
        "bit_exact": not bool(np.any(unequal)),
        "differing_cells": int(np.count_nonzero(unequal)),
        "count": int(np.count_nonzero(registered)),
        "candidate_nonfinite": int(np.count_nonzero(
            ~np.isfinite(candidate) & registered)),
        "oracle_nonfinite": int(np.count_nonzero(
            ~np.isfinite(oracle) & registered)),
        "finite_absolute_max": float(np.max(delta[finite], initial=0.0)),
        "candidate_min": float(np.min(candidate[registered])),
        "candidate_max": float(np.max(candidate[registered])),
        "oracle_min": float(np.min(oracle[registered])),
        "oracle_max": float(np.max(oracle[registered])),
    }


def _first_nonbit(rows: list[dict]) -> dict | None:
    return next((row for row in rows if not row["bit_exact"]), None)


def _score_upstream(trace, oracle: dict[str, np.ndarray], active: dict[str, np.ndarray],
                    area: np.ndarray, index: int) -> list[dict]:
    """Score the recorded prefix without rejecting the known later overflow."""

    step = index + 1
    prefix = f"j{step:03d}"
    previous = f"j{index:03d}"
    candidate = {
        "entry_u": r97._native_u(trace["u_entry"][index]),
        "entry_v": r97._native_v(trace["v_entry"][index]),
        "entry_ssh": np.asarray(trace["eta_entry"][index]),
        "entry_inverse_u": r97._native_u(trace["inverse_depth_u"][index]),
        "entry_inverse_v": r97._native_v(trace["inverse_depth_v"][index]),
        "mid_u": r97._native_u(trace["u_mid"][index]),
        "mid_v": r97._native_v(trace["v_mid"][index]),
        "mid_ssh": np.asarray(trace["eta_mid"][index]),
        "mid_depth_u": r97._native_u(trace["transport_face_depth_u"][index]),
        "mid_depth_v": r97._native_v(trace["transport_face_depth_v"][index]),
        "transport_u": r97._native_u(trace["transport_metric_u"][index]),
        "transport_v": r97._native_v(trace["transport_metric_v"][index]),
        "continuity_du": np.asarray(trace["continuity_du"][index]),
        "continuity_dv": np.asarray(trace["continuity_dv"][index]),
        "continuity_divergence": np.asarray(trace["continuity_divergence"][index]),
        "after_ssh": np.asarray(trace["eta_continuity"][index]),
    }
    oracle_du = oracle[f"{prefix}_zhU"] - np.roll(
        oracle[f"{prefix}_zhU"], 1, axis=1)
    oracle_v_south = np.concatenate([
        np.zeros_like(oracle[f"{prefix}_zhV"][:1]),
        oracle[f"{prefix}_zhV"][:-1]], axis=0)
    oracle_dv = oracle[f"{prefix}_zhV"] - oracle_v_south
    oracle_divergence = (oracle_du + oracle_dv) * (1.0 / area)
    reference = {
        "entry_u": oracle[f"{previous}_ua_new"],
        "entry_v": oracle[f"{previous}_va_new"],
        "entry_ssh": oracle[f"{previous}_ssha_e"],
        "entry_inverse_u": oracle[f"{previous}_hur_e"],
        "entry_inverse_v": oracle[f"{previous}_hvr_e"],
        "mid_u": oracle[f"{prefix}_ua_ext"],
        "mid_v": oracle[f"{prefix}_va_ext"],
        "mid_ssh": oracle[f"{prefix}_sshp2_mid"],
        "mid_depth_u": oracle[f"{prefix}_hup2_e"],
        "mid_depth_v": oracle[f"{prefix}_hvp2_e"],
        "transport_u": oracle[f"{prefix}_zhU"],
        "transport_v": oracle[f"{prefix}_zhV"],
        "continuity_du": oracle_du,
        "continuity_dv": oracle_dv,
        "continuity_divergence": oracle_divergence,
        "after_ssh": oracle[f"{prefix}_ssha_e"],
    }
    faces = {
        "entry_u": "u", "entry_v": "v", "entry_ssh": "t",
        "entry_inverse_u": "u", "entry_inverse_v": "v",
        "mid_u": "u", "mid_v": "v", "mid_ssh": "t",
        "mid_depth_u": "u", "mid_depth_v": "v",
        "transport_u": "u", "transport_v": "v",
        "continuity_du": "t", "continuity_dv": "t",
        "continuity_divergence": "t", "after_ssh": "t",
    }
    rows = []
    for name in UPSTREAM_ORDER:
        rows.append({"name": name, **_score(
            candidate[name], reference[name], active[faces[name]])})
    return rows


def classify(report: dict, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "registry":
        report["registered_count"] -= 1
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "depth-replay":
        report["registered_rows"]["raw_plus_oracle_face_ssh"]["count"] -= 1
    elif plant == "ssh-replay":
        report["registered_rows"]["oracle_eta_face_ssh_replay"]["count"] -= 1

    admission = report["admission"]
    require(admission["rank_coverage"] == "exactly-once",
            "rank coverage moved")
    require(len(admission["records"]) == 2 and all(
        row["icycle"] == r167.EXPECTED_SUBSTEPS
        and row["groups"] == r167.EXPECTED_GROUPS
        for row in admission["records"]), "record census moved")
    require(len(admission["terminal_restart_comparisons"]) == 20,
            "restart census moved")
    require(report["registered_count"] == 41,
            "round-167 native registered-face census moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER,
            "exit-depth operand source order moved")
    require(tuple(report["upstream_order"]) == UPSTREAM_ORDER,
            "upstream source order moved")
    registered = report["registered_rows"]
    require(all(row["count"] == report["registered_count"]
                for row in registered.values()),
            "registered replay coverage moved")
    require(report["completed_kt"] == 7,
            "complete private arm no longer reaches kt=7")

    raw_exact = registered["raw_plus_oracle_face_ssh"]["bit_exact"]
    face_owns = (
        raw_exact and registered["candidate_sum_replay"]["bit_exact"]
        and registered["candidate_face_ssh"]["differing_cells"]
        == report["registered_count"])
    eta_upstream = (
        registered["oracle_eta_face_ssh_replay"]["bit_exact"]
        and registered["candidate_after_ssh_stencil"]["differing_cells"] > 0)
    report["prediction_dispositions"] = {
        "R168-P1": "CONFIRMED" if raw_exact else "REFUTED",
        "R168-P2": "CONFIRMED" if face_owns else "REFUTED",
        "R168-P3": "CONFIRMED" if eta_upstream else "REFUTED",
        "R168-P4": "CONFIRMED" if report["first_upstream_nonbit"] else "REFUTED",
        "R168-P5": "CONFIRMED",
    }
    report["status"] = "PASS_ROUND168_EXIT_DEPTH_OPERAND_WALK"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            baseline_root: Path, expect_commit: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_reference_face_depths,
        _nemo_ssh_avg_apply,
        _nemo_ssh_avg_prep,
        _nemo_ssh_avg_reference_depth_override,
        nemo_source_round,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    stamp = worktree_stamp()
    require(stamp["clean"], "round-168 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-168 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-168 walk requires production JIT on CPU")

    admission = check_record.run(
        spg_root, baseline_root, "none", expected_kt=r167.EXPECTED_KT,
        prefix=r167.PREFIX, expected_magic=r167.MAGIC)
    oracle, census = r97.assemble_record(
        spg_root, prefix=r167.PREFIX, expected_kt=r167.EXPECTED_KT,
        expected_magic=r167.MAGIC)
    require(census["coverage"] == "exactly-once",
            "assembled record coverage moved")

    card, state, freshwater, surface = r166._setup(deck_root, frame_root)
    hooks = r166._hooks(card)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    for kt in range(1, 8):
        state = jax.device_get(ordinary.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        print(f"PROGRESS round168 complete kt={kt}", file=sys.stderr, flush=True)

    traced = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks._replace(expose_barotropic_substeps=True))
    observed = jax.device_get(traced.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    trace = observed.substeps
    require(trace["eta_entry"].shape[0] == r167.EXPECTED_SUBSTEPS,
            "production trace substep count moved")

    index = 1
    prefix = "j002"
    masks = phase3_gate.expected_masks(card)
    active = {
        "t": np.asarray(masks["ssh"], dtype=bool),
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }
    candidate_depth = r97._native_u(trace["face_depth_u_exit"][index])
    candidate_inverse = r97._native_u(trace["r1_face_depth_u_exit"][index])
    candidate_face_ssh = r97._native_u(trace["face_ssh_u_exit"][index])
    raw_compact = np.asarray(hooks.barotropic_reference_face_depth_override[0])
    candidate_raw = r97._native_u(raw_compact)
    oracle_depth = oracle[f"{prefix}_hu_e"]
    oracle_face_ssh = oracle[f"{prefix}_sshu_a"]
    registered = ~np.isfinite(candidate_inverse)
    require(int(np.count_nonzero(registered)) == 41,
            "round-167 registered native face set moved")

    raw_plus_oracle = np.asarray(jax.device_get(nemo_source_round(
        jnp.asarray(candidate_raw) + jnp.asarray(oracle_face_ssh))))
    candidate_sum = np.asarray(jax.device_get(nemo_source_round(
        jnp.asarray(candidate_raw) + jnp.asarray(candidate_face_ssh))))

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "rung-0 card has no raw NEMO masks")
    u_mask = jnp.asarray(r97._to_model_u(
        np.max(np.asarray(raw.umask), axis=-1)), dtype=jnp.float64)
    v_mask = jnp.asarray(r97._to_model_v(
        np.max(np.asarray(raw.vmask), axis=-1)), dtype=jnp.float64)
    prep = _nemo_ssh_avg_prep(
        jnp.asarray(state.H_bathy.data, dtype=jnp.float64),
        jnp.asarray(state.land_mask.data, dtype=jnp.float64),
        card.recipe.grid, jnp.float64)
    prep = _nemo_ssh_avg_reference_depth_override(
        prep, _nemo_literal_reference_face_depths(
            card.recipe.z_coord, jnp.float64), jnp.float64)
    oracle_eta_replay = jax.device_get(_nemo_ssh_avg_apply(
        jnp.asarray(oracle[f"{prefix}_ssha_e"]), u_mask, v_mask,
        card.recipe.grid, jnp.asarray(card.recipe.grid.area), prep,
        return_literal_inverse=True, return_ssh_average=True))
    oracle_eta_face_ssh = r97._native_u(oracle_eta_replay[4])

    # A face differs if either T cell in its west/east stencil differs.  This
    # is the exact stencil used by dynspg_ts.f90:633-636.
    eta_unequal = _bits(np.asarray(trace["eta_continuity"][index])) != _bits(
        oracle[f"{prefix}_ssha_e"])
    eta_stencil = eta_unequal | np.roll(eta_unequal, -1, axis=1)

    registered_rows = {
        "candidate_raw_depth": _registered_row(
            candidate_raw, candidate_raw, registered),
        "candidate_face_ssh": _registered_row(
            candidate_face_ssh, oracle_face_ssh, registered),
        "candidate_exit_depth": _registered_row(
            candidate_depth, oracle_depth, registered),
        "raw_plus_oracle_face_ssh": _registered_row(
            raw_plus_oracle, oracle_depth, registered),
        "candidate_sum_replay": _registered_row(
            candidate_sum, candidate_depth, registered),
        "oracle_eta_face_ssh_replay": _registered_row(
            oracle_eta_face_ssh, oracle_face_ssh, registered),
        "candidate_after_ssh_stencil": {
            "bit_exact": not bool(np.any(eta_stencil & registered)),
            "differing_cells": int(np.count_nonzero(eta_stencil & registered)),
            "count": int(np.count_nonzero(registered)),
        },
    }
    full_rows = {
        "raw_depth_u": _score(candidate_raw, candidate_raw, active["u"]),
        "face_ssh_u": _score(candidate_face_ssh, oracle_face_ssh, active["u"]),
        "exit_depth_u": _score(candidate_depth, oracle_depth, active["u"]),
        "exit_inverse_u": _score(
            candidate_inverse, oracle[f"{prefix}_hur_e"], active["u"]),
    }
    upstream_rows = _score_upstream(
        trace, oracle, active, np.asarray(card.recipe.grid.area), index)
    require(tuple(row["name"] for row in upstream_rows) == UPSTREAM_ORDER,
            "round-129 upstream row registry moved")
    first_upstream = _first_nonbit(upstream_rows)

    raw_report = {
        "format": "nemo-testcase-l4-orca2-round168-exit-operands-v1",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": admission,
        "assembled_record": census,
        "completed_kt": 7,
        "kt": 8,
        "substep": 2,
        "registered_count": int(np.count_nonzero(registered)),
        "registered_first_j_i": list(map(int, np.argwhere(registered)[0])),
        "source_order": list(SOURCE_ORDER),
        "upstream_order": list(UPSTREAM_ORDER),
        "registered_rows": registered_rows,
        "full_rows": full_rows,
        "upstream_rows": upstream_rows,
        "first_upstream_nonbit": first_upstream,
        "worktree": stamp,
    }
    return classify(raw_report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.spg_root,
                args.baseline_root, args.expect_commit)),
                "measurement requires deck/frame/SPG/baseline roots and commit")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.baseline_root, args.expect_commit)
        else:
            require(args.report_in is not None,
                    "classification requires --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError, check_record.Refusal,
            r97.GateError, r167.GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_ROUND168_EXIT_DEPTH_OPERAND_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
