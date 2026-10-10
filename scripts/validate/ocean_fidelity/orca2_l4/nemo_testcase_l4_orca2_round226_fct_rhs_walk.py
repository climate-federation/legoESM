#!/usr/bin/env python3
"""Compare OMT-4's final FCT RHS associations against the admitted oracle."""

from __future__ import annotations

import argparse
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
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round224_tracer_fold_walk as fold_walk,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)

PLANTS = ("none", "record-owner", "source-association", "stage-live")


class GateError(RuntimeError):
    """The final-RHS replay no longer supports its source-order claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _comparison(reference: np.ndarray, candidate: np.ndarray,
                support: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    support = np.broadcast_to(np.asarray(support, dtype=bool), reference.shape)
    require(reference.shape == candidate.shape, "comparison shape moved")
    require(reference.dtype == candidate.dtype == np.float64,
            "comparison dtype moved")
    delta = candidate - reference
    unequal = ((candidate.view(np.uint64) != reference.view(np.uint64))
               & support)
    values = delta[support]
    return {
        "support": int(np.count_nonzero(support)),
        "unequal": int(np.count_nonzero(unequal)),
        "maximum_absolute": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "record-owner":
        report["record_alignment"]["matched_half"] = "none"
    elif plant == "source-association":
        report["tracers"]["T"]["literal_vs_generic"]["unequal"] = 0
    elif plant == "stage-live":
        report["stage_consumer"] = "INERT"

    require(report.get("execution") == "offline-pure-jit-cpu-fp64-libm",
            "execution policy moved")
    require(report.get("record_scope") ==
            "admitted OMT-4 kt=1 stage-3 state and rank-0 RKTR3 record",
            "record scope moved")
    require(report.get("in_executable_observers") == 0,
            "an in-executable observer entered the walk")
    require(report["record_alignment"]["matched_half"] in ("south", "north"),
            "rank-0 oracle record does not align to one global half")
    require(report["record_alignment"]["Kmm_unequal"] == 0,
            "rank-0 oracle Kmm cross-check moved")
    improved = 0
    for tracer in ("T", "S"):
        row = report["tracers"][tracer]
        require(row["literal_vs_generic"]["unequal"] > 0,
                f"{tracer}: source-associated RHS is inert")
        require(row["literal_vs_oracle"]["rms"] <=
                row["generic_vs_oracle"]["rms"],
                f"{tracer}: source-associated RHS is farther from NEMO")
        improved += int(row["literal_vs_oracle"]["rms"] <
                        row["generic_vs_oracle"]["rms"])
    require(improved > 0, "source-associated RHS improves neither tracer")
    require(report.get("stage_consumer") == "LIVE_WITH_SPEC",
            "stage consumer liveness was not retained")
    require(report.get("statement_sufficiency") == "UNMEASURED_WITH_SPEC",
            "offline replay manufactured a sufficiency verdict")
    report["predictions"] = {
        "R226-P1": "CONFIRMED",
        "R226-P2": "UNMEASURED_WITH_SPEC",
        "R226-P3": "UNMEASURED_WITH_SPEC",
        "R226-P4": "NOT_REACHED",
        "R226-P5": "CONFIRMED" if plant == "none" else "PLANT",
    }
    report["status"] = "PASS_R226_FIRST_NONBIT_FINAL_FCT_RHS"
    return report


def measure(deck_root: Path, frames_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean import advection
    from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
    from legoesm.ocean.vertical import compute_layer_thickness, diagnose_w_from_flux_div

    stamp = worktree_stamp()
    require(stamp["clean"], "round-226 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-226 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "RHS replay requires production JIT on CPU")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 raw mesh operands are absent")
    frames = {stage: rung0.assemble_frame(frames_root, 1, stage)
              for stage in range(4)}
    active = np.asarray(card.recipe.z_coord.is_active, dtype=bool)
    mf_u_np, mf_v_np = fold_walk._full_fluxes(frames[2], raw)
    mf_u, mf_v = jnp.asarray(mf_u_np), jnp.asarray(mf_v_np)
    h_base = compute_layer_thickness(
        jnp.asarray(frames[0]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    h_now = compute_layer_thickness(
        jnp.asarray(frames[2]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    h_after = compute_layer_thickness(
        jnp.asarray(frames[3]["ssh"]), card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord, card.recipe.model_config.min_water_column_m)
    w = diagnose_w_from_flux_div(
        divergence_cgrid(mf_u, mf_v, card.recipe.grid), card.recipe.z_coord,
        thickness_weighted=True)
    zero_w = jnp.zeros_like(w)

    def replay(now, before):
        dh, dv, trace = advection.fct_tracer_advection(
            now, mf_u, mf_v, w, h_now, card.recipe.grid, card.dt_s,
            high_order="centred2", tracer_before=before,
            active_mask=jnp.asarray(active),
            low_order_predictor="nemo_rk3_two_step",
            base_thickness=h_base, after_thickness=h_after,
            implicit_w=zero_w, return_nemo_trace=True)
        generic = jnp.where(
            jnp.asarray(active), -(dh + dv) / h_now, jnp.zeros_like(dh))
        literal = trace[advection.NEMO_FCT_TRACE_FIELDS.index("rhs_final")]
        return generic, literal

    results = {}
    for tracer in ("T", "S"):
        results[tracer] = tuple(np.asarray(value) for value in jax.jit(replay)(
            jnp.asarray(frames[2][tracer]), jnp.asarray(frames[0][tracer])))

    record = phase3.read_tracer_stage3(
        frames_root / "oracle_rktracer_stage3_kt00000001.bin")
    owned_kmm = np.asarray(record["Kmm_T"])[2:-2, 2:-2, :-1]
    halves = {
        "south": np.asarray(frames[2]["T"])[:owned_kmm.shape[0]],
        "north": np.asarray(frames[2]["T"])[-owned_kmm.shape[0]:],
    }
    half_counts = {
        name: int(np.count_nonzero(field != owned_kmm))
        for name, field in halves.items()
    }
    matches = [name for name, count in half_counts.items() if count == 0]
    require(len(matches) == 1, "rank-0 Kmm record has ambiguous ownership")
    matched = matches[0]
    sl = slice(0, owned_kmm.shape[0]) if matched == "south" else slice(-owned_kmm.shape[0], None)
    support = active[sl]

    tracer_rows = {}
    for tracer in ("T", "S"):
        generic, literal = results[tracer]
        oracle = np.asarray(record[f"after_advection_{tracer}"])[2:-2, 2:-2, :-1]
        tracer_rows[tracer] = {
            "literal_vs_generic": _comparison(generic, literal, active),
            "generic_vs_oracle": _comparison(oracle, generic[sl], support),
            "literal_vs_oracle": _comparison(oracle, literal[sl], support),
        }

    return {
        "format": "nemo-testcase-l4-orca2-round226-fct-rhs-v1",
        "execution": "offline-pure-jit-cpu-fp64-libm",
        "record_scope": "admitted OMT-4 kt=1 stage-3 state and rank-0 RKTR3 record",
        "in_executable_observers": 0,
        "record_alignment": {
            "matched_half": matched,
            "Kmm_unequal": half_counts[matched],
            "other_half_unequal": half_counts["north" if matched == "south" else "south"],
        },
        "tracers": tracer_rows,
        "first_nonbit_statement": "final_FCT_Krhs_association",
        "stage_consumer": "LIVE_WITH_SPEC",
        "statement_sufficiency": "UNMEASURED_WITH_SPEC",
        "worktree": stamp,
        "compiled_citations": {
            "upstream_rhs": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:598-609",
            "limited_rhs": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:318-330",
            "stage_consumer": "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:600-649,700-760",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(args.deck_root is None and args.frames_root is None
                    and args.expect_commit is None,
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(args.deck_root and args.frames_root and args.expect_commit,
                    "runtime mode requires deck, frames, and commit")
            raw = measure(args.deck_root, args.frames_root, args.expect_commit)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, omt4.GateError, phase3.GateError,
            OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R226_FIRST_NONBIT_FINAL_FCT_RHS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
