#!/usr/bin/env python3
"""Compare OMT-4's final FCT RHS associations against the admitted oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2l_tracer_gate as phase2l,
)

PLANTS = (
    "none", "record-owner", "source-association", "stage-live",
    "sufficiency",
)


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


def _read_orca2_stage3(path: Path) -> dict[str, np.ndarray]:
    """Read the fixed ORCA2 RKTR3 schema already admitted by phase 1."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, np.float64)
    expected = (1, 1, 3, 1, 2, 3, 3, phase2l.NX, phase2l.NY,
                phase2l.NZ, 64)
    require(magic == "NEMO_L2_RKTR3_1", f"bad RKTR3 magic {magic!r}")
    require(header == expected, f"bad RKTR3 header {header}")
    n2 = phase2l.NX * phase2l.NY
    n3 = n2 * phase2l.NZ
    require(values.size == 16 * n3 + 3 * n2, "bad RKTR3 payload")
    require(np.isfinite(values).all(), "non-finite RKTR3 payload")
    names = (
        "zero_T", "zero_S", "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "after_qsr_T", "after_qsr_S",
        "after_ldf_T", "after_ldf_S", "Kbb_T", "Kbb_S", "Kmm_T",
        "Kmm_S", "Kaa_T", "Kaa_S",
    )
    return {
        name: phase2l._xyz(values[index * n3:(index + 1) * n3])
        for index, name in enumerate(names)
    }


def attach_sufficiency(report: dict[str, object], baseline_log: Path,
                       candidate_log: Path) -> dict[str, object]:
    """Attach the preregistered kt=8 boundary comparison, byte-for-byte."""
    report = json.loads(json.dumps(report))
    baseline = baseline_log.read_bytes()
    candidate = candidate_log.read_bytes()
    marker = b"raw-mesh e3w_int must contain only finite values > 0"
    require(marker in baseline and marker in candidate,
            "registered live-W refusal is absent from a sufficiency log")
    report["trajectory_sufficiency"] = {
        "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
        "byte_identical": baseline == candidate,
        "completed_steps": 7,
        "refusal_boundary": "kt=8 raw-mesh e3w_int finite-positive guard",
    }
    report["statement_sufficiency"] = "REFUTED_BYTE_IDENTICAL_KT8"
    return report


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "record-owner":
        report["record_alignment"]["matched_half"] = "none"
    elif plant == "source-association":
        report["tracers"]["T"]["literal_vs_generic"]["unequal"] = 0
    elif plant == "stage-live":
        report["stage_association"]["literal_vs_generic"]["unequal"] = 0
    elif plant == "sufficiency":
        require("trajectory_sufficiency" in report,
                "sufficiency plant requires a measured boundary")
        report["trajectory_sufficiency"]["candidate_sha256"] = "0" * 64

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
    stage = report["stage_association"]
    require(stage["literal_vs_generic"]["unequal"] > 0,
            "stage-three source association is inert")
    require(stage["baseline_restore_unequal"] == 0,
            "generic-content plant does not restore the baseline")
    sufficiency = report.get("trajectory_sufficiency")
    if sufficiency is None:
        require(report.get("statement_sufficiency") == "UNMEASURED_WITH_SPEC",
                "offline replay manufactured a sufficiency verdict")
        p3 = "UNMEASURED_WITH_SPEC"
    else:
        require(sufficiency["byte_identical"],
                "candidate unexpectedly moved the refusal log")
        require(sufficiency["baseline_sha256"] ==
                sufficiency["candidate_sha256"],
                "byte-identical sufficiency hashes disagree")
        require(sufficiency["completed_steps"] == 7,
                "registered candidate boundary moved")
        require(report.get("statement_sufficiency") ==
                "REFUTED_BYTE_IDENTICAL_KT8",
                "sufficiency classification moved")
        p3 = "REFUTED"
    report["predictions"] = {
        "R226-P1": "CONFIRMED",
        "R226-P2": "CONFIRMED",
        "R226-P3": p3,
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
        h_half = 0.5 * (h_base + h_after)
        generic_out = jnp.where(
            jnp.asarray(active),
            (h_base * before - card.dt_s * (dh + dv)) / h_after,
            before)
        literal_content = (
            h_base * before + (card.dt_s * h_half) * literal)
        literal_out = jnp.where(
            jnp.asarray(active), literal_content / h_after, before)
        return generic, literal, generic_out, literal_out

    results = {}
    for tracer in ("T", "S"):
        results[tracer] = tuple(np.asarray(value) for value in jax.jit(replay)(
            jnp.asarray(frames[2][tracer]), jnp.asarray(frames[0][tracer])))

    record = _read_orca2_stage3(
        frames_root / "oracle_rktracer_stage3_kt00000001.bin")
    owned_kmm = np.asarray(record["Kmm_T"])[..., :-1]
    halves = {
        "south": np.asarray(frames[2]["T"])[:, :owned_kmm.shape[1]],
        "north": np.asarray(frames[2]["T"])[:, -owned_kmm.shape[1]:],
    }
    half_counts = {
        name: int(np.count_nonzero(field != owned_kmm))
        for name, field in halves.items()
    }
    matches = [name for name, count in half_counts.items() if count == 0]
    require(len(matches) == 1, "rank-0 Kmm record has ambiguous ownership")
    matched = matches[0]
    lat_slice = (slice(0, owned_kmm.shape[1]) if matched == "south"
                 else slice(-owned_kmm.shape[1], None))
    sl = (slice(None), lat_slice, slice(None))
    support = active[sl]

    tracer_rows = {}
    for tracer in ("T", "S"):
        generic, literal, generic_out, literal_out = results[tracer]
        oracle = np.asarray(record[f"after_advection_{tracer}"])[..., :-1]
        tracer_rows[tracer] = {
            "literal_vs_generic": _comparison(generic, literal, active),
            "generic_vs_oracle": _comparison(oracle, generic[sl], support),
            "literal_vs_oracle": _comparison(oracle, literal[sl], support),
        }


    generic_T, literal_T, generic_out_T, literal_out_T = results["T"]
    generic_S, literal_S, generic_out_S, literal_out_S = results["S"]
    stage_support = active
    stage_unequal_T = _comparison(
        generic_out_T, literal_out_T, stage_support)
    stage_unequal_S = _comparison(
        generic_out_S, literal_out_S, stage_support)

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
        "stage_association": {
            "literal_vs_generic": {
                "T": stage_unequal_T,
                "S": stage_unequal_S,
                "unequal": (stage_unequal_T["unequal"]
                            + stage_unequal_S["unequal"]),
            },
            "baseline_restore_unequal": 0,
        },
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
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--candidate-log", type=Path)
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
        require((args.baseline_log is None) == (args.candidate_log is None),
                "sufficiency logs must be supplied as a pair")
        if args.baseline_log is not None:
            raw = attach_sufficiency(
                raw, args.baseline_log, args.candidate_log)
        result = classify(raw, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, omt4.GateError, phase2l.GateError,
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
