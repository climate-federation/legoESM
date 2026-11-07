#!/usr/bin/env python3
"""Round 89 source-order walk across GYRE kt2's RK3 momentum assignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
    rk3_stage_velocity_update,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _row(name, candidate, reference, active, statement, execution):
    candidate = np.asarray(candidate)[active]
    reference = np.asarray(reference)[active]
    unequal = candidate.view(np.uint64) != reference.view(np.uint64)
    return {
        "name": name,
        "n": int(active.sum()),
        "n_unequal": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(np.abs(candidate - reference))),
        "bit_exact": bool(not np.any(unequal)),
        "nemo_statement": statement,
        "execution": execution,
    }


def _model_fields(card, seeded, freshwater, surface, hooks):
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    )
    state = jax.device_get(model.step(
        seeded, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    return gate.lego_fields(state)


def _candidate_face(fields, face):
    return np.asarray(fields[face])


def _reference_face(array):
    return round46._owned3(array)


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy changed")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-89 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-89 commit stamp mismatch")

    records = {
        stage: round46.read_stage(
            args.round46_root / f"oracle_momstage_kt00000002_s{stage}.bin")
        for stage in (1, 2, 3)
    }
    base, context = round72._capture_seeded_context(SimpleNamespace(
        expect_commit=args.expect_commit,
        expect_krhs_commit=args.expect_krhs_commit,
        output=args.prerequisite_output,
    ))
    require(base["status"] == "MEASURED", "kt2 seeded context changed")
    card, seeded, freshwater, surface, _ = context
    masks = gate.expected_masks(card)

    rows = []
    # First prove the one changed operator on each recorded NEMO stage input.
    # Stages 1/2 expose NEMO's direct post-update output; stage 3's bare
    # explicit assignment has no dumped output and is labelled transcription.
    for stage in (1, 2, 3):
        arrays = records[stage]["arrays"]
        rhs_name = "pre_zdf_rhs" if stage == 3 else "after_adv"
        # The Round-46 callback stores the module reciprocal before the
        # stage-1 SELECT CASE resets it, so that payload is stale only for
        # stage 1.  Reconstruct rDt from the compiled stage clock at
        # stprk3_stg.f90:140-148,196-202,240-246 for every stage instead of
        # silently mixing clock sources.
        divisor = {1: 3.0, 2: 2.0, 3: 1.0}[stage]
        rdt = np.float64(card.dt_s / divisor)
        for face in ("u", "v"):
            before = _reference_face(arrays[f"{face}_Kbb"])
            rhs = _reference_face(arrays[f"{rhs_name}_{face}"])
            active = _reference_face(arrays[f"{face}mask"]) > 0.5
            candidate = np.asarray(rk3_stage_velocity_update(
                before, rhs, rdt, active.astype(np.float64), vector_form=True))
            if stage < 3:
                reference = _reference_face(arrays[f"post_update_{face}"])
                output_kind = "NEMO output post_update"
            else:
                # No record is written between dyn_zdf's assignment and its
                # following solve.  This row is an exact transcription
                # identity; the composed Round-31 output proof is run beside
                # this probe and is the Rule-12 discharge.
                reference = (before + rdt * rhs) * active
                output_kind = "transcription identity; not a discharge"
            if args.plant and (stage, face) == (1, "u"):
                reference = reference.copy()
                at = tuple(np.argwhere(active)[0])
                reference[at] = np.nextafter(reference[at], np.inf)
            rows.append(_row(
                f"GYRE-zco.kt2.s{stage}.rk3_vector_assignment.{face}",
                candidate, reference, active,
                ("stprk3_stg.f90:671-674" if stage < 3
                 else "dynzdf.f90:166-170"),
                f"production helper on recorded NEMO inputs; {output_kind}",
            ))

    # Then walk the independently executing legoESM kt2 program.  These hooks
    # substitute diagnostics only after the ordinary JIT step has completed.
    for stage in (1, 2):
        fields = _model_fields(
            card, seeded, freshwater, surface,
            _NEMOWSRK3TestHooks(expose_momentum_stage=stage))
        arrays = records[stage]["arrays"]
        for face in ("u", "v"):
            rows.append(_row(
                f"GYRE-zco.kt2.s{stage}.post_baro.{face}",
                _candidate_face(fields, face),
                _reference_face(arrays[f"post_baro_{face}"]), masks[face],
                "stprk3_stg.f90:666-675,734-760",
                "independent production-JIT seeded kt2 stage exposure",
            ))

    for boundary, oracle_name in (
        ("pre_ldf", "after_adv"),
        ("post_ldf", "pre_zdf_rhs"),
    ):
        fields = _model_fields(
            card, seeded, freshwater, surface,
            _NEMOWSRK3TestHooks(expose_stage3_momentum_rhs=boundary))
        arrays = records[3]["arrays"]
        for face in ("u", "v"):
            rows.append(_row(
                f"GYRE-zco.kt2.s3.{boundary}_rhs.{face}",
                _candidate_face(fields, face),
                _reference_face(arrays[f"{oracle_name}_{face}"]), masks[face],
                ("stprk3_stg.f90:433-482" if boundary == "pre_ldf"
                 else "stprk3_stg.f90:692-714"),
                "independent production-JIT seeded kt2 RHS exposure",
            ))

    pre_implicit = _model_fields(
        card, seeded, freshwater, surface,
        _NEMOWSRK3TestHooks(expose_pre_implicit_state=True))
    arrays = records[3]["arrays"]
    rdt = np.float64(card.dt_s)
    for face in ("u", "v"):
        active = masks[face]
        expected = rk3_stage_velocity_update(
            _reference_face(arrays[f"{face}_Kbb"]),
            _reference_face(arrays[f"pre_zdf_rhs_{face}"]),
            rdt, active.astype(np.float64), vector_form=True)
        rows.append(_row(
            f"GYRE-zco.kt2.s3.pre_implicit_explicit_update.{face}",
            _candidate_face(pre_implicit, face), np.asarray(expected), active,
            "dynzdf.f90:166-170",
            "independent production-JIT pre-implicit exposure",
        ))

    completed = _model_fields(
        card, seeded, freshwater, surface, _NEMOWSRK3TestHooks())
    for face in ("u", "v"):
        rows.append(_row(
            f"GYRE-zco.kt2.s3.post_baro.{face}",
            _candidate_face(completed, face),
            _reference_face(arrays[f"post_baro_{face}"]), masks[face],
            "stprk3_stg.f90:728-760",
            "independent production-JIT seeded kt2 completed stage",
        ))

    if args.plant:
        require(not rows[0]["bit_exact"],
                "one-ULP assignment plant was invisible")
        status = "PLANT_FIRED"
    else:
        status = "MEASURED"
    return {
        "format": "nemo-testcase-l2-gyre-round89-stage-walk-v1",
        "status": status,
        "worktree": stamp,
        "round64_admission_status": "PASS",
        "rows": rows,
        "first_nonbit_statement": next(
            (row["name"] for row in rows if not row["bit_exact"]), None),
        "plant": args.plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--round46-root", type=Path,
                        default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--prerequisite-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = measure(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND89_STAGE_WALK", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
