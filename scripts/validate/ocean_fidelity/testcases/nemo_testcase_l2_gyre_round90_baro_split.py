#!/usr/bin/env python3
"""Split live kt2 stage-one barotropic correction against its direct record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
import nemo_testcase_l2_gyre_round90_baro_record as round90  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.dynamics.barotropic_common import (  # noqa: E402
    _ascending_level_sum,
    rk3_stage_barotropic_correction,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _native(value, face: str):
    value = np.asarray(value)
    return value[:, 1:, ...] if face == "u" else value[1:, ...]


def _owned(value):
    return np.asarray(value)[2:-2, 2:-2]


def _stats(candidate, reference, active) -> dict[str, object]:
    candidate = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64)[active])
    reference = np.ascontiguousarray(np.asarray(reference, dtype=np.float64)[active])
    require(candidate.size > 0, "comparison has no active values")
    require(candidate.shape == reference.shape,
            f"comparison shapes differ: {candidate.shape} != {reference.shape}")
    require(np.isfinite(candidate).all() and np.isfinite(reference).all(),
            "comparison has a non-finite value")
    unequal = candidate.view(np.uint64) != reference.view(np.uint64)
    return {
        "n": int(candidate.size),
        "n_unequal": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(np.abs(candidate - reference))),
        "bit_exact": bool(not np.any(unequal)),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
    }


@jax.jit
def _correction_terms(raw, thickness, reciprocal, target, mask):
    products = raw * thickness
    own_mean = _ascending_level_sum(products) * reciprocal
    correction = target - own_mean
    final = (raw + correction[..., None]) * mask
    return products, own_mean, correction, final


@jax.jit
def _correction(raw, target, thickness, reciprocal, mask):
    return rk3_stage_barotropic_correction(
        raw, target, thickness, reciprocal, mask)


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy changed")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-92 operand measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            "Round-92 commit stamp mismatch")

    correction_record = round90.read_record(
        args.record_root / "oracle_baro_correction_kt00000002_s1.bin")
    arrays = correction_record["arrays"]
    stage = round46.read_stage(
        args.record_root / "oracle_momstage_kt00000002_s1.bin")
    stage_arrays = stage["arrays"]
    require(stage["header"]["kt"] == 2 and stage["header"]["stage"] == 1,
            "correction companion is not the kt2 stage-one record")

    base, context = round72._capture_seeded_context(SimpleNamespace(
        expect_commit=args.expect_commit,
        expect_krhs_commit=args.expect_krhs_commit,
        output=args.prerequisite_output,
    ))
    require(base["status"] == "MEASURED", "kt2 seeded context changed")
    card, seeded, freshwater, surface, _ = context
    masks = gate.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    ordinary = jax.device_get(ordinary_model.step(
        seeded, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    )
    trace = jax.device_get(trace_model.step(
        seeded, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    ordinary_fields = gate.lego_fields(ordinary)
    trace_fields = gate.lego_fields(trace.state_after)
    observer = {
        name: _stats(trace_fields[name], ordinary_fields[name], masks[name])
        for name in ("T", "S", "u", "v", "ssh")
    }
    require(all(row["bit_exact"] for row in observer.values()),
            "live trace changed ordinary output")

    live_h_u, live_h_v, live_r1_u, live_r1_v = (
        trace.barotropic_correction_geometry)
    live_geometry = {
        "u": (_native(live_h_u, "u"), _native(live_r1_u, "u")),
        "v": (_native(live_h_v, "v"), _native(live_r1_v, "v")),
    }

    rows: list[dict[str, object]] = []
    faces: dict[str, dict[str, object]] = {}
    cross_record: dict[str, dict[str, object]] = {}
    for face in ("u", "v"):
        state_index = 0 if face == "u" else 1
        mask3 = _owned(arrays[f"baro_{face}mask"])[..., :nlev] > 0.5
        mask2 = mask3[..., 0]
        oracle_kbb = round46._owned3(stage_arrays[f"{face}_Kbb"], nlev)
        oracle_rhs = round46._owned3(stage_arrays[f"after_adv_{face}"], nlev)
        oracle_raw = _owned(arrays[f"baro_raw_{face}"])[..., :nlev]
        oracle_target = _owned(arrays[f"baro_target_{face}"])
        # The Round-90 writer's ``baro_e3*_0`` payload is a static reference
        # and cannot calibrate the compiled QCO statement, whose preprocessed
        # source consumes ``e3*_3d`` at Kaa. The byte-identical companion
        # stage record from the same admitted run carries that live Kaa array.
        oracle_h = round46._owned3(
            stage_arrays[f"e3{face}_Kaa"], nlev)
        oracle_r1 = _owned(arrays[f"baro_r1_h{face}_0"])
        oracle_z = np.asarray(arrays[
            "baro_zub" if face == "u" else "baro_zvb"])
        oracle_final = _owned(arrays[f"baro_final_{face}"])[..., :nlev]

        companion_raw = round46._owned3(
            stage_arrays[f"post_update_{face}"], nlev)
        companion_final = round46._owned3(
            stage_arrays[f"post_baro_{face}"], nlev)
        cross_record[face] = {
            "raw": _stats(companion_raw, oracle_raw, mask3),
            "final": _stats(companion_final, oracle_final, mask3),
            "round90_static_e3_vs_executing_Kaa": _stats(
                _owned(arrays[f"baro_e3{face}_0"])[..., :nlev],
                oracle_h, mask3),
        }
        require(all(cross_record[face][name]["bit_exact"]
                    for name in ("raw", "final")),
                f"{face} correction and companion records disagree")

        live_kbb = _native(trace.stage_states[0][state_index], face)
        live_rhs = _native(trace.stage_rhs[0][state_index], face)
        live_raw = _native(trace.stage_raw_velocities[0][state_index], face)
        live_target = _native(trace.barotropic_targets[state_index], face)
        live_final = _native(trace.stage_states[1][state_index], face)
        live_h, live_r1 = live_geometry[face]

        oracle_terms = tuple(np.asarray(value) for value in _correction_terms(
            jnp.asarray(oracle_raw), jnp.asarray(oracle_h),
            jnp.asarray(oracle_r1), jnp.asarray(oracle_target),
            jnp.asarray(mask3, dtype=jnp.float64)))
        live_terms = tuple(np.asarray(value) for value in _correction_terms(
            jnp.asarray(live_raw), jnp.asarray(live_h),
            jnp.asarray(live_r1), jnp.asarray(live_target),
            jnp.asarray(mask3, dtype=jnp.float64)))
        oracle_products, oracle_mean, computed_z, oracle_replay = oracle_terms
        live_products, live_mean, live_z, live_replay = live_terms

        planted_target = np.array(oracle_target, copy=True)
        if args.plant and face == "u":
            candidates = np.argwhere(mask2 & (planted_target != 0.0))
            require(candidates.size > 0, "target plant found no nonzero wet cell")
            at = tuple(candidates[np.argmax(np.abs(
                planted_target[tuple(candidates.T)]))])
            planted_target[at] = np.nextafter(planted_target[at], np.inf)
        clean_calibrated = np.asarray(_correction(
            jnp.asarray(oracle_raw), jnp.asarray(oracle_target),
            jnp.asarray(oracle_h), jnp.asarray(oracle_r1),
            jnp.asarray(mask3, dtype=jnp.float64)))
        calibration_target = planted_target if args.plant and face == "u" else oracle_target
        calibrated = np.asarray(_correction(
            jnp.asarray(oracle_raw), jnp.asarray(calibration_target),
            jnp.asarray(oracle_h), jnp.asarray(oracle_r1),
            jnp.asarray(mask3, dtype=jnp.float64)))

        named = (
            ("Kbb", live_kbb, oracle_kbb, mask3, "DIRECT"),
            ("RHS", live_rhs, oracle_rhs, mask3, "DIRECT"),
            ("raw_Kaa", live_raw, oracle_raw, mask3, "DIRECT"),
            ("e3_ref", live_h, oracle_h, mask3, "DIRECT"),
            ("r1_depth_ref", live_r1, oracle_r1, mask2, "DIRECT"),
            ("weighted_products", live_products, oracle_products, mask3, "DERIVED"),
            ("own_mean", live_mean, oracle_mean, mask2, "DERIVED"),
            ("target", live_target, oracle_target, mask2, "DIRECT"),
            ("correction", live_z, oracle_z, mask2, "DIRECT"),
            ("final", live_final, oracle_final, mask3, "DIRECT"),
        )
        face_rows = {}
        for boundary, candidate, reference, active, provenance in named:
            result = _stats(candidate, reference, active)
            result.update({
                "name": f"GYRE-zco.kt2.s1.baro.{boundary}.{face}",
                "provenance": provenance,
            })
            rows.append(result)
            face_rows[boundary] = result

        direct_z_replay = _stats(computed_z, oracle_z, mask2)
        direct_z_replay["name"] = f"GYRE-zco.kt2.s1.baro.given_z_replay.{face}"
        direct_z_replay["provenance"] = "DIRECT NEMO OUTPUT CALIBRATION"
        rows.append(direct_z_replay)
        face_rows["given_z_replay"] = direct_z_replay
        given_input = _stats(clean_calibrated, oracle_final, mask3)
        given_input["name"] = f"GYRE-zco.kt2.s1.baro.given_input_replay.{face}"
        given_input["provenance"] = "DIRECT NEMO OUTPUT CALIBRATION"
        rows.append(given_input)
        face_rows["given_input_replay"] = given_input
        replay_actual = _stats(live_replay, live_final, mask3)
        replay_actual["name"] = f"GYRE-zco.kt2.s1.baro.live_terms_replay.{face}"
        replay_actual["provenance"] = "LIVE TRACE SELF-CALIBRATION"
        rows.append(replay_actual)
        face_rows["live_terms_replay"] = replay_actual
        faces[face] = {
            "rows": face_rows,
            "plant_effect": _stats(calibrated, clean_calibrated, mask3),
        }

    if args.plant:
        require(not faces["u"]["plant_effect"]["bit_exact"],
                "one-ULP target plant was invisible")
        status = "PLANT_FIRED"
    else:
        calibration_exact = all(
            faces[face]["rows"][name]["bit_exact"]
            for face in ("u", "v")
            for name in ("given_z_replay", "given_input_replay",
                         "live_terms_replay"))
        status = "MEASURED" if calibration_exact else "REFUTED"

    return {
        "format": "nemo-testcase-l2-gyre-round92-baro-split-v1",
        "status": status,
        "worktree": stamp,
        "dtype": {
            "state": str(np.asarray(trace.stage_states[0][0]).dtype),
            "geometry": str(np.asarray(live_h_u).dtype),
            "record": str(np.asarray(arrays["baro_raw_u"]).dtype),
        },
        "source": {
            "assignment": "stprk3_stg.f90:668-687",
            "correction": "stprk3_stg.f90:728-765",
            "target": "dynspg_ts.f90:763-804",
            "reciprocal": "domain.f90:195-215",
        },
        "base_status": base["status"],
        "observer": observer,
        "cross_record_identity": cross_record,
        "post_hoc_instrument_retraction": (
            "REFUTED: the initial static-versus-executing-thickness hypothesis "
            "does not explain the calibration miss. The compiled writer's "
            "e3u_3d/e3v_3d payloads and the companion e3u_Kaa/e3v_Kaa payloads "
            "are bit-identical; the companion remains a direct cross-record "
            "identity check."),
        "rows": rows,
        "faces": faces,
        "first_nonbit_direct": next(
            (row["name"] for row in rows
             if row["provenance"] == "DIRECT" and not row["bit_exact"]),
            None),
        "plant": args.plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--prerequisite-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        print(f"ROUND92 BARO SPLIT FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND92_BARO_SPLIT", report["status"])
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
