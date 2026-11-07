#!/usr/bin/env python3
"""Round-66 reciprocal substitution of the GYRE stage-3 content operands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46
import nemo_testcase_l2_gyre_round54_tracer_decomposition as round54
import nemo_testcase_overflow_barotropic_gate as baro
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from legoesm.ocean.vertical import compute_layer_thickness

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
OPERANDS = ("tracer_Kbb", "e3t_Kbb", "p2dt", "e3t_Kmm", "Krhs")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def transposed(value):
    """Map NEMO inner-domain i,j,k storage to legoESM j,i,k storage."""
    return np.ascontiguousarray(
        np.asarray(value, dtype=np.float64).transpose(1, 0, 2))


def content_statement(operands: dict[str, np.ndarray | np.float64]):
    """Compiled trazdf.f90:548-562 association, without reassociation."""
    left = operands["e3t_Kbb"] * operands["tracer_Kbb"]
    tendency = np.float64(operands["p2dt"]) * operands["e3t_Kmm"]
    tendency = tendency * operands["Krhs"]
    return left + tendency


def reciprocal_substitutions(
    live: dict[str, np.ndarray | np.float64],
    oracle: dict[str, np.ndarray | np.float64],
    oracle_content: np.ndarray,
    wet: np.ndarray,
) -> dict:
    """Score one-operand substitutions in both directions."""
    rows = {
        "live_baseline": round54.field_stats(
            content_statement(live), oracle_content, wet),
        "oracle_baseline": round54.field_stats(
            content_statement(oracle), oracle_content, wet),
        "oracle_into_live": {},
        "live_into_oracle": {},
    }
    for name in OPERANDS:
        arm = dict(live)
        arm[name] = oracle[name]
        rows["oracle_into_live"][name] = round54.field_stats(
            content_statement(arm), oracle_content, wet)
        reverse = dict(oracle)
        reverse[name] = live[name]
        rows["live_into_oracle"][name] = round54.field_stats(
            content_statement(reverse), oracle_content, wet)
    return rows


def _capture_final_content_call(card, seeded, freshwater, surface):
    """Run the production step and capture the operands of its final WS call."""
    captured: list[dict[str, np.ndarray]] = []
    captured_ldf: list[dict[str, np.ndarray]] = []
    real_pair_step = model_module._nemo_ws_rk3_tracer_pair_step
    real_gm_redi = model_module.gm_redi_tracer_tendency_latlon

    def sink(*values):
        names = (
            "tracer_Kbb_T", "tracer_Kbb_S", "e3t_Kbb", "e3t_Kaa",
            "p2dt", "tracer_Kmm_T", "tracer_Kmm_S", "source_T",
            "source_S", "content_T", "content_S", "advection_content_T",
            "advection_content_S",
        )
        captured.append({
            name: np.asarray(value, dtype=np.float64)
            for name, value in zip(names, values, strict=True)
        })

    def capture_pair_step(*args, **kwargs):
        result = real_pair_step(*args, **kwargs)
        if kwargs.get("return_final_content", False):
            resume = kwargs.get("resume")
            require(resume is not None and resume[0] == 2,
                    "final WS content call did not resume from stage 2")
            source_t, source_s = kwargs["stage_source_rates"][2]
            jax.debug.callback(
                sink,
                args[0], args[1], args[6], args[7], jnp.asarray(args[11]),
                resume[1], resume[2], source_t, source_s,
                result[2], result[3], result[4], result[5],
                ordered=True,
            )
        return result

    def ldf_sink(dT, dS):
        captured_ldf.append({
            "T": np.asarray(dT, dtype=np.float64),
            "S": np.asarray(dS, dtype=np.float64),
        })

    def capture_gm_redi(*args, **kwargs):
        result = real_gm_redi(*args, **kwargs)
        require(len(result) == 2,
                "GYRE GM/Redi unexpectedly returned a bolus transport")
        jax.debug.callback(ldf_sink, result[0], result[1], ordered=True)
        return result

    model_module._nemo_ws_rk3_tracer_pair_step = capture_pair_step
    model_module.gm_redi_tracer_tendency_latlon = capture_gm_redi
    try:
        trace = model_module.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                expose_live_stage_operands=True),
        ).step(seeded, dt=card.dt_s, freshwater=freshwater,
               surface_forcing=surface)
        trace = jax.device_get(trace)
    finally:
        model_module._nemo_ws_rk3_tracer_pair_step = real_pair_step
        model_module.gm_redi_tracer_tendency_latlon = real_gm_redi
    require(len(captured) >= 1,
            "final content callback did not fire")
    # ``step`` exposes the captured arrays in both the state and private trace
    # pytrees.  Current JAX may therefore execute this ordered callback twice.
    # Collapse only byte-identical observations; a genuinely distinct second
    # production evaluation remains a hard failure.
    for duplicate in captured[1:]:
        require(duplicate.keys() == captured[0].keys(),
                "duplicate content callback schema changed")
        differences = {}
        for key in captured[0]:
            left = np.asarray(duplicate[key], dtype=np.float64)
            right = np.asarray(captured[0][key], dtype=np.float64)
            changed = left.view(np.uint64) != right.view(np.uint64)
            if np.any(changed):
                differences[key] = {
                    "cells_unequal": int(np.count_nonzero(changed)),
                    "max_abs": float(np.max(np.abs(left - right))),
                }
        require(not differences,
                "final content callback observed distinct evaluations: "
                f"{differences}")
    require(len(captured_ldf) >= 1, "GM/Redi callback did not fire")
    for duplicate in captured_ldf[1:]:
        require(all(np.array_equal(duplicate[key], captured_ldf[0][key])
                    for key in captured_ldf[0]),
                "GM/Redi callback observed distinct evaluations")
    return trace, captured[0], captured_ldf[0]


def measure(args) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "round-66 producer worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(),
            f"commit stamp mismatch: {stamp['commit']} != {args.expect_commit}")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    admission = json.loads(args.admission.read_text())
    require(admission.get("verdict") == "PASS", "round-64 record not admitted")
    require(admission.get("byte_identical_records") == 43,
            "round-64 exact-record census changed")
    require(len(admission.get("classified_changed_records", [])) == 20,
            "round-64 changed-record census changed")
    require(admission.get("admitted_difference_count") == 132,
            "round-64 admitted-value census changed")
    producer = args.producer_commit.read_text().strip().lower()
    require(producer == args.expect_record_commit.lower(),
            f"record producer mismatch: {producer}")
    round54._verify_r63_stamp(args.krhs_record, args.krhs_stamp, producer)
    record = round54._read_r63_stream(args.krhs_record, kind="krhs")
    calibration = round54._calibrate_r63_krhs(record["arrays"])

    previous_tke = round46.read_stage(
        args.stage_root / "oracle_momstage_kt00000001_s1.bin")
    stage3 = round46.read_stage(
        args.stage_root / "oracle_momstage_kt00000002_s3.bin")
    entry2 = gate.read_entry(
        args.entry_root / "oracle_step_entry_kt00000002.bin")
    frames = gate.read_bt(
        args.entry_root / "oracle_bt_frames_kt00000001.bin", 1)
    card = build_nemo_testcase_card("GYRE-zco")
    require(card.recipe.model_config.tracer_time_integrator == "rk3_ws",
            "GYRE card no longer selects the WS-RK3 tracer program")
    require(card.recipe.model_config.tracer_advection == "fct2",
            "GYRE card no longer selects FCT2")
    masks = gate.expected_masks(card)
    base_model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    initial = card.recipe.initial_state
    freshwater1, surface1 = gate._surface_forcings(card, initial, 1)
    free_entry = base_model.step(initial, dt=card.dt_s,
                                 freshwater=freshwater1,
                                 surface_forcing=surface1)
    seeded = baro.state_from_oracle_entry(free_entry, entry2, masks)
    seeded = round54._bridge_barotropic(seeded, frames, masks)
    seeded = round54._bridge_tke(seeded, previous_tke["arrays"])
    freshwater2, surface2 = gate._surface_forcings(card, seeded, 2)
    trace, captured, captured_ldf = _capture_final_content_call(
        card, seeded, freshwater2, surface2)

    geom = trace.stage_geometry[2]
    stage_state = trace.stage_states[2]
    h_after = compute_layer_thickness(
        trace.barotropic_targets[4], seeded.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    active = jnp.asarray(card.recipe.z_coord.is_active, dtype=jnp.float64)
    wall = active if getattr(
        card.recipe.model_config, "tracer_wall_neumann_fill", True) else None

    def final_flux_divergences():
        return model_module.compute_advection_flux_div_pair(
            stage_state[2], stage_state[3], "fct2",
            geom[0], geom[1], geom[2], geom[3], geom[4], geom[5],
            card.recipe.grid, card.dt_s,
            recon_fill_mask=wall,
            linssh_top_flux=getattr(
                card.recipe.z_coord, "linear_free_surface", False),
            tr_a_before=seeded.T.data, tr_b_before=seeded.S.data,
            fct_low_order_predictor="nemo_rk3_two_step",
            fct_base_thickness=captured["e3t_Kbb"],
            fct_after_thickness=h_after,
            fct_implicit_w=geom[6],
        )

    pair_t, pair_s = jax.device_get(jax.jit(final_flux_divergences)())
    flux_div = {
        "T": np.asarray(pair_t[0] + pair_t[1], dtype=np.float64),
        "S": np.asarray(pair_s[0] + pair_s[1], dtype=np.float64),
    }
    content_h_kmm = np.asarray(
        np.float64(0.5) * (captured["e3t_Kbb"] + captured["e3t_Kaa"]),
        dtype=np.float64)
    transport_h_kmm = np.asarray(geom[3], dtype=np.float64)
    p2dt = np.float64(np.asarray(captured["p2dt"]).item())
    require(p2dt == np.float64(card.dt_s),
            f"live p2dt {p2dt!r} != full card dt {card.dt_s!r}")

    arrays = record["arrays"]
    stage_arrays = stage3["arrays"]
    rows = {}
    operand_identity = {}
    split_identity = {}
    live_krhs_identity = {}
    ldf_boundary_rows = {}
    for name, stage_index in (("T", 2), ("S", 3)):
        wet = masks[name]
        source = captured[f"source_{name}"]
        # The split helper consumes the already thickness-weighted FCT
        # divergence but weights the source with its locally reconstructed
        # Kmm.  This effective Krhs makes that production statement exactly
        # expressible in NEMO's five-operand association.  Keep the transport
        # Kmm identity separate below: equating the two would hide the operand
        # distinction this walk is intended to test.
        krhs_live = -flux_div[name] / content_h_kmm + source
        live = {
            "tracer_Kbb": captured[f"tracer_Kbb_{name}"],
            "e3t_Kbb": captured["e3t_Kbb"],
            "p2dt": p2dt,
            "e3t_Kmm": content_h_kmm,
            "Krhs": krhs_live,
        }
        oracle = {
            "tracer_Kbb": transposed(arrays[f"{name}_Kbb"]),
            "e3t_Kbb": transposed(arrays["e3t_Kbb"]),
            "p2dt": np.float64(arrays["p2dt"]),
            "e3t_Kmm": transposed(arrays["e3t_Kmm"]),
            "Krhs": transposed(arrays[f"after_ldf_{name}"]),
        }
        oracle_content = transposed(arrays[f"content_{name}"])
        rows[name] = reciprocal_substitutions(
            live, oracle, oracle_content, wet)
        operand_identity[name] = {
            operand: round54.field_stats(
                np.broadcast_to(live[operand], oracle_content.shape),
                np.broadcast_to(oracle[operand], oracle_content.shape), wet)
            for operand in OPERANDS
        }
        operand_identity[name]["tracer_Kmm"] = round54.field_stats(
            np.asarray(stage_state[stage_index], dtype=np.float64),
            round46._owned3(stage_arrays[f"{name}_Kmm"]), wet)
        operand_identity[name]["transport_e3t_Kmm"] = round54.field_stats(
            transport_h_kmm, oracle["e3t_Kmm"], wet)
        split_identity[name] = {
            "captured_split_content_vs_oracle": round54.field_stats(
                captured[f"content_{name}"], oracle_content, wet),
            "live_statement_vs_captured_split": round54.field_stats(
                content_statement(live), captured[f"content_{name}"], wet),
            "captured_advection_rebuild": round54.field_stats(
                live["e3t_Kbb"] * live["tracer_Kbb"]
                - live["p2dt"] * flux_div[name],
                captured[f"advection_content_{name}"], wet),
        }
        live_krhs_identity[name] = round54.field_stats(
            krhs_live, oracle["Krhs"], wet)
        live_krhs_identity[name]["transport_normalized"] = round54.field_stats(
            -flux_div[name] / transport_h_kmm + source,
            oracle["Krhs"], wet)
        ldf_rate = captured_ldf[name] * np.asarray(active, dtype=np.float64)
        boundary_arrays = {
            boundary: transposed(arrays[f"{boundary}_{name}"])
            for boundary in ("after_adv", "after_sbc", "after_qsr", "after_ldf")
        }
        oracle_ldf = boundary_arrays["after_ldf"] - boundary_arrays["after_qsr"]
        corrected_krhs = krhs_live + ldf_rate
        corrected_operands = dict(live)
        corrected_operands["Krhs"] = corrected_krhs
        reverse_operands = dict(oracle)
        reverse_operands["Krhs"] = oracle["Krhs"] - ldf_rate
        ldf_boundary_rows[name] = {
            "live_vs_cumulative": {
                boundary: round54.field_stats(krhs_live, values, wet)
                for boundary, values in boundary_arrays.items()
            },
            "captured_ldf_vs_oracle_increment": round54.field_stats(
                ldf_rate, oracle_ldf, wet),
            "live_plus_ldf_vs_after_ldf": round54.field_stats(
                corrected_krhs, oracle["Krhs"], wet),
            "live_plus_ldf_content_vs_oracle": round54.field_stats(
                content_statement(corrected_operands), oracle_content, wet),
            "oracle_minus_live_ldf_content_vs_oracle": round54.field_stats(
                content_statement(reverse_operands), oracle_content, wet),
        }

    planted_oracle = {
        key: (np.array(value, copy=True) if np.ndim(value) else value)
        for key, value in {
            "tracer_Kbb": transposed(arrays["T_Kbb"]),
            "e3t_Kbb": transposed(arrays["e3t_Kbb"]),
            "p2dt": np.float64(arrays["p2dt"]),
            "e3t_Kmm": transposed(arrays["e3t_Kmm"]),
            "Krhs": transposed(arrays["after_ldf_T"]),
        }.items()
    }
    first_wet = tuple(np.argwhere(masks["T"])[0])
    baseline_plant = content_statement(planted_oracle)
    planted_oracle["tracer_Kbb"][first_wet] = np.nextafter(
        planted_oracle["tracer_Kbb"][first_wet], np.float64(np.inf))
    plant_row = round54.field_stats(
        content_statement(planted_oracle), baseline_plant, masks["T"])
    require(plant_row["cells_unequal"] == 1 and plant_row["max_abs"] > 0.0,
            "one-ULP operand plant did not fire")
    planted_ldf = np.array(captured_ldf["T"], copy=True)
    planted_ldf[first_wet] = np.nextafter(
        planted_ldf[first_wet], np.float64(np.inf))
    ldf_plant_row = round54.field_stats(
        planted_ldf, captured_ldf["T"], masks["T"])
    require(ldf_plant_row["cells_unequal"] == 1,
            "one-ULP captured-LDF plant did not fire")

    t_ldf_content = ldf_boundary_rows["T"]["live_plus_ldf_content_vs_oracle"]
    t_baseline = rows["T"]["live_baseline"]
    ldf_confirmed = (
        t_ldf_content["max_abs"] <= 2.0e-6
        and t_ldf_content["max_abs"] <= t_baseline["max_abs"] / 5.0
    )

    return {
        "format": "nemo-testcase-l2-gyre-round66-content-operands-v1",
        "status": "MEASURED",
        "worktree": stamp,
        "record_producer": producer,
        "admission_counts": {"exact": 43, "total": 63, "changed": 20,
                             "admitted": 132},
        "calibration": calibration,
        "dtype": {"live_content": str(captured["content_T"].dtype),
                  "live_geometry": str(captured["e3t_Kbb"].dtype),
                  "record": str(arrays["content_T"].dtype)},
        "p2dt": {"live": p2dt, "oracle": np.float64(arrays["p2dt"]),
                 "card_dt_s": np.float64(card.dt_s)},
        "operand_identity": operand_identity,
        "live_krhs_identity": live_krhs_identity,
        "ldf_boundary_rows": ldf_boundary_rows,
        "split_statement_identity": split_identity,
        "substitution_rows": rows,
        "controls": {"one_ulp_operand": plant_row,
                     "one_ulp_captured_ldf": ldf_plant_row},
        "prediction": "left-product tracer level (Kmm used where Kbb required)",
        "falsifier": "no reciprocal single substitution reduces T max to <=2e-6",
        "ldf_prediction": "CONFIRMED" if ldf_confirmed else "REFUTED",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-root", type=Path,
                        default=ROOT / "year_owners/nemo_seed0")
    parser.add_argument("--stage-root", type=Path,
                        default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--krhs-record", type=Path, default=(
        ROOT / "round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin"))
    parser.add_argument("--krhs-stamp", type=Path, default=(
        ROOT / "round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin.stamp"))
    parser.add_argument("--producer-commit", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/producer_commit.txt")
    parser.add_argument("--admission", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/round64_admission.json")
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    print(
        "ROUND66 CONTENT OPERANDS PASS: "
        f"T_live_max={report['substitution_rows']['T']['live_baseline']['max_abs']:.12e} "
        f"S_live_max={report['substitution_rows']['S']['live_baseline']['max_abs']:.12e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
