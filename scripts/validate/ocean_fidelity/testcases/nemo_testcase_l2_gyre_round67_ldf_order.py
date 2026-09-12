#!/usr/bin/env python3
"""Round-67 ordered Krhs substitutions and pre-edit LDF-routing prediction."""

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
import nemo_testcase_l2_gyre_round54_tracer_decomposition as round54
import nemo_testcase_l2_gyre_round66_content_operands as round66
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
TRACERS = ("T", "S")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _copy_operands(values: dict) -> dict:
    return {
        name: (np.array(value, dtype=np.float64, copy=True)
               if np.ndim(value) else np.float64(value))
        for name, value in values.items()
    }


def pytree_exact_census(got, want) -> dict:
    """Count exact differences over every array leaf of two state pytrees."""
    got_leaves, got_tree = jax.tree_util.tree_flatten(got)
    want_leaves, want_tree = jax.tree_util.tree_flatten(want)
    require(got_tree == want_tree, "state pytree structures differ")
    require(len(got_leaves) == len(want_leaves), "state leaf counts differ")
    cells = 0
    unequal = 0
    max_abs = 0.0
    for got_leaf, want_leaf in zip(got_leaves, want_leaves, strict=True):
        got_array = np.asarray(got_leaf)
        want_array = np.asarray(want_leaf)
        require(got_array.shape == want_array.shape,
                "corresponding state leaf shapes differ")
        cells += got_array.size
        unequal += int(np.count_nonzero(got_array != want_array))
        if got_array.size and np.issubdtype(got_array.dtype, np.number):
            max_abs = max(
                max_abs,
                float(np.max(np.abs(
                    got_array.astype(np.float64)
                    - want_array.astype(np.float64)))))
    return {
        "leaves": len(got_leaves),
        "cells": cells,
        "cells_unequal": unequal,
        "max_abs": max_abs,
    }


def cumulative_substitutions(
    *, live: dict, oracle: dict, oracle_content: np.ndarray,
    boundaries: dict[str, np.ndarray], source: np.ndarray,
    qsr: np.ndarray, ldf: np.ndarray, wet: np.ndarray,
    routed_krhs: np.ndarray | None = None,
) -> dict:
    """Replace cumulative Krhs boundaries in NEMO's compiled order."""
    live_adv = live["Krhs"] - source
    live_sbc = source - qsr
    components = {
        "advection": (live_adv, boundaries["after_adv"]),
        "sbc": (live_sbc, boundaries["after_sbc"] - boundaries["after_adv"]),
        "qsr": (qsr, boundaries["after_qsr"] - boundaries["after_sbc"]),
        "ldf": (ldf, boundaries["after_ldf"] - boundaries["after_qsr"]),
    }
    arms = {
        "routed_live": (
            live["Krhs"] + ldf if routed_krhs is None else routed_krhs),
        "nemo_post_sbc": boundaries["after_sbc"] + qsr + ldf,
        "nemo_post_qsr": boundaries["after_qsr"] + ldf,
        "nemo_post_ldf": boundaries["after_ldf"],
    }
    component_rows = {
        name: round54.field_stats(got, want, wet)
        for name, (got, want) in components.items()
    }
    arm_rows = {}
    arm_content = {}
    for name, krhs in arms.items():
        operands = dict(live)
        operands["Krhs"] = krhs
        content = round66.content_statement(operands)
        arm_content[name] = content
        arm_rows[name] = {
            "krhs": round54.field_stats(krhs, oracle["Krhs"], wet),
            "content": round54.field_stats(content, oracle_content, wet),
        }
    return {
        "component_rows": component_rows,
        "arm_rows": arm_rows,
        "arm_content": arm_content,
    }


def _one_ulp_plant(content: np.ndarray, wet: np.ndarray) -> dict:
    planted = np.array(content, dtype=np.float64, copy=True)
    first_wet = tuple(np.argwhere(wet)[0])
    planted[first_wet] = np.nextafter(
        planted[first_wet], np.float64(np.inf))
    row = round54.field_stats(planted, content, wet)
    require(row["cells_unequal"] == 1 and row["max_abs"] > 0.0,
            "one-ULP cumulative-content plant did not fire")
    return row


def production_fct_content(
    advection_content: np.ndarray,
    source: np.ndarray,
    ldf: np.ndarray,
    operands: dict,
) -> np.ndarray:
    """Use the production FCT content and NEMO's Kmm source association."""
    return (
        advection_content
        + operands["p2dt"] * operands["e3t_Kmm"] * (source + ldf)
    )


def measure(args) -> dict:
    reciprocal_calls: list[dict] = []
    pair_sources: list[tuple[np.ndarray, ...]] = []
    qsr_calls: list[np.ndarray] = []
    ldf_calls: list[tuple[np.ndarray, np.ndarray]] = []
    step_calls: list[tuple] = []

    real_reciprocal = round66.reciprocal_substitutions
    real_pair = model_module._nemo_ws_rk3_tracer_pair_step
    real_qsr = model_module._nemo_qsr_stage3_rate
    real_ldf = model_module.gm_redi_tracer_tendency_latlon
    real_step = model_module.LatLonCGridOceanModel.step

    def reciprocal_capture(live, oracle, oracle_content, wet):
        reciprocal_calls.append({
            "live": _copy_operands(live),
            "oracle": _copy_operands(oracle),
            "oracle_content": np.array(oracle_content, dtype=np.float64, copy=True),
            "wet": np.array(wet, dtype=bool, copy=True),
        })
        return real_reciprocal(live, oracle, oracle_content, wet)

    def pair_sink(
        source_t, source_s, tracer_t, tracer_s, content_t, content_s,
        advection_content_t, advection_content_s,
    ):
        pair_sources.append(tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                source_t, source_s, tracer_t, tracer_s, content_t, content_s,
                advection_content_t, advection_content_s)))

    def pair_capture(*values, **kwargs):
        result = real_pair(*values, **kwargs)
        if kwargs.get("return_final_content", False):
            source_t, source_s = kwargs["stage_source_rates"][2]
            jax.debug.callback(
                pair_sink, source_t, source_s, values[0], values[1],
                result[2], result[3], result[4], result[5],
                ordered=True)
        return result

    def qsr_sink(value):
        qsr_calls.append(np.asarray(value, dtype=np.float64))

    def qsr_capture(*values, **kwargs):
        result = real_qsr(*values, **kwargs)
        jax.debug.callback(qsr_sink, values[2], ordered=True)
        return result

    def ldf_sink(value_t, value_s):
        ldf_calls.append((np.asarray(value_t, dtype=np.float64),
                          np.asarray(value_s, dtype=np.float64)))

    def ldf_capture(*values, **kwargs):
        result = real_ldf(*values, **kwargs)
        require(len(result) == 2,
                "GYRE GM/Redi unexpectedly returned bolus transports")
        jax.debug.callback(ldf_sink, result[0], result[1], ordered=True)
        return result

    def step_capture(self, state, dt, freshwater=None, surface_forcing=None,
                     *values, **kwargs):
        step_calls.append((self, state, dt, freshwater, surface_forcing))
        return real_step(self, state, dt, freshwater, surface_forcing,
                         *values, **kwargs)

    round66.reciprocal_substitutions = reciprocal_capture
    model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
    model_module._nemo_qsr_stage3_rate = qsr_capture
    model_module.gm_redi_tracer_tendency_latlon = ldf_capture
    model_module.LatLonCGridOceanModel.step = step_capture
    try:
        base = round66.measure(args)
    finally:
        round66.reciprocal_substitutions = real_reciprocal
        model_module._nemo_ws_rk3_tracer_pair_step = real_pair
        model_module._nemo_qsr_stage3_rate = real_qsr
        model_module.gm_redi_tracer_tendency_latlon = real_ldf
        model_module.LatLonCGridOceanModel.step = real_step

    require(len(reciprocal_calls) == 2,
            f"expected T/S reciprocal calls, got {len(reciprocal_calls)}")
    require(pair_sources and qsr_calls and ldf_calls and len(step_calls) >= 2,
            "one or more production captures did not fire")
    # round66's final production call is the kt=2 oracle-seeded call. Multiple
    # callback observations can occur under JAX, but the last observation must
    # carry that call's exact Kbb tracer.
    (source_t, source_s, pair_t, pair_s, content_t, content_s,
     advection_content_t, advection_content_s) = pair_sources[-1]
    _, seeded, dt, freshwater, surface = step_calls[-1]
    require(np.array_equal(pair_t, np.asarray(seeded.T.data, dtype=np.float64))
            and np.array_equal(pair_s, np.asarray(seeded.S.data, dtype=np.float64)),
            "last source capture is not the oracle-seeded kt=2 call")

    record = round54._read_r63_stream(args.krhs_record, kind="krhs")
    arrays = record["arrays"]
    qsr_t = qsr_calls[-1]
    qsr = {"T": qsr_t, "S": np.zeros_like(qsr_t)}
    ldf_t, ldf_s = ldf_calls[-1]
    ldf = {"T": ldf_t, "S": ldf_s}
    sources = {"T": source_t, "S": source_s}
    captured_content = {"T": content_t, "S": content_s}
    captured_advection_content = {
        "T": advection_content_t,
        "S": advection_content_s,
    }

    results = {}
    routed_content = {}
    implementation_content = {}
    implementation_oracle_content = {}
    production_prediction_content = {}
    production_prediction_rows = {}
    production_baseline_rebuild = {}
    production_vs_round67_routed = {}
    for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
        boundaries = {
            boundary: round66.transposed(arrays[f"{boundary}_{name}"])
            for boundary in ("after_adv", "after_sbc", "after_qsr", "after_ldf")
        }
        live = capture["live"]
        source = sources[name]
        routed_krhs = None
        if args.expect_model_order == "after":
            # The production content already contains LDF.  Remove it only
            # from the diagnostic decomposition; score the captured Krhs
            # directly as the routed arm so subtraction/addition rounding
            # cannot masquerade as a production discrepancy.
            routed_krhs = live["Krhs"]
            live = dict(live)
            live["Krhs"] = live["Krhs"] - ldf[name]
            source = source - ldf[name]
        result = cumulative_substitutions(
            live=live, oracle=capture["oracle"],
            oracle_content=capture["oracle_content"], boundaries=boundaries,
            source=source, qsr=qsr[name], ldf=ldf[name],
            wet=capture["wet"], routed_krhs=routed_krhs)
        routed_content[name] = result.pop("arm_content")["routed_live"]
        implementation_content[name] = round54.field_stats(
            captured_content[name], routed_content[name], capture["wet"])
        implementation_oracle_content[name] = round54.field_stats(
            captured_content[name], capture["oracle_content"], capture["wet"])
        current_ldf = (
            np.zeros_like(ldf[name])
            if args.expect_production_route == "after" else ldf[name]
        )
        production_baseline = production_fct_content(
            captured_advection_content[name], sources[name],
            np.zeros_like(ldf[name]), capture["live"])
        production_prediction = production_fct_content(
            captured_advection_content[name], sources[name],
            current_ldf, capture["live"])
        production_prediction_content[name] = production_prediction
        production_baseline_rebuild[name] = round54.field_stats(
            production_baseline, captured_content[name], capture["wet"])
        production_prediction_rows[name] = {
            "content_vs_oracle": round54.field_stats(
                production_prediction, capture["oracle_content"], capture["wet"]),
            "implementation_vs_prediction": round54.field_stats(
                captured_content[name], production_prediction, capture["wet"]),
        }
        production_vs_round67_routed[name] = round54.field_stats(
            production_prediction, routed_content[name], capture["wet"])
        results[name] = result

    content_criterion = {name: True for name in TRACERS}
    if args.expect_model_order == "after":
        prediction = json.loads(args.prediction_report.read_text())
        for name in TRACERS:
            predicted = prediction["cumulative_substitutions"][name][
                "arm_rows"]["routed_live"]["content"]
            actual = implementation_oracle_content[name]
            association = implementation_content[name]
            content_criterion[name] = bool(
                actual["max_abs"]
                <= predicted["max_abs"] + association["max_abs"])

    post_ldf = results["T"]["arm_rows"]["nemo_post_ldf"]["content"]
    oracle_exact = base["substitution_rows"]["T"]["oracle_baseline"]
    require(oracle_exact["cells_unequal"] == 0,
            "all-oracle five-operand statement did not rebuild exact content")
    expected_post_ldf = (
        base["substitution_rows"]["T"]["oracle_into_live"]["Krhs"])
    require(post_ldf == expected_post_ldf,
            "post-LDF single substitution did not reproduce round-66 Krhs row")
    plant = _one_ulp_plant(
        round66.transposed(arrays["content_T"]), reciprocal_calls[0]["wet"])

    model, seeded, dt, freshwater, surface = step_calls[-1]
    baseline_state = jax.device_get(type(model)(
        model.grid, model.z_coord, model.config).step(
            seeded, dt, freshwater=freshwater, surface_forcing=surface))

    def run_native_source_arm(route: bool):
        pair_start = len(pair_sources)
        ldf_start = len(ldf_calls)
        model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
        model_module.gm_redi_tracer_tendency_latlon = ldf_capture
        try:
            arm_state = jax.device_get(type(model)(
                model.grid, model.z_coord, model.config,
                _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                    route_gm_redi_stage3_source=route),
            ).step(seeded, dt, freshwater=freshwater,
                   surface_forcing=surface))
        finally:
            model_module._nemo_ws_rk3_tracer_pair_step = real_pair
            model_module.gm_redi_tracer_tendency_latlon = real_ldf
        pair_run = pair_sources[pair_start:]
        ldf_run = ldf_calls[ldf_start:]
        require(pair_run and ldf_run,
                "native source arm did not execute the production captures")
        for duplicate in pair_run[1:]:
            require(all(np.array_equal(value, reference)
                        for value, reference in zip(
                            duplicate, pair_run[0], strict=True)),
                    "native source arm observed distinct content evaluations")
        for duplicate in ldf_run[1:]:
            require(all(np.array_equal(value, reference)
                        for value, reference in zip(
                            duplicate, ldf_run[0], strict=True)),
                    "native source arm observed distinct LDF evaluations")
        return arm_state, pair_run[0], ldf_run[0]

    false_state, false_pair, false_ldf = run_native_source_arm(False)
    route_enabled = not args.plant_native_null
    native_state, native_pair, native_ldf = run_native_source_arm(route_enabled)
    require(all(np.array_equal(value, reference)
                for value, reference in zip(
                    false_ldf, native_ldf, strict=True)),
            "native arm changed the live GM/Redi result")

    false_pair_for_control = list(false_pair)
    if args.plant_native_content_ulp:
        planted_content = np.array(false_pair_for_control[4], copy=True)
        first_wet = tuple(np.argwhere(reciprocal_calls[0]["wet"])[0])
        planted_content[first_wet] = np.nextafter(
            planted_content[first_wet], np.float64(np.inf))
        false_pair_for_control[4] = planted_content

    false_state_exact = pytree_exact_census(false_state, baseline_state)
    native_state_move = pytree_exact_census(native_state, false_state)
    false_content_exact = {
        "T": round54.field_stats(
            false_pair_for_control[4], captured_content["T"],
            reciprocal_calls[0]["wet"]),
        "S": round54.field_stats(
            false_pair_for_control[5], captured_content["S"],
            reciprocal_calls[1]["wet"]),
    }

    override_state = type(model)(
        model.grid, model.z_coord, model.config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            pre_implicit_tracer_content_override=(
                jnp.asarray(routed_content["T"]),
                jnp.asarray(routed_content["S"]),
            )),
    ).step(seeded, dt, freshwater=freshwater, surface_forcing=surface)
    baseline_fields = gate.lego_fields(baseline_state)
    override_fields = gate.lego_fields(jax.device_get(override_state))
    production_override_state = type(model)(
        model.grid, model.z_coord, model.config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            pre_implicit_tracer_content_override=(
                jnp.asarray(production_prediction_content["T"]),
                jnp.asarray(production_prediction_content["S"]),
            )),
    ).step(seeded, dt, freshwater=freshwater, surface_forcing=surface)
    production_override_fields = gate.lego_fields(
        jax.device_get(production_override_state))
    native_fields = gate.lego_fields(native_state)
    entry3 = gate.read_entry(args.entry_root / "oracle_step_entry_kt00000003.bin")
    kt3 = {}
    production_kt3 = {}
    native_source_rows = {}
    for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
        tracer_index = TRACERS.index(name)
        native_source = native_pair[tracer_index]
        false_source = false_pair[tracer_index]
        native_content = native_pair[4 + tracer_index]
        expected_source = (
            false_source
            + native_ldf[tracer_index] * capture["wet"].astype(np.float64))
        oracle_entry = entry3[name][..., :baseline_fields[name].shape[-1]]
        kt3[name] = {
            "baseline": round54.field_stats(
                baseline_fields[name], oracle_entry, capture["wet"]),
            "routed_content_override": round54.field_stats(
                override_fields[name], oracle_entry, capture["wet"]),
            "override_vs_baseline": round54.field_stats(
                override_fields[name], baseline_fields[name], capture["wet"]),
        }
        production_kt3[name] = {
            "baseline": round54.field_stats(
                baseline_fields[name], oracle_entry, capture["wet"]),
            "production_content_override": round54.field_stats(
                production_override_fields[name], oracle_entry, capture["wet"]),
            "override_vs_baseline": round54.field_stats(
                production_override_fields[name], baseline_fields[name],
                capture["wet"]),
        }
        native_source_rows[name] = {
            "source_injection": round54.field_stats(
                native_source, expected_source, capture["wet"]),
            "content_vs_oracle": round54.field_stats(
                native_content, capture["oracle_content"], capture["wet"]),
            "content_vs_false": round54.field_stats(
                native_content, false_pair[4 + tracer_index], capture["wet"]),
            "content_vs_host_reconstruction": round54.field_stats(
                native_content, production_prediction_content[name],
                capture["wet"]),
            "kt3_vs_oracle": round54.field_stats(
                native_fields[name], oracle_entry, capture["wet"]),
            "kt3_vs_false": round54.field_stats(
                native_fields[name], baseline_fields[name], capture["wet"]),
            "finite": bool(
                np.all(np.isfinite(native_source))
                and np.all(np.isfinite(native_content))
                and np.all(np.isfinite(np.asarray(native_fields[name])))),
        }

    prediction_match = {name: True for name in TRACERS}
    override_match = {name: True for name in TRACERS}
    if args.expect_model_order == "after":
        for name in TRACERS:
            prediction_match[name] = bool(
                kt3[name]["baseline"]
                == prediction["kt3_prediction"][name]["routed_content_override"])
            override_match[name] = bool(
                kt3[name]["override_vs_baseline"]["cells_unequal"] == 0)

    status = (
        "CONFIRMED"
        if (all(content_criterion.values())
            and all(prediction_match.values())
            and all(override_match.values()))
        else "REFUTED")

    production_criteria = {
        "baseline_rebuild_exact": {
            name: production_baseline_rebuild[name]["cells_unequal"] == 0
            for name in TRACERS
        },
        "implementation_exact": {
            name: production_prediction_rows[name][
                "implementation_vs_prediction"]["cells_unequal"] == 0
            for name in TRACERS
        },
        "frozen_prediction_exact": {name: True for name in TRACERS},
        "frozen_kt3_exact": {name: True for name in TRACERS},
        "override_equals_implementation": {name: True for name in TRACERS},
    }
    if args.expect_production_route == "after":
        production_report = json.loads(
            args.production_prediction_report.read_text())
        frozen_rows = production_report["production_fct_prediction"]
        for name in TRACERS:
            production_criteria["frozen_prediction_exact"][name] = bool(
                production_prediction_rows[name]["content_vs_oracle"]
                == frozen_rows["content_rows"][name]["content_vs_oracle"])
            production_criteria["frozen_kt3_exact"][name] = bool(
                production_kt3[name]["baseline"]
                == frozen_rows["kt3"][name]["production_content_override"])
            production_criteria["override_equals_implementation"][name] = bool(
                production_kt3[name]["override_vs_baseline"][
                    "cells_unequal"] == 0)
        status = (
            "CONFIRMED"
            if all(all(rows.values()) for rows in production_criteria.values())
            else "REFUTED")
    else:
        round67_report = json.loads(args.prediction_report.read_text())
        prediction_t = production_prediction_rows["T"]["content_vs_oracle"]
        baseline_t = base["substitution_rows"]["T"]["live_baseline"]
        improves_20x = bool(
            prediction_t["max_abs"] * 20.0 <= baseline_t["max_abs"])
        within_round67_floor = {}
        for name in TRACERS:
            frozen = round67_report["cumulative_substitutions"][name][
                "arm_rows"]["routed_live"]["content"]["max_abs"]
            association = production_vs_round67_routed[name]["max_abs"]
            within_round67_floor[name] = bool(
                production_prediction_rows[name]["content_vs_oracle"][
                    "max_abs"] <= frozen + association)
        production_criteria["improves_t_20x"] = {"T": improves_20x}
        production_criteria["within_round67_floor"] = within_round67_floor
        status = (
            "CONFIRMED"
            if (all(production_criteria["baseline_rebuild_exact"].values())
                and improves_20x
                and all(within_round67_floor.values()))
            else "REFUTED")

    native_criteria = {}
    if args.round69_native:
        round68_report = json.loads(args.production_prediction_report.read_text())
        frozen_host = round68_report["production_fct_prediction"]
        host_retraction_preserved = {
            name: bool(
                production_baseline_rebuild[name]
                == frozen_host["baseline_rebuild"][name])
            for name in TRACERS
        }
        native_improves_t_20x = bool(
            native_source_rows["T"]["content_vs_oracle"]["max_abs"] * 20.0
            <= base["substitution_rows"]["T"]["live_baseline"]["max_abs"])
        native_floors = {
            "T": np.float64(5.954039670541533e-5),
            "S": np.float64(7.651457963220310e-6),
        }
        within_native_floor = {
            name: bool(
                native_source_rows[name]["content_vs_oracle"]["max_abs"]
                <= native_floors[name]
                + native_source_rows[name][
                    "content_vs_host_reconstruction"]["max_abs"])
            for name in TRACERS
        }
        native_criteria = {
            "default_state_exact": false_state_exact["cells_unequal"] == 0,
            "default_content_exact": {
                name: false_content_exact[name]["cells_unequal"] == 0
                for name in TRACERS
            },
            "same_step_source_exact": {
                name: native_source_rows[name]["source_injection"][
                    "cells_unequal"] == 0
                for name in TRACERS
            },
            "routed_content_moves": {
                name: native_source_rows[name]["content_vs_false"][
                    "cells_unequal"] > 0
                for name in TRACERS
            },
            "routed_kt3_moves": {
                name: native_source_rows[name]["kt3_vs_false"][
                    "cells_unequal"] > 0
                for name in TRACERS
            },
            "finite": {
                name: native_source_rows[name]["finite"]
                for name in TRACERS
            },
            "improves_t_20x": native_improves_t_20x,
            "within_native_floor": within_native_floor,
            "host_retraction_preserved": host_retraction_preserved,
            "host_retraction_census": bool(
                production_baseline_rebuild["T"]["cells_unequal"] == 3
                and production_baseline_rebuild["S"]["cells_unequal"] == 0),
        }

        def all_true(value) -> bool:
            if isinstance(value, dict):
                return all(all_true(item) for item in value.values())
            return bool(value)

        status = (
            "CONFIRMED"
            if all(all_true(value) for value in native_criteria.values())
            else "REFUTED")

    return {
        "format": (
            "nemo-testcase-l2-gyre-round69-native-source-v1"
            if args.round69_native
            else "nemo-testcase-l2-gyre-round68-production-fct-v1"),
        "status": status,
        "worktree": base["worktree"],
        "record_producer": base["record_producer"],
        "admission_counts": base["admission_counts"],
        "dtype": base["dtype"],
        "cumulative_substitutions": results,
        "implementation_content_vs_routed": implementation_content,
        "implementation_content_vs_oracle": implementation_oracle_content,
        "kt3_prediction": kt3,
        "production_fct_prediction": {
            "content_rows": production_prediction_rows,
            "baseline_rebuild": production_baseline_rebuild,
            "vs_round67_recomputed_routed": production_vs_round67_routed,
            "kt3": production_kt3,
            "criteria": production_criteria,
        },
        "native_source_arm": {
            "enabled": bool(args.round69_native),
            "route_enabled": bool(route_enabled),
            "default_state_exact": false_state_exact,
            "default_content_exact": false_content_exact,
            "state_move": native_state_move,
            "rows": native_source_rows,
            "criteria": native_criteria,
        },
        "criteria": {
            "content_within_floor_plus_association": content_criterion,
            "kt3_exact_pre_edit_prediction": prediction_match,
            "content_override_equals_implementation": override_match,
        },
        "controls": {"one_ulp_content": plant,
                     "all_oracle_exact": oracle_exact,
                     "post_ldf_round66_reproduction": post_ldf},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-record-commit", required=True)
    parser.add_argument(
        "--expect-model-order", choices=("before", "after"), required=True)
    parser.add_argument(
        "--expect-production-route", choices=("before", "after"),
        default="before")
    parser.add_argument("--round69-native", action="store_true")
    parser.add_argument("--plant-native-null", action="store_true")
    parser.add_argument("--plant-native-content-ulp", action="store_true")
    parser.add_argument(
        "--prediction-report", type=Path,
        default=ROOT / "round67/round67_ldf_order_before.json")
    parser.add_argument(
        "--production-prediction-report", type=Path,
        default=ROOT / "round68/round68_production_fct_before.json")
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
    parser.add_argument("--admission", type=Path, default=(
        ROOT / "round64/oracle_krhs_split/round64_admission.json"))
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except Exception as error:
        print(f"GATE FAILED: {error}", file=sys.stderr)
        return 1
    if args.round69_native:
        row = report["native_source_arm"]["rows"]["T"]
        print(
            f"ROUND69 NATIVE SOURCE {report['status']}: "
            f"content_T={row['content_vs_oracle']['max_abs']:.12e} "
            f"kt3_T={row['kt3_vs_oracle']['max_abs']:.12e}")
    else:
        row = report["production_fct_prediction"]["content_rows"]["T"]
        kt3 = report["production_fct_prediction"]["kt3"]["T"][
            "production_content_override"]
        print(
            f"ROUND68 PRODUCTION FCT {report['status']}: "
            f"content_T={row['content_vs_oracle']['max_abs']:.12e} "
            f"kt3_T={kt3['max_abs']:.12e}")
    return 0 if report["status"] == "CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
