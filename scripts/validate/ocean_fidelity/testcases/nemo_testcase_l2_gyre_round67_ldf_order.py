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
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean import advection as advection_module
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


def worsening_census(
    baseline: np.ndarray,
    candidate: np.ndarray,
    oracle: np.ndarray,
    wet: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Count candidate cells worsening by more than two row-scale fp64 ULPs."""
    scale = np.float64(max(float(np.max(np.abs(oracle[wet]))), 1.0))
    ulp = np.spacing(scale)
    degradation = np.abs(candidate - oracle) - np.abs(baseline - oracle)
    worsened = np.asarray(wet, dtype=bool) & (degradation > 2.0 * ulp)
    return {
        "row_scale_ulp": float(ulp),
        "cells_worsened": int(np.count_nonzero(worsened)),
        "max_worsening_ulps": float(
            max(0.0, float(np.max(degradation[wet]))) / ulp),
    }, worsened


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


def _replace_kmm_resume(resume, override):
    """Replace only selected stage-2 tracers at the final WS helper boundary."""
    require(resume is not None and resume[0] == 2,
            "final WS helper did not receive the stage-2 Kmm tracer")
    if override is None:
        return resume
    target_t, target_s = override
    return (
        2,
        resume[1] if target_t is None else target_t,
        resume[2] if target_s is None else target_s,
    )


def measure(args) -> dict:
    require(sum((args.round69_native, args.round70_pair, args.round71_kmm,
                 args.round111_fct_split)) <= 1,
            "round-69, round-70, round-71, and round-111 modes are exclusive")
    require(not (args.plant_pair_null_fct and args.plant_pair_content_ulp),
            "round-70 plants are mutually exclusive")
    require(args.round70_pair or not (
        args.plant_pair_null_fct or args.plant_pair_content_ulp),
        "round-70 plants require --round70-pair")
    require(not (args.plant_kmm_null and args.plant_kmm_content_ulp),
            "round-71 plants are mutually exclusive")
    require(args.round71_kmm or not (
        args.plant_kmm_null or args.plant_kmm_content_ulp),
        "round-71 plants require --round71-kmm")
    require(args.round111_fct_split or not args.plant_fct_split_ulp,
            "the FCT split plant requires --round111-fct-split")
    reciprocal_calls: list[dict] = []
    pair_sources: list[tuple[np.ndarray, ...]] = []
    pair_kmm: list[tuple[np.ndarray, np.ndarray]] = []
    qsr_calls: list[np.ndarray] = []
    ldf_calls: list[tuple[np.ndarray, np.ndarray]] = []
    step_calls: list[tuple] = []
    fct_split_calls: list[tuple[np.ndarray, ...]] = []

    real_reciprocal = round66.reciprocal_substitutions
    real_pair = model_module._nemo_ws_rk3_tracer_pair_step
    real_qsr = model_module._nemo_qsr_stage3_rate
    real_ldf = model_module.gm_redi_tracer_tendency_latlon
    real_step = model_module.LatLonCGridOceanModel.step
    real_fct = advection_module.fct_tracer_advection
    active_kmm_override: tuple[object | None, object | None] | None = None
    capture_fct_split = False

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
        nonlocal capture_fct_split
        call_kwargs = kwargs
        if kwargs.get("return_final_content", False):
            resume = _replace_kmm_resume(
                kwargs.get("resume"), active_kmm_override)
            resume_t, resume_s = resume[1], resume[2]
            if resume is not kwargs.get("resume"):
                call_kwargs = dict(kwargs)
                call_kwargs["resume"] = resume
            jax.debug.callback(
                lambda t, s: pair_kmm.append((
                    np.asarray(t, dtype=np.float64),
                    np.asarray(s, dtype=np.float64))),
                resume_t, resume_s, ordered=True)
        prior_capture = capture_fct_split
        capture_fct_split = bool(
            args.round111_fct_split
            and kwargs.get("return_final_content", False))
        try:
            result = real_pair(*values, **call_kwargs)
        finally:
            capture_fct_split = prior_capture
        if kwargs.get("return_final_content", False):
            source_t, source_s = kwargs["stage_source_rates"][2]
            jax.debug.callback(
                pair_sink, source_t, source_s, values[0], values[1],
                result[2], result[3], result[4], result[5],
                ordered=True)
        return result

    def fct_split_sink(*values):
        fct_split_calls.append(tuple(
            np.asarray(value, dtype=np.float64) for value in values))

    def fct_capture(*values, **kwargs):
        if not capture_fct_split:
            return real_fct(*values, **kwargs)
        result = real_fct(*values, return_nemo_split=True, **kwargs)
        div_h, div_w, split = result
        low_h, low_w, anti_h, anti_w = split
        h_kmm = values[4]
        h_safe = jnp.maximum(h_kmm, jnp.asarray(1.0e-10, h_kmm.dtype))
        low_div = nemo_source_round(low_h + low_w)
        anti_div = nemo_source_round(anti_h + anti_w)
        upstream_rhs = nemo_source_round(-low_div / h_safe)
        anti_rhs = nemo_source_round(-anti_div / h_safe)
        if args.plant_fct_split_ulp:
            active = kwargs.get("active_mask")
            require(active is not None,
                    "the production FCT split plant requires an active mask")
            at = jnp.argmax(jnp.ravel(active > 0.5))
            flat = jnp.ravel(upstream_rhs)
            upstream_rhs = flat.at[at].set(jnp.nextafter(
                flat[at], jnp.asarray(jnp.inf, flat.dtype))).reshape(
                    upstream_rhs.shape)
        split_rhs = nemo_source_round(upstream_rhs + anti_rhs)
        combined_rhs = nemo_source_round(
            -nemo_source_round(div_h + div_w) / h_safe)
        base = kwargs.get("tracer_before")
        if base is None:
            base = values[0]
        h_kbb = kwargs.get("base_thickness")
        if h_kbb is None:
            h_kbb = h_kmm
        dt = jnp.asarray(values[6], dtype=h_kmm.dtype)
        split_content = nemo_source_round(
            nemo_source_round(h_kbb * base)
            + nemo_source_round(nemo_source_round(dt * h_kmm) * split_rhs))
        jax.debug.callback(
            fct_split_sink, upstream_rhs, anti_rhs, split_rhs, combined_rhs,
            split_content, low_div, anti_div, ordered=True)
        return div_h, div_w

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
    advection_module.fct_tracer_advection = fct_capture
    model_module._nemo_qsr_stage3_rate = qsr_capture
    model_module.gm_redi_tracer_tendency_latlon = ldf_capture
    model_module.LatLonCGridOceanModel.step = step_capture
    try:
        base = round66.measure(args)
    finally:
        round66.reciprocal_substitutions = real_reciprocal
        model_module._nemo_ws_rk3_tracer_pair_step = real_pair
        advection_module.fct_tracer_advection = real_fct
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
    oracle_advection_content = {}
    for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
        boundaries = {
            boundary: round66.transposed(arrays[f"{boundary}_{name}"])
            for boundary in ("after_adv", "after_sbc", "after_qsr", "after_ldf")
        }
        oracle_adv_operands = dict(capture["oracle"])
        oracle_adv_operands["Krhs"] = boundaries["after_adv"]
        oracle_advection_content[name] = round66.content_statement(
            oracle_adv_operands)
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

    round111_matrix = {}
    if args.round111_fct_split:
        require(len(fct_split_calls) >= 2,
                "production FCT split callbacks did not execute for T/S")
        selected_splits = fct_split_calls[-2:]
        split_rows = {}
        split_observations = {}
        for name, capture, values in zip(
            TRACERS, reciprocal_calls, selected_splits, strict=True
        ):
            (upstream_rhs, anti_rhs, split_rhs, combined_rhs,
             split_content, low_div, anti_div) = values
            oracle_upstream = round66.transposed(arrays[f"adv_up1_{name}"])
            oracle_final = round66.transposed(arrays[f"after_adv_{name}"])
            wet = capture["wet"]
            first_wet = tuple(np.argwhere(wet)[0])
            split_rows[name] = {
                "upstream_rhs_vs_adv_up1": round54.field_stats(
                    upstream_rhs, oracle_upstream, wet),
                "split_rhs_vs_after_adv": round54.field_stats(
                    split_rhs, oracle_final, wet),
                "combined_rhs_vs_after_adv": round54.field_stats(
                    combined_rhs, oracle_final, wet),
                "split_vs_combined": round54.field_stats(
                    split_rhs, combined_rhs, wet),
                "split_advection_content_vs_oracle": round54.field_stats(
                    split_content, oracle_advection_content[name], wet),
                "anti_rhs_vs_posthoc_boundary_difference": round54.field_stats(
                    anti_rhs, oracle_final - oracle_upstream, wet),
                "low_content_divergence": round54.field_stats(
                    low_div, np.zeros_like(low_div), wet),
                "anti_content_divergence": round54.field_stats(
                    anti_div, np.zeros_like(anti_div), wet),
            }
            split_observations[name] = {
                "first_wet_index": [int(index) for index in first_wet],
                "upstream_rhs_bits": int(np.asarray(
                    upstream_rhs[first_wet], dtype="=f8").view("=u8")),
                "split_rhs_bits": int(np.asarray(
                    split_rhs[first_wet], dtype="=f8").view("=u8")),
            }
        local_frozen = {
            "T": {"content": np.float64(5.743498263655056e-5),
                  "kt3": np.float64(8.600420500215478e-7)},
            "S": {"content": np.float64(7.387909136014059e-6),
                  "kt3": np.float64(6.979443156751586e-8)},
        }
        split_criteria = {
            "callbacks_observed": len(fct_split_calls) >= 2,
            "upstream_predicted_nonbit": {
                name: split_rows[name]["upstream_rhs_vs_adv_up1"][
                    "cells_unequal"] > 0
                for name in TRACERS
            },
            "final_predicted_nonbit": {
                name: split_rows[name]["combined_rhs_vs_after_adv"][
                    "cells_unequal"] > 0
                for name in TRACERS
            },
            "frozen_content_max": {
                name: bool(
                    implementation_oracle_content[name]["max_abs"]
                    == local_frozen[name]["content"])
                for name in TRACERS
            },
        }
        candidate_eligible = bool(all(
            split_rows[name][boundary]["cells_unequal"] == 0
            for name in TRACERS
            for boundary in (
                "upstream_rhs_vs_adv_up1", "split_rhs_vs_after_adv")
        ))
        plant_fired = False
        if args.plant_fct_split_ulp:
            frozen = json.loads(args.fct_split_report.read_text())[
                "round111_fct_split"]
            plant_fired = bool(any(
                split_observations[name]["upstream_rhs_bits"]
                != frozen["observations"][name]["upstream_rhs_bits"]
                for name in TRACERS
            ))
            require(plant_fired,
                    "one-ULP production FCT split plant was not observed")
        round111_matrix = {
            "status": (
                "PLANT-FIRED" if plant_fired else
                "CONFIRMED" if all(
                    all(value.values()) if isinstance(value, dict) else value
                    for value in split_criteria.values())
                else "REFUTED"),
            "plant_ulp": bool(args.plant_fct_split_ulp),
            "candidate_eligible": candidate_eligible,
            "rows": split_rows,
            "observations": split_observations,
            "criteria": split_criteria,
            "caveat": (
                "The anti-boundary subtraction is post-hoc and is not an "
                "input proof; eligibility uses only directly recorded rows."),
        }

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

    pair_matrix = {}
    kmm_matrix = {}
    if args.round70_pair or args.round71_kmm:
        original_fct_target = {
            name: np.array(value, dtype=np.float64, copy=True)
            for name, value in oracle_advection_content.items()
        }
        fct_target = {
            name: np.array(value, dtype=np.float64, copy=True)
            for name, value in original_fct_target.items()
        }
        if args.plant_pair_content_ulp:
            first_wet = tuple(np.argwhere(reciprocal_calls[0]["wet"])[0])
            fct_target["T"][first_wet] = np.nextafter(
                fct_target["T"][first_wet], np.float64(np.inf))

        def run_pair_arm(
            *, route_ldf: bool, use_fct: bool,
            kmm_override: tuple[object | None, object | None] | None = None,
        ):
            nonlocal active_kmm_override
            pair_start = len(pair_sources)
            kmm_start = len(pair_kmm)
            ldf_start = len(ldf_calls)
            override = None
            if use_fct:
                override = (jnp.asarray(fct_target["T"]),
                            jnp.asarray(fct_target["S"]))
            model_module._nemo_ws_rk3_tracer_pair_step = pair_capture
            model_module.gm_redi_tracer_tendency_latlon = ldf_capture
            active_kmm_override = kmm_override
            try:
                arm_state = jax.device_get(type(model)(
                    model.grid, model.z_coord, model.config,
                    _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                        route_gm_redi_stage3_source=route_ldf,
                        stage3_advection_content_override=override),
                ).step(seeded, dt, freshwater=freshwater,
                       surface_forcing=surface))
            finally:
                active_kmm_override = None
                model_module._nemo_ws_rk3_tracer_pair_step = real_pair
                model_module.gm_redi_tracer_tendency_latlon = real_ldf
            pair_run = pair_sources[pair_start:]
            kmm_run = pair_kmm[kmm_start:]
            ldf_run = ldf_calls[ldf_start:]
            require(pair_run and kmm_run and ldf_run,
                    "FCT arm did not execute production captures")
            require(all(all(np.array_equal(value, reference)
                            for value, reference in zip(
                                duplicate, pair_run[0], strict=True))
                        for duplicate in pair_run[1:]),
                    "round-70 arm observed distinct content evaluations")
            require(all(all(np.array_equal(value, reference)
                            for value, reference in zip(
                                duplicate, ldf_run[0], strict=True))
                        for duplicate in ldf_run[1:]),
                    "round-70 arm observed distinct LDF evaluations")
            require(all(all(np.array_equal(value, reference)
                            for value, reference in zip(
                                duplicate, kmm_run[0], strict=True))
                        for duplicate in kmm_run[1:]),
                    "FCT arm observed distinct Kmm evaluations")
            return arm_state, pair_run[0], ldf_run[0], kmm_run[0]

        untouched = run_pair_arm(route_ldf=False, use_fct=False)
        ldf_only = run_pair_arm(route_ldf=True, use_fct=False)
        if args.plant_pair_null_fct:
            fct_target = {
                "T": np.array(untouched[1][6], copy=True),
                "S": np.array(untouched[1][7], copy=True),
            }
        fct_only = run_pair_arm(route_ldf=False, use_fct=True)
        paired = run_pair_arm(route_ldf=True, use_fct=True)
        arms = {
            "untouched": untouched,
            "ldf_only": ldf_only,
            "fct_only": fct_only,
            "paired": paired,
        }
        entry3_pair = gate.read_entry(
            args.entry_root / "oracle_step_entry_kt00000003.bin")
        arm_rows = {}
        criteria = {
            "ordinary_state_exact": pytree_exact_census(
                untouched[0], baseline_state)["cells_unequal"] == 0,
            "ordinary_content_exact": {},
            "same_step_ldf_exact": {},
            "fct_target_exact": {},
            "active_content_moves": {},
            "active_state_moves": {},
            "finite": {},
            "paired_content_ceiling": {},
            "paired_improves_ldf_eightfold": {},
            "paired_kt3_improves_both": {},
            "ldf_worsening_removed_99pct": {},
        }
        for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
            index = TRACERS.index(name)
            wet = capture["wet"]
            oracle_content = capture["oracle_content"]
            oracle_entry = entry3_pair[name][..., :np.asarray(
                gate.lego_fields(baseline_state)[name]).shape[-1]]
            untouched_content = untouched[1][4 + index]
            untouched_adv = untouched[1][6 + index]
            untouched_field = np.asarray(
                gate.lego_fields(untouched[0])[name], dtype=np.float64)
            arm_rows[name] = {}
            for arm_name, (arm_state, arm_pair, arm_ldf, _) in arms.items():
                field = np.asarray(
                    gate.lego_fields(arm_state)[name], dtype=np.float64)
                content = arm_pair[4 + index]
                arm_rows[name][arm_name] = {
                    "content_vs_oracle": round54.field_stats(
                        content, oracle_content, wet),
                    "content_vs_untouched": round54.field_stats(
                        content, untouched_content, wet),
                    "kt3_vs_oracle": round54.field_stats(
                        field, oracle_entry, wet),
                    "kt3_vs_untouched": round54.field_stats(
                        field, untouched_field, wet),
                    "source_vs_expected": round54.field_stats(
                        arm_pair[index],
                        (untouched[1][index] + untouched[2][index]
                         * wet.astype(np.float64)
                         if arm_name in ("ldf_only", "paired")
                         else untouched[1][index]), wet),
                    "advection_content_vs_target": round54.field_stats(
                        arm_pair[6 + index],
                        (fct_target[name] if arm_name in ("fct_only", "paired")
                         else untouched_adv), wet),
                }
                criteria["finite"][f"{name}_{arm_name}"] = bool(
                    np.all(np.isfinite(field))
                    and np.all(np.isfinite(content))
                    and np.all(np.isfinite(arm_ldf[index])))
            criteria["ordinary_content_exact"][name] = bool(
                round54.field_stats(
                    untouched_content, captured_content[name], wet)[
                        "cells_unequal"] == 0)
            criteria["same_step_ldf_exact"][name] = bool(all(
                np.array_equal(arm[2][index], untouched[2][index])
                for arm in arms.values()))
            frozen_target_row = round54.field_stats(
                fct_only[1][6 + index], original_fct_target[name], wet)
            criteria["fct_target_exact"][name] = bool(
                frozen_target_row["cells_unequal"] == 0)
            for arm_name in ("ldf_only", "fct_only", "paired"):
                criteria["active_content_moves"][f"{name}_{arm_name}"] = bool(
                    arm_rows[name][arm_name]["content_vs_untouched"][
                        "cells_unequal"] > 0)
                criteria["active_state_moves"][f"{name}_{arm_name}"] = bool(
                    arm_rows[name][arm_name]["kt3_vs_untouched"][
                        "cells_unequal"] > 0)
            paired_max = arm_rows[name]["paired"]["content_vs_oracle"][
                "max_abs"]
            ldf_max = arm_rows[name]["ldf_only"]["content_vs_oracle"][
                "max_abs"]
            ceiling = 6.0e-6 if name == "T" else 1.0e-6
            criteria["paired_content_ceiling"][name] = bool(
                paired_max <= ceiling)
            criteria["paired_improves_ldf_eightfold"][name] = bool(
                paired_max * 8.0 <= ldf_max)
            paired_kt3 = arm_rows[name]["paired"]["kt3_vs_oracle"]["max_abs"]
            criteria["paired_kt3_improves_both"][name] = bool(
                paired_kt3
                < arm_rows[name]["untouched"]["kt3_vs_oracle"]["max_abs"]
                and paired_kt3
                < arm_rows[name]["ldf_only"]["kt3_vs_oracle"]["max_abs"])
            ldf_worsening, ldf_mask = worsening_census(
                untouched_field,
                np.asarray(gate.lego_fields(ldf_only[0])[name]),
                oracle_entry, wet)
            paired_worsening, paired_mask = worsening_census(
                untouched_field,
                np.asarray(gate.lego_fields(paired[0])[name]),
                oracle_entry, wet)
            retained = int(np.count_nonzero(ldf_mask & paired_mask))
            denominator = ldf_worsening["cells_worsened"]
            removed_fraction = (
                1.0 if denominator == 0 else 1.0 - retained / denominator)
            criteria["ldf_worsening_removed_99pct"][name] = bool(
                removed_fraction >= 0.99)
            arm_rows[name]["worsening"] = {
                "ldf_only": ldf_worsening,
                "paired": paired_worsening,
                "ldf_cells_still_worsened_by_pair": retained,
                "ldf_worsening_removed_fraction": removed_fraction,
                "frozen_target": frozen_target_row,
            }

        def all_true(value) -> bool:
            if isinstance(value, dict):
                return all(all_true(item) for item in value.values())
            return bool(value)

        pair_matrix = {
            "status": (
                "CONFIRMED" if all_true(criteria) else "REFUTED"),
            "plant_null_fct": bool(args.plant_pair_null_fct),
            "plant_content_ulp": bool(args.plant_pair_content_ulp),
            "rows": arm_rows,
            "criteria": criteria,
        }

        if args.round71_kmm:
            stage3 = round66.round46.read_stage(
                args.stage_root / "oracle_momstage_kt00000002_s3.bin")
            require(stage3["header"] == {
                "version": 1, "kt": 2, "stage": 3,
                "Kbb": 3, "Kmm": 2, "Krhs": 1, "Kaa": 1,
                "jpi": 36, "jpj": 26, "jpk": 31, "jpkm1": 30,
                "ntsi": 3, "ntei": 34, "ntsj": 3, "ntej": 24,
                "bits": 64,
            }, "round-46 kt2 stage-3 header changed")
            original_kmm_target = {
                name: np.asarray(round66.round46._owned3(
                    stage3["arrays"][f"{name}_Kmm"]), dtype=np.float64)
                for name in TRACERS
            }
            kmm_target = {
                name: np.array(value, dtype=np.float64, copy=True)
                for name, value in original_kmm_target.items()
            }
            if args.plant_kmm_null:
                kmm_target = {
                    "T": np.array(ldf_only[3][0], copy=True),
                    "S": np.array(ldf_only[3][1], copy=True),
                }
            if args.plant_kmm_content_ulp:
                first_wet = tuple(np.argwhere(reciprocal_calls[0]["wet"])[0])
                kmm_target["T"][first_wet] = np.nextafter(
                    kmm_target["T"][first_wet], np.float64(np.inf))

            kmm_t = run_pair_arm(
                route_ldf=True, use_fct=False,
                kmm_override=(jnp.asarray(kmm_target["T"]), None))
            kmm_s = run_pair_arm(
                route_ldf=True, use_fct=False,
                kmm_override=(None, jnp.asarray(kmm_target["S"])))
            kmm_ts = run_pair_arm(
                route_ldf=True, use_fct=False,
                kmm_override=(jnp.asarray(kmm_target["T"]),
                              jnp.asarray(kmm_target["S"])))
            kmm_arms = {"kmm_T": kmm_t, "kmm_S": kmm_s, "kmm_TS": kmm_ts}
            kmm_rows = {name: {} for name in TRACERS}
            kmm_criteria = {
                "round70_rows_exact": False,
                "round70_criteria_exact": False,
                "source_operand_non_bit": {},
                "injected_target_exact": {},
                "cross_tracer_isolation": {},
                "joint_equals_single": {},
                "same_step_ldf_exact": {},
                "active_content_moves": {},
                "active_state_moves": {},
                "finite": {},
                "kt3_improves_ldf_twofold": {},
                "kt3_remains_worse_than_output_pair": {},
                "retained_set_reduced_half": {},
                "retained_debt_remains": {},
            }
            frozen_round70 = json.loads(args.round70_report.read_text())[
                "round70_pair_matrix"]
            kmm_criteria["round70_rows_exact"] = bool(
                pair_matrix["rows"] == frozen_round70["rows"])
            kmm_criteria["round70_criteria_exact"] = bool(
                pair_matrix["criteria"] == frozen_round70["criteria"])

            source_split = {}
            for name, capture in zip(TRACERS, reciprocal_calls, strict=True):
                index = TRACERS.index(name)
                wet = capture["wet"]
                oracle_entry = entry3_pair[name][..., :np.asarray(
                    gate.lego_fields(baseline_state)[name]).shape[-1]]
                untouched_field = np.asarray(
                    gate.lego_fields(untouched[0])[name], dtype=np.float64)
                ldf_field = np.asarray(
                    gate.lego_fields(ldf_only[0])[name], dtype=np.float64)
                paired_field = np.asarray(
                    gate.lego_fields(paired[0])[name], dtype=np.float64)
                original_target = original_kmm_target[name]
                live_kmm = ldf_only[3][index]
                source_row = round54.field_stats(
                    live_kmm, original_target, wet)
                expected_count = 17994 if name == "T" else 16769
                kmm_criteria["source_operand_non_bit"][name] = bool(
                    source_row["cells_unequal"] == expected_count
                    and source_row["max_abs"] > 0.0)

                for arm_name, arm in kmm_arms.items():
                    field = np.asarray(
                        gate.lego_fields(arm[0])[name], dtype=np.float64)
                    content = arm[1][4 + index]
                    kmm_rows[name][arm_name] = {
                        "content_vs_oracle": round54.field_stats(
                            content, capture["oracle_content"], wet),
                        "content_vs_ldf_only": round54.field_stats(
                            content, ldf_only[1][4 + index], wet),
                        "kt3_vs_oracle": round54.field_stats(
                            field, oracle_entry, wet),
                        "kt3_vs_ldf_only": round54.field_stats(
                            field, ldf_field, wet),
                        "injected_kmm_vs_original_target": round54.field_stats(
                            arm[3][index], original_target, wet),
                    }
                    kmm_criteria["finite"][f"{name}_{arm_name}"] = bool(
                        np.all(np.isfinite(field))
                        and np.all(np.isfinite(content))
                        and np.all(np.isfinite(arm[3][index])))
                    kmm_criteria["same_step_ldf_exact"][
                        f"{name}_{arm_name}"] = bool(np.array_equal(
                            arm[2][index], ldf_only[2][index]))

                target_arm = kmm_t if name == "T" else kmm_s
                other_arm = kmm_s if name == "T" else kmm_t
                kmm_criteria["injected_target_exact"][name] = bool(
                    kmm_rows[name]["kmm_TS"][
                        "injected_kmm_vs_original_target"][
                            "cells_unequal"] == 0)
                kmm_criteria["cross_tracer_isolation"][name] = bool(
                    round54.field_stats(
                        np.asarray(gate.lego_fields(other_arm[0])[name]),
                        ldf_field, wet)["cells_unequal"] == 0
                    and round54.field_stats(
                        other_arm[1][4 + index],
                        ldf_only[1][4 + index], wet)["cells_unequal"] == 0)
                kmm_criteria["joint_equals_single"][name] = bool(
                    round54.field_stats(
                        np.asarray(gate.lego_fields(kmm_ts[0])[name]),
                        np.asarray(gate.lego_fields(target_arm[0])[name]),
                        wet)["cells_unequal"] == 0
                    and round54.field_stats(
                        kmm_ts[1][4 + index], target_arm[1][4 + index], wet)[
                            "cells_unequal"] == 0)
                kmm_criteria["active_content_moves"][name] = bool(
                    kmm_rows[name]["kmm_TS"]["content_vs_ldf_only"][
                        "cells_unequal"] > 0)
                kmm_criteria["active_state_moves"][name] = bool(
                    kmm_rows[name]["kmm_TS"]["kt3_vs_ldf_only"][
                        "cells_unequal"] > 0)

                ldf_worsening, ldf_mask = worsening_census(
                    untouched_field, ldf_field, oracle_entry, wet)
                paired_worsening, paired_mask = worsening_census(
                    untouched_field, paired_field, oracle_entry, wet)
                kmm_field = np.asarray(
                    gate.lego_fields(kmm_ts[0])[name], dtype=np.float64)
                kmm_worsening, kmm_mask = worsening_census(
                    untouched_field, kmm_field, oracle_entry, wet)
                retained_mask = ldf_mask & paired_mask
                new_pair_mask = (~ldf_mask) & paired_mask & wet
                retained_count = int(np.count_nonzero(retained_mask))
                new_pair_count = int(np.count_nonzero(new_pair_mask))
                expected_retained = 1375 if name == "T" else 1891
                expected_new = 1800 if name == "T" else 6374
                require(retained_count == expected_retained,
                        f"round-70 retained {name} set changed")
                require(new_pair_count == expected_new,
                        f"round-70 newly-worsened {name} set changed")
                retained_kmm = int(np.count_nonzero(retained_mask & kmm_mask))
                new_pair_kmm = int(np.count_nonzero(new_pair_mask & kmm_mask))
                retained_removed_fraction = 1.0 - retained_kmm / retained_count
                source_split[name] = {
                    "live_kmm_vs_oracle": source_row,
                    "ldf_only_worsening": ldf_worsening,
                    "output_pair_worsening": paired_worsening,
                    "kmm_pair_worsening": kmm_worsening,
                    "round70_retained_cells": retained_count,
                    "round70_newly_worsened_cells": new_pair_count,
                    "retained_cells_still_worsened_by_kmm": retained_kmm,
                    "retained_cells_removed_fraction": retained_removed_fraction,
                    "round70_new_cells_worsened_by_kmm": new_pair_kmm,
                }
                kmm_max = kmm_rows[name]["kmm_TS"][
                    "kt3_vs_oracle"]["max_abs"]
                ldf_max = arm_rows[name]["ldf_only"][
                    "kt3_vs_oracle"]["max_abs"]
                output_pair_max = arm_rows[name]["paired"][
                    "kt3_vs_oracle"]["max_abs"]
                kmm_criteria["kt3_improves_ldf_twofold"][name] = bool(
                    2.0 * kmm_max <= ldf_max)
                kmm_criteria["kt3_remains_worse_than_output_pair"][name] = bool(
                    kmm_max > output_pair_max)
                kmm_criteria["retained_set_reduced_half"][name] = bool(
                    retained_removed_fraction >= 0.5)
                kmm_criteria["retained_debt_remains"][name] = bool(
                    retained_kmm > 0)

            kmm_matrix = {
                "status": (
                    "CONFIRMED" if all_true(kmm_criteria) else "REFUTED"),
                "plant_null": bool(args.plant_kmm_null),
                "plant_content_ulp": bool(args.plant_kmm_content_ulp),
                "dtype": {
                    "live_kmm": str(np.asarray(ldf_only[3][0]).dtype),
                    "injected_kmm": str(np.asarray(kmm_ts[3][0]).dtype),
                    "record_kmm": str(original_kmm_target["T"].dtype),
                },
                "rows": kmm_rows,
                "source_split": source_split,
                "criteria": kmm_criteria,
            }

    if args.round69_native:
        require(args.expect_model_order == "after",
                "the landed round-69 gate requires production LDF routing")
        require(not args.plant_native_null and not args.plant_native_content_ulp,
                "round-69 plants belong to the committed pre-edit instrument")
    native_state = baseline_state
    native_pair = (
        source_t, source_s, pair_t, pair_s, content_t, content_s,
        advection_content_t, advection_content_s)
    native_ldf = (ldf_t, ldf_s)
    false_pair = native_pair
    false_pair_for_control = list(false_pair)
    route_enabled = args.expect_model_order == "after"
    false_state_exact = pytree_exact_census(baseline_state, baseline_state)
    native_state_move = pytree_exact_census(native_state, baseline_state)
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
        expected_source = native_source
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

    if args.round111_fct_split:
        round111_matrix["local_rows"] = {
            name: {
                "content_vs_oracle": implementation_oracle_content[name],
                "kt3_vs_oracle": kt3[name]["baseline"],
            }
            for name in TRACERS
        }
        round111_matrix["criteria"]["frozen_kt3_max"] = {
            name: bool(
                kt3[name]["baseline"]["max_abs"]
                == local_frozen[name]["kt3"])
            for name in TRACERS
        }
        if not args.plant_fct_split_ulp:
            round111_matrix["status"] = (
                "CONFIRMED" if all(
                    all(value.values()) if isinstance(value, dict) else value
                    for value in round111_matrix["criteria"].values())
                else "REFUTED")

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
        native_prediction = json.loads(args.native_prediction_report.read_text())
        frozen_native = native_prediction["native_source_arm"]
        native_improves_t_20x = bool(
            native_source_rows["T"]["content_vs_oracle"]["max_abs"] * 20.0
            <= round68_report["implementation_content_vs_oracle"]["T"][
                "max_abs"])
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
            "frozen_content_metrics_exact": {
                name: bool(
                    native_source_rows[name]["content_vs_oracle"]
                    == frozen_native["rows"][name]["content_vs_oracle"])
                for name in TRACERS
            },
            "frozen_kt3_metrics_exact": {
                name: bool(
                    native_source_rows[name]["kt3_vs_oracle"]
                    == frozen_native["rows"][name]["kt3_vs_oracle"])
                for name in TRACERS
            },
            "private_arm_removed": bool(
                "route_gm_redi_stage3_source"
                not in model_module._NEMOWSRK3TestHooks._fields),
            "finite": {
                name: native_source_rows[name]["finite"]
                for name in TRACERS
            },
            "improves_t_20x": native_improves_t_20x,
            "within_native_floor": within_native_floor,
            "round68_host_retraction_retained": bool(
                frozen_host["baseline_rebuild"]["T"]["cells_unequal"] == 3
                and frozen_host["baseline_rebuild"]["S"][
                    "cells_unequal"] == 0),
            "native_association_census_reproduced": {
                name: bool(
                    production_baseline_rebuild[name]["cells_unequal"]
                    == frozen_native["rows"][name][
                        "content_vs_host_reconstruction"]["cells_unequal"]
                    and production_baseline_rebuild[name]["max_abs"]
                    == frozen_native["rows"][name][
                        "content_vs_host_reconstruction"]["max_abs"])
                for name in TRACERS
            },
            "pre_edit_plants": {
                "null_exit": 1,
                "content_ulp_exit": 1,
            },
        }

        def all_true(value) -> bool:
            if isinstance(value, dict):
                return all(all_true(item) for item in value.values())
            return bool(value)

        status = (
            "CONFIRMED"
            if all(all_true(value) for value in native_criteria.values())
            else "REFUTED")
    if args.round70_pair:
        status = pair_matrix["status"]
    if args.round71_kmm:
        status = kmm_matrix["status"]
    if args.round111_fct_split:
        status = round111_matrix["status"]

    return {
        "format": (
            "nemo-testcase-l2-gyre-round111-fct-split-v1"
            if args.round111_fct_split else
            "nemo-testcase-l2-gyre-round71-fct-kmm-v1"
            if args.round71_kmm else
            "nemo-testcase-l2-gyre-round70-fct-ldf-pair-v1"
            if args.round70_pair else
            "nemo-testcase-l2-gyre-round69-native-source-v1"
            if args.round69_native else
            "nemo-testcase-l2-gyre-round68-production-fct-v1"),
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
        "round70_pair_matrix": pair_matrix,
        "round71_kmm_matrix": kmm_matrix,
        "round111_fct_split": round111_matrix,
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
    parser.add_argument("--round70-pair", action="store_true")
    parser.add_argument("--plant-pair-null-fct", action="store_true")
    parser.add_argument("--plant-pair-content-ulp", action="store_true")
    parser.add_argument("--round71-kmm", action="store_true")
    parser.add_argument("--plant-kmm-null", action="store_true")
    parser.add_argument("--plant-kmm-content-ulp", action="store_true")
    parser.add_argument("--round111-fct-split", action="store_true")
    parser.add_argument("--plant-fct-split-ulp", action="store_true")
    parser.add_argument(
        "--fct-split-report", type=Path,
        default=ROOT / "round111/fct_split_jit.json")
    parser.add_argument(
        "--prediction-report", type=Path,
        default=ROOT / "round67/round67_ldf_order_before.json")
    parser.add_argument(
        "--production-prediction-report", type=Path,
        default=ROOT / "round68/round68_production_fct_before.json")
    parser.add_argument(
        "--native-prediction-report", type=Path,
        default=ROOT / "round69/round69_native_source_before.json")
    parser.add_argument(
        "--round70-report", type=Path,
        default=ROOT / "round70/round70_pair_before.json")
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
    if args.round111_fct_split:
        row = report["round111_fct_split"]["rows"]["T"]
        prefix = (
            "STATUS PLANT-FIRED" if report["status"] == "PLANT-FIRED"
            else f"ROUND111 FCT SPLIT {report['status']}")
        print(
            f"{prefix}: "
            f"upstream={row['upstream_rhs_vs_adv_up1']['max_abs']:.12e} "
            f"final={row['split_rhs_vs_after_adv']['max_abs']:.12e}")
    elif args.round71_kmm:
        row = report["round71_kmm_matrix"]["rows"]["T"]
        ldf_max = report["round70_pair_matrix"]["rows"]["T"][
            "ldf_only"]["kt3_vs_oracle"]["max_abs"]
        print(
            f"ROUND71 FCT KMM {report['status']}: "
            f"ldf_T={ldf_max:.12e} "
            f"kmm_T={row['kmm_TS']['kt3_vs_oracle']['max_abs']:.12e}")
    elif args.round70_pair:
        row = report["round70_pair_matrix"]["rows"]["T"]
        print(
            f"ROUND70 FCT LDF PAIR {report['status']}: "
            f"ldf_T={row['ldf_only']['content_vs_oracle']['max_abs']:.12e} "
            f"pair_T={row['paired']['content_vs_oracle']['max_abs']:.12e}")
    elif args.round69_native:
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
