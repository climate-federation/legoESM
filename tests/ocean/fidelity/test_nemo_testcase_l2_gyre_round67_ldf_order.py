from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
    / "nemo_testcase_l2_gyre_round67_ldf_order.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("round67_ldf_order", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cumulative_substitutions_follow_compiled_order():
    module = _module()
    shape = (2, 2, 2)
    wet = np.ones(shape, dtype=bool)
    base = {
        "tracer_Kbb": np.full(shape, 2.0),
        "e3t_Kbb": np.full(shape, 3.0),
        "p2dt": np.float64(4.0),
        "e3t_Kmm": np.full(shape, 5.0),
    }
    oracle_parts = {
        "advection": np.full(shape, 0.125),
        "sbc": np.full(shape, 0.25),
        "qsr": np.full(shape, 0.5),
        "ldf": np.full(shape, 1.0),
    }
    live_parts = dict(oracle_parts)
    live_parts["advection"] = np.full(shape, 0.625)
    boundaries = {
        "after_adv": oracle_parts["advection"],
        "after_sbc": oracle_parts["advection"] + oracle_parts["sbc"],
        "after_qsr": (oracle_parts["advection"] + oracle_parts["sbc"]
                      + oracle_parts["qsr"]),
        "after_ldf": sum(oracle_parts.values()),
    }
    live = dict(base)
    live["Krhs"] = (live_parts["advection"] + live_parts["sbc"]
                    + live_parts["qsr"])
    oracle = dict(base)
    oracle["Krhs"] = boundaries["after_ldf"]
    oracle_content = module.round66.content_statement(oracle)
    result = module.cumulative_substitutions(
        live=live, oracle=oracle, oracle_content=oracle_content,
        boundaries=boundaries,
        source=live_parts["sbc"] + live_parts["qsr"],
        qsr=live_parts["qsr"], ldf=live_parts["ldf"], wet=wet)

    assert result["component_rows"]["advection"]["cells_unequal"] == wet.size
    for name in ("sbc", "qsr", "ldf"):
        assert result["component_rows"][name]["cells_unequal"] == 0
    assert result["arm_rows"]["routed_live"]["content"]["cells_unequal"] == wet.size
    for name in ("nemo_post_sbc", "nemo_post_qsr", "nemo_post_ldf"):
        assert result["arm_rows"][name]["content"]["cells_unequal"] == 0


def test_content_plant_is_non_vacuous():
    module = _module()
    content = np.arange(8.0, dtype=np.float64).reshape(2, 2, 2) + 1.0
    wet = np.ones_like(content, dtype=bool)
    row = module._one_ulp_plant(content, wet)
    assert row["cells_unequal"] == 1
    assert row["max_abs"] > 0.0


def test_production_fct_content_uses_captured_advection_and_kmm_source():
    module = _module()
    advection = np.array([[[2.0, 3.0]]], dtype=np.float64)
    source = np.array([[[0.25, -0.5]]], dtype=np.float64)
    ldf = np.array([[[-0.25, 0.75]]], dtype=np.float64)
    operands = {
        "p2dt": np.float64(4.0),
        "e3t_Kmm": np.array([[[5.0, 7.0]]], dtype=np.float64),
    }
    got = module.production_fct_content(advection, source, ldf, operands)
    want = advection + 4.0 * operands["e3t_Kmm"] * (source + ldf)
    np.testing.assert_array_equal(got, want)

    planted = np.array(advection, copy=True)
    planted[0, 0, 0] = np.nextafter(planted[0, 0, 0], np.inf)
    changed = module.production_fct_content(planted, source, ldf, operands)
    assert np.count_nonzero(changed != got) == 1


def test_pytree_exact_census_checks_every_leaf_and_fires_on_one_ulp():
    module = _module()
    reference = {
        "a": np.array([1.0, 2.0], dtype=np.float64),
        "b": (np.array([3], dtype=np.int32),),
    }
    exact = module.pytree_exact_census(reference, reference)
    assert exact == {
        "leaves": 2, "cells": 3, "cells_unequal": 0, "max_abs": 0.0}

    planted = {
        "a": np.array(reference["a"], copy=True),
        "b": reference["b"],
    }
    planted["a"][0] = np.nextafter(planted["a"][0], np.inf)
    changed = module.pytree_exact_census(planted, reference)
    assert changed["cells_unequal"] == 1
    assert changed["max_abs"] > 0.0


def test_landed_native_source_route_executes_on_the_real_generic_card():
    module = _module()
    hooks = module.model_module._NEMOWSRK3TestHooks
    assert "route_gm_redi_stage3_source" not in hooks._fields
    assert "stage3_advection_content_override" in hooks._fields

    from legoesm.ocean.fidelity.nemo_recipe import (
        _NEMO_GYRE_DT_S,
        build_nemo_gyre_recipe,
    )

    def observed_stage3_source(gm_value):
        recipe = build_nemo_gyre_recipe()
        observations = []
        real_gm = module.model_module.gm_redi_tracer_tendency_latlon
        real_pair = module.model_module._nemo_ws_rk3_tracer_pair_step

        def sentinel_gm(*args, **kwargs):
            result = real_gm(*args, **kwargs)
            assert len(result) == 2
            return tuple(module.jnp.full_like(value, gm_value)
                         for value in result)

        def capture_pair(*args, **kwargs):
            if kwargs.get("return_final_content", False):
                source_t, source_s = kwargs["stage_source_rates"][2]
                module.jax.debug.callback(
                    lambda t, s: observations.append((
                        np.asarray(t), np.asarray(s))),
                    source_t, source_s, ordered=True)
            return real_pair(*args, **kwargs)

        module.model_module.gm_redi_tracer_tendency_latlon = sentinel_gm
        module.model_module._nemo_ws_rk3_tracer_pair_step = capture_pair
        try:
            model = module.model_module.LatLonCGridOceanModel(
                recipe.grid, recipe.z_coord, recipe.model_config)
            result = model.step(
                recipe.initial_state, dt=_NEMO_GYRE_DT_S)
            module.jax.device_get(result)
            module.jax.effects_barrier()
        finally:
            module.model_module.gm_redi_tracer_tendency_latlon = real_gm
            module.model_module._nemo_ws_rk3_tracer_pair_step = real_pair
        assert observations
        for duplicate in observations[1:]:
            np.testing.assert_array_equal(duplicate[0], observations[0][0])
            np.testing.assert_array_equal(duplicate[1], observations[0][1])
        return observations[0]

    zero = observed_stage3_source(0.0)
    planted = observed_stage3_source(0.125)
    for baseline, candidate in zip(zero, planted, strict=True):
        delta = candidate - baseline
        assert np.all((delta == 0.0) | (delta == 0.125))
        assert np.count_nonzero(delta == 0.125) > 0
        assert np.count_nonzero(delta == 0.0) > 0


def test_worsening_census_uses_two_row_scale_ulps():
    module = _module()
    oracle = np.array([1.0, 2.0], dtype=np.float64)
    baseline = np.array(oracle, copy=True)
    candidate = np.array(oracle, copy=True)
    candidate[0] += 2.0 * np.spacing(np.float64(2.0))
    row, mask = module.worsening_census(
        baseline, candidate, oracle, np.ones(2, dtype=bool))
    assert row["cells_worsened"] == 0
    candidate[0] += np.spacing(np.float64(2.0))
    row, mask = module.worsening_census(
        baseline, candidate, oracle, np.ones(2, dtype=bool))
    assert row["cells_worsened"] == 1
    assert mask.tolist() == [True, False]


def test_helper_applies_fct_override_before_source(monkeypatch):
    module = _module()

    def zero_flux_pair(a, b, *args, **kwargs):
        zeros = np.zeros_like(np.asarray(a))
        return ((module.jnp.asarray(zeros), module.jnp.asarray(zeros)),
                (module.jnp.asarray(zeros), module.jnp.asarray(zeros)))

    monkeypatch.setattr(
        module.model_module, "compute_advection_flux_div_pair", zero_flux_pair)
    tracer = module.jnp.ones((1, 1, 1), dtype=module.jnp.float64)
    ones = module.jnp.ones_like(tracer)
    override_a = module.jnp.full_like(tracer, 7.0)
    override_b = module.jnp.full_like(tracer, 11.0)
    source_a = module.jnp.full_like(tracer, 0.5)
    source_b = module.jnp.full_like(tracer, -0.25)
    result = module.model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", ones, ones,
        module.jnp.ones((1, 1, 2)), ones, ones, ones, ones, object(), 2.0,
        ones, stage_source_rates=((source_a, source_b),) * 3,
        stage3_advection_content_override=(override_a, override_b),
        return_final_content=True)
    np.testing.assert_array_equal(np.asarray(result[4]), np.asarray(override_a))
    np.testing.assert_array_equal(np.asarray(result[5]), np.asarray(override_b))
    np.testing.assert_array_equal(np.asarray(result[2]),
                                  np.asarray(override_a + 2.0 * source_a))
    np.testing.assert_array_equal(np.asarray(result[3]),
                                  np.asarray(override_b + 2.0 * source_b))


def test_round71_kmm_resume_replaces_only_selected_tracer():
    module = _module()
    live_t = np.array([1.0, 2.0], dtype=np.float64)
    live_s = np.array([3.0, 4.0], dtype=np.float64)
    target_t = np.array([5.0, 6.0], dtype=np.float64)
    target_s = np.array([7.0, 8.0], dtype=np.float64)
    resume = (2, live_t, live_s)

    assert module._replace_kmm_resume(resume, None) is resume
    t_only = module._replace_kmm_resume(resume, (target_t, None))
    s_only = module._replace_kmm_resume(resume, (None, target_s))
    both = module._replace_kmm_resume(resume, (target_t, target_s))

    assert t_only[1] is target_t and t_only[2] is live_s
    assert s_only[1] is live_t and s_only[2] is target_s
    assert both[1] is target_t and both[2] is target_s


def test_round71_kmm_resume_rejects_a_non_stage2_boundary():
    module = _module()
    with np.testing.assert_raises_regex(
        RuntimeError, "stage-2 Kmm tracer"
    ):
        module._replace_kmm_resume((1, np.ones(1), np.ones(1)), None)


def _round112_synthetic_values(module):
    cell = (22, 32, 30)
    return {
        "p2dt": module.jnp.asarray(14400.0, dtype=module.jnp.float64),
        "p_u": module.jnp.ones((22, 33, 30), dtype=module.jnp.float64),
        "p_v": module.jnp.zeros((23, 32, 30), dtype=module.jnp.float64),
        "p_w": module.jnp.zeros((22, 32, 31), dtype=module.jnp.float64),
        "e3t": module.jnp.ones(cell, dtype=module.jnp.float64),
        "r3t_kbb": module.jnp.zeros((22, 32), dtype=module.jnp.float64),
        "r3t_kmm": module.jnp.zeros((22, 32), dtype=module.jnp.float64),
        "tmask": module.jnp.ones(cell, dtype=module.jnp.float64),
        "wmask": module.jnp.ones(cell, dtype=module.jnp.float64),
        "r1_area": module.jnp.ones((22, 32), dtype=module.jnp.float64),
        "h_kbb": module.jnp.ones(cell, dtype=module.jnp.float64),
        "h_kmm": module.jnp.ones(cell, dtype=module.jnp.float64),
    }


def test_round112_literal_plant_reaches_first_face_under_jit():
    module = _module()
    values = _round112_synthetic_values(module)
    values["tmask"] = values["tmask"].at[0, 0, 0].set(0.0)
    base = module.jnp.ones((22, 32, 30), dtype=module.jnp.float64)

    run = module.jax.jit(
        lambda transport: module._round112_literal_rows(
            base, {**values, "p_u": transport}))
    ordinary = run(values["p_u"])
    planted_transport = values["p_u"].at[0, 1, 0].set(
        module.jnp.nextafter(
            values["p_u"][0, 1, 0], module.jnp.asarray(module.jnp.inf)))
    planted = run(planted_transport)

    assert len(ordinary) == len(module.ROUND112_ROWS)
    assert np.count_nonzero(
        np.asarray(planted[0]) != np.asarray(ordinary[0])) == 1
    assert np.asarray(planted[0])[0, 1, 0] != np.asarray(ordinary[0])[0, 1, 0]
    assert np.asarray(ordinary[-1])[0, 0, 0].view(np.uint64) == 0


def test_round112_duplicate_callback_guard_rejects_distinct_execution():
    module = _module()
    ordinary = (np.array([1.0], dtype=np.float64),)
    planted = (np.array([np.nextafter(1.0, np.inf)], dtype=np.float64),)
    np.testing.assert_array_equal(
        module._round112_collapse([ordinary, ordinary], "synthetic")[0],
        ordinary[0])
    with np.testing.assert_raises_regex(
        RuntimeError, "distinct duplicate executions"
    ):
        module._round112_collapse([ordinary, planted], "synthetic")


def test_round113_live_input_materialization_and_transport_substitution():
    module = _module()
    tracer = module.jnp.ones((2, 3, 2), dtype=module.jnp.float64)
    grid = SimpleNamespace(
        dy_u=module.jnp.full((2, 4), 2.0),
        dx_v=module.jnp.full((3, 3), 3.0),
        area_T=module.jnp.full((2, 3), 5.0),
    )
    values = (
        tracer,
        module.jnp.ones((2, 4, 2)),
        module.jnp.ones((3, 3, 2)),
        module.jnp.ones((2, 3, 3)),
        module.jnp.full_like(tracer, 7.0), grid, 11.0,
    )
    kwargs = {
        "tracer_before": module.jnp.full_like(tracer, 13.0),
        "active_mask": module.jnp.ones_like(tracer),
        "base_thickness": module.jnp.full_like(tracer, 17.0),
    }
    materialized = module._round113_input_tuple(
        values, kwargs, tracer="T")
    np.testing.assert_array_equal(materialized[1], 2.0)
    np.testing.assert_array_equal(materialized[2], 3.0)
    np.testing.assert_array_equal(materialized[3], 5.0)
    np.testing.assert_array_equal(materialized[8], 0.2)

    bundle = {
        "p_u": np.full((2, 4, 2), 19.0),
        "p_v": np.full((3, 3, 2), 23.0),
        "p_w": np.full((2, 3, 3), 29.0),
    }
    patched, same_kwargs = module._round113_patch_inputs(
        values, kwargs, bundle, "T", "transport")
    np.testing.assert_array_equal(np.asarray(patched[1]) * 2.0, 19.0)
    np.testing.assert_array_equal(np.asarray(patched[2]) * 3.0, 23.0)
    np.testing.assert_array_equal(np.asarray(patched[3]) * 5.0, 29.0)
    assert same_kwargs["tracer_before"] is kwargs["tracer_before"]

    for family, changed_index in (
        ("transport_u", 1), ("transport_v", 2), ("transport_w", 3),
    ):
        one, _ = module._round113_patch_inputs(
            values, kwargs, bundle, "T", family)
        for index in (1, 2, 3):
            if index == changed_index:
                assert not np.array_equal(np.asarray(one[index]), values[index])
            else:
                np.testing.assert_array_equal(np.asarray(one[index]), values[index])


def test_round113_pair_callback_guard_fires_on_one_ulp():
    module = _module()
    row = tuple(np.array([float(index + 1)]) for index in range(11))
    calls = [row, row, row, row]
    collapsed = module._round113_collapse_calls(calls, "synthetic")
    assert all(np.array_equal(value, expected)
               for pair in collapsed for value, expected
               in zip(pair, row, strict=True))
    planted = list(row)
    planted[1] = np.nextafter(planted[1], np.inf)
    with np.testing.assert_raises_regex(
        RuntimeError, "distinct duplicate executions"
    ):
        module._round113_collapse_calls(
            [row, row, tuple(planted), row], "synthetic")


def test_round114_u_statement_and_score_name_first_direct_factor():
    module = _module()
    shape3 = (2, 3, 2)
    shape2 = shape3[:2]
    e2u = module.jnp.full(shape2, 3.0, dtype=module.jnp.float64)
    e3u = module.jnp.full(shape3, 2.0, dtype=module.jnp.float64)
    u = module.jnp.full(shape3, 0.25, dtype=module.jnp.float64)
    un_adv = module.jnp.full(shape2, 1.5, dtype=module.jnp.float64)
    r1_hu = module.jnp.full(shape2, 0.5, dtype=module.jnp.float64)
    uu_b = module.jnp.full(shape2, 0.125, dtype=module.jnp.float64)
    umask = module.jnp.ones(shape3, dtype=module.jnp.float64)
    zub, corrected, zfu = module.jax.jit(module._round114_u_statement)(
        e2u, e3u, u, un_adv, r1_hu, uu_b, umask)
    np.testing.assert_array_equal(np.asarray(zub), 0.625)
    np.testing.assert_array_equal(np.asarray(corrected), 0.875)
    np.testing.assert_array_equal(np.asarray(zfu), 5.25)

    observation = (
        np.asarray(e2u), np.asarray(e3u), np.ones(shape2), np.asarray(u),
        np.asarray(un_adv), np.asarray(r1_hu), np.asarray(uu_b),
        np.asarray(umask), np.asarray(zub), np.asarray(corrected),
        np.asarray(zfu),
    )
    bundle = {
        "e2u": np.asarray(e2u), "e3u_kmm": np.asarray(e3u),
        "one_plus_r3u": np.ones(shape2), "u_kmm": np.asarray(u),
        "un_adv": np.asarray(un_adv), "uu_b_kmm": np.asarray(uu_b),
        "umask": np.asarray(umask), "zfu": np.asarray(zfu),
    }
    planted = list(observation)
    planted[1] = np.array(planted[1], copy=True)
    planted[1][0, 0, 0] = np.nextafter(planted[1][0, 0, 0], np.inf)
    score = module._round114_score_u(tuple(planted), bundle)
    assert score["rows"]["e2u"]["classification"] == "BIT"
    assert score["rows"]["e3u_kmm"]["cells_unequal"] == 1
    assert score["first_direct_nonbit"] == "e3u_kmm"
    assert score["rows"]["r1_hu_kmm"]["classification"] == (
        "UNMEASURED_WITH_SPEC")


def test_round114_u_duplicate_guard_rejects_changed_factor():
    module = _module()
    ordinary = tuple(
        np.array([float(index + 1)], dtype=np.float64)
        for index in range(len(module.ROUND114_U_INPUTS)))
    np.testing.assert_equal(
        module._round114_collapse_u([ordinary, ordinary], "synthetic"),
        ordinary)
    planted = list(ordinary)
    planted[1] = np.nextafter(planted[1], np.inf)
    with np.testing.assert_raises_regex(
        RuntimeError, "distinct duplicate U executions"
    ):
        module._round114_collapse_u(
            [ordinary, tuple(planted)], "synthetic")


def test_round115_geometry_trace_scores_first_nonbit_ssh_input():
    module = _module()
    shape = (2, 3)
    eta_full = np.array(
        [[0.125, 0.25, 0.375], [0.5, 0.625, 0.75]],
        dtype=np.float64)
    eta_half = eta_full * np.float64(0.5)
    area_t = np.full(shape, 8.0, dtype=np.float64)
    r1_hu0 = np.full(shape, 0.5, dtype=np.float64)
    r1_area_u = np.full(shape, 0.25, dtype=np.float64)
    full_trace = tuple(np.asarray(value) for value in module.jax.jit(
        module._round115_u_trace)(eta_full, area_t, r1_hu0, r1_area_u))
    scalar_trace = module._round115_scalar_u_trace(
        eta_full, area_t, r1_hu0, r1_area_u)
    for compiled, scalar in zip(full_trace, scalar_trace, strict=True):
        np.testing.assert_array_equal(compiled, scalar)

    half_trace = tuple(np.asarray(value) for value in module.jax.jit(
        module._round115_u_trace)(eta_half, area_t, r1_hu0, r1_area_u))
    r3u_before = np.zeros(shape, dtype=np.float64)
    interpolated = np.asarray(module.jax.jit(module._round115_half_ratio)(
        r3u_before, full_trace[5]))
    np.testing.assert_array_equal(interpolated, half_trace[5])
    one_plus_half = module._round115_u_redundant(half_trace[6])

    observed_full = np.array(eta_full, copy=True)
    observed_full[0, 0] = np.nextafter(observed_full[0, 0], np.inf)
    observation = (
        observed_full, eta_half, area_t, r1_hu0, r1_area_u,
        np.asarray(one_plus_half), full_trace[5], half_trace[5],
        *full_trace, full_trace[5], interpolated,
        np.float64(1.0) + interpolated,
    )
    bundle = {
        "wet_t": np.ones(shape, dtype=bool),
        "wet_u_native": np.ones(shape, dtype=bool),
        "wet_u_redundant": np.ones((2, 4), dtype=bool),
        "area_t": area_t,
        "ssh_n1": eta_full,
        "ssh_half": eta_half,
        "r1_hu0": r1_hu0,
        "r1_area_u": r1_area_u,
        "scalar_trace": scalar_trace,
        "r3u_full": full_trace[5],
        "r3u_half": interpolated,
        "one_plus_r3u_half": np.asarray(one_plus_half),
    }
    score = module._round115_score_geometry(observation, bundle)
    assert score["first_nonbit_input"] == "full_step_ssh"
    assert score["inputs"]["full_step_ssh"]["cells_unequal"] == 1
    assert score["direct_outputs"]["oracle_full_r3u"][
        "classification"] == "BIT"
    assert score["direct_outputs"]["oracle_interpolated_half_r3u"][
        "classification"] == "BIT"


def test_round115_geometry_ulp_plant_reaches_product_and_half_ratio_under_jit():
    module = _module()
    eta = module.jnp.array(
        [[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]],
        dtype=module.jnp.float64)
    area = module.jnp.ones_like(eta)
    r1_depth = module.jnp.ones_like(eta)
    r1_area = module.jnp.ones_like(eta)
    before = module.jnp.zeros_like(eta)

    def run(ssh):
        trace = module._round115_u_trace(ssh, area, r1_depth, r1_area)
        half = module._round115_half_ratio(before, trace[5])
        return trace[0], trace[1], trace[5], half

    ordinary = module.jax.jit(run)(eta)
    planted_eta = eta.at[0, 0].set(module.jnp.nextafter(
        eta[0, 0], module.jnp.asarray(module.jnp.inf)))
    planted = module.jax.jit(run)(planted_eta)
    assert (np.any(np.asarray(planted[0]) != np.asarray(ordinary[0]))
            or np.any(np.asarray(planted[1]) != np.asarray(ordinary[1])))
    assert np.any(np.asarray(planted[2]) != np.asarray(ordinary[2]))
    assert np.any(np.asarray(planted[3]) != np.asarray(ordinary[3]))


def test_round115_geometry_duplicate_guard_rejects_changed_execution():
    module = _module()
    ordinary = tuple(
        np.array([float(index + 1)], dtype=np.float64)
        for index in range(len(module.ROUND115_OBSERVATION_FIELDS)))
    np.testing.assert_equal(
        module._round115_collapse([ordinary, ordinary], "synthetic"),
        ordinary)
    planted = list(ordinary)
    planted[0] = np.nextafter(planted[0], np.inf)
    with np.testing.assert_raises_regex(
        RuntimeError, "distinct duplicate executions"
    ):
        module._round115_collapse(
            [ordinary, tuple(planted)], "synthetic")
