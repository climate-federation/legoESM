from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

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


def test_landed_native_source_route_has_no_private_selector():
    module = _module()
    hooks = module.model_module._NEMOWSRK3TestHooks
    assert "route_gm_redi_stage3_source" not in hooks._fields
    assert "stage3_advection_content_override" in hooks._fields
    source = inspect.getsource(
        module.model_module.LatLonCGridOceanModel._step_impl)
    assert "if _cfg_b.gm_redi is not None" in source
    assert source.count("_stage_source_rates[2][0] + dT_gm * active_3d") == 1
    assert source.count("_stage_source_rates[2][1] + dS_gm * active_3d") == 1


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
