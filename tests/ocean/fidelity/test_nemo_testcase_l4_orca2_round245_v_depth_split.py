from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round245_v_depth_split as gate,
)


def _difference(unequal: int = 0):
    return {
        "support": 1, "unequal": unequal, "signed_zero_only": 0,
        "max_abs": float(unequal), "argmax": [0] if unequal else [0],
        "first_nonfinite": None,
    }


def _report(label: str):
    substitutions = {name: _difference(0) for name in gate.OPERAND_ORDER}
    substitutions["hv_0"] = _difference(35)
    return {
        "status": "PASS_R245_V_DEPTH_OPERAND_SPLIT",
        "label": label,
        "worktree": {"commit": gate.PAIR_MEASUREMENT_COMMITS["baseline"]},
        "source": {"lines": "487-519"},
        "production_operand_route": "D114 raw hv_0; current compact area/mask",
        "raw_surface_masks": False,
        "raw_v_surface_mask": False,
        "t_pivot_north_neighbor": False,
        "passivity": {"eta": True, "u": True},
        "substeps": list(gate.SUBSTEPS),
        "operand_order": list(gate.OPERAND_ORDER),
        "operand_rows": {
            "hv_0": _difference(0), "r1_e1e2v": _difference(0),
            "e1e2t": _difference(1), "ssvmask": _difference(0),
        },
        "rows": [
            {
                "substep": step,
                "baseline_replay_vs_production": _difference(0),
                "baseline_vs_nemo": _difference(35),
                "eta_mid_owned": _difference(0),
                "pair": {
                    "v_mid": _difference(35),
                    "face_depth_v_mid": _difference(35),
                    "metric_transport_v": _difference(35),
                },
                "cumulative_substitution": copy.deepcopy(substitutions),
            }
            for step in gate.SUBSTEPS
        ],
        "one_bit_control": _difference(1),
    }


def test_classification_and_every_plant_fail_closed():
    reports = [_report("independent"), _report("given_nemo_entry")]
    result = gate.classify(reports)
    assert result["status"] == "HELD_R245_V_DEPTH_OPERAND_NAMED"
    assert result["first_effective_operand"] == "e1e2t"
    for plant in gate.CLASSIFY_PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(reports, plant=plant)


def test_atomic_pair_classification_and_every_pair_plant_fail_closed():
    baseline = [_report("independent"), _report("given_nemo_entry")]
    raw_mask = copy.deepcopy(baseline)
    pair = copy.deepcopy(baseline)
    for report in raw_mask:
        report["worktree"]["commit"] = gate.PAIR_MEASUREMENT_COMMITS["raw-mask arm"]
        report["raw_v_surface_mask"] = True
        report["operand_rows"]["ssvmask"] = _difference(0)
        for row in report["rows"]:
            row["pair"]["face_depth_v_mid"] = _difference(20)
    for report in pair:
        report["worktree"]["commit"] = gate.PAIR_MEASUREMENT_COMMITS["atomic pair"]
        report["raw_v_surface_mask"] = True
        report["t_pivot_north_neighbor"] = True
        for row in report["rows"]:
            for field in row["pair"].values():
                field.update(_difference(0))
    result = gate.classify_pair(baseline, raw_mask, pair)
    assert result["status"] == "HELD_R245_MASK_T_HALO_CANCELLING_PAIR_NAMED"
    assert result["unit"] == ["raw_v_surface_ssvmask", "t_pivot_north_neighbor"]
    for plant in gate.PAIR_PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify_pair(baseline, raw_mask, pair, plant=plant)


def test_v_depth_replay_is_sensitive_to_each_live_operand():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)

    class Fold:
        is_active = False

    class Grid:
        fold = Fold()

    eta = jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float64)
    hv = jnp.ones((3, 2), dtype=jnp.float64)
    r1 = jnp.ones((3, 2), dtype=jnp.float64)
    area = jnp.ones((2, 2), dtype=jnp.float64)
    mask = jnp.ones((3, 2), dtype=jnp.float64)
    baseline = np.asarray(gate._v_depth_replay(eta, hv, r1, area, mask, Grid()))
    for name, changed in (
        ("hv", hv.at[1, 0].set(2.0)),
        ("r1", r1.at[1, 0].set(2.0)),
        ("area", area.at[0, 0].set(2.0)),
        ("mask", mask.at[1, 0].set(0.0)),
    ):
        args = {"eta": eta, "hv_0": hv, "r1_e1e2v": r1,
                "area_t": area, "ssvmask": mask, "grid": Grid()}
        args[{"hv": "hv_0", "r1": "r1_e1e2v", "area": "area_t",
              "mask": "ssvmask"}[name]] = changed
        assert not np.array_equal(np.asarray(gate._v_depth_replay(**args)), baseline)


def test_raw_mask_mapping_is_native_east_north_to_compact_west_south():
    class Raw:
        umask = np.asarray([[[1.0], [0.0]], [[0.0], [1.0]]])
        vmask = np.asarray([[[0.0], [1.0]], [[1.0], [0.0]]])

    class Z:
        nemo_een_barotropic = Raw()

    class Recipe:
        z_coord = Z()

    class Card:
        recipe = Recipe()

    u, v = gate._raw_compact_face_masks(Card())
    np.testing.assert_array_equal(u, [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(v, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
