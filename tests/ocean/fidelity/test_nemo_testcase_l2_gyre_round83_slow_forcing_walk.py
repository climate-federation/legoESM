from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import jax


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round83_slow_forcing_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round83_slow_forcing_walk", SCRIPT)
assert SPEC and SPEC.loader
WALK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WALK)


def test_bottom_value_selects_deepest_wet_face_level() -> None:
    values = np.array([[[1.0, 2.0, 99.0], [3.0, 88.0, 77.0]]])
    mask = np.array([[[True, True, False], [True, False, False]]])
    np.testing.assert_array_equal(WALK.bottom_value(values, mask), [[2.0, 3.0]])


def test_source_chain_preserves_compiled_statement_association() -> None:
    # The final model level is NEMO's non-contributing jpk slot.
    rhs = np.array([[[1.0, 2.0, 99.0]]])
    e3 = np.array([[[3.0, 4.0, 99.0]]])
    mask3 = np.ones_like(rhs)
    chain = WALK.source_chain(
        rhs=rhs,
        e3=e3,
        mask3=mask3,
        reciprocal_ref=np.array([[0.5]]),
        inverse_depth=np.array([[0.25]]),
        drag_coefficient=np.array([[2.0]]),
        bottom_velocity=np.array([[5.0]]),
        barotropic_velocity=np.array([[1.0]]),
        rho_reciprocal=np.float64(0.1),
        stress=np.array([[8.0]]),
        coriolis=np.array([[0.75]]),
        mask2=np.ones((1, 1)),
    )
    # depth=(3*1 + 4*2)*.5=5.5; drag=(.25*2)*(5-1)=2;
    # wind=(.1*8)*.25=.2; final=5.5+2+.2-.75=6.95.
    np.testing.assert_array_equal(chain["depth_mean"], [[5.5]])
    np.testing.assert_array_equal(chain["post_drag"], [[7.5]])
    np.testing.assert_allclose(chain["post_wind"], [[7.7]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(chain["final"], [[6.95]], rtol=0.0, atol=0.0)


def test_direct_record_replay_retains_noncontributing_bottom_slot() -> None:
    recorded = np.arange(6 * 7 * 31.0).reshape(6, 7, 31)
    assert WALK.owned3(recorded).shape == (2, 3, 30)
    retained = WALK.owned3_with_bottom(recorded)
    assert retained.shape == (2, 3, 31)
    np.testing.assert_array_equal(retained[..., -1], recorded[2:-2, 2:-2, -1])


def test_comparison_detects_one_ulp_on_an_active_cell() -> None:
    oracle = np.array([[1.0, 2.0]])
    candidate = oracle.copy()
    candidate[0, 1] = np.nextafter(candidate[0, 1], np.inf)
    row = WALK.comparison(candidate, oracle, np.array([[True, True]]))
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1
    assert row["absolute_max"] > 0.0


def test_round117_compiled_accumulator_order_keeps_after_adv_identity() -> None:
    terms = [np.asarray([value], dtype=np.float64) for value in range(1, 11)]
    rows = jax.device_get(jax.jit(WALK.round117_source_order_accumulators)(
        *terms))
    assert rows["after_hpg_u"][0] == 1.0
    assert rows["after_ldf_u"][0] == 4.0
    assert rows["after_vor_u"][0] == 9.0
    assert rows["after_keg_u"][0] == 16.0
    assert rows["after_zad_u"][0] == 25.0
    np.testing.assert_array_equal(rows["after_adv_u"], rows["after_zad_u"])


def test_round117_first_nonbit_obeys_boundary_then_face_order() -> None:
    exact = {"bit_exact": True, "absolute_max": 0.0}
    rows = {
        face: {boundary: dict(exact) for boundary in WALK.ROUND117_BOUNDARIES}
        for face in ("u", "v")
    }
    rows["v"]["after_hpg"] = {"bit_exact": False, "absolute_max": 2.0}
    rows["u"]["after_ldf"] = {"bit_exact": False, "absolute_max": 3.0}
    first = WALK.round117_first_nonbit(rows)
    assert first["boundary"] == "after_hpg"
    assert first["face"] == "v"


def test_round117_native_override_retains_excluded_halo() -> None:
    full_u = np.arange(8.0).reshape(2, 4)
    native_u = np.full((2, 3), 42.0)
    replaced_u = WALK._full_from_native(full_u, native_u, "u")
    np.testing.assert_array_equal(replaced_u[:, 0], full_u[:, 0])
    np.testing.assert_array_equal(replaced_u[:, 1:], native_u)

    full_v = np.arange(8.0).reshape(4, 2)
    native_v = np.full((3, 2), -7.0)
    replaced_v = WALK._full_from_native(full_v, native_v, "v")
    np.testing.assert_array_equal(replaced_v[0], full_v[0])
    np.testing.assert_array_equal(replaced_v[1:], native_v)


def test_round117_one_ulp_plant_survives_subtract() -> None:
    incoming = np.asarray([[1.0, 2.0]], dtype=np.float64)
    coriolis = np.asarray([[0.25, 0.5]], dtype=np.float64)
    mask = np.asarray([[True, True]])
    planted, location, _ = WALK._round117_propagating_ulp(
        incoming, coriolis, mask)
    assert np.count_nonzero(planted.view(np.uint64) != incoming.view(np.uint64)) == 1
    assert ((planted - coriolis)[location].view(np.uint64)
            != (incoming - coriolis)[location].view(np.uint64))
