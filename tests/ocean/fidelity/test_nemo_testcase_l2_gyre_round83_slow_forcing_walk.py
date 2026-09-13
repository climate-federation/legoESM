from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


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


def test_comparison_detects_one_ulp_on_an_active_cell() -> None:
    oracle = np.array([[1.0, 2.0]])
    candidate = oracle.copy()
    candidate[0, 1] = np.nextafter(candidate[0, 1], np.inf)
    row = WALK.comparison(candidate, oracle, np.array([[True, True]]))
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1
    assert row["absolute_max"] > 0.0
