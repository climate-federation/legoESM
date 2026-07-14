"""Iter-925 regression sentinel: pin FV3 production rest state behavior.

Per Ralph loop step 4 ("ocean rest state" / atmospheric rest state):
the FV3EdgeShallowWaterModel with `apply_fortran_xppm_boundary=True`
(iter-893 production matrix) on a uniform-h, zero-velocity initial
state should:

1. Hold winds at machine precision after multi-step (max|u_d|, |v_d|
   < 1e-12 after 1-day C36 = 288 RK3 steps).
2. Hold `h` to within ~1e-4 m of H0=1000 m (this is the float64
   round-off floor of the flux divergence, NOT a physical drift).
3. Be bit-identical to the iter-820 baseline (`apply_fortran_xppm_
   boundary=False`).  Rationale: PPM reconstruction of a constant
   field gives q_R = q_L = H0 regardless of which boundary formula
   is used, so the iter-893 fix is a no-op on rest state.

Property #3 is the strongest gate.  Any iter that makes iter-893
non-trivial on rest state would be hiding a bug — the boundary
formula MUST be exact on constants.

Cost: ~30 s wall (two C36 1-day rest-state trajectories shared
via module fixtures, but each is much faster than W2 because no
PPM reconstruction work happens on the constant field).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


N = 36
DT = 300.0
NSTEPS = int(86400 / DT)
H0 = 1000.0


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


def _run_rest_state(apply_xppm: bool) -> dict:
    grid = create_cubed_sphere(N)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(N),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=apply_xppm,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    state = FV3EdgeShallowWaterState(
        h=jnp.ones((6, N, N)) * H0,
        u_d=jnp.zeros((6, N, N + 1)),
        v_d=jnp.zeros((6, N + 1, N)),
        h_s=jnp.zeros((6, N, N)),
    )
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(NSTEPS):
            state = model.step(state, DT)
    return {
        "h_drift_max": float(jnp.max(jnp.abs(state.h - H0))),
        "u_d_max": float(jnp.max(jnp.abs(state.u_d))),
        "v_d_max": float(jnp.max(jnp.abs(state.v_d))),
        "h_field": np.asarray(state.h),
        "u_d_field": np.asarray(state.u_d),
        "v_d_field": np.asarray(state.v_d),
    }


@pytest.fixture(scope="module")
def _rest_state_iter893():
    return _run_rest_state(apply_xppm=True)


@pytest.fixture(scope="module")
def _rest_state_iter820():
    return _run_rest_state(apply_xppm=False)


def test_iter925_production_rest_state_winds_at_machine_precision(
    _rest_state_iter893,
):
    """u_d and v_d should stay at machine precision (~1e-15) after a
    full day at C36 with iter-893 flag.  iter-925 measured u_d_max =
    4.18e-15, v_d_max = 4.17e-15.  Pin under 1e-12.
    """
    out = _rest_state_iter893
    assert out["u_d_max"] < 1e-12, (
        f"Rest state u_d_max = {out['u_d_max']:.3e} m/s exceeds "
        f"machine-precision gate (1e-12).  iter-925 baseline: 4.18e-15."
    )
    assert out["v_d_max"] < 1e-12, (
        f"Rest state v_d_max = {out['v_d_max']:.3e} m/s exceeds "
        f"machine-precision gate (1e-12).  iter-925 baseline: 4.17e-15."
    )


def test_iter925_production_rest_state_h_drift_below_round_off_floor(
    _rest_state_iter893,
):
    """h drift should stay below ~1e-3 m of H0=1000 m (round-off
    accumulation floor).  iter-925 measured 1.010e-4 m.  Pin under
    1e-3 to allow margin.
    """
    out = _rest_state_iter893
    assert out["h_drift_max"] < 1e-3, (
        f"Rest state max|h - H0| = {out['h_drift_max']:.3e} m exceeds "
        f"the 1e-3 m round-off floor.  iter-925 baseline: 1.010e-4 m.  "
        f"This signals the flux-divergence path is no longer round-off "
        f"-bound on constants."
    )


def test_iter925_iter893_is_bit_exact_no_op_on_rest_state(
    _rest_state_iter893, _rest_state_iter820,
):
    """The iter-893 PPM boundary fix should be a BIT-EXACT no-op on
    rest state: PPM reconstruction of a constant field gives q_R =
    q_L = H0 regardless of which boundary formula is used.

    iter-925 verified bit-exact match: iter-820 and iter-893 both
    give h_drift_max = 1.010e-4 m and u_d_max = 4.18e-15 m/s.

    If a future iter makes iter-893 non-trivial on rest state, it
    means the boundary formula has acquired a non-zero contribution
    on constants — likely a bug.
    """
    on = _rest_state_iter893
    off = _rest_state_iter820
    np.testing.assert_array_equal(
        on["h_field"], off["h_field"],
        err_msg=(
            "iter-893 PPM boundary fix is no longer bit-exact on "
            "rest state h field — non-zero contribution on constants."
        ),
    )
    np.testing.assert_array_equal(
        on["u_d_field"], off["u_d_field"],
        err_msg=(
            "iter-893 PPM boundary fix is no longer bit-exact on "
            "rest state u_d field."
        ),
    )
    np.testing.assert_array_equal(
        on["v_d_field"], off["v_d_field"],
        err_msg=(
            "iter-893 PPM boundary fix is no longer bit-exact on "
            "rest state v_d field."
        ),
    )
