"""Iter-766b: measure W2 C36 1-day with the iter-766 `fortran_a2b_corner_avg`
flag ON, vs default (OFF).  Direct test whether Fortran's a2b_ord4 3-point
cube-corner average reduces the mode-A v-wind artifact.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    """Match run_atmosphere_test_matrix._div_damp_cube (quadratic, not quartic)."""
    return ref_coeff * (ref_n / n) ** 2
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests", "unit"))
from test_williamson2_cdgrid import williamson2_initial_condition


n = 36
nsteps = 86400 // 60
dt = 60.0
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

h0, u_d0, v_d0, h_s = williamson2_initial_condition(cdgrid)

weights = get_cubedsphere_to_latlon_weights(n, 360, 181)


def run(fortran_a2b_corner_avg: bool):
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        fortran_a2b_corner_avg=fortran_a2b_corner_avg,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState)
    state = FV3EdgeShallowWaterState(h=h0, u_d=u_d0, v_d=v_d0, h_s=h_s)
    for _ in range(nsteps):
        state = model.step(state, dt)
    return state


def measure(state, name):
    h = np.asarray(state.h)
    h_err_l2 = float(np.sqrt(np.mean((h - np.asarray(h0)) ** 2)))
    h_err_linf = float(np.max(np.abs(h - np.asarray(h0))))
    # Compute v_north via the model's D→A interpolation then local rotation.
    # Use the same approach as the matrix script: unpack state velocity.
    from legoesm.core.operators_cdgrid import fv3_d2cc
    u_cc, v_cc = fv3_d2cc(state.u_d, state.v_d, cdgrid)
    u_cc_np = np.asarray(u_cc)
    v_cc_np = np.asarray(v_cc)
    # Convert grid-aligned -> geographic via grid angle.
    cos_a = np.cos(np.asarray(grid.angle))
    sin_a = np.sin(np.asarray(grid.angle))
    u_east = cos_a * u_cc_np - sin_a * v_cc_np
    v_north = sin_a * u_cc_np + cos_a * v_cc_np
    v_ll = np.asarray(apply_cubedsphere_to_latlon(v_north, weights))
    v_ll_linf = float(np.max(np.abs(v_ll)))
    v_north_linf = float(np.max(np.abs(v_north)))
    print(f"[{name}] h_L2={h_err_l2:.3e}  h_Linf={h_err_linf:.3e}  "
          f"v_north_Linf={v_north_linf:.3e}  v_ll_Linf={v_ll_linf:.3e}")
    return h_err_l2, h_err_linf, v_ll_linf


# Baseline
state_off = run(fortran_a2b_corner_avg=False)
measure(state_off, "OFF (2-pt-avg, default)")

# With flag
state_on = run(fortran_a2b_corner_avg=True)
measure(state_on, "ON  (Fortran a2b 3-pt)  ")
