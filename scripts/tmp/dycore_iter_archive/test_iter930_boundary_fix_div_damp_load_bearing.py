"""Iter-930 sentinel: pin the 4-way (`boundary_fix` × `div_damp`)
production matrix on W2 C36 1-day to quantify which is the load-
bearing stabilizer.

iter-929 identified a 3-stage causal chain at the cube vertex:
  1. Bare A-L: 1 % imperfect cancellation at 8 cube vertices.
  2. boundary_fix: smooths row 0/n-1 with row 1/n-2 → spreads
     cube-vertex error.
  3. div_damp: amplifies the contaminated row-1 gradient ~7×.

iter-930 quantifies the integrated W2 1-day impact of toggling
the two stabilizers (with `apply_fortran_xppm_boundary=True`):

| `boundary_fix` | `div_damp` | v_ll_Linf  | h_L2   | h_Linf |
|----------------|------------|-----------|--------|--------|
| True (prod)    | 8×         | 0.1319    | 0.512  | 8.18   |
| False          | 8×         | 0.6380    | 1.751  | 34.29  |
| True           | 0          | 0.1857    | 0.645  | 6.02   |
| False          | 0          | 0.6616    | 1.826  | 20.72  |

Findings:
- `boundary_fix=False` → v_ll_Linf 4.8× worse regardless of div_damp.
- `div_damp=0` (with boundary_fix on) → v_ll_Linf only 1.4× worse.
- **`boundary_fix` is the dominant integrated-error stabilizer**
  for W2, even though iter-907 found div_damp dominates the
  INSTANTANEOUS hot-spot tendency magnitude.
- Production matrix (both on) is the BEST of the 4 cells.

This sentinel pins all 4 cells with ±5 % tolerance so future iters
that propose to remove `boundary_fix` (the "Python-only stabilizer"
per user issue #5) without a Fortran-faithful replacement are
caught — removal would push v_ll_Linf from 0.13 to 0.64 m/s.
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
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


N = 36
DT = 300.0
NSTEPS = int(86400 / DT)
TOL_PCT = 0.05  # ±5 %


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


def _run_w2_metrics(*, boundary_fix: bool, div_damp_factor: float) -> dict:
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp_factor * _div_damp_cube(N),
        boundary_fix=boundary_fix,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(NSTEPS):
            state = model.step(state, DT)
    h_err = np.asarray(state.h - sw.h.data)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (
        np.asarray(state.u_d)[:, :, :-1] + np.asarray(state.u_d)[:, :, 1:]
    )
    v_cc = 0.5 * (
        np.asarray(state.v_d)[:, :-1, :] + np.asarray(state.v_d)[:, 1:, :]
    )
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    return {
        "h_L2": float(np.sqrt(np.mean(h_err**2))),
        "h_Linf": float(np.abs(h_err).max()),
        "v_ll_Linf": float(
            np.abs(apply_cubedsphere_to_latlon(v_north, weights)).max()),
    }


@pytest.mark.parametrize(
    "boundary_fix, div_damp_factor, expected",
    [
        (True,  8.0, {"v_ll_Linf": 0.1319, "h_L2": 0.5116, "h_Linf":  8.184}),
        (False, 8.0, {"v_ll_Linf": 0.6380, "h_L2": 1.7514, "h_Linf": 34.29}),
        (True,  0.0, {"v_ll_Linf": 0.1857, "h_L2": 0.6447, "h_Linf":  6.023}),
        (False, 0.0, {"v_ll_Linf": 0.6616, "h_L2": 1.8256, "h_Linf": 20.72}),
    ],
    ids=[
        "production_bf_dd8",
        "no_bf_dd8",
        "bf_no_dd",
        "no_bf_no_dd",
    ],
)
def test_iter930_w2_4way_matrix_pinned(
    boundary_fix, div_damp_factor, expected,
):
    """Pin one of 4 (boundary_fix × div_damp) cells on W2 C36 1-day.

    Each parametric case takes ~15 s; total ~60 s for the four.
    """
    measured = _run_w2_metrics(
        boundary_fix=boundary_fix,
        div_damp_factor=div_damp_factor,
    )
    for metric, target in expected.items():
        assert np.isfinite(measured[metric]), (
            f"{metric} is NaN — the configuration is destabilized."
        )
        rel = abs(measured[metric] - target) / abs(target)
        assert rel < TOL_PCT, (
            f"({measured=}) {metric} drifted {rel*100:.2f}% from "
            f"iter-930's {target}."
        )


def test_iter930_boundary_fix_is_dominant_stabilizer_for_w2_v_ll(
):
    """Removing boundary_fix worsens v_ll_Linf 4× more than removing div_damp.

    iter-930 measured at the iter-893 production matrix:
      bf on,  dd 8× (production):  v_ll_Linf 0.132 m/s
      bf on,  dd 0:                v_ll_Linf 0.186 m/s  (+41 %)
      bf off, dd 8×:               v_ll_Linf 0.638 m/s  (+384 %)

    This sentinel pins the inequality:
      `bf=False` worsens v_ll more than 3× compared to `dd=0`.

    Refutes the casual reading of iter-907 "div_damp is 29× more
    than boundary_fix at hot spots" — that statement is about
    INSTANTANEOUS hot-spot tendencies, not integrated v_ll_Linf.
    """
    base = _run_w2_metrics(boundary_fix=True, div_damp_factor=8.0)
    no_bf = _run_w2_metrics(boundary_fix=False, div_damp_factor=8.0)
    no_dd = _run_w2_metrics(boundary_fix=True, div_damp_factor=0.0)

    bf_penalty = no_bf["v_ll_Linf"] - base["v_ll_Linf"]
    dd_penalty = no_dd["v_ll_Linf"] - base["v_ll_Linf"]
    ratio = bf_penalty / max(dd_penalty, 1e-30)
    assert ratio > 3.0, (
        f"boundary_fix penalty {bf_penalty:.3e} m/s should be > 3× "
        f"the div_damp penalty {dd_penalty:.3e} m/s.  Got ratio {ratio:.2f}."
    )
