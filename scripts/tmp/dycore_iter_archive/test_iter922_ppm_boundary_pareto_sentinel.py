"""Iter-922 regression sentinel: pin the iter-893 default Pareto-non-
dominated position vs `fortran_faithful_ppm_left/right/both`.

Background.  iter-921 discovered the v_ll_Linf vs h_err_max trade-off
when toggling `apply_fortran_xppm_boundary`.  iter-922 extends the
audit to the iter-892/iter-900/iter-903 sub-flags
(`fortran_faithful_ppm_left/right`) and finds:

| label | xppm  | faithful_L | faithful_R | v_ll_Linf | h_err_max |
|-------|-------|------------|------------|-----------|-----------|
| A     | False |            |            | 0.1593    | 4.62 m    |
| B     | True  | False      | False      | 0.1319    | 8.18 m    | <- production
| C     | True  | True       | False      | 0.2027    | 10.02 m   |
| D     | True  | False      | True       | 0.1989    | 10.59 m   |
| E     | True  | True       | True       | 0.1713    | 8.62 m    |

Cases C, D, and E are PARETO-DOMINATED by case B (iter-893 default):
B has both lower v_ll_Linf AND lower h_err_max than each.

This sentinel pins that ordering by running ONE comparison case
(faithful_left=True, B vs C) and verifying B is strictly better on
both metrics.  Any future change to the iter-892 default-LEFT
formula that loses Pareto dominance over the strict-Fortran LEFT
formula on (v_ll_Linf, h_err_max) fires this test.

Why one case is sufficient: iter-892's LEFT 1-cell-shifted formula
is the LOAD-BEARING choice — it's why iter-893 default is better
than strict Fortran.  If C beats B on either metric, the iter-892
choice has been compromised and the production matrix needs
re-evaluation.

Cost: ~80 s wall (two C36 1-day trajectories with a module fixture
preventing duplicate runs across the file's tests).
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


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


def _run(faithful_left: bool, faithful_right: bool) -> dict:
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
        div_damp=8.0 * _div_damp_cube(N),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        fortran_faithful_ppm_left=faithful_left,
        fortran_faithful_ppm_right=faithful_right,
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
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return {
        "h_err_max": float(np.abs(h_err).max()),
        "v_ll_Linf": float(np.abs(v_ll).max()),
    }


@pytest.fixture(scope="module")
def _run_iter893_default():
    """Case B: iter-893 default (faithful_left=False, faithful_right=False)."""
    return _run(faithful_left=False, faithful_right=False)


@pytest.fixture(scope="module")
def _run_strict_fortran_left():
    """Case C: strict Fortran LEFT only (faithful_left=True, faithful_right=False)."""
    return _run(faithful_left=True, faithful_right=False)


def test_iter922_iter893_default_pareto_dominates_strict_fortran_left(
    _run_iter893_default, _run_strict_fortran_left,
):
    """Case B (iter-893 default) Pareto-dominates Case C (strict Fortran LEFT).

    Specifically: B has both LOWER v_ll_Linf AND LOWER h_err_max than C.
    """
    b = _run_iter893_default
    c = _run_strict_fortran_left
    assert np.isfinite(b["v_ll_Linf"]) and np.isfinite(b["h_err_max"])
    assert np.isfinite(c["v_ll_Linf"]) and np.isfinite(c["h_err_max"])
    assert b["v_ll_Linf"] < c["v_ll_Linf"], (
        f"iter-893 default v_ll_Linf={b['v_ll_Linf']:.4f} >= "
        f"strict Fortran LEFT v_ll_Linf={c['v_ll_Linf']:.4f}.  "
        f"The iter-892 default-LEFT 1-cell-shifted formula has lost its "
        f"v_ll_Linf advantage over strict Fortran — production matrix "
        f"needs re-evaluation."
    )
    assert b["h_err_max"] < c["h_err_max"], (
        f"iter-893 default h_err_max={b['h_err_max']:.3f} m >= "
        f"strict Fortran LEFT h_err_max={c['h_err_max']:.3f} m.  "
        f"The iter-892 default-LEFT 1-cell-shifted formula has lost its "
        f"h_err_max advantage over strict Fortran — production matrix "
        f"needs re-evaluation."
    )


def test_iter922_iter893_pareto_baseline_pinned(_run_iter893_default):
    """Pin the iter-893 default Pareto position at (0.1319, 8.18) within ±5 %.

    This is a redundant but defensive duplicate of iter-921's pin so this
    test file alone catches drift even when run in isolation.
    """
    b = _run_iter893_default
    assert abs(b["v_ll_Linf"] - 0.1319) / 0.1319 < 0.05, (
        f"iter-893 v_ll_Linf={b['v_ll_Linf']:.4f} drifted from 0.1319 m/s."
    )
    assert abs(b["h_err_max"] - 8.184) / 8.184 < 0.05, (
        f"iter-893 h_err_max={b['h_err_max']:.3f} drifted from 8.184 m."
    )


def test_iter922_strict_fortran_left_pinned(_run_strict_fortran_left):
    """Pin the strict-Fortran LEFT measurement at (0.2027, 10.02) within ±10 %.

    Wider tolerance because this is the worse-of-two case and may have
    higher iter-to-iter variability under unrelated changes.
    """
    c = _run_strict_fortran_left
    assert abs(c["v_ll_Linf"] - 0.2027) / 0.2027 < 0.10, (
        f"strict Fortran LEFT v_ll_Linf={c['v_ll_Linf']:.4f} drifted "
        f"from 0.2027 m/s."
    )
    assert abs(c["h_err_max"] - 10.015) / 10.015 < 0.10, (
        f"strict Fortran LEFT h_err_max={c['h_err_max']:.3f} drifted "
        f"from 10.015 m."
    )
