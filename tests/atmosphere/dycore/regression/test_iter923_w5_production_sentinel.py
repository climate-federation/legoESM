"""Iter-923 regression sentinel: pin Williamson 5 (mountain) C36
day-3 production diagnostics.

Background.  iter-921 found a v_ll_Linf vs h_err_max Pareto trade-off
on W2 when toggling `apply_fortran_xppm_boundary`.  iter-923 ran
the same A/B on W5 (3-day mountain run) and measured ALL 7
diagnostics (h_max, h_min, h_change_max, h_change_l2, v_north_max,
v_ll_Linf, h_max_vert_deviation) within ±1 % across the swap.

W5 is therefore STRUCTURALLY INSENSITIVE to the iter-893 PPM
boundary fix — there is no Pareto trade-off to make explicit.  But
the iter-921 audit also revealed that previously-pinned single-
metric sentinels can miss multi-metric drift; iter-923 closes that
gap on W5 too by pinning multiple W5 production diagnostics.

Cost: ~70 s wall (one C36 3-day W5 trajectory cached via module
fixture).
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
    williamson_test5,
)


N = 36
DT = 300.0
DAYS = 3
NSTEPS = int(round(DAYS * 86400 / DT))
TOL_PCT = 0.05  # ±5 % around iter-923 measurements


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


@pytest.fixture(scope="module")
def _w5_production_iter893():
    """Run W5 day-3 production once and cache the diagnostics."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test5(grid)
    u0 = 20.0
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
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(NSTEPS):
            state = model.step(state, DT)
    h_final = np.asarray(state.h)
    h_initial = np.asarray(sw.h.data)
    h_change = h_final - h_initial
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
        "h_max": float(h_final.max()),
        "h_min": float(h_final.min()),
        "h_change_max": float(np.abs(h_change).max()),
        "h_change_l2": float(np.sqrt(np.mean(h_change**2))),
        "v_north_max": float(np.abs(v_north).max()),
        "v_ll_Linf": float(np.abs(v_ll).max()),
    }


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("h_max",        5965.0),    # 5.9650e+03
        ("h_min",        3903.9),    # 3.9039e+03
        ("h_change_max", 260.4),     # 2.6043e+02
        ("h_change_l2",  32.83),     # 3.2834e+01
        ("v_north_max",  24.78),     # 2.4781e+01
        ("v_ll_Linf",    24.58),     # 2.4583e+01
    ],
)
def test_iter923_w5_production_metric_pinned(
    _w5_production_iter893, metric, expected,
):
    """Pin each W5 day-3 production diagnostic within ±5 %.

    The 6 metrics are jointly pinned by the same module-fixture
    trajectory — one C36 3-day run feeds all 6 assertions.
    """
    measured = _w5_production_iter893[metric]
    assert np.isfinite(measured), (
        f"W5 {metric} is NaN — production path destabilized."
    )
    rel = abs(measured - expected) / abs(expected)
    assert rel < TOL_PCT, (
        f"W5 {metric} = {measured:.4e} drifted by {rel*100:.2f} % from "
        f"iter-923's {expected:.4e} baseline (tolerance ±5 %).  "
        f"If this is intentional, re-baseline this parametric entry."
    )


def test_iter923_w5_h_remains_in_physical_range(_w5_production_iter893):
    """Sanity check on the full h field range.

    iter-923 measured h_max=5965 m (within 1 m of the iter-820 5964.3 m)
    and h_min=3903.9 m (within 1 m of iter-820's 3903.7 m).  Any
    catastrophic destabilization (NaNs, huge ranges) fires this gate
    irrespective of the per-metric pins.
    """
    out = _w5_production_iter893
    assert 5800 < out["h_max"] < 6100, (
        f"W5 h_max={out['h_max']:.1f} m outside physical range [5800, 6100]."
    )
    assert 3700 < out["h_min"] < 4100, (
        f"W5 h_min={out['h_min']:.1f} m outside physical range [3700, 4100]."
    )
