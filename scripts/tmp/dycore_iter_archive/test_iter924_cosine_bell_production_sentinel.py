"""Iter-924 regression sentinel: pin cosine bell C36 day-1 production
diagnostics with `apply_fortran_xppm_boundary=True` (iter-893
production matrix).

Background.  iter-921→923 audit:

| test         | Δ across iter-820↔iter-893  | trade-off?  |
|--------------|-----------------------------|-------------|
| W2 1-day     | v_ll −17 %, h_err_max +77 % | YES (Pareto) |
| W5 day-3     | all 6 metrics ≤ 1 %         | NO          |
| Cosine bell  | all 4 metrics ≤ 0.13 %      | NO          |

Cosine bell is the pure-transport leg of the cube-sphere validation
triad and is structurally insensitive to the iter-893 PPM-boundary
fix.  iter-924 closes the production-sentinel loop by adding multi-
metric pins on cosine bell day-1 production.

Cost: ~25 s wall (one C36 1-day cosine-bell trajectory cached via
module fixture).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
)
from legoesm.core.fv3_sw_core import d2a2c_vect
from legoesm.core.fv_tp_2d import transport_step
from legoesm.grids.cubed_sphere import create_cubed_sphere
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere,
    cosine_bell_exact,
)


N = 36
DT = 1440.0
DAYS = 1.0
BETA = jnp.pi / 4.0
NSTEPS = int(round(DAYS * 86400 / DT))
TOL_PCT = 0.05  # ±5 % around iter-924 measurements


def _div_damp_cube(n: int, ref_n: int = 48, ref_coeff: float = 1.5e7) -> float:
    return ref_coeff * (ref_n / n) ** 2


@pytest.fixture(scope="module")
def _cb_production_iter893():
    """Cosine bell day-1 with apply_fortran_xppm_boundary=True."""
    grid = create_cubed_sphere(N)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=_div_damp_cube(N),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid
    state = cosine_bell_cubesphere(grid, cdgrid, BETA)
    _, _, _, _, ut, vt = d2a2c_vect(state.u_d, state.v_d, cdgrid)
    mass_init = float(jnp.sum(state.h * grid.area))
    h = state.h
    for _ in range(NSTEPS):
        h = transport_step(
            h, ut, vt, DT, cdgrid,
            mass_target=mass_init,
            apply_fortran_xppm_boundary=True,
        )
    t_s = DAYS * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, BETA)
    h_num = np.asarray(h)
    h_ex = np.asarray(h_exact)
    h_err = h_num - h_ex
    return {
        "h_num_max": float(np.abs(h_num).max()),
        "Linf_err": float(np.max(np.abs(h_err))),
        "L2_err": float(np.sqrt(np.mean(h_err**2))),
        "rel_L2": float(
            np.sqrt(np.mean(h_err**2)) / np.sqrt(np.mean(h_ex**2))
        ),
        "mass_drift": float(
            (np.asarray(h * grid.area).sum() - mass_init) / mass_init
        ),
    }


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("h_num_max",  895.7),     # 8.9574e+02
        ("Linf_err",   121.3),     # 1.2129e+02
        ("L2_err",     8.114),     # 8.1136e+00
        ("rel_L2",     0.1194),    # 1.1938e-01
    ],
)
def test_iter924_cb_production_metric_pinned(
    _cb_production_iter893, metric, expected,
):
    """Pin each cosine bell day-1 production diagnostic within ±5 %."""
    measured = _cb_production_iter893[metric]
    assert np.isfinite(measured)
    rel = abs(measured - expected) / abs(expected)
    assert rel < TOL_PCT, (
        f"Cosine bell {metric} = {measured:.4e} drifted by "
        f"{rel*100:.2f} % from iter-924's {expected:.4e}.  "
        f"Tolerance ±5 %.  See diag_iter823_cb_visual.py for "
        f"the equivalent visual diagnostic."
    )


def test_iter924_cb_mass_conservation_within_observed_band(
    _cb_production_iter893,
):
    """`transport_step(... mass_target=...)` enforces mass at each step
    via a target.  iter-924 measured drift |Δm/m| ~ 1.92e-7 across
    60 steps — not machine precision but well below the bell-mass
    scale (~12 % of the bell amplitude).  Pin within 5e-7 to catch
    catastrophic conservation failure without false-firing on the
    expected accumulation drift.
    """
    drift = _cb_production_iter893["mass_drift"]
    assert abs(drift) < 5e-7, (
        f"Cosine bell mass-drift |Δm/m| = {abs(drift):.3e} exceeds "
        f"the 5e-7 conservation gate.  iter-924 baseline: 1.92e-7."
    )
