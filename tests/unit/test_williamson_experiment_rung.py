"""Williamson TC2 as a Stage-A2 ``Experiment`` rung — the dycore A2-exit rung.

This is the second of the two A2-exit rungs (the first being the single-column
gradient check): it wraps a real cubed-sphere shallow-water integration in the
``Experiment`` harness, so "run Williamson TC2 and verify it" is one call with a
pass/fail verdict.  Reference bounds are grounded in the existing C16 unit test
(``test_williamson2_cdgrid``): the steady geostrophic flow's height field must
not drift beyond ``l2_norm < 0.5`` and mass must conserve to ``rel_err < 1e-4``.

A *test*, not a ``src`` module, because it reuses the test-side Williamson IC
builder (the in-repo characterization half of the §6 matrices lives in tests/).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.experiments import Experiment, MetricCheck
# Import the IC from the side-effect-free shared test_cases helper (NOT from
# test_williamson2_cdgrid, which globally mutates jax x64 at import).
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson2_cdgrid_initial_condition,
)

_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="dycore validation needs JAX_ENABLE_X64=1",
)


class WilliamsonTC2Experiment(Experiment):
    """Steady geostrophic flow (Williamson 1992 TC2) preserved by the SW dycore.

    TC2 is an exact steady state, so any drift of the height field from the
    analytic (= initial) solution is numerical error.
    """

    name = "williamson_tc2_cdgrid"
    description = (
        "Williamson (1992) TC2 steady geostrophic flow on the cubed-sphere C-D "
        "grid shallow-water core; height drift + mass conservation."
    )

    def __init__(self, n: int = 16, n_days: float = 5.0, dt: float = 600.0) -> None:
        self.n = n
        self.n_days = n_days
        self.dt = dt

    def run(self) -> dict[str, float]:
        from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterConfig,
            CDGridShallowWaterModel,
        )
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

        grid = create_cubed_sphere(self.n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        dx_min = float(jnp.min(grid.dx))
        config = CDGridShallowWaterConfig(
            A_h=1e4, hyperdiff_coeff=dx_min ** 4 / (86400.0 * 10)
        )
        model = CDGridShallowWaterModel(grid, config)
        state0 = williamson2_cdgrid_initial_condition(cdgrid)

        n_steps = int(self.n_days * 86400 / self.dt)
        state = state0
        for _ in range(n_steps):
            state = model.step(state, self.dt)

        area = cdgrid.base.area
        total_area = jnp.sum(area)

        # Area-weighted normalized L2 height-field drift from the steady IC.
        h_err = state.h - state0.h
        l2 = float(jnp.sqrt(jnp.sum(h_err ** 2 * area) / total_area))
        h_range = float(jnp.max(state0.h) - jnp.min(state0.h))
        l2_norm = l2 / max(h_range, 1.0)

        # Mass conservation.
        mass0 = float(jnp.sum(state0.h * area))
        massf = float(jnp.sum(state.h * area))
        mass_rel = abs(massf - mass0) / abs(mass0)

        all_finite = bool(
            jnp.all(jnp.isfinite(state.h))
            and jnp.all(jnp.isfinite(state.u_d))
            and jnp.all(jnp.isfinite(state.v_d))  # every prognostic field
        )
        return {
            "l2_height_error": l2_norm,
            "mass_rel_error": mass_rel,
            "all_finite": float(all_finite),
        }

    @property
    def checks(self) -> dict[str, MetricCheck]:
        return {
            "l2_height_error": MetricCheck(reference=0.5, kind="below"),
            "mass_rel_error": MetricCheck(reference=1e-4, kind="below"),
            "all_finite": MetricCheck(reference=1.0, kind="above"),  # must be finite
        }


@_needs_x64
def test_williamson_tc2_experiment_rung_passes() -> None:
    """The Williamson TC2 Experiment rung runs and passes (A2 exit criterion).

    Runs the same 5-day C16 window the grounded reference bounds come from, so the
    rung's verdict matches the established characterization (not a weaker subset).
    """
    result = WilliamsonTC2Experiment(n=16, n_days=5.0).evaluate()
    assert result.passed, result.summary()
    assert result.metrics["l2_height_error"] < 0.5
    assert result.metrics["mass_rel_error"] < 1e-4
