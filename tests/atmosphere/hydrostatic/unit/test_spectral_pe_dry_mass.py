"""Spectral-lane DRY-mass convention (review 2026-10-10 F13 remainder).

Before: physics water tendencies entered the spectral PE as dq/dt with no
surface-pressure change, and the mass anchor held the TOTAL ``∫ p_s dA``, so
precipitated water stayed behind as dry air.  Now the physics water carries
its mass into the lnps tendency (tracers re-weighted onto the new layer
masses) and the anchor holds the global DRY mass (one p_s factor; the
transport fixer pre-subtracts its share of the tracer masses).

Setup: the F21 test's T21 L10 flow over a mountain (mass anchor on, x64) plus
a physics stand-in that removes vapour at a known rate (a pure rain sink) and
a passive non-water tracer.  After N steps:

* the global dry mass equals its initial value (old code: grows by the rain);
* ``∫ p_s dA`` falls by exactly the water that left (old code: unchanged);
* the vapour mass decays as the sink prescribes and the passenger's mass is
  untouched.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.conservation import dry_surface_pressure
from legoesm.core.field import Field
from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels
from tests.atmosphere.hydrostatic.unit.test_spectral_pe_tracer_mass import (
    NLEV, _setup,
)

jax.config.update("jax_enable_x64", True)

N_STEPS, DT, K_SINK = 10, 600.0, 2.0e-5


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(n_max=21)


def _rain_physics(s, grid, sigma_coord):
    """Vapour sink -K q_v (all of it rains out); the passenger gets none."""
    z = lambda f: f.replace(data=jnp.zeros_like(f.data))  # noqa: E731
    tr = {k: (v.replace(data=-K_SINK * v.data) if k == "q_v"
              else v.replace(data=jnp.zeros_like(v.data)))
          for k, v in s.tracers.items()}
    return s._replace(vor_hat=z(s.vor_hat), div_hat=z(s.div_hat),
                      T_hat=z(s.T_hat), lnps_hat=z(s.lnps_hat),
                      phis_hat=z(s.phis_hat), tracers=tr)


def _masses(grid, coord, s):
    """(∫ p_s dA, ∫ water dp dA, ∫ passenger dp dA) [Pa m^2]."""
    ps = jnp.exp(sh_synthesis(grid, s.lnps_hat.data))
    dp = coord.layer_thickness_dp(ps)
    a = grid.grid_area
    return (float(jnp.sum(a * ps)),
            float(jnp.sum(a[..., None] * dp * s.tracers["q_v"].data)),
            float(jnp.sum(a[..., None] * dp * s.tracers["tr"].data)))


def spectral_dry_mass(grid, coord, s):
    """∫ dry p_s dA [Pa m^2], written here from the shared dry surface
    pressure (not the module's helper) so the test also runs on the old
    code."""
    ps = jnp.exp(sh_synthesis(grid, s.lnps_hat.data))
    return jnp.sum(grid.grid_area * dry_surface_pressure(ps, s.tracers, coord))


def _with_passenger(s):
    q = s.tracers["q_v"]
    return s._replace(tracers={
        # no uniform offset: on this setup a UNIFORM tracer drifts 6e-8 in
        # 10 steps on origin/main too (pre-existing, outside F13)
        "q_v": q, "tr": Field(data=0.5 * q.data, name="tr",
                              dims=q.dims, units="kg/kg")})


@pytest.mark.parametrize("coord_kind", ["sigma", "hybrid"])
def test_rain_leaves_the_column_mass_and_dry_mass_is_kept(grid, coord_kind):
    coord = (create_sigma_coordinate(NLEV) if coord_kind == "sigma"
             else make_hybrid_levels(NLEV, p_top_Pa=200.0, stretching=2.0))
    model, s = _setup(grid, coord)
    s = _with_passenger(s)
    dry0 = float(spectral_dry_mass(grid, coord, s))
    P0, W0, R0 = _masses(grid, coord, s)
    for _ in range(N_STEPS):
        s = model.step(s, DT, physics_fn=_rain_physics)
    P1, W1, R1 = _masses(grid, coord, s)
    dry1 = float(spectral_dry_mass(grid, coord, s))

    # the anchor holds the DRY mass (old: total held, dry grows by the rain)
    assert abs(dry1 - dry0) / dry0 < 1e-12, (dry1 - dry0) / dry0
    # the vapour decays as the sink prescribes (RK error only) ...
    w_exp = W0 * np.exp(-K_SINK * N_STEPS * DT)
    assert abs(W1 - w_exp) / (W0 - w_exp) < 1e-4, (W1, w_exp)
    # ... and that water left the column mass (old: P1 == P0)
    assert abs((P1 - P0) - (W1 - W0)) / (W0 - W1) < 1e-6, (P1 - P0, W1 - W0)
    # a passenger's mass is untouched by rain, re-weighting and anchor
    assert abs(R1 - R0) / R0 < 1e-8, (R1 - R0) / R0


@pytest.mark.parametrize("coord_kind", ["sigma", "hybrid"])
def test_dry_anchor_is_exact_and_leaves_mixing_ratios(grid, coord_kind):
    """One anchor call from a perturbed state: the dry mass hits the target
    (the affine solve is exact up to the transform's roundoff) and the
    mixing ratios are untouched (no re-weighting: the transport fixer
    pre-subtracts the anchor's share); the dry mass actually moved
    (non-vacuity)."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        anchor_lnps_to_dry_mass)
    coord = (create_sigma_coordinate(NLEV) if coord_kind == "sigma"
             else make_hybrid_levels(NLEV, p_top_Pa=200.0, stretching=2.0))
    _, s = _setup(grid, coord)
    s = _with_passenger(s)
    target = float(spectral_dry_mass(grid, coord, s))
    pert = s._replace(lnps_hat=s.lnps_hat.replace(
        data=s.lnps_hat.data.at[0].add(1e-3)))
    assert abs(float(spectral_dry_mass(grid, coord, pert)) - target) \
        / target > 1e-5
    out = anchor_lnps_to_dry_mass(grid, coord, pert, target)
    assert abs(float(spectral_dry_mass(grid, coord, out)) - target) \
        / target < 1e-11
    for k in ("q_v", "tr"):
        np.testing.assert_array_equal(np.asarray(out.tracers[k].data),
                                      np.asarray(pert.tracers[k].data))
