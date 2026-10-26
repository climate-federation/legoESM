"""The classical WB arm's prescribed-flux hand-off through the real rollout.

Pins delivery, not plumbing: the split-radiation STATEFUL ``spectral_rollout``
(the path the WB classical arm trains on) must carry the four prescribed
turbulent fluxes from the forcing dict into the surface scheme for every
step, the radiative LW boundary condition into the radiation call, and refuse
a partial flux set.  T10 / 4 levels / gray radiation so it runs in seconds.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis_3d
from legoesm.training.aimip_params import (
    AIMIPClassicalParams,
    make_aimip_classical_spectral_physics,
)
from legoesm.training.neural_gcm_spectral import (
    carry_to_spectral_state,
    spectral_rollout,
)

from tests.unit.test_learned_column import _mini_spectral_state

NLEV = 4
DT = 600.0
N_STEPS = 2


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(n_max=10)
    carry, sigma = _mini_spectral_state(grid, NLEV)
    state = carry_to_spectral_state(carry, grid)
    non_rad, rad = make_aimip_classical_spectral_physics(
        AIMIPClassicalParams.from_defaults(), grid, dt=DT, radiation="gray",
        split_rad=True, turbulence_scheme="louis")
    ncol = len(grid.lat) * len(grid.lon)
    base = {
        "T_sfc": jnp.full((ncol,), 300.0),
        "sic": jnp.zeros((ncol,)),
        "day_of_year": jnp.asarray(180.0),
        "seconds_of_day": jnp.asarray(43200.0),
    }
    pe = SpectralPEConfig(time_integrator="ssp_rk3")

    def run(forcing):
        return spectral_rollout(
            state, non_rad, grid, sigma, pe, DT, N_STEPS, None, None,
            forcing_base=forcing, rad_physics_fn=rad, rad_update_interval=1)

    def lowest_T_mean(out):
        return float(sh_synthesis_3d(grid, out.T_hat.data)[..., -1].mean())

    return grid, ncol, base, run, lowest_T_mean


def _fluxes(ncol, shf=0.0, tau_x=0.0, tau_y=0.0):
    return {
        "sfc_shf": jnp.full((ncol,), shf),
        "sfc_lhf": jnp.zeros((ncol,)),
        "sfc_tau_x": jnp.full((ncol,), tau_x),
        "sfc_tau_y": jnp.full((ncol,), tau_y),
    }


def test_prescribed_sensible_flux_reaches_the_lowest_level(setup):
    """+200 W/m2 warms, -200 cools, symmetrically about the interactive run."""
    grid, ncol, base, run, lowest_T = setup
    t0 = lowest_T(run(base))
    t_up = lowest_T(run({**base, **_fluxes(ncol, shf=200.0)}))
    t_dn = lowest_T(run({**base, **_fluxes(ncol, shf=-200.0)}))
    assert t_up > t0 > t_dn
    # Louis applies the flux linearly as the diffusion's lower boundary
    # condition, so the two departures from the interactive run are mirror
    # images (they need not vanish: the interactive flux is not zero).
    d_up, d_dn = t_up - t0, t0 - t_dn
    assert abs(d_up - d_dn) < 0.05 * max(d_up, d_dn)
    assert d_up > 0.01  # K over two 600 s steps at 200 W/m2: not a rounding


def test_prescribed_stress_scales_the_momentum_tendency_linearly(setup):
    """x3 stress -> x3 vorticity departure from the zero-stress run."""
    grid, ncol, base, run, _ = setup
    rng = np.random.default_rng(0)
    tx = jnp.asarray(rng.normal(0.0, 0.1, ncol))
    ty = jnp.asarray(rng.normal(0.0, 0.1, ncol))
    ref = run({**base, **_fluxes(ncol)}).vor_hat.data
    d1 = run({**base, **_fluxes(ncol), "sfc_tau_x": tx, "sfc_tau_y": ty}
             ).vor_hat.data - ref
    d3 = run({**base, **_fluxes(ncol), "sfc_tau_x": 3 * tx, "sfc_tau_y": 3 * ty}
             ).vor_hat.data - ref
    assert float(jnp.max(jnp.abs(d1))) > 0.0
    ratio = float(jnp.max(jnp.abs(d3)) / jnp.max(jnp.abs(d1)))
    assert ratio == pytest.approx(3.0, rel=0.05)


def test_prescribed_lw_up_changes_the_radiative_state(setup):
    grid, ncol, base, run, _ = setup
    a = run({**base, "sfc_lw_up": jnp.full((ncol,), 500.0)}).T_hat.data
    b = run({**base, "sfc_lw_up": jnp.full((ncol,), 300.0)}).T_hat.data
    assert float(jnp.max(jnp.abs(a - b))) > 0.0


def test_partial_flux_set_is_refused(setup):
    grid, ncol, base, run, _ = setup
    partial = {**base, "sfc_shf": jnp.zeros((ncol,)),
               "sfc_lhf": jnp.zeros((ncol,))}
    with pytest.raises(KeyError, match="sfc_tau_x"):
        run(partial)
