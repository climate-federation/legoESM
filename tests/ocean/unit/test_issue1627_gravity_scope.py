"""How much the run's gravity actually moves the convection trigger (#1627).

The issue scoped the defect as five parts in a hundred thousand on a pressure,
which reads as harmless. A reviewer refused that reading on a good argument:
the adiabatic convection trigger compares a displaced parcel's density against
its surroundings -- a small difference of large numbers -- and the error is
sign-fixed, so it nudges every marginal layer the same way.

An earlier attempt to measure this was wrong three times and was deleted rather
than quoted. This one is deliberately modest: it computes, with the model's own
equation of state on a realistic column, what the two gravities do to the
density AND to the layer-to-layer density difference the trigger consumes. It
answers the magnitude question and NOT the marginal-layer question, which an
idealised column cannot answer.

The result: the pressure error is nearly uniform with depth, so most of it
CANCELS in the vertical difference, and what reaches the trigger is about one
part in seven thousand of a typical stratification.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG, ConstantsConfig
from legoesm.ocean.eos import wright_eos


def _rest_ocean():
    """A small stratified rest state -- the cheapest real state available."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=8, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0, H_max=4000.0,
    )
    return state, z_coord, grid

def _column(nz: int = 30, depth_m: float = 5000.0):
    dz = np.full(nz, depth_m / nz)
    T = np.linspace(18.0, 2.0, nz)
    S = np.linspace(35.0, 34.7, nz)
    rho_ref = np.full(nz, 1027.0)
    return dz, T, S, rho_ref


def _cell_pressure(g: float, rho_ref, dz):
    """Cell-centre hydrostatic pressure, the integral both helpers build."""
    return g * np.cumsum(rho_ref * dz) - 0.5 * g * rho_ref * dz


def _density(T, S, p):
    return np.array([float(wright_eos(jnp.asarray(T[k]), jnp.asarray(S[k]),
                                      jnp.asarray(p[k]))) for k in range(len(T))])


def test_the_two_gravities_differ_by_the_amount_the_issue_states():
    g_lib = ConstantsConfig().g
    g_nemo = NEMO_CONSTANTS_CONFIG.g
    assert (g_nemo - g_lib) / g_lib == pytest.approx(5.0e-5, rel=0.05)


def test_the_helpers_really_do_respond_to_the_gravity_they_are_handed():
    """The patched call sites' argument has to reach the answer.

    A reviewer's point, and a fair one: the column arithmetic below builds its
    own pressure and never calls the helper the patch threads gravity into, so
    on its own it could pass with every added argument deleted. This drives the
    real two-pass helper on a real state and shows the argument changes what it
    returns -- without which the whole change would be decorative.
    """
    from legoesm.ocean.eos import compute_ocean_rho

    state, z_coord, _grid = _rest_ocean()
    jac = jnp.ones_like(jnp.asarray(state.eta.data))

    rho_lib = compute_ocean_rho(state, z_coord, jac, g=ConstantsConfig().g)
    rho_nemo = compute_ocean_rho(state, z_coord, jac, g=NEMO_CONSTANTS_CONFIG.g)
    d = np.abs(np.asarray(rho_nemo) - np.asarray(rho_lib))
    assert d.max() > 0.0, (
        "the helper returns the same density for two different gravities, so "
        "the argument the patch threads reaches nothing")
    # and the size is the one the column arithmetic below predicts
    assert (d / np.asarray(rho_lib)).max() < 1.0e-5


def test_the_gravity_error_largely_cancels_in_the_stability_difference():
    """The number the issue asked for, and the reviewer's objection answered.

    Density itself shifts by about a milligram per cubic metre. What the
    convection trigger consumes is the DIFFERENCE between adjacent layers, and
    because the pressure error grows almost linearly with depth the shift is
    nearly the same in both, so it cancels to about one part in seven thousand
    of a typical stratification.

    This bounds the MAGNITUDE. It does not say how many marginal layers flip,
    which needs the real distribution of near-neutral layers in a run, not an
    idealised column -- so nothing here licenses calling the defect harmless.
    """
    dz, T, S, rho_ref = _column()
    rho_lib = _density(T, S, _cell_pressure(ConstantsConfig().g, rho_ref, dz))
    rho_nemo = _density(T, S, _cell_pressure(NEMO_CONSTANTS_CONFIG.g, rho_ref, dz))

    d_rho = np.abs(rho_nemo - rho_lib)
    assert d_rho.max() < 2.0e-3, f"density shift {d_rho.max():.3e} kg/m^3"
    assert (d_rho / rho_lib).max() < 2.0e-6

    # the quantity the trigger compares
    d_stability = np.abs(np.diff(rho_nemo - rho_lib))
    typical = np.abs(np.diff(rho_lib)).mean()
    ratio = d_stability.max() / typical
    assert ratio < 1.0e-3, (
        f"the gravity difference moves the layer-to-layer density difference by "
        f"{ratio:.2e} of a typical stratification ({d_stability.max():.2e} against "
        f"{typical:.2e} kg/m^3)")
    # and it is not zero either -- a test that passed on an inert quantity
    # would be measuring nothing
    assert d_stability.max() > 1.0e-6
