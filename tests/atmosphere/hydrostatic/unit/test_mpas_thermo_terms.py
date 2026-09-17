"""The diagnostic split of the MPAS thermodynamic tendency.

``mpas_hydrostatic_tendencies(..., return_thermo_terms=True)`` hands back the
three arrays the dycore itself sums into dT/dt.  The split exists so a heating
budget can attribute the "dynamics" residual without re-deriving anything from
the state; it must be inert unless it is asked for.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig, ThermoTerms, mpas_hydrostatic_tendencies,
    vertical_del4_T_tendency)
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.grids.voronoi import create_voronoi_mesh


@pytest.fixture(autouse=True)
def fp64_policy():
    """fp64 storage: the split is an exact identity, and at fp32 the residual
    of three cancelling terms is ~1e-4 relative, which would force a tolerance
    loose enough to hide a genuinely wrong term."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _fixture(nu_vert4_T=0.0, spin=3, K_h=0.0):
    """A spun-up state: the Held-Suarez initial condition is at rest, so the
    horizontal advection term is identically zero there and could not tell a
    working split from a broken one."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel)
    from legoesm.atmosphere.idealized.held_suarez_topo import (
        held_suarez_topo_init_mpas)
    mesh = create_voronoi_mesh(3)
    coord = create_sigma_coordinate(10)
    cfg = MPASPrimitiveEquationConfig(fix_mass=False, nu_vert4_T=nu_vert4_T,
                                      K_h=K_h)
    state = held_suarez_topo_init_mpas(mesh, coord)
    model = MPASPrimitiveEquationModel(mesh, coord, cfg)
    for _ in range(spin):
        state = model.step(state, 300.0)
    return mesh, coord, cfg, state


def test_terms_sum_to_the_thermodynamic_tendency():
    mesh, coord, cfg, state = _fixture()
    tend, terms = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg, None, 0.0, return_thermo_terms=True)
    assert isinstance(terms, ThermoTerms)
    parts = (terms.horiz_adv + terms.horiz_diff + terms.vert_adv
             + terms.adiabatic_ps)
    np.testing.assert_allclose(np.asarray(parts), np.asarray(tend.dT_dt.data),
                               rtol=0.0, atol=0.0)   # same order, same dtype
    # and each term must actually carry signal, else the split is decorative
    for f in ("horiz_adv", "vert_adv", "adiabatic_ps"):
        assert float(jnp.max(jnp.abs(getattr(terms, f)))) > 0.0, f
    assert float(jnp.max(jnp.abs(terms.horiz_diff))) == 0.0   # K_h = 0 here
    # the coordinate vertical velocity comes back too, at interfaces
    assert terms.sigma_dot.shape[1] in (tend.dT_dt.data.shape[1],
                                        tend.dT_dt.data.shape[1] + 1)
    assert bool(jnp.all(jnp.isfinite(terms.sigma_dot)))
    assert float(jnp.max(jnp.abs(terms.sigma_dot))) > 0.0


def test_horizontal_diffusion_is_reported_separately():
    """With K_h > 0 the Laplacian diffusion of T must appear in its OWN term.

    It used to be folded into the advection term, which made a diffusive warming
    of the winter polar cap read as resolved poleward heat transport.  The sum
    still has to close, so this cannot be satisfied by double counting.
    """
    K_h = 1.0e5                                   # production-scale value
    mesh, coord, cfg, state = _fixture(K_h=K_h)
    tend, terms = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg, None, 0.0, return_thermo_terms=True)
    assert float(jnp.max(jnp.abs(terms.horiz_diff))) > 0.0
    parts = (terms.horiz_adv + terms.horiz_diff + terms.vert_adv
             + terms.adiabatic_ps)
    np.testing.assert_allclose(np.asarray(parts), np.asarray(tend.dT_dt.data),
                               rtol=0.0, atol=0.0)
    # and the advection term must be the K_h = 0 advection, unchanged
    _, terms0 = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg._replace(K_h=0.0), None, 0.0,
        return_thermo_terms=True)
    np.testing.assert_allclose(np.asarray(terms.horiz_adv),
                               np.asarray(terms0.horiz_adv),
                               rtol=0.0, atol=0.0)


def test_split_excludes_the_vertical_filter():
    """With the del4 filter on, dT/dt carries it but the three terms do not,
    so a budget can report the filter as its own row without double counting."""
    nu4 = 2.0e-6
    mesh, coord, cfg, state = _fixture(nu_vert4_T=nu4)
    tend, terms = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg, None, 0.0, return_thermo_terms=True)
    parts = (terms.horiz_adv + terms.horiz_diff + terms.vert_adv
             + terms.adiabatic_ps)
    filt = vertical_del4_T_tendency(state.T.data, nu4, coord.dsigma)
    assert float(jnp.max(jnp.abs(filt))) > 0.0           # control: it is active
    np.testing.assert_allclose(np.asarray(parts + filt),
                               np.asarray(tend.dT_dt.data),
                               rtol=0.0, atol=0.0)


def test_production_call_is_unchanged():
    """Omitting the flag returns the tendencies object itself, not a tuple, and
    the same numbers; the flag is a static Python bool so jit sees neither."""
    mesh, coord, cfg, state = _fixture()
    plain = mpas_hydrostatic_tendencies(state, mesh, coord, cfg, None, 0.0)
    # the tendencies object is itself a NamedTuple, so "not a tuple" is the
    # wrong check — what matters is that no ThermoTerms came back with it
    assert not (len(plain) == 2 and isinstance(plain[1], ThermoTerms))
    assert hasattr(plain, "dT_dt")
    with_terms, _ = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg, None, 0.0, return_thermo_terms=True)
    np.testing.assert_allclose(np.asarray(plain.dT_dt.data),
                               np.asarray(with_terms.dT_dt.data),
                               rtol=0.0, atol=0.0)
    jitted = jax.jit(lambda s: mpas_hydrostatic_tendencies(
        s, mesh, coord, cfg, None, 0.0).dT_dt.data)
    np.testing.assert_allclose(np.asarray(jitted(state)),
                               np.asarray(plain.dT_dt.data),
                               rtol=1e-12, atol=0.0)
