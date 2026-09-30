"""The energy PV flux is fenced off from the momentum-update fusion.

Removing the barrier is a silent performance regression (the fused kernel
spills registers above ~80k cells/GPU), so its presence in the traced
tendency is pinned here; the barrier must not change values or gradients.
"""
from __future__ import annotations

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig, mpas_hydrostatic_tendencies)
from legoesm.grids.vertical import create_sigma_coordinate


def _helpers():
    path = pathlib.Path(__file__).with_name("test_mpas_atmosphere.py")
    spec = importlib.util.spec_from_file_location("_mpas_atm_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup():
    h = _helpers()
    mesh, nlev = h._make_mesh(level=2), 4
    sig = create_sigma_coordinate(nlev, dtype=jnp.float64)
    state = h._add_perturbation_hydro(h._make_hydrostatic_state(mesh, nlev), mesh, nlev)
    return state, mesh, sig


def test_barrier_only_when_fenced():
    state, mesh, sig = _setup()
    cfg = MPASPrimitiveEquationConfig()
    assert cfg.pv_scheme == "energy"

    def jaxpr(fence):
        return str(jax.make_jaxpr(lambda s: mpas_hydrostatic_tendencies(
            s, mesh, sig, cfg, fence_pv_flux=fence))(state))

    assert "optimization_barrier" in jaxpr(True)
    assert "optimization_barrier" not in jaxpr(False)


def test_sharded_step_fences_both_call_sites():
    import inspect

    import legoesm.parallel.sharded_dynamics as sd
    src = inspect.getsource(sd.make_voronoi_sharded_step)
    assert src.count("fence_pv_flux=True") == 2


def test_barrier_is_value_and_gradient_neutral_under_jit():
    state, mesh, sig = _setup()
    cfg = MPASPrimitiveEquationConfig()

    def tend(s, fence=True):
        return mpas_hydrostatic_tendencies(
            s, mesh, sig, cfg, fence_pv_flux=fence).du_dt.data

    np.testing.assert_array_equal(np.asarray(tend(state)),
                                  np.asarray(tend(state, fence=False)))
    eager, jitted = np.asarray(tend(state)), np.asarray(jax.jit(tend)(state))
    assert np.all(np.isfinite(eager))
    np.testing.assert_allclose(jitted, eager, rtol=1e-12, atol=1e-18)

    u0 = state.u.data

    def loss(u):
        return jnp.sum(tend(state._replace(u=state.u.replace(data=u))) ** 2)

    check_grads(loss, (u0,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)


def test_barrier_gradient_equals_unfenced_gradient_under_jit():
    state, mesh, sig = _setup()
    cfg = MPASPrimitiveEquationConfig()

    def grad_u(fence):
        def loss(u):
            s = state._replace(u=state.u.replace(data=u))
            return jnp.sum(mpas_hydrostatic_tendencies(
                s, mesh, sig, cfg, fence_pv_flux=fence).du_dt.data ** 2)
        return np.asarray(jax.jit(jax.grad(loss))(state.u.data))

    fenced, unfenced = grad_u(True), grad_u(False)
    assert np.all(np.isfinite(fenced)) and np.any(fenced != 0.0)
    np.testing.assert_allclose(fenced, unfenced, rtol=1e-12, atol=1e-18)
