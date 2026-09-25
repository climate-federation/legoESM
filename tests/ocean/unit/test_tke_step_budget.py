"""The TKE solver's per-term step budget closes, and asking for it changes
nothing.

Every term of :class:`TKEStepBudget` is formed from the same matrix entries
the step was solved with, so their sum must equal ``tke_new - tke_old`` to
round-off on every surface-condition path the solver has (Neumann flux,
pinned Dirichlet, NEMO z=0 row with the 1.5-split dissipation, Langmuir and
nn_etau sources). Float64 is required: the identity is checked at 1e-12
relative, which float32 cannot resolve.

Non-vacuity: the identity must FAIL when the transport term is dropped, so a
budget that silently zeroed transport (the term the 2026-09-25 Mode-A
instrument exists to read) would be caught here.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    TKEStepBudget,
    _solve_tke_backward_euler,
    tke_vertical_mixing,
)


def _column(nlev=8):
    shape = (1, 2, nlev)
    T = jnp.asarray(np.linspace(18.0, 12.0, nlev))[None, None, :] * jnp.ones(shape)
    S = jnp.full(shape, 35.0)
    rho = jnp.asarray(1025.0 + 0.05 * np.arange(nlev))[None, None, :] * jnp.ones(shape)
    u = jnp.asarray(0.02 * np.arange(nlev))[None, None, :] * jnp.ones(shape)
    v = jnp.zeros(shape)
    dz_half = jnp.full((1, 2, nlev - 1), 10.0)
    z_interface = -10.0 * jnp.arange(1, nlev)
    tau_x = jnp.full((1, 2), 0.1)
    tau_y = jnp.zeros((1, 2))
    return dict(
        u_cell=u, v_cell=v, T_cell=T, S_cell=S, rho_cell=rho,
        dz_half=dz_half, tau_x_surface=tau_x, tau_y_surface=tau_y,
        rho_0=constants.rho_ocean, dz_ref=jnp.full((nlev,), 10.0),
        jacobian=jnp.ones((1, 2)), dz_surface=jnp.full((1, 2), 10.0),
        z_interface=z_interface, lat_deg=jnp.full((1, 2), 1.0),
        taum_surface=jnp.full((1, 2), 0.1),
    )


_CASES = {
    "neumann_flux": TKEConfig(),
    "interior_pinned": TKEConfig(surface_bc="nemo_dirichlet",
                                 tke_surface_bc_level="interior_pinned"),
    "nemo_z0_split_lc_etau": TKEConfig(
        surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0",
        dissipation_discretization="nemo_1p5_split", tke_mxl_choice=3,
        lc=True, etau_mode="below_ml", etau_htau_mode="latitude",
        alpha_tke=1.0),
}


def _sum_terms(b: TKEStepBudget, skip=()):
    return sum(getattr(b, f) for f in b._fields
               if f not in ("residual",) + tuple(skip))


@pytest.mark.parametrize("name", sorted(_CASES))
def test_budget_closes_and_primal_unchanged(name):
    cfg = _CASES[name]
    col = _column()
    tke_old = jnp.asarray(np.geomspace(1e-3, 1e-6, 7))[None, None, :] * jnp.ones((1, 2, 7))
    plain = tke_vertical_mixing(tke_old=tke_old, dt=600.0, cfg=cfg,
                                n_iterations=1, **col)
    with_b = tke_vertical_mixing(tke_old=tke_old, dt=600.0, cfg=cfg,
                                 n_iterations=1, return_budget=True, **col)
    assert plain.budget is None
    assert np.array_equal(np.asarray(plain.tke_new), np.asarray(with_b.tke_new))
    b = with_b.budget
    change = np.asarray(with_b.tke_new - tke_old)
    scale = float(np.max(np.abs(change)))
    assert scale > 0.0
    assert float(np.max(np.abs(np.asarray(b.residual)))) <= 1e-12 * scale
    total = np.asarray(_sum_terms(b))
    assert np.allclose(total, change, rtol=0.0, atol=1e-12 * scale)
    # every term carries signal somewhere (a zero term is a silent no-op)
    assert float(np.max(np.abs(np.asarray(b.transport)))) > 0.0
    assert float(np.max(np.asarray(b.production))) > 0.0
    assert float(np.max(np.abs(np.asarray(b.dissipation)))) > 0.0
    # non-vacuity: without transport the identity is broken
    without_transport = np.asarray(_sum_terms(b, skip=("transport",)))
    assert not np.allclose(without_transport, change, rtol=0.0, atol=1e-12 * scale)


def test_pinned_row_is_reported_as_pin_only():
    cfg = _CASES["interior_pinned"]
    col = _column()
    tke_old = jnp.full((1, 2, 7), 1e-5)
    out = tke_vertical_mixing(tke_old=tke_old, dt=600.0, cfg=cfg,
                              n_iterations=1, return_budget=True, **col)
    b = out.budget
    top = lambda a: np.asarray(a)[..., 0]
    assert np.allclose(top(b.pin), top(out.tke_new - tke_old))
    for f in ("production", "external", "buoyancy", "dissipation", "transport"):
        assert np.all(top(getattr(b, f)) == 0.0)


def test_etau_term_is_the_post_solve_injection():
    cfg = _CASES["nemo_z0_split_lc_etau"]
    col = _column()
    tke_old = jnp.full((1, 2, 7), 1e-5)
    out = tke_vertical_mixing(tke_old=tke_old, dt=600.0, cfg=cfg,
                              n_iterations=1, return_budget=True, **col)
    assert float(np.max(np.asarray(out.budget.etau))) > 0.0
    assert float(np.max(np.asarray(out.budget.external))) > 0.0   # Langmuir
    no_etau = tke_vertical_mixing(
        tke_old=tke_old, dt=600.0, cfg=cfg._replace(etau_mode="none"),
        n_iterations=1, return_budget=True, **col)
    assert np.all(np.asarray(no_etau.budget.etau) == 0.0)


def test_budget_jit_parity():
    cfg = _CASES["nemo_z0_split_lc_etau"]
    col = _column()
    tke_old = jnp.full((1, 2, 7), 1e-5)

    def run(tke_old):
        return tke_vertical_mixing(tke_old=tke_old, dt=600.0, cfg=cfg,
                                   n_iterations=1, return_budget=True, **col)

    eager = run(tke_old).budget
    jitted = jax.jit(run)(tke_old).budget
    for f in eager._fields:
        assert np.allclose(np.asarray(getattr(eager, f)),
                           np.asarray(getattr(jitted, f)), rtol=1e-12, atol=1e-18)


def test_literal_matrix_refuses_budget():
    shape = (1, 1, 3)
    cfg = TKEConfig(tke_matrix_evaluation="nemo_literal",
                    dissipation_discretization="nemo_1p5_split",
                    surface_bc="nemo_dirichlet", tke_surface_bc_level="nemo_z0")
    zeros = jnp.zeros(shape)
    with pytest.raises(ValueError, match="return_budget"):
        _solve_tke_backward_euler(
            e_old=jnp.full(shape, 1e-6), K_M_old=zeros, K_H_old=zeros,
            P_s=zeros, N2=zeros, l_eps=jnp.ones(shape), dz_half=jnp.ones(shape),
            surface_flux=jnp.zeros((1, 1)), dt=1.0, cfg=cfg,
            return_budget=True)


@pytest.mark.parametrize("name", ["neumann_flux", "nemo_z0_split_lc_etau"])
def test_terms_are_assigned_not_just_closing(name):
    """Analytic expectation per term, so a compensating mis-assignment (e.g.
    part of the implicit dissipation landing in transport with the sum still
    closing) is caught: a uniform TKE column, no shear, neutral N^2, zero
    stress and no Langmuir/etau can only DISSIPATE. The solve is implicit, so
    a depth-varying dissipation length makes e_new slightly non-uniform and
    transport then acts on that: at dt = 1e-3 s that transport is O(K dt/dz^2)
    ~ 1e-7 of the change, while a mis-assigned dissipation would be O(1) of it.
    Transport must therefore be below 1e-4 of the change on every interior row
    and dissipation must carry the change to the same tolerance."""
    cfg = _CASES[name]._replace(lc=False, etau_mode="none") \
        if name == "nemo_z0_split_lc_etau" else _CASES[name]
    col = _column()
    nlev = 8
    shape = (1, 2, nlev)
    col.update(
        u_cell=jnp.zeros(shape), v_cell=jnp.zeros(shape),
        T_cell=jnp.full(shape, 15.0), S_cell=jnp.full(shape, 35.0),
        rho_cell=jnp.full(shape, 1025.0),
        tau_x_surface=jnp.zeros((1, 2)), tau_y_surface=jnp.zeros((1, 2)),
        taum_surface=jnp.zeros((1, 2)))
    tke_old = jnp.full((1, 2, nlev - 1), 1e-4)
    out = tke_vertical_mixing(tke_old=tke_old, dt=1e-3, cfg=cfg,
                              n_iterations=1, return_budget=True, **col)
    b = out.budget
    change = np.asarray(out.tke_new - tke_old)
    scale = float(np.max(np.abs(change)))
    assert scale > 0.0                       # it does dissipate
    tol = 1e-4 * scale
    unp = ~np.asarray(b.pin != 0.0)
    # interior rows that see no pinned neighbour: k=1..nlev-3 always qualify
    inner = unp.copy(); inner[..., 0] = False; inner[..., -1] = False
    assert inner.any()
    assert np.max(np.abs(np.asarray(b.transport)[inner])) <= tol
    assert np.max(np.abs(np.asarray(b.production)[inner])) == 0.0
    assert np.max(np.abs(np.asarray(b.external)[inner])) == 0.0
    assert np.max(np.abs(np.asarray(b.buoyancy)[inner])) <= tol
    assert np.allclose(np.asarray(b.dissipation)[inner] + np.asarray(b.floor)[inner],
                       change[inner], rtol=0.0, atol=tol)
