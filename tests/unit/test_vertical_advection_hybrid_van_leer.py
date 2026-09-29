"""Conservative limited tracer vertical transport on the hybrid (CAM L32) lane.

The advective upwind operator the MPAS hybrid lane used for every tracer
creates water: measured +0.32..+0.36 kg/m2/day globally on the production
CAM6 run (days 30/45/55), the size of that run's own P - E + dW/dt budget gap.
These tests pin the replacement's defining properties and that the MPAS
tracer path runs it.
"""
from __future__ import annotations

import importlib.util
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.grids.vertical import (
    compute_mass_flux_from_cumsum, dp_from_hybrid, make_cam6_l32_levels,
    vertical_advection_hybrid, vertical_advection_hybrid_van_leer)


def _column(seed=0, ncol=6):
    """CAM L32 columns with varying p_s and a continuity-built mass flux."""
    rng = np.random.default_rng(seed)
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.asarray(1.0e5 + 3.0e3 * rng.standard_normal(ncol))
    div = jnp.asarray(1.0e-5 * rng.standard_normal((ncol, nlev)))
    cums = jnp.cumsum(div * dp_from_hybrid(coord, p_s), axis=-1)
    F = compute_mass_flux_from_cumsum(cums, cums[..., -1:], coord)
    q = jnp.asarray(1.0e-2 * np.exp(-0.5 * ((np.arange(nlev) - 24) / 3.0) ** 2)
                    * (1.0 + 0.3 * rng.random((ncol, nlev))))
    return coord, p_s, F, q


def _product_rule_residual(op, coord, p_s, F, q):
    dp = dp_from_hybrid(coord, p_s)
    col = jnp.sum(dp * op(q, F, p_s, coord) - q * (F[..., 1:] - F[..., :-1]), axis=-1)
    scale = jnp.sum(jnp.abs(F[..., 1:-1] * jnp.diff(q, axis=-1)), axis=-1)
    return float(jnp.max(jnp.abs(col) / scale))


def test_column_water_telescopes_and_the_upwind_operator_does_not():
    coord, p_s, F, q = _column()
    assert _product_rule_residual(vertical_advection_hybrid_van_leer, coord, p_s, F, q) < 1e-12
    assert _product_rule_residual(vertical_advection_hybrid, coord, p_s, F, q) > 1e-3


def test_constant_field_has_zero_tendency():
    coord, p_s, F, _ = _column(1)
    c = jnp.full((p_s.shape[0], coord.A_full.shape[0]), 7.3e-3)
    assert float(jnp.max(jnp.abs(vertical_advection_hybrid_van_leer(c, F, p_s, coord)))) < 1e-18


def test_positive_and_bounded_under_repeated_steps():
    """A non-negative layer with sharp edges stays in [0, max] over many
    forward-Euler steps at a per-face Courant sum of ~0.3 (production max 0.14)."""
    coord, p_s, F, _ = _column(2)
    nlev = coord.A_full.shape[0]
    dp = dp_from_hybrid(coord, p_s)
    q = jnp.zeros((p_s.shape[0], nlev)).at[:, 18:22].set(1.0e-3)
    rate = (jnp.abs(F[..., :-1]) + jnp.abs(F[..., 1:])) / dp
    dt = 0.3 / float(jnp.max(rate))
    for _ in range(200):
        q = q + dt * vertical_advection_hybrid_van_leer(q, F, p_s, coord)
    assert float(jnp.min(q)) >= -1e-18
    assert float(jnp.max(q)) <= 1.0e-3 * (1 + 1e-12)


def test_mpas_hybrid_tracer_path_runs_the_conservative_operator(monkeypatch):
    """Names the symbol the MPAS hybrid tracer path calls, and fails if the
    lane goes back to the upwind operator."""
    path = pathlib.Path(__file__).with_name("test_mpas_atmosphere.py")
    spec = importlib.util.spec_from_file_location("_mpas_atm_helpers_vl", path)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    from legoesm.core.field import Field

    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    mesh = h._make_mesh(level=2)
    state = h._add_perturbation_hydro(h._make_hydrostatic_state(mesh, nlev), mesh, nlev)
    qv = jnp.asarray(np.linspace(1e-6, 1.5e-2, nlev)[None, :].repeat(mesh.nCells, 0))
    state = state._replace(tracers={"q_v": Field(
        data=qv, name="q_v", dims=("nCells", "nlev"), units="kg/kg")})
    calls = []
    real = pe.vertical_advection_hybrid_van_leer

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(pe, "vertical_advection_hybrid_van_leer", spy)
    out = pe.mpas_hydrostatic_tendencies(
        state, mesh, coord, pe.MPASPrimitiveEquationConfig(nu_del2=1.0e5))
    assert calls, "the MPAS hybrid tracer path no longer calls the conservative operator"
    assert np.all(np.isfinite(np.asarray(out.tracer_tendencies["q_v"].data)))


def test_boundary_donor_faces_keep_a_full_column_in_range():
    """Load-bearing: without the donor faces next to the top and bottom layers
    a non-constant column under steady descent leaves its starting range
    (independent review measured 1.13..1.99 -> 0.44 after 2000 steps)."""
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.array([1.0e5])
    dp = dp_from_hybrid(coord, p_s)
    q0 = jnp.asarray(1.13 + 0.86 * np.random.default_rng(3).random((1, nlev)))
    for sign in (1.0, -1.0):
        F = jnp.concatenate([jnp.zeros((1, 1)), jnp.full((1, nlev - 1), sign),
                             jnp.zeros((1, 1))], axis=-1)
        dt = 0.4 / float(jnp.max((jnp.abs(F[..., :-1]) + jnp.abs(F[..., 1:])) / dp))
        q = q0
        for _ in range(2000):
            q = q + dt * vertical_advection_hybrid_van_leer(q, F, p_s, coord)
        assert float(jnp.min(q)) >= float(jnp.min(q0)) - 1e-12
        assert float(jnp.max(q)) <= float(jnp.max(q0)) + 1e-12
