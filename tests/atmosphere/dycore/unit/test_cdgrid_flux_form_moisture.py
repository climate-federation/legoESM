"""End-to-end gate: flux-form moisture transport in the live cdgrid PE step (#771).

The advective -(u·∇q) form (default) does not conserve column water under a
divergent wind and needs a one-signed max(q, 0) clip.  With
``moisture_flux_form=True`` the horizontal transport moves to a mass-conserving,
monotone post-RK3 substep (flux_form_tracer_step + co-transported δp + a
per-tracer mass fixer).  This test drives a real cubed-sphere PE model under a
strong divergent wind and asserts the flux-form path conserves column water
far better AND stays positive, while the advective path drifts and clips.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from types import SimpleNamespace

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.field import Field
from legoesm.core.state import FV3HydrostaticState
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    CDGridPrimitiveEquationConfig,
)


def test_flux_form_scatter_guard_predicate():
    """The fail-closed predicate: MPI (single-rank / replicated / face-scatter) is
    now ALLOWED — the reductions are allreduce-aware AND the transport + wind
    reconstruction are 4D-halo (transport_step_4d / d2a2c_vect_4d), so there is no
    vmap(pad_halo).  Only SPMD face-sharding still fails closed (#811)."""
    f = CDGridPrimitiveEquationModel._flux_form_scatter_blocked
    _topo3 = SimpleNamespace(local_face_ids=(0, 1, 2))   # rank owns 3 of 6 faces
    _topo6 = SimpleNamespace(local_face_ids=(0, 1, 2, 3, 4, 5))
    _band = SimpleNamespace(band_id=0)                   # lat-lon: no faces attr
    # Genuinely scattered MPI: now ALLOWED (4D-halo transport + winds, #811).
    assert f("mpi", _topo3, None, 3) is False
    # Replicated cube MPI: <6 owned faces but FULL 6-face state (lead==6).
    assert f("mpi", _topo3, None, 6) is False
    # Single-rank MPI: owns all 6 faces.
    assert f("mpi", _topo6, None, 6) is False
    # Serial / local backend.
    assert f("local", None, None, 6) is False
    # Lat-lon band MPI (no local_face_ids) must not false-positive.
    assert f("mpi", _band, None, 96) is False
    # SPMD with an active mesh -> fail-closed; without a mesh -> allowed.
    assert f("spmd", None, object(), 6) is True
    assert f("spmd", None, None, 6) is False


def _model(n, nlev, flux_form):
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    cfg = CDGridPrimitiveEquationConfig(moisture_flux_form=flux_form)
    return CDGridPrimitiveEquationModel(grid, sigma, cfg), grid, sigma


def _divergent_state(n, nlev):
    # Strong, spatially varying (divergent) D-grid corner winds.
    jj = jnp.arange(n + 1) / n
    uu = (30.0 * jnp.sin(2.0 * jnp.pi * jj))[None, None, :, None] \
        * jnp.ones((6, n + 1, n + 1, nlev))
    vv = (25.0 * jnp.cos(2.0 * jnp.pi * jj))[None, :, None, None] \
        * jnp.ones((6, n + 1, n + 1, nlev))
    # A sharp positive moisture blob on face 0 (fronts stress the limiter).
    q = jnp.full((6, n, n, nlev), 1e-4).at[
        0, n // 3:2 * n // 3, n // 3:2 * n // 3, :].set(2e-2)
    return FV3HydrostaticState(
        u_d=Field(uu, name="u_d"), v_d=Field(vv, name="v_d"),
        T=Field(jnp.full((6, n, n, nlev), 280.0), name="T"),
        p_s=Field(jnp.full((6, n, n), 1.0e5), name="p_s"),
        phis=Field(jnp.zeros((6, n, n)), name="phis"),
        tracers={"q_v": Field(q, name="q_v")})


def _column_water(model, sigma, state):
    dp = sigma.layer_thickness_dp(state.p_s.data)
    area = model.cdgrid.base.area
    return float(jnp.sum(area[..., None] * dp * state.tracers["q_v"].data,
                         dtype=jnp.float64))


def _run(flux_form, n=24, nlev=8, nsteps=10, dt=120.0):
    model, grid, sigma = _model(n, nlev, flux_form)
    state = _divergent_state(n, nlev)
    w0 = _column_water(model, sigma, state)
    min_q = 1.0e9
    max_drift = 0.0
    for _ in range(nsteps):
        state = model.step(state, dt)
        min_q = min(min_q, float(jnp.min(state.tracers["q_v"].data)))
        max_drift = max(
            max_drift, abs(_column_water(model, sigma, state) - w0) / w0)
    final_drift = abs(_column_water(model, sigma, state) - w0) / w0
    return final_drift, min_q, max_drift


def test_flux_form_conserves_and_stays_positive():
    drift_adv, minq_adv, _ = _run(flux_form=False)
    drift_ff, minq_ff, _ = _run(flux_form=True)

    # Advective drifts substantially (the #771 non-conservation) and clips to 0.
    assert drift_adv > 1e-4, f"advective drift unexpectedly small: {drift_adv:.2e}"
    assert minq_adv <= 0.0 + 1e-20, "advective should hit the q>=0 clip"

    # Flux-form conserves column water FAR better and never goes negative.
    assert drift_ff < drift_adv / 100.0, \
        f"flux-form drift {drift_ff:.2e} not << advective {drift_adv:.2e}"
    assert drift_ff < 1e-4, f"flux-form drift too large: {drift_ff:.2e}"
    assert minq_ff > -1e-12, \
        f"flux-form produced spurious negatives: {minq_ff:.2e}"


@pytest.mark.slow
def test_flux_form_conservation_is_bounded_not_accumulating():
    """The #771 acceptance property, over a longer run: the advective form
    ACCUMULATES column-water error (final ≈ max — the monotone day-150 drift),
    while flux-form stays BOUNDED (max drift small, no runaway) and positive.
    A single-step or short run cannot distinguish transient from accumulating
    drift; this gate can.  (Serial-only; the scattered/SPMD path is fail-closed
    until the flux-form reductions are allreduce-aware — #771 follow-up.)"""
    final_adv, _minq_adv, max_adv = _run(flux_form=False, nsteps=40)
    final_ff, minq_ff, max_ff = _run(flux_form=True, nsteps=40)
    # Advective ACCUMULATES: final drift is large and ≈ its max (monotone growth,
    # not a transient), and it clips.
    assert final_adv > 1e-2, f"advective should accumulate: {final_adv:.2e}"
    assert final_adv > 0.5 * max_adv
    # Flux-form BOUNDED (no runaway), >= ~20x better, and never negative.
    assert max_ff < 1e-3, f"flux-form max drift not bounded: {max_ff:.2e}"
    assert max_ff < max_adv / 20.0, \
        f"flux-form max {max_ff:.2e} not << advective max {max_adv:.2e}"
    assert minq_ff > -1e-12, f"flux-form went negative: {minq_ff:.2e}"


def test_flux_form_off_is_the_advective_path():
    """Default (moisture_flux_form=False) must keep the advective behaviour —
    the substep must not run when off (guards the OFF-is-unchanged contract)."""
    model_off, grid, sigma = _model(12, 6, flux_form=False)
    state = _divergent_state(12, 6)
    # One step must not raise and must NOT invoke the flux-form substep path;
    # the advective path is not positive-definite, so a clip is expected — we
    # only assert the run completes and produces finite moisture.
    out = model_off.step(state, 120.0)
    q = out.tracers["q_v"].data
    assert bool(jnp.all(jnp.isfinite(q)))
    assert q.shape == state.tracers["q_v"].data.shape


def test_dry_state_unaffected_by_flag():
    """No tracers ⟹ the substep is skipped entirely (guard against touching a
    dry run)."""
    grid = create_cubed_sphere(12)
    sigma = create_sigma_coordinate(6)
    model = CDGridPrimitiveEquationModel(
        grid, sigma, CDGridPrimitiveEquationConfig(moisture_flux_form=True))
    n, nlev = 12, 6
    dry = FV3HydrostaticState(
        u_d=Field(0.3 * jnp.ones((6, n + 1, n + 1, nlev)), name="u_d"),
        v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
        T=Field(jnp.full((6, n, n, nlev), 280.0), name="T"),
        p_s=Field(jnp.full((6, n, n), 1.0e5), name="p_s"),
        phis=Field(jnp.zeros((6, n, n)), name="phis"),
        tracers=None)
    out = model.step(dry, 120.0)          # must not raise
    assert out.tracers is None
