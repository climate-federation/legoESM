"""The CANONICAL differentiable ocean model reproduces the MITgcm laminar gyre.

Issue #519 item 1: the bespoke, scipy-sparse (non-differentiable)
``GyreFaithfulModel`` was deleted.  The MITgcm ``tutorial_barotropic_gyre``
equilibrium oracle now runs through the shipped, end-to-end ``jax.grad``-able
``LatLonCGridOceanModel`` on the gyre recipe (``barotropic_solver="implicit_cn"``).

These tests assert the SAME three physical properties the bespoke stepper was
kept for, now measured on the canonical model:

1. spin-up forms a western-intensified free-surface dipole at the gyre scale,
2. the marginally-resolved (Munk δ≈1.7-cell) gyre stays LAMINAR (well below the
   turbulent 0.15-0.37 band) and equilibrates near MITgcm's |u|max≈0.031 /
   |v|max≈0.084,
3. the equilibrium is steady / not growing,

PLUS a ``jax.grad`` smoke check through the gyre step — the differentiability the
scipy Helmholtz solver lacked, the entire point of the deletion.

Canonical-vs-MITgcm equilibrium gap (documented, physical): at 15000 steps
(~0.57 yr) the canonical implicit_cn path plateaus at |u|max≈0.027 / |v|max≈0.079,
i.e. ~12% / ~6% below MITgcm's 0.031 / 0.084.  The residual is the canonical
barotropic/baroclinic operator-split's slightly higher effective dissipation on
a Munk layer resolved by only ~1.7 cells (δ_Munk=(A_h/β)^{1/3}≈33 km vs dx=20 km);
it is the SAME ~13% gap the recipe docstring records, far inside the laminar band.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.fidelity import mitgcm_barotropic_gyre_recipe as gyre

os.environ.setdefault("JAX_ENABLE_X64", "1")


def _build_model():
    """Canonical model + rest state on the MITgcm barotropic-gyre recipe."""
    r = gyre.build_gyre_recipe()  # implicit_cn split, MITgcm-faithful numerics
    model = LatLonCGridOceanModel(r.geometry, r.z_coord, r.config)
    state = r.state
    model._ensure_vertex_mask(state)
    return model, state, r


def _chunk_fn(model, recipe):
    """JIT-compiled 1000-step scan of the canonical step (laminar spin-up)."""

    @jax.jit
    def chunk(state):
        def body(st, _):
            st = model._step_impl(
                st, recipe.dt_s, surface_forcing=recipe.wind_forcing
            )
            return st, None

        st, _ = jax.lax.scan(body, state, None, length=1000)
        return st

    return chunk


def test_spins_up_a_dipole_at_the_right_scale():
    """10 steps from rest: a wind-curl free-surface dipole at the gyre scale.

    Velocities are O(1e-4 m/s) after 10 steps (matching MITgcm's 10-step
    spin-up and the deleted bespoke stepper's own 10-step magnitude — the spin-up
    transient, not the multi-year equilibrium 0.031)."""
    model, state, r = _build_model()
    s = state
    for _ in range(10):
        s = model.step(s, r.dt_s, surface_forcing=r.wind_forcing)
    eta = np.asarray(s.eta.data)
    u = np.asarray(s.u.data)
    v = np.asarray(s.v.data)
    assert np.all(np.isfinite(eta)) and np.all(np.isfinite(u))
    # Wind-curl forcing tilts the free surface into a dipole (min < 0 < max).
    assert eta.min() < 0.0 < eta.max()
    # Same order as MITgcm after 10 steps (u,v ~ 1e-4 m/s spin-up), not a blow-up.
    assert 1e-5 < np.abs(u).max() < 1e-2, f"|u|max={np.abs(u).max():.2e}"
    assert 1e-5 < np.abs(v).max() < 1e-2, f"|v|max={np.abs(v).max():.2e}"


def test_grad_flows_through_the_gyre_step():
    """jax.grad smoke check: differentiate a free-surface energy after 3 gyre
    steps w.r.t. the wind-stress amplitude.

    The whole point of deleting the bespoke stepper: its scipy-sparse Helmholtz
    solve broke end-to-end ``jax.grad``.  The canonical implicit free surface is
    pure JAX, so a finite, non-zero gradient flows through the step."""
    model, state, r = _build_model()
    base_tau_x = r.wind_forcing.tau_x

    def loss(amp):
        frc = r.wind_forcing._replace(tau_x=base_tau_x * amp)
        st = state
        for _ in range(3):
            st = model._step_impl(st, r.dt_s, surface_forcing=frc)
        return jnp.sum(st.eta.data**2)

    val, grad = jax.value_and_grad(loss)(1.0)
    assert np.isfinite(float(val)) and val > 0.0
    assert np.isfinite(float(grad))
    # eta energy grows with wind amplitude near amp=1, so d(eta^2)/d(amp) > 0.
    assert float(grad) > 0.0, f"grad={float(grad):.3e} not positive"


@pytest.mark.slow
def test_reproduces_mitgcm_laminar_equilibrium():
    """15000 steps (~0.57 yr): the canonical model stays LAMINAR and equilibrates
    near MITgcm's |u|max≈0.031 / |v|max≈0.084, NOT the turbulent 0.15-0.37 band.

    Thresholds are widened on the low side vs the bespoke stepper's
    [0.025, 0.040] / [0.065, 0.105] to admit the documented ~12% / ~6% gap of the
    canonical operator-split path (see module docstring): |u|max∈[0.022, 0.040],
    |v|max∈[0.060, 0.105].  Both upper bounds stay far below the turbulent
    attractor, so the test still FAILS on a turbulent overshoot."""
    model, state, r = _build_model()
    chunk = _chunk_fn(model, r)
    s = state
    for _ in range(15):  # 15000 steps
        s = chunk(s)
    u = np.asarray(s.u.data)
    v = np.asarray(s.v.data)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
    umax = float(np.abs(u).max())
    vmax = float(np.abs(v).max())
    # Near MITgcm's laminar 0.031 / 0.084 (canonical split runs ~12% / ~6% low),
    # and well below the turbulent band.
    assert 0.022 < umax < 0.040, f"|u|max={umax:.4f} not at MITgcm's laminar 0.031"
    assert 0.060 < vmax < 0.105, f"|v|max={vmax:.4f} not at MITgcm's laminar 0.084"


@pytest.mark.slow
def test_steady_not_growing():
    """Laminar = bounded/steady: |u|max barely changes between ~0.5 and ~0.95 yr
    (a turbulent attractor would still be growing/oscillating an order larger)."""
    model, state, r = _build_model()
    chunk = _chunk_fn(model, r)
    s = state
    for _ in range(13):  # 13000 steps (~0.49 yr)
        s = chunk(s)
    u_mid = float(np.abs(np.asarray(s.u.data)).max())
    for _ in range(12):  # +12000 steps (~0.95 yr total)
        s = chunk(s)
    u_late = float(np.abs(np.asarray(s.u.data)).max())
    assert abs(u_late - u_mid) < 0.01, f"not steady: {u_mid:.4f} -> {u_late:.4f}"
    assert u_late < 0.05, f"|u|max={u_late:.4f} drifted out of the laminar band"
