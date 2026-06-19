"""Tight finite-difference-vs-autodiff gradient correctness for the lat-lon
C-grid ocean step, on a deliberately SMOOTH config.

Why this exists alongside ``test_step_gradient_matrix.py``: that matrix gates
gradient *health* (finite, flowing) and the *transpose* consistency
``reverse-mode == forward-mode`` across many kinky variants (limiters, KPP,
convection). Reverse==forward is necessary but NOT sufficient — a
self-consistent but wrong adjoint (e.g. a custom_vjp solving the wrong
transposed system symmetrically) passes rev==fwd yet gives the wrong
derivative. Only a finite-difference cross-check pins the gradient to its true
value, and the matrix's FD gate is intentionally loose (5%, one T-direction)
because limiter/where kinks defeat tight FD.

Here we remove the kinks — linear-in-the-perturbation advection (``centered``,
unlimited), constant background mixing (no KPP soft-argmax), stable
stratification (no convective ``where``), differentiable barotropic substepping
— so the loss is genuinely smooth and central-difference FD matches AD to
~1e-6 in fp64. We then assert that tight agreement for:

  1. each prognostic INPUT field (T, S, u, v, eta) independently — the u/v/eta
     directions exercise the barotropic-solve / Coriolis / pressure-gradient
     adjoints, which the matrix only covers via rev==fwd; and
  2. the GM/Redi eddy-closure PARAMETERS (kappa_GM, kappa_Redi) — the
     calibration gradients training actually optimizes, which no existing
     ocean test cross-checks against FD at all.

Everything runs under an explicitly-forced fp64 compute policy (the models
default to float32 regardless of JAX_ENABLE_X64; at float32 these gates would
measure precision floors, not correctness — cf. test_step_gradient_matrix).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

N_LAT, N_LON, NLEV = 8, 12, 4
H_MAX = 1000.0
DT = 600.0
N_STEPS = 2

# Tolerances (fp64). Observed worst-case during bring-up: per-field ~2.6e-6,
# parameters ~1.3e-4 (kappa_Redi). These gates carry ~40x (field) / ~8x
# (param) margin over that noise floor, yet a deliberately-injected 1%
# gradient error trips them at rel~1e-2 (≥80x / ≥10x over tol) and a sign flip
# at rel~2 — so they stay green across CI hardware while still catching a wrong
# sign/transpose. (Empirically confirmed by error-injection during review.)
FIELD_TOL = 1e-4
PARAM_TOL = 1e-3

# Random directions: a fixed per-field seed (NOT hash(), which is salted per
# process) so the test is bit-reproducible.
_FIELD_SEED = {"T": 1, "S": 2, "u": 3, "v": 4, "eta": 5}


@pytest.fixture(scope="module", autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(prev)


def _smooth_config(kappa_gm=500.0, kappa_redi=500.0):
    """A C-infinity (kink-free) lat-lon C-grid config: unlimited centered
    tracer advection, constant background viscosity/diffusivity, GM/Redi with
    no EKE closure, differentiable barotropic substepping."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig

    return LatLonCGridOceanConfig(
        A_h=1000.0, K_h=100.0, A_v=1e-3, K_v=1e-4, bottom_drag_r=1e-3,
        tracer_advection="centered", n_barotropic_substeps=4,
        differentiable_barotropic=True, enable_runtime_checks=False,
        gm_redi=GMRediConfig(kappa_GM=kappa_gm, kappa_Redi=kappa_redi),
    )


def _grid_state():
    """Build the grid + a stably-stratified rest state (warm surface decaying
    with depth → N^2 > 0 everywhere, so no convective adjustment fires)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(N_LAT, N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=H_MAX, land_lat_threshold=85.0)
    lat = jnp.linspace(-1.0, 1.0, N_LAT)[:, None, None]
    lon = jnp.linspace(-1.0, 1.0, N_LON)[None, :, None]
    lev = jnp.linspace(0.0, 1.0, NLEV)[None, None, :]
    # Warm, monotonically-decreasing-with-depth perturbation: stable column.
    T = state.T.data + 1.5 * jnp.exp(
        -((lat / 0.5) ** 2 + (lon / 0.5) ** 2) - 3.0 * lev)
    state = state._replace(T=state.T.replace(data=T))
    return grid, z_coord, state


def _masked_unit_dir(state, field):
    """A random unit perturbation for ``field``, masked to the wet domain (so
    we never differentiate a land cell that the step leaves untouched)."""
    base = getattr(state, field).data
    d = jax.random.normal(jax.random.PRNGKey(_FIELD_SEED[field]), base.shape)
    if field == "u":
        d = d * state.u_mask.data[..., None]
    elif field == "v":
        d = d * state.v_mask.data[..., None]
    elif field == "eta":
        d = d * state.land_mask.data
    else:  # T, S — cell-centred tracers
        d = d * state.land_mask.data[..., None]
    return d / jnp.linalg.norm(d)


def _rollout_loss(model, state):
    s = state
    for _ in range(N_STEPS):
        s = model._step_impl(s, DT, surface_forcing=None)
    wet3 = state.land_mask.data[..., None] * jnp.ones((1, 1, NLEV))
    return (jnp.sum((s.T.data * wet3) ** 2)
            + jnp.sum((s.S.data * wet3) ** 2)
            + 1e4 * (jnp.sum(s.u.data ** 2) + jnp.sum(s.v.data ** 2))
            + 1e4 * jnp.sum(s.eta.data ** 2))


@pytest.mark.parametrize("field", ["T", "S", "u", "v", "eta"])
def test_initial_state_grad_matches_fd(field):
    """Directional gradient of a 2-step rollout w.r.t. each prognostic input
    field matches a central-difference FD to tight (fp64) tolerance."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid, z_coord, state = _grid_state()
    model = LatLonCGridOceanModel(grid, z_coord, _smooth_config())
    d = _masked_unit_dir(state, field)
    fld = getattr(state, field)

    def loss(eps):
        s = state._replace(**{field: fld.replace(data=fld.data + eps * d)})
        return _rollout_loss(model, s)

    loss_j = jax.jit(loss)
    ad = float(jax.jit(jax.grad(loss))(0.0))
    h = 1e-4
    fd = (float(loss_j(h)) - float(loss_j(-h))) / (2.0 * h)
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
    assert rel < FIELD_TOL, (
        f"{field}: AD-vs-FD rel={rel:.2e} (ad={ad:.8e}, fd={fd:.8e}) "
        f"— smooth-config gradient is wrong, not merely rev==fwd-consistent")


@pytest.mark.parametrize("param", ["kappa_GM", "kappa_Redi"])
def test_gm_redi_param_grad_matches_fd(param):
    """Gradient of a 2-step rollout w.r.t. a GM/Redi eddy-closure parameter —
    the calibration gradient training optimizes — matches central-difference
    FD. The model is rebuilt inside the differentiated function so the traced
    parameter threads through construction (validate() does not numerically
    compare the kappa fields). NOT jitted: model construction calls
    ``ensure_geometry(grid)`` → ``create_latlon_geometry``, which is host-side
    grid setup that is not trace-safe under ``jax.jit``; reverse-mode AD alone
    handles it eagerly. The box is tiny, so un-jitted is fast enough."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    grid, z_coord, state = _grid_state()
    k0 = 500.0

    def loss(theta):
        cfg = (_smooth_config(kappa_gm=theta)
               if param == "kappa_GM"
               else _smooth_config(kappa_redi=theta))
        model = LatLonCGridOceanModel(grid, z_coord, cfg)
        return _rollout_loss(model, state)

    ad = float(jax.grad(loss)(k0))
    h = 0.02 * k0  # ~10.0: large enough that fp64 roundoff (not truncation,
    # the loss is near-linear in kappa here) dominates the FD error floor.
    fd = (float(loss(k0 + h)) - float(loss(k0 - h))) / (2.0 * h)
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-300)
    assert abs(ad) > 0.0, f"{param}: gradient is exactly zero (path severed?)"
    assert rel < PARAM_TOL, (
        f"{param}: AD-vs-FD rel={rel:.2e} (ad={ad:.8e}, fd={fd:.8e})")
