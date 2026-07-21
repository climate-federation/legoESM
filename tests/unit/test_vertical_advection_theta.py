"""#930: the MPAS thermodynamic vertical transport used to compute the two
large, near-cancelling terms  -σ̇·∂T/∂σ  (a first difference, ×2 at the 2Δz
Nyquist) and  κ·T·σ̇/σ  (a point value, ×1)  SEPARATELY, leaving a spurious
2Δz residual that the 1/σ prefactor amplifies at the stretched top levels.

The cure is the θ-form (``vertical_advection_theta`` for σ, and the new
``vertical_advection_theta_hybrid`` for the hybrid coordinate): advect
potential temperature θ = T·(p₀/p)^κ and convert back, so the two terms cancel
ANALYTICALLY before discretization.

These tests pin the decisive property: on a dry adiabat (θ = const) the θ-form
returns ~0 to machine precision, while the OLD split form does NOT.  Run under
``JAX_ENABLE_X64=1`` (the machine-precision bound is x64-tight; a float32
fallback bound is used otherwise so the *contrast* is still asserted).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    make_hybrid_levels,
    pressure_from_hybrid,
    vertical_advection,
    vertical_advection_hybrid,
    vertical_advection_theta,
    vertical_advection_theta_hybrid,
)

from legoesm import constants

NLEV = 24
P_S = 1.0e5
THETA_CONST_K = 300.0

# x64-tight "machine zero" bound for the θ-form; loosen under float32 so the
# key contrast assertion (split ≫ θ-form) still stands without a false red.
# Under float32 (`_to_f64` is a no-op, since jax forbids f64) the θ round-trip
# floors at ~1e-7, so the contrast is ~5e4 not ~1e14 — use a dtype-aware
# factor so a stray fp32 invocation still asserts the property, not a false red.
_X64 = jnp.zeros(()).dtype == jnp.float64
_THETA_ZERO_TOL = 1e-8 if _X64 else 1e-4
_CONTRAST_FACTOR = 1e6 if _X64 else 1e4


def _to_f64(coord):
    """Cast a coordinate NamedTuple's float arrays to float64.

    ``create_sigma_coordinate`` / ``make_hybrid_levels`` build the vertical
    coordinate in float32 (finite-volume storage default), which caps the
    θ round-trip ``(p/p₀)^κ·(p₀/p)^κ`` at ~1e-7 relative.  The θ-form
    cancellation is a property of the OPERATOR (exact in exact arithmetic),
    so we exercise it in float64 to state the machine-precision claim
    honestly — the split-form contrast below is computed with the SAME
    float64 coordinate for a fair, dtype-controlled comparison.
    """
    def _cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float64)
        return x
    return jax.tree_util.tree_map(_cast, coord)


def _smooth_interface_profile(n_half, amp):
    """A smooth interface field, zero at both boundaries (top & surface)."""
    s = jnp.linspace(0.0, 1.0, n_half)
    return amp * jnp.sin(jnp.pi * s)  # (n_half,)


# ---------------------------------------------------------------------------
# Dry-adiabat cancellation — the #930 decider (σ coordinate)
# ---------------------------------------------------------------------------

def test_theta_form_zero_on_dry_adiabat_sigma():
    """θ=const column: ``vertical_advection_theta`` ~0, split form NOT ~0."""
    coord = _to_f64(create_sigma_coordinate(NLEV))
    p_s = jnp.full((4, 3), P_S)
    sigma_full = coord.sigma_full  # (nlev,)
    p_full = sigma_full * p_s[..., None]

    # Dry adiabat: θ = T·(p₀/p)^κ = const  ⇒  T = θ·(p/p₀)^κ.
    T = THETA_CONST_K * (p_full / constants.p_ref) ** constants.kappa

    # A realistic smooth σ̇ (dσ/dt) at interfaces, zero at top & surface.
    sigma_dot_1d = _smooth_interface_profile(NLEV + 1, amp=5.0e-4)
    sigma_dot = jnp.broadcast_to(sigma_dot_1d, p_s.shape + (NLEV + 1,))

    # θ-form (the fix): cancellation-free.
    theta_tend = vertical_advection_theta(T, sigma_dot, p_s, coord)

    # OLD split form: -σ̇·∂T/∂σ and κ·T·σ̇/σ computed SEPARATELY.
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    adv_T = vertical_advection(T, sigma_dot, coord)
    adiabatic_sigmadot = constants.kappa * T * sigma_dot_full / sigma_full
    split_tend = adv_T + adiabatic_sigmadot

    theta_max = float(jnp.max(jnp.abs(theta_tend)))
    split_max = float(jnp.max(jnp.abs(split_tend)))

    # θ-form is ~0 to (x64) machine precision …
    assert theta_max < _THETA_ZERO_TOL, f"θ-form not ~0: {theta_max:.3e}"
    # … the split form leaves an appreciable spurious residual …
    assert split_max > 1e-4, f"split residual unexpectedly tiny: {split_max:.3e}"
    # … and the θ-form is dramatically smaller (the whole point of #930).
    assert split_max > _CONTRAST_FACTOR * (theta_max + 1e-300)


def test_theta_form_zero_on_dry_adiabat_hybrid():
    """θ=const column: ``vertical_advection_theta_hybrid`` ~0, split NOT ~0."""
    coord = _to_f64(make_hybrid_levels(NLEV, p_top_Pa=200.0))
    p_s = jnp.full((4, 3), P_S)
    p_full = pressure_from_hybrid(coord, p_s, full=True)

    T = THETA_CONST_K * (p_full / constants.p_ref) ** constants.kappa

    # Smooth vertical mass flux F [Pa/s] at interfaces, zero at top & surface.
    mf_1d = _smooth_interface_profile(NLEV + 1, amp=50.0)
    mass_flux = jnp.broadcast_to(mf_1d, p_s.shape + (NLEV + 1,))

    # θ-form (the fix).
    theta_tend = vertical_advection_theta_hybrid(T, mass_flux, p_s, coord)

    # OLD split form: -F·∂T/∂p and κ·T·F/p computed SEPARATELY.
    F_full = 0.5 * (mass_flux[..., :-1] + mass_flux[..., 1:])
    adv_T = vertical_advection_hybrid(T, mass_flux, p_s, coord)
    adiabatic_F = constants.kappa * T * F_full / p_full
    split_tend = adv_T + adiabatic_F

    theta_max = float(jnp.max(jnp.abs(theta_tend)))
    split_max = float(jnp.max(jnp.abs(split_tend)))

    assert theta_max < _THETA_ZERO_TOL, f"θ-form not ~0: {theta_max:.3e}"
    assert split_max > 1e-4, f"split residual unexpectedly tiny: {split_max:.3e}"
    assert split_max > _CONTRAST_FACTOR * (theta_max + 1e-300)


# ---------------------------------------------------------------------------
# Direct unit tests for the new helper
# ---------------------------------------------------------------------------

def test_theta_hybrid_shape_and_finite():
    """Shape (..., nlev) and finiteness on a non-trivial column."""
    coord = make_hybrid_levels(NLEV, p_top_Pa=200.0)
    p_s = jnp.full((6, 4, 4), P_S)
    key = jax.random.PRNGKey(0)
    T = 250.0 + 30.0 * jax.random.uniform(key, (6, 4, 4, NLEV))
    mass_flux = jnp.zeros((6, 4, 4, NLEV + 1)).at[..., 5].set(50.0)

    tend = vertical_advection_theta_hybrid(T, mass_flux, p_s, coord)
    assert tend.shape == (6, 4, 4, NLEV)
    assert bool(jnp.all(jnp.isfinite(tend)))


def test_theta_hybrid_zero_mass_flux_zero_tendency():
    """Zero mass flux ⇒ zero tendency for any temperature field."""
    coord = make_hybrid_levels(NLEV, p_top_Pa=200.0)
    p_s = jnp.full((2, 3), P_S)
    key = jax.random.PRNGKey(1)
    T = 200.0 + 80.0 * jax.random.uniform(key, (2, 3, NLEV))
    mass_flux = jnp.zeros((2, 3, NLEV + 1))

    tend = vertical_advection_theta_hybrid(T, mass_flux, p_s, coord)
    np.testing.assert_allclose(np.asarray(tend), 0.0, atol=1e-12)


def test_theta_hybrid_uniform_theta_column_is_zero():
    """A single-column dry adiabat (θ=const) returns ~0 (helper in isolation)."""
    coord = _to_f64(make_hybrid_levels(NLEV, p_top_Pa=200.0))
    p_s = jnp.array([P_S])
    p_full = pressure_from_hybrid(coord, p_s, full=True)
    T = THETA_CONST_K * (p_full / constants.p_ref) ** constants.kappa
    mass_flux = jnp.broadcast_to(
        _smooth_interface_profile(NLEV + 1, amp=40.0), (1, NLEV + 1))

    tend = vertical_advection_theta_hybrid(T, mass_flux, p_s, coord)
    assert float(jnp.max(jnp.abs(tend))) < _THETA_ZERO_TOL


def test_theta_hybrid_differentiable():
    """jit+grad through the helper w.r.t. T is finite (AD/JIT safe)."""
    coord = make_hybrid_levels(NLEV, p_top_Pa=200.0)
    p_s = jnp.full((2, 2), P_S)
    mass_flux = jnp.broadcast_to(
        _smooth_interface_profile(NLEV + 1, amp=30.0), (2, 2, NLEV + 1))

    def _scalar(T):
        return jnp.sum(vertical_advection_theta_hybrid(T, mass_flux, p_s, coord) ** 2)

    T = jnp.full((2, 2, NLEV), 260.0)
    g = jax.jit(jax.grad(_scalar))(T)
    assert g.shape == T.shape
    assert bool(jnp.all(jnp.isfinite(g)))
