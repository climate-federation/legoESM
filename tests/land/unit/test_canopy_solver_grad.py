"""Implicit-differentiation consistency for the canopy Newton closure.

The solver wraps a Newton fixed point in a ``custom_vjp`` that returns the
implicit-function-theorem adjoint.  These tests verify it is *consistent*:

- ``jax.grad`` through ``solve_canopy_closure`` matches central finite
  differences at a well-converged operating point — for both an atmospheric
  forcing (La) and a trainable physics parameter (Vcmax25).  This is the gold
  standard, and it catches the two bugs fixed alongside the DifferBESS sync:
  the biased ``lstsq(rcond=1e-4)`` adjoint (now an exact solve) and the
  all-or-nothing NaN mask.
- Per-field masking: the Monin-Obukhov scan is not differentiable w.r.t. its
  aerodynamic forcing (e.g. Ta -> NaN cotangent), but that no longer poisons
  the finite gradients of the other inputs (La, Vcmax25, ...).
- The cotangent w.r.t. the initial guess ``x0`` is zero (a fixed point does
  not depend on where the iteration started).
- A non-converged solve (max_iters = 1) returns a zero gradient.

Run under JAX_ENABLE_X64=1 for a clean FD comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure

_a = jnp.asarray


def _bundle(La=_a(350.0), Vc3=_a(40.0), Ta=_a(298.0)):
    """A physically reasonable single-column midday forcing bundle (scalars)."""
    return CanopyForcingBundle(
        LAI=_a(3.0), SZA=_a(30.0), La=La, epsf=_a(0.97), epss=_a(0.96), fSun=_a(0.6),
        APAR_Sun=_a(800.0), APAR_Sh=_a(200.0),
        Vcmax25_Sun=Vc3, Vcmax25_Sh=_a(20.0),
        Vcmax25_C4Sun=_a(0.0), Vcmax25_C4Sh=_a(0.0),
        ASW_Sun=_a(400.0), ASW_Sh=_a(100.0), ASW_Soil=_a(50.0),
        Ts_bc=_a(300.0),
        Ca=_a(400.0), Ps=_a(101325.0), Ta=Ta,
        lam=_a(2.5e6), Cp=_a(1005.0), rhoa=_a(1.2), Tv_atm=_a(298.5), q_atm=_a(0.012),
        m=_a(9.0), b0=_a(0.01), alf=_a(0.3), TgC=_a(20.0), fC4=_a(0.0),
        fStress_soil=_a(0.8),
        ur=_a(3.0), CI=_a(0.75), z0m=_a(0.5), displa=_a(3.0), z0=_a(10.0),
        cv=_a(0.0135), d_leaf=_a(0.025),
        r_soil_surface=_a(0.0),
    )


_X0 = jnp.array([298.0, 298.0, 280.0, 280.0, 298.0, 0.013])
# Tight-but-reachable tolerance: the forward converges (so the IFT identity
# holds) and x* is an accurate root (so IFT and FD agree).
_CFG = CanopyConfig(tol=1e-6, max_iters=300)


def _tf_sun_of_La(La):
    return solve_canopy_closure(_X0, _bundle(La=La), _CFG)[0][0]


def _tf_sun_of_Vc3(Vc3):
    return solve_canopy_closure(_X0, _bundle(Vc3=Vc3), _CFG)[0][0]


def test_ift_grad_matches_fd_forcing():
    La0 = _a(350.0)
    g_ad = jax.grad(_tf_sun_of_La)(La0)
    g_fd = (_tf_sun_of_La(La0 + 0.1) - _tf_sun_of_La(La0 - 0.1)) / 0.2
    assert float(g_ad) > 0.0  # more incoming LW -> warmer sunlit leaf
    assert jnp.allclose(g_ad, g_fd, rtol=1e-3, atol=1e-5)


def test_ift_grad_matches_fd_trainable_param():
    Vc0 = _a(40.0)
    g_ad = jax.grad(_tf_sun_of_Vc3)(Vc0)
    g_fd = (_tf_sun_of_Vc3(Vc0 + 0.05) - _tf_sun_of_Vc3(Vc0 - 0.05)) / 0.1
    assert jnp.isfinite(g_ad)
    assert jnp.allclose(g_ad, g_fd, rtol=2e-3, atol=1e-6)


def test_per_field_masking_keeps_finite_grads():
    """A NaN cotangent for the MOST forcing (Ta) must not zero La's gradient."""
    def tf(La, Ta):
        return solve_canopy_closure(_X0, _bundle(La=La, Ta=Ta), _CFG)[0][0]

    g_La, g_Ta = jax.grad(tf, argnums=(0, 1))(_a(350.0), _a(298.0))
    assert float(g_La) > 0.0           # preserved, finite, nonzero
    assert jnp.isfinite(g_La)
    assert float(g_Ta) == 0.0          # MOST-forcing NaN cotangent zeroed


def test_initial_guess_cotangent_is_zero():
    bundle = _bundle()

    def tf_of_x0(x0):
        return solve_canopy_closure(x0, bundle, _CFG)[0][0]

    g_x0 = jax.grad(tf_of_x0)(_X0)
    assert jnp.allclose(g_x0, 0.0, atol=1e-8)


def test_nonconverged_returns_zero_gradient():
    cfg_bad = CanopyConfig(tol=1e-12, max_iters=1)

    def tf_bad(La):
        return solve_canopy_closure(_X0, _bundle(La=La), cfg_bad)[0][0]

    g = jax.grad(tf_bad)(_a(350.0))
    assert float(g) == 0.0


def test_cold_calm_forcing_stays_finite():
    """Fix 3a (non-finite Newton-step guard): at the calm cold-night MOST edge the
    aerodynamic Jacobian can degenerate so ``jnp.linalg.solve`` returns a NaN step
    — and ``jnp.clip(nan)`` stays NaN, which would poison every coupled land leaf
    and force the atomic land guard to revert the step.  The solver now sanitises a
    non-finite step to 0 BEFORE the clip, so the closure returns its last finite
    iterate.  This locks in a finite solve for the extreme cold / near-calm /
    no-sunlight forcing that regime lives in."""
    stressed = _bundle(La=_a(180.0), Ta=_a(233.0))._replace(
        SZA=_a(89.0), fSun=_a(0.0), APAR_Sun=_a(0.0), APAR_Sh=_a(0.0),
        ASW_Sun=_a(0.0), ASW_Sh=_a(0.0), ASW_Soil=_a(0.0),
        ur=_a(1.0e-4), Ts_bc=_a(235.0), Tv_atm=_a(233.0), q_atm=_a(1.0e-4))
    x0 = jnp.array([235.0, 235.0, 233.0, 233.0, 235.0, 1.0e-4])
    out, _n_iters = solve_canopy_closure(x0, stressed, _CFG)
    assert bool(jnp.all(jnp.isfinite(out)))
