"""Second-order differentiability (Hessian) health of the bulk-flux kernels.

A differentiable ESM must be healthy beyond first order: Hessians of the
air-sea fluxes with respect to their inputs feed curvature-based
applications (uncertainty quantification, Laplace approximations in DA,
sensitivity/rectification diagnostics).  First-order tests
(``test_bulk_flux_differentiability``, Taylor tests) do not cover this —
``max``/``clip``/``where`` branches and the gustiness ``cbrt`` floor have
well-defined first derivatives but can inject NaN/Inf or garbage only at
second order.

Two gates:

1. **Analytic reference** — for the constant-coefficient bulk formula the
   sensible-heat Hessian in the basis (u, v, T_sfc, T_air) is known in
   closed form.  With K = rho*c_p*Ch, U = |(u,v)|, unit vector (uh, vh)
   and dT = T_sfc − T_air:

       d2F/du2      =  K*dT*vh^2/U        (convexity of the modulus)
       d2F/dv2      =  K*dT*uh^2/U
       d2F/dudv     = −K*dT*uh*vh/U
       d2F/du dTsfc =  K*uh    d2F/du dTair = −K*uh   (likewise v, vh)
       temperature-temperature block = 0  (bilinear in each temperature)

   ``jax.hessian`` through ``simple_bulk_fluxes`` must reproduce every
   entry — an exact, machine-precision check of second-order AD.

2. **Finiteness sweep** — ``jacfwd(jacrev(...))`` through the full MOST
   solver (fori_loop, scheme psi functions, zeta clips, wind floors,
   gustiness w* cbrt) must be finite for coare3 (gustiness on and off)
   and large_yeager across calm/stable/unstable/neutral/high-wind states.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from legoesm.core.bulk_flux import (  # noqa: E402
    compute_most_fluxes,
    simple_bulk_fluxes,
)

from legoesm import constants  # noqa: E402


def test_constant_coeff_sensible_hessian_matches_analytic():
    cd, ch = 1.2e-3, 1.1e-3
    u, v = 4.0, -3.0            # speed 5, off-axis so every entry is nonzero
    t_s, t_a = 301.0, 297.5
    q_s, q_a = 0.022, 0.011
    rho = 1.17

    def sh(x):
        uu, vv, ts, ta = x
        wind = jnp.sqrt(uu ** 2 + vv ** 2)
        out = simple_bulk_fluxes(
            uu[None], vv[None], ta[None], jnp.full((1,), q_a),
            ts[None], jnp.full((1,), q_s), jnp.full((1,), rho),
            wind[None], cd, ch)
        return out[2][0]

    x0 = jnp.array([u, v, t_s, t_a])
    hess = np.asarray(jax.hessian(sh)(x0))

    k = rho * constants.c_pd * ch
    spd = np.hypot(u, v)
    uh, vh = u / spd, v / spd
    dt = t_s - t_a
    expected = np.array([
        [k * dt * vh ** 2 / spd, -k * dt * uh * vh / spd,  k * uh, -k * uh],
        [-k * dt * uh * vh / spd, k * dt * uh ** 2 / spd,  k * vh, -k * vh],
        [k * uh,                  k * vh,                  0.0,     0.0],
        [-k * uh,                -k * vh,                  0.0,     0.0],
    ])
    np.testing.assert_allclose(hess, expected, rtol=1e-12, atol=1e-12)


# (scheme, gustiness_w_zi): coare3 scheme-native (gustiness on), coare3
# explicitly gust-free, and large_yeager (no gustiness by construction).
_CONFIGS = [("coare3", None), ("coare3", 0.0), ("large_yeager", None)]

# (u, v, T_air, q_air, T_sfc, q_sfc, rho) spanning the regimes where the
# kernel's nonsmooth pieces activate: convective gustiness (calm unstable),
# stable psi branch, the Stanton switch neighborhood (near-neutral), the
# Charnock ramp (high wind).
_STATES = {
    "unstable_moderate": (6.0, 3.0, 295.0, 0.012, 300.0, 0.024, 1.15),
    "stable_moderate": (6.0, 3.0, 303.0, 0.010, 299.0, 0.022, 1.15),
    "calm_unstable": (0.3, 0.2, 296.0, 0.013, 302.0, 0.026, 1.12),
    "near_neutral": (8.0, 0.0, 299.99, 0.017642, 300.0, 0.018, 1.14),
    "high_wind": (22.0, 10.0, 298.0, 0.008, 300.0, 0.023, 1.18),
}


@pytest.mark.parametrize("scheme,gust", _CONFIGS,
                         ids=[f"{s}-gust{g}" for s, g in _CONFIGS])
def test_most_flux_hessian_finite_across_regimes(scheme, gust):
    def f(x):
        u, v, ta, qa, ts, qs, rho = x
        out = compute_most_fluxes(
            u[None], v[None], ta[None], qa[None], ts[None], qs[None],
            rho[None], scheme=scheme, gustiness_w_zi=gust)
        # tau_x, shflx, lhflx: one scalar stack covers momentum + both
        # heat fluxes (tau_y is tau_x with u<->v roles).
        return jnp.stack([out[0][0], out[2][0], out[3][0]])

    hess = jax.jit(jax.jacfwd(jax.jacrev(f)))  # (3, 7, 7)
    for name, state in _STATES.items():
        h = np.asarray(hess(jnp.array(state)))
        assert np.all(np.isfinite(h)), (
            f"{scheme} gust={gust} {name}: non-finite Hessian entries")
        # Garbage-magnitude guard: entries are physically bounded well
        # below this (measured max ~6e5, from rho*L_v curvature terms).
        assert np.abs(h).max() < 1e9, (
            f"{scheme} gust={gust} {name}: Hessian blow-up "
            f"max|h|={np.abs(h).max():.3e}")
