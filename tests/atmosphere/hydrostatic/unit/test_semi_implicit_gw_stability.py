"""Truncation-parametrized gravity-wave stability gate for ssp_rk3_step_si (#920).

The Hoskins-Simmons semi-implicit (SI) correction in
``legoesm.timestepping.semi_implicit.si_correction`` is meant to make the
SSP-RK3-SI step *unconditionally* stable for the linearized gravity-wave (GW)
subsystem, removing the GW CFL constraint.  A formulation bug (issue #920) made
the per-step amplification ``|lambda(kappa)| > 1`` and *growing* with
``kappa = omega_n * dt`` (external GW speed times the mode frequency), so the
step was unconditionally *unstable*.  Because ``kappa_max`` grows with the
spectral truncation ``n_max``, the defect was invisible at the T21 scale of the
existing SI tests but produced an epoch-0 NaN at T85+/dt=1800.

This test reproduces the *exact* per-mode linear amplification of the real
``ssp_rk3_step_si`` and asserts it stays inside the unit circle across
``n_max in {21, 42, 85, 106}`` -- the truncation-parametrized acceptance
criterion the T21-only smoke tests could not provide.

Construction (mirrors the scalar (D, Psi) gravity-wave oscillator of the
offline stability analysis, but drives the *real* integrator):

  * Real ``GaussianGrid`` (non-rotating: ``omega=0`` decouples vorticity so the
    (div, T) pair is a clean GW oscillator) and a single-level sigma coordinate
    so ``precompute_si_matrices`` yields a real, positive scalar ``Gamma``.
  * A linear explicit tendency consistent with that ``si_data``:
        dD/dt   =  eps_n * T ,     eps_n = Lambda_n * Gamma / T_ref
        dT/dt   = -T_ref * D
    where ``Lambda_n = n(n+1)/a^2``.  With ``gamma = T_ref`` this matches the
    ``-alpha*dt*T_ref*delta_div`` temperature correction inside
    ``si_correction`` exactly, so the closed (D, T) loop is the trapezoidal
    theta-method solve the SI scheme is supposed to implement.  The implicit
    coupling ``M_n = alpha^2 dt^2 eps_n gamma`` then equals
    ``alpha^2 dt^2 Lambda_n Gamma`` -- i.e. the actual ``si_matrices``.
  * The (D, T) 2x2 amplification block is the *exact* Jacobian of one real
    ``ssp_rk3_step_si`` at rest (``jax.jacfwd``; the map is linear so this is
    the amplification matrix itself).

Pre-fix this test FAILS (max ``|lambda|`` ~ 2.8 at T21, ~1.8e4 at T85);
post-fix it PASSES with ``|lambda| <= 1`` at alpha=0.5 (neutral Crank-Nicolson)
and ``|lambda| < 1`` at alpha>0.5 (damping) for every mode and truncation.

SCOPE / KNOWN LIMITATION (issue #920, second contributor).  This is an
*integrator-formulation* gate: it locks in the two-line ``si_correction`` fix
for a *well-conditioned* gravity-wave oscillator.  ``nlev=1`` is used
deliberately so ``precompute_si_matrices`` yields a real, positive *scalar*
``Gamma`` -- a clean single-mode oscillator with real frequency, exactly the
system the fix is derived for.  It does NOT certify the full multi-level
spectral dycore: the discretized ``Gamma`` is non-normal with complex
eigenvalues at ``nlev >= 4`` (real+positive only at ``nlev <= 2``), which
injects a *separate* growing gravity-wave mode (isolated ``|lambda| ~ 4.4`` at
T106/nlev=8) that the trapezoidal SI cannot neutralize and hyperdiffusion
cannot damp.  A real T85/nlev=8/dt=1800 SI rollout therefore still blows up at
alpha=0.5 *after* this fix (it only delays the NaN); stabilizing it needs a
follow-up ``Gamma`` symmetrization/positivity fix.  This test guards the
formulation regression only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.spectral_pe import SpectralHydrostaticState
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.timestepping.semi_implicit import (
    precompute_si_matrices,
    ssp_rk3_step_si,
)

jax.config.update("jax_enable_x64", True)

# Linearization reference temperature [K] (the SI ``si_T_ref`` default); the
# scalar GW oscillator is built around this reference, not a physical constant
# lookup -- it only sets the (D, T) coupling scale.
_T_REF_K = 300.0
_DT_S = 1800.0  # 30 min external time step -- the #920 configuration


@pytest.fixture(autouse=True)
def _x64_fp64():
    orig_x64 = jax.config.jax_enable_x64
    orig_pol = get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_pol)
    jax.config.update("jax_enable_x64", orig_x64)


def _gw_oscillator_step(n_max, alpha, dt=_DT_S, T_ref=_T_REF_K):
    """Return ``(step_fn, grid, si_data, eps_per_mode)`` for the GW oscillator.

    ``step_fn(div, T)`` applies the real ``ssp_rk3_step_si`` to a full spectral
    state carrying the perturbation arrays ``div`` (n_sh, 1) and ``T`` (n_sh, 1).
    """
    grid = create_gaussian_grid(n_max=n_max, omega=0.0)          # non-rotating
    sigma = create_sigma_coordinate(1, sigma_top=0.01)           # nlev=1 -> scalar Gamma
    si_data = precompute_si_matrices(grid, sigma, T_ref=T_ref, alpha=alpha, dt=dt)

    Gamma = jnp.asarray(si_data.Gamma)[0, 0]                     # real positive scalar
    # dD/dt coefficient per spectral coefficient: eps_n = Lambda_n * Gamma / T_ref.
    eps = (si_data.eigenvalues[grid.ls] * Gamma / T_ref)        # (n_sh,)

    n_sh = grid.n_sh
    zero_3d = jnp.zeros((n_sh, 1), dtype=jnp.complex128)
    zero_2d = jnp.zeros((n_sh,), dtype=jnp.complex128)

    def _state(div, T):
        return SpectralHydrostaticState(
            vor_hat=Field(data=zero_3d, name="vor_hat", dims=()),
            div_hat=Field(data=div, name="div_hat", dims=()),
            T_hat=Field(data=T, name="T_hat", dims=()),
            lnps_hat=Field(data=zero_2d, name="lnps_hat", dims=()),
            phis_hat=Field(data=zero_2d, name="phis_hat", dims=()),
        )

    def _tendency(s):
        # Linearized gravity-wave oscillator consistent with ``si_data``.
        dD = eps[:, None] * s.T_hat.data
        dT = -T_ref * s.div_hat.data
        return s._replace(
            vor_hat=s.vor_hat.replace(data=jnp.zeros_like(s.vor_hat.data)),
            div_hat=s.div_hat.replace(data=dD),
            T_hat=s.T_hat.replace(data=dT),
            lnps_hat=s.lnps_hat.replace(data=jnp.zeros_like(s.lnps_hat.data)),
            phis_hat=s.phis_hat.replace(data=jnp.zeros_like(s.phis_hat.data)),
        )

    def step_fn(div, T):
        out = ssp_rk3_step_si(_state(div, T), _tendency, dt, si_data, grid)
        return out.div_hat.data, out.T_hat.data

    return step_fn, grid, si_data, eps


def _amp_block_at_mode(step_fn, grid, j):
    """Exact 2x2 (D, T) amplification block at spectral coefficient ``j``."""
    n_sh = grid.n_sh
    zero_3d = jnp.zeros((n_sh, 1), dtype=jnp.complex128)

    def _map(vec):  # vec: real (2,) = (D, T) at coeff j -> (2,) after one step
        vc = vec.astype(jnp.complex128)
        div = zero_3d.at[j, 0].set(vc[0])
        T = zero_3d.at[j, 0].set(vc[1])
        out_div, out_T = step_fn(div, T)
        return jnp.stack([out_div[j, 0], out_T[j, 0]])

    block = np.asarray(jax.jacfwd(_map)(jnp.zeros(2)))
    # m=0 zonal coefficients live in the real subspace; imaginary leakage is
    # roundoff.  Guard against a non-real block (would signal a broken setup).
    assert np.abs(block.imag).max() < 1e-9, (
        f"amplification block acquired an imaginary part {np.abs(block.imag).max():.2e}"
    )
    return block.real


def _mode_indices(grid, n_max):
    """Representative m=0 coefficients spanning n=1..n_max (fastest = n_max)."""
    ls = np.asarray(grid.ls)
    ms = np.asarray(grid.ms)
    targets = sorted({1, n_max // 4, n_max // 2, (3 * n_max) // 4, n_max} - {0})
    out = []
    for n in targets:
        idx = np.where((ls == n) & (ms == 0))[0]
        if len(idx):
            out.append((n, int(idx[0])))
    return out


@pytest.mark.parametrize("n_max", [21, 42, 85, 106])
def test_ssp_rk3_step_si_stable_across_truncation(n_max):
    """|lambda| <= 1 for every resolved GW mode at alpha=0.5 (neutral CN), and
    strictly < 1 at alpha=0.55 (damping), across T21..T106.

    The fastest-resolved mode ``n=n_max`` (largest ``kappa``) is the one whose
    amplification the #920 bug blew up (|lambda| ~ 1.8e4 at T85); it must now sit
    on/inside the unit circle.
    """
    for alpha, tol, strict in ((0.5, 1.0 + 1e-9, False), (0.55, 1.0, True)):
        step_fn, grid, _si, _eps = _gw_oscillator_step(n_max, alpha)
        for n, j in _mode_indices(grid, n_max):
            block = _amp_block_at_mode(step_fn, grid, j)
            rho = float(np.abs(np.linalg.eigvals(block)).max())
            assert rho <= tol, (
                f"T{n_max} alpha={alpha}: mode n={n} amplification "
                f"|lambda|={rho:.6f} exceeds {tol:.9f} -- SI gravity-wave "
                f"instability (issue #920) is present"
            )
            if strict:
                assert rho < 1.0, (
                    f"T{n_max} alpha={alpha}>0.5: mode n={n} |lambda|={rho:.6f} "
                    f"is not strictly damping"
                )


def test_gravity_wave_amplification_blows_up_without_the_fix_signature():
    """Documents the discriminating signature: at T85 the fastest mode carries a
    large ``kappa`` (external-mode GW), so a mis-centered SI correction amplifies
    it by many orders of magnitude per step.  Here we assert only that the
    *fixed* scheme keeps that same mode neutral -- the companion parametrized
    test is the actual gate; this pins the physical scale so a future regression
    reads clearly (pre-fix this mode had |lambda| ~ 1.8e4)."""
    step_fn, grid, si_data, eps = _gw_oscillator_step(85, alpha=0.5)
    ls = np.asarray(grid.ls)
    ms = np.asarray(grid.ms)
    j = int(np.where((ls == 85) & (ms == 0))[0][0])
    # kappa = omega*dt for the top mode; must be well into the "unstable-if-buggy"
    # regime (>> 1) so this test is non-vacuous.
    Gamma = float(jnp.asarray(si_data.Gamma)[0, 0])
    kappa = _DT_S * np.sqrt(float(si_data.eigenvalues[85]) * Gamma)
    assert kappa > 5.0, f"top-mode kappa={kappa:.2f} too small to exercise the bug"
    block = _amp_block_at_mode(step_fn, grid, j)
    rho = float(np.abs(np.linalg.eigvals(block)).max())
    assert rho <= 1.0 + 1e-9, f"T85 top-mode |lambda|={rho:.6f} > 1 (regression of #920)"
