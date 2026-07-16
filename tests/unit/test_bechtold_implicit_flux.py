"""Direct tests for the IMPLICIT conservative flux-form mass-flux transport.

The implicit (backward-Euler / theta-blended) conservative flux solve is a
selectable option on the shared mass-flux kernel
(:func:`legoesm.atmosphere.physics.convection.mass_flux.apply_mass_flux_kernel`
with ``subsidence_solve="implicit_flux"``; Bechtold opts in via
``BechtoldConfig(subsidence_solve="implicit_flux")``).

These tests pin the properties that motivated the redesign:

1. **Conservation** -- the transport conserves column MSE ``h = c_p T + g z
   + L_v q_v`` and total water ``q_v + q_c`` to MACHINE PRECISION
   (telescoping flux form), unlike the advective default which leaks a
   resolution-dependent fraction.  Non-vacuous: the same assertion FAILS
   for ``subsidence_solve="advective"``.
2. **Stability under time integration** -- integrating a pure-convection
   column for >=200 steps at the production ``dt`` from a 2dz temperature
   checkerboard stays finite and bounded (no checkerboard growth).  This
   is the gate the prior EXPLICIT flux-form attempt failed (NaN by
   ~day 0.77).
3. (RCE realism is exercised by the SCM-RCE harness, not here.)
4. **Advective byte-identity** -- the default ``"advective"`` path is
   bit-identical to a direct advective call (the dispatch is a no-op for
   the default), and the dispatch RAISES on an unknown selector.
5. **AD-safe** -- ``jax.grad`` is finite (and non-zero) through the
   implicit tridiagonal solve; ``jit``/``vmap`` match eager.

Convention note (stated at the kernel): ``z`` up, surface-LAST arrays;
``M >= 0`` updraft mass flux; compensating subsidence ``-M`` (downward).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
from legoesm.atmosphere.physics.convection.config import BechtoldConfig
from legoesm.atmosphere.physics.convection.mass_flux import (
    apply_mass_flux_kernel,
    apply_mass_flux_kernel_implicit_flux,
)


def _column(ncol=2, nlev=16, T_sfc=302.0, q_sfc=16e-3, lapse_rate=7.5,
            p_s=1.0e5, p_top=5.0e3):
    """Surface-last conditionally-unstable column (matches test_bechtold)."""
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), p_half_inner,
         jnp.full((ncol, 1), p_s)], axis=1)
    z_full = -8500.0 * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    q = q_sfc * jnp.exp(-z_full / 3000.0)
    u = jnp.broadcast_to(jnp.linspace(0, 25, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros_like(u)
    return T, q, p_full, p_half, u, v


def _col_int(x, dp):
    return jnp.sum(x * dp / constants.g, axis=1)


def _synthetic_kernel_inputs(seed=0, ncol=3, nlev=20):
    """A PHYSICALLY-VALID surface-last column (monotonic sigma pressure so
    Δp > 0 and the interface mass flux vanishes at the top/surface faces)
    with a randomly-perturbed but well-ordered thermodynamic state and a
    bell-shaped mass flux that is zero at the boundaries (the telescoping
    flux form requires F_top = F_surface = 0)."""
    rng = np.random.default_rng(seed)
    p_s, p_top = 1.0e5, 5.0e3
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), p_half_inner,
         jnp.full((ncol, 1), p_s)], axis=1
    )
    z = -8500.0 * jnp.log(p_full / p_s)
    T = 300.0 - 6.5e-3 * z + jnp.array(rng.normal(0, 1.5, (ncol, nlev)))
    q_v = jnp.array(np.abs(0.014 * np.exp(-np.asarray(z) / 3000.0)
                          + rng.normal(0, 3e-4, (ncol, nlev))))
    T_u = T + 2.0
    q_v_u = q_v * 1.1
    q_c_u = jnp.array(np.abs(rng.normal(1e-3, 3e-4, (ncol, nlev))))
    # Bell-shaped mass flux: ZERO at base (k=nlev-1) and top (k=0), peak
    # mid-column (the interface helper sets the top/surface FACES to 0).
    M = jnp.broadcast_to(
        0.04 * jnp.sin(jnp.pi * jnp.linspace(0, 1, nlev))[None, :], (ncol, nlev)
    )
    rho = p_full / (constants.R_d * jnp.maximum(T, 100.0))
    return T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho


# ---------------------------------------------------------------------------
# Gate 4 -- dispatch hardening + advective byte-identity
# ---------------------------------------------------------------------------

def test_advective_dispatch_byte_identical():
    (T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho) = (
        _synthetic_kernel_inputs()
    )
    out_default = apply_mass_flux_kernel(
        T, q_v, p_full, T_u, q_v_u, q_c_u, M, z, rho, 7.5e-5, 0.05,
    )
    out_explicit = apply_mass_flux_kernel(
        T, q_v, p_full, T_u, q_v_u, q_c_u, M, z, rho, 7.5e-5, 0.05,
        subsidence_solve="advective",
    )
    for a, b in zip(out_default, out_explicit):
        assert jnp.array_equal(a, b), "advective dispatch is not byte-identical"


def test_dispatch_raises_on_unknown():
    (T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho) = (
        _synthetic_kernel_inputs()
    )
    with pytest.raises(ValueError, match="unknown subsidence_solve"):
        apply_mass_flux_kernel(
            T, q_v, p_full, T_u, q_v_u, q_c_u, M, z, rho, 7.5e-5, 0.05,
            subsidence_solve="bogus",
        )


def test_implicit_requires_p_half_and_dt():
    (T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho) = (
        _synthetic_kernel_inputs()
    )
    with pytest.raises(ValueError, match="requires p_half and dt"):
        apply_mass_flux_kernel(
            T, q_v, p_full, T_u, q_v_u, q_c_u, M, z, rho, 7.5e-5, 0.05,
            subsidence_solve="implicit_flux",
        )


# ---------------------------------------------------------------------------
# Gate 1 -- conservation (machine precision; non-vacuous vs advective)
# ---------------------------------------------------------------------------

def test_implicit_kernel_conserves_mse_and_water_to_machine_precision():
    (T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho) = (
        _synthetic_kernel_inputs()
    )
    dp = p_half[:, 1:] - p_half[:, :-1]
    dt = 1800.0
    dT, dqv, dqc = apply_mass_flux_kernel_implicit_flux(
        T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho, 7.5e-5, 0.05, dt,
    )
    H = _col_int(constants.c_pd * dT, dp)
    Q = _col_int(constants.L_v * dqv, dp)
    C = _col_int(constants.L_v * dqc, dp)
    mse_rel = jnp.abs(H + Q + C) / (jnp.abs(H) + jnp.abs(Q) + jnp.abs(C) + 1e-10)
    water = _col_int(dqv + dqc, dp)
    water_scale = _col_int(jnp.abs(dqv), dp) + 1e-15
    water_rel = jnp.abs(water) / water_scale
    assert jnp.all(mse_rel < 1e-9), f"implicit MSE residual not machine-zero: {mse_rel}"
    assert jnp.all(water_rel < 1e-9), f"implicit total-water not conserved: {water_rel}"
    assert jnp.all(dqc >= -1e-12), "implicit condensate source went negative"

    dTa, dqva, dqca = apply_mass_flux_kernel(
        T, q_v, p_full, T_u, q_v_u, q_c_u, M, z, rho, 7.5e-5, 0.05,
    )
    Ha = _col_int(constants.c_pd * dTa, dp)
    Qa = _col_int(constants.L_v * dqva, dp)
    Ca = _col_int(constants.L_v * dqca, dp)
    mse_rel_adv = jnp.abs(Ha + Qa + Ca) / (
        jnp.abs(Ha) + jnp.abs(Qa) + jnp.abs(Ca) + 1e-10
    )
    assert jnp.all(mse_rel_adv > 100.0 * mse_rel), (
        "advective residual not materially larger -- test would pass even "
        "if implicit were reverted (vacuous)"
    )


def test_bechtold_implicit_mse_conservation_within_tolerance():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    dp = ph[:, 1:] - ph[:, :-1]

    def _rel(ss, use_ifs_cape_closure=True):
        out, _, _ = bechtold_convection(
            T=T, q_v=q, p_full=pf, p_half=ph, u=u, v=v,
            conv_prog_profile=cpp, conv_stoch_state=stoch, prng_key=None,
            dt=1800.0,
            config=BechtoldConfig(
                enable_stochastic=False, enable_cmt=False, subsidence_solve=ss,
                use_ifs_cape_closure=use_ifs_cape_closure,
            ),
            moisture_convergence=jnp.zeros_like(T),
        )
        # The latent-heat sink C is the FULL detrained condensate: with the
        # default #929 rain split (precip_efficiency=0.7) precip_efficiency of
        # it moves from dq_c_conv_dt into dq_r_conv_dt, but the latent heat of
        # ALL of it is already booked in dT_dt (H), so the enthalpy budget must
        # sum dq_c + dq_r (the split re-partitions water downstream; it does
        # not change the scheme's internal energy balance).
        _dqr = out.dq_r_conv_dt if out.dq_r_conv_dt is not None else 0.0
        H = float(jnp.sum(out.dT_dt * dp / constants.g, axis=1).mean()) * constants.c_pd
        Q = float(jnp.sum(out.dq_v_dt * dp / constants.g, axis=1).mean()) * constants.L_v
        C = float(jnp.sum((out.dq_c_conv_dt + _dqr) * dp / constants.g, axis=1).mean()) * constants.L_v
        return abs(H + Q + C) / (abs(H) + abs(Q) + abs(C) + 1e-10)

    # DEFAULT scheme (IFS cape closure ON since 2026-07-16): both solves must
    # meet the hard bar.  The closure changes the M_u magnitude regime, so the
    # legacy adv>impl ORDERING is not guaranteed here (raw-tendency budget
    # closure is delegated to the orchestrator rebalance per the contract);
    # the anti-vacuity ordering claim is asserted on the LEGACY closure below,
    # where it was established.
    rel_impl = _rel("implicit_flux")
    rel_adv = _rel("advective")
    assert rel_impl < 0.30, f"implicit Bechtold MSE residual {rel_impl*100:.1f}% >= 30%"
    assert rel_adv < 0.30, f"advective Bechtold MSE residual {rel_adv*100:.1f}% >= 30%"
    rel_impl_legacy = _rel("implicit_flux", use_ifs_cape_closure=False)
    rel_adv_legacy = _rel("advective", use_ifs_cape_closure=False)
    assert rel_impl_legacy < 0.30, (
        f"legacy implicit MSE residual {rel_impl_legacy*100:.1f}% >= 30%")
    assert rel_adv_legacy > rel_impl_legacy, (
        "legacy advective residual not larger than implicit -- guard would be vacuous"
    )


def test_bechtold_implicit_condensate_non_negative():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    out, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), jnp.zeros((ncol,)),
        None, 1800.0,
        BechtoldConfig(subsidence_solve="implicit_flux"),
        moisture_convergence=jnp.zeros_like(T),
    )
    assert jnp.all(out.dq_c_conv_dt >= -1e-12)


# ---------------------------------------------------------------------------
# Gate 2 -- stability under MULTI-STEP time integration (the prior NaN gate)
# ---------------------------------------------------------------------------

def test_implicit_kernel_stable_under_long_integration_checkerboard():
    ncol, nlev = 2, 24
    p_s, p_top = 1.0e5, 5.0e3
    sig = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sig[None, :] * jnp.full((ncol, 1), p_s)
    phi = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), phi, jnp.full((ncol, 1), p_s)], 1
    )
    z = -8500.0 * jnp.log(p_full / p_s)
    rho = p_full / (constants.R_d * 250.0)
    M = jnp.broadcast_to(
        0.05 * jnp.sin(jnp.pi * jnp.linspace(0, 1, nlev))[None, :], (ncol, nlev)
    )
    T_env0 = 300.0 - 6.5e-3 * z
    T_u = T_env0 + 3.0
    q_u = jnp.full((ncol, nlev), 5e-3)
    q_c_u = jnp.full((ncol, nlev), 1e-3)
    dt, nsteps = 1800.0, 300
    checker = jnp.where(jnp.arange(nlev) % 2 == 0, 1.0, -1.0)[None, :]
    T = T_env0 + 5.0 * checker
    q = jnp.full((ncol, nlev), 8e-3)

    @jax.jit
    def step(T, q):
        dT, dqv, _ = apply_mass_flux_kernel_implicit_flux(
            T, q, p_full, p_half, T_u, q_u, q_c_u, M, z, rho, 7.5e-5, 0.05, dt,
        )
        return T + dt * dT, jnp.clip(q + dt * dqv, 0.0, None)

    # 2dz-mode amplitude relative to the local smooth profile, isolated by
    # a 1-2-1 vertical smoother: the instability signature is the
    # grid-scale (2dz) residual GROWING without bound; the legitimate
    # detrainment reshaping is smooth and does not project onto this.
    def grid_mode(T):
        smooth = jnp.zeros_like(T)
        smooth = smooth.at[:, 1:-1].set(
            0.25 * T[:, :-2] + 0.5 * T[:, 1:-1] + 0.25 * T[:, 2:]
        )
        smooth = smooth.at[:, 0].set(T[:, 0]).at[:, -1].set(T[:, -1])
        return float(jnp.max(jnp.abs(T - smooth)))

    g0 = grid_mode(T)            # initial 2dz residual (~the 5 K checker)
    maxes = []
    for _ in range(nsteps):
        T, q = step(T, q)
        maxes.append(float(jnp.max(jnp.abs(T))))
    assert bool(jnp.all(jnp.isfinite(T))), "implicit transport went non-finite"
    # A stable heated column (no radiation/surface flux) stays physically
    # bounded; the prior EXPLICIT flux form blew to ~2.7e4 K here.
    assert max(maxes) < 400.0, (
        f"implicit transport unbounded (max|T|={max(maxes):.1f} K over the run)"
    )
    # The 2dz grid mode must DAMP, not grow: backward-Euler removes the
    # checkerboard (it does not amplify the highest wavenumber).
    assert grid_mode(T) <= g0 + 1e-6, (
        f"2dz grid mode grew: {g0:.3e} -> {grid_mode(T):.3e} K"
    )


def test_bechtold_implicit_multistep_finite_bounded():
    T0, q0, pf, ph, u, v = _column(nlev=24)
    ncol, nlev = T0.shape
    checker = jnp.where(jnp.arange(nlev) % 2 == 0, 1.0, -1.0)[None, :]
    T = T0 + 2.0 * checker
    cfg = BechtoldConfig(
        subsidence_solve="implicit_flux", enable_cmt=False,
        enable_stochastic=False,
    )
    dt, nsteps = 1800.0, 200

    @jax.jit
    def step(T, q, cpp):
        out, M, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, jnp.zeros((ncol,)), None, dt, cfg,
            jnp.zeros_like(T),
        )
        return T + dt * out.dT_dt, jnp.clip(q + dt * out.dq_v_dt, 0.0, None), M

    cpp = jnp.zeros((ncol, nlev))
    q = q0
    for _ in range(nsteps):
        T, q, cpp = step(T, q, cpp)
        if not bool(jnp.all(jnp.isfinite(T))):
            break
    assert bool(jnp.all(jnp.isfinite(T))), "Bechtold implicit went non-finite"
    assert float(jnp.max(jnp.abs(T))) < 1.0e3, "Bechtold implicit T runaway"
    assert bool(jnp.all(jnp.isfinite(q)))


# ---------------------------------------------------------------------------
# Gate 5 -- AD-safe through the implicit tridiagonal solve; jit/vmap
# ---------------------------------------------------------------------------

def test_implicit_grad_finite_through_tridiagonal_solve():
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))

    def loss_eps(eps):
        out, _, _ = bechtold_convection(
            T, q, pf, ph, u, v, cpp, stoch, None, 1800.0,
            BechtoldConfig(
                subsidence_solve="implicit_flux", epsilon_deep=eps,
                enable_cmt=False, enable_stochastic=False,
            ),
            moisture_convergence=jnp.zeros_like(T),
        )
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

    g_eps = jax.grad(loss_eps)(jnp.asarray(1.75e-3))
    assert bool(jnp.isfinite(g_eps)), "non-finite grad through implicit solve"
    assert g_eps != 0.0, "dead gradient through implicit solve"

    def loss_T(Tx):
        out, _, _ = bechtold_convection(
            Tx, q, pf, ph, u, v, cpp, stoch, None, 1800.0,
            BechtoldConfig(
                subsidence_solve="implicit_flux", enable_cmt=False,
                enable_stochastic=False,
            ),
            moisture_convergence=jnp.zeros_like(Tx),
        )
        return jnp.sum(out.dT_dt ** 2)

    g_T = jax.grad(loss_T)(T)
    assert bool(jnp.all(jnp.isfinite(g_T))), "non-finite dL/dT (implicit)"


def test_implicit_jit_and_vmap_match_eager():
    (T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho) = (
        _synthetic_kernel_inputs(ncol=4)
    )
    dt = 1800.0
    eager = apply_mass_flux_kernel_implicit_flux(
        T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho, 7.5e-5, 0.05, dt,
    )

    def f(*args):
        return apply_mass_flux_kernel_implicit_flux(*args, 7.5e-5, 0.05, dt)

    jitted = jax.jit(f)(T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho)
    for a, b in zip(eager, jitted):
        assert jnp.allclose(a, b, atol=1e-10, rtol=1e-7), "jit != eager"

    vin = [jnp.stack([a, a]) for a in
           (T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho)]
    vout = jax.vmap(
        lambda *a: apply_mass_flux_kernel_implicit_flux(*a, 7.5e-5, 0.05, dt)
    )(*vin)
    for member, e in zip(vout, eager):
        assert jnp.allclose(member[0], e, atol=1e-10, rtol=1e-7), "vmap != eager"


def test_implicit_theta_blend_conserves():
    (T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho) = (
        _synthetic_kernel_inputs()
    )
    dp = p_half[:, 1:] - p_half[:, :-1]
    dt = 1800.0
    for theta in (0.5, 0.7, 1.0):
        dT, dqv, dqc = apply_mass_flux_kernel_implicit_flux(
            T, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M, z, rho,
            7.5e-5, 0.05, dt, theta_implicit=theta,
        )
        H = _col_int(constants.c_pd * dT, dp)
        Q = _col_int(constants.L_v * dqv, dp)
        C = _col_int(constants.L_v * dqc, dp)
        rel = jnp.abs(H + Q + C) / (jnp.abs(H) + jnp.abs(Q) + jnp.abs(C) + 1e-10)
        assert jnp.all(rel < 1e-9), f"theta={theta} MSE residual {rel}"


def test_stochastic_draw_chunk_invariance():
    """A1 increment 2 leaf gate: the per-GLOBAL-column fold_in draw makes
    the stochastic AR1 update decomposition-invariant — running the leaf
    on a contiguous CHUNK of columns (with that chunk's global col_index)
    must reproduce exactly the corresponding slice of the full-domain
    run.  The legacy bulk normal(key, (ncol,)) draw fails this."""
    import jax

    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    assert ncol >= 2, "need >=2 columns to split"
    cpp = jnp.zeros((ncol, nlev))
    stoch0 = jnp.linspace(-0.5, 0.5, ncol)
    key = jax.random.PRNGKey(7)
    cfg = BechtoldConfig(enable_stochastic=True, enable_cmt=False)
    ids = jnp.arange(ncol, dtype=jnp.int32)

    def _run(sl):
        _, _, stoch_new = bechtold_convection(
            T=T[sl], q_v=q[sl], p_full=pf[sl], p_half=ph[sl],
            u=u[sl], v=v[sl],
            conv_prog_profile=cpp[sl], conv_stoch_state=stoch0[sl],
            prng_key=key, dt=1800.0, config=cfg,
            moisture_convergence=jnp.zeros_like(T[sl]),
            col_index=ids[sl],
        )
        return np.asarray(stoch_new)

    full = _run(slice(None))
    half = ncol // 2
    np.testing.assert_array_equal(full[:half], _run(slice(0, half)))
    np.testing.assert_array_equal(full[half:], _run(slice(half, ncol)))
    # Non-vacuity: the innovation actually moved the AR1 state.
    assert float(np.max(np.abs(full - np.asarray(
        jnp.exp(-1800.0 / cfg.stochastic_decorrelation) * stoch0)))) > 1e-8
