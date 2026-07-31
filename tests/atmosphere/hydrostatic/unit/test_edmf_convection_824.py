"""Regression tests for the EDMF-as-convection day-5 AMIP blowup (#824).

EDMF selected as the *convection* scheme died at day 5 in full-physics AMIP
(non-finite winds) even though its SCM lane is validated.  Two independent
defects, both fixed here and pinned by these tests:

1. **Non-detraining, top-heavy updraft.**  ``w_u`` came from a POSITIVE-only
   buoyancy integral (``clip(B·dz, 0, None)``), which FROZE at its peak so the
   mass flux ``M = ρ·a_u·w_u`` plateaued at its cap from the level of neutral
   buoyancy all the way to the stratosphere gate — putting the strongest
   compensating subsidence in the thin-mass upper troposphere.  Fix: the SIGNED
   buoyancy integral (``w² = w₀² + 2∫B dz``) lets the plume decelerate above the
   LNB, so ``M`` detrains to ~0 near its top like every stable scheme.
   ``test_edmf_updraft_detrains_above_neutral_buoyancy``.
2. **Non-conservative explicit subsidence.**  EDMF used the default explicit
   ``advective`` solve (``"conserves": ["none"]``), leaking column static energy
   that accumulated over ~720 steps to the day-5 wall.  Fix: default to the
   CONSERVATIVE, damping ``implicit_flux`` solve (mirrors Bechtold).
   ``test_edmf_conserves_mse_and_water_implicit`` (non-vacuous vs advective) and
   ``test_edmf_multistep_stable_large_dt`` (the day-5 regression).

Convention (stated at the kernel): ``z`` up, arrays surface-LAST (index 0 = model
top, -1 = surface); ``M ≥ 0`` updraft; compensating subsidence ``-M`` (downward).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.convection.config import ConvectiveEDMFConfig
from legoesm.atmosphere.physics.convection.mass_flux import (
    edmf_convection,
    updraft_velocity_from_buoyancy,
)


def _column(ncol=3, nlev=24, T_sfc=302.0, q_sfc=17e-3, lapse_rate=8.0,
            p_s=1.0e5, p_top=5.0e3):
    """Surface-last, conditionally-unstable column (drives EDMF convection)."""
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), p_half_inner,
         jnp.full((ncol, 1), p_s)], axis=1)
    z = -8500.0 * jnp.log(p_full / p_s)
    # Constant-lapse troposphere with an isothermal ~200 K lower stratosphere
    # floor (a realistic tropopause) — otherwise the constant lapse extrapolated
    # to the ~25 km model top gives an unphysical ~100 K, which is the IC, not a
    # drift, and would trip the multistep physical-range guard.
    T = jnp.maximum(jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z,
                    200.0)
    q_v = q_sfc * jnp.exp(-z / 3000.0)
    return T, q_v, p_full, p_half


def _col_int(x, dp):
    return jnp.sum(x * dp / constants.g, axis=1)


_DZ = 500.0  # m — uniform layer thickness for the controlled w_u helper tests


def test_edmf_defaults_to_conservative_implicit_solve():
    """The blowup fix ships as the DEFAULT (not an opt-in): the config must
    select the conservative solve out of the box."""
    cfg = ConvectiveEDMFConfig()
    assert cfg.subsidence_solve == "implicit_flux"
    assert 0.5 <= cfg.theta_implicit <= 1.0
    assert cfg.w_u_max > cfg.w_u_min > 0.0


def test_edmf_implicit_conserves_mse_far_better_than_advective():
    """The default ``implicit_flux`` solve conserves column VAPOR MSE
    (c_p T + L_v q_v) and total water DRAMATICALLY better than the legacy
    ``advective`` solve — the non-conservation that accumulated to the day-5
    blowup.

    INVARIANT UPDATED (EDMF unpaired-latent fix): this test previously asserted
    ``H + Q + C ≈ 0``, which was the correct invariant only while EDMF handed
    microphysics condensate whose latent heat had NOT been released.  EDMF now
    adds the paired ``+(L_v/c_p) dq_c`` condensation warming (as Kain-Fritsch,
    Bechtold, Tiedtke, ZM and Arakawa-Wu all do), so the conserved column
    quantity is the VAPOR MSE ``h = c_p T + L_v q_v``: a correctly paired
    scheme now has ``H + Q ≈ 0`` and hence ``H + Q + C ≈ C``.  Asserting the
    old form would fail by exactly one latent throughput.

    The implicit flux form telescopes, so the residual is fp ROUNDOFF of the ∫s
    telescoping (amplified by the large g·z carried in s and by the signal
    strength — the same kernel reaches ~1e-9 only on the strong hand-crafted
    Bechtold input, cf. test_bechtold_implicit_flux).  We therefore assert the
    residual is small in ABSOLUTE terms AND far below the advective leak, which
    is the property that fixes the blowup — not a machine-precision claim."""
    T, q_v, p_full, p_half = _column(
        ncol=3, nlev=24, T_sfc=304.0, q_sfc=19e-3, lapse_rate=9.0)  # strong CAPE
    ncol = T.shape[0]
    dp = p_half[:, 1:] - p_half[:, :-1]
    a_u = jnp.full(ncol, ConvectiveEDMFConfig().a_u_init)

    def _residuals(solve):
        cfg = ConvectiveEDMFConfig(subsidence_solve=solve)
        out, _ = edmf_convection(T, q_v, p_full, p_half, a_u, dt=1800.0, config=cfg)
        H = _col_int(constants.c_pd * out.dT_dt, dp)
        Q = _col_int(constants.L_v * out.dq_v_dt, dp)
        C = _col_int(constants.L_v * out.dq_c_conv_dt, dp)
        mse_scale = jnp.abs(H) + jnp.abs(Q) + jnp.abs(C) + 1e-10
        water = _col_int(out.dq_v_dt + out.dq_c_conv_dt, dp)
        water_scale = _col_int(jnp.abs(out.dq_v_dt), dp) + 1e-15
        # VAPOR MSE (H + Q), not H + Q + C -- see the docstring.
        return (jnp.max(jnp.abs(H + Q) / mse_scale),
                jnp.max(jnp.abs(water) / water_scale), out)

    mse_imp, water_imp, out_imp = _residuals("implicit_flux")
    mse_adv, water_adv, _ = _residuals("advective")

    assert jnp.all(out_imp.dq_c_conv_dt >= -1e-12), "condensate source went negative"
    # Implicit conserves MSE + total water to roundoff (far below the advective
    # leak that drove the day-5 blowup).
    assert float(mse_imp) < 1e-4, f"implicit vapor-MSE residual too large: {float(mse_imp):.2e}"
    assert float(water_imp) < 1e-4, f"implicit total-water residual too large: {float(water_imp):.2e}"
    # Non-vacuous: implicit is at least ~20x more conservative than advective
    # (the real ratio is far larger), so the test FAILS if the fix is reverted.
    assert float(mse_imp) < float(mse_adv) / 20.0, (
        f"implicit MSE ({float(mse_imp):.2e}) not materially better than "
        f"advective ({float(mse_adv):.2e}) — vacuous")


def test_updraft_velocity_integrates_surface_up_not_top_down():
    """DIRECTION of the signed-buoyancy integral, tested on the extracted
    ``updraft_velocity_from_buoyancy`` core with a CONTROLLED input (the emergent
    tendency profile can't isolate this — compensating subsidence warms the whole
    lower troposphere).  A single positive buoyancy SPIKE at a mid level must
    produce updraft velocity ABOVE the spike (surface-up integration carries the
    kinetic energy upward) and ~the floor BELOW it; a wrong TOP-DOWN integral does
    the opposite, so this pins the direction (codex #824 review)."""
    nlev = 20
    dz = jnp.full((1, nlev), _DZ)
    spike = nlev // 2                                  # index; 0 = top, -1 = surface
    B = jnp.zeros((1, nlev)).at[0, spike].set(0.02)    # +buoyancy spike, one level
    w = updraft_velocity_from_buoyancy(B, dz, w_u_min=0.1, w_u_max=50.0)[0]
    # ABOVE the spike (smaller index → higher altitude): velocity carried upward.
    assert float(jnp.max(w[:spike])) > 1.0, "no updraft velocity ABOVE the spike"
    # BELOW the spike (larger index → toward the surface): still at the floor
    # (the plume has not yet reached the buoyant layer).  Top-down would light up.
    assert float(jnp.max(w[spike + 1:])) < 0.2, (
        "updraft velocity present BELOW the spike — top-down integration?")


def test_updraft_velocity_detrains_above_neutral_buoyancy():
    """DETRAINMENT.  Buoyant below (surface half), stable above (a capping
    inversion): the signed integral makes w_u rise through the buoyant layer then
    DECREASE back toward the floor above the level of neutral buoyancy — it does
    NOT plateau at its peak (the #824 fix; the old positive-only clip froze it)."""
    nlev = 24
    dz = jnp.full((1, nlev), _DZ)
    lower = nlev // 2                                  # boundary index (the LNB)
    # index 0..lower-1 = upper half (stable, B<0); lower..nlev-1 = lower half (B>0).
    B = jnp.concatenate([
        jnp.full((1, lower), -0.01),
        jnp.full((1, nlev - lower), 0.01),
    ], axis=1)
    w = updraft_velocity_from_buoyancy(B, dz, w_u_min=0.1, w_u_max=50.0)[0]
    peak = float(jnp.max(w))
    # Peak at/above the LNB (top of the buoyant layer), i.e. a low-ish index.
    assert int(jnp.argmax(w)) <= lower + 1
    # w decays from the peak toward the model top (detrainment), back near floor.
    assert float(w[0]) < 0.3 * peak, "updraft does not detrain aloft (plateau?)"


def test_updraft_velocity_capped_and_ad_safe():
    """w_u is bounded by w_u_max even for a huge-CAPE column, and the profile is
    reverse-mode differentiable (finite, non-trivial gradient)."""
    nlev = 24  # deep enough that 2*integral (=1200) exceeds w_u_max^2 (=900): cap ACTIVE
    dz = jnp.full((2, nlev), _DZ)
    B = jnp.full((2, nlev), 0.05)                      # enormous, uniform buoyancy
    w = updraft_velocity_from_buoyancy(B, dz, w_u_min=0.1, w_u_max=30.0)
    assert float(jnp.max(w)) <= 30.0 + 1e-6, "w_u exceeds the w_u_max cap"
    assert float(jnp.max(w)) > 29.0, "w_u_max cap not actually exercised (vacuous)"

    def loss(b):
        return jnp.sum(updraft_velocity_from_buoyancy(b, dz, 0.1, 30.0) ** 2)

    g = jax.grad(loss)(B)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_edmf_multistep_stable_large_dt():
    """The day-5 blowup regression: integrate a pure-EDMF column for 400 steps at
    the production physics dt (600 s) with NO per-step clipping — so an unstable
    subsidence would surface as a moisture/temperature runaway rather than being
    masked — and require the state to stay finite and physically bounded, with
    the convective heating bounded below a runaway ceiling throughout.  The old
    non-conservative + non-detraining scheme drifted to a non-finite state."""
    T, q_v, p_full, p_half = _column(ncol=2, nlev=24)
    cfg = ConvectiveEDMFConfig()
    a_u = jnp.full(T.shape[0], cfg.a_u_init)
    dt = 600.0

    @jax.jit
    def _step(carry, _):
        T, q_v, a_u = carry
        out, a_u_new = edmf_convection(T, q_v, p_full, p_half, a_u, dt=dt, config=cfg)
        T_new = T + dt * out.dT_dt
        q_v_new = q_v + dt * out.dq_v_dt              # NO clip — let instability show
        return (T_new, q_v_new, a_u_new), jnp.max(jnp.abs(out.dT_dt))

    (T_f, q_f, _), max_tend = jax.lax.scan(
        _step, (T, q_v, a_u), None, length=400)

    assert bool(jnp.all(jnp.isfinite(T_f))), "temperature went non-finite (#824 blowup)"
    assert bool(jnp.all(jnp.isfinite(q_f))), "moisture went non-finite"
    assert bool(jnp.all(jnp.isfinite(max_tend))), "tendency went non-finite mid-run"
    assert bool(jnp.all((T_f > 150.0) & (T_f < 400.0))), f"temperature unphysical: {T_f}"
    # Bounded convective heating throughout: a physical ceiling (0.05 K/s ≈
    # 180 K/hr) far above any real convective tendency (~1e-4 K/s) but far below
    # a blowup.  NOT a monotone-decrease assertion — a healthy damped scheme can
    # transiently grow after a thermodynamic adjustment (codex #824 review).
    assert float(jnp.max(max_tend)) < 0.05, (
        f"convective heating exceeded the runaway ceiling: {float(jnp.max(max_tend)):.3e} K/s")


def test_edmf_advective_still_selectable_and_raises_on_unknown():
    """The legacy advective solve stays reachable via config, and an unknown
    selector RAISES at the kernel (dispatch-hardening, CLAUDE.md) rather than
    silently running default physics."""
    T, q_v, p_full, p_half = _column(ncol=1, nlev=12)
    a_u = jnp.full(1, 0.1)
    out, _ = edmf_convection(
        T, q_v, p_full, p_half, a_u, dt=600.0,
        config=ConvectiveEDMFConfig(subsidence_solve="advective"))
    assert jnp.all(jnp.isfinite(out.dT_dt))
    with pytest.raises(ValueError, match="unknown subsidence_solve"):
        edmf_convection(
            T, q_v, p_full, p_half, a_u, dt=600.0,
            config=ConvectiveEDMFConfig(subsidence_solve="bogus"))


def test_edmf_implicit_is_differentiable():
    """jax.grad through the default (implicit tridiagonal) EDMF path is finite
    and non-trivial — the conservative solve preserves end-to-end AD."""
    T, q_v, p_full, p_half = _column(ncol=2, nlev=12)
    cfg = ConvectiveEDMFConfig()
    a_u = jnp.full(2, cfg.a_u_init)

    def loss(T_in):
        out, _ = edmf_convection(T_in, q_v, p_full, p_half, a_u, dt=600.0, config=cfg)
        return jnp.sum(out.dT_dt ** 2)

    g = jax.grad(loss)(T)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.max(jnp.abs(g))) > 0.0
