"""Gates for the Simmons-Burridge vertical transport + frictional heating.

Motivating measurement (2026-08-10, scripts/validate/aimip_t_budget.py):
the legacy upwind advective-form vertical transport carries a systematic
-36 W/m^2 (-0.32..-0.47 K/day) mass-weighted global-mean temperature sink
at T63L8 — the dominant term of the dycore's -0.48 K/day zero-physics
cooling. These tests gate the fix:

1. the SB centered form's flux-form telescoping identity (exact algebra);
2. the mass-weighted T sink of the SB form is far below the upwind form's,
   and the per-column conservation pairing holds exactly (upwind violates
   it — non-vacuity);
3. frictional heating returns EXACTLY the KE the hyperdiffusion removes
   (global energy identity, fp64);
4. dispatch hardening: unknown scheme raises; sb_centered + hybrid raises;
5. differentiability: jax.grad flows through the new path.

All fp64 (JAX_ENABLE_X64 required — numerics/conservation policy).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig,
    _compute_sigma_dot_gaussian,
    _vertical_advection_sigma_gaussian,
    _vertical_advection_sigma_sb,
    isothermal_rest_state_spectral,
    spectral_pe_tendencies,
)
from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis,
    sh_synthesis,
    sh_synthesis_3d,
)
from legoesm.grids.vertical import create_sigma_coordinate

NLEV = 8


@pytest.fixture(scope="module")
def grid():
    return create_gaussian_grid(21, dealiasing="quadratic")


@pytest.fixture(scope="module")
def sigma():
    return create_sigma_coordinate(NLEV, sigma_top=0.05)


def _balanced_state(grid, sigma, seed=0):
    """A smooth, dynamically active state: isothermal rest + large-scale
    vor/div/T/lnps perturbations (band-limited, so dealiasing is inert)."""
    state = isothermal_rest_state_spectral(grid, sigma)
    key = jax.random.PRNGKey(seed)
    lat = np.asarray(grid.lat)[:, None]
    lon = np.asarray(grid.lon)[None, :]
    k1, k2, k3 = jax.random.split(key, 3)

    def _field(k, amp):
        a1, a2, a3 = jax.random.uniform(k, (3,), minval=-1.0, maxval=1.0)
        pat = (a1 * np.cos(2 * lon) * np.cos(lat) ** 2
               + a2 * np.sin(3 * lon) * np.cos(lat) ** 3
               + a3 * np.cos(lat) ** 2)
        return amp * jnp.asarray(pat)

    vor_g = jnp.stack([_field(jax.random.fold_in(k1, i), 2e-5)
                       for i in range(NLEV)], axis=-1)
    div_g = jnp.stack([_field(jax.random.fold_in(k2, i), 5e-6)
                       for i in range(NLEV)], axis=-1)
    T_pert = jnp.stack([_field(jax.random.fold_in(k3, i), 8.0)
                        for i in range(NLEV)], axis=-1)

    from legoesm.grids.gaussian import sh_analysis_3d
    return state._replace(
        vor_hat=state.vor_hat.replace(
            data=state.vor_hat.data + sh_analysis_3d(grid, vor_g)),
        div_hat=state.div_hat.replace(
            data=state.div_hat.data + sh_analysis_3d(grid, div_g)),
        T_hat=state.T_hat.replace(
            data=state.T_hat.data + sh_analysis_3d(grid, T_pert)),
        lnps_hat=state.lnps_hat.replace(
            data=sh_analysis(grid, jnp.log(1.0e5 * (1.0 + 0.02 * jnp.asarray(
                np.cos(2 * lon) * np.cos(lat) ** 2))))),
    )


def _mass_weighted_mean(grid, sigma, ps, X):
    w = np.asarray(grid.grid_area)
    dsig = np.asarray(sigma.dsigma)
    num = (w[:, :, None] * np.asarray(ps)[:, :, None]
           * dsig[None, None, :] * np.asarray(X)).sum()
    den = (w[:, :, None] * np.asarray(ps)[:, :, None]
           * dsig[None, None, :]).sum()
    return float(num / den)


# ---------------------------------------------------------------- 1. algebra

def test_sb_form_equals_flux_form_minus_field_times_sigmadot_divergence(
        grid, sigma):
    """SB centered advective form == flux-form divergence of the centered
    interface flux MINUS field * d(sigma_dot)/d(sigma). Exact identity,
    which is what makes the mass-weighted integral telescope against
    continuity."""
    key = jax.random.PRNGKey(1)
    shape = (grid.n_lat, grid.n_lon, NLEV)
    f = jax.random.normal(key, shape, dtype=jnp.float64)
    div = 1e-5 * jax.random.normal(jax.random.fold_in(key, 1), shape,
                                   dtype=jnp.float64)
    sigma_dot, _ = _compute_sigma_dot_gaussian(div, sigma)

    adv = _vertical_advection_sigma_sb(f, sigma_dot, sigma)

    # Flux form: F_j = sigma_dot_j * (f_k + f_{k+1})/2 at interior
    # interfaces, zero at top/bottom (sigma_dot BCs).
    f_half = 0.5 * (f[..., :-1] + f[..., 1:])
    F = sigma_dot[..., 1:-1] * f_half
    pad = ((0, 0),) * (F.ndim - 1)
    F_full = jnp.pad(F, (*pad, (1, 1)))            # (..., nlev+1)
    flux_div = jnp.diff(F_full, axis=-1) / sigma.dsigma
    dsdot = jnp.diff(sigma_dot, axis=-1) / sigma.dsigma
    expected = -(flux_div - f * dsdot)

    np.testing.assert_allclose(np.asarray(adv), np.asarray(expected),
                               rtol=1e-12, atol=1e-12)


# ------------------------------------------------- 2. the sink, side by side

def test_sb_mass_weighted_T_sink_far_below_upwind(grid, sigma):
    """On a balanced active state, the upwind form's mass-weighted global
    T tendency is systematically nonzero while the SB form's pairs with
    continuity. Assert the SB |mean| is at most half the upwind |mean|
    (measured at T63L8 on ERA5 the ratio is ~30x; T21 synthetic is a
    weaker but still decisive contrast)."""
    state = _balanced_state(grid, sigma)
    T = sh_synthesis_3d(grid, state.T_hat.data)
    div = sh_synthesis_3d(grid, state.div_hat.data)
    ps = jnp.exp(sh_synthesis(grid, state.lnps_hat.data))
    sigma_dot, _ = _compute_sigma_dot_gaussian(div, sigma)

    up = _vertical_advection_sigma_gaussian(T, sigma_dot, sigma)
    sb = _vertical_advection_sigma_sb(T, sigma_dot, sigma)

    m_up = abs(_mass_weighted_mean(grid, sigma, ps, up))
    m_sb = abs(_mass_weighted_mean(grid, sigma, ps, sb))
    assert m_up > 0.0
    assert m_sb < 0.5 * m_up, (m_sb, m_up)


def test_sb_column_telescoping_identity_upwind_violates_it(grid, sigma):
    """THE conservation pairing, asserted exactly: for the SB form,

        sum_k dsigma_k * [ adv_k - f_k * (d sigma_dot/d sigma)_k ] = 0

    per column (the interface fluxes telescope and the boundary fluxes are
    zero), which is precisely the term continuity provides — so the
    mass-weighted transport budget closes. NON-VACUITY partner: the upwind
    form violates the same identity by a finite residual on the same
    inputs. (A synthetic multi-step drift comparison is NOT used here: on a
    violently unbalanced random state the total drift is adjustment-
    dominated and cannot discriminate the schemes — measured 0.381 vs
    0.361 K; the end-to-end acceptance gate is the ERA5-IC --global-t
    probe, recorded in the dev note.)"""
    key = jax.random.PRNGKey(7)
    shape = (grid.n_lat, grid.n_lon, NLEV)
    f = jax.random.normal(key, shape, dtype=jnp.float64)
    div = 1e-5 * jax.random.normal(jax.random.fold_in(key, 1), shape,
                                   dtype=jnp.float64)
    sigma_dot, _ = _compute_sigma_dot_gaussian(div, sigma)
    dsdot = jnp.diff(sigma_dot, axis=-1) / sigma.dsigma

    sb = _vertical_advection_sigma_sb(f, sigma_dot, sigma)
    # adv - f*dsdot = -(flux divergence), whose dsigma-weighted column sum
    # telescopes to the (zero) boundary fluxes.
    resid_sb = ((sb - f * dsdot) * sigma.dsigma).sum(axis=-1)
    np.testing.assert_allclose(np.asarray(resid_sb), 0.0, atol=1e-12)

    up = _vertical_advection_sigma_gaussian(f, sigma_dot, sigma)
    resid_up = ((up - f * dsdot) * sigma.dsigma).sum(axis=-1)
    assert float(jnp.abs(resid_up).max()) > 1e-8, (
        "upwind unexpectedly satisfies the identity — test is vacuous")


# ---------------------------------------------- 3. frictional-heating identity

def test_frictional_heating_returns_exactly_the_hyperdiff_KE(grid, sigma):
    """Global energy identity: cp * mass-weighted integral of the frictional
    heating == -(mass-weighted integral of u*du_hd + v*dv_hd). Computed by
    differencing the T tendency with the flag on vs off (everything else
    identical), so the test pins the SHIPPED term, not a re-derivation."""
    state = _balanced_state(grid, sigma, seed=3)
    # Add power ABOVE the 2/3 dealias cutoff (l > 14 at T21): the delivered
    # vor/div tendencies are dealiased, so those modes lose no KE — heating
    # computed from the UNMASKED hyperdiff stack would manufacture energy
    # here and break the identity (codex P1; this line makes the test fail
    # against that defect).
    hi = jnp.where((jnp.asarray(grid.ls) >= 18)[:, None],
                   1e-5, 0.0).astype(state.vor_hat.data.dtype)
    state = state._replace(
        vor_hat=state.vor_hat.replace(data=state.vor_hat.data + hi))
    cfg_off = SpectralPEConfig(hyperdiff_coeff=1e16, hyperdiff_order=2,
                               frictional_heating=False)
    cfg_on = cfg_off._replace(frictional_heating=True)

    t_off = spectral_pe_tendencies(state, grid, sigma, cfg_off, None)
    t_on = spectral_pe_tendencies(state, grid, sigma, cfg_on, None)

    heat = sh_synthesis_3d(grid, t_on.T_hat.data - t_off.T_hat.data)
    ps = jnp.exp(sh_synthesis(grid, state.lnps_hat.data))

    # KE tendency of the hyperdiffusion alone, from the vor/div tendency
    # difference against a zero-hyperdiff config.
    cfg_none = cfg_off._replace(hyperdiff_coeff=0.0)
    t_none = spectral_pe_tendencies(state, grid, sigma, cfg_none, None)
    from legoesm.grids.gaussian import uv_from_vordiv_3d
    du_cos, dv_cos = uv_from_vordiv_3d(
        grid, t_off.vor_hat.data - t_none.vor_hat.data,
        t_off.div_hat.data - t_none.div_hat.data)
    u_cos, v_cos = uv_from_vordiv_3d(grid, state.vor_hat.data,
                                     state.div_hat.data)
    cos2 = np.clip(np.asarray(grid.cos_lat), 1e-8, None)[:, None, None] ** 2
    dKE = np.asarray(u_cos * du_cos + v_cos * dv_cos) / cos2

    w = np.asarray(grid.grid_area)[:, :, None] * np.asarray(ps)[:, :, None] \
        * np.asarray(sigma.dsigma)[None, None, :]
    lhs = float((w * np.asarray(heat)).sum()) * constants.c_pd
    rhs = -float((w * dKE).sum())
    assert rhs > 0.0, "hyperdiff must remove KE on an active state"
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6)


# ------------------------------------------------------- 4. dispatch hardening

def test_unknown_scheme_raises(grid, sigma):
    state = _balanced_state(grid, sigma)
    cfg = SpectralPEConfig(vertical_advection_scheme="centred")  # typo
    with pytest.raises(ValueError, match="vertical_advection_scheme"):
        spectral_pe_tendencies(state, grid, sigma, cfg, None)


def test_hybrid_sb_telescoping_identity(grid):
    """Same conservation pairing on the HYBRID coordinate: for the hybrid SB
    form, sum_k dp_k * [adv_k - f_k * (d mdot/d p)_k] telescopes to the
    (zero) boundary mass fluxes. Upwind violates it (non-vacuity)."""
    from legoesm.grids.vertical import (
        compute_mass_flux_hybrid,
        create_hybrid_coordinate,
        dp_from_hybrid,
        vertical_advection_hybrid,
        vertical_advection_hybrid_sb,
    )

    B_half = jnp.linspace(0.05, 1.0, NLEV + 1, dtype=jnp.float64)
    hybrid = create_hybrid_coordinate(
        NLEV, jnp.zeros(NLEV + 1, dtype=jnp.float64), B_half)

    key = jax.random.PRNGKey(11)
    shape = (grid.n_lat, grid.n_lon, NLEV)
    f = jax.random.normal(key, shape, dtype=jnp.float64)
    div = 1e-5 * jax.random.normal(jax.random.fold_in(key, 1), shape,
                                   dtype=jnp.float64)
    ps = 1.0e5 * (1.0 + 0.02 * jax.random.normal(
        jax.random.fold_in(key, 2), shape[:2], dtype=jnp.float64))
    mass_flux, _ = compute_mass_flux_hybrid(div, ps, hybrid)
    dp = dp_from_hybrid(hybrid, ps)
    dmdot = jnp.diff(mass_flux, axis=-1) / dp

    sb = vertical_advection_hybrid_sb(f, mass_flux, ps, hybrid)
    resid_sb = ((sb - f * dmdot) * dp).sum(axis=-1)
    # tolerance relative to the flux scale (dp ~ 1e4 Pa, mdot ~ 1e-1 Pa/s)
    np.testing.assert_allclose(np.asarray(resid_sb), 0.0, atol=1e-8)

    up = vertical_advection_hybrid(f, mass_flux, ps, hybrid)
    resid_up = ((up - f * dmdot) * dp).sum(axis=-1)
    assert float(jnp.abs(resid_up).max()) > 1e-4, (
        "upwind unexpectedly satisfies the identity — test is vacuous")


def test_hybrid_sb_runs_through_full_tendencies(grid):
    """End-to-end: spectral_pe_tendencies on the hybrid coordinate with
    sb_centered produces finite tendencies (the dispatch reaches the hybrid
    SB helper, not a guard)."""
    from legoesm.grids.vertical import create_hybrid_coordinate

    B_half = jnp.linspace(0.05, 1.0, NLEV + 1, dtype=jnp.float64)
    hybrid = create_hybrid_coordinate(
        NLEV, jnp.zeros(NLEV + 1, dtype=jnp.float64), B_half)
    state = _balanced_state(grid, create_sigma_coordinate(NLEV, sigma_top=0.05))
    cfg = SpectralPEConfig(vertical_advection_scheme="sb_centered")
    t = spectral_pe_tendencies(state, grid, hybrid, cfg, None)
    for leaf in (t.vor_hat.data, t.div_hat.data, t.T_hat.data,
                 t.lnps_hat.data):
        assert bool(jnp.all(jnp.isfinite(jnp.abs(leaf))))


# ------------------------------------------------------- 5. differentiability

def test_grad_flows_through_sb_and_frictional_heating(grid, sigma):
    state = _balanced_state(grid, sigma, seed=4)
    cfg = SpectralPEConfig(hyperdiff_coeff=1e16, hyperdiff_order=2,
                           vertical_advection_scheme="sb_centered",
                           frictional_heating=True)

    def loss(T_hat_data):
        s = state._replace(T_hat=state.T_hat.replace(data=T_hat_data))
        t = spectral_pe_tendencies(s, grid, sigma, cfg, None)
        return jnp.sum(jnp.abs(t.T_hat.data) ** 2)

    g = jax.grad(loss)(state.T_hat.data)
    assert bool(jnp.all(jnp.isfinite(g.real))) and float(
        jnp.abs(g).max()) > 0.0
