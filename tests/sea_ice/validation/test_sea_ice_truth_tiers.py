"""Truth-tier validation for the sea-ice group (physics-validator).

Validates, per tier and per scheme, the in-scope sea-ice physics:
  - rheology / ice_strength (Hibler 1979) + VP constitutive law
  - EVP / mEVP momentum solvers (Hunke & Dukowicz 1997 / Kimmritz 2015)
  - ITD ``linear_remap`` (simple) + ``lipscomb_2001_remap``
  - delta-Eddington / Maykut-Untersteiner albedo feedback + SW closure

Tiers:
  1. units  2. signs  3. conservation  4. differentiability  5. idealized

Run: JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/sea_ice/validation/test_sea_ice_truth_tiers.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ice import dynamics as D
from legoesm.ice import itd as I
from legoesm.ice import rheology as R
from legoesm.ice import shortwave as SW
from legoesm.ice.config import SeaIceConfig


# ============================================================================
# rheology / ice_strength
# ============================================================================

def test_ice_strength_hibler_form_and_signs():
    """P = P* h exp(-C(1-A)): zero at h=0, increases with h, decreases as A drops."""
    P_star, C = 27500.0, 20.0
    # h = 0 -> zero strength
    assert float(R.ice_strength(jnp.array([0.0]), jnp.array([1.0]), P_star, C)[0]) == 0.0
    # At A = 1, P = P* h exactly
    h = jnp.array([0.5, 1.0, 2.0])
    P = R.ice_strength(h, jnp.ones_like(h), P_star, C)
    np.testing.assert_allclose(np.asarray(P), P_star * np.asarray(h), rtol=1e-12)
    # Monotone in h
    assert float(P[0]) < float(P[1]) < float(P[2])
    # Decreasing concentration weakens ice (exp decay)
    P_hi = float(R.ice_strength(jnp.array([1.0]), jnp.array([1.0]), P_star, C)[0])
    P_lo = float(R.ice_strength(jnp.array([1.0]), jnp.array([0.5]), P_star, C)[0])
    assert P_lo < P_hi
    np.testing.assert_allclose(P_lo, P_star * np.exp(-C * 0.5), rtol=1e-12)


def test_vp_stress_signs_and_rest_state():
    """Convergence -> compressive normal stress; shear -> co-signed sigma_12;
    zero strain -> isotropic -P/2 (the EVP rest target)."""
    P = jnp.array([1.0e4])
    e = 2.0
    d = 1.0e-4
    # Pure convergence (eps_11 = eps_22 < 0)
    s11, s22, s12 = R.vp_stress(
        jnp.array([-d]), jnp.array([-d]), jnp.array([0.0]),
        P, R.delta_deformation(jnp.array([-d]), jnp.array([-d]), jnp.array([0.0]), e), e,
    )
    assert float(s11[0]) < 0.0 and float(s22[0]) < 0.0
    # Pure shear -> sigma_12 same sign as eps_12
    s11b, s22b, s12b = R.vp_stress(
        jnp.array([0.0]), jnp.array([0.0]), jnp.array([d]),
        P, R.delta_deformation(jnp.array([0.0]), jnp.array([0.0]), jnp.array([d]), e), e,
    )
    assert float(s12b[0]) > 0.0
    # Rest state
    z = jnp.array([0.0])
    s11r, _, s12r = R.vp_stress(z, z, z, P, R.delta_deformation(z, z, z, e), e)
    np.testing.assert_allclose(float(s11r[0]), -float(P[0]) / 2.0, rtol=1e-12)
    assert float(s12r[0]) == 0.0


def test_delta_deformation_grad_finite_at_rest():
    """grad of Delta at zero strain must be finite (no 0*inf=NaN from sqrt'(0))."""
    def f(args):
        a, b, c = args
        return jnp.sum(R.delta_deformation(a, b, c, 2.0))

    z = jnp.zeros(4)
    g = jax.grad(f)((z, z, z))
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))


def test_mevp_stress_update_converges_to_vp():
    """Tier 5: fixed strain, iterate mEVP relaxation -> VP target."""
    P = jnp.array([2.0e4])
    e = 2.0
    e11, e22, e12 = jnp.array([-1e-5]), jnp.array([3e-6]), jnp.array([2e-6])
    Delta = R.delta_deformation(e11, e22, e12, e)
    vp = R.vp_stress(e11, e22, e12, P, Delta, e)
    s = (jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]))
    for _ in range(3000):
        s = R.mevp_stress_update(*s, e11, e22, e12, P, e, 50.0)
    for got, want in zip(s, vp):
        np.testing.assert_allclose(float(got[0]), float(want[0]), rtol=1e-3)


def test_mevp_stress_update_rejects_bad_alpha():
    z = jnp.array([0.0])
    for bad in (0.5, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            R.mevp_stress_update(z, z, z, z, z, z, jnp.array([1e4]), 2.0, bad)


# ============================================================================
# EVP / mEVP momentum solvers
# ============================================================================

def _latlon_dyn_fields():
    grid = create_latlon_grid(n_lat=12, n_lon=24)
    shp = D._grid_coriolis(grid).shape
    z = jnp.zeros(shp)
    return grid, shp, z


def test_evp_freedrift_limit_matches_zubov():
    """Tier 5: uniform field (no stress divergence) -> air/ocean drag balance.

    Steady ice velocity must equal the analytic 1-D free-drift fixed point
    ui = k Ua/(1+k), k = sqrt(rho_air Cai / (rho_oc Coi)).
    """
    grid, shp, z = _latlon_dyn_fields()
    cfg = SeaIceConfig()
    h = jnp.full(shp, 1.0)
    A = jnp.ones(shp)
    wind_u = jnp.full(shp, 10.0)
    u, v, s11, s22, s12 = z, z, z, z, z
    for _ in range(8):  # several dynamic steps to reach steady drift
        u, v, s11, s22, s12 = D.evp_solver(
            u, v, s11, s22, s12, h, A, wind_u, z, z, z, grid, dt=3600.0,
            N_evp=120, differentiable=False,
        )
    u = np.asarray(u)
    assert np.all(np.isfinite(u))
    k = np.sqrt(constants.rho_air * cfg.drag_atm
                / (constants.rho_ocean * cfg.drag_ocean))
    ui_analytic = k * 10.0 / (1.0 + k)
    interior = u[4:8]
    assert np.all(interior > 0.0)  # downwind drift
    np.testing.assert_allclose(np.mean(interior), ui_analytic, rtol=0.05)


def test_evp_steady_velocity_mass_independent():
    """m_ice = rho_ice*h (no double conc weighting): steady drift independent of h."""
    grid, shp, z = _latlon_dyn_fields()
    wind_u = jnp.full(shp, 10.0)
    A = jnp.ones(shp)
    steady = {}
    for hval in (0.5, 1.0, 4.0):
        u, v, s11, s22, s12 = z, z, z, z, z
        h = jnp.full(shp, hval)
        for _ in range(8):
            u, v, s11, s22, s12 = D.evp_solver(
                u, v, s11, s22, s12, h, A, wind_u, z, z, z, grid, dt=3600.0,
                N_evp=120, differentiable=False,
            )
        steady[hval] = float(np.mean(np.asarray(u)[4:8]))
    vals = list(steady.values())
    assert max(vals) - min(vals) < 0.02 * np.mean(vals)


def test_mevp_freedrift_finite_and_downwind():
    grid, shp, z = _latlon_dyn_fields()
    h = jnp.full(shp, 1.0)
    A = jnp.ones(shp)
    wind_u = jnp.full(shp, 10.0)
    u, *_ = D.mevp_solver(
        z, z, z, z, z, h, A, wind_u, z, z, z, grid, dt=3600.0,
        N_mevp=200, alpha_mevp=500.0, beta_mevp=500.0, differentiable=False,
    )
    u = np.asarray(u)
    assert np.all(np.isfinite(u))
    assert np.all(u[4:8] > 0.0)


@pytest.mark.parametrize("solver,kw", [
    (D.evp_solver, dict(N_evp=30)),
    (D.mevp_solver, dict(N_mevp=30, alpha_mevp=500.0, beta_mevp=500.0)),
])
def test_solver_ad_finite_nonzero(solver, kw):
    """Tier 4: grad through the scan path w.r.t. P_star and wind is finite,
    and the wind sensitivity is non-zero (physics implies it)."""
    grid, shp, z = _latlon_dyn_fields()
    h = jnp.full(shp, 1.0)
    A = jnp.ones(shp)
    wind_u = jnp.full(shp, 10.0)

    def loss_P(P_star):
        u, v, *_ = solver(z, z, z, z, z, h, A, wind_u, z, z, z, grid,
                          dt=3600.0, P_star=P_star, differentiable=True, **kw)
        return jnp.sum(u ** 2 + v ** 2)

    gP = float(jax.grad(loss_P)(jnp.asarray(2.75e4)))
    assert np.isfinite(gP)

    def loss_w(wu):
        u, v, *_ = solver(z, z, z, z, z, h, A, wu, z, z, z, grid,
                          dt=3600.0, differentiable=True, **kw)
        return jnp.sum(u ** 2)

    gw = jax.grad(loss_w)(wind_u)
    assert np.all(np.isfinite(np.asarray(gw)))
    assert float(jnp.sum(jnp.abs(gw))) > 0.0


def test_evp_jit_matches_eager():
    import functools
    grid, shp, z = _latlon_dyn_fields()
    h = jnp.full(shp, 1.0)
    A = jnp.ones(shp)
    wind_u = jnp.full(shp, 10.0)
    eager = D.evp_solver(z, z, z, z, z, h, A, wind_u, z, z, z, grid,
                         dt=3600.0, N_evp=20, differentiable=True)[0]
    jitted = jax.jit(functools.partial(
        D.evp_solver, grid=grid, dt=3600.0, N_evp=20, differentiable=True))
    je = jitted(z, z, z, z, z, h, A, wind_u, z, z, z)[0]
    np.testing.assert_allclose(np.asarray(eager), np.asarray(je), rtol=1e-9, atol=1e-12)


def test_strain_rates_uniform_field_zero_interior():
    """Solid-body translation -> zero interior strain on lat-lon + cubed-sphere."""
    grid = create_latlon_grid(n_lat=20, n_lon=40)
    shp = (grid.lat.shape[0], 40)
    e11, e22, e12 = R.strain_rates(jnp.full(shp, 0.5), jnp.zeros(shp), grid)
    mid = slice(8, 12)
    assert np.max(np.abs(np.asarray(e11)[mid])) < 1e-9
    assert np.max(np.abs(np.asarray(e22)[mid])) < 1e-9

    cs = create_cubed_sphere(n=8)
    e11c, e22c, e12c = R.strain_rates(
        jnp.full((6, 8, 8), 0.3), jnp.full((6, 8, 8), -0.2), cs)
    assert np.all(np.isfinite(np.asarray(e11c)))
    assert np.all(np.isfinite(np.asarray(e12c)))


# ============================================================================
# ITD remap
# ============================================================================

def test_linear_remap_volume_conserved_always():
    """Tier 3: ice VOLUME (= sum h*a) conserved exactly even when clamp fires."""
    n_cat = 5
    a = jnp.array([[0.2, 0.3, 0.1, 0.05, 0.0]])
    h_old = jnp.array([[0.3, 1.0, 2.0, 3.0, 0.0]])
    h_new = jnp.array([[0.05, 1.6, 2.2, 3.5, 0.0]])  # cat1 grows past its bound
    h_r, a_r = I.linear_remap(h_old, a, h_new, a, n_cat)
    V_in = float(jnp.sum(h_new * a))
    V_out = float(jnp.sum(h_r * a_r))
    np.testing.assert_allclose(V_out, V_in, rtol=1e-9, atol=1e-12)


def test_linear_remap_exact_area_when_no_clamp():
    """Area + volume both exact when growth stays within category bins."""
    n_cat = 5
    a = jnp.array([[0.2, 0.3, 0.1, 0.05, 0.0]])
    h_old = jnp.array([[0.3, 1.0, 2.0, 3.0, 0.0]])
    h_new = jnp.array([[0.35, 1.05, 2.05, 3.1, 0.0]])  # all inside bins
    h_r, a_r = I.linear_remap(h_old, a, h_new, a, n_cat)
    np.testing.assert_allclose(float(jnp.sum(a_r)), float(jnp.sum(a)), atol=1e-12)
    np.testing.assert_allclose(
        float(jnp.sum(h_r * a_r)), float(jnp.sum(h_new * a)), atol=1e-12)


def test_linear_remap_area_not_conserved_when_clamp_fires_representative():
    """CHARACTERIZATION of a documented approximation: the 'simple' remap's
    volume-conserving rescale does NOT conserve total ice area when the
    thickness clamp fires (volume IS conserved).

    This REPRESENTATIVE case (cat-1 grows past its upper bound) gains ~5.8%
    area.  NOTE: area non-conservation is not sign-bounded in general — other
    low-area multi-category inputs can LOSE area substantially while still
    conserving volume; this test characterizes one specific clamp case, it does
    NOT assert a global area bound (codex caveat).  The general invariant is
    VOLUME conservation only.

    This is why multi-category configs WITH tracers require ``itd_remap=
    'lipscomb2001'`` (area-conserving), and why ``_step_dynamic`` re-applies
    ``_cap_multicat_concentration``.  See itd.linear_remap docstring + the
    sea_ice.py:310 gating.
    """
    n_cat = 5
    a = jnp.array([[0.2, 0.3, 0.1, 0.05, 0.0]])
    h_old = jnp.array([[0.3, 1.0, 2.0, 3.0, 0.0]])
    h_new = jnp.array([[0.05, 1.6, 2.2, 3.5, 0.0]])
    h_r, a_r = I.linear_remap(h_old, a, h_new, a, n_cat)
    A_in, A_out = float(jnp.sum(a)), float(jnp.sum(a_r))
    # Volume exact (the general invariant); area NOT conserved in this case.
    np.testing.assert_allclose(
        float(jnp.sum(h_r * a_r)), float(jnp.sum(h_new * a)), atol=1e-9)
    assert abs(A_out - A_in) > 1e-3  # area is not conserved (documented)
    # The downstream sum-cap is a NO-OP here (A_out < 1) so the inflation is
    # not corrected — exactly why the tracer-safe path requires lipscomb2001.
    assert A_out < 1.0 and A_out > A_in


def test_lipscomb_conserves_area_volume_salt():
    """Tier 3 (the tracer-safe scheme): area + volume + salt to machine precision."""
    n_cat = 5
    a = jnp.array([[0.2, 0.3, 0.1, 0.05, 0.0]])
    h_old = jnp.array([[0.3, 1.0, 2.0, 3.0, 0.0]])
    h_new = jnp.array([[0.35, 1.2, 2.1, 3.2, 0.0]])
    T_new = jnp.full((1, 5), 263.0)
    S_new = jnp.full((1, 5), 4.0)
    out = I.lipscomb_2001_remap(
        h_old, a, h_new, a, n_cat, dt=3600.0, T_new=T_new, S_new=S_new)
    V_in, V_out = float(jnp.sum(h_new * a)), float(jnp.sum(out["h"] * out["a"]))
    A_in, A_out = float(jnp.sum(a)), float(jnp.sum(out["a"]))
    salt_in = float(jnp.sum(S_new * h_new * a))
    salt_out = float(jnp.sum(out["S"] * out["h"] * out["a"]))
    np.testing.assert_allclose(V_out, V_in, rtol=1e-6)
    np.testing.assert_allclose(A_out, A_in, rtol=1e-6)
    np.testing.assert_allclose(salt_out, salt_in, rtol=1e-5)


def test_linear_remap_temperature_clamp_to_surface_melt():
    n_cat = 5
    a = jnp.array([[0.2, 0.3, 0.1, 0.05, 0.0]])
    h_old = jnp.array([[0.3, 1.0, 2.0, 3.0, 0.0]])
    h_new = jnp.array([[0.05, 1.6, 2.2, 3.5, 0.0]])
    T_new = jnp.full((1, 5), 290.0)  # above melt -> must clamp at T_freeze
    _, _, T_r = I.linear_remap(h_old, a, h_new, a, n_cat, T_new=T_new,
                               T_max=constants.T_freeze)
    assert float(jnp.max(T_r)) <= constants.T_freeze + 1e-9


def test_remap_ad_finite():
    """Tier 4: grad through linear_remap w.r.t. h_new is finite."""
    n_cat = 5
    a = jnp.array([[0.2, 0.3, 0.1, 0.05, 0.0]])
    h_old = jnp.array([[0.3, 1.0, 2.0, 3.0, 0.0]])

    def f(h_new):
        h_r, a_r = I.linear_remap(h_old, a, h_new, a, n_cat)
        return jnp.sum(h_r * a_r)

    g = jax.grad(f)(jnp.array([[0.3, 1.1, 2.1, 3.1, 0.0]]))
    assert np.all(np.isfinite(np.asarray(g)))


# ============================================================================
# Albedo feedback + SW closure
# ============================================================================

def test_albedo_feedback_signs():
    """Tier 2/5: warming lowers albedo (positive ice-albedo feedback);
    thicker ice raises it; snow raises, ponds lower (delta-Eddington)."""
    h = jnp.array([2.0])
    a_cold = float(SW.maykut_untersteiner_albedo(jnp.array([253.0]), h)[0])
    a_warm = float(SW.maykut_untersteiner_albedo(jnp.array([constants.T_freeze]), h)[0])
    assert a_warm < a_cold
    a_thin = float(SW.maykut_untersteiner_albedo(jnp.array([253.0]), jnp.array([0.05]))[0])
    a_thick = float(SW.maykut_untersteiner_albedo(jnp.array([253.0]), jnp.array([2.0]))[0])
    assert a_thin < a_thick

    base = dict(T_sfc=jnp.array([253.0]), h_ice=jnp.array([2.0]))
    a_bare = float(SW.delta_eddington_albedo(
        h_snow=jnp.array([0.0]), pond_area=jnp.array([0.0]),
        pond_depth=jnp.array([0.0]), **base)[0][0])
    a_snow = float(SW.delta_eddington_albedo(
        h_snow=jnp.array([0.2]), pond_area=jnp.array([0.0]),
        pond_depth=jnp.array([0.0]), **base)[0][0])
    a_pond = float(SW.delta_eddington_albedo(
        h_snow=jnp.array([0.0]), pond_area=jnp.array([0.6]),
        pond_depth=jnp.array([0.3]), **base)[0][0])
    assert a_snow > a_bare
    assert a_pond < a_bare
    for v in (a_cold, a_warm, a_bare, a_snow, a_pond):
        assert 0.0 <= v <= 1.0


@pytest.mark.parametrize("scheme", ["constant", "maykut_untersteiner", "delta_eddington"])
def test_sw_energy_closure(scheme):
    """Tier 3: reflected + absorbed + penetrated == sw_down (no SW energy leak)."""
    sw = jnp.array([300.0])
    r = SW.compute_ice_sw(
        sw, jnp.array([253.0]), jnp.array([2.0]), jnp.array([0.0]),
        jnp.array([0.0]), jnp.array([0.0]), scheme=scheme, albedo_const=0.65)
    total = r.albedo_eff * sw + r.sw_absorbed_surface + r.sw_penetrated
    np.testing.assert_allclose(np.asarray(total), np.asarray(sw), rtol=1e-9)
    assert float(r.sw_absorbed_surface[0]) >= 0.0
    assert float(r.sw_penetrated[0]) >= 0.0


def test_albedo_grad_finite():
    """Tier 4: dα/dT <= 0 finite; dα/dh finite at the open-water limit h=0."""
    def fT(T):
        return jnp.sum(SW.maykut_untersteiner_albedo(T, jnp.array([2.0])))

    gT = float(jax.grad(fT)(jnp.array([260.0]))[0])
    assert np.isfinite(gT) and gT <= 0.0

    def fh(h):
        return jnp.sum(SW.maykut_untersteiner_albedo(jnp.array([253.0]), h))

    gh = float(jax.grad(fh)(jnp.array([0.0]))[0])
    assert np.isfinite(gh)


@pytest.mark.parametrize("scheme", ["maykut_untersteiner", "delta_eddington"])
def test_multicat_response_albedo_is_area_weighted(scheme):
    """REGRESSION (codex): in multi-category mode the coupler-facing tile albedo
    must be the AREA-WEIGHTED mean of the per-category albedos, NOT the albedo
    evaluated on the area-aggregated state.  For the nonlinear MU /
    delta-Eddington schemes the two differ by tens of W/m² for a thin+thick mix.
    """
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ice.itd import aggregate_state
    from legoesm.ice.sea_ice import step_sea_ice
    from legoesm.ice.state import init_dynamic_ice_state
    from legoesm.ice.shortwave import compute_ice_sw

    grid = create_cubed_sphere(n=4)
    n_cat = 2
    shape = (6, 4, 4, n_cat)
    state = init_dynamic_ice_state(shape, n_categories=n_cat)
    # Thin (0.05 m) + thick (2.0 m) categories, equal area, cold surface.
    h = jnp.zeros(shape).at[..., 0].set(0.05).at[..., 1].set(2.0)
    conc = jnp.full(shape, 0.5)
    T = jnp.full(shape, 253.0)
    state = state._replace(
        h_ice=state.h_ice.replace(data=h),
        concentration=state.concentration.replace(data=conc),
        T_ice=state.T_ice.replace(data=T),
    )
    sshape = (6, 4, 4)
    forcing = AtmToSurface(
        sw_down=jnp.full(sshape, 300.0), lw_down=jnp.full(sshape, 250.0),
        T_lowest=jnp.full(sshape, 250.0), q_lowest=jnp.full(sshape, 1e-3),
        u_lowest=jnp.full(sshape, 2.0), v_lowest=jnp.zeros(sshape),
        p_lowest=jnp.full(sshape, 1.0e5), p_surface=jnp.full(sshape, 1.0e5),
        rho_lowest=jnp.full(sshape, 1.3),
        cos_zenith=jnp.full(sshape, 0.5), co2_ppmv=jnp.full(sshape, 400.0),
        precip_total=jnp.zeros(sshape), precip_snow=jnp.zeros(sshape),
        has_radiation=jnp.ones(sshape), has_precipitation=jnp.zeros(sshape),
    )
    config = SeaIceConfig(n_categories=n_cat, shortwave_scheme=scheme,
                          itd_remap="lipscomb2001")
    _, resp = step_sea_ice(
        state, forcing, jnp.full(sshape, 271.35),
        jnp.zeros(sshape), jnp.zeros(sshape), config, 1.0, 3600.0, grid=grid)

    # Expected: area-weighted per-category albedo on the POST-STEP state.
    # The thermo barely moves the cold thin/thick mix in one step, so compare
    # the response albedo against the per-category weighting of the INPUT mix as
    # an order-of-magnitude check, and against the (wrong) aggregate-state value.
    sw_cat = compute_ice_sw(
        forcing.sw_down[..., None], T, h, jnp.zeros(shape),
        jnp.zeros(shape), jnp.zeros(shape), scheme=scheme, albedo_const=config.albedo_ice)
    conc_agg = jnp.sum(conc, axis=-1)
    alpha_weighted = jnp.sum(sw_cat.albedo_eff * conc, axis=-1) / conc_agg
    h_agg, T_agg, _ = aggregate_state(h, T, conc)
    sw_agg = compute_ice_sw(
        forcing.sw_down, T_agg, h_agg, jnp.zeros(sshape),
        jnp.zeros(sshape), jnp.zeros(sshape), scheme=scheme,
        albedo_const=config.albedo_ice)
    alpha_agg_wrong = sw_agg.albedo_eff

    resp_a = np.asarray(resp.albedo)
    # The response albedo must track the AREA-WEIGHTED value, far from the
    # biased aggregate-state value (which is ~0.2 higher here).
    assert np.allclose(resp_a, np.asarray(alpha_weighted), atol=2e-2)
    assert np.max(np.abs(resp_a - np.asarray(alpha_agg_wrong))) > 0.1


def test_ice_albedo_config_feedback_monotone():
    """legoesm.surface_albedo.ice_albedo: warming -> lower albedo, finite grad."""
    from legoesm.surface_albedo import ice_albedo, IceAlbedoConfig
    cfg = IceAlbedoConfig()
    a_cold = float(ice_albedo(jnp.array([cfg.T_freeze - 10.0]), cfg)[0])
    a_warm = float(ice_albedo(jnp.array([cfg.T_freeze + 1.0]), cfg)[0])
    assert a_cold == pytest.approx(cfg.alpha_ice_cold)
    assert a_warm == pytest.approx(cfg.alpha_ice_warm)
    assert a_warm < a_cold
    g = jax.grad(lambda T: jnp.sum(ice_albedo(T, cfg)))(jnp.array([cfg.T_freeze - 2.0]))
    assert np.all(np.isfinite(np.asarray(g)))
