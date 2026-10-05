"""Unit tests for :mod:`legoesm.atmosphere.forcing.column_large_scale_extract`.

Stage-4 grid-side extractor: ω from continuity + horizontal advective
tendencies on a lat-lon GCM state.  Analytic checks on the pure cores plus
clean uniform-field cases on a real LatLonGrid (uniform flow/field ⇒ zero
divergence/gradient ⇒ zero ω/advection), and the assembled ColumnLargeScaleState.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.forcing.idealized.column_forcing import (
    ColumnLargeScaleState,
    build_column_scm_forcing,
)
from legoesm.atmosphere.forcing.column_large_scale_extract import (
    _MIN_GEOSTROPHIC_LAT_DEG,
    advective_tendency,
    extract_column_forcing_latlon,
    geostrophic_wind_from_gradients,
    omega_from_divergence,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

jax.config.update("jax_enable_x64", True)


def test_advective_tendency_analytic():
    u = jnp.full((3,), 2.0)
    v = jnp.zeros((3,))
    dphi_dx = jnp.full((3,), 3.0)
    dphi_dy = jnp.full((3,), -1.0)
    tend = advective_tendency(u, v, dphi_dx, dphi_dy)
    # -(2*3 + 0*-1) = -6
    np.testing.assert_allclose(np.asarray(tend), -6.0, rtol=1e-12)


def test_advective_tendency_zero_for_uniform_field():
    u = jnp.full((4,), 5.0)
    v = jnp.full((4,), 3.0)
    tend = advective_tendency(u, v, jnp.zeros((4,)), jnp.zeros((4,)))
    np.testing.assert_array_equal(np.asarray(tend), np.zeros(4))


def test_omega_zero_for_zero_divergence():
    sigma = create_sigma_coordinate(6)
    div = jnp.zeros((2, 3, 6))
    p_s = jnp.full((2, 3), 1.0e5)
    omega = omega_from_divergence(div, p_s, sigma)
    assert omega.shape == (2, 3, 6)
    np.testing.assert_allclose(np.asarray(omega), 0.0, atol=1e-10)


def test_omega_nonzero_finite_for_divergence():
    sigma = create_sigma_coordinate(6)
    # Uniform convergence (div<0) in the lower half, divergence aloft.
    div = jnp.zeros((1, 1, 6)).at[0, 0, 3:].set(-1e-5).at[0, 0, :3].set(1e-5)
    p_s = jnp.full((1, 1), 1.0e5)
    omega = omega_from_divergence(div, p_s, sigma)
    assert bool(jnp.all(jnp.isfinite(omega)))
    # ω at the model top (σ→0) is ~0; non-trivial in the interior.
    assert float(jnp.max(jnp.abs(omega))) > 0.0


def test_les_omega_uses_hybrid_continuity_over_terrain():
    """For a HYBRID coordinate over terrain (``p_s ≠ p_ref``) the LES large-scale ω must
    close the HYBRID continuity, not the pure-sigma form (~tens-of-% wrong over terrain).

    iter 348 fix: ``omega_from_divergence`` now REUSES the canonical dycore blocks
    :func:`compute_mass_flux_hybrid` + :func:`compute_omega_hybrid` (the SAME the
    spectral / primitive-eq hydrostatic dycore uses for hybrid) when handed a
    :class:`HybridSigmaPressureCoordinate`.  Verify the output equals that canonical
    hybrid ω EXACTLY, AND differs materially from the OLD pure-sigma ω (non-vacuity —
    the branch genuinely changed the answer over terrain).
    """
    from legoesm.grids.vertical import (
        compute_mass_flux_hybrid,
        compute_omega_hybrid,
        compute_pressure_velocity,
        compute_sigma_dot_and_total,
        make_hybrid_levels,
    )

    nlev = 12
    hc = make_hybrid_levels(nlev, p_top_Pa=100.0)
    div = jnp.linspace(2e-6, -1e-6, nlev)[None, :]            # (1, nlev) subsidence-like
    p_s = jnp.array([7.0e4])                                  # 700-hPa terrain (p_s != p_ref)
    omega = omega_from_divergence(div, p_s, hc)               # (1, nlev)

    # (1) Equals the canonical hybrid ω the dycore uses (ω_k = B_full·∂p_s/∂t + F_k,
    #     ∂p_s/∂t = -D_total_p/B_range, D_total_p = Σ_k (∇·v)_k dp_k).
    mass_flux, D_total_p = compute_mass_flux_hybrid(div, p_s, hc)
    dp_s_dt = -D_total_p[..., 0] / hc.B_range
    omega_hybrid = compute_omega_hybrid(mass_flux, p_s, dp_s_dt, hc)
    np.testing.assert_allclose(np.asarray(omega), np.asarray(omega_hybrid), rtol=1e-12)

    # (2) NON-VACUITY: the OLD pure-sigma ω (σ·∂p_s/∂t + p_s·σ̇ on the effective σ=A+B)
    #     differs materially over terrain, so the branch is not a no-op.
    sigma_dot, d_total = compute_sigma_dot_and_total(div, hc)
    dp_s_dt_sig = -p_s * d_total[..., 0] / (1.0 - jnp.asarray(hc.sigma_half[0]))
    omega_sigma = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_sig, hc)
    rel = float(jnp.max(jnp.abs(omega - omega_sigma)) / jnp.max(jnp.abs(omega_sigma)))
    assert rel > 0.1, (
        f"hybrid vs pure-sigma ω differ by only {rel:.1%} over terrain — expected >10%; "
        "the hybrid branch may not be active")


def test_omega_hybrid_branch_reduces_to_pure_sigma_when_a_is_zero():
    """The two ``omega_from_divergence`` branches must AGREE when the hybrid coordinate
    degenerates to pure sigma (``A=0, B=σ``) — the docstring's 'byte-identical when
    A=0, B=σ' reduction.  This locks the hybrid generalization
    (``compute_mass_flux_hybrid`` + ``compute_omega_hybrid``, the ``HybridSigma…``
    branch) against the proven DIRECT pure-sigma closure
    (``compute_sigma_dot_and_total`` + ``compute_pressure_velocity``, the
    ``SigmaCoordinate`` branch).  The per-branch tests above validate each branch
    against its OWN manual form; only this catches a refactor of EITHER branch that
    breaks their AGREEMENT in the degenerate case (while leaving the over-terrain
    hybrid case intact).  Run over terrain (``p_s ≠ p_ref``) so it is non-trivial."""
    from legoesm.grids.vertical import create_hybrid_coordinate

    nlev = 10
    sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)         # pure-sigma branch
    # A HYBRID coordinate that IS pure sigma: A=0 everywhere, B=σ_half.
    hybrid_pure = create_hybrid_coordinate(
        nlev,
        A_half=jnp.zeros(nlev + 1, dtype=jnp.float64),
        B_half=jnp.asarray(sigma.sigma_half, dtype=jnp.float64),
        dtype=jnp.float64,
    )

    div = jnp.linspace(2e-6, -1e-6, nlev)[None, :]                   # (1, nlev)
    p_s = jnp.array([7.0e4])                                         # 700-hPa terrain

    omega_sigma = omega_from_divergence(div, p_s, sigma)            # SigmaCoordinate branch
    omega_hybrid = omega_from_divergence(div, p_s, hybrid_pure)     # HybridSigma… branch, A=0

    # The hybrid generalization reduces to the pure-sigma closure (fp64; the two paths
    # build ω from DIFFERENT intermediates — mass flux vs σ̇ — so allow op-order fp noise).
    np.testing.assert_allclose(
        np.asarray(omega_hybrid), np.asarray(omega_sigma), rtol=1e-9, atol=1e-12)
    # Non-vacuity: the continuity is actually active (nonzero ω from a divergent column).
    assert float(jnp.max(jnp.abs(omega_sigma))) > 0.0


def test_omega_matches_manual_continuity_with_sign():
    """Nonzero column-integrated divergence: verify the dp_s/dt closure + sign
    against an explicit compute_sigma_dot_and_total + compute_pressure_velocity."""
    from legoesm.grids.vertical import (
        compute_pressure_velocity,
        compute_sigma_dot_and_total,
    )

    nlev = 6
    sigma = create_sigma_coordinate(nlev)
    D = 2e-5  # uniform divergence at every level (net mass export)
    div = jnp.full((1, 1, nlev), D)
    p_s = jnp.full((1, 1), 1.0e5)

    omega = omega_from_divergence(div, p_s, sigma)

    sigma_dot, d_total = compute_sigma_dot_and_total(div, sigma)
    sigma_top = float(sigma.sigma_half[0])
    dp_s_dt = -p_s * d_total[..., 0] / (1.0 - sigma_top)
    omega_manual = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma)
    # Float32-precision agreement (the sigma coordinate is float32 internally);
    # the point is the dp_s/dt closure + sign, not bit-exactness.
    np.testing.assert_allclose(np.asarray(omega), np.asarray(omega_manual), rtol=1e-6)

    # Net divergence (mass export) ⇒ surface pressure FALLS ⇒ dp_s/dt < 0.
    assert float(dp_s_dt[0, 0]) < 0.0
    # Δσ sums to (1 − σ_top) ⇒ D_total = D·(1 − σ_top); the /(1−σ_top) factor in
    # dp_s/dt then cancels it ⇒ dp_s/dt = −p_s·D exactly.
    assert float(d_total[0, 0, 0]) == pytest.approx(D * (1.0 - sigma_top), rel=1e-5)
    assert float(dp_s_dt[0, 0]) == pytest.approx(-1.0e5 * D, rel=1e-5)


def _latlon_state(n_lat=8, n_lon=16, nlev=6):
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    shape = (n_lat, n_lon, nlev)
    T = jnp.full(shape, 280.0)
    q_v = jnp.full(shape, 5e-3)
    u = jnp.full(shape, 10.0)
    v = jnp.full(shape, 0.0)
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    return grid, sigma, T, q_v, u, v, p_s


def test_extract_uniform_state_zero_forcing():
    """A horizontally-uniform state ⇒ zero advection AND zero subsidence."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(20.0)), col_index=(4, 8),
    )
    assert isinstance(ls, ColumnLargeScaleState)
    assert ls.T.shape == (6,)
    np.testing.assert_allclose(np.asarray(ls.theta_adv), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ls.qv_adv), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(ls.omega), 0.0, atol=1e-8)


def test_omega_from_divergence_rejects_unsupported_coordinate():
    """Dispatch hardening (iter 350, CLAUDE.md): ``omega_from_divergence`` must RAISE on a
    coordinate that is neither sigma nor hybrid rather than silently using the pure-sigma
    closure — a height/z* coordinate has no hydrostatic σ continuity, so diagnosing ω from
    σ pressures there would be silently wrong.  Synthetic-violation self-test: a
    non-(Sigma|Hybrid) coordinate object ⇒ ``ValueError``."""
    class _NotACoord:
        """Stands in for a non-(Sigma|Hybrid) coordinate, e.g. a height/z* coord."""

    div = jnp.zeros((1, 6))
    p_s = jnp.array([1.0e5])
    with pytest.raises(ValueError, match="unsupported vertical coordinate"):
        omega_from_divergence(div, p_s, _NotACoord())


def test_extract_latlon_threads_hybrid_coord_to_omega_over_terrain():
    """END-TO-END (iter 348): the latlon extractor must thread a HYBRID coordinate all the
    way to ``omega_from_divergence`` so the column's large-scale ω over terrain uses the
    hybrid continuity — not silently re-default to pure-sigma.  A non-uniform wind (nonzero
    divergence) + a 700-hPa terrain ``p_s`` + a hybrid coordinate ⇒ the extracted ω must
    EQUAL the direct hybrid ω computed on the extractor's OWN divergence operator
    (:func:`_divergence_latlon_3d`), gathered at the column.  Non-vacuity: that hybrid ω
    differs materially (>10%) from the pure-sigma ω the pre-fix extractor produced.
    """
    from legoesm.atmosphere.forcing.column_large_scale_extract import (
        _divergence_latlon_3d,
    )
    from legoesm.grids.vertical import (
        compute_pressure_velocity,
        compute_sigma_dot_and_total,
        make_hybrid_levels,
    )

    n_lat, n_lon, nlev = 8, 16, 10
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    hc = make_hybrid_levels(nlev, p_top_Pa=100.0)
    # Longitude-varying zonal wind ⇒ nonzero horizontal divergence ⇒ nonzero ω.
    lon = jnp.linspace(0.0, 2.0 * jnp.pi, n_lon, endpoint=False)[None, :, None]
    u = (10.0 + 5.0 * jnp.sin(lon)) * jnp.ones((n_lat, n_lon, nlev))
    v = jnp.zeros((n_lat, n_lon, nlev))
    T = jnp.full((n_lat, n_lon, nlev), 280.0)
    q_v = jnp.full((n_lat, n_lon, nlev), 5e-3)
    p_s = jnp.full((n_lat, n_lon), 7.0e4)            # terrain (p_s != p_ref)
    col = (4, 8)
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=hc,
        lat_rad=float(jnp.deg2rad(20.0)), col_index=col,
    )
    assert bool(jnp.all(jnp.isfinite(ls.omega)))
    assert float(jnp.max(jnp.abs(ls.omega))) > 0.0   # genuinely nonzero subsidence

    # (1) The extracted ω equals the direct HYBRID ω on the extractor's own divergence.
    div_3d = _divergence_latlon_3d(u, v, grid)
    omega_hybrid = omega_from_divergence(div_3d, p_s, hc)[col]
    np.testing.assert_allclose(np.asarray(ls.omega), np.asarray(omega_hybrid), rtol=1e-12)

    # (2) NON-VACUITY: that differs materially from the pure-sigma ω (the pre-iter-348
    #     value), so the hybrid coordinate genuinely changed the answer end-to-end.
    sigma_dot, d_total = compute_sigma_dot_and_total(div_3d, hc)
    dp_s_dt_sig = -p_s * d_total[..., 0] / (1.0 - jnp.asarray(hc.sigma_half[0]))
    omega_sigma = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_sig, hc)[col]
    rel = float(jnp.max(jnp.abs(omega_hybrid - omega_sigma))
                / jnp.max(jnp.abs(omega_sigma)))
    assert rel > 0.1, f"hybrid vs pure-sigma column ω differ by only {rel:.1%} over terrain"


def test_extract_latlon_is_surface_flux_free():
    """The latlon extractor produces NO surface boundary condition (prescribe='none',
    no T_s / w_th_s / w_qv_s) — matching the cubed-sphere lock and iter-148's
    build_column_les_setup guard: wiring a surface flux into ANY extractor (to improve
    convective columns) without also adding the LES surface-flux bottom BC would make
    that guard reject the column, so the invariant is locked across the dispatch."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(20.0)), col_index=(4, 8),
    )
    assert ls.prescribe == "none"
    assert ls.T_s is None and ls.w_th_s is None and ls.w_qv_s is None


def test_extract_nonuniform_gives_finite_forcing():
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    # Impose a zonal temperature gradient (warmer to the east).
    n_lat, n_lon, nlev = T.shape
    lon_ramp = jnp.linspace(0.0, 10.0, n_lon)[None, :, None]
    T = T + lon_ramp
    # Also impose a zonal humidity ramp (drier east).
    q_v = q_v - 1e-4 * jnp.linspace(0.0, 5.0, n_lon)[None, :, None]
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(20.0)), col_index=(4, 8),
    )
    assert bool(jnp.all(jnp.isfinite(ls.theta_adv)))
    assert float(jnp.max(jnp.abs(ls.theta_adv))) > 0.0

    # Wiring/sign check: the extractor's theta_adv at the column equals
    # -(u·∂θ/∂x + v·∂θ/∂y) from the SAME operator gradient (v=0 here).
    from legoesm.atmosphere.forcing.column_large_scale_extract import (
        _gradient_latlon_3d,
    )
    from legoesm.atmosphere.physics._shared import exner_function

    # Match the extractor's float64 cast of sigma_full for a bit-exact compare.
    sigma_full64 = jnp.asarray(sigma.sigma_full, dtype=T.dtype)
    theta = T / exner_function(p_s[..., None] * sigma_full64)
    th_x, th_y = _gradient_latlon_3d(theta, grid)
    expected = -(u * th_x + v * th_y)[4, 8, :]
    np.testing.assert_allclose(np.asarray(ls.theta_adv), np.asarray(expected), rtol=1e-10)
    # u>0 (eastward) up a warm-east gradient ⇒ cold advection (theta_adv < 0).
    assert bool(jnp.all(ls.theta_adv < 0.0))
    # Drier east ⇒ u>0 brings moister air ⇒ qv_adv > 0.
    assert bool(jnp.all(ls.qv_adv > 0.0))


def test_extract_feeds_build_column_scm_forcing():
    """The extracted state drives the iter-5 SCMForcing assembler end-to-end."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    T = T + jnp.linspace(0.0, 5.0, T.shape[1])[None, :, None]
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(30.0)), col_index=(4, 8),
    )
    forcing = build_column_scm_forcing(ls)  # omega present ⇒ subsidence set
    assert forcing.subsidence_w is not None
    assert forcing.theta_adv is not None
    # Coriolis at 30N.
    from legoesm import constants
    assert forcing.f_c == pytest.approx(2.0 * constants.Omega * 0.5, rel=1e-9)


def test_extract_jit():
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()

    def run(T_in):
        # col_index closed over as a static constant (documented contract).
        ls = extract_column_forcing_latlon(
            T=T_in, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
            lat_rad=0.5, col_index=(4, 8),
        )
        return ls.omega, ls.theta_adv, ls.qv_adv

    omega, th_adv, qv_adv = jax.jit(run)(T)
    for arr in (omega, th_adv, qv_adv):
        assert bool(jnp.all(jnp.isfinite(arr)))


# --- geostrophic wind (iter 29) --------------------------------------------

def test_geostrophic_wind_from_gradients_nh_sign_and_magnitude():
    """Φ decreasing northward (∂Φ/∂y<0), f>0 ⇒ westerly geostrophic wind
    u_g = -∂Φ/∂y/f > 0; v_g from ∂Φ/∂x only."""
    f = 1.0e-4
    T_v = jnp.full((4,), 280.0)
    dphi_dy = jnp.full((4,), -1.0e-2)   # Φ down toward north
    dphi_dx = jnp.zeros((4,))
    zero = jnp.zeros((4,))
    u_g, v_g = geostrophic_wind_from_gradients(dphi_dx, dphi_dy, zero, zero, T_v, f, dlnp_dlnps=jnp.ones(4))
    np.testing.assert_allclose(np.asarray(u_g), 1.0e-2 / f, rtol=1e-12)  # +100
    np.testing.assert_allclose(np.asarray(v_g), 0.0, atol=1e-12)
    assert bool(jnp.all(u_g > 0.0))  # westerly


def test_geostrophic_wind_from_gradients_lnps_term():
    """The +R_d·T_v·∂ln p_s term enters the pressure-surface gradient: with
    ∂Φ/∂·|σ=0, v_g = R_d·T_v·∂ln p_s/∂x / f."""
    from legoesm import constants

    f = 1.0e-4
    T_v = jnp.full((3,), 280.0)
    dlnps_dx = jnp.full((3,), 1.0e-6)
    zero = jnp.zeros((3,))
    u_g, v_g = geostrophic_wind_from_gradients(zero, zero, dlnps_dx, zero, T_v, f, dlnp_dlnps=jnp.ones(3))
    expected_v = constants.R_d * 280.0 * 1.0e-6 / f
    np.testing.assert_allclose(np.asarray(v_g), expected_v, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(u_g), 0.0, atol=1e-12)


def test_geostrophic_wind_hybrid_resting_isothermal_over_slope_is_small():
    """Resting isothermal atmosphere over a 1 km / 100 km slope on CAM6 L32
    hybrid levels: pressure surfaces are flat, so the true geostrophic wind is 0.
    The pure-sigma σ→p term (``∂ln p/∂ln p_s = 1`` at every level) puts ~950 m/s
    of false wind on the pure-pressure top levels; the hybrid factor ``B·p_s/p``
    must remove it."""
    from legoesm import constants
    from legoesm.atmosphere.forcing.column_large_scale_extract import (
        _geostrophic_wind_column,
    )
    from legoesm.grids.vertical import make_cam6_l32_levels

    coord = make_cam6_l32_levels()
    nlev = coord.n_levels
    T0, dx = 250.0, 1.0e5
    z_s = jnp.array([0.0, 1000.0, 2000.0])
    phis = constants.g * z_s
    p_s = constants.p_ref * jnp.exp(-phis / (constants.R_d * T0))
    T = jnp.full((3, nlev), T0)
    q_v = jnp.zeros((3, nlev))

    def grad_fn(field, _grid):
        gx = jnp.zeros_like(field).at[1].set((field[2] - field[0]) / (2.0 * dx))
        return gx, jnp.zeros_like(field)

    u_g, v_g = _geostrophic_wind_column(
        T=T, q_v=q_v, p_s=p_s, grid=None, sigma_coord=coord, grad_fn=grad_fn,
        lat_rad=np.deg2rad(45.0), col_index=(1,), phis=phis,
    )
    np.testing.assert_allclose(np.asarray(u_g), 0.0, atol=1e-12)
    # Residual (measured 4.5 m/s on the top levels) is the geopotential's linear
    # hypsometric layer thickness R·T·dp/(g·p_mid) over the p_s-dependent lower
    # layers, not the coordinate term; the pure-sigma term gives ~950 m/s here.
    assert float(jnp.max(jnp.abs(v_g))) < 10.0, np.asarray(v_g)


def test_geostrophic_wind_southern_hemisphere_reverses_sign():
    """In the SOUTHERN hemisphere ``f < 0``, so the SAME pressure gradient yields the
    OPPOSITE geostrophic wind — hemisphere antisymmetry.

    The NH test (``f>0``) cannot catch an ``abs(f)`` / sign bug in the ``/f``
    division: ``geostrophic_wind_from_gradients`` divides by the SIGNED column
    Coriolis (caller computes ``f = 2Ω sin(lat)``, negative south of the equator), so
    for ``Φ`` decreasing northward the NH gets a WESTERLY (u_g>0) and the SH an
    EASTERLY (u_g<0) geostrophic wind of equal magnitude.  A mishandled sign would
    give a physically backwards SH forcing while the NH test stayed green.
    """
    T_v = jnp.full((4,), 280.0)
    dphi_dy = jnp.full((4,), -1.0e-2)   # Φ down toward north (same gradient as the NH test)
    dphi_dx = jnp.full((4,), 3.0e-3)
    zero = jnp.zeros((4,))
    f_nh = 1.0e-4
    f_sh = -1.0e-4                       # same |f|, southern hemisphere
    u_nh, v_nh = geostrophic_wind_from_gradients(dphi_dx, dphi_dy, zero, zero, T_v, f_nh, dlnp_dlnps=jnp.ones(4))
    u_sh, v_sh = geostrophic_wind_from_gradients(dphi_dx, dphi_dy, zero, zero, T_v, f_sh, dlnp_dlnps=jnp.ones(4))
    # Exact value from the signed-f formula, and the antisymmetry u_sh = -u_nh.
    np.testing.assert_allclose(np.asarray(u_sh), 1.0e-2 / f_sh, rtol=1e-12)  # -100
    assert bool(jnp.all(u_sh < 0.0))                    # EASTERLY in the SH
    np.testing.assert_allclose(np.asarray(u_sh), -np.asarray(u_nh), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(v_sh), -np.asarray(v_nh), rtol=1e-12)


def test_extract_populates_geostrophic_wind_extratropics():
    """An extratropical column (|lat| ≥ cutoff) gets a finite (nlev,) u_geo/v_geo."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    nlev = T.shape[-1]
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(35.0)), col_index=(6, 8),
    )
    assert ls.u_geo is not None and ls.v_geo is not None
    assert ls.u_geo.shape == (nlev,) and ls.v_geo.shape == (nlev,)
    assert bool(jnp.all(jnp.isfinite(ls.u_geo)))
    assert bool(jnp.all(jnp.isfinite(ls.v_geo)))


def test_extract_geostrophic_orographic_term_from_phis():
    """Over terrain, passing ``phis`` (surface geopotential = g·z_s) adds the
    orographic gradient ∇phis to the geostrophic geopotential gradient: the
    u_geo/v_geo delta equals ∓∇phis/f (computed via the SAME grad operator), is
    σ-independent (phis does not vary with level), and is NON-zero. Flat/ocean
    (phis=None) is unchanged (covered by the tests above)."""
    import math

    from legoesm.atmosphere.forcing.column_large_scale_extract import (
        _gradient_latlon_3d,
    )

    from legoesm import constants

    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    col, lat_deg = (6, 8), 35.0
    # Smooth terrain g·z_s with BOTH lat and lon structure (∂x and ∂y nonzero).
    lat2d = jnp.asarray(grid.lat)[:, None]       # radians (n_lat, 1)
    lon2d = jnp.asarray(grid.lon)[None, :]       # (1, n_lon)
    phis = constants.g * (800.0 * jnp.cos(lat2d) * jnp.sin(lon2d))   # m²/s², ~800 m

    kw = dict(T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
              lat_rad=float(jnp.deg2rad(lat_deg)), col_index=col)
    ls_flat = extract_column_forcing_latlon(**kw)
    ls_oro = extract_column_forcing_latlon(**kw, phis=phis)

    # Expected orographic delta via the SAME operator: Δv = +∂phis/∂x / f,
    # Δu = -∂phis/∂y / f, gathered at the column, constant across levels.
    f_c = 2.0 * constants.Omega * math.sin(math.radians(lat_deg))
    dphis_dx, dphis_dy = _gradient_latlon_3d(phis[..., None], grid)
    exp_dv = float(np.asarray(dphis_dx)[col + (0,)]) / f_c
    exp_du = -float(np.asarray(dphis_dy)[col + (0,)]) / f_c
    du = np.asarray(ls_oro.u_geo) - np.asarray(ls_flat.u_geo)
    dv = np.asarray(ls_oro.v_geo) - np.asarray(ls_flat.v_geo)
    np.testing.assert_allclose(du, exp_du, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(dv, exp_dv, rtol=1e-6, atol=1e-9)
    # non-vacuous: the orographic term actually moves the geostrophic wind
    assert abs(exp_dv) > 1e-3 or abs(exp_du) > 1e-3
    # σ-independent: the delta is the same at every level
    np.testing.assert_allclose(du, du[0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dv, dv[0], rtol=1e-9, atol=1e-12)


def test_extract_geostrophic_orographic_phis_shape_mismatch_raises():
    """A ``phis`` not co-located with ``p_s`` (wrong spatial shape) FAILS LOUDLY
    rather than silently broadcast-adding the wrong orography. The 8×16 grid makes
    a transposed 16×8 field detectably wrong (a square grid would hide it)."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()   # p_s is (8, 16)
    kw = dict(T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
              lat_rad=float(jnp.deg2rad(35.0)), col_index=(6, 8))
    with pytest.raises(ValueError, match="co-located with p_s"):
        extract_column_forcing_latlon(**kw, phis=jnp.zeros((16, 8)))   # transposed
    with pytest.raises(ValueError, match="co-located with p_s"):
        extract_column_forcing_latlon(**kw, phis=jnp.zeros((8,)))      # wrong rank


def test_extract_no_geostrophic_wind_near_equator():
    """Within the equatorial cutoff, geostrophic balance is ill-posed ⇒ None
    (so build_column_scm_forcing disables geostrophic relaxation there)."""
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    lat = 0.5 * _MIN_GEOSTROPHIC_LAT_DEG  # inside the cutoff
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(lat)), col_index=(4, 8),
    )
    assert ls.u_geo is None and ls.v_geo is None
    forcing = build_column_scm_forcing(ls)
    assert forcing.u_geo is None  # geostrophic relaxation disabled


def test_extract_geostrophic_cutoff_is_exclusive_at_the_boundary():
    """The equatorial cutoff ``abs(lat_deg) < _MIN_GEOSTROPHIC_LAT_DEG`` is STRICT
    (``<``): a column EXACTLY at the cutoff latitude still gets a geostrophic wind;
    only columns strictly INSIDE the band are excluded.

    The existing equator test sits at HALF the cutoff (well inside → None), so it
    doesn't pin the boundary operator.  A future ``<``→``<=`` change would silently
    drop the boundary column's geostrophic relaxation (or vice-versa) — the
    geostrophic analog of the iter-225 campaign-health inclusive-threshold lock.
    """
    grid, sigma, T, q_v, u, v, p_s = _latlon_state()
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=float(jnp.deg2rad(_MIN_GEOSTROPHIC_LAT_DEG)),  # EXACTLY at the cutoff
        col_index=(4, 8),
    )
    # |lat| == cutoff is NOT < cutoff ⇒ geostrophic balance IS supplied (finite),
    # not None (which a strictly-inside column would get).
    assert ls.u_geo is not None and ls.v_geo is not None
    assert bool(jnp.all(jnp.isfinite(ls.u_geo))) and bool(jnp.all(jnp.isfinite(ls.v_geo)))


def test_extract_geostrophic_thermal_wind_westerly():
    """Warm equator / cold pole (Φ decreasing poleward) ⇒ westerly (u_geo>0)
    geostrophic wind at an NH column — the thermal-wind midlatitude westerlies."""
    n_lat, n_lon, nlev = 16, 32, 6
    grid = create_latlon_grid(n_lat, n_lon, dtype=jnp.float64)
    sigma = create_sigma_coordinate(nlev)
    lat2d = jnp.asarray(grid.grid_lat)[:, :, None]  # (n_lat, n_lon, 1) radians
    T = 240.0 + 50.0 * jnp.cos(lat2d) * jnp.ones((n_lat, n_lon, nlev))  # warm equator
    q_v = jnp.full((n_lat, n_lon, nlev), 5e-3)
    u = jnp.zeros((n_lat, n_lon, nlev))
    v = jnp.zeros((n_lat, n_lon, nlev))
    p_s = jnp.full((n_lat, n_lon), 1.0e5)
    # pick a clearly-NH column (lat well above the cutoff)
    i_nh = int(np.argmin(np.abs(np.asarray(grid.grid_lat)[:, 0] - np.deg2rad(45.0))))
    lat_rad = float(np.asarray(grid.grid_lat)[i_nh, 0])
    ls = extract_column_forcing_latlon(
        T=T, q_v=q_v, u=u, v=v, p_s=p_s, grid=grid, sigma_coord=sigma,
        lat_rad=lat_rad, col_index=(i_nh, 0),
    )
    assert ls.u_geo is not None
    # westerly (eastward) geostrophic wind through the column depth.
    assert bool(jnp.all(ls.u_geo > 0.0))
