"""Balanced DCMIP-2025 TC3 initial state for the spectral NH dycore.

The former IC (latitude-independent u = u_eq(z), rho' = theta' = 0) cannot
be balanced near the poles: u^2 tan(lat)/a diverges, and on the radius/60
sphere it drove polar |w| to 3.3 m/s in 4 min at T21 while the bubbles made
< 0.1 m/s.  The balanced IC (u = u_eq cos(lat), hydrostatic + cyclostrophic
theta', rho'; Klemp et al. 2015 / DCMIP2016) must stay near rest.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants as c  # noqa: E402
from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import (  # noqa: E402
    TC3_PARAMS,
    cyclostrophic_balance_theta_exner,
)
from legoesm.atmosphere.dynamics.gcm.spectral_nh import (  # noqa: E402
    SpectralCompressibleEulerModel,
    SpectralNHConfig,
    dcmip25_tc3_init_spectral,
)
from legoesm.grids.gaussian import (  # noqa: E402
    create_gaussian_grid,
    sh_synthesis_3d,
)


def test_balance_profiles_satisfy_both_relations():
    """Residuals of d(exner)/dz = -g/(c_p theta) and
    u^2 tan(lat) = -c_p theta d(exner)/d(lat), u = u_eq cos(lat), on an
    independent finite-difference check (smooth shear profile)."""
    z = np.linspace(0.0, 20000.0, 201)
    theta_eq = 300.0 * np.exp(1.2e-5 * z)
    exner_eq = 1.0 - np.concatenate([[0.0], np.cumsum(
        0.5 * (c.g / (c.c_pd * theta_eq[1:]) + c.g / (c.c_pd * theta_eq[:-1]))
        * np.diff(z))])
    u_eq = 30.0 * np.tanh(z / 5000.0)
    lat = np.linspace(-1.3, 1.3, 53)
    th, ex = cyclostrophic_balance_theta_exner(lat, z, theta_eq, exner_eq, u_eq)
    assert np.max(np.abs(th[len(lat) // 2] - theta_eq)) < 1e-9   # equator
    assert np.max(np.abs(th - theta_eq)) > 0.5                    # non-trivial
    # hydrostatic residual (interior levels, centred differences)
    hyd = np.gradient(ex, z, axis=1) + c.g / (c.c_pd * th)
    assert np.max(np.abs(hyd[:, 2:-2])) < 2e-3 * c.g / (c.c_pd * 300.0)
    # cyclostrophic residual (interior latitudes)
    u = u_eq[None, :] * np.cos(lat)[:, None]
    cyc = u ** 2 * np.tan(lat)[:, None] + c.c_pd * th * np.gradient(ex, lat, axis=0)
    scale = np.max(np.abs(u ** 2 * np.tan(lat)[:, None]))
    assert np.max(np.abs(cyc[2:-2])) < 2e-3 * scale


def _no_bubble_wmax(balanced, nsteps=20):
    x = float(TC3_PARAMS["small_earth_factor"])
    state, hc, tm = dcmip25_tc3_init_spectral(
        create_gaussian_grid(21), n_levels=40,
        params={"bubble_dtheta": 0.0}, balanced=balanced)
    cfg = SpectralNHConfig(
        n_acoustic_substeps=25, semi_implicit_acoustic=True,
        sponge_width=8000.0, sponge_coeff=0.15,
        hyperdiff_coeff=2.338e15 * 8.0 / x ** 4, small_earth_factor=x,
        rotating=False, fix_mass=True, anchor_mass_to_initial=True)
    g = create_gaussian_grid(21, radius=c.R_earth / x)
    model = SpectralCompressibleEulerModel(g, hc, tm, cfg)
    for _ in range(nsteps):
        state = model.step(state, 3.0)
    return float(jnp.max(jnp.abs(sh_synthesis_3d(g, state.w_hat.data))))


def test_balanced_tc3_stays_near_rest_without_bubbles():
    """No bubble, no physics, 60 s at T21: the balanced IC keeps |w| tiny;
    the unbalanced one (control) does not -- measured 4 min: 0.019 vs
    3.27 m/s."""
    w_bal = _no_bubble_wmax(True)
    w_unbal = _no_bubble_wmax(False)
    assert w_unbal > 0.3, w_unbal          # the control really is unbalanced
    assert w_bal < 0.05 * w_unbal, (w_bal, w_unbal)


def test_balance_refuses_unconverged_iteration():
    z = np.linspace(0.0, 20000.0, 81)
    th = 300.0 + 0.004 * z
    ex = 1.0 - c.g / (c.c_pd * 0.004) * np.log(th / 300.0)   # hydrostatic
    with pytest.raises(RuntimeError, match="not converged"):
        cyclostrophic_balance_theta_exner(
            np.linspace(-1, 1, 9), z, th, ex, 30.0 * np.tanh(z / 5000.0),
            n_iter=1)


def test_legacy_encoding_state_is_refused():
    """A state from the former (unweighted / kg-kg) encodings is refused,
    not silently re-read as J-weighted."""
    state, hc, tm = dcmip25_tc3_init_spectral(create_gaussian_grid(21), n_levels=10)
    x = float(TC3_PARAMS["small_earth_factor"])
    model = SpectralCompressibleEulerModel(
        create_gaussian_grid(21, radius=c.R_earth / x), hc, tm,
        SpectralNHConfig(small_earth_factor=x, rotating=False))
    old = state._replace(rho_prime_hat=state.rho_prime_hat.replace(name="rho_prime_hat"))
    with pytest.raises(ValueError, match="J-weighted"):
        model.compute_dry_mass(old)
    with pytest.raises(ValueError, match="J-weighted"):
        model.step(old, 1.0)
    kgkg = state._replace(tracers_hat=state.tracers_hat.replace(units="kg/kg"))
    with pytest.raises(ValueError):
        model.step(kgkg, 1.0)
    bad_rho = state._replace(rho_prime_hat=state.rho_prime_hat.replace(units="kg/kg"))
    with pytest.raises(ValueError):
        model.compute_dry_mass(bad_rho)


def test_balance_rejects_degenerate_inputs():
    z = np.linspace(0.0, 20000.0, 81)
    th = 300.0 + 0.004 * z
    ex = 1.0 - c.g / (c.c_pd * 0.004) * np.log(th / 300.0)   # hydrostatic
    u = 30.0 * np.tanh(z / 5000.0)
    with pytest.raises(ValueError, match="n_quad"):
        cyclostrophic_balance_theta_exner(np.zeros(3), z, th, ex, u, n_quad=2)
    with pytest.raises(ValueError, match="latitudes"):
        cyclostrophic_balance_theta_exner(np.array([0.0, 2.0]), z, th, ex, u)
    poles32 = np.array([-np.pi / 2, np.pi / 2], dtype=np.float32)   # accepted
    th_p, _ = cyclostrophic_balance_theta_exner(poles32, z, th, ex, u)
    assert np.all(np.isfinite(th_p))
    # Equatorial profiles that are not hydrostatic are refused.
    with pytest.raises(ValueError, match="hydrostatic"):
        cyclostrophic_balance_theta_exner(
            np.zeros(3), z, th, np.full_like(z, 0.9), u)
    # A (hydrostatic) exner profile that the cyclostrophic correction
    # drives through 0 is refused.
    ex_small = ex - ex.min() + 1e-4
    with pytest.raises(RuntimeError, match="exner"):
        cyclostrophic_balance_theta_exner(
            np.linspace(-1.5, 1.5, 9), z, th, ex_small, 300.0 * np.ones_like(z))
