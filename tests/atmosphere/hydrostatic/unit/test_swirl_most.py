"""swirl-lm MOST port (physics/turbulence/swirl_most.py) against a NumPy transcription.

``_swirl_reference`` re-states swirl-lm 98f96f80's TensorFlow code line by line
(``boundary_condition/monin_obukhov_similarity_theory.py``: _stability_correction_
function, _richardson_number, _normalized_height, _surface_shear_stress_and_heat_flux,
_exchange_coefficient, surface_flux_update_fn; ``numerics/root_finder.newton_method``;
the sign flips in ``numerics/diffusion.py`` and ``equations/utils.py``). RULE 5: the
hard switch (``stability_width_K = 0``) matches it to 1e-12; the smooth default's gap
is pinned per case; gradients through the smooth default are checked.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.swirl_most import (
    SwirlMOSTConfig,
    bulk_richardson,
    normalized_height,
    stability_functions,
    swirl_most_fluxes,
)

jax.config.update("jax_enable_x64", True)

_CB = dict(z0_m=1.0e-3, gamma_m=15.0, gamma_h=15.0, beta_m=4.8, beta_h=7.8,
           gustiness_m_s=1.0, kappa=0.4, g_m_s2=9.81)


def _swirl_reference(u, v, theta, theta_l, q_t, rho, z, t_s, q_s, *, density_free):
    """swirl-lm's surface fluxes, kinematic and positive up (see module docstring)."""
    p = _CB
    ln_z = np.log(z / p["z0_m"])

    def psi(zeta, th):
        b = th - t_s
        xm = np.maximum(1.0 - p["gamma_m"] * zeta, 0.0) ** 0.25
        um = (2.0 * np.log(0.5 * (1.0 + xm)) + np.log(0.5 * (1.0 + xm ** 2))
              - 2.0 * np.arctan(xm) + 0.5 * np.pi)
        yh = np.maximum(1.0 - p["gamma_h"] * zeta, 0.0) ** 0.5
        uh = 2.0 * 1.0 * np.log(0.5 * (1.0 + yh))
        pm = np.where(b < 0.0, um, np.where(b > 0.0, -p["beta_m"] * zeta, 0.0))
        ph = np.where(b < 0.0, uh, np.where(b > 0.0, -p["beta_h"] * zeta, 0.0))
        return pm, ph

    def dnn(a, b):
        return 0.0 if b == 0.0 else a / b

    def zeta_of(th):
        r_b = p["g_m_s2"] * z * dnn(th - t_s, (u ** 2 + v ** 2) * th)

        def err(zeta):
            pm, ph = psi(zeta, th)
            return r_b - zeta * (ln_z - ph) / (ln_z - pm) ** 2

        res = np.finfo(np.float64).resolution
        eps = np.power(2.0, np.ceil(np.log(10.0 * res) / np.log(2.0)))
        x = 0.0
        for _ in range(10):
            dx = eps * abs(x)
            dx = 1e-4 if dx == 0.0 else dx
            df = (err(x + dx / 2.0) - err(x - dx / 2.0)) / dx
            x = x - dnn(err(x), df)
        return x

    u_mag = np.sqrt(p["gustiness_m_s"] ** 2 + u ** 2 + v ** 2)
    # scalars: stability from theta; c_h = kappa^2 / ((ln - psi_h)(ln - psi_m))
    pm, ph = psi(zeta_of(theta), theta)
    c_h = dnn(p["kappa"] ** 2, (ln_z - ph) * (ln_z - pm))
    # swirl scalar_flux = -rho c_h |U| (phi_1 - phi_s); diffusion.py negates it into
    # the face slot (rho D dphi/dz), whose physical upward flux is minus the slot.
    def up(phi1, phis):
        slot = -(-rho * c_h * u_mag * (phi1 - phis))
        return -slot / rho                 # kinematic, positive up

    w_th = up(theta_l, t_s)
    w_q = up(q_t, q_s)
    # momentum: stability from theta_l; C_d = kappa^2 / (rho (ln - psi_m)^2)
    pm_m, _ = psi(zeta_of(theta_l), theta_l)
    c_d = p["kappa"] ** 2 / (rho * (ln_z - pm_m) ** 2)
    tau13 = -rho * c_d * u * u_mag        # utils.py negates it into the 2*mu*S slot,
    tau23 = -rho * c_d * v * u_mag        # so the physical rho*u'w' equals tau13
    if density_free:
        uw, vw = tau13 / rho, tau23 / rho  # the LES: rho*u'w' = tau13 [Pa]
    else:
        uw, vw = tau13, tau23              # intended: u'w' = -kappa^2/(..)^2 u |U|
    return np.array([w_th, w_q, uw, vw])


# name: (u, v, theta, theta_l, q_t) at z = 70 m, rho 1.17; surface theta_l 297.126, q_t 0.0192
_CASES = {
    "trade_unstable": (-5.0, 2.0, 296.5, 296.4, 0.0170),
    "weak_wind_unstable": (0.5, 0.2, 297.07, 297.06, 0.0185),
    "stable": (3.0, 0.0, 297.626, 297.62, 0.0180),
    "calm_unstable": (0.0, 0.0, 296.6, 296.6, 0.0175),
    "exact_neutral": (4.0, 1.0, 297.126, 297.126, 0.0190),
    "near_neutral_unstable": (4.0, 1.0, 297.121, 297.121, 0.0190),
}
_Z, _RHO, _TS, _QS = 70.0, 1.17, 297.126, 0.0192


def _port(case, **cfg_kw):
    u, v, th, thl, qt = _CASES[case]
    cfg = SwirlMOSTConfig(**_CB, **cfg_kw)
    f = swirl_most_fluxes(jnp.asarray(u), jnp.asarray(v), jnp.asarray(th),
                          jnp.asarray(thl), jnp.asarray(qt), jnp.asarray(_RHO),
                          jnp.asarray(_Z), jnp.asarray(_TS), jnp.asarray(_QS), cfg)
    return np.array([float(f.w_theta_l), float(f.w_q_t), float(f.uw), float(f.vw)])


@pytest.mark.parametrize("density_free", [False, True])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_hard_switch_matches_swirl_reference(case, density_free):
    u, v, th, thl, qt = _CASES[case]
    ref = _swirl_reference(u, v, th, thl, qt, _RHO, _Z, _TS, _QS, density_free=density_free)
    got = _port(case, stability_width_K=0.0, density_free_drag=density_free)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-18)


# Smooth default (stability_width_K = 0.01 K, stable_continuation_width 1e-3) minus the
# hard switch, relative to the hard value, per case and output (w_theta_l, w_q_t, uw,
# vw); re-measured 2026-11-03 with the bounded stable continuation ("stable": 4e-15).
_SMOOTH_GAP = {
    "calm_unstable": (0.0, 0.0, 0.0, 0.0),
    "exact_neutral": (0.0, 0.0, 0.0, 0.0),
    "near_neutral_unstable": (-0.002781087479986738, -0.0027810874799866843,
                              -0.0018606553129383288, -0.0018606553129383288),
    "stable": (0.0, 0.0, 0.0, 0.0),
    "trade_unstable": (0.0, 0.0, 0.0, 0.0),
    "weak_wind_unstable": (-0.002179263799452728, -0.0021792637994526806,
                           -0.0006145890968793437, -0.0006145890968794418),
}


def _rel_gap(case):
    hard = _port(case, stability_width_K=0.0)
    soft = _port(case)
    return np.where(hard == 0.0, soft - hard, (soft - hard) / np.where(hard == 0.0, 1.0, hard))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_smooth_default_gap_is_the_recorded_one(case):
    np.testing.assert_allclose(_rel_gap(case), _SMOOTH_GAP[case], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("case", ["trade_unstable", "stable", "near_neutral_unstable"])
def test_gradients_through_the_smooth_default(case):
    u, v, th, thl, qt = _CASES[case]
    cfg = SwirlMOSTConfig(**_CB)

    def f(x):
        out = swirl_most_fluxes(x[0], x[1], x[2], x[3], x[4], jnp.asarray(_RHO),
                                jnp.asarray(_Z), jnp.asarray(_TS), jnp.asarray(_QS), cfg)
        return jnp.stack([out.w_theta_l, 1e3 * out.w_q_t, out.uw, out.vw])

    x0 = jnp.asarray([u, v, th, thl, 1e3 * qt]) * jnp.asarray([1, 1, 1, 1, 1e-3])
    check_grads(f, (x0,), order=1, modes=("fwd", "rev"), atol=1e-6, rtol=1e-4)
    g = jax.jacrev(f)(x0)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.abs(g[0, 3])) > 0.0  # heat flux responds to the air theta_l


def test_equilibrium_gives_zero_fluxes():
    cfg = SwirlMOSTConfig(**_CB)
    f = swirl_most_fluxes(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(_TS),
                          jnp.asarray(_TS), jnp.asarray(_QS), jnp.asarray(_RHO),
                          jnp.asarray(_Z), jnp.asarray(_TS), jnp.asarray(_QS), cfg)
    assert [float(x) for x in f] == [0.0, 0.0, 0.0, 0.0]


def test_signs_warm_moist_surface_and_drag():
    f = _port("trade_unstable")
    assert f[0] > 0.0 and f[1] > 0.0          # upward heat and moisture
    assert f[2] > 0.0 and f[3] < 0.0          # u < 0, v > 0: flux opposes the wind


def test_density_free_drag_is_one_over_rho():
    a = _port("trade_unstable")
    b = _port("trade_unstable", density_free_drag=True)
    np.testing.assert_allclose(b[2:], a[2:] / _RHO, rtol=1e-14)
    np.testing.assert_allclose(b[:2], a[:2], rtol=1e-14)


def test_negative_width_refused():
    from legoesm.atmosphere.physics.turbulence.swirl_most import validate_swirl_most
    with pytest.raises(ValueError, match="stability_width_K"):
        validate_swirl_most(SwirlMOSTConfig(**_CB, stability_width_K=-1.0))


def test_continuation_width_refused_on_the_smooth_path():
    from legoesm.atmosphere.physics.turbulence.swirl_most import validate_swirl_most
    with pytest.raises(ValueError, match="stable_continuation_width"):
        validate_swirl_most(SwirlMOSTConfig(**_CB, stable_continuation_width=0.0))
    validate_swirl_most(SwirlMOSTConfig(**_CB, stability_width_K=0.0,
                                        stable_continuation_width=0.0))


def _ri_residual(wind, b, **cfg_kw):
    cfg = SwirlMOSTConfig(**_CB, **cfg_kw)
    th, z, u = jnp.asarray(_TS + b), jnp.asarray(_Z), jnp.asarray(wind)
    zeta = normalized_height(th, jnp.asarray(_TS), u, jnp.zeros(()), z, cfg)
    ln_z = jnp.log(z / cfg.z0_m)
    pm, ph = stability_functions(zeta, th - _TS, cfg)
    rb = bulk_richardson(th, jnp.asarray(_TS), u, jnp.zeros(()), z, cfg)
    return float(jnp.abs(rb - zeta * (ln_z - ph) / (ln_z - pm) ** 2) / max(abs(float(rb)), 1e-12))


def test_smooth_solve_converges_wherever_the_hard_switch_does():
    # calm air near neutral: the blend once returned unconverged iterates here
    checked = 0
    for wind in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0):
        for b in (-0.05, -0.02, -0.01, -0.006, -0.003, -0.001,
                  0.001, 0.003, 0.006, 0.01, 0.02, 0.05):
            if _ri_residual(wind, b, stability_width_K=0.0) > 1e-8:
                continue                      # no root in the reference either
            checked += 1
            assert _ri_residual(wind, b) < 1e-10, (wind, b)
    assert checked >= 70


def test_kappa_default_is_the_repo_constant():
    assert SwirlMOSTConfig().kappa == constants.kappa_von_karman
