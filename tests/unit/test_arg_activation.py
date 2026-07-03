"""Unit tests for physically-based ARG2000 aerosol activation.

Covers the Abdul-Razzak & Ghan (2000) modal activation scheme
(:mod:`legoesm.atmosphere.physics.microphysics.arg_activation`):

  * monotonicity of CDNC in aerosol number N_a and updraft w;
  * activated fraction in [0, 1];
  * competition limit (very high N_a -> fraction < 1) and clean-air limit
    (low N_a -> near-full activation of the accumulation mode);
  * sanity magnitude for a marine-ish accumulation mode (CDNC of order tens-
    hundreds cm^-3, cf. Abdul-Razzak & Ghan 2000 Fig. 1 / CliMA verification);
  * the ``scheme="proxy"`` dispatch reproduces the Andreae (2009) proxy exactly;
  * unknown scheme raises ValueError (dispatch hardening);
  * jax.grad of CDNC w.r.t. N_a and w is finite and positive.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics.arg_activation import (  # noqa: E402
    ActivationConfig,
    activated_nc_field,
    arg_cdnc,
    arg_cdnc_from_config,
)
from legoesm.atmosphere.physics.microphysics.aerosol_activation import (  # noqa: E402
    CCNFromAODConfig,
    specified_nc_field,
)

_PER_CM3 = 1.0e6  # cm^-3 -> m^-3

# Marine-ish accumulation mode reference environment (matches the module's
# idealized-test contract and the CliMA cross-check: CDNC ~ 70 cm^-3).
_R_G = 0.05e-6      # 50 nm dry geometric-mean radius [m]
_SIGMA = 2.0        # geometric standard deviation [-]
_KAPPA = 0.6        # ammonium-sulfate-ish hygroscopicity [-]
_W = 0.3            # updraft [m/s]
_T = 285.0          # temperature [K]
_P = 9.5e4          # pressure [Pa]


def _one_mode(n_cm3, w=_W, T=_T, p=_P, r_g=_R_G, sigma=_SIGMA, kappa=_KAPPA):
    """CDNC [1/m^3], shape (1,), for a single lognormal mode at (w, T, p).

    ``n_cm3`` and ``w`` may be traced scalars (used by the grad tests); T, p,
    r_g, sigma, kappa are the fixed reference environment.
    """
    w_f = jnp.broadcast_to(jnp.asarray(w, dtype=float), (1,))
    T_f = jnp.broadcast_to(jnp.asarray(T, dtype=float), (1,))
    p_f = jnp.broadcast_to(jnp.asarray(p, dtype=float), (1,))
    number = jnp.reshape(jnp.asarray(n_cm3, dtype=float) * _PER_CM3, (1,))
    return arg_cdnc(
        w_f, T_f, p_f,
        number,
        jnp.array([r_g]),
        jnp.array([sigma]),
        jnp.array([kappa]),
    )


def test_activated_fraction_in_unit_interval():
    # Sweep number and updraft; the per-mode activated fraction must stay in
    # [0, 1] everywhere.
    n = jnp.array([10.0, 100.0, 1000.0, 1.0e4]) * _PER_CM3   # (4,)
    w = jnp.array([0.05, 0.3, 1.0, 3.0])                     # (4,)
    T = jnp.full((4,), _T)
    p = jnp.full((4,), _P)
    _, _, frac = arg_cdnc(
        w, T, p,
        jnp.array([100.0 * _PER_CM3]), jnp.array([_R_G]),
        jnp.array([_SIGMA]), jnp.array([_KAPPA]),
        return_diagnostics=True,
    )
    assert jnp.all(frac >= 0.0)
    assert jnp.all(frac <= 1.0)
    # And with the number varying too:
    _, _, frac2 = arg_cdnc(
        jnp.full((4,), _W), T, p,
        n, jnp.array([_R_G]), jnp.array([_SIGMA]), jnp.array([_KAPPA]),
        return_diagnostics=True,
    )
    assert jnp.all((frac2 >= 0.0) & (frac2 <= 1.0))


def test_cdnc_monotone_in_aerosol_number():
    n_cm3 = jnp.array([20.0, 50.0, 100.0, 300.0, 1000.0])
    cdnc = jnp.array([float(_one_mode(x)[0]) for x in n_cm3]) / _PER_CM3
    # More aerosol -> more activated droplets (strictly increasing).
    assert jnp.all(jnp.diff(cdnc) > 0.0), cdnc


def test_cdnc_monotone_in_updraft():
    w = jnp.array([0.05, 0.1, 0.3, 0.6, 1.5])
    cdnc = jnp.array([float(_one_mode(100.0, w=float(x))[0]) for x in w]) / _PER_CM3
    # Stronger updraft -> higher S_max -> more activation (strictly increasing).
    assert jnp.all(jnp.diff(cdnc) > 0.0), cdnc


def test_competition_high_na_fraction_below_one():
    # Polluted: very high N_a -> supersaturation competition -> fraction < 1.
    frac_clean = float(_one_mode(20.0)[0]) / (20.0 * _PER_CM3)
    frac_polluted = float(_one_mode(5000.0)[0]) / (5000.0 * _PER_CM3)
    assert frac_polluted < 0.9, frac_polluted          # clear competition
    assert frac_polluted < frac_clean                  # monotone in competition


def test_clean_air_near_full_activation():
    # Pristine accumulation mode -> nearly all particles activate.
    frac_clean = float(_one_mode(10.0)[0]) / (10.0 * _PER_CM3)
    assert frac_clean > 0.9, frac_clean


def test_sanity_magnitude_marine_mode():
    # Marine-ish: N_a ~ 100 cm^-3, w ~ 0.3 m/s -> CDNC of order tens-hundreds
    # cm^-3 (ARG2000 Fig. 1; CliMA cross-check gave ~70 cm^-3). Wide band to
    # absorb the ~2% D_v/k_a difference vs CliMA's transport constants.
    cdnc_cm3 = float(_one_mode(100.0)[0]) / _PER_CM3
    assert 30.0 < cdnc_cm3 < 150.0, cdnc_cm3
    # Activated fraction should be a substantial but sub-unity fraction.
    assert 0.4 < cdnc_cm3 / 100.0 < 0.98, cdnc_cm3


def test_proxy_scheme_reproduces_andreae_exactly():
    # Regression guard: the default "proxy" dispatch must be BIT-IDENTICAL to
    # the existing Andreae (2009) specified_nc_field.
    ncol, nlev = 4, 6
    aer = jnp.full((ncol, nlev), 0.075 / nlev)
    got = activated_nc_field(ActivationConfig(), (ncol, nlev), aerosol_od=aer)
    ref = specified_nc_field(aer, (ncol, nlev))
    assert jnp.array_equal(got, ref)
    # Explicit proxy config equals the default CCNFromAODConfig path too.
    cfg = ActivationConfig(scheme="proxy", proxy=CCNFromAODConfig())
    got2 = activated_nc_field(cfg, (ncol, nlev), aerosol_od=aer)
    assert jnp.array_equal(got2, ref)


def test_arg_scheme_dispatch_matches_core():
    # The "arg" dispatch and the core arg_cdnc agree (prescribed single mode).
    ncol, nlev = 3, 5
    T = jnp.full((ncol, nlev), _T)
    p = jnp.full((ncol, nlev), _P)
    cfg = ActivationConfig(
        scheme="arg",
        mode_number_cm3=(100.0,), mode_r_g_um=(0.05,),
        mode_sigma_g=(2.0,), mode_kappa=(0.6,), w_char_m_s=_W,
    )
    got = activated_nc_field(cfg, (ncol, nlev), T=T, p=p)
    ref = arg_cdnc(
        jnp.full((ncol, nlev), _W), T, p,
        jnp.array([100.0 * _PER_CM3]), jnp.array([0.05e-6]),
        jnp.array([2.0]), jnp.array([0.6]),
    )
    assert got.shape == (ncol, nlev)
    assert jnp.allclose(got, ref, rtol=1e-10)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown activation scheme"):
        activated_nc_field(
            ActivationConfig(scheme="typo"), (2, 2),
            aerosol_od=jnp.ones((2, 2)) * 0.1,
        )


def test_proxy_without_aerosol_raises():
    with pytest.raises(ValueError, match="requires an aerosol_od"):
        activated_nc_field(ActivationConfig(scheme="proxy"), (2, 2))


def test_arg_without_tp_raises():
    with pytest.raises(ValueError, match="requires T and p"):
        activated_nc_field(ActivationConfig(scheme="arg"), (2, 2))


def test_prognostic_feed_requires_a_configured_mode():
    # Feeding a prognostic number with NO configured mode shape must raise
    # (the prognostic mode inherits mode-0's r_g/sigma/kappa).
    cfg = ActivationConfig(
        scheme="arg", mode_number_cm3=(), mode_r_g_um=(),
        mode_sigma_g=(), mode_kappa=())
    T = jnp.full((2,), _T)
    p = jnp.full((2,), _P)
    with pytest.raises(ValueError, match="at least one configured"):
        arg_cdnc_from_config(cfg, T, p, aerosol_number=jnp.full((2,), 1.0e8))


def test_arg_mode_tuples_must_have_equal_length():
    # A ragged config (r_g present but sigma_g empty) must raise, not enter
    # broken broadcasting.
    cfg = ActivationConfig(
        scheme="arg", mode_number_cm3=(100.0,), mode_r_g_um=(0.05,),
        mode_sigma_g=(), mode_kappa=(0.6,))
    T = jnp.full((2,), _T)
    p = jnp.full((2,), _P)
    with pytest.raises(ValueError, match="equal length"):
        arg_cdnc_from_config(cfg, T, p)


def test_differentiable_positive_grad_in_number_and_updraft():
    # grad of CDNC w.r.t. aerosol number and updraft is finite and positive.
    def cdnc_of_n(n_cm3):
        return _one_mode(n_cm3)[0]

    def cdnc_of_w(w):
        return _one_mode(100.0, w=w)[0]

    g_n = jax.grad(cdnc_of_n)(100.0)
    g_w = jax.grad(cdnc_of_w)(0.3)
    assert jnp.isfinite(g_n) and g_n > 0.0, g_n
    assert jnp.isfinite(g_w) and g_w > 0.0, g_w


def test_multimodal_consistency_and_giant_ccn_competition():
    # Two modes: (a) total CDNC = sum_i N_i * frac_i (self-consistency);
    # (b) adding a coarse "giant CCN" mode LOWERS S_max (extra condensation
    # sink) and thus suppresses the accumulation mode's activated fraction —
    # the well-known giant-CCN supersaturation-competition effect.
    T = jnp.full((2,), _T)
    p = jnp.full((2,), _P)
    w = jnp.full((2,), _W)
    cdnc1, smax1, frac1 = arg_cdnc(
        w, T, p, jnp.array([100.0 * _PER_CM3]),
        jnp.array([_R_G]), jnp.array([_SIGMA]), jnp.array([_KAPPA]),
        return_diagnostics=True)
    n = jnp.array([100.0 * _PER_CM3, 5.0 * _PER_CM3])
    r_g = jnp.array([_R_G, 0.5e-6])                 # add a coarse mode
    sig = jnp.array([_SIGMA, 2.2])
    kap = jnp.array([_KAPPA, _KAPPA])
    cdnc2, smax2, frac2 = arg_cdnc(
        w, T, p, n, r_g, sig, kap, return_diagnostics=True)
    assert cdnc2.shape == (2,)
    assert frac2.shape == (2, 2)
    # (a) total equals the per-mode activated-number sum.
    manual = jnp.sum(n[:, None] * frac2, axis=0)
    assert jnp.allclose(cdnc2, manual, rtol=1e-10)
    # (b) coarse mode adds a positive term to 1/S_max^2 -> S_max drops -> the
    # accumulation mode activates a SMALLER fraction.
    assert jnp.all(smax2 < smax1)
    assert jnp.all(frac2[0] < frac1[0])
