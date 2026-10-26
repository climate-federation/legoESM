"""Direct tests for the ePBL vertical-mixing closure (``ocean/physics/vertical_mixing/epbl.py``).

The faithfulness claims in ``epbl.py``'s docstring are pinned here: each
ported kernel is checked against an independent reimplementation written
straight from the MOM6 Fortran, and the column solver is checked for the
invariants the scheme is supposed to have (energy limitation, the ordering of
the diagnosed depth with the surface buoyancy forcing, differentiability).

Non-vacuity: every kernel test is written so that perturbing the coefficient
it pins changes the asserted number (the reimplementations are independent
expressions, not calls back into the module under test).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.eos import eos_density_derivatives, make_eos_fn
from legoesm.ocean.physics.vertical_mixing.config import (
    EPBLConfig,
    VALID_VERTICAL_MIXING_SCHEMES,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.epbl import (
    _EQD_ML_C,
    _eqdisc_shape,
    _eqdisc_v0h,
    _find_mstar,
    _find_pe_chg,
    _find_pe_chg_orig,
    epbl_diagnostics,
    epbl_vertical_mixing,
    surface_tke_forcing,
    validate_epbl_config,
)

from legoesm import constants

# The ePBL comparison runs pin MOM6's constants to legoESM's, so the tests
# read them from the single canonical source rather than restating literals.
G = constants.g
RHO0 = 1025.0          # const-ok: ConstantsConfig.rho_0 default, a config value not a constant
C_SW = 3994.0          # const-ok: ConstantsConfig.c_sw default, a config value not a constant
OMEGA = constants.Omega


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _column(nz=50, dz=1.0, t_top=20.0, t_range=5.0, salinity=35.0):
    """The 50 m / 1 m reference column, built with MOM6's TS_CONFIG=linear."""
    h = jnp.full((nz,), dz)
    k = np.arange(nz)
    T = jnp.asarray(t_top - t_range * ((k + 0.5) / nz))
    S = jnp.full((nz,), salinity)
    eos = make_eos_fn()
    d_pres = (G * RHO0) * h
    p_lay = jnp.cumsum(d_pres) - 0.5 * d_pres
    rho = eos(T, S, p_lay)
    drho_dT, drho_dS = eos_density_derivatives(eos, T, S, p_lay)
    dSV_dT = -drho_dT / rho**2
    dSV_dS = -drho_dS / rho**2
    return h, T, S, dSV_dT, dSV_dS


def _ref_cfg(**over):
    """The override set used for the MOM6 comparison runs."""
    base = dict(
        dt_s=1200.0, mstar_scheme="reichl_h18", nstar=0.06, omega_frac=0.001,
        tke_decay=0.001, mstar_convect_coef=0.66, use_mld_iteration=True,
        max_mld_its=30, mixlen_exponent=1.0, translay_scale=0.01,
        wt_scheme="reichl_h18", vstar_scale_fac=0.5477, vstar_surf_fac=1.8258,
        rh18_mstar_cn3=-6.0, wstar_ustar_coef=15.0, fixed_mstar=0.0,
        mld_tol=0.1, mld_bisection=True, eqdisc_shape=True, eqdisc_v0h=True,
    )
    base.update(over)
    return EPBLConfig(**base)


def _b_flux(q_net, T0, S0):
    """MOM6 buoyancyFlux: positive when the surface flux is stabilizing."""
    eos = make_eos_fn()
    drho_dT, _ = eos_density_derivatives(
        eos, jnp.asarray([T0]), jnp.asarray([S0]), jnp.zeros(1))
    return float(-(G / RHO0) * drho_dT[0] * (q_net / (RHO0 * C_SW)))


# ---------------------------------------------------------------------------
# Kernel: find_PE_chg_orig
# ---------------------------------------------------------------------------


def test_find_pe_chg_orig_matches_independent_reimplementation():
    """``_find_pe_chg_orig`` reproduces MOM6's find_PE_chg_orig algebra."""
    args = dict(kddt_h=0.37, h_k=1.3, b_den_1=2.1,
                dte_term=-0.8, dse_term=0.15,
                dt_km1_t2=0.42, ds_km1_t2=-0.07,
                dt_to_dpe_k=3.1e3, ds_to_dpe_k=-2.4e3,
                dt_to_dpe_a=2.7e3, ds_to_dpe_a=-1.9e3,
                pres_z=1.1e4,
                dt_to_dcolht_k=1.7e-4, ds_to_dcolht_k=-8.0e-5,
                dt_to_dcolht_a=1.3e-4, ds_to_dcolht_a=-6.0e-5)
    pe, dpe, pe_max, dpe0 = _find_pe_chg_orig(**args)

    # Independent transcription of MOM_energetic_PBL.F90:3460-3519.
    K, h_k, b = args["kddt_h"], args["h_k"], args["b_den_1"]
    b1 = 1.0 / (b + K)
    b1Kd = K * b1
    I_Kr = 1.0 / (h_k * b + (b + h_k) * K)
    dT_k = (K * I_Kr) * args["dte_term"]
    dS_k = (K * I_Kr) * args["dse_term"]
    dT_km1 = b1Kd * (dT_k + args["dt_km1_t2"])
    dS_km1 = b1Kd * (dS_k + args["ds_km1_t2"])
    pe_ref = (args["dt_to_dpe_k"] * dT_k + args["dt_to_dpe_a"] * dT_km1
              + args["ds_to_dpe_k"] * dS_k + args["ds_to_dpe_a"] * dS_km1)
    ch = (args["dt_to_dcolht_k"] * dT_k + args["dt_to_dcolht_a"] * dT_km1
          + args["ds_to_dcolht_k"] * dS_k + args["ds_to_dcolht_a"] * dS_km1)
    if ch < 0.0:
        pe_ref -= args["pres_z"] * ch
    assert float(pe) == pytest.approx(pe_ref, rel=1e-12)

    # The derivative must agree with a finite difference of PE_chg itself.
    eps = 1e-7
    hi = _find_pe_chg_orig(**{**args, "kddt_h": K + eps})[0]
    lo = _find_pe_chg_orig(**{**args, "kddt_h": K - eps})[0]
    assert float(dpe) == pytest.approx(float((hi - lo) / (2 * eps)), rel=1e-5)

    # dPE_max is the infinite-diffusivity limit; dPEc_dKd_0 the slope at zero.
    huge = _find_pe_chg_orig(**{**args, "kddt_h": 1.0e12})[0]
    assert float(pe_max) == pytest.approx(float(huge), rel=1e-6)
    small = 1e-9
    slope0 = _find_pe_chg_orig(**{**args, "kddt_h": small})[0] / small
    assert float(dpe0) == pytest.approx(float(slope0), rel=1e-5)


def test_find_pe_chg_condensed_agrees_with_orig_on_a_shared_case():
    """The two PE kernels are two forms of the same quantity.

    MOM6 keeps both (``EPBL_ORIGINAL_PE_CALC``) and documents them as
    algebraically equivalent; this pins that equivalence in the port, which
    is the strongest available check on the condensed form.
    """
    hp_a, h_k = 2.1, 1.3
    T_a, T_b, S_a, S_b = 12.0, 11.4, 35.1, 35.0
    dt_to_dpe_a, ds_to_dpe_a = 2.7e3, -1.9e3
    dt_to_dpe_k, ds_to_dpe_k = 3.1e3, -2.4e3
    dt_ch_a, ds_ch_a = 1.3e-4, -6.0e-5
    dt_ch_k, ds_ch_k = 1.7e-4, -8.0e-5
    pres = 1.1e4
    kddt = 0.37

    # Th_a/Sh_a with no mixing above means the "orig" inputs reduce to these.
    th_a, sh_a = hp_a * T_a, hp_a * S_a
    th_b, sh_b = h_k * T_b, h_k * S_b
    pe_new, _, _, _ = _find_pe_chg(
        0.0, kddt, hp_a, h_k, th_a, sh_a, th_b, sh_b,
        dt_to_dpe_a, ds_to_dpe_a, dt_to_dpe_k, ds_to_dpe_k,
        pres, dt_ch_a, ds_ch_a, dt_ch_k, ds_ch_k)
    pe_orig, _, _, _ = _find_pe_chg_orig(
        kddt, h_k, hp_a,
        hp_a * (T_a - T_b), hp_a * (S_a - S_b),
        T_b - T_a, S_b - S_a,
        dt_to_dpe_k, ds_to_dpe_k, dt_to_dpe_a, ds_to_dpe_a,
        pres, dt_ch_k, ds_ch_k, dt_ch_a, ds_ch_a)
    assert float(pe_new) == pytest.approx(float(pe_orig), rel=1e-10)


# ---------------------------------------------------------------------------
# Kernel: find_mstar
# ---------------------------------------------------------------------------


def test_mstar_constant_scheme_returns_the_configured_value():
    cfg = EPBLConfig(mstar_scheme="constant", fixed_mstar=1.2,
                     mstar_convect_coef=0.0)
    m = _find_mstar(cfg, jnp.asarray(0.0), jnp.asarray(0.01),
                    jnp.asarray(30.0), jnp.asarray(9.4e-5))
    assert float(m) == pytest.approx(1.2, rel=1e-12)


def test_mstar_reichl_h18_matches_independent_reimplementation():
    """RH18 mstar = mstar_N + mstar_S, then the convective reduction."""
    cfg = _ref_cfg()
    b_flux, u_star, bld, absf = 3.0e-8, 0.014, 22.0, 9.37e-5
    got = float(_find_mstar(cfg, jnp.asarray(b_flux), jnp.asarray(u_star),
                            jnp.asarray(bld), jnp.asarray(absf)))

    msn = cfg.rh18_mstar_cn2 * math.exp(
        cfg.rh18_mstar_cn3 * bld * absf / u_star)
    mstar_n = (cfg.rh18_mstar_cn1 * msn) / (1.0 + msn)
    mstar_s = cfg.rh18_mstar_cs1 * (
        max(0.0, b_flux) ** 2 * bld / (u_star**5 * max(absf, 1e-20))
    ) ** cfg.rh18_mstar_cs2
    mstar = mstar_n + mstar_s
    t1 = -bld * min(0.0, b_flux)
    t2 = 2.0 * mstar * u_star**3
    red = ((1.0 - cfg.mstar_convect_coef) * t1 + t2) / (t1 + t2)
    assert got == pytest.approx(mstar * red, rel=1e-12)


def test_mstar_convective_reduction_bites_only_when_cooling():
    """MSTAR_CONV_ADJ must reduce mstar under cooling and not under heating."""
    cfg = _ref_cfg()
    kw = dict(u_star=jnp.asarray(0.014), bld=jnp.asarray(22.0),
              absf=jnp.asarray(9.37e-5))
    cool = float(_find_mstar(cfg, jnp.asarray(-3.0e-7), **kw))
    warm = float(_find_mstar(cfg, jnp.asarray(0.0), **kw))
    no_adj = float(_find_mstar(_ref_cfg(mstar_convect_coef=0.0),
                               jnp.asarray(-3.0e-7), **kw))
    assert cool < no_adj, "cooling must reduce mstar when MSTAR_CONV_ADJ > 0"
    assert warm == pytest.approx(
        float(_find_mstar(_ref_cfg(mstar_convect_coef=0.0),
                          jnp.asarray(0.0), **kw)), rel=1e-12), (
        "with no convection the reduction factor must be exactly 1")


# ---------------------------------------------------------------------------
# Kernel: equation-discovery shape and velocity scale
# ---------------------------------------------------------------------------


def test_eqdisc_shape_has_its_maximum_at_sigma_max_and_floors_below_the_bl():
    """Sane et al. (2026) Eq. 3: quadratic to the peak, cubic to the base."""
    cfg = _ref_cfg()
    nz = 50
    z_int = jnp.arange(nz + 1, dtype=jnp.float64)
    mld = 20.0
    shape = _eqdisc_shape(cfg, z_int, jnp.asarray(9.37e-5),
                          jnp.asarray(-1.0e-7), jnp.asarray(0.014),
                          jnp.asarray(mld))
    s = np.asarray(shape)
    assert s[0] == 0.0, "the surface interface carries no mixing"
    # The analytic peak is 1 at z = sigma_max * MLD, which falls between
    # interfaces, so the sampled maximum sits just below it.
    assert 0.99 < s.max() <= 1.0, (
        "the quadratic peaks at 1 by construction")
    below = s[z_int > mld]
    assert np.allclose(below, cfg.eqd_shape_function_epsilon), (
        "below the boundary layer the shape must sit on its epsilon floor")

    # sigma_max must obey its caps.
    c = _EQD_ML_C
    u_star, absf, b_flux = 0.014, 9.37e-5, -1.0e-7
    lh = min(max((-b_flux * mld) / u_star**3, -cfg.eqd_lh_cap), cfg.eqd_lh_cap)
    eh = min((mld * absf) / u_star, cfg.eqd_eh_upper_cap)
    f = 1.0 / (c[3] + c[4] * math.exp(-c[5] * lh)) + c[6]
    sm = (f * eh) / (c[1] * (f * eh) + c[2])
    sm = min(max(sm, cfg.eqd_sigma_max_lower_cap), cfg.eqd_sigma_max_upper_cap)
    assert np.argmax(s) == pytest.approx(round(sm * mld), abs=1.5)


def test_eqdisc_v0h_matches_independent_reimplementation_both_branches():
    cfg = _ref_cfg()
    c = _EQD_ML_C
    for b_flux in (2.0e-7, -2.0e-7):
        u_star, mld = 0.014, 18.0
        got = float(_eqdisc_v0h(cfg, jnp.asarray(b_flux),
                                jnp.asarray(u_star), jnp.asarray(mld)))
        bc = min(max(b_flux, cfg.eqd_bflux_lower_cap), cfg.eqd_bflux_upper_cap)
        b_h = abs(bc) * mld
        b13 = math.copysign(abs(b_h) ** (1 / 3), b_h)
        if bc >= 0.0:
            den = c[15] * b_h + c[16] * u_star * b13**2 + u_star**3
            ref = c[14] * u_star**4 / den
        else:
            den = c[17] * b13**2 + c[18] * u_star**2
            ref = b_h / den + c[14] * u_star
        ref = min(max(ref, cfg.eqd_v0_lower_cap), cfg.eqd_v0_upper_cap)
        assert got == pytest.approx(ref, rel=1e-12), f"b_flux={b_flux}"


# ---------------------------------------------------------------------------
# Surface TKE forcing
# ---------------------------------------------------------------------------


def test_surface_tke_forcing_sign_and_magnitude():
    """Cooling RELEASES energy (positive); warming COSTS energy (negative)."""
    h, T, S, dSV_dT, dSV_dS = _column()
    kw = dict(h_top=h[0], dSV_dT_top=dSV_dT[0], dSV_dS_top=dSV_dS[0],
              g=G, rho_0=RHO0, c_sw=C_SW, dt=1200.0)
    cool = float(surface_tke_forcing(jnp.asarray(-200.0), None, **kw))
    warm = float(surface_tke_forcing(jnp.asarray(200.0), None, **kw))
    zero = float(surface_tke_forcing(jnp.asarray(0.0), None, **kw))
    assert cool > 0.0 and warm < 0.0 and zero == 0.0
    assert cool == pytest.approx(-warm, rel=1e-12), "linear in the heat flux"

    # Independent transcription of MOM_diabatic_aux.F90:1147.
    d_temp = -200.0 * 1200.0 / (RHO0 * C_SW * float(h[0]))
    ref = -(0.5 * float(h[0]) * G * RHO0**2) * d_temp * float(dSV_dT[0])
    assert cool == pytest.approx(ref, rel=1e-12)


# ---------------------------------------------------------------------------
# Column solver behaviour
# ---------------------------------------------------------------------------


def _run_column(q_net, cfg=None, tau=0.2, lat=40.0):
    cfg = cfg or _ref_cfg()
    h, T, S, dSV_dT, dSV_dS = _column()
    absf = abs(2 * OMEGA * math.sin(math.radians(lat)))
    tke = jnp.zeros_like(T).at[0].set(surface_tke_forcing(
        jnp.asarray(q_net), None, h[0], dSV_dT[0], dSV_dS[0],
        g=G, rho_0=RHO0, c_sw=C_SW, dt=cfg.dt_s))
    return epbl_diagnostics(
        T[None, :] * 0.0, T[None, :] * 0.0, T[None, :], S[None, :],
        dSV_dT[None, :], dSV_dS[None, :], h, cfg,
        tau_x=jnp.asarray([tau]), tau_y=jnp.asarray([0.0]),
        B_flux=jnp.asarray([_b_flux(q_net, float(T[0]), float(S[0]))]),
        tke_forcing=tke[None, :], absf=jnp.asarray([absf]),
        g=G, rho_0=RHO0)


def test_column_produces_finite_non_negative_diffusivity():
    kd, mld, mixvel, mixlen = _run_column(-200.0)
    assert bool(jnp.isfinite(kd).all()), "Kd must be finite everywhere"
    assert bool((kd >= 0.0).all()), "a negative diffusivity is unphysical"
    assert float(kd[0, 0]) == 0.0 and float(kd[0, -1]) == 0.0, (
        "no mixing through the surface or the sea floor")
    assert 0.0 < float(mld[0]) <= 50.0, "the depth must lie inside the column"


def test_no_wind_and_no_buoyancy_forcing_gives_no_mixing():
    """With no energy input there is nothing to pay for mixing with."""
    kd, mld, _, _ = _run_column(0.0, tau=0.0)
    assert float(jnp.max(kd)) == pytest.approx(0.0, abs=1e-14)


def test_stronger_wind_deepens_the_boundary_layer():
    """mech_TKE ~ u_star^3, so more stress must buy more mixing."""
    _, mld_weak, _, _ = _run_column(0.0, tau=0.05)
    _, mld_strong, _, _ = _run_column(0.0, tau=0.4)
    assert float(mld_strong[0]) > float(mld_weak[0])


def test_diffusivity_is_capped_by_the_available_energy():
    """The invariant that defines ePBL: mixing cannot outrun the forcing.

    Halving the energy input can only reduce the column-integrated
    diffusivity; it can never increase it.
    """
    kd_lo, _, _, _ = _run_column(0.0, tau=0.1)
    kd_hi, _, _, _ = _run_column(0.0, tau=0.2)
    assert float(jnp.sum(kd_lo)) < float(jnp.sum(kd_hi))


def test_mld_iteration_is_self_consistent():
    """The returned depth must be a fixed point of the iteration.

    Feeding the converged depth back in as the first guess has to reproduce
    it to within the tolerance, which is what "converged" means.
    """
    cfg = _ref_cfg()
    _, mld, _, _ = _run_column(-200.0, cfg)
    h, T, S, dSV_dT, dSV_dS = _column()
    absf = abs(2 * OMEGA * math.sin(math.radians(40.0)))
    tke = jnp.zeros_like(T).at[0].set(surface_tke_forcing(
        jnp.asarray(-200.0), None, h[0], dSV_dT[0], dSV_dS[0],
        g=G, rho_0=RHO0, c_sw=C_SW, dt=cfg.dt_s))
    one_pass = EPBLConfig(**{**cfg._asdict(), "max_mld_its": 1})
    _, mld2, _, _ = epbl_diagnostics(
        T[None, :] * 0.0, T[None, :] * 0.0, T[None, :], S[None, :],
        dSV_dT[None, :], dSV_dS[None, :], h, one_pass,
        tau_x=jnp.asarray([0.2]), tau_y=jnp.asarray([0.0]),
        B_flux=jnp.asarray([_b_flux(-200.0, float(T[0]), float(S[0]))]),
        tke_forcing=tke[None, :], absf=jnp.asarray([absf]),
        mld_guess=mld, g=G, rho_0=RHO0)
    assert abs(float(mld2[0]) - float(mld[0])) < cfg.mld_tol


def test_gradient_flows_through_the_column_solver():
    """End-to-end reverse-mode AD: the repo's hard requirement.

    Both fixed-trip-count iterations and every ``jnp.where`` branch must
    leave a finite, non-zero cotangent on a physically influential input.
    """
    h, T, S, dSV_dT, dSV_dS = _column()
    cfg = _ref_cfg(max_mld_its=6, max_kd_its=5)
    absf = abs(2 * OMEGA * math.sin(math.radians(40.0)))

    def loss(tau):
        tke = jnp.zeros_like(T).at[0].set(surface_tke_forcing(
            jnp.asarray(-200.0), None, h[0], dSV_dT[0], dSV_dS[0],
            g=G, rho_0=RHO0, c_sw=C_SW, dt=cfg.dt_s))
        out = epbl_vertical_mixing(
            T[None, :] * 0.0, T[None, :] * 0.0, T[None, :], S[None, :],
            dSV_dT[None, :], dSV_dS[None, :], h, cfg,
            tau_x=tau[None], tau_y=jnp.zeros(1),
            B_flux=jnp.asarray([_b_flux(-200.0, float(T[0]), float(S[0]))]),
            tke_forcing=tke[None, :], absf=jnp.asarray([absf]),
            g=G, rho_0=RHO0)
        return jnp.sum(out.K_v)

    gr = jax.grad(loss)(jnp.asarray(0.2))
    assert jnp.isfinite(gr), "the gradient must be finite"
    assert abs(float(gr)) > 0.0, (
        "wind stress demonstrably changes K_v, so a zero gradient means the "
        "AD path is severed (a stop_gradient or a dead where branch)")


def test_jit_and_eager_agree():
    """A divergence here means a Python branch on a traced value."""
    h, T, S, dSV_dT, dSV_dS = _column()
    cfg = _ref_cfg(max_mld_its=6)
    absf = abs(2 * OMEGA * math.sin(math.radians(40.0)))
    tke = jnp.zeros_like(T).at[0].set(surface_tke_forcing(
        jnp.asarray(-200.0), None, h[0], dSV_dT[0], dSV_dS[0],
        g=G, rho_0=RHO0, c_sw=C_SW, dt=cfg.dt_s))
    args = (T[None, :] * 0.0, T[None, :] * 0.0, T[None, :], S[None, :],
            dSV_dT[None, :], dSV_dS[None, :])
    bf = jnp.asarray([_b_flux(-200.0, float(T[0]), float(S[0]))])

    def run(*a):
        return epbl_vertical_mixing(
            *a, h, cfg, tau_x=jnp.asarray([0.2]), tau_y=jnp.asarray([0.0]),
            B_flux=bf, tke_forcing=tke[None, :], absf=jnp.asarray([absf]),
            g=G, rho_0=RHO0).K_v

    assert np.allclose(np.asarray(run(*args)),
                       np.asarray(jax.jit(run)(*args)), rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Dispatch and validation
# ---------------------------------------------------------------------------


def test_epbl_is_a_canonical_scheme():
    assert "epbl" in VALID_VERTICAL_MIXING_SCHEMES


def test_unknown_mstar_scheme_raises():
    with pytest.raises(ValueError, match="mstar_scheme"):
        validate_epbl_config(EPBLConfig(mstar_scheme="rh18_typo"))


def test_unknown_velocity_scale_scheme_raises():
    with pytest.raises(ValueError, match="wt_scheme"):
        validate_epbl_config(EPBLConfig(wt_scheme="cuberoot"))


def test_both_equation_discovery_velocity_scales_at_once_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_epbl_config(EPBLConfig(eqdisc_v0=True, eqdisc_v0h=True))


def test_non_positive_timestep_raises():
    with pytest.raises(ValueError, match="dt_s"):
        validate_epbl_config(EPBLConfig(dt_s=0.0))


def test_explicit_diffusion_is_rejected():
    """ePBL is implicit-only; asking it to apply diffusion must fail loudly."""
    from legoesm.ocean.physics.vertical_mixing.integration import (
        make_vertical_mixing_physics,
    )
    with pytest.raises(ValueError, match="implicit_vertical_mixing"):
        make_vertical_mixing_physics(
            VerticalMixingConfig(scheme="epbl"), apply_diffusion=True)


def test_config_defaults_are_the_mom6_defaults():
    """Canary: these defaults ARE the MOM6 defaults, which is the port's
    contract (except ``vonkar``: legoESM uses 0.4 where MOM6 uses 0.41, so it
    is not listed here). A change here silently changes every run that does
    not set the field explicitly, so it must be a deliberate edit to this
    list."""
    c = EPBLConfig()
    assert (c.mstar_scheme, c.fixed_mstar, c.nstar) == ("constant", 1.2, 0.2)
    assert (c.tke_decay, c.ekman_scale_coef, c.omega_frac) == (2.5, 1.0, 0.0)
    assert (c.mixlen_exponent, c.translay_scale) == (2.0, 0.1)
    assert (c.wt_scheme, c.vstar_scale_fac, c.vstar_surf_fac) == (
        "cube_root_tke", 1.0, 1.2)
    assert (c.use_mld_iteration, c.mld_bisection, c.mld_tol) == (True, False, 1.0)
    assert (c.max_mld_its, c.mixlen_exponent) == (20, 2.0)
    assert (c.mke_to_tke_effic, c.mstar_convect_coef) == (0.0, 0.0)
    assert (c.rh18_mstar_cn1, c.rh18_mstar_cn2, c.rh18_mstar_cn3) == (
        0.275, 8.0, -5.0)
    assert (c.rh18_mstar_cs1, c.rh18_mstar_cs2) == (0.2, 0.4)
    assert (c.mstar_coef, c.c_ek) == (0.3, 0.085)
    assert (c.eqdisc_shape, c.eqdisc_v0, c.eqdisc_v0h) == (False, False, False)


# --- operator-split ordering option (k_after_surface_flux) -----------------

def _epbl_scm(q_net, k_after_surface_flux, nsteps=6, dt=1200.0, set_dt_s=True,
              dt_s=None):
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.scm import OceanColumnModel
    from legoesm.ocean.scm_forcing import OceanSCMForcing
    from legoesm.ocean.vertical import create_ocean_z_star

    nz = 20
    zc = create_ocean_z_star(n_levels=nz, H_max=20.0, dz_surface=1.0,
                             dz_deep=1.0)
    cfg = EPBLConfig(k_after_surface_flux=k_after_surface_flux)
    if set_dt_s:
        cfg = cfg._replace(dt_s=dt if dt_s is None else dt_s)
    phys = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="epbl", epbl=cfg),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None)
    lat = 40.0
    forcing = OceanSCMForcing(
        f_c=2.0 * OMEGA * math.sin(math.radians(lat)),
        tau_x=lambda t: 0.2, tau_y=lambda t: 0.0, q_net=lambda t: q_net)
    t_profile = 20.0 - 5.0 * (np.arange(nz) + 0.5) / nz
    scm = OceanColumnModel.create(
        nlev=nz, dt=dt, T_profile=t_profile, S_profile=35.0,
        physics_config=phys, z_coord=zc, latitude_deg=lat, forcing=forcing,
        implicit_vertical_mixing=True)
    return scm, scm.run(nsteps=nsteps, save_every=nsteps)[0]


@pytest.mark.parametrize("q_net", [-200.0, 200.0])
def test_k_after_surface_flux_changes_forced_column(q_net):
    """The option must actually reach the SCM step (non-vacuous): with a
    surface heat flux the two orderings give different columns."""
    _, base = _epbl_scm(q_net, False)
    _, mom6 = _epbl_scm(q_net, True)
    assert np.abs(np.asarray(mom6.T.data) - np.asarray(base.T.data)).max() > 1e-4


def test_k_after_surface_flux_is_inert_without_flux():
    """With zero heat flux both orderings give the same column."""
    _, base = _epbl_scm(0.0, False)
    _, mom6 = _epbl_scm(0.0, True)
    np.testing.assert_allclose(np.asarray(mom6.T.data),
                               np.asarray(base.T.data), atol=1e-12)


def test_scm_injects_model_dt_into_epbl():
    """The SCM must run ePBL on its own dt, not EPBLConfig's 1800 s default:
    a config that never sets dt_s gives the same column as one set to dt."""
    assert EPBLConfig().dt_s != 1200.0  # else the comparison is vacuous
    _, explicit = _epbl_scm(-200.0, False, dt=1200.0)
    _, injected = _epbl_scm(-200.0, False, dt=1200.0, set_dt_s=False)
    np.testing.assert_array_equal(np.asarray(injected.T.data),
                                  np.asarray(explicit.T.data))
    np.testing.assert_array_equal(np.asarray(injected.u.data),
                                  np.asarray(explicit.u.data))


def test_scm_rejects_user_dt_s_that_disagrees_with_model_dt():
    """A dt_s moved off the default that disagrees with the SCM dt raises
    instead of being silently overwritten."""
    assert EPBLConfig().dt_s not in (900.0, 1200.0)
    with pytest.raises(ValueError, match="900.0 s but the SCM timestep is "
                                         "1200.0 s"):
        _epbl_scm(-200.0, False, nsteps=1, dt=1200.0, dt_s=900.0)


def test_k_after_surface_flux_off_by_default():
    assert EPBLConfig().k_after_surface_flux is False


def test_k_after_surface_flux_rejected_outside_scm():
    """Any driver that builds ePBL physics with the flag set must fail
    loudly: only the SCM implements the MOM6 step order."""
    from legoesm.ocean.physics.vertical_mixing.integration import (
        make_vertical_mixing_physics,
    )
    with pytest.raises(NotImplementedError, match="k_after_surface_flux"):
        make_vertical_mixing_physics(
            VerticalMixingConfig(
                scheme="epbl", epbl=EPBLConfig(k_after_surface_flux=True)),
            apply_diffusion=False)
