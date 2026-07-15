"""Implicit-differentiation consistency for the canopy Newton closure.

The solver wraps a Newton fixed point in a ``custom_vjp`` that returns the
implicit-function-theorem adjoint.  These tests verify it is *consistent*:

- ``jax.grad`` through ``solve_canopy_closure`` matches central finite
  differences at a well-converged operating point — for both an atmospheric
  forcing (La) and a trainable physics parameter (Vcmax25).  This is the gold
  standard, and it catches the two bugs fixed alongside the DifferBESS sync:
  the biased ``lstsq(rcond=1e-4)`` adjoint (now an exact solve) and the
  all-or-nothing NaN mask.
- Per-field masking: the Monin-Obukhov scan is not differentiable w.r.t. its
  aerodynamic forcing (e.g. Ta -> NaN cotangent), but that no longer poisons
  the finite gradients of the other inputs (La, Vcmax25, ...).
- The cotangent w.r.t. the initial guess ``x0`` is zero (a fixed point does
  not depend on where the iteration started).
- A non-converged solve (max_iters = 1) returns a zero gradient.

Run under JAX_ENABLE_X64=1 for a clean FD comparison.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure

_a = jnp.asarray


def _bundle(La=_a(350.0), Vc3=_a(40.0), Ta=_a(298.0)):
    """A physically reasonable single-column midday forcing bundle (scalars)."""
    return CanopyForcingBundle(
        LAI=_a(3.0), SZA=_a(30.0), La=La, epsf=_a(0.97), epss=_a(0.96), fSun=_a(0.6),
        APAR_Sun=_a(800.0), APAR_Sh=_a(200.0),
        Vcmax25_Sun=Vc3, Vcmax25_Sh=_a(20.0),
        Vcmax25_C4Sun=_a(0.0), Vcmax25_C4Sh=_a(0.0),
        ASW_Sun=_a(400.0), ASW_Sh=_a(100.0), ASW_Soil=_a(50.0),
        Ts_bc=_a(300.0),
        Ca=_a(400.0), Ps=_a(101325.0), Ta=Ta,
        lam=_a(2.5e6), Cp=_a(1005.0), rhoa=_a(1.2), Tv_atm=_a(298.5), q_atm=_a(0.012),
        m=_a(9.0), b0=_a(0.01), alf=_a(0.3), TgC=_a(20.0), fC4=_a(0.0),
        fStress_soil=_a(0.8),
        ur=_a(3.0), CI=_a(0.75), z0m=_a(0.5), displa=_a(3.0), z0=_a(10.0),
        cv=_a(0.0135), d_leaf=_a(0.025),
        r_soil_surface=_a(0.0),
    )


_X0 = jnp.array([298.0, 298.0, 280.0, 280.0, 298.0, 0.013])
# Tight-but-reachable tolerance: the forward converges (so the IFT identity
# holds) and x* is an accurate root (so IFT and FD agree).
_CFG = CanopyConfig(tol=1e-6, max_iters=300)


def _tf_sun_of_La(La):
    return solve_canopy_closure(_X0, _bundle(La=La), _CFG)[0][0]


def _tf_sun_of_Vc3(Vc3):
    return solve_canopy_closure(_X0, _bundle(Vc3=Vc3), _CFG)[0][0]


def test_ift_grad_matches_fd_forcing():
    La0 = _a(350.0)
    g_ad = jax.grad(_tf_sun_of_La)(La0)
    g_fd = (_tf_sun_of_La(La0 + 0.1) - _tf_sun_of_La(La0 - 0.1)) / 0.2
    assert float(g_ad) > 0.0  # more incoming LW -> warmer sunlit leaf
    assert jnp.allclose(g_ad, g_fd, rtol=1e-3, atol=1e-5)


def test_ift_grad_matches_fd_trainable_param():
    Vc0 = _a(40.0)
    g_ad = jax.grad(_tf_sun_of_Vc3)(Vc0)
    g_fd = (_tf_sun_of_Vc3(Vc0 + 0.05) - _tf_sun_of_Vc3(Vc0 - 0.05)) / 0.1
    assert jnp.isfinite(g_ad)
    assert jnp.allclose(g_ad, g_fd, rtol=2e-3, atol=1e-6)


def test_per_field_masking_keeps_finite_grads():
    """Grads through the canopy closure are finite for every forcing field.

    Previously the 4-regime MOST stability produced a NaN cotangent for the air
    temperature Ta (out-of-domain sqrt/log/cbrt in the where-combined branches),
    which a per-field mask had to zero to keep La's gradient finite.  With the
    MOST branches now grad-safe (2026-07-09 audit F5), Ta's cotangent flows as a
    genuine finite value that agrees with finite differences, so BOTH La and Ta
    carry finite, physically-signed, correct gradients.
    """
    def tf(La, Ta):
        return solve_canopy_closure(_X0, _bundle(La=La, Ta=Ta), _CFG)[0][0]

    g_La, g_Ta = jax.grad(tf, argnums=(0, 1))(_a(350.0), _a(298.0))
    assert float(g_La) > 0.0           # preserved, finite, nonzero
    assert jnp.isfinite(g_La)
    assert jnp.isfinite(g_Ta)          # MOST is grad-safe now: real Ta gradient
    # The Ta gradient is now genuine (not a masked-away NaN): validate it against
    # a central finite difference.
    g_Ta_fd = (tf(_a(350.0), _a(298.0 + 0.05)) - tf(_a(350.0), _a(298.0 - 0.05))) / 0.1
    assert jnp.allclose(g_Ta, g_Ta_fd, rtol=5e-3, atol=1e-5)


def test_initial_guess_cotangent_is_zero():
    bundle = _bundle()

    def tf_of_x0(x0):
        return solve_canopy_closure(x0, bundle, _CFG)[0][0]

    g_x0 = jax.grad(tf_of_x0)(_X0)
    assert jnp.allclose(g_x0, 0.0, atol=1e-8)


def test_nonconverged_returns_zero_gradient():
    cfg_bad = CanopyConfig(tol=1e-12, max_iters=1)

    def tf_bad(La):
        return solve_canopy_closure(_X0, _bundle(La=La), cfg_bad)[0][0]

    g = jax.grad(tf_bad)(_a(350.0))
    assert float(g) == 0.0


def test_cold_calm_forcing_stays_finite():
    """Fix 3a (non-finite Newton-step guard): at the calm cold-night MOST edge the
    aerodynamic Jacobian can degenerate so ``jnp.linalg.solve`` returns a NaN step
    — and ``jnp.clip(nan)`` stays NaN, which would poison every coupled land leaf
    and force the atomic land guard to revert the step.  The solver now sanitises a
    non-finite step to 0 BEFORE the clip, so the closure returns its last finite
    iterate.  This locks in a finite solve for the extreme cold / near-calm /
    no-sunlight forcing that regime lives in."""
    stressed = _bundle(La=_a(180.0), Ta=_a(233.0))._replace(
        SZA=_a(89.0), fSun=_a(0.0), APAR_Sun=_a(0.0), APAR_Sh=_a(0.0),
        ASW_Sun=_a(0.0), ASW_Sh=_a(0.0), ASW_Soil=_a(0.0),
        ur=_a(1.0e-4), Ts_bc=_a(235.0), Tv_atm=_a(233.0), q_atm=_a(1.0e-4))
    x0 = jnp.array([235.0, 235.0, 233.0, 233.0, 235.0, 1.0e-4])
    out, _n_iters = solve_canopy_closure(x0, stressed, _CFG)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_canopy_forward_reports_gross_and_net():
    """DECISIVE gross/net split at the boundary two_leaf_canopy consumes.

    ``canopy_forward`` must return BOTH the NET ``An_*`` (which drives the leaf
    energy/CO2 flux + SIF) and the GROSS ``Agross_*`` (the carbon-model GPP).
    For the lit sunlit leaf the gross-minus-net gap is exactly the dark
    respiration Rd, and the GROSS GPP aggregation that ``two_leaf_canopy``
    exports is STRICTLY LARGER than the pre-fix NET aggregation — the
    foliar-respiration double-count fix.  Fails on the pre-fix code: the flux
    dict had no ``Agross_*`` keys and GPP used net ``An``.
    """
    from legoesm.land.canopy.solver import canopy_forward
    from legoesm.land.canopy.photosynthesis import _rd_atkin, _TGC_LO, _TGC_HI
    from legoesm.land.surface_scheme.two_leaf_canopy import _G_C_PER_UMOL_CO2

    bundle = _bundle()          # pure-C3 midday column (fC4 = 0)
    x_star, _ = solve_canopy_closure(_X0, bundle, _CFG)
    fluxes = canopy_forward(
        x_star, bundle, _CFG.LE_module, _CFG.stomatal_model,
        _CFG.le_cap_mode, _CFG.use_ta_for_photosynthesis)

    for k in ("An_Sun", "An_Sh", "Agross_Sun", "Agross_Sh"):
        assert k in fluxes, f"canopy_forward dict missing {k}"

    An_Sun = float(fluxes["An_Sun"]);  Ag_Sun = float(fluxes["Agross_Sun"])
    An_Sh  = float(fluxes["An_Sh"]);   Ag_Sh  = float(fluxes["Agross_Sh"])

    # Gross >= net for both leaf classes.
    assert Ag_Sun >= An_Sun
    assert Ag_Sh >= An_Sh

    # In the light (An_Sun > 0), gross - net == Rd exactly.  Pure C3, and
    # use_ta_for_photosynthesis defaults False so T_phot is the leaf T (x_star).
    assert An_Sun > 0.0
    Tf_Sun = x_star[0]
    Rd_sun = _rd_atkin(Tf_Sun, jnp.clip(bundle.TgC, _TGC_LO, _TGC_HI),
                       bundle.Vcmax25_Sun)
    assert jnp.allclose(Ag_Sun - An_Sun, Rd_sun, rtol=1e-6, atol=1e-6)

    # DECISIVE: the GROSS GPP two_leaf_canopy exports strictly exceeds the NET
    # aggregation the pre-fix code exported (both in gC/m2/s).
    gpp_gross   = (Ag_Sun + Ag_Sh) * _G_C_PER_UMOL_CO2
    gpp_net_old = (An_Sun + An_Sh) * _G_C_PER_UMOL_CO2
    assert gpp_gross > gpp_net_old


def test_two_leaf_public_export_gpp_is_gross():
    """PUBLIC-boundary lock: ``compute_two_leaf_canopy_fluxes().gpp`` is the
    GROSS aggregation ``(Agross_Sun+Agross_Sh)*conv``, NOT the net
    ``(An_Sun+An_Sh)*conv``.  Reverting the two_leaf_canopy GPP line to net
    makes THIS test fail (the prior canopy_forward test does not, because it
    never exercises the public surface-scheme entry point).

    ``SurfaceFluxOutput`` does not expose the per-leaf An/Agross, so we
    reproduce the canopy's forward assembly with the SAME public helpers the
    source uses and a FROZEN soil callback (``soil_thermal_fn`` returns
    ``T_soil_top`` so every Picard pass keeps ``Ts_bc == T_soil_top`` and the
    converged state is a single deterministic solve).  Pure C3 (fC4=0), no soil
    stress (w_frac_rz=1) so the reconstructed bundle matches the internal one
    bit-for-bit, giving ``out.gpp == gross_sum`` to fp precision.
    """
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.canopy import CanopyLandParams
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.canopy.radiative_transfer import (
        split_sw_components, canopy_shortwave_rt)
    from legoesm.land.canopy.stability import (
        compute_aerodynamics, sat_specific_humidity)
    from legoesm.land.canopy.solver import (
        CanopyForcingBundle, solve_canopy_closure, canopy_forward)
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme import two_leaf_canopy as tlc

    ncol = 1
    cc = TwoLeafCanopyConfig(max_iters=60)
    land_config = MultiLayerLandConfig(surface_scheme=cc)
    conv = tlc._G_C_PER_UMOL_CO2

    forcing = AtmToSurface(
        sw_down=jnp.full(ncol, 700.0), lw_down=jnp.full(ncol, 380.0),
        precip_total=jnp.zeros(ncol), precip_snow=jnp.zeros(ncol),
        T_lowest=jnp.full(ncol, 298.0), q_lowest=jnp.full(ncol, 0.012),
        u_lowest=jnp.full(ncol, 3.0), v_lowest=jnp.full(ncol, 0.5),
        p_lowest=jnp.full(ncol, 98000.0), p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.18), cos_zenith=jnp.full(ncol, 0.8),
        co2_ppmv=jnp.full(ncol, 420.0),
        has_radiation=True, has_precipitation=False)
    params = CanopyLandParams(
        LAI=jnp.full(ncol, 3.0), hc=jnp.full(ncol, 5.0),
        fC4=jnp.zeros(ncol), FNonVeg=jnp.zeros(ncol),
        CI=jnp.full(ncol, 0.75), kn=jnp.full(ncol, 0.3),
        Vcmax25_C3_leaf=jnp.full(ncol, 60.0), Vcmax25_C4_leaf=jnp.full(ncol, 40.0),
        m_C3=jnp.full(ncol, 9.0), m_C4=jnp.full(ncol, 4.0),
        b0_C3=jnp.full(ncol, 0.01), b0_C4=jnp.full(ncol, 0.04),
        alf=jnp.full(ncol, 0.3), TgC=jnp.full(ncol, 20.0),
        ALB_VIS=jnp.full(ncol, 0.1), ALB_NIR=jnp.full(ncol, 0.2),
        emissivity=jnp.full(ncol, 0.97), rz0m=jnp.full(ncol, 0.055),
        rd=jnp.full(ncol, 0.67))
    T_soil_top = jnp.full(ncol, 296.0)
    wind = jnp.sqrt(jnp.full(ncol, 3.0) ** 2 + jnp.full(ncol, 0.5) ** 2)
    wdir_x = jnp.full(ncol, 3.0) / wind
    wdir_y = jnp.full(ncol, 0.5) / wind
    dt = 1800.0

    def _frozen_soil(G, dt_):     # keeps Ts_bc == T_soil_top every Picard pass
        return T_soil_top

    def _public(p):
        return tlc.compute_two_leaf_canopy_fluxes(
            T_soil_top=T_soil_top, forcing=forcing, canopy_config=cc,
            land_config=land_config, canopy_params=p,
            w_frac_rz=jnp.ones(ncol), wind_speed=wind,
            wind_dir_x=wdir_x, wind_dir_y=wdir_y,
            soil_thermal_fn=_frozen_soil, dt=dt)

    out = _public(params)
    assert out.gpp is not None
    assert bool(jnp.all(jnp.isfinite(out.gpp)))
    assert float(out.gpp[0]) > 0.0                         # lit canopy assimilates

    # ---- Reconstruct the converged state with the SAME public helpers ----
    LAI = params.LAI
    z0m, displa = compute_aerodynamics(params.hc, LAI, params.rz0m, params.rd)
    z_ref = jnp.maximum(jnp.full(ncol, land_config.z_ref),
                        displa + 10.0 * z0m + 2.0)
    SZA = jnp.degrees(jnp.arccos(jnp.clip(forcing.cos_zenith, 0.0, 1.0)))
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        forcing.sw_down, forcing.cos_zenith)
    sw = canopy_shortwave_rt(
        PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV, SZA, LAI, params.CI,
        params.ALB_VIS, params.ALB_NIR, params.Vcmax25_C3_leaf,
        params.Vcmax25_C4_leaf, params.kn, params.FNonVeg)
    Ta = forcing.T_lowest
    Ps = forcing.p_surface
    q_atm = forcing.q_lowest
    Ca = forcing.co2_ppmv
    q_c_init = 0.5 * (sat_specific_humidity(T_soil_top, Ps) + q_atm)
    Ci_init = Ca * tlc._CI_CA_C3          # chi = _CI_CA_C3 - _..._MINUS_C4*fC4, fC4=0
    x0 = jnp.stack([Ta, Ta, Ci_init, Ci_init, Ta, q_c_init], axis=-1)  # (ncol, 6)
    bundle = CanopyForcingBundle(
        LAI=LAI, SZA=SZA, La=forcing.lw_down,
        epsf=jnp.full(ncol, cc.epsf), epss=jnp.full(ncol, cc.epss),
        fSun=sw.fSun, APAR_Sun=sw.APAR_Sun, APAR_Sh=sw.APAR_Sh,
        Vcmax25_Sun=sw.Vcmax25_C3Sun, Vcmax25_Sh=sw.Vcmax25_C3Sh,
        Vcmax25_C4Sun=sw.Vcmax25_C4Sun, Vcmax25_C4Sh=sw.Vcmax25_C4Sh,
        ASW_Sun=sw.ASW_Sun, ASW_Sh=sw.ASW_Sh, ASW_Soil=sw.ASW_Soil,
        Ts_bc=T_soil_top, Ca=Ca, Ps=Ps, Ta=Ta,
        lam=jnp.full(ncol, constants.L_v), Cp=jnp.full(ncol, constants.c_pd),
        rhoa=forcing.rho_lowest, Tv_atm=Ta * (1.0 + tlc._VIRT_T_COEF * q_atm),
        q_atm=q_atm,
        m=params.m_C3, b0=params.b0_C3, alf=params.alf, TgC=params.TgC,
        fC4=params.fC4, fStress_soil=jnp.ones(ncol),
        ur=wind, CI=params.CI, z0m=z0m, displa=displa, z0=z_ref,
        cv=jnp.full(ncol, cc.cv), d_leaf=jnp.full(ncol, tlc._DEFAULT_D_LEAF),
        r_soil_surface=jnp.zeros(ncol))
    x_star, _ = jax.vmap(lambda x, b: solve_canopy_closure(x, b, cc))(x0, bundle)
    d = jax.vmap(lambda x, b: canopy_forward(
        x, b, cc.LE_module, cc.stomatal_model, cc.le_cap_mode,
        cc.use_ta_for_photosynthesis))(x_star, bundle)

    gross_sum = (d["Agross_Sun"] + d["Agross_Sh"]) * conv
    net_sum = (d["An_Sun"] + d["An_Sh"]) * conv

    # (a) public gpp == GROSS aggregate (bit-faithful reconstruction).
    assert jnp.allclose(out.gpp, gross_sum, rtol=1e-4, atol=1e-12)
    # (b) GROSS strictly exceeds NET in the light, and the public gpp is NOT
    #     the net aggregate (would fail on a reverted gpp line).
    assert float(gross_sum[0]) > float(net_sum[0])
    assert not bool(jnp.allclose(out.gpp, net_sum, rtol=1e-4, atol=1e-12))

    # (c) a finite, NONZERO gradient flows through the exported gpp path.
    def _sum_gpp(vc):
        p = params._replace(Vcmax25_C3_leaf=jnp.full(ncol, vc))
        return jnp.sum(_public(p).gpp)

    g = jax.grad(_sum_gpp)(60.0)
    assert jnp.isfinite(g)
    assert abs(float(g)) > 0.0
