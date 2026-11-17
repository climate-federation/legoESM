"""LMD94-style K-Profile Parameterization (KPP).

Boundary-layer parameterization following Large, McWilliams & Doney (1994)
with:

- Bulk Richardson number BL-depth diagnosis with linear interpolation
  of the crossing depth between model levels.
- Turbulent velocity scales w_s(sigma) from surface forcing (u_star, B_f).
- Cubic shape function G(sigma) = sigma * (1 - sigma)^2.
- Non-local tracer transport for unstable (convective) conditions only.
- Interior mixing: Richardson-number dependent + convective instability
  enhancement for statically unstable layers below the BL.

The caller should provide surface wind stress and buoyancy flux when
available.  If tau_x/tau_y are None, a simplified u_star proxy from
surface speed is used.

Faithfulness
------------
``tests/ocean/unit/test_kpp_lmd94_faithful.py`` pins the LMD94 closed forms to
round-off (rel 1e-12) against an INDEPENDENT reimplementation with the published
Appendix B constants, and canaries the ``KPPConfig`` defaults against them.

FAITHFUL to LMD94:
- ``_kpp_velocity_scales`` — the App. B similarity scales w_m/w_s in all three
  regimes.  In THIS module's sign convention (zeta = d/L_MO, negative for stable
  forcing): stable (kappa*u*/(1+5*|zeta|)), weakly unstable ((1+16|zeta|)^{1/4,1/2}
  Businger-Dyer), and convective (kappa*(a*u*^3 + c*kappa*B_f*d)^{1/3}), with the
  momentum/scalar joins at |zeta| = zeta_m = 0.2 and zeta_s = 1.0 and the
  surface-layer cap d_eff = min(d, epsilon*h).
- ``_kpp_unresolved_shear_variance`` — the Eq. 23 V_t^2 with the explicit
  (-beta_T)^1/2 prefactor and Cv = 1.6 (a prior 2.236x-too-large bug is fixed).
- ``_boundary_layer_depth`` — the Eq. 21 bulk-Richardson TERM.  Two structures
  (``KPPConfig.bld_structure``): ``"legacy"`` differences top-cell potential
  density (p = 0) and picks the crossing by a sigmoid-weighted average of
  cell-centre depths; ``"cvmix"`` follows CVMix/MOM6 (surface-layer averaged
  reference at the local pressure, (1 - eps/2) Ri scaling, first-crossing
  quadratic interpolation) — see DEPARTURES.
- ``KPPConfig`` App. B constants (kappa, Ri_c=0.3, Cv=1.6, a_m, c_m, a_s, c_s,
  zeta_m, zeta_s, 16/5 Businger coefficients).

DEPARTURES (documented, canaried in the test):
- ``_kpp_shape_function`` G(sigma) = sigma*(1-sigma)^2 is the REDUCED cubic:
  G(1) = G'(1) = 0, so both K_bl and dK_bl/dz vanish at the BL base.  The full
  LMD94 App. B / Eq. D cubic G = sigma*(1 + a2*sigma + a3*sigma^2) fixes a2, a3
  to MATCH the boundary-layer diffusivity AND its derivative to the interior K at
  sigma = 1; the reduced form cannot represent a nonzero interior-matched base
  value/slope, so it omits that matching construction (base entrainment is
  under-represented whenever the interior K/dK is nonzero there).
- ``bld_structure="legacy"``: the BL-depth crossing is a sigmoid-weighted
  average of cell-CENTRE depths; at the hard limit it is the centre of the
  first supercritical cell for a single crossing (the average of the
  upward-crossing centres for several), never interpolated; the BL diffusivity is
  evaluated at centres and averaged to interfaces.
- ``bld_structure="cvmix"``: the first-crossing selection is smooth
  (sigmoid of sharpness ``crossing_sharpness`` on Ri_b - Ri_c); at the hard
  limit it reproduces CVMix ``compute_OBL_depth`` (quadratic) except at
  Ri_b == Ri_c exactly, where the sigmoid gives 1/2 and CVMix's strict ``>``
  gives 0.
- AD-safety floors/caps (d_eff cap, w >= 1e-10, N^2 >= 0, base clips) and the
  optional Langmuir-turbulence enhancement (off by default).

References
----------
- Large, W. G., McWilliams, J. C., & Doney, S. C. (1994). Oceanic
  vertical mixing: A review and a model with a nonlocal boundary layer
  parameterization. Rev. Geophys., 32, 363-403.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.ocean.eos import (
    compute_buoyancy_frequency,
)
from legoesm.ocean.eos import (
    rho_0 as rho_0_ref,
)
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing._shared import (
    eos_salinity,
    richardson_number,
)
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.physics.vertical_mixing.tke import (
    nemo_tke_effective_ice_fraction,
)
from legoesm.ocean.vertical import OceanZStarCoordinate

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "LMD94 K-Profile Parameterization: diagnose the boundary-layer depth "
        "from a bulk Richardson criterion, set K_v/A_v in the boundary layer "
        "from similarity velocity scales and a cubic shape function, add "
        "non-local (counter-gradient) tracer transport for convective columns, "
        "and Ri-dependent interior mixing below."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "degC", "S": "psu", "rho": "kg/m^3",
        "eta": "m", "jacobian": "1 (z-star dimensionless)",
        "tau_x": "N/m^2", "tau_y": "N/m^2", "B_f": "m^2/s^3",
        "Q_sfc_T": "degC m/s", "Q_sfc_S": "psu m/s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "K_v": "m^2/s", "A_v": "m^2/s",
    },
    "sign_convention": (
        "Diffusivities K_v, A_v >= 0; B_f>0 is destabilising (convective) and "
        "wind/buoyancy forcing set the boundary-layer depth; the down-gradient "
        "diffusion and the non-local (counter-gradient) transport it applies "
        "are flux-form redistributions (the non-local flux vanishes at the "
        "surface and BL base), with surface fluxes applied separately; z "
        "positive up. The non-local (counter-gradient) tracer tendencies are "
        "ALWAYS applied; the local down-gradient diffusion is applied when "
        "apply_diffusion=True, otherwise the K_v/A_v profiles are handed to the "
        "implicit solver (which applies them conservatively)."
    ),
    # Both the always-applied non-local transport (vanishing at the surface and
    # BL base) and the flux-form local diffusion (no-flux interior BC, surface
    # fluxes separate) are conservative redistributions, so the scheme conserves
    # column-integrated heat (energy), salt and momentum whether the local
    # diffusion is applied here or deferred to the implicit solver.
    "conserves": ["energy", "salt", "momentum"],
    "differentiable": True,
    "reference": "Large, McWilliams & Doney (1994), Rev. Geophys. 32, 363-403",
    "idealized_test": (
        "tests/ocean/unit/test_kpp_langmuir.py + "
        "tests/ocean/unit/test_vmix_k_profiles_direct.py — wind/convective "
        "forcing deepens the boundary layer and enhances K within it; a "
        "stratified rest column gives near-background K."
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def _kpp_ice_attenuation(ice_frac, eice):
    """Under-ice attenuation factor ``(1 - eff)`` for the KPP velocity scales.

    KPP analogue of the TKE closure's ice-fraction suppression: compact sea
    ice caps the surface, so the surface-forcing-driven turbulent velocity
    scales — and hence both the boundary-layer depth (via ``V_t^2``) and the
    mixing coefficients — are reduced under ice.  ``eice`` uses NEMO's
    ``nn_eice`` numbering even though NEMO applies the control to the
    Langmuir/wave-TKE sources while KPP applies it to the broader w-scale:

    - ``0`` (default, BIT-IDENTICAL): no attenuation (returns 1.0).
    - ``1``: eff = tanh(10*fi)     -> factor 1 - tanh(10*fi).
    - ``2``: eff = fi              -> factor 1 - fi.
    - ``3``: eff = min(4*fi, 1)    -> factor max(0, 1 - 4*fi)  (matches NEMO
      nn_eice=3; mixing fully suppressed at fi >= 0.25).

    ``ice_frac`` None (no coupler ice field) -> 1.0.  Unknown ``eice`` raises
    even when off/None (contract: validate the STATIC config value first).
    """
    if eice not in (0, 1, 2, 3):
        raise ValueError(
            f"Unknown KPPConfig.eice={eice!r}; expected NEMO nn_eice "
            "0, 1, 2 or 3.")
    if eice == 0 or ice_frac is None:
        return 1.0
    eff = nemo_tke_effective_ice_fraction(ice_frac, eice)
    return jnp.maximum(1.0 - eff, 0.0)[..., jnp.newaxis]


def _kpp_velocity_scales(u_star, B_f, d, h_bl_col, cfg, ice_frac=None):
    """LMD94 (Appendix B) turbulent velocity scales at depth(s) ``d``.

    Returns the momentum (``w_m``) and scalar (``w_s``) similarity velocity
    scales for a boundary layer of depth ``h_bl_col``.  Factored so the
    bulk-Richardson ``V_t^2`` (in ``_boundary_layer_depth``) and the
    boundary-layer diffusivities (in ``kpp_vertical_mixing``) share ONE
    velocity-scale implementation instead of duplicating it.

    ``ice_frac`` (0-1 sea-ice concentration, or None) drives the under-ice
    attenuation ``(1 - eff)`` (``KPPConfig.eice``; NEMO nn_eice) applied to the
    FINAL ``w_m``/``w_s``: it shrinks both the mixing coefficients AND ``V_t^2``
    (so the bulk-Ri boundary layer shoals), the KPP analogue of the TKE
    closure's under-ice wave-TKE suppression.  ``eice=0`` -> factor 1 ->
    bit-identical.

    SHOALING SCOPE (codex): reducing ``w_s`` shoals ``h_bl`` for the intended
    stable, monotone bulk-Ri column (lower positive ``V_t^2`` -> higher positive
    ``Ri_b`` -> the ``Ri_b > Ri_crit`` crossing moves up).  It is NOT an
    unconditional guarantee: where the local density difference is negative or
    the Ri profile is non-monotone, the soft multi-crossing weighting can move
    ``h_bl`` the other way.  The Arctic winter halocline (strongly stable,
    monotone) is the target regime; the sign is validated there.

    Parameters
    ----------
    u_star, B_f : (...,) friction velocity [m/s] and surface buoyancy forcing
        [m^2/s^3] (positive = destabilising, this module's sign convention).
    d : (..., nlev) depth(s) below the surface [m, positive down].
    h_bl_col : (..., 1) boundary-layer depth [m].

    Returns
    -------
    (w_m, w_s) : each (..., nlev) [m/s].
    """
    kappa = cfg.kappa_vk
    # Surface-layer cap d_eff = min(d, epsilon*h): the similarity scale is held
    # constant below sigma = epsilon (LMD94 App. A/B).
    d_eff = jnp.minimum(d, cfg.epsilon_lmd * h_bl_col)
    B_f_e = B_f[..., jnp.newaxis]
    ustar_e = u_star[..., jnp.newaxis]
    # zeta = d/L_MO with B_f in the numerator (CVMix form): no buoyancy-flux floor
    # is needed (u* > 0 on both u* paths), and weak fluxes keep their own zeta.
    zeta = d_eff * kappa * B_f_e / ustar_e ** 3
    abs_zeta = jnp.abs(zeta)
    Bf_pos = jnp.maximum(B_f_e, 0.0)
    is_unstable = B_f_e > 0.0
    base16 = jnp.maximum(1.0 + cfg.businger_unstable_coeff * abs_zeta, 1.0)
    w_m_weak = kappa * ustar_e * jnp.power(base16, 0.25)
    w_s_weak = kappa * ustar_e * jnp.power(base16, 0.5)
    # Convective scales (kappa OUTSIDE the cube root); floored base keeps the
    # cube root + gradient finite near the join (F-OCEAN-2 pattern).
    w_m_conv = kappa * jnp.power(
        jnp.maximum(cfg.a_m * ustar_e ** 3 + cfg.c_m * kappa * Bf_pos * d_eff, 1e-30),
        1.0 / 3.0,
    )
    w_s_conv = kappa * jnp.power(
        jnp.maximum(cfg.a_s * ustar_e ** 3 + cfg.c_s * kappa * Bf_pos * d_eff, 1e-30),
        1.0 / 3.0,
    )
    w_m_unstable = jnp.where(abs_zeta <= cfg.zeta_m_abs, w_m_weak, w_m_conv)
    w_s_unstable = jnp.where(abs_zeta <= cfg.zeta_s_abs, w_s_weak, w_s_conv)
    # Shared stable suppression: zeta < 0 for stable forcing under this sign
    # convention, so max(-zeta, 0) drives the suppression.
    w_stable = (kappa * ustar_e
                / jnp.maximum(1.0 + cfg.businger_stable_coeff * jnp.maximum(-zeta, 0.0), 1.0))
    w_m = jnp.maximum(jnp.where(is_unstable, w_m_unstable, w_stable), 1e-10)
    w_s = jnp.maximum(jnp.where(is_unstable, w_s_unstable, w_stable), 1e-10)
    # Under-ice attenuation (NEMO nn_eice; KPPConfig.eice) — applied to the
    # final scales so it flows into BOTH V_t^2 (BL depth) and the K profiles.
    # The 1e-10 floors above are re-imposed so a full-ice factor of 0 does not
    # produce an exactly-zero scale that would divide-by-zero downstream.
    _att = _kpp_ice_attenuation(ice_frac, getattr(cfg, "eice", 0))
    w_m = jnp.maximum(w_m * _att, 1e-10)
    w_s = jnp.maximum(w_s * _att, 1e-10)
    return w_m, w_s


def _kpp_shape_function(sigma):
    """LMD94 boundary-layer shape function G(sigma) = sigma*(1-sigma)^2.

    This is the REDUCED KPP cubic: the full LMD94 (App. B / Eq. D) is
    G(sigma) = sigma*(1 + a2*sigma + a3*sigma^2) with a2, a3 fixed so that both
    the boundary-layer diffusivity AND its vertical derivative MATCH the interior
    K at sigma = 1.  Here G(1) = 0 AND G'(1) = 0, so both K_bl and dK_bl/dz vanish
    at the boundary-layer base; the reduced form cannot represent a nonzero
    interior-matched base value/slope, so it omits that matching construction (a
    documented faithfulness gap — base entrainment is under-represented whenever
    the interior K/dK is nonzero there).  ``sigma`` is clipped to [0, 1] first
    (outside the BL, G = 0).  G peaks at sigma = 1/3 with G(1/3) = 4/27.
    """
    sigma_clip = jnp.clip(sigma, 0.0, 1.0)
    return sigma_clip * (1.0 - sigma_clip) ** 2


def _kpp_unresolved_shear_variance(N, d, w_s, cfg, eps):
    """LMD94 Eq. 23 unresolved-shear velocity variance V_t^2 [m^2/s^2].

        V_t^2(d) = Cv * (-beta_T)^1/2 / (Ri_c * kappa^2) * (c_s*eps_lmd)^-1/2
                   * d * N * w_s(d)

    The (-beta_T)^1/2 = sqrt(0.2) prefactor (``cfg.neg_beta_T``) is applied
    EXPLICITLY; Cv keeps its standalone LMD94 value 1.6 (it does NOT absorb
    sqrt(0.2)).  The d*N*w_s factor [m * 1/s * m/s] makes V_t^2 a velocity
    squared, dimensionally additive with the resolved shear du^2+dv^2.  Only
    stable stratification contributes (the caller passes N = sqrt(max(N^2, 0))).
    """
    return (cfg.Cv * cfg.neg_beta_T ** 0.5 * N * d * w_s
            / (cfg.Ri_crit * cfg.kappa_vk ** 2
               * jnp.sqrt(jnp.maximum(cfg.c_s * cfg.epsilon_lmd, eps))))


_BLD_STRUCTURES = ("legacy", "cvmix")
# CVMix compute_bulk_Richardson: a zero denominator gives num * 1e10.
_CVMIX_ZERO_DENOM_RI_FACTOR = 1e10


def _mom6_surface_layer_deltas(T, S, u, v, dz, z_depth, cfg, eos_fn, g):
    """MOM6 KPP_compute_BLD reference differences (MOM_CVMix_KPP.F90 1300-1428).

    For each centre k the reference is the thickness average of T, S, u, v over
    the surface layer [0, SL_k], SL_k = epsilon * max(d_k, top-cell thickness),
    and both density operands are evaluated at the pressure of the top
    interface of k (rho_0 * g * z_top_k, Boussinesq).  Returns
    (delta_rho [kg/m^3], delta_V2 [m^2/s^2]), each (..., nlev).
    """
    z_bot = jnp.cumsum(dz, axis=-1)
    z_top = z_bot - dz
    sl = cfg.epsilon_lmd * jnp.maximum(z_depth, dz[..., :1])
    nlev = z_bot.shape[-1]
    # Cell holding SL_k: first bottom interface strictly below it (MOM6's
    # overlap min(max(0, SL - z_top), dz) summed over cells).
    j = jax.vmap(lambda zb, x: jnp.searchsorted(zb, x, side="right"))(
        z_bot.reshape(-1, nlev), sl.reshape(-1, nlev)).reshape(sl.shape)
    j = jnp.minimum(j, nlev - 1)
    wet = sl > 0.0
    sl_safe = jnp.where(wet, sl, 1.0)

    def sl_mean(q):
        cum_top = jnp.cumsum(q * dz, axis=-1) - q * dz
        part = (jnp.take_along_axis(cum_top, j, axis=-1)
                + jnp.take_along_axis(q, j, axis=-1)
                * (sl - jnp.take_along_axis(z_top, j, axis=-1)))
        # Dry column (zero thickness): reference = top cell, so delta = 0 and no
        # zero salinity reaches a sqrt(S) EOS.
        return jnp.where(wet, part / sl_safe, q[..., :1])

    S_eos = eos_salinity(S, eos_fn)
    p_top = rho_0_ref * g * z_top
    delta_rho = (eos_fn(T, S_eos, p_top)
                 - eos_fn(sl_mean(T), sl_mean(S_eos), p_top))
    delta_V2 = (u - sl_mean(u)) ** 2 + (v - sl_mean(v)) ** 2
    return delta_rho, delta_V2


def _cvmix_bulk_richardson(delta_rho, delta_V2, V_t2, z_depth, cfg, g):
    """CVMix compute_bulk_Richardson: (1 - eps/2) * d * g*delta_rho/rho_0 over
    (delta_V2 + V_t2); a zero denominator gives num * 1e10."""
    num = (1.0 - 0.5 * cfg.epsilon_lmd) * g * delta_rho * z_depth / rho_0_ref
    den = delta_V2 + V_t2
    return jnp.where(den != 0.0, num / jnp.where(den != 0.0, den, 1.0),
                     num * _CVMIX_ZERO_DENOM_RI_FACTOR)


def _cvmix_bracket_offset(f0, fH, m, H, beta):
    """Offset in [0, H] of the crossing in one bracket of CVMix's quadratic.

    q(x) = a*x^2 + m*x + q0 through (0, q0), (H, qH) with slope m at 0, where
    the end values are clamped smoothly, q0 = -softplus(-beta*f0)/beta < 0 and
    qH = softplus(beta*fH)/beta > 0 (beta = crossing_sharpness).  Every bracket
    then has exactly one root in (0, H) and it is simple, so the offset is
    smooth in (f0, fH, m) (a root-selection rule on the raw quadratic has fold
    seams).  At the hard limit the clamps are exact for the bracket CVMix
    selects (f0 <= 0 < fH), giving CVMix's root.  When a clamp underflows to 0
    the bracket's sigmoid weight underflows with it.
    """
    q0 = -jax.nn.softplus(-beta * f0) / beta
    qH = jax.nn.softplus(beta * fH) / beta
    H_safe = jnp.where(H > 0.0, H, 1.0)
    a = (qH - q0 - m * H) / H_safe ** 2
    disc = m * m - 4.0 * a * q0
    # sqrt has an infinite derivative at 0: double-where.
    root_d = jnp.where(disc > 0.0, jnp.sqrt(jnp.where(disc > 0.0, disc, 1.0)), 0.0)
    # The root is the upward zero: 2 q0 / (-m - sqrt(D)) for m > 0 (no
    # cancellation), else (-m + sqrt(D)) / (2a) with a > 0 (q0 < 0 < qH, m <= 0).
    r_pos = 2.0 * q0 / jnp.where(m > 0.0, -m - root_d, -1.0)
    r_neg = (-m + root_d) / jnp.where(a > 0.0, 2.0 * a, 1.0)
    return jnp.clip(jnp.where(m > 0.0, r_pos, r_neg), 0.0, H)


def _cvmix_first_crossing_depth(Ri_b, z_depth, cfg):
    """CVMix compute_OBL_depth_low (non-Stokes, quadratic), smoothed (RULE 5).

    s_k = sigmoid(crossing_sharpness * (Ri_b_k - Ri_c)) replaces CVMix's
    "first k with Ri_b_k > Ri_c"; the soft first-crossing weights
    w_k = s_k * prod_{j<k}(1 - s_j) select: k = 0 -> d_0; k >= 1 -> the root of
    CVMix's quadratic through (d_{k-1}, Ri_{k-1}), (d_k, Ri_k) with the
    backward-difference slope at d_{k-1} (zero slope for k = 1); no crossing
    (weight prod_j(1 - s_j)) -> d_{nlev-1}.  The hard limit (sharpness -> inf)
    is CVMix's depth.  The bracket offset is ``_cvmix_bracket_offset``
    (closed form, continuous in Ri_b).
    """
    s = jax.nn.sigmoid(cfg.crossing_sharpness * (Ri_b - cfg.Ri_crit))
    surv = jnp.cumprod(1.0 - s, axis=-1)
    pad_axes = ((0, 0),) * (s.ndim - 1)
    surv_prev = jnp.pad(surv[..., :-1], (*pad_axes, (1, 0)), constant_values=1.0)
    w = s * surv_prev
    w_none = surv[..., -1]

    d0, d1 = z_depth[..., :-1], z_depth[..., 1:]
    y0, y1 = Ri_b[..., :-1], Ri_b[..., 1:]
    H = d1 - d0
    dprev = d0[..., 1:] - d0[..., :-1]
    slope_prev = (y0[..., 1:] - y0[..., :-1]) / jnp.where(dprev > 0.0, dprev, 1.0)
    m = jnp.pad(slope_prev, (*pad_axes, (1, 0)))   # bracket 1: zero slope
    x = _cvmix_bracket_offset(y0 - cfg.Ri_crit, y1 - cfg.Ri_crit, m, H,
                              cfg.crossing_sharpness)
    h_bracket = d0 + x

    return (w[..., 0] * z_depth[..., 0]
            + jnp.sum(w[..., 1:] * h_bracket, axis=-1)
            + w_none * z_depth[..., -1])


def _boundary_layer_depth(
    rho: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    u_star: jnp.ndarray,
    B_f: jnp.ndarray,
    cfg: KPPConfig,
    g: float = constants.g,
    h_bl_prev: jnp.ndarray | None = None,
    eos_fn=None,
    ice_frac: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Estimate boundary layer depth h via bulk Richardson number.

    ``bld_structure="legacy"``: sigmoid-weighted average of cell-centre depths
    around the Ri_b = Ri_crit crossing (hard limit: centre of the first
    supercritical cell for one crossing, average of upward-crossing centres for
    several; never interpolated).  ``"cvmix"``: CVMix first-crossing
    quadratic interpolation, smoothed (``_cvmix_first_crossing_depth``).

    Returns shape (...) boundary layer depth [m, positive downward].
    """
    eps = _EPS

    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    # Depth of cell centers below surface (positive downward)
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual

    if cfg.bld_structure not in _BLD_STRUCTURES:
        raise ValueError(
            f"Unknown KPPConfig.bld_structure={cfg.bld_structure!r} "
            f"(boundary-layer structure variant); expected one of {_BLD_STRUCTURES}.")
    # Buoyancy (density) and velocity differences from the surface for the
    # bulk Richardson number.  LMD94 Eq. 21 requires density referenced to a
    # COMMON pressure: the previous ``rho - rho[..., :1]`` differenced two
    # IN-SITU densities rho(T,S,p_hydro) evaluated at DIFFERENT hydrostatic
    # pressures, so in warm/deep/stratified columns the compressibility +
    # thermobaric contribution inflated delta_rho and the boundary layer was
    # diagnosed too shallow (the documented low-latitude MLD-too-shallow
    # bias).  "legacy" references every level to the SURFACE pressure (p = 0
    # Pa => potential density) and to the top cell; MOM6 instead uses the
    # local pressure and a surface-layer average ("cvmix", see
    # _mom6_surface_layer_deltas).  ``eos_fn`` defaults to
    # ``wright_eos`` so the reference density matches how the in-situ ``rho``
    # was built when the caller does not thread an explicit EOS.  (N^2 / V_t^2
    # below still use the in-situ ``rho`` gradient — a smaller secondary
    # effect, left unchanged so this fix stays surgical.)
    if eos_fn is None:
        from legoesm.ocean.eos import wright_eos
        eos_fn = wright_eos
    # Floored salinity for sqrt(S) EOSes (infinite dRho/dS at S = 0), as
    # the density above; Wright as is (#2243).
    if cfg.bld_structure == "cvmix":
        delta_rho, delta_V2 = _mom6_surface_layer_deltas(
            T, S, u, v, dz_actual, z_depth, cfg, eos_fn, g)
    else:
        rho_surf_ref = eos_fn(T, eos_salinity(S, eos_fn), jnp.zeros_like(T))
        delta_rho = rho_surf_ref - rho_surf_ref[..., :1]
        delta_u = u - u[..., :1]
        delta_v = v - v[..., :1]
        delta_V2 = delta_u**2 + delta_v**2

    # LMD94 Eq. 23 unresolved-shear variance:
    #   V_t^2(d) = Cv * (-beta_T)^1/2 / (Ri_c * kappa^2) * (c_s*eps)^-1/2
    #              * d * N * w_s(d)                                  [m^2/s^2]
    # The (-beta_T)^1/2 = sqrt(0.2) = 0.4472 prefactor is applied EXPLICITLY
    # (``cfg.neg_beta_T``); Cv keeps its standard LMD94 value (1.6).  A prior
    # comment claimed (-beta_T)^1/2 was "folded into Cv" — it was not, so
    # V_t^2 was 1/sqrt(0.2) = 2.236x too large, biasing the diagnosed boundary-
    # layer depth deep in weak-shear convective columns (where V_t^2 dominates
    # the resolved shear).  The d*N*w_s(d) factor — m * (1/s) *
    # (m/s) — makes V_t^2 a velocity-squared, dimensionally additive with
    # delta_V2 = du^2 + dv^2 [m^2/s^2].  The turbulent velocity scale w_s(d)
    # [m/s] is essential and was previously dropped (the old non-canonical
    # form max(Ri_c*h - d, 0)*d/h was a LENGTH, giving V_t^2 in [m/s]).  w_s
    # is evaluated with the previous-step boundary-layer depth (h_bl_prev) so
    # V_t^2 does not depend on the very h it helps to compute.  N is floored
    # at sqrt(1e-30) for an AD-safe gradient (no max()-induced 0*inf).
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    N2_full = jnp.concatenate([N2[..., :1], N2], axis=-1)
    # N = sqrt(max(N^2, 0)): only STABLE stratification (N^2 > 0) contributes to
    # the unresolved-shear variance (LMD94 / CVMix); convective layers (N^2 < 0)
    # give V_t^2 = 0 there.  (Floored at 1e-30 for an AD-safe gradient.)
    N_full = jnp.sqrt(jnp.maximum(N2_full, 1e-30))  # [1/s]
    max_depth = z_depth[..., -1]
    h_est = max_depth if h_bl_prev is None else h_bl_prev
    h_safe = jnp.maximum(h_est[..., jnp.newaxis], eps)
    _, w_s_vt = _kpp_velocity_scales(u_star, B_f, z_depth, h_safe, cfg,
                                     ice_frac=ice_frac)
    V_t2 = _kpp_unresolved_shear_variance(N_full, z_depth, w_s_vt, cfg, eps)

    if cfg.bld_structure == "cvmix":
        Ri_b = _cvmix_bulk_richardson(delta_rho, delta_V2, V_t2, z_depth, cfg, g)
        h = _cvmix_first_crossing_depth(Ri_b, z_depth, cfg)
        return jnp.maximum(h, dz_actual[..., 0])

    # Bulk Richardson number
    Ri_b = (g * delta_rho * z_depth) / (
        rho_0_ref * jnp.maximum(delta_V2 + V_t2, eps)
    )

    # --- Differentiable soft interpolation of crossing depth ---
    # Instead of argmax (non-differentiable), use a sigmoid-weighted
    # average over all levels.  Each level contributes a weight
    # proportional to how much Ri_b crosses Ri_crit there.
    #
    # Weight at level k = sigmoid(sharpness * (Ri_b[k] - Ri_crit))
    #                    - sigmoid(sharpness * (Ri_b[k-1] - Ri_crit))
    # This is ~1 at the crossing level and ~0 elsewhere.
    sharpness = cfg.crossing_sharpness
    sig = jax.nn.sigmoid(sharpness * (Ri_b - cfg.Ri_crit))  # (..., nlev)

    # Crossing weight: difference of adjacent sigmoid values.  ``jnp.pad``
    # along the trailing axis is one HLO op; the previous
    # ``concatenate([zeros_like(sig[..., :1]), sig[..., :-1]])`` allocated
    # a fresh zero buffer and concatenated.
    pad_axes = ((0, 0),) * (sig.ndim - 1)
    sig_prev = jnp.pad(sig[..., :-1], (*pad_axes, (1, 0)))
    w_cross = sig - sig_prev  # (..., nlev), peaks at crossing level
    w_cross = jnp.maximum(w_cross, 0.0)
    # ``w_sum`` and the column-stability mean both reduce over the
    # level axis — fuse into one stacked sum so XLA fires a single
    # column-axis kernel.
    _nlev = sig.shape[-1]
    _stack_pair = jnp.sum(jnp.stack([w_cross, sig], axis=-1), axis=-2)
    w_sum = _stack_pair[..., 0:1]  # keepdims=True equivalent
    column_stability = _stack_pair[..., 1] / _nlev  # mean = sum / nlev
    w_norm = w_cross / jnp.maximum(w_sum, eps)

    # Crossing-based depth estimate
    h_crossing = jnp.sum(w_norm * z_depth, axis=-1)  # (...)

    # Fallback for columns where Ri_b never crosses Ri_crit:
    # - If column is mostly unstable (sig ≈ 0): BL extends to full depth
    # - If column is mostly stable (sig ≈ 1): BL is one layer
    # ``column_stability`` already computed above (~0 = all unstable, ~1 = all stable).
    max_depth = z_depth[..., -1]
    min_depth = dz_actual[..., 0]
    h_fallback = (1.0 - column_stability) * max_depth + column_stability * min_depth

    # Blend: use crossing depth when crossing signal is strong, fallback otherwise
    crossing_strength = w_sum[..., 0]
    blend = jax.nn.sigmoid(cfg.crossing_sharpness * (crossing_strength - cfg.crossing_threshold))
    h = blend * h_crossing + (1.0 - blend) * h_fallback

    # At least one layer thick
    h = jnp.maximum(h, dz_actual[..., 0])

    return h


def kpp_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    eta: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: KPPConfig,
    g: float = constants.g,
    tau_x: jnp.ndarray | None = None,
    tau_y: jnp.ndarray | None = None,
    B_f: jnp.ndarray | None = None,
    Q_sfc_T: jnp.ndarray | None = None,
    Q_sfc_S: jnp.ndarray | None = None,
    h_bl_prev: jnp.ndarray | None = None,
    apply_diffusion: bool = True,
    dt: float | None = None,
    eos_fn=None,
    u_stokes: jnp.ndarray | None = None,
    ice_frac: jnp.ndarray | None = None,
) -> VerticalMixingOutput:
    """Apply LMD94-style KPP vertical mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    eta : array (6, n, n)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : KPPConfig
    g : float
    tau_x, tau_y : array (6, n, n) or None
        Surface wind stress [Pa]. If None, a proxy from surface speed is used.
    B_f : array (6, n, n) or None
        Surface buoyancy flux [m^2/s^3], positive = destabilizing (convective).
        If None, estimated from surface density gradient.
    Q_sfc_T : array (6, n, n) or None
        Surface kinematic heat flux [K*m/s] for non-local transport (LMD94
        Eq. 19).  If None, falls back to diagnosed K_sfc * dT/dz proxy.
    Q_sfc_S : array (6, n, n) or None
        Surface kinematic salt flux [PSU*m/s]. Same convention as Q_sfc_T.
    h_bl_prev : array (6, n, n) or None
        BL depth from the previous time step [m, positive downward].
        Used to break the implicit V_t-h_bl coupling in the Ri_b diagnosis
        (LMD94 Eq. 23).  If None, uses the full column depth as estimate.

    Returns
    -------
    VerticalMixingOutput
    """
    eps = _EPS

    # --- Friction velocity ---
    if tau_x is not None and tau_y is not None:
        # Proper u_star from wind stress: u_star = sqrt(|tau| / rho_0)
        tau_mag = jnp.sqrt(tau_x**2 + tau_y**2 + eps)
        u_star = jnp.sqrt(tau_mag / rho_0_ref)
    else:
        # Simplified proxy: u_star ~ ustar_speed_ratio * |U_surface|
        speed_sfc = jnp.sqrt(u[..., 0]**2 + v[..., 0]**2 + eps)
        u_star = jnp.maximum(speed_sfc * cfg.ustar_speed_ratio, 1e-4)  # coeff-ok: u_star floor [m/s]

    # --- Surface buoyancy flux ---
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    if B_f is None:
        # When the caller does not supply a surface buoyancy flux, set
        # B_f = 0 (no convective non-local transport).  The previous
        # ``K_bg · g/ρ₀ · drho_dz_sfc`` proxy is ~1e-10 m²/s³ for
        # realistic density gradients (~1e-3 kg/m⁴), which is 2-4
        # orders of magnitude below realistic destabilizing B_f
        # (1e-8 to 1e-7 m²/s³).  A wrong-magnitude fallback masks
        # missing surface forcing without the user noticing — a
        # zero-flux fallback fails closed (forward integration is
        # consistent with no surface buoyancy forcing) and forces the
        # caller to provide B_f explicitly when convective KPP matters.
        B_f = jnp.zeros_like(rho[..., 0])

    # --- Boundary layer depth ---
    h_bl = _boundary_layer_depth(
        rho, T, S, u, v, z_coord, jacobian, u_star, B_f, cfg, g,
        h_bl_prev=h_bl_prev, eos_fn=eos_fn, ice_frac=ice_frac,
    )

    # --- Depth coordinate ---
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual
    sigma = z_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)

    # --- Shape function G(sigma) = sigma * (1 - sigma)^2 (LMD94 reduced cubic) ---
    # See _kpp_shape_function: this is the non-matching cubic, a documented
    # faithfulness gap (G(1)=G'(1)=0 => under-represented base entrainment).
    G = _kpp_shape_function(sigma)
    sigma_clip = jnp.clip(sigma, 0.0, 1.0)

    # --- Turbulent velocity scales w_m (momentum) / w_s (scalar) ---
    # LMD94 Appendix B; shared with the bulk-Richardson V_t^2 via
    # _kpp_velocity_scales (one implementation, no duplication).  The scalar
    # scale w_s uses the steeper unstable exponent so K_v > A_v (Pr_t < 1) in
    # convection (F-OCEAN-1), while w_m feeds the momentum viscosity.
    d = sigma_clip * h_bl[..., jnp.newaxis]
    w_m, w_s = _kpp_velocity_scales(
        u_star, B_f, d, h_bl[..., jnp.newaxis], cfg, ice_frac=ice_frac,
    )

    # --- Langmuir turbulence enhancement (KPP-Langmuir) ---
    # Langmuir circulations (wind + Stokes-drift shear) enhance surface
    # boundary-layer mixing.  Multiply the KPP velocity scales by the
    # enhancement factor eps_L = sqrt(1 + C_L / La_t^2) >= 1, where La_t =
    # sqrt(u* / u_s0) is the turbulent Langmuir number (McWilliams & Sullivan
    # 2000; Li et al. 2016 CVMix).  With a surface Stokes-drift input
    # ``u_stokes`` (from a wave model / forcing) La_t is spatially resolved;
    # without one it falls back to the fully-developed-sea value
    # ``langmuir_number_default`` (~0.3, Van Roekel et al. 2012) — a uniform
    # enhancement where wind mixing dominates.  eps_L >= 1 always (C_L, La_t >
    # 0) so this only ENHANCES mixing, never reduces it; it scales the BL
    # diffusivity K = h*w*G (conservation of the implicit solve is unaffected).
    # Gated on the static config bool (feature gating); default off leaves
    # w_m / w_s byte-identical to classical KPP.  The bulk-Richardson MLD
    # diagnosis (V_t^2) is intentionally NOT enhanced here — Langmuir
    # boundary-layer deepening is a documented refinement.
    if cfg.enable_langmuir:
        if u_stokes is not None:
            la_t = jnp.sqrt(u_star / jnp.maximum(u_stokes, 1e-4))  # coeff-ok: Stokes floor [m/s]
        else:
            la_t = jnp.full_like(u_star, cfg.langmuir_number_default)
        # Floor the radicand at 1.0 so eps_L >= 1 for ANY config value (a
        # traced-safe safety floor: Langmuir must ENHANCE, never reduce, mixing,
        # and it avoids a NaN sqrt if langmuir_coeff is set negative).  Both
        # langmuir params are tunable (tier 2) so they may be traced — a Python
        # branch/raise would break the training trace; the floor is a no-op for
        # the spec-bounded positive range.
        eps_langmuir = jnp.sqrt(jnp.maximum(
            1.0, 1.0 + cfg.langmuir_coeff / jnp.maximum(la_t, 1e-3) ** 2  # coeff-ok: La_t floor
        ))[..., jnp.newaxis]
        w_m = w_m * eps_langmuir
        w_s = w_s * eps_langmuir
    else:
        eps_langmuir = 1.0

    # --- BL viscosity (momentum, w_m) and diffusivity (scalar, w_s) ---
    K_bl_m_full = jnp.minimum(h_bl[..., jnp.newaxis] * w_m * G, cfg.K_max)
    K_bl_s_full = jnp.minimum(h_bl[..., jnp.newaxis] * w_s * G, cfg.K_max)

    # --- Interior mixing: Richardson-number dependent ---
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    # Interface spacing — also reused by the surface T/S gradient terms below.
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    # Interior Ri = N^2 / S^2 (#518: shared helper; no clip here — KPP clamps
    # downstream via Ri / Ri_0).
    Ri_int = richardson_number(N2, u, v, dz_actual, eps=eps, clip_negative=False)
    # LMD94 interior shear instability: K = K_0 * (1 - (Ri/Ri_0)^2)^3
    # for Ri < Ri_0, zero above.
    Ri_ratio = jnp.clip(Ri_int / cfg.Ri_0, 0.0, 1.0)
    Ri_shear_curve = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3

    # Interior static instability: enhanced mixing where N2 < 0
    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)

    # Build separate interior floors for tracer (K_v) and momentum (A_v).
    # The shear-instability + convective enhancement is shared, but the
    # background floors differ (K_bg = 1e-5 for tracers, A_bg = 1e-4 for
    # momentum).  Previously both branches used cfg.K_bg, which dropped
    # the momentum interior viscosity by an order of magnitude (codex
    # adversarial review iter-1, finding #6).
    K_interior = Ri_shear_curve + cfg.K_bg + K_conv
    A_interior = Ri_shear_curve + cfg.A_bg + K_conv

    # --- K at interfaces (average of full level K_bl), split m/s ---
    K_bl_m_half = 0.5 * (K_bl_m_full[..., :-1] + K_bl_m_full[..., 1:])
    K_bl_s_half = 0.5 * (K_bl_s_full[..., :-1] + K_bl_s_full[..., 1:])

    # sigma at interfaces
    if cfg.bld_structure == "cvmix":
        # CVMix SimpleShapes: K = h * w(sigma) * G(sigma) AT the interfaces.
        z_half_depth = jnp.cumsum(dz_actual, axis=-1)[..., :-1]
        # max(h, eps) guards h = 0 only; it does not clean a NaN h (a fully dry
        # column's h is NaN via the shared N^2 / V_t^2 0/0, in both structures).
        sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
        sig_i = jnp.clip(sigma_half, 0.0, 1.0)
        w_m_i, w_s_i = _kpp_velocity_scales(
            u_star, B_f, sig_i * h_bl[..., jnp.newaxis], h_bl[..., jnp.newaxis],
            cfg, ice_frac=ice_frac)
        G_i = _kpp_shape_function(sig_i)
        K_bl_m_half = jnp.minimum(
            h_bl[..., jnp.newaxis] * w_m_i * eps_langmuir * G_i, cfg.K_max)
        K_bl_s_half = jnp.minimum(
            h_bl[..., jnp.newaxis] * w_s_i * eps_langmuir * G_i, cfg.K_max)
    else:
        z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
        sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    in_bl = sigma_half < 1.0

    # Combine BL and interior.  Each branch uses its own background
    # floor (K_bg for tracers, A_bg for momentum) so the merged field
    # honors the configured background levels in BOTH the BL and the
    # interior.
    K_v = jnp.where(in_bl, K_bl_s_half + cfg.K_bg, K_interior)   # scalar  ← w_s
    A_v = jnp.where(in_bl, K_bl_m_half + cfg.A_bg, A_interior)   # momentum ← w_m
    K_v = jnp.minimum(K_v, cfg.K_max)
    A_v = jnp.minimum(A_v, cfg.K_max)

    # --- Apply diffusion ---
    # When ``apply_diffusion`` is False, the local diffusion tendency is
    # zeroed; the caller is expected to apply the K_v/A_v profiles via an
    # implicit (backward-Euler) solver after the explicit step.  The
    # non-local KPP transport (counter-gradient flux) below is *not* a
    # diffusion and is always returned in dT/dS.
    if apply_diffusion:
        # Pass ``dt`` (when provided) so the explicit-Euler CFL cap
        # added in clean_physics iter-5 fires inside
        # ``vertical_diffusion_variable_K``.  KPP's own ``cfg.K_max``
        # bounds K from above but cannot enforce ``K·dt/dz²≤½`` on
        # thin upper layers; the leaf cap is the safety net.
        vel = jnp.stack([u, v], axis=0)
        vel_tend = jax.vmap(
            lambda q: vertical_diffusion_variable_K(
                q, z_coord, jacobian, A_v, dt=dt,
            ),
            in_axes=0, out_axes=0,
        )(vel)

        tracers = jnp.stack([T, S], axis=0)
        tr_tend = jax.vmap(
            lambda q: vertical_diffusion_variable_K(
                q, z_coord, jacobian, K_v, dt=dt,
            ),
            in_axes=0, out_axes=0,
        )(tracers)
    else:
        zero_uv = jnp.zeros_like(u)
        vel_tend = jnp.stack([zero_uv, zero_uv], axis=0)
        zero_T = jnp.zeros_like(T)
        tr_tend = jnp.stack([zero_T, zero_T], axis=0)

    # --- Non-local flux for T, S (LMD94 Eq. 19) ---
    #
    # LMD94 defines a counter-gradient term:
    #   gamma_T(sigma) = C_s * Q_0 / (w_s(sigma) * h)   [K/m]
    # where Q_0 is the surface kinematic heat flux [K*m/s].
    #
    # The non-local tendency is  -d/dz(K_bl * gamma_T).
    # Substituting K_bl = h * w_s * G(sigma):
    #   K_bl * gamma_T = h * w_s * G * C_s * Q_0 / (w_s * h) = C_s * Q_0 * G(sigma)
    #
    # So the non-local tendency reduces to:
    #   dT/dt_nonlocal = -d/dz[ C_s * Q_0 * G(sigma) ]            [K/s]
    #
    # We discretize this as the vertical divergence of the non-local
    # flux F_nl = C_s * Q_0 * G(sigma) evaluated at interfaces.

    # Surface kinematic heat/salt flux for non-local transport (LMD94 Eq. 19).
    # Use the IMPOSED surface flux when available (from bulk formulas or
    # prescribed forcing).  Fall back to diagnosed K_sfc * dT/dz proxy
    # only when no external flux is provided (issue #168 bug 2).
    if Q_sfc_T is not None:
        Q_T = Q_sfc_T  # [K*m/s]
    else:
        dT_dz_sfc = (T[..., 0] - T[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_s_full[..., 0]
        Q_T = K_sfc * dT_dz_sfc

    if Q_sfc_S is not None:
        Q_S = Q_sfc_S  # [PSU*m/s]
    else:
        dS_dz_sfc = (S[..., 0] - S[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_s_full[..., 0]
        Q_S = K_sfc * dS_dz_sfc

    # Only apply non-local transport for unstable (convective) columns.
    is_unstable_col = B_f > 0.0

    # G(sigma) at interior interfaces (half levels between full levels)
    sigma_half_full = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    sigma_half_clip = jnp.clip(sigma_half_full, 0.0, 1.0)
    G_half = sigma_half_clip * (1.0 - sigma_half_clip) ** 2  # (..., nlev-1)

    # --- Temperature non-local tendency ---
    # Non-local flux at interfaces: F_nl = C_s * Q_T * G_half  [K*m/s]
    # G_half = 0 for sigma_half >= 1 (outside BL) so F_T automatically
    # vanishes below the BL — the divergence ``-dF/dz`` is naturally
    # restricted to the BL.  The previous in-BL mask
    # ``in_bl_full & is_unstable_col`` zeroed the compensating
    # tendency in the layer whose CENTER sigma >= 1 but whose TOP
    # interface sigma_half < 1, breaking column conservation when h_bl
    # cut through a grid cell (codex adversarial review iter-1,
    # finding #1).  Keep only the column-level ``is_unstable_col``
    # gate.
    F_T = cfg.gamma_T * Q_T[..., jnp.newaxis] * G_half  # (..., nlev-1)
    # Tendency = -dF/dz at full levels (zero-flux BCs at surface and bottom).
    # AD-safe divisor: dry columns have ``dz_actual = 0`` and the
    # column-level ``is_unstable_col`` mask scrubs the forward value,
    # but the 0/0 division produces NaN gradients in the backward
    # pass.  Safe denominator (``where dz>0, dz, 1``) gives clean
    # gradients while the where-mask still zeroes the forward output.
    dz_safe = jnp.where(dz_actual > 0.0, dz_actual, 1.0)
    dT_nonlocal_top = -F_T[..., :1] / dz_safe[..., :1]
    dT_nonlocal_int = (F_T[..., :-1] - F_T[..., 1:]) / dz_safe[..., 1:-1]
    dT_nonlocal_bot = F_T[..., -1:] / dz_safe[..., -1:]
    dT_nonlocal = jnp.concatenate(
        [dT_nonlocal_top, dT_nonlocal_int, dT_nonlocal_bot], axis=-1
    )  # (..., nlev)  [K/s]
    dT_nonlocal = jnp.where(
        is_unstable_col[..., jnp.newaxis] & (dz_actual > 0.0),
        dT_nonlocal, 0.0,
    )

    # --- Salinity non-local tendency ---
    F_S = cfg.gamma_S * Q_S[..., jnp.newaxis] * G_half  # (..., nlev-1)
    dS_nonlocal_top = -F_S[..., :1] / dz_safe[..., :1]
    dS_nonlocal_int = (F_S[..., :-1] - F_S[..., 1:]) / dz_safe[..., 1:-1]
    dS_nonlocal_bot = F_S[..., -1:] / dz_safe[..., -1:]
    dS_nonlocal = jnp.concatenate(
        [dS_nonlocal_top, dS_nonlocal_int, dS_nonlocal_bot], axis=-1
    )  # (..., nlev)  [psu/s]
    dS_nonlocal = jnp.where(
        is_unstable_col[..., jnp.newaxis] & (dz_actual > 0.0),
        dS_nonlocal, 0.0,
    )

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0] + dT_nonlocal,
        dS_dt=tr_tend[1] + dS_nonlocal,
        K_v=K_v,
        A_v=A_v,
    )
