"""Canopy radiative transfer for the two-leaf energy balance model.

Shortwave: Sellers (1985) / Ryu et al. (2011) two-stream approach.
  - Separates sunlit and shaded leaf fractions.
  - Handles PAR, NIR, and UV independently.
  - Distributes Vcmax25 using a nitrogen extinction profile.

Longwave: Ryu et al. (2011) extinction-based scheme.

SW decomposition: Liu-Jordan clearness-index direct/diffuse split.

All functions are pure JAX, JIT-compatible, and differentiable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants

# Module-local conversion factor (not a physical constant per se).
_APAR_CONVERSION = 4.56           # [W m-2] → [μmol m-2 s-1] for PAR


# ---------------------------------------------------------------------------
# Output NamedTuples
# ---------------------------------------------------------------------------

class CanopySWOutput(NamedTuple):
    """Outputs of the shortwave radiative transfer calculation."""
    fSun: jax.Array           # Sunlit fraction of LAI [-]
    APAR_Sun: jax.Array       # Absorbed PAR by sunlit leaves [μmol m-2 s-1]
    APAR_Sh: jax.Array        # Absorbed PAR by shaded leaves [μmol m-2 s-1]
    ASW_Sun: jax.Array        # Absorbed shortwave by sunlit leaves [W m-2]
    ASW_Sh: jax.Array         # Absorbed shortwave by shaded leaves [W m-2]
    ASW_Soil: jax.Array       # Absorbed shortwave by soil [W m-2]
    Vcmax25_C3Sun: jax.Array  # Canopy-integrated C3 Vcmax25, sunlit [μmol m-2 s-1]
    Vcmax25_C3Sh: jax.Array   # Canopy-integrated C3 Vcmax25, shaded
    Vcmax25_C4Sun: jax.Array  # Canopy-integrated C4 Vcmax25, sunlit
    Vcmax25_C4Sh: jax.Array   # Canopy-integrated C4 Vcmax25, shaded


class CanopyLWOutput(NamedTuple):
    """Outputs of the longwave radiative transfer calculation."""
    ALW_Sun: jax.Array     # Net absorbed LW by sunlit leaves [W m-2 ground]
    ALW_Sh: jax.Array      # Net absorbed LW by shaded leaves [W m-2 ground]
    ALW_Soil: jax.Array    # Net absorbed LW by soil [W m-2 ground]
    Ls: jax.Array          # Upward LW emitted by soil [W m-2]
    Lcanopy_up: jax.Array  # Kernel-weighted canopy LW emitted upward [W m-2 ground]
    gap_LW: jax.Array      # LW gap fraction exp(-kd L_eff) [-]


# ---------------------------------------------------------------------------
# SW decomposition: sw_down → PAR/NIR/UV direct/diffuse components
# ---------------------------------------------------------------------------

@jax.jit
def split_sw_components(
    sw_down: jax.Array,
    cos_zenith: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decompose broadband sw_down into PAR/NIR/UV direct and diffuse.

    Uses fixed spectral fractions (Weiss & Norman 1985):
      PAR ≈ 48 % of sw_down
      NIR ≈ 50 % of sw_down
      UV  ≈  2 % of sw_down

    Direct/diffuse split follows the Liu-Jordan (1960) clearness-index
    approach as adapted for instantaneous radiation (Erbs et al. 1982).
    All arithmetic is differentiable; the night guard uses jnp.where.

    Parameters
    ----------
    sw_down : array, shape (ncol,)
        Downward broadband shortwave radiation [W m-2].
    cos_zenith : array, shape (ncol,)
        Cosine of the solar zenith angle [-].

    Returns
    -------
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV : arrays, each (ncol,)
        Direct and diffuse PAR [W m-2], direct and diffuse NIR [W m-2],
        and total UV [W m-2].
    """
    # Night guard: avoid division by zero and set all components to zero
    is_day = cos_zenith > 0.01

    # Extra-terrestrial irradiance (top-of-atmosphere, W m-2)
    I0 = 1361.0  # solar constant
    I_ext = I0 * jnp.where(is_day, cos_zenith, 1.0)  # avoid zero denominator

    # Clearness index k_t = sw_down / I_ext  (clamped to [0, 1])
    k_t = jnp.clip(sw_down / I_ext, 0.0, 1.0)

    # Diffuse fraction f_d (Erbs et al. 1982, Eq. 1)
    # Piecewise polynomial: valid for k_t in [0, 1]
    f_d_low  = 1.0 - 0.09 * k_t
    f_d_mid  = 0.9511 - 0.1604 * k_t + 4.388 * k_t**2 - 16.638 * k_t**3 + 12.336 * k_t**4
    f_d_high = 0.165
    f_d = jnp.where(k_t <= 0.22, f_d_low,
          jnp.where(k_t <= 0.80, f_d_mid, f_d_high))
    f_d = jnp.clip(f_d, 0.0, 1.0)

    # Broadband direct and diffuse
    sw_diff = f_d * sw_down
    sw_dir  = sw_down - sw_diff

    # Spectral fractions
    PAR_dir  = 0.48 * sw_dir
    PAR_diff = 0.48 * sw_diff
    NIR_dir  = 0.50 * sw_dir
    NIR_diff = 0.50 * sw_diff
    UV       = 0.02 * sw_down

    # Apply night guard
    zero = jnp.zeros_like(sw_down)
    PAR_dir  = jnp.where(is_day, PAR_dir,  zero)
    PAR_diff = jnp.where(is_day, PAR_diff, zero)
    NIR_dir  = jnp.where(is_day, NIR_dir,  zero)
    NIR_diff = jnp.where(is_day, NIR_diff, zero)
    UV       = jnp.where(is_day, UV,       zero)

    return PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV


# ---------------------------------------------------------------------------
# Shortwave radiative transfer
# ---------------------------------------------------------------------------

@jax.jit
def canopy_shortwave_rt(
    PAR_dir: jax.Array,
    PAR_diff: jax.Array,
    NIR_dir: jax.Array,
    NIR_diff: jax.Array,
    UV: jax.Array,
    SZA: jax.Array,
    LAI: jax.Array,
    CI: jax.Array,
    ALB_VIS: jax.Array,
    ALB_NIR: jax.Array,
    Vcmax25_C3_leaf: jax.Array,
    Vcmax25_C4_leaf: jax.Array,
    kn: jax.Array,
    FNonVeg: jax.Array,
) -> CanopySWOutput:
    """Two-leaf shortwave radiative transfer (Sellers 1985, Ryu et al. 2011).

    Separates absorbed PAR and shortwave into sunlit and shaded components,
    and distributes Vcmax25 according to the nitrogen extinction profile.

    Parameters
    ----------
    PAR_dir, PAR_diff : (ncol,) [W m-2]
    NIR_dir, NIR_diff : (ncol,) [W m-2]
    UV                : (ncol,) [W m-2]  total UV
    SZA               : (ncol,) solar zenith angle [degrees]
    LAI               : (ncol,) [m2 m-2]
    CI                : (ncol,) clumping index [-]
    ALB_VIS, ALB_NIR  : (ncol,) visible and NIR albedo [-]
    Vcmax25_C3_leaf,
    Vcmax25_C4_leaf   : (ncol,) per-leaf Vcmax25 [μmol m-2 s-1]
    kn                : (ncol,) nitrogen extinction coefficient [-]
    FNonVeg           : (ncol,) non-vegetated fraction [-]

    Returns
    -------
    CanopySWOutput NamedTuple.
    """
    mskNight = SZA > 89.0

    # ---- Scattering/reflectance coefficients (Sellers 1985) ----
    sigma_P   = 0.175           # PAR leaf scattering coefficient
    rho_PSoil = 0.15 * (1.0 - FNonVeg)   # PAR soil reflectance
    sigma_N   = 0.825           # NIR leaf scattering coefficient
    rho_NSoil = 0.30 * (1.0 - FNonVeg)   # NIR soil reflectance

    # ---- Extinction coefficients (Ryu et al. 2011 Table A1) ----
    cos_sza = jnp.cos(jnp.radians(SZA))
    cos_sza_safe = jnp.where(mskNight, 0.01, cos_sza)   # avoid /0

    kb     = 0.5  / cos_sza_safe    # beam extinction
    kk_Pb  = 0.46 / cos_sza_safe    # beam + scattered PAR
    kb     = jnp.where(mskNight, 50.0, kb)
    kk_Pb  = jnp.where(mskNight, 50.0, kk_Pb)
    kk_Pd  = 0.72                   # diffuse PAR extinction
    kk_Nb  = kb * jnp.sqrt(1.0 - sigma_N)    # beam NIR
    kk_Nd  = 0.35 * jnp.sqrt(1.0 - sigma_N)  # diffuse NIR

    # ---- Sunlit fraction (integrated Beer's law, Eq. 1 in Ryu 2011) ----
    L_CI = LAI * CI
    fSun = jnp.where(
        LAI > 0,
        jnp.clip((1.0 / kb) * (1.0 - jnp.exp(-kb * L_CI)) / LAI, 0.0, 1.0),
        0.0,
    )

    # Pre-compute repeated exponentials
    exp_kk_Pd = jnp.exp(-kk_Pd * L_CI)
    exp_kk_Nd = jnp.exp(-kk_Nd * L_CI)

    # ---- PAR absorption ----
    # Total absorbed downward PAR
    Q_PDn = ((1.0 - ALB_VIS) * PAR_dir  * (1.0 - jnp.exp(-kk_Pb * L_CI))
           + (1.0 - ALB_VIS) * PAR_diff * (1.0 - exp_kk_Pd))
    # Sunlit: beam + diffuse + scattered
    Q_PbSunDn  = PAR_dir * (1.0 - sigma_P) * (1.0 - jnp.exp(-kb * L_CI))
    Q_PdSunDn  = (PAR_diff * (1.0 - ALB_VIS)
                  * (1.0 - jnp.exp(-(kk_Pd + kb) * L_CI))
                  * kk_Pd / (kk_Pd + kb))
    Q_PsSunDn  = jnp.maximum(
        PAR_dir * ((1.0 - ALB_VIS)
                   * (1.0 - jnp.exp(-(kk_Pb + kb) * L_CI))
                   * kk_Pb / (kk_Pb + kb)
                   - (1.0 - sigma_P) * (1.0 - jnp.exp(-2.0 * kb * L_CI)) / 2.0),
        0.0,
    )
    Q_PSunDn   = Q_PbSunDn + Q_PdSunDn + Q_PsSunDn
    Q_PShDn    = jnp.maximum(Q_PDn - Q_PSunDn, 0.0)
    # Soil
    I_PSoil    = ((1.0 - ALB_VIS) * PAR_dir
                + (1.0 - ALB_VIS) * PAR_diff
                - (Q_PSunDn + Q_PShDn))
    APAR_Soil  = (1.0 - rho_PSoil) * I_PSoil
    # Reflected soil PAR absorbed by leaves
    Q_PSunUp   = I_PSoil * rho_PSoil * exp_kk_Pd
    Q_PShUp    = I_PSoil * rho_PSoil * (1.0 - exp_kk_Pd)
    APAR_Sun   = Q_PSunDn + Q_PSunUp
    APAR_Sh    = Q_PShDn  + Q_PShUp

    # ---- NIR absorption ----
    Q_NSunDn = (NIR_dir  * (1.0 - sigma_N) * (1.0 - jnp.exp(-kb * L_CI))
              + NIR_diff * (1.0 - ALB_NIR)
                         * (1.0 - jnp.exp(-(kk_Nd + kb) * L_CI))
                         * kk_Nd / (kk_Nd + kb)
              + NIR_dir  * ((1.0 - ALB_NIR)
                            * (1.0 - jnp.exp(-(kk_Nb + kb) * L_CI))
                            * kk_Nb / (kk_Nb + kb)
                            - (1.0 - sigma_N) * (1.0 - jnp.exp(-2.0 * kb * L_CI)) / 2.0))
    Q_NShDn  = ((1.0 - ALB_NIR) * NIR_dir  * (1.0 - jnp.exp(-kk_Nb * L_CI))
               + (1.0 - ALB_NIR) * NIR_diff * (1.0 - exp_kk_Nd)
               - Q_NSunDn)
    I_NSoil  = ((1.0 - ALB_NIR) * NIR_dir
              + (1.0 - ALB_NIR) * NIR_diff
              - (Q_NSunDn + Q_NShDn))
    ANIR_Soil = (1.0 - rho_NSoil) * I_NSoil
    Q_NSunUp  = I_NSoil * rho_NSoil * exp_kk_Nd
    Q_NShUp   = I_NSoil * rho_NSoil * (1.0 - exp_kk_Nd)
    ANIR_Sun  = Q_NSunDn + Q_NSunUp
    ANIR_Sh   = Q_NShDn  + Q_NShUp

    # ---- UV (treated as narrow PAR-like band) ----
    # Split UV into beam/diffuse proportional to PAR fractions
    total_PAR = PAR_dir + PAR_diff + 1e-5
    UV_dir  = UV * PAR_dir  / total_PAR
    UV_diff = UV - UV_dir
    Q_U    = ((1.0 - 0.05) * UV_diff * (1.0 - jnp.exp(-kk_Pb * L_CI))
            + (1.0 - 0.05) * UV_diff * (1.0 - exp_kk_Pd))
    AUV_Sun  = Q_U * fSun
    AUV_Sh   = Q_U * (1.0 - fSun)
    AUV_Soil = (1.0 - 0.05) * UV - Q_U

    # ---- Total absorbed shortwave ----
    ASW_Sun  = APAR_Sun  + ANIR_Sun  + AUV_Sun
    ASW_Sh   = APAR_Sh   + ANIR_Sh   + AUV_Sh
    ASW_Soil = APAR_Soil + ANIR_Soil + AUV_Soil

    # ---- Convert APAR W m-2 → μmol m-2 s-1 ----
    APAR_Sun = APAR_Sun * _APAR_CONVERSION
    APAR_Sh  = APAR_Sh  * _APAR_CONVERSION

    # ---- Vcmax25 canopy integration via nitrogen extinction profile ----
    # (Sellers 1985 / DifferBESS: exponential N profile with extinction kn)
    kn_kb_Lc = kn + kb * LAI
    LAI_Vc3  = LAI * Vcmax25_C3_leaf
    # Guard against kn*CI → 0
    kn_CI_safe = jnp.maximum(kn * CI, 1e-6)
    Vcmax25_C3Tot = LAI_Vc3 * (1.0 - jnp.exp(-kn * CI)) / kn_CI_safe
    Vcmax25_C3Sun = LAI_Vc3 * (1.0 - jnp.exp(-CI * kn_kb_Lc)) / jnp.maximum(kn_kb_Lc, 1e-6)
    Vcmax25_C3Sh  = Vcmax25_C3Tot - Vcmax25_C3Sun

    LAI_Vc4  = LAI * Vcmax25_C4_leaf
    Vcmax25_C4Tot = LAI_Vc4 * (1.0 - jnp.exp(-kn * CI)) / kn_CI_safe
    Vcmax25_C4Sun = LAI_Vc4 * (1.0 - jnp.exp(-CI * kn_kb_Lc)) / jnp.maximum(kn_kb_Lc, 1e-6)
    Vcmax25_C4Sh  = Vcmax25_C4Tot - Vcmax25_C4Sun

    # ---- Night / bare soil guard ----
    zero = jnp.zeros_like(LAI)
    APAR_Sun      = jnp.where(LAI > 0, APAR_Sun,      zero)
    APAR_Sh       = jnp.where(LAI > 0, APAR_Sh,       zero)
    ASW_Sun       = jnp.where(LAI > 0, ASW_Sun,        zero)
    ASW_Sh        = jnp.where(LAI > 0, ASW_Sh,         zero)
    Vcmax25_C3Sun = jnp.where(LAI > 0, Vcmax25_C3Sun,  zero)
    Vcmax25_C3Sh  = jnp.where(LAI > 0, Vcmax25_C3Sh,   zero)
    Vcmax25_C4Sun = jnp.where(LAI > 0, Vcmax25_C4Sun,  zero)
    Vcmax25_C4Sh  = jnp.where(LAI > 0, Vcmax25_C4Sh,   zero)

    # Night guard: when direct-beam radiation is small, the two-stream
    # quadrature still returns a non-zero fSun (~1/ngauss for the first
    # Gauss point) that would make ``Rb_Sun = rb / (LAI · fSun)``
    # artificially finite and let the canopy closure assign spurious
    # temperatures to a negligible sunlit leaf fraction at night.  Ramp
    # fSun smoothly to 0 via a tanh weight centred at
    # ``sw_dir = 30 W/m²`` with half-width 20 W/m².  The wide transition
    # (sw_dir ∈ [0, 60]) is intentional: with a 30-min model timestep
    # the dusk forcing ``sw_dir`` typically drops from O(50) to 0 in one
    # step, so a narrow transition would still produce a kink at the
    # first night step.  By the time sw_dir > 60 the day_weight is ~1
    # and the sunlit fraction is fully active.
    sw_dir_total = PAR_dir + NIR_dir
    day_weight = 0.5 * (1.0 + jnp.tanh((sw_dir_total - 30.0) / 20.0))
    fSun = fSun * day_weight

    return CanopySWOutput(
        fSun=fSun,
        APAR_Sun=APAR_Sun,
        APAR_Sh=APAR_Sh,
        ASW_Sun=ASW_Sun,
        ASW_Sh=ASW_Sh,
        ASW_Soil=ASW_Soil,
        Vcmax25_C3Sun=Vcmax25_C3Sun,
        Vcmax25_C3Sh=Vcmax25_C3Sh,
        Vcmax25_C4Sun=Vcmax25_C4Sun,
        Vcmax25_C4Sh=Vcmax25_C4Sh,
    )


# ---------------------------------------------------------------------------
# Longwave radiative transfer
# ---------------------------------------------------------------------------

@jax.jit
def canopy_longwave_rt(
    LAI: jax.Array,
    CI: jax.Array,
    SZA: jax.Array,
    Ts: jax.Array,
    Tf_Sun: jax.Array,
    Tf_Sh: jax.Array,
    La: jax.Array,
    epsf: float | jax.Array,
    epss: float | jax.Array,
) -> CanopyLWOutput:
    """Absorbed longwave radiation by sunlit/shaded leaves and soil.

    Two-big-leaf longwave transfer with Beer-law extinction against the
    **effective LAI** ``L_eff = LAI * CI`` (clumping correction, consistent
    with the shortwave routine), following Ryu et al. (2011) / CABLE / CLM.

    Sunlit and shaded leaves absorb sky and soil longwave through separate
    depth-weighted two-stream kernels (``W_*_sky``, ``W_*_soil``) and emit
    with their own temperature.  By Kirchhoff / detailed balance the same
    kernels weight the canopy emission reaching the soil:

        canopy LW reaching soil = W_sun_soil * Lf_Sun + W_sh_soil * Lf_Sh

    which is exact (under the kd kernel) when ``Tf_Sun != Tf_Sh`` and reduces
    to the bulk ``W_tot * Lf`` form in the isothermal limit.  This replaces
    the earlier ``(1 - exp(-kd LAI)) (Ls + La - 2 Lf_Sh)`` shaded form, whose
    ``2 Lf_Sh`` self-emission term spuriously warmed shaded leaves when
    ``Tf_Sun > Tf_Sh`` over hot soil.  The canopy is always fully coupled to
    the soil (no decoupling option).

    Emissivity convention (inherited from DifferBESS / Ryu et al. 2011): each
    surface emits ``eps * sigma * T^4`` but absorbs incident LW with unit
    absorptivity and reflection is neglected (the "near-black" big-leaf
    approximation, valid for the leaf/soil emissivities ~0.96-0.97).  This is
    NOT a strict gray-body: at a hypothetical isothermal equilibrium with
    eps < 1 the net flux is O(1 - eps) rather than exactly zero (the
    ``test_longwave_isothermal_blackbody_zero`` check therefore uses eps = 1).
    A strictly conservative formulation would track reflected/multiply-
    scattered LW; that diverges from the DifferBESS oracle and is left as a
    documented follow-up.  The top-of-canopy ``lw_net`` boundary diagnostic in
    ``two_leaf_canopy`` IS gray-body consistent (applies the effective
    emissivity to both up- and down-welling).

    Parameters
    ----------
    LAI     : (ncol,) leaf area index [m2/m2] (one-sided, nominal)
    CI      : (ncol,) clumping index [-]; effective LAI for radiation = LAI*CI
    SZA     : (ncol,) solar zenith angle [degrees]
    Ts      : (ncol,) soil surface temperature [K]
    Tf_Sun  : (ncol,) sunlit leaf temperature [K]
    Tf_Sh   : (ncol,) shaded leaf temperature [K]
    La      : (ncol,) incoming atmospheric longwave [W m-2]
    epsf    : leaf emissivity (scalar or array)
    epss    : soil emissivity (scalar or array)

    Returns
    -------
    CanopyLWOutput NamedTuple (fluxes per unit ground area [W m-2 ground]).
    """
    SZA_clamped = jnp.clip(SZA, 0.0, 89.0)
    cos_sza     = jnp.cos(jnp.radians(SZA_clamped))

    # Extinction coefficients (Ryu et al. 2011 Table A1)
    kb = 0.5 / jnp.maximum(cos_sza, 0.01)   # direct-beam
    kd = 0.78                                # diffuse

    # Effective LAI for radiation (clumping correction): clumped canopies
    # have larger gap fractions, so LW transmission uses L_eff, not LAI.
    L_eff    = LAI * CI
    kd_L_eff = kd * L_eff

    # Stefan-Boltzmann emitted flux densities (per unit one-sided leaf face)
    Ls     = epss * constants.sigma_sb * Ts**4
    Lf_Sun = epsf * constants.sigma_sb * Tf_Sun**4
    Lf_Sh  = epsf * constants.sigma_sb * Tf_Sh**4

    # Total one-sided diffuse-LW canopy interception weight.
    W_tot = 1.0 - jnp.exp(-kd_L_eff)

    # Sunlit absorption weights (per unit ground area, dimensionless):
    #   sky-origin : kd * int_0^L exp(-(kb+kd)x) dx
    #                  = kd (1 - exp(-(kb+kd) L_eff)) / (kb + kd)
    #   soil-origin: kd * int_0^L exp(-kb x) exp(-kd (L-x)) dx
    #                  = kd (exp(-kb L_eff) - exp(-kd L_eff)) / (kd - kb)
    #                  (kb -> kd limit is L_eff * exp(-kd L_eff))
    W_sun_sky = kd * (1.0 - jnp.exp(-(kb + kd) * L_eff)) / (kb + kd)

    # The soil-origin integral has a removable singularity at kb = kd
    # (SZA ~ 50°).  Write it via the relative exponential
    #   ratio_soil = (exp(-kb L) - exp(-kd L)) / (kd - kb)
    #              = exp(-kd L) * L * exprel((kd-kb) L),
    #   exprel(y) = (exp(y) - 1) / y,   exprel(0) = 1,
    # which is finite and SMOOTH through kb = kd.  A near-zero Taylor branch
    # (exprel ≈ 1 + y/2) keeps both the value and its kb/SZA derivative correct
    # there (the earlier constant-limit guard zeroed the kb derivative inside
    # the window), with a guarded denominator so AD never sees 0/0.
    y       = (kd - kb) * L_eff
    y_safe  = jnp.where(jnp.abs(y) < 1.0e-6, 1.0, y)
    exprel  = jnp.where(jnp.abs(y) < 1.0e-6, 1.0 + 0.5 * y, jnp.expm1(y) / y_safe)
    W_sun_soil = kd * jnp.exp(-kd_L_eff) * L_eff * exprel

    # Shaded weights are the residual interception weights.
    W_sh_sky  = W_tot - W_sun_sky
    W_sh_soil = W_tot - W_sun_soil

    # Guard tiny negative roundoff outside [0, W_tot].
    W_sun_sky  = jnp.clip(W_sun_sky,  0.0, W_tot)
    W_sun_soil = jnp.clip(W_sun_soil, 0.0, W_tot)
    W_sh_sky   = jnp.clip(W_sh_sky,   0.0, W_tot)
    W_sh_soil  = jnp.clip(W_sh_soil,  0.0, W_tot)

    # Net absorbed LW per leaf class: each class absorbs from the soil below
    # and the sky above and emits with its own temperature.
    ALW_Sun = W_sun_soil * (Ls - Lf_Sun) + W_sun_sky * (La - Lf_Sun)
    ALW_Sh  = W_sh_soil  * (Ls - Lf_Sh)  + W_sh_sky  * (La - Lf_Sh)

    # Soil absorbed LW: depth-weighted per-class canopy emission (detailed
    # balance) + gap-transmitted atmospheric LW - soil's own emission.
    gap_LW = jnp.exp(-kd_L_eff)
    ALW_Soil = (
        W_sun_soil * Lf_Sun
        + W_sh_soil * Lf_Sh
        + gap_LW * La
        - Ls
    )

    # Kernel-weighted canopy LW emitted upward to the atmosphere (for the
    # top-of-canopy LST / lw_up diagnostic — uses the same sky kernels as the
    # sky->leaf absorption, by detailed balance).
    Lcanopy_up = W_sun_sky * Lf_Sun + W_sh_sky * Lf_Sh

    return CanopyLWOutput(
        ALW_Sun=ALW_Sun,
        ALW_Sh=ALW_Sh,
        ALW_Soil=ALW_Soil,
        Ls=Ls,
        Lcanopy_up=Lcanopy_up,
        gap_LW=gap_LW,
    )
