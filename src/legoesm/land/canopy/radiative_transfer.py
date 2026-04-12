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

import jax
import jax.numpy as jnp
from typing import NamedTuple

# Physical constants (local; do not import from legoesm.constants to keep
# this module self-contained for potential standalone use)
_STEFAN_BOLTZMANN = 5.670373e-8   # [W m-2 K-4]
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
    ALW_Sun: jax.Array   # Net absorbed LW by sunlit leaves [W m-2]
    ALW_Sh: jax.Array    # Net absorbed LW by shaded leaves [W m-2]
    ALW_Soil: jax.Array  # Net absorbed LW by soil [W m-2]
    Ls: jax.Array        # Upward LW emitted by soil [W m-2]


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
    SZA: jax.Array,
    Ts: jax.Array,
    Tf_mean: jax.Array,
    Tf_Sun: jax.Array,
    Tf_Sh: jax.Array,
    La: jax.Array,
    epsf: float | jax.Array,
    epss: float | jax.Array,
) -> CanopyLWOutput:
    """Absorbed longwave radiation by sunlit/shaded leaves and soil.

    Uses the Beer-law extinction approach from Ryu et al. (2011).

    Parameters
    ----------
    LAI     : (ncol,) leaf area index [m2/m2]
    SZA     : (ncol,) solar zenith angle [degrees]
    Ts      : (ncol,) soil surface temperature [K]
    Tf_mean : (ncol,) mean foliage temperature [K] (area-weighted average)
    Tf_Sun  : (ncol,) sunlit leaf temperature [K]
    Tf_Sh   : (ncol,) shaded leaf temperature [K]
    La      : (ncol,) incoming atmospheric longwave [W m-2]
    epsf    : leaf emissivity (scalar or array)
    epss    : soil emissivity (scalar or array)

    Returns
    -------
    CanopyLWOutput NamedTuple.
    """
    SZA_clamped = jnp.clip(SZA, 0.0, 89.0)
    cos_sza     = jnp.cos(jnp.radians(SZA_clamped))

    # Extinction coefficients (Ryu et al. 2011 Table A1)
    kb = 0.5 / jnp.maximum(cos_sza, 0.01)
    kd = 0.78

    # Stefan-Boltzmann emitted fluxes
    Ls    = epss * _STEFAN_BOLTZMANN * Ts**4
    Lf_Sun = epsf * _STEFAN_BOLTZMANN * Tf_Sun**4
    Lf_Sh  = epsf * _STEFAN_BOLTZMANN * Tf_Sh**4
    Lf     = epsf * _STEFAN_BOLTZMANN * Tf_mean**4

    kd_LAI = kd * LAI

    # ``kd - kb`` is negative for SZA > ~50° (kb > 0.78) and positive below.
    # The earlier ``max(kd - kb, 1e-6)`` was a safety against division by
    # zero when ``kd == kb`` (SZA ≈ 50°), but it silently clipped any
    # negative denominator to +1e-6 — which flips the sign and amplifies
    # the sunlit LW term by ~1e6 at night (SZA → 90°, kb → 50).  This
    # was the root cause of nocturnal Newton divergence for dense canopies.
    # Use a sign-preserving guard that only intervenes at |kd - kb| < 1e-6.
    kdb = kd - kb
    kdb_safe = jnp.where(jnp.abs(kdb) < 1e-6, 1e-6, kdb)

    # Net absorbed LW by sunlit leaves
    ALW_Sun = (
        (Ls - Lf_Sun) * kd * (jnp.exp(-kd_LAI) - jnp.exp(-kb * LAI)) / kdb_safe
        + kd * (La - Lf_Sun) * (1.0 - jnp.exp(-(kb + kd) * LAI)) / (kd + kb)
    )

    # Net absorbed LW by shaded leaves
    ALW_Sh = (1.0 - jnp.exp(-kd_LAI)) * (Ls + La - 2.0 * Lf_Sh) - ALW_Sun

    # Net absorbed LW by soil
    ALW_Soil = (1.0 - jnp.exp(-kd_LAI)) * Lf + jnp.exp(-kd_LAI) * La - Ls

    return CanopyLWOutput(
        ALW_Sun=ALW_Sun,
        ALW_Sh=ALW_Sh,
        ALW_Soil=ALW_Soil,
        Ls=Ls,
    )
