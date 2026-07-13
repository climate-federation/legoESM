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

# --- Weiss & Norman (1985) broadband → spectral fractions [-] ---
_PAR_FRACTION = 0.48
_NIR_FRACTION = 0.50
_UV_FRACTION  = 0.02

# --- Erbs et al. (1982) diffuse-fraction correlation in clearness index k_t ---
_ERBS_KT_LOW    = 0.22     # k_t below which the low-clearness branch applies
_ERBS_KT_HIGH   = 0.80     # k_t above which f_d saturates to _ERBS_FD_HIGH
_ERBS_LOW_SLOPE = 0.09     # low branch: f_d = 1 - 0.09 k_t
_ERBS_MID_C0 = 0.9511      # mid branch: quartic polynomial in k_t
_ERBS_MID_C1 = 0.1604
_ERBS_MID_C2 = 4.388
_ERBS_MID_C3 = 16.638
_ERBS_MID_C4 = 12.336
_ERBS_FD_HIGH = 0.165      # diffuse fraction for k_t > _ERBS_KT_HIGH

# --- Sellers (1985) / Ryu et al. (2011) two-stream scattering & extinction [-] ---
_SIGMA_PAR    = 0.175      # PAR leaf scattering coefficient
_SIGMA_NIR    = 0.825      # NIR leaf scattering coefficient
_RHO_PAR_SOIL = 0.15       # PAR soil reflectance factor (× (1 - FNonVeg))
_RHO_NIR_SOIL = 0.30       # NIR soil reflectance factor (× (1 - FNonVeg))
_KPB_PAR      = 0.46       # beam + scattered PAR extinction numerator (/cos SZA)
_KD_PAR       = 0.72       # diffuse PAR extinction
_KD_NIR_COEF  = 0.35       # diffuse NIR extinction coefficient
_RHO_UV       = 0.05       # UV reflectance (leaf + soil, PAR-like band)
_KD_LW        = 0.78       # diffuse longwave extinction
_KB_BEAM      = 0.5        # direct-beam extinction numerator = G-function for a
                           # spherical (uniform) leaf-angle distribution (Ryu 2011)

# --- Night ramp for the sunlit fraction (numerics; see canopy_shortwave_rt) ---
_NIGHT_RAMP_CENTER_WM2    = 30.0   # tanh centre in direct-beam SW [W m-2]
_NIGHT_RAMP_HALFWIDTH_WM2 = 20.0   # tanh half-width [W m-2]


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
    Lcanopy_up: jax.Array  # Kernel-weighted canopy LW emitted upward (= S_up) [W m-2 ground]
    gap_LW: jax.Array      # LW gap fraction exp(-kd L_eff) [-]
    LW_out: jax.Array      # Total top-of-canopy upward LW escaping to atmosphere
                           # (S_up + t_c*U_g + r_c*La) [W m-2 ground]; conservative
                           # land->atmosphere flux
    eps_col: jax.Array     # Atmosphere-equivalent column LW emissivity 1 - R_col,
                           # R_col = dLW_out/dLa = r_c + t_c^2*rho/(1-rho*r_c) [-]
    LW_emit: jax.Array     # Emission-only upward LW = LW_out - R_col*La [W m-2 ground];
                           # eps_col*sigma*T^4 + (1-eps_col)*La = LW_out exactly


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
    is_day = cos_zenith > 0.01  # coeff-ok: night guard, cos(zenith) floor to avoid /0

    # Extra-terrestrial irradiance (top-of-atmosphere, W m-2)
    I0 = constants.S_0  # total solar irradiance
    I_ext = I0 * jnp.where(is_day, cos_zenith, 1.0)  # avoid zero denominator

    # Clearness index k_t = sw_down / I_ext  (clamped to [0, 1])
    k_t = jnp.clip(sw_down / I_ext, 0.0, 1.0)

    # Diffuse fraction f_d (Erbs et al. 1982, Eq. 1)
    # Piecewise polynomial: valid for k_t in [0, 1]
    f_d_low  = 1.0 - _ERBS_LOW_SLOPE * k_t
    f_d_mid  = (_ERBS_MID_C0 - _ERBS_MID_C1 * k_t + _ERBS_MID_C2 * k_t**2
                - _ERBS_MID_C3 * k_t**3 + _ERBS_MID_C4 * k_t**4)
    f_d_high = _ERBS_FD_HIGH
    f_d = jnp.where(k_t <= _ERBS_KT_LOW, f_d_low,
          jnp.where(k_t <= _ERBS_KT_HIGH, f_d_mid, f_d_high))
    f_d = jnp.clip(f_d, 0.0, 1.0)

    # Broadband direct and diffuse
    sw_diff = f_d * sw_down
    sw_dir  = sw_down - sw_diff

    # Spectral fractions
    PAR_dir  = _PAR_FRACTION * sw_dir
    PAR_diff = _PAR_FRACTION * sw_diff
    NIR_dir  = _NIR_FRACTION * sw_dir
    NIR_diff = _NIR_FRACTION * sw_diff
    UV       = _UV_FRACTION * sw_down

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
    mskNight = SZA > 89.0   # coeff-ok: night mask, solar-zenith threshold [deg]

    # ---- Scattering/reflectance coefficients (Sellers 1985) ----
    sigma_P   = _SIGMA_PAR                  # PAR leaf scattering coefficient
    rho_PSoil = _RHO_PAR_SOIL * (1.0 - FNonVeg)   # PAR soil reflectance
    sigma_N   = _SIGMA_NIR                  # NIR leaf scattering coefficient
    rho_NSoil = _RHO_NIR_SOIL * (1.0 - FNonVeg)   # NIR soil reflectance

    # ---- Extinction coefficients (Ryu et al. 2011 Table A1) ----
    cos_sza = jnp.cos(jnp.radians(SZA))
    cos_sza_safe = jnp.where(mskNight, 0.01, cos_sza)   # coeff-ok: /0 guard on cos(SZA) at night

    kb     = _KB_BEAM / cos_sza_safe    # beam extinction
    kk_Pb  = _KPB_PAR / cos_sza_safe    # beam + scattered PAR
    kb     = jnp.where(mskNight, 50.0, kb)   # coeff-ok: night sentinel extinction (kills beam)
    kk_Pb  = jnp.where(mskNight, 50.0, kk_Pb)   # coeff-ok: night sentinel extinction (kills beam)
    kk_Pd  = _KD_PAR                # diffuse PAR extinction
    kk_Nb  = kb * jnp.sqrt(1.0 - sigma_N)          # beam NIR
    kk_Nd  = _KD_NIR_COEF * jnp.sqrt(1.0 - sigma_N)  # diffuse NIR

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
    total_PAR = PAR_dir + PAR_diff + 1e-5   # coeff-ok: /0 guard on UV beam/diffuse split
    UV_dir  = UV * PAR_dir  / total_PAR
    UV_diff = UV - UV_dir
    # Beam component uses beam extinction kk_Pb, diffuse uses diffuse kk_Pd —
    # mirrors the PAR Q_PDn split above.  (Previously both terms used UV_diff,
    # applying beam extinction to the diffuse flux and dropping UV_dir, which
    # mis-partitioned the beam/diffuse and leaf/soil UV split.)
    Q_U    = ((1.0 - _RHO_UV) * UV_dir  * (1.0 - jnp.exp(-kk_Pb * L_CI))
            + (1.0 - _RHO_UV) * UV_diff * (1.0 - exp_kk_Pd))
    AUV_Sun  = Q_U * fSun
    AUV_Sh   = Q_U * (1.0 - fSun)
    AUV_Soil = (1.0 - _RHO_UV) * UV - Q_U

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
    day_weight = 0.5 * (1.0 + jnp.tanh(
        (sw_dir_total - _NIGHT_RAMP_CENTER_WM2) / _NIGHT_RAMP_HALFWIDTH_WM2))
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

    Conservative gray-body emissivity (β = 1/2 isotropic leaf scatter): each
    surface ABSORBS ``eps * incident`` (Kirchhoff: absorptivity = emissivity)
    and REFLECTS ``(1 - eps)``.  The diffuse canopy stream splits into absorbed
    ``a_c = eps_f * W_tot``, back-scattered ``r_c = (1-eps_f) W_tot / 2`` and
    transmitted ``t_c`` (``a_c + r_c + t_c = 1``), and the canopy<->ground
    interreflection is summed in closed form ``1/(1 - rho * r_c)`` (one division;
    ``rho * r_c <= 6e-4`` for physical eps).  This is EXACTLY energy-conserving
    (``ALW_Sun + ALW_Sh + ALW_Soil = La - LW_out``) and gives zero net flux at
    isothermal equilibrium for ANY eps_f, eps_s; it reduces exactly to the
    previous near-black scheme at eps_f = eps_s = 1 (then r_c = 0, t_c = gap_LW).
    ``LW_out`` is the true top-of-canopy upward LW (emission + reflection) for the
    coupler and the LST diagnostic.  Ported from DifferBESS; full derivation in
    docs/canopy_longwave_graybody_conservation.md §5.

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
    SZA_clamped = jnp.clip(SZA, 0.0, 89.0)   # coeff-ok: clamp SZA < 90 deg to keep cos(SZA) > 0
    cos_sza     = jnp.cos(jnp.radians(SZA_clamped))

    # Extinction coefficients (Ryu et al. 2011 Table A1)
    kb = _KB_BEAM / jnp.maximum(cos_sza, 0.01)   # coeff-ok: /0 guard on cos(SZA); direct-beam
    kd = _KD_LW                              # diffuse

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

    gap_LW = jnp.exp(-kd_L_eff)

    # ---- Conservative gray-body two-leaf longwave (β = 1/2 isotropic) ----
    # Keeps the physical leaf/soil emissivities εf, εs < 1.  Each surface absorbs
    # ε·incident (Kirchhoff) and reflects (1−ε); the diffuse canopy stream splits
    # into absorbed ``a_c``, back-scattered ``r_c`` (canopy LW reflectance) and
    # transmitted ``t_c``, with ``a_c + r_c + t_c = 1``.  The canopy↔ground
    # interreflection is summed in closed form ``1/(1 − ρ·r_c)`` (one division,
    # ρ·r_c ≤ 6e-4 for physical ε, so the denominator stays ≈1).  This is exactly
    # conservative (``ΣALW = La − LW_out``), zero-net at isothermal equilibrium for
    # any εf, εs, and reduces EXACTLY to the previous near-black scheme at
    # εf=εs=1 (then r_c=0, t_c=gap_LW).  Ported from DifferBESS
    # CanopyLongwaveRadiation; see docs/canopy_longwave_graybody_conservation.md §5.
    a_c = epsf * W_tot                          # absorbed (absorptivity = εf)
    r_c = 0.5 * (1.0 - epsf) * W_tot            # back-scattered (β = 1/2)
    t_c = gap_LW + 0.5 * (1.0 - epsf) * W_tot   # transmitted (gap + forward-scatter)
    rho = 1.0 - epss                            # ground LW reflectance

    # Per-class canopy emission to the sky / ground hemispheres (``Lf`` already
    # carries εf, so ``S_up`` ≡ εf·(W_sun_sky·B_Sun + W_sh_sky·B_Sh)).
    S_up   = W_sun_sky  * Lf_Sun + W_sh_sky  * Lf_Sh
    S_down = W_sun_soil * Lf_Sun + W_sh_soil * Lf_Sh

    # Closed-form canopy↔ground interreflection (note εs·B_g ≡ Ls):
    S_d = t_c * La + S_down                       # primary downward source
    U_g = (Ls + rho * S_d) / (1.0 - rho * r_c)    # upward LW leaving the ground
    D_g = S_d + r_c * U_g                          # downward LW onto the ground

    # Net absorbed LW per leaf class: absorb εf of the sky stream (above) and of
    # the ground upwelling U_g (below); emit εf·B to both hemispheres.
    ALW_Sun = epsf * (W_sun_sky * La + W_sun_soil * U_g) - (W_sun_sky + W_sun_soil) * Lf_Sun
    ALW_Sh  = epsf * (W_sh_sky  * La + W_sh_soil  * U_g) - (W_sh_sky  + W_sh_soil ) * Lf_Sh
    ALW_Soil = epss * D_g - Ls                    # ≡ εs·(D_g − B_g)

    # Top-of-canopy upward LW escaping to the atmosphere: canopy up-emission +
    # ground upwelling transmitted out + sky reflected off the canopy top.
    LW_out = S_up + t_c * U_g + r_c * La

    # Atmosphere-equivalent column representation so the coupler's property-
    # coupling LW boundary ``eps*sigma*T^4 + (1-eps)*La`` reproduces this LW_out
    # EXACTLY.  The column LW reflectance R_col = dLW_out/dLa is the fraction of
    # incident sky LW the canopy+ground column reflects back up — canopy back-
    # scatter r_c PLUS sky LW transmitted to the ground, reflected, and
    # re-transmitted out through the multiple-reflection chain
    # t_c^2*rho/(1-rho*r_c) (differentiate U_g w.r.t. La: dU_g/dLa = rho*t_c/
    # (1-rho*r_c)).  The remaining LW_out - R_col*La is the La-independent
    # emission of the column.  Then eps_col*sigma*T_emit^4 + (1-eps_col)*La
    # = LW_emit + R_col*La = LW_out for any eps_f, eps_s, LAI.
    R_col   = r_c + t_c * t_c * rho / (1.0 - rho * r_c)
    eps_col = 1.0 - R_col
    LW_emit = LW_out - R_col * La

    # Back-compat: kernel-weighted canopy up-emission (≡ S_up).
    Lcanopy_up = S_up

    return CanopyLWOutput(
        ALW_Sun=ALW_Sun,
        ALW_Sh=ALW_Sh,
        ALW_Soil=ALW_Soil,
        Ls=Ls,
        Lcanopy_up=Lcanopy_up,
        gap_LW=gap_LW,
        LW_out=LW_out,
        eps_col=eps_col,
        LW_emit=LW_emit,
    )
