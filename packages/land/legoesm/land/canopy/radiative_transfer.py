"""Canopy radiative transfer for the two-leaf energy balance model.

Shortwave: Sellers (1985) / Ryu et al. (2011) two-stream approach.
  - Separates sunlit and shaded leaf fractions.
  - Handles PAR, NIR, and UV independently.
  - Distributes Vcmax25 using a nitrogen extinction profile.

Longwave: Ryu et al. (2011) extinction-based scheme.

SW decomposition: Liu-Jordan clearness-index direct/diffuse split.

All functions are pure JAX, JIT-compatible, and differentiable.

Faithfulness
------------
The shortwave path is pinned term-for-term by
``tests/land/unit/test_canopy_rt_faithful.py`` against an independent scalar
oracle (rel 1e-9) that reimplements the published closed forms:

  * Direct/diffuse split (``split_sw_components``): the Erbs et al. (1982)
    piecewise diffuse-fraction correlation f_d(k_t) — linear ``1 - 0.09 k_t``
    for k_t <= 0.22, the quartic mid branch for 0.22 < k_t <= 0.80, constant
    0.165 above — on the clearness index k_t = sw_down / (S_0 cos Z).  The
    downstream PAR/NIR/UV = 0.48/0.50/0.02 spectral fractions are FIXED MODEL
    assumptions (broadly after Weiss & Norman 1985 for the visible/NIR split;
    the 2% UV band is a model choice, NOT a Weiss-Norman published form).
  * Two-stream absorption (``canopy_shortwave_rt``, Ryu et al. 2011): the beam
    kb = G / cos Z (spherical leaf-angle G = 0.5), the scattered-beam
    kk_Pb = 0.46 / cos Z and diffuse kk_Pd = 0.72 PAR extinctions (Ryu 2011
    Table A1, attributed there to de Pury & Farquhar 1997), the integrated-Beer
    sunlit fraction fSun = (1 - e^{-kb L_CI}) / (kb LAI) with the foliar-clumping
    L_CI = LAI CI, the beam+diffuse+scattered sunlit/shaded/soil partition with
    the k/(k + kb) two-stream weights, and the exponential-nitrogen-profile
    canopy Vcmax25 integral (sunlit = LAI Vc (1 - e^{-CI(kn + kb LAI)})/(kn +
    kb LAI)) — all Ryu (2011) forms, with Sellers (1985) the two-stream lineage.
    MODEL-SPECIFIC choices layered on top: the fixed 2% UV band, the
    canopy-cover scaling of the ground-reflected soil term, and the tanh dusk ramp.

The Erbs / spectral-fraction / Sellers-Ryu scheme coefficients are transcribed
as independent oracle literals and canaried against the module constants
(module == literal == value), so no scheme coefficient is sourced from the SUT;
S_0 is the shared ``legoesm.constants`` value.  DEPARTURES / numeric guards
(documented, not the pure closed form; reproduced by the oracle so the pin
stays exact): the cos Z > 0.01 and SZA > 89 deg night guards with the
kb = kk_Pb = 50 night sentinel, the fSun clip to [0, 1] and LAI > 0 zeroing,
the max(., 0) floors on the scattered-sunlit and shaded down-fluxes, the
max(kn CI, 1e-6) / max(kn + kb LAI, 1e-6) profile-integral guards, and the
tanh day-weight ramp (centre 30, half-width 20 W m-2 in beam SW) that fades
fSun TOWARD zero at dusk (weight ~0.047 at zero beam SW — never exactly zero).
The SZA > 89 deg sentinel only swaps in the sentinel extinction; it does NOT
zero fSun (a nonzero-beam night still leaves fSun ~ 0.007).  fSun is exactly
zero on the LAI <= 0 (bare-soil) where()-branch, or whenever L_CI = LAI CI = 0
(e.g. CI = 0).  The longwave path
is covered separately by test_canopy_rt.py (conservation, isothermal-zero,
clumping, kb == kd smoothness / AD).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
_RHO_PAR_SOIL = 0.15       # PAR soil reflectance factor (x canopy cover)
_RHO_NIR_SOIL = 0.30       # NIR soil reflectance factor (x canopy cover)
_KPB_PAR      = 0.46       # beam + scattered PAR extinction numerator (/cos SZA)
_KD_PAR       = 0.72       # diffuse PAR extinction
_KD_NIR_COEF  = 0.35       # diffuse NIR extinction coefficient
_RHO_UV       = 0.05       # UV reflectance (leaf + soil, PAR-like band)
_KD_LW        = 0.78       # diffuse longwave extinction
_KB_BEAM      = 0.5        # direct-beam extinction numerator = G-function for a
                           # spherical (uniform) leaf-angle distribution (Ryu 2011)


def broadband_albedo(ALB_VIS: jax.Array, ALB_NIR: jax.Array) -> jax.Array:
    """Broadband reflectance the two-leaf RT realises for these band albedos:
    the band albedo weights PAR and NIR, the UV band reflects ``_RHO_UV``
    regardless.  ``1 - sum(absorbed)/sw_down`` of :func:`canopy_shortwave_rt`
    equals this exactly (the soil-reflection terms cancel in the sum)."""
    return (_PAR_FRACTION * ALB_VIS + _NIR_FRACTION * ALB_NIR
            + _UV_FRACTION * _RHO_UV)


def _floor_soil_absorption(total: jax.Array, q_sun: jax.Array, q_sh: jax.Array):
    """Return ``(I_soil, q_sun, q_sh)`` with the ground's share of ``total``
    floored at zero and the leaf shares scaled to keep the sum exact.

    The two-leaf closure derives the sunlit-leaf beam term from the LEAF
    scattering coefficient while the column total is set by the band albedo
    supplied to it; under a bright band (snow at 0.7-0.8 beneath leaves) the
    leaf terms can exceed the whole column's absorption and the remainder
    handed to the ground goes negative.  A snow surface under a canopy absorbs
    little and the leaves absorb most, so the excess belongs to the leaves:
    scale both leaf classes down to the column total and give the ground
    nothing.  Untouched wherever the ground share is already positive."""
    q_can = q_sun + q_sh
    i_soil = jnp.maximum(total - q_can, 0.0)
    scale = jnp.where(q_can > 0.0, (total - i_soil) / jnp.maximum(q_can, 1e-30), 1.0)
    return i_soil, q_sun * scale, q_sh * scale

def canopy_cover(LAI: jax.Array, CI: jax.Array) -> jax.Array:
    """Fraction of the ground shaded by foliage, ``1 - exp(-G CI LAI)``.

    The complement of the canopy GAP fraction, with the same spherical-leaf
    ``G = 0.5`` and clumping the beam extinction uses.  It vanishes like
    ``0.5 CI LAI`` as the leaf area does, which is the property everything
    downstream needs: a term describing radiation intercepted BY LEAVES must
    go to zero with the leaves.

    This is computed here rather than taken as an input because it used to be
    one: a per-column ``FNonVeg`` field that two builders filled in two
    incompatible ways — the flux-tower builder with this gap fraction, the
    global builder with a BINARY dominant-plant-type flag. At a leaf area of
    0.019 those disagree by 0.99, and the binary value left the ground-reflected
    radiation heating a canopy that had no leaves to absorb it: measured, the
    sunlit leaf temperature ran 124 K from freezing and the closure failed to
    solve in 405 of 4500 sampled weather states. Deriving it from ``LAI`` and
    ``CI``, which the caller already passes, makes the inconsistency
    unrepresentable.

    The same expression appears as the bare-vs-dense weight in
    ``stability.compute_below_canopy_resistance``; that one keeps its own form
    because it may later want stem area or snow burial, which this one must not.
    """
    return -jnp.expm1(-_KB_BEAM * CI * LAI)

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

    Returns
    -------
    CanopySWOutput NamedTuple.
    """
    mskNight = SZA > 89.0   # coeff-ok: night mask, solar-zenith threshold [deg]

    # ---- Scattering/reflectance coefficients (Sellers 1985) ----
    sigma_P   = _SIGMA_PAR                  # PAR leaf scattering coefficient
    # Ground-reflected radiation returned UPWARD INTO THE CANOPY, so it is
    # scaled by the foliage that can intercept it — NOT the bare-ground albedo,
    # which is applied to the incident flux further down.
    cover    = canopy_cover(LAI, CI)
    rho_PSoil = _RHO_PAR_SOIL * cover       # PAR soil reflectance into the canopy
    sigma_N   = _SIGMA_NIR                  # NIR leaf scattering coefficient
    rho_NSoil = _RHO_NIR_SOIL * cover       # NIR soil reflectance into the canopy

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
    # Soil: the column total minus the leaves, floored at zero
    I_PSoil, Q_PSunDn, Q_PShDn = _floor_soil_absorption(
        (1.0 - ALB_VIS) * (PAR_dir + PAR_diff), Q_PSunDn, Q_PShDn)
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
    I_NSoil, Q_NSunDn, Q_NShDn = _floor_soil_absorption(
        (1.0 - ALB_NIR) * (NIR_dir + NIR_diff), Q_NSunDn, Q_NShDn)
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
# CLM5 big-leaf two-stream column albedo (canopy over a snowy ground)
# ---------------------------------------------------------------------------
# CTSM 5.1 SurfaceAlbedoMod.F90 TwoStream (lines 1314-1338 preamble, 1378-1510
# albedo) and clm_varcon.F90 205-208 (intercepted-snow optics).  Only the
# column ALBEDO outputs (albd, albi) are ported; the absorbed/transmitted
# partitions are not used here (the two-leaf RT keeps doing that).
_CLM_CHIL_MIN = -0.4         # xl clip range (SurfaceAlbedoMod 1323)
_CLM_CHIL_MAX = 0.6
_CLM_CHIL_ZERO = 0.01        # |chil| <= 0.01 -> 0.01 (1324)
_CLM_PHI1_A = 0.633          # phi1 = 0.5 - 0.633 chil - 0.330 chil^2 (1325)
_CLM_PHI1_B = 0.330
_CLM_PHI2_A = 0.877          # phi2 = 0.877 (1 - 2 phi1) (1326)
_CLM_COSZ_MIN = 0.001        # cosz floor (1322)
_CLM_TEMP0_MIN = 1.0e-6      # temp0 floor, bugzilla 2431 (1334)
_CLM_EXP_MAX = 40.0          # optical-depth cap in exp(-t) (1428-1430)
_CLM_MPE = 1.0e-6            # rho/tau and weight floors (823-831)
_CLM_OMEGA_SNOW = (0.8, 0.4)  # omegas(vis, nir), clm_varcon 207-208
_CLM_BETA_SNOW = 0.5         # betads = betais, clm_varcon 205-206


def clm5_two_stream_albedo(vai, f_leaf, rhol, taul, rhos, taus, xl, cosz,
                           alb_ground, fcansno=0.0, omega_snow=0.0):
    """CLM5 big-leaf two-stream albedo of a canopy over a ground of albedo
    ``alb_ground`` for ONE waveband.  Returns ``(albd, albi)``: direct-beam and
    diffuse column albedo.

    ``vai`` = exposed elai + esai, ``f_leaf`` = elai / vai (CLM ``wl``), the
    leaf/stem reflectance and transmittance of that band, ``xl`` the leaf
    orientation index, ``fcansno`` the snow-covered canopy fraction with
    ``omega_snow`` its scattering (``_CLM_OMEGA_SNOW``).  Where ``vai <= 0`` the
    canopy is absent and the column albedo IS the ground albedo (CLM's
    non-vegetated filter).  Same ground albedo for beam and diffuse (the model
    carries one per band)."""
    veg = vai > 0.0
    vai_s = jnp.where(veg, vai, 1.0)          # keep the masked branch finite
    ws = 1.0 - f_leaf
    rho = jnp.maximum(rhol * f_leaf + rhos * ws, _CLM_MPE)
    tau = jnp.maximum(taul * f_leaf + taus * ws, _CLM_MPE)
    cz = jnp.maximum(cosz, _CLM_COSZ_MIN)
    chil = jnp.clip(xl, _CLM_CHIL_MIN, _CLM_CHIL_MAX)
    chil = jnp.where(jnp.abs(chil) <= _CLM_CHIL_ZERO, _CLM_CHIL_ZERO, chil)
    phi1 = 0.5 - _CLM_PHI1_A * chil - _CLM_PHI1_B * chil * chil
    phi2 = _CLM_PHI2_A * (1.0 - 2.0 * phi1)
    gdir = phi1 + phi2 * cz
    ext = gdir / cz
    avmu = (1.0 - phi1 / phi2 * jnp.log((phi1 + phi2) / phi1)) / phi2
    temp0 = jnp.maximum(gdir + phi2 * cz, _CLM_TEMP0_MIN)
    temp1 = phi1 * cz
    temp2 = 1.0 - temp1 / temp0 * jnp.log((temp1 + temp0) / temp1)
    # Leaf/stem single-scattering parameters (1381-1385).
    omegal = rho + tau
    asu = 0.5 * omegal * gdir / temp0 * temp2
    betadl = (1.0 + avmu * ext) / (omegal * avmu * ext) * asu
    betail = 0.5 * ((rho + tau) + (rho - tau) * ((1.0 + chil) / 2.0) ** 2) / omegal
    # Intercepted-snow adjustment (1396-1398).
    om = (1.0 - fcansno) * omegal + fcansno * omega_snow
    betad = ((1.0 - fcansno) * omegal * betadl
             + fcansno * omega_snow * _CLM_BETA_SNOW) / om
    betai = ((1.0 - fcansno) * omegal * betail
             + fcansno * omega_snow * _CLM_BETA_SNOW) / om
    # Common terms (1408-1420).
    b = 1.0 - om + om * betai
    c1 = om * betai
    t0 = avmu * ext
    d = t0 * om * betad
    f = t0 * om * (1.0 - betad)
    h = jnp.sqrt(b * b - c1 * c1) / avmu
    sigma = t0 * t0 - (b * b - c1 * c1)
    p1 = b + avmu * h
    p2 = b - avmu * h
    p3 = b + t0
    p4 = b - t0
    s1 = jnp.exp(-jnp.minimum(h * vai_s, _CLM_EXP_MAX))
    s2 = jnp.exp(-jnp.minimum(ext * vai_s, _CLM_EXP_MAX))
    # Direct beam (1434-1462).
    # Floored divisor: alb_ground = 0 would make u1 infinite and, through the
    # jnp.where below, NaN the gradient even where the canopy is absent.
    u1 = b - c1 / jnp.maximum(alb_ground, _CLM_MPE)
    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    h1 = -d * p4 - c1 * f
    tmp6 = d - h1 * p3 / sigma
    tmp7 = (d - c1 - h1 / sigma * (u1 + t0)) * s2
    h2 = (tmp6 * tmp2 / s1 - p2 * tmp7) / d1
    h3 = -(tmp6 * tmp3 * s1 - p1 * tmp7) / d1
    albd = h1 / sigma + h2 + h3
    # Diffuse (1480-1510): same u1 since the ground albedo is shared.
    h7 = (c1 * tmp2) / (d1 * s1)
    h8 = (-c1 * tmp3 * s1) / d1
    albi = h7 + h8
    return (jnp.where(veg, albd, alb_ground), jnp.where(veg, albi, alb_ground))


# --- CLM5 snow burial of short vegetation (SatellitePhenologyMod.F90 173-188) ---
_CLM_TALL_PFT_MAX = 11       # nbrdlf_dcd_brl_shrub: PFTs 1..11 use the hbot..htop rule
_CLM_SHORT_BEND = 0.8        # grass/crop burial height = 0.8 htop (20% bending)
_CLM_SHORT_HMIN = 0.05       # floor on that burial height [m]
_CLM_VAI_CUT = 0.05          # elai, esai < 0.05 -> 0
_N_CLM_PFT = 17              # natural PFTs 0..16 (surfdata npft, CLM5 order)


def _clm5_pft_optics():
    """Guarded front end of :func:`_clm5_pft_optics_cached`: the tower-site
    override switch is checked on EVERY call, not only when the cache fills."""
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLclm_varctl
    if MLclm_varctl.pftcon_val != 0:
        raise ValueError("canopy snow albedo needs the CLM default PFT optics; "
                         "pftcon_val is set to a tower-site override")
    return _clm5_pft_optics_cached()


@functools.lru_cache(maxsize=1)
def _clm5_pft_optics_cached():
    """CLM5 default per-PFT leaf/stem optics, rows 0..16, columns (vis, nir):
    ``(xl, rhol, taul, rhos, taus)`` from the repo's port of CLM pftconMod, as
    host NumPy arrays (built eagerly even when first called inside a jit trace,
    so the cache never holds a tracer)."""
    from legoesm.land.canopy.clm_ml_backend.clm_src_main import pftconMod as _pc
    from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis
    with jax.ensure_compile_time_eval():
        pc = _pc.InitRead(_pc.InitAllocate())
    from legoesm.land.surface_params import N_PFT_CLM5
    if N_PFT_CLM5 != _N_CLM_PFT:
        raise ValueError(f"surfdata PFT count {N_PFT_CLM5} != CLM5 optics rows {_N_CLM_PFT}")
    rows = slice(0, _N_CLM_PFT)
    two = lambda a: np.stack([np.asarray(a)[rows, ivis], np.asarray(a)[rows, inir]],
                             axis=-1)
    return (np.asarray(pc.xl)[rows], two(pc.rhol), two(pc.taul), two(pc.rhos),
            two(pc.taus))


def clm5_exposed_area(LAI, SAI, htop, hbot, pft_index, f_snow, snow_depth):
    """CLM5 leaf and stem area left exposed above the snow, ``(elai, esai)``
    (SatellitePhenologyMod.F90 173-188): trees and shrubs (PFT 1..11) are
    buried from ``hbot`` up to ``htop``; grasses and crops up to 0.8 ``htop``
    (20% bending, floor 0.05 m); the burial applies on the snow-covered
    fraction ``f_snow``; exposed areas below 0.05 are set to zero."""
    ip = jnp.clip(jnp.round(pft_index).astype(jnp.int32), 0, _N_CLM_PFT - 1)
    tall = (ip > 0) & (ip <= _CLM_TALL_PFT_MAX)
    ol = jnp.clip(snow_depth - hbot, 0.0, htop - hbot)
    fb_tall = 1.0 - ol / jnp.maximum(htop - hbot, 1e-6)
    hb = jnp.maximum(_CLM_SHORT_HMIN, _CLM_SHORT_BEND * htop)
    fb_short = 1.0 - jnp.clip(snow_depth, 0.0, hb) / hb
    fb = jnp.where(tall, fb_tall, fb_short)

    def _exposed(x):
        e = jnp.maximum(x * (1.0 - f_snow) + x * fb * f_snow, 0.0)
        return jnp.where(e < _CLM_VAI_CUT, 0.0, e)

    return _exposed(LAI), _exposed(SAI)


def canopy_masked_snow_albedo(alb_snowfree, alb_snowy, band, LAI, SAI, htop,
                              hbot, pft_index, f_snow, snow_depth, cosz,
                              f_diffuse):
    """Column albedo of one band with the canopy hiding the snow (CLM5).

    DELTA form: the snow-free column albedo is left exactly as supplied and only
    the snow increment is passed through the canopy,

        alb = alb_snowfree + TS(elai, esai; alb_snowy) - TS(tlai, tsai; alb_snowfree)

    with TS the CLM5 two-stream column albedo (:func:`clm5_two_stream_albedo`)
    as the step's light sees it, ``f_diffuse * albi + (1 - f_diffuse) * albd``
    (beam albedo at ``cosz``; the model carries one albedo per band, and the
    canopy RT reflects that one value off both beam and diffuse light, so the
    blend makes the reflected total exact for this step's split), elai/esai the
    CLM5 snow-buried leaf/stem area and ``alb_snowy`` the ground albedo with
    snow already blended on (the unmasked model value).  No exposed plant area
    (none, or all of it buried) -> ``alb_snowy`` exactly; no snow ->
    ``alb_snowfree`` exactly.  ``band`` 0 =
    visible, 1 = near-infrared.  Intercepted canopy snow is not represented
    (fcansno = 0): a bare-branch bound."""
    xl_t, rhol_t, taul_t, rhos_t, taus_t = (jnp.asarray(t) for t in _clm5_pft_optics())
    ip = jnp.clip(jnp.round(pft_index).astype(jnp.int32), 0, _N_CLM_PFT - 1)
    elai, esai = clm5_exposed_area(LAI, SAI, htop, hbot, pft_index, f_snow,
                                   snow_depth)

    def _ts(lai, sai, ground):
        vai = lai + sai
        f_leaf = lai / jnp.maximum(vai, _CLM_MPE)
        albd, albi = clm5_two_stream_albedo(
            vai, f_leaf, rhol_t[ip, band], taul_t[ip, band], rhos_t[ip, band],
            taus_t[ip, band], xl_t[ip], cosz, ground)
        return f_diffuse * albi + (1.0 - f_diffuse) * albd

    tlai = jnp.where(LAI < _CLM_VAI_CUT, 0.0, LAI)
    tsai = jnp.where(SAI < _CLM_VAI_CUT, 0.0, SAI)
    # Parenthesised so a zero increment returns alb_snowfree bit-exactly.  No
    # EXPOSED plant area (none at all, or all buried) returns alb_snowy
    # bit-exactly, the unmasked model value: CTSM puts elai + esai = 0 in its
    # non-vegetated albedo filter (SurfaceAlbedoMod 805-814).
    masked = alb_snowfree + (_ts(elai, esai, alb_snowy)
                             - _ts(tlai, tsai, alb_snowfree))
    return jnp.where(elai + esai > 0.0, masked, alb_snowy)


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
