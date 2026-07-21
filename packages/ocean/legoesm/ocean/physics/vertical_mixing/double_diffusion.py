"""Double-diffusive vertical mixing (hybrid: Large 1994/CVMix + NEMO/Merryfield).

Salt fingering and diffusive convection add to the closure's tracer
diffusivities ``avt`` (heat) / ``avs`` (salt) wherever the water column is
statically stable (``N^2 > 0``) but doubly unstable in one component.  The
regime is set by the density ratio at each W-interface::

    R_rho = (alpha * dT/dz) / (beta * dS/dz)

with ``alpha``, ``beta`` the model-EOS thermal-expansion / haline-contraction
coefficients (reuse ``eos.eos_density_derivatives`` — never a re-derived
linear pair).  z is positive up; dT/dz, dS/dz are interface gradients.

* Salt fingering (``1 < R_rho < R_c``, warm-salty over cold-fresh):
  strong salt flux, weaker heat flux::

      avs = rn_avts * (1 - ((R_rho - 1)/(R_c - 1))^2)^3
      avt = 0.7 * avs / R_rho

  NOTE the salt-diffusivity SHAPE here is the Large et al. (1994) / CVMix
  (Aug-2012 doc Eq. 6.6) CUBIC with a hard cutoff at ``R_c = rn_hsbfr`` — NOT
  NEMO zdfddm's own salt-fingering form.  NEMO uses the rational
  ``zavfs = rn_avts / (1 + (R_rho/rn_hsbfr)^6)`` gated by its masks
  (``zmsks`` N^2>0, ``zmskf`` R_rho>1, ``zmskr`` R_rho<1000).  WITHIN that mask
  (1 < R_rho < 1000, N^2 > 0) NEMO applies NO cutoff at R_c — it decays smoothly
  and equals rn_avts/2 at R_rho = rn_hsbfr, whereas this module's cubic is 0
  there; NEMO's own cutoff is the far ``R_rho >= 1000`` mask.  The heat/salt
  ratio ``avt = 0.7 avs / R_rho`` IS the NEMO/Merryfield FLUX-ratio convention
  (flux ratio gamma = 0.7 -> diffusivity ratio 0.7/R_rho, before any k_max cap),
  which differs from Large/CVMix's constant 0.7 (their Eq. 6.7 sets separate
  amplitudes kappa0_S = 1e-4, kappa0_T = 0.7e-4).  See "Faithfulness" below.

* Diffusive convection (``0 < R_rho < 1``, cold-fresh over warm-salty;
  Kelley 1984 / Federov 1988 fit)::

      avt = 1.5e-6 * 0.909 * exp(4.6 * exp(-0.54*(1/R_rho - 1)))
      avs = avt * (1.85 - 0.85/R_rho) * R_rho     (0.5 <= R_rho < 1)
          = avt * 0.15 * R_rho                    (0   <  R_rho < 0.5)

  This regime matches NEMO zdfddm (== Large 1994 == CVMix Eq. 6.10-6.12) on the
  open interval, EXCLUDING the AD-guard zones (the |beta_dSdz|>=eps denominator
  floor and the R_rho clip to (eps, 1-eps); e.g. a floored denominator can move
  R_rho out of the raw regime entirely): the prefactor 1.5e-6*0.909 = 1.3635e-6
  matches NEMO's ``zavdt`` literal, and the salt factor
  (1.85 - 0.85/R_rho)*R_rho = 1.85*R_rho - 0.85 matches NEMO's ``zavds`` form.
  The two salt branches agree at R_rho = 0.5,
  so avs is CONTINUOUS there, BUT the value differs from NEMO at that exact point:
  this module's ``where(R>=0.5, hi, lo)`` picks the hi branch (0.075*avt), whereas
  NEMO's dual-STRICT masks (zmskd2: R<0.5, zmskd3: 0.5<R<1) both vanish at R=0.5,
  giving NEMO avds=0 there (a measure-zero boundary departure — see Faithfulness).

Outside both windows (``R_rho <= 0`` single-signed stratification, or
``R_rho >= R_c`` fingering-stable) the contribution is zero, as is any
statically-unstable interface (``N^2 <= 0`` — the convection scheme owns
those).  Momentum (``avm``) is NOT modified — a DEPARTURE from NEMO zdfddm,
which adds ``MAX(avt_ddm, avs_ddm)`` to ``p_avm`` (this leaf returns only the
tracer ``(avt, avs)``, no momentum enhancement).

Faithfulness
------------
Oracles: CVMix Aug-2012 documentation (Griffies et al.) Sec. 6.2-6.3 Eqs.
6.6-6.12, and the NEMO ``zdfddm.F90`` source (the module's declared reference).
``tests/ocean/unit/test_ddm_merryfield_faithful.py`` pins every closed form to
round-off (rel 1e-12) against an INDEPENDENT reimplementation and canaries the
numerically-material departures; the measure-zero / guard-boundary NEMO
differences (below) are DOCUMENTED and, where practical, canaried.

FAITHFUL (on the open regime intervals, away from the AD-guard zones — the
|beta_dSdz|>=eps denominator floor and the R_rho clip near the interval ends):
- Salt-fingering salt diffusivity: the Large 1994 / CVMix Eq. 6.6 cubic
  ``rn_avts*(1-((R_rho-1)/(R_c-1))^2)^3``.
- Salt-fingering heat/salt ratio ``0.7/R_rho`` (NEMO/Merryfield flux ratio),
  before the k_max cap.
- Diffusive convection avt and the two-branch avs (NEMO == CVMix Eq. 6.10-6.12,
  prefactor 1.3635e-6); the VALUE is continuous at R_rho = 0.5 (derivative jumps).

DEPARTURES (documented; canaried where numerically material):
- The salt-fingering amplitude does NOT match the DECLARED NEMO oracle: WITHIN
  NEMO's mask (1 < R_rho < 1000, N^2 > 0) NEMO's rational
  ``rn_avts/(1+(R_rho/rn_hsbfr)^6)`` applies NO cutoff at R_c, while this module's
  Large/CVMix cubic hard-cuts at R_c.  They agree only near R_rho=1; at R_rho=R_c
  (masks all 1) the cubic is 0 while NEMO's rational is rn_avts/2.  [canaried]
- The cutoff default ``rn_hsbfr = 1.6`` is NEMO's namelist default, NOT CVMix's
  R_rho^0 = 2.55 (Eq. 6.5).  [canaried]
- The heat/salt ratio 0.7/R_rho is NEMO/Merryfield, NOT CVMix/Large's constant
  0.7 (their separate kappa0_T = 0.7e-4 amplitude).  [canaried]
- Momentum: NEMO adds ``MAX(avt, avs)`` to ``avm``; this leaf returns only the
  tracer ``(avt, avs)`` and does NOT enhance momentum.  [canaried]
- Single-signed / unstable gating: NEMO clamps the ratio ``zrau = MAX(1e-20,
  zdt/zds)``, so a single-signed interface (raw ratio <= 0) becomes R_rho=1e-20
  and enters NEMO's DIFFUSIVE-CONVECTION regime (avt ~ 1.3635e-6); this module
  sign-PRESERVES the ratio, keeping R_rho <= 0 -> ZERO output.  NEMO's stability
  mask is ``rn2 + 1e-12 <= 0`` (a tiny-negative N^2 still counts as stable),
  whereas this module requires STRICT N^2 > 0.  [canaried: module side = 0]
- Diffusive-convection R_rho = 0.5 point: NEMO's dual-strict masks give avds = 0
  there; this module's ``where(R>=0.5,...)`` returns the hi branch 0.075*avt.
  [canaried]
- A ``k_max`` safety cap clips both outputs (inactive at the default k_max=1e-2
  vs the ~1e-4 fingering / ~1e-6 convection magnitudes; a small k_max caps both,
  breaking the 0.7/R_rho ratio toward 1).  [canaried]
- AD-guards (not in the oracle): a sign-preserving |beta_dSdz| >= eps floor, and
  R_rho clipped to (1+eps, R_c-eps) [fingering] / (eps, 1-eps) [convection]
  before the power/exp, eps = float32 machine epsilon.  The MATERIALLY-observable
  clips are canaried: fingering UPPER (R_c-eps) and BOTH convection bounds
  (eps, 1-eps).  The fingering LOWER clamp (1+eps) is a DEFENSIVE no-op — the
  cubic is flat-maximal at R_rho->1, so it changes avs by only ~1e-13 (documented,
  not a discriminating canary).  [canaried: floor + 3 of the 4 clip layers]

Integration status
------------------
Increment 1 (this module): the leaf tendency ``compute_ddm_diffusivity`` +
its unit test ``tests/ocean/unit/test_double_diffusion.py``.  Increment 2
wires ``DoubleDiffusionConfig`` into ``VerticalMixingConfig.ddm`` and adds a
``ddm_K_profile`` to ``k_profiles.py`` that ADDS ``avt_ddm`` to the heat
diffusivity and returns the salt-heat delta ``avs_ddm - avt_ddm`` so the
implicit tracer solve routes S through a SEPARATE salinity diffusivity
(``implicit_vertical_diffusion_ocean_batched`` already supports per-field K;
the shared-K ``_pair`` fast path stays byte-identical when ``ddm.enabled`` is
False).  See ``docs/ocean/specs/double_diffusion_zdfddm.md``.

References
----------
Merryfield, W. J., Holloway, G. & Gargett, A. E. (1999). A global ocean
model with double-diffusive mixing. *JPO*, 29, 1124-1142.
Large, W. G., McWilliams, J. C. & Doney, S. C. (1994). Oceanic vertical
mixing. *Rev. Geophys.*, 32, 363-403 (salt-fingering cubic + convective fit).
Griffies, S. M. et al. (2012). CVMix documentation (Aug 14 2012), Sec. 6.2-6.3
Eqs. 6.6-6.12 (double-diffusion oracle used here).
Kelley, D. E. (1984). Effective diffusivities within oceanic thermohaline
staircases. *JGR*, 89, 10484-10488.
NEMO ``zdfddm.F90`` (forge.nemo-ocean.eu) — the declared reference; note its
salt-fingering avfs is a rational form, a documented departure (see above).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# --- Salt-fingering heat/salt flux ratio (NEMO zdfddm / Merryfield 1999) ---
# gamma = 0.7 is the salt-finger FLUX ratio -> diffusivity ratio 0.7/R_rho
# (the salt-diffusivity CUBIC amplitude is Large 1994 / CVMix Eq. 6.6, on rn_avts).
_FINGER_HEAT_RATIO = 0.7          # avt = _FINGER_HEAT_RATIO * avs / R_rho
# --- Diffusive-convection fit (Kelley 1984 / Federov 1988; == NEMO/CVMix Eq. 6.10-6.12) ---
_DC_AVT_PREFACTOR = 1.5e-6        # [m^2/s]  (1.5e-6*0.909 = 1.3635e-6 = NEMO zavdt literal)
_DC_AVT_C0 = 0.909
_DC_AVT_C1 = 4.6
_DC_AVT_C2 = -0.54
_DC_AVS_HI_A = 1.85               # avs/avt slope, 0.5 <= R_rho < 1
_DC_AVS_HI_B = 0.85
_DC_AVS_LO = 0.15                 # avs/avt slope, R_rho < 0.5
_DC_RRHO_SPLIT = 0.5
_EPS = float(jnp.finfo(jnp.float32).eps)

__physics_contract__ = {
    "summary": (
        "Double-diffusive vertical mixing (hybrid: Large 1994/CVMix Eq. 6.6 "
        "salt-fingering cubic + NEMO/Merryfield 0.7/R_rho heat flux-ratio + "
        "NEMO/CVMix Eq. 6.10-6.12 diffusive convection): salt fingering "
        "(1<R_rho<R_c) and diffusive convection (0<R_rho<1) add regime-dependent "
        "avt/avs from R_rho = (alpha dT/dz)/(beta dS/dz). Additive (avt, avs)."
    ),
    "inputs": {
        "N2": "1/s^2 (Brunt-Vaisala at the W-interface)",
        "alpha_dTdz": "1/s^2 (alpha * dT/dz, alpha from eos)",
        "beta_dSdz": "1/s^2 (beta * dS/dz, beta from eos)",
    },
    "outputs": {"avt_ddm": "m^2/s", "avs_ddm": "m^2/s"},
    "sign_convention": (
        "avt_ddm, avs_ddm >= 0; both zero outside the fingering (1,R_c) and "
        "diffusive-convection (0,1) R_rho windows and where N^2<=0. z up. "
        "Additive to the closure avt/avs on the implicit path (conserves "
        "column heat and salt via flux-form implicit diffusion, no-flux BC)."
    ),
    "conserves": ["energy", "salt"],
    # "differentiable": True is the codebase convention = AD-COMPATIBLE / defined
    # VJP (no stop_gradient / nondiff ops; jax.grad returns finite selected-branch
    # gradients a.e.), NOT everywhere-smooth. The scheme is NOT even C0 where a
    # where-mask switches a NONZERO branch on/off: avs JUMPS from 0 (R_rho<=1) to
    # ~rn_avts as R_rho->1+ (the cubic is maximal at R_rho=1), and there are
    # analogous jumps at R_rho=0 and the N^2 gate. (At R_c the RAW cubic ->0
    # continuously, matching the 0 above; the AD clip evaluating at R_c-eps leaves
    # a negligible O(eps^3) implementation plateau then 0 at R_c — a tiny
    # implementation jump, not a material one.) The R_rho=0.5 DC join is continuous
    # in VALUE (both salt branches = 0.075*avt there; the DERIVATIVE differs). These
    # jumps are BY the Large/CVMix/NEMO piecewise construction (smoothing the gates
    # would break oracle fidelity); at a gate the gradient is the one-sided
    # selected-branch value, not a true subgradient.
    "differentiable": True,
    "reference": (
        "Large et al. (1994) Rev.Geophys. 32 + CVMix Aug-2012 Eqs. 6.6-6.12 "
        "(fingering cubic, diffusive convection); NEMO zdfddm / Merryfield et al. "
        "(1999) JPO 29 (0.7/R_rho heat flux-ratio)"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_double_diffusion.py + "
        "tests/ocean/unit/test_ddm_merryfield_faithful.py"
    ),
}

__param_spec__ = {
    "DoubleDiffusionConfig": {
        "scheme_key": "ocean.vm.ddm",
        "excluded": {
            "k_max": "numerics: bound (diffusivity cap)",
        },
        "params": {
            "rn_avts": {
                "units": "m^2/s", "bounds": (1e-5, 5e-4), "tunable_tier": 1,
                "transform": "none", "category": "double_diffusion",
                "reference": "NEMO namzdf_ddm rn_avts", "shape": None,
            },
            "rn_hsbfr": {
                "units": "1", "bounds": (1.2, 2.0), "tunable_tier": 2,
                "transform": "none", "category": "double_diffusion",
                # Default 1.6 = NEMO namzdf_ddm value. ROLE differs: here it is the
                # Large/CVMix cubic HARD cutoff R_c; in NEMO it is the RATIONAL
                # decay scale (avfs = rn_avts/2 at R_rho = rn_hsbfr, no cutoff).
                "reference": "NEMO namzdf_ddm rn_hsbfr (cubic cutoff R_c here)",
                "shape": None,
            },
        },
    },
}


class DoubleDiffusionConfig(NamedTuple):
    """Double-diffusive mixing config (Large 1994/CVMix cubic + NEMO/Merryfield
    0.7/R flux-ratio + NEMO/CVMix convection; default OFF; additive avt/avs)."""

    enabled: bool = False
    rn_avts: float = 1e-4     # max salt-fingering salt diffusivity [m^2/s]
    rn_hsbfr: float = 1.6     # salt-fingering cutoff density ratio R_c [1]
    k_max: float = 1e-2       # diffusivity cap [m^2/s]


def compute_ddm_diffusivity(
    N2: jnp.ndarray,
    alpha_dTdz: jnp.ndarray,
    beta_dSdz: jnp.ndarray,
    cfg: DoubleDiffusionConfig,
):
    """Additive double-diffusive (avt_ddm, avs_ddm) at each W-interface.

    Pure, traced-safe: all regime selection via ``jnp.where`` (no Python
    branching on the arrays).  Zero everywhere when ``N2<=0`` or R_rho is
    outside both double-diffusive windows.
    """
    # R_rho = (alpha dT/dz)/(beta dS/dz); floor |denominator| to _EPS while
    # PRESERVING sign, so a vanishing salinity gradient can't blow up the ratio
    # or flip its sign (traced-safe, no branch).
    denom = jnp.where(beta_dSdz >= 0.0,
                      jnp.maximum(beta_dSdz, _EPS),
                      jnp.minimum(beta_dSdz, -_EPS))
    R_rho = alpha_dTdz / denom
    stable = N2 > 0.0

    # --- Salt fingering: 1 < R_rho < R_c ---
    Rc = cfg.rn_hsbfr
    finger = stable & (R_rho > 1.0) & (R_rho < Rc)
    # Clip R_rho into (1, Rc) before the cubic so the where-masked-out lanes
    # can't produce NaN/negative under the power.
    Rf = jnp.clip(R_rho, 1.0 + _EPS, Rc - _EPS)
    frac = (Rf - 1.0) / (Rc - 1.0)                       # in (0,1)
    avs_finger = cfg.rn_avts * (1.0 - frac * frac) ** 3   # >= 0
    # Heat ratio uses the ACTUAL in-window R_rho (Merryfield: 0.7*avs/R_rho),
    # NOT the clipped Rf (which would err by ~_EPS within eps of 1 / Rc — codex
    # r1).  In-window R_rho>1 so the divide is safe; out-of-window lanes divide
    # by 1.0 (finite, AD-safe — a raw R_rho there could be 0/negative and, though
    # masked below, would poison reverse-mode grads via 0*inf).
    R_finger = jnp.where(finger, R_rho, 1.0)
    avt_finger = _FINGER_HEAT_RATIO * avs_finger / R_finger

    # --- Diffusive convection: 0 < R_rho < 1 ---
    dc = stable & (R_rho > 0.0) & (R_rho < 1.0)
    Rd = jnp.clip(R_rho, _EPS, 1.0 - _EPS)
    avt_dc = (_DC_AVT_PREFACTOR * _DC_AVT_C0
              * jnp.exp(_DC_AVT_C1 * jnp.exp(_DC_AVT_C2 * (1.0 / Rd - 1.0))))
    avs_dc = jnp.where(
        Rd >= _DC_RRHO_SPLIT,
        avt_dc * (_DC_AVS_HI_A - _DC_AVS_HI_B / Rd) * Rd,
        avt_dc * _DC_AVS_LO * Rd,
    )

    avt = jnp.where(finger, avt_finger, jnp.where(dc, avt_dc, 0.0))
    avs = jnp.where(finger, avs_finger, jnp.where(dc, avs_dc, 0.0))
    avt = jnp.clip(avt, 0.0, cfg.k_max)
    avs = jnp.clip(avs, 0.0, cfg.k_max)
    return avt, avs
