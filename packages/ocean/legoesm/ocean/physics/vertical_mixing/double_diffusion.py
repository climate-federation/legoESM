"""Double-diffusive vertical mixing (NEMO zdfddm; Merryfield et al. 1999).

Salt fingering and diffusive convection add to the closure's tracer
diffusivities ``avt`` (heat) / ``avs`` (salt) wherever the water column is
statically stable (``N^2 > 0``) but doubly unstable in one component.  The
regime is set by the density ratio at each W-interface::

    R_rho = (alpha * dT/dz) / (beta * dS/dz)

with ``alpha``, ``beta`` the model-EOS thermal-expansion / haline-contraction
coefficients (reuse ``eos.eos_density_derivatives`` — never a re-derived
linear pair).  z is positive up; dT/dz, dS/dz are interface gradients.

* Salt fingering (``1 < R_rho < R_c``, warm-salty over cold-fresh):
  strong salt flux, weaker heat flux (Schmitt 1981 / NEMO namzdf_ddm)::

      avs = rn_avts * (1 - ((R_rho - 1)/(R_c - 1))^2)^3
      avt = 0.7 * avs / R_rho

* Diffusive convection (``0 < R_rho < 1``, cold-fresh over warm-salty;
  Kelley 1984 / Federov 1988 fit)::

      avt = 1.5e-6 * 0.909 * exp(4.6 * exp(-0.54*(1/R_rho - 1)))
      avs = avt * (1.85 - 0.85/R_rho) * R_rho     (0.5 <= R_rho < 1)
          = avt * 0.15 * R_rho                    (0   <  R_rho < 0.5)

Outside both windows (``R_rho <= 0`` single-signed stratification, or
``R_rho >= R_c`` fingering-stable) the contribution is zero, as is any
statically-unstable interface (``N^2 <= 0`` — the convection scheme owns
those).  Momentum (``avm``) is not modified, matching NEMO zdfddm.

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
Kelley, D. E. (1984). Effective diffusivities within oceanic thermohaline
staircases. *JGR*, 89, 10484-10488.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

# --- Salt-fingering fit (Schmitt 1981; NEMO zdfddm.F90 namzdf_ddm) ---
_FINGER_HEAT_RATIO = 0.7          # avt = _FINGER_HEAT_RATIO * avs / R_rho
# --- Diffusive-convection fit (Kelley 1984; Federov 1988) ---
_DC_AVT_PREFACTOR = 1.5e-6        # [m^2/s]
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
        "Double-diffusive vertical mixing (NEMO zdfddm, Merryfield 1999): "
        "salt fingering (1<R_rho<R_c) and diffusive convection (0<R_rho<1) "
        "add regime-dependent avt/avs from the density ratio R_rho = "
        "(alpha dT/dz)/(beta dS/dz). Returns an ADDITIVE (avt_ddm, avs_ddm)."
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
    # JAX-differentiable everywhere (grad flows; no stop_gradient / nondiff
    # ops) — the codebase convention for where/clip-based closures. It is
    # piecewise-C0 (not C1) at the regime boundaries (R_rho = 1, R_c, 0.5) BY
    # NEMO CONSTRUCTION: the Merryfield/Kelley fits are piecewise, so smoothing
    # the joins would break oracle fidelity. Subgradients at the kinks are
    # well-defined and finite (codex r1 differentiability note).
    "differentiable": True,
    "reference": "Merryfield, Holloway & Gargett (1999), JPO 29, 1124-1142",
    "idealized_test": "tests/ocean/unit/test_double_diffusion.py",
}

__param_spec__ = {
    "rn_avts": {
        "units": "m^2/s", "bounds": (1e-5, 5e-4), "tunable_tier": 1,
        "transform": "none", "category": "double_diffusion",
        "reference": "NEMO namzdf_ddm rn_avts", "shape": None,
    },
    "rn_hsbfr": {
        "units": "1", "bounds": (1.2, 2.0), "tunable_tier": 2,
        "transform": "none", "category": "double_diffusion",
        "reference": "NEMO namzdf_ddm (fingering cutoff R_c)", "shape": None,
    },
    "k_max": {
        "units": "m^2/s", "bounds": (1e-3, 1e-1), "tunable_tier": 0,
        "transform": "none", "category": "double_diffusion",
        "reference": "numerical cap", "shape": None,
    },
}


class DoubleDiffusionConfig(NamedTuple):
    """NEMO zdfddm double-diffusive mixing (default OFF; additive avt/avs)."""

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
