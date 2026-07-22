"""
JAX translation of MLCanopyTurbulenceMod Fortran module.

Canopy turbulence parameterisation: roughness sublayer (RSL) theory
of Harman and Finnigan (2008), wind speed profile, and aerodynamic
conductances.

Public routines
---------------
- :func:`CanopyTurbulence`: main driver.
- :func:`LookupPsihatINI`: initialise RSL psihat look-up tables from
  a NetCDF file.

Private helpers
---------------
- :func:`_HF2008`: HF2008 RSL canopy turbulence.
- :func:`_GetObu`: solve for Obukhov length.
- :func:`_ObuFunc`: ObuFunc callback for hybrid root solver.
- :func:`_GetBeta`: β = u*/u(h) for a given stability.
- :func:`_GetPrSc`: Prandtl/Schmidt number at canopy top.
- :func:`_GetPsiRSL`: RSL-modified stability functions.
- :func:`_phim_monin_obukhov`: MO φ for momentum.
- :func:`_phic_monin_obukhov`: MO φ for scalars.
- :func:`_psim_monin_obukhov`: MO ψ for momentum.
- :func:`_psic_monin_obukhov`: MO ψ for scalars.
- :func:`_LookupPsihat`: bilinear interpolation in psihat table.
- :func:`_RoughnessLength`: roughness length for momentum.
- :func:`_WindProfile`: wind speed above and within canopy.
- :func:`_AerodynamicConductance`: aerodynamic conductances.

Original Fortran module: MLCanopyTurbulenceMod
Fortran lines 1-560
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import grav, vkc  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import pi as rpi
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog, rslfile  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import (  # noqa: F401
    LcL_max,
    LcL_min,
    Pr0,
    Pr1,
    Pr2,
    aH12,
    beta_neutral_max,
    c2,
    cd,
    cr,
    dtLgridH,
    dtLgridM,
    eta_max,
    mmdry,
    mmh2o,
    nL,
    nZ,
    psigridH,
    psigridM,
    ra_max,
    z0mg,
    zdtgridH,
    zdtgridM,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import (  # noqa: F401
    DIFFERENTIABLE_MODE,
    GridInfo,
    HF_extension_type,
    sparse_canopy_type,
    turb_type,
)
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLMathToolsMod import hybrid, hybrid_scalar  # noqa: F401

# Python-float aliases for aH12 elements — used inside _ObuFuncPure which is called
# on every solver iteration; accessing aH12[i] (a JAX array) would force an XLA sync
# each time, adding ~100–200 syncs per _GetObu call.
_aH12_0: float = 0.89
_aH12_1: float = -0.07
_aH12_2: float = 2.19

# ---------------------------------------------------------------------------
# Cached 1D views of psihat look-up table grids — populated by LookupPsihatINI.
# _LookupPsihat creates a 276-element negated array (-zdtg) on every call for
# np.searchsorted; with ~220 calls/patch/step that's ~110,000 allocations/run.
# Using module-level pre-computed arrays eliminates those allocations and the
# 2D→1D slice operations (dtLgrid[0], zdtgrid[:, 0]) done on every call.
# ---------------------------------------------------------------------------
_zdtgM_1d: np.ndarray = np.empty(0)  # zdtgridM[:, 0]  (populated after init)
_dtLgM_1d: np.ndarray = np.empty(0)  # dtLgridM[0]
_neg_zdtgM_1d: np.ndarray = np.empty(0)  # -zdtgridM[:, 0]
_nZ_M: int = nZ
_nL_M: int = nL
_zdtgH_1d: np.ndarray = np.empty(0)  # zdtgridH[:, 0]
_dtLgH_1d: np.ndarray = np.empty(0)  # dtLgridH[0]
_neg_zdtgH_1d: np.ndarray = np.empty(0)  # -zdtgridH[:, 0]
_nZ_H: int = nZ
_nL_H: int = nL

# Pre-converted JAX array versions of lookup tables — populated by LookupPsihatINI.
# _LookupPsihatM/H called jnp.asarray(...) on every invocation (~10K times per
# forward pass), triggering repeated CPU→GPU copies of the psihat grid.
# These module-level JAX arrays eliminate that overhead.
_dtLgM_jax: jax.Array | None = None
_zdtgM_jax: jax.Array | None = None
_neg_zdtgM_jax: jax.Array | None = None
_psigridM_jax: jax.Array | None = None
_dtLgH_jax: jax.Array | None = None
_zdtgH_jax: jax.Array | None = None
_neg_zdtgH_jax: jax.Array | None = None
_psigridH_jax: jax.Array | None = None

# ===========================================================================
# Private: Monin-Obukhov stability functions
# ===========================================================================


def _phim_monin_obukhov(zeta):
    """
    Monin-Obukhov φ stability function for momentum.

    Mirrors Fortran function ``phim_monin_obukhov`` (lines 330-343).

    Reference: Bonan et al. (2018), eq. (A10).

    .. code-block:: none

        zeta < 0 (unstable): phi = 1 / (1 - 16*zeta)^(1/4)
        zeta ≥ 0 (stable):   phi = 1 + 5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        φ for momentum.
    """
    zeta = jnp.asarray(zeta)
    # jnp.abs avoids sqrt(negative) NaN for stable (zeta>0) branch gradient
    unstable = 1.0 / jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    stable = 1.0 + 5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def _phic_monin_obukhov(zeta):
    """
    Monin-Obukhov φ stability function for scalars.

    Mirrors Fortran function ``phic_monin_obukhov`` (lines 345-358).

    Reference: Bonan et al. (2018), eq. (A11).

    .. code-block:: none

        zeta < 0 (unstable): phi = 1 / (1 - 16*zeta)^(1/2)
        zeta ≥ 0 (stable):   phi = 1 + 5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        φ for scalars.
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids sqrt(negative) NaN for stable (zeta>0) branch gradient
    unstable = 1.0 / jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10))
    stable = 1.0 + 5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def _psim_monin_obukhov(zeta):
    """
    Monin-Obukhov ψ stability function for momentum.

    Mirrors Fortran function ``psim_monin_obukhov`` (lines 360-378).

    Reference: Bonan et al. (2018), eq. (A12).

    .. code-block:: none

        zeta < 0 (unstable):
            x   = (1 - 16*zeta)^(1/4)
            psi = 2*log((1+x)/2) + log((1+x^2)/2) - 2*atan(x) + pi/2
        zeta ≥ 0 (stable):
            psi = -5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        ψ for momentum.
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids jnp.abs(0)^(-3/4) * sign(0) = inf * 0 = NaN in backward.
    # When zeta >= 0 the stable branch is selected; this branch expression is
    # discarded but its VJP is still evaluated. Using jnp.maximum clips the
    # derivative at the boundary (d(max(u,1e-10))/du = 0 when u <= 1e-10)
    # so the unselected-branch gradient is 0 * finite = 0, not 0 * inf = NaN.
    # Forward values are unchanged for any zeta actually selected (zeta < 0).
    x = jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    unstable = (
        2.0 * jnp.log((1.0 + x) / 2.0)
        + jnp.log((1.0 + x * x) / 2.0)
        - 2.0 * jnp.arctan(x)
        + rpi * 0.5
    )
    stable = -5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


def _psic_monin_obukhov(zeta):
    """
    Monin-Obukhov ψ stability function for scalars.

    Mirrors Fortran function ``psic_monin_obukhov`` (lines 380-395).

    Reference: Bonan et al. (2018), eq. (A13).

    .. code-block:: none

        zeta < 0 (unstable):
            x   = (1 - 16*zeta)^(1/4)
            psi = 2*log((1+x^2)/2)
        zeta ≥ 0 (stable):
            psi = -5*zeta

    Args:
        zeta: Monin-Obukhov stability parameter (JAX scalar or float).

    Returns:
        ψ for scalars.
    """
    zeta = jnp.asarray(zeta)
    # jnp.maximum avoids jnp.abs(0)^(-3/4) * sign(0) = inf * 0 = NaN in backward.
    # Same reasoning as _psim_monin_obukhov above.
    x = jnp.sqrt(jnp.sqrt(jnp.maximum(1.0 - 16.0 * zeta, 1.0e-10)))
    unstable = 2.0 * jnp.log((1.0 + x * x) / 2.0)
    stable = -5.0 * zeta
    return jnp.where(zeta < 0.0, unstable, stable)


# ===========================================================================
# Private: RSL psihat look-up table interpolation
# ===========================================================================


def _LookupPsihat(
    zdt: float,
    dtL: float,
    zdtgrid,  # shape (nZ, 1)
    dtLgrid,  # shape (1, nL)
    psigrid,  # shape (nZ, nL)
) -> float:
    """
    Bilinear interpolation of the RSL psihat function from a look-up table.

    Mirrors Fortran subroutine ``LookupPsihat`` (lines 398-448).

    Reference: Bonan et al. (2018), eq. (A21).

    The returned value is the unscaled amplitude ``A[(z-h)/(h-d),(h-d)/L]``;
    it must be multiplied by ``c1`` before it fully represents psihat.

    Look-up table grid conventions (1-based in Fortran, 0-based here):

    - ``zdtgrid`` decreases with increasing row index (height above canopy
      decreasing toward zero as index increases); boundary clamping applies
      when ``zdt`` exceeds the table range.
    - ``dtLgrid`` increases with increasing column index; boundary clamping
      applies when ``dtL`` is outside the table range.

    Args:
        zdt: Normalised height above canopy ``(z-h)/(h-d)``.
        dtL: Stability ratio ``(h-d)/L``.
        zdtgrid: Grid of ``zdt`` values; shape ``(nZ, 1)``.
        dtLgrid: Grid of ``dtL`` values; shape ``(1, nL)``.
        psigrid: Psihat values on the grid; shape ``(nZ, nL)``.

    Returns:
        Interpolated (unscaled) psihat value.
    """
    dtLg = jnp.asarray(dtLgrid[0])  # 1-D, length nL, increasing
    zdtg = jnp.asarray(zdtgrid[:, 0])  # 1-D, length nZ, DECREASING
    nZ_tbl = len(zdtg)
    nL_tbl = len(dtLg)

    dtL = jnp.asarray(dtL)
    zdt = jnp.asarray(zdt)

    # --- dtL bracketing — Fortran lines 412-428 ---
    at_left_L = dtL <= dtLg[0]
    at_right_L = dtL > dtLg[-1]
    L2_raw = jnp.clip(jnp.searchsorted(dtLg, dtL, side="right"), 0, nL_tbl - 1)
    L1_raw = jnp.maximum(L2_raw - 1, 0)
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_L = jnp.maximum(dtLg[L2_raw] - dtLg[L1_raw], 1.0e-30)
    wL1_raw = (dtLg[L2_raw] - dtL) / denom_L
    wL2_raw = 1.0 - wL1_raw
    at_bnd_L = at_left_L | at_right_L
    wL1 = jnp.where(at_bnd_L, 0.5, wL1_raw)
    wL2 = jnp.where(at_bnd_L, 0.5, wL2_raw)
    L1 = jnp.where(at_left_L, 0, jnp.where(at_right_L, nL_tbl - 1, L1_raw))
    L2 = jnp.where(at_left_L, 0, jnp.where(at_right_L, nL_tbl - 1, L2_raw))

    # --- zdt bracketing — Fortran lines 430-448 (zdtg DECREASES) ---
    neg_zdtg = -zdtg
    at_top_Z = zdt > zdtg[0]
    at_bottom_Z = zdt < zdtg[-1]
    Z1_raw = jnp.clip(jnp.searchsorted(neg_zdtg, -zdt, side="right") - 1, 0, nZ_tbl - 2)
    Z2_raw = Z1_raw + 1
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_Z = jnp.maximum(zdtg[Z1_raw] - zdtg[Z2_raw], 1.0e-30)
    wZ1_raw = (zdt - zdtg[Z2_raw]) / denom_Z
    wZ2_raw = 1.0 - wZ1_raw
    at_bnd_Z = at_top_Z | at_bottom_Z
    wZ1 = jnp.where(at_bnd_Z, 0.5, wZ1_raw)
    wZ2 = jnp.where(at_bnd_Z, 0.5, wZ2_raw)
    Z1 = jnp.where(at_top_Z, 0, jnp.where(at_bottom_Z, nZ_tbl - 1, Z1_raw))
    Z2 = jnp.where(at_top_Z, 0, jnp.where(at_bottom_Z, nZ_tbl - 1, Z2_raw))

    psigrid = jnp.asarray(psigrid)
    # Bilinear interpolation — Fortran lines 450-452
    return (
        wZ1 * wL1 * psigrid[Z1, L1]
        + wZ2 * wL1 * psigrid[Z2, L1]
        + wZ1 * wL2 * psigrid[Z1, L2]
        + wZ2 * wL2 * psigrid[Z2, L2]
    )


# ---------------------------------------------------------------------------
# Fast specialized lookups using module-level pre-computed 1D arrays.
# These eliminate the per-call slice and negation overhead of _LookupPsihat.
# Called by _GetPsiRSL after LookupPsihatINI has populated the caches.
# ---------------------------------------------------------------------------


def _LookupPsihatM(zdt, dtL):
    """Momentum psihat lookup using module-level cached JAX arrays.

    Uses pre-converted _*_jax arrays (populated by LookupPsihatINI) to avoid
    jnp.asarray() on every call — eliminates ~10K GPU copies per forward pass.
    """
    dtL = jnp.asarray(dtL)
    zdt = jnp.asarray(zdt)
    # Use pre-converted JAX arrays; fall back to on-the-fly conversion only
    # before LookupPsihatINI has been called (should not happen in practice).
    dtLg = _dtLgM_jax if _dtLgM_jax is not None else jnp.asarray(_dtLgM_1d)
    zdtg = _zdtgM_jax if _zdtgM_jax is not None else jnp.asarray(_zdtgM_1d)
    neg_zdtg = _neg_zdtgM_jax if _neg_zdtgM_jax is not None else jnp.asarray(_neg_zdtgM_1d)
    nL = _nL_M
    nZ = _nZ_M

    at_bnd_L = (dtL <= dtLg[0]) | (dtL > dtLg[-1])
    L2_raw = jnp.clip(jnp.searchsorted(dtLg, dtL, side="right"), 0, nL - 1)
    L1_raw = jnp.maximum(L2_raw - 1, 0)
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_L = jnp.maximum(dtLg[L2_raw] - dtLg[L1_raw], 1.0e-30)
    wL1 = jnp.where(at_bnd_L, 0.5, (dtLg[L2_raw] - dtL) / denom_L)
    wL2 = 1.0 - wL1
    L1 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L1_raw))
    L2 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L2_raw))

    at_bnd_Z = (zdt > zdtg[0]) | (zdt < zdtg[-1])
    Z1_raw = jnp.clip(jnp.searchsorted(neg_zdtg, -zdt, side="right") - 1, 0, nZ - 2)
    Z2_raw = Z1_raw + 1
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_Z = jnp.maximum(zdtg[Z1_raw] - zdtg[Z2_raw], 1.0e-30)
    wZ1 = jnp.where(at_bnd_Z, 0.5, (zdt - zdtg[Z2_raw]) / denom_Z)
    wZ2 = 1.0 - wZ1
    Z1 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z1_raw))
    Z2 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z2_raw))

    pg = _psigridM_jax if _psigridM_jax is not None else jnp.asarray(psigridM)
    return (
        wZ1 * wL1 * pg[Z1, L1]
        + wZ2 * wL1 * pg[Z2, L1]
        + wZ1 * wL2 * pg[Z1, L2]
        + wZ2 * wL2 * pg[Z2, L2]
    )


def _LookupPsihatH(zdt, dtL):
    """Heat/scalar psihat lookup using module-level cached JAX arrays.

    Uses pre-converted _*_jax arrays (populated by LookupPsihatINI) to avoid
    jnp.asarray() on every call — eliminates ~10K GPU copies per forward pass.
    """
    dtL = jnp.asarray(dtL)
    zdt = jnp.asarray(zdt)
    dtLg = _dtLgH_jax if _dtLgH_jax is not None else jnp.asarray(_dtLgH_1d)
    zdtg = _zdtgH_jax if _zdtgH_jax is not None else jnp.asarray(_zdtgH_1d)
    neg_zdtg = _neg_zdtgH_jax if _neg_zdtgH_jax is not None else jnp.asarray(_neg_zdtgH_1d)
    nL = _nL_H
    nZ = _nZ_H

    at_bnd_L = (dtL <= dtLg[0]) | (dtL > dtLg[-1])
    L2_raw = jnp.clip(jnp.searchsorted(dtLg, dtL, side="right"), 0, nL - 1)
    L1_raw = jnp.maximum(L2_raw - 1, 0)
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_L = jnp.maximum(dtLg[L2_raw] - dtLg[L1_raw], 1.0e-30)
    wL1 = jnp.where(at_bnd_L, 0.5, (dtLg[L2_raw] - dtL) / denom_L)
    wL2 = 1.0 - wL1
    L1 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L1_raw))
    L2 = jnp.where(dtL <= dtLg[0], 0, jnp.where(dtL > dtLg[-1], nL - 1, L2_raw))

    at_bnd_Z = (zdt > zdtg[0]) | (zdt < zdtg[-1])
    Z1_raw = jnp.clip(jnp.searchsorted(neg_zdtg, -zdt, side="right") - 1, 0, nZ - 2)
    Z2_raw = Z1_raw + 1
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug
    denom_Z = jnp.maximum(zdtg[Z1_raw] - zdtg[Z2_raw], 1.0e-30)
    wZ1 = jnp.where(at_bnd_Z, 0.5, (zdt - zdtg[Z2_raw]) / denom_Z)
    wZ2 = 1.0 - wZ1
    Z1 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z1_raw))
    Z2 = jnp.where(zdt > zdtg[0], 0, jnp.where(zdt < zdtg[-1], nZ - 1, Z2_raw))

    pg = _psigridH_jax if _psigridH_jax is not None else jnp.asarray(psigridH)
    return (
        wZ1 * wL1 * pg[Z1, L1]
        + wZ2 * wL1 * pg[Z2, L1]
        + wZ1 * wL2 * pg[Z1, L2]
        + wZ2 * wL2 * pg[Z2, L2]
    )


# ===========================================================================
# Private: Pure-Python scalar versions of MO stability and psihat functions.
# These are used ONLY inside the iterative root solvers (_ObuFuncPure,
# _z0m_func) which receive plain Python floats.  Replacing jnp.* with
# math.* eliminates ~8600 JAX eager-dispatch overhead calls per
# CanopyTurbulence invocation, giving ~50× speedup for those code paths.
# The original JAX versions are kept for vectorised/JIT-compiled use.
# ===========================================================================


def _phim_scalar(zeta: float) -> float:
    """Scalar (math.*) version of _phim_monin_obukhov for root-solver use."""
    if zeta < 0.0:
        return 1.0 / math.sqrt(math.sqrt(1.0 - 16.0 * zeta))
    return 1.0 + 5.0 * zeta


def _phic_scalar(zeta: float) -> float:
    """Scalar version of _phic_monin_obukhov."""
    if zeta < 0.0:
        return 1.0 / math.sqrt(1.0 - 16.0 * zeta)
    return 1.0 + 5.0 * zeta


def _psim_scalar(zeta: float) -> float:
    """Scalar version of _psim_monin_obukhov."""
    if zeta < 0.0:
        x = math.sqrt(math.sqrt(abs(1.0 - 16.0 * zeta)))
        return (
            2.0 * math.log((1.0 + x) / 2.0)
            + math.log((1.0 + x * x) / 2.0)
            - 2.0 * math.atan(x)
            + math.pi / 2.0
        )
    return -5.0 * zeta


def _psic_scalar(zeta: float) -> float:
    """Scalar version of _psic_monin_obukhov."""
    if zeta < 0.0:
        x = math.sqrt(math.sqrt(abs(1.0 - 16.0 * zeta)))
        return 2.0 * math.log((1.0 + x * x) / 2.0)
    return -5.0 * zeta


def _LookupPsihatM_scalar(zdt: float, dtL: float) -> float:
    """Scalar numpy-based psihat momentum lookup (no JAX dispatch overhead)."""
    dtLg = _dtLgM_1d  # numpy 1-D arrays populated by LookupPsihatINI
    zdtg = _zdtgM_1d
    neg_zdtg = _neg_zdtgM_1d
    pg = psigridM  # numpy 2-D array
    nL_t = _nL_M
    nZ_t = _nZ_M

    # --- dtL bracketing ---
    at_bnd_L = (dtL <= dtLg[0]) or (dtL > dtLg[-1])
    L2_raw = int(min(max(np.searchsorted(dtLg, dtL, side="right"), 0), nL_t - 1))
    L1_raw = max(L2_raw - 1, 0)
    if at_bnd_L:
        wL1 = wL2 = 0.5
        L1 = L2 = 0 if dtL <= dtLg[0] else nL_t - 1
    else:
        denom_L = dtLg[L2_raw] - dtLg[L1_raw] if L2_raw != L1_raw else 1.0
        wL1 = (dtLg[L2_raw] - dtL) / denom_L
        wL2 = 1.0 - wL1
        L1, L2 = L1_raw, L2_raw

    # --- zdt bracketing (zdtg DECREASES) ---
    at_bnd_Z = (zdt > zdtg[0]) or (zdt < zdtg[-1])
    Z1_raw = int(max(min(np.searchsorted(neg_zdtg, -zdt, side="right") - 1, nZ_t - 2), 0))
    Z2_raw = Z1_raw + 1
    if at_bnd_Z:
        wZ1 = wZ2 = 0.5
        Z1 = Z2 = 0 if zdt > zdtg[0] else nZ_t - 1
    else:
        denom_Z = zdtg[Z1_raw] - zdtg[Z2_raw] if zdtg[Z1_raw] != zdtg[Z2_raw] else 1.0
        wZ1 = (zdt - zdtg[Z2_raw]) / denom_Z
        wZ2 = 1.0 - wZ1
        Z1, Z2 = Z1_raw, Z2_raw

    return (
        wZ1 * wL1 * float(pg[Z1, L1])
        + wZ2 * wL1 * float(pg[Z2, L1])
        + wZ1 * wL2 * float(pg[Z1, L2])
        + wZ2 * wL2 * float(pg[Z2, L2])
    )


def _LookupPsihatH_scalar(zdt: float, dtL: float) -> float:
    """Scalar numpy-based psihat scalar lookup (no JAX dispatch overhead)."""
    dtLg = _dtLgH_1d
    zdtg = _zdtgH_1d
    neg_zdtg = _neg_zdtgH_1d
    pg = psigridH
    nL_t = _nL_H
    nZ_t = _nZ_H

    at_bnd_L = (dtL <= dtLg[0]) or (dtL > dtLg[-1])
    L2_raw = int(min(max(np.searchsorted(dtLg, dtL, side="right"), 0), nL_t - 1))
    L1_raw = max(L2_raw - 1, 0)
    if at_bnd_L:
        wL1 = wL2 = 0.5
        L1 = L2 = 0 if dtL <= dtLg[0] else nL_t - 1
    else:
        denom_L = dtLg[L2_raw] - dtLg[L1_raw] if L2_raw != L1_raw else 1.0
        wL1 = (dtLg[L2_raw] - dtL) / denom_L
        wL2 = 1.0 - wL1
        L1, L2 = L1_raw, L2_raw

    at_bnd_Z = (zdt > zdtg[0]) or (zdt < zdtg[-1])
    Z1_raw = int(max(min(np.searchsorted(neg_zdtg, -zdt, side="right") - 1, nZ_t - 2), 0))
    Z2_raw = Z1_raw + 1
    if at_bnd_Z:
        wZ1 = wZ2 = 0.5
        Z1 = Z2 = 0 if zdt > zdtg[0] else nZ_t - 1
    else:
        denom_Z = zdtg[Z1_raw] - zdtg[Z2_raw] if zdtg[Z1_raw] != zdtg[Z2_raw] else 1.0
        wZ1 = (zdt - zdtg[Z2_raw]) / denom_Z
        wZ2 = 1.0 - wZ1
        Z1, Z2 = Z1_raw, Z2_raw

    return (
        wZ1 * wL1 * float(pg[Z1, L1])
        + wZ2 * wL1 * float(pg[Z2, L1])
        + wZ1 * wL2 * float(pg[Z1, L2])
        + wZ2 * wL2 * float(pg[Z2, L2])
    )


def _GetPsiRSL_scalar(
    za: float,
    hc: float,
    disp: float,
    obu: float,
    beta: float,
    PrSc: float,
) -> tuple:
    """Scalar (pure Python/math.*) version of _GetPsiRSL for root-solver use."""
    dt = hc - disp
    zdt_za = (za - hc) / dt
    zdt_hc = 0.0
    dtL = dt / obu

    # Momentum
    phim = _phim_scalar((hc - disp) / obu)
    c1_m = (1.0 - vkc / (2.0 * beta * phim)) * math.exp(0.5 * c2)
    psim_hat1 = _LookupPsihatM_scalar(zdt_za, dtL) * c1_m
    psim_hat2 = _LookupPsihatM_scalar(zdt_hc, dtL) * c1_m
    psim1 = _psim_scalar((za - disp) / obu)
    psim2 = _psim_scalar((hc - disp) / obu)
    psim = -psim1 + psim2 + psim_hat1 - psim_hat2 + vkc / beta

    # Scalars
    phic = _phic_scalar((hc - disp) / obu)
    c1_c = (1.0 - PrSc * vkc / (2.0 * beta * phic)) * math.exp(0.5 * c2)
    psic_hat1 = _LookupPsihatH_scalar(zdt_za, dtL) * c1_c
    psic_hat2 = _LookupPsihatH_scalar(zdt_hc, dtL) * c1_c
    psic1 = _psic_scalar((za - disp) / obu)
    psic2 = _psic_scalar((hc - disp) / obu)
    psic = -psic1 + psic2 + psic_hat1 - psic_hat2

    return psim, psic, psim2, psim_hat2


# ===========================================================================
# Private: beta = u*/u(h)
# ===========================================================================


def _GetBeta(beta_neutral: float, LcL: float) -> float:
    """
    Calculate β = u*/u(h) for the current Obukhov length stability.

    Mirrors Fortran subroutine ``GetBeta`` (lines 245-290).

    Reference: Bonan et al. (2018), eqs. (A22)-(A24).

    **Unstable** (``LcL ≤ 0``): solves the quadratic equation for β²:

    .. code-block:: none

        β² = (-b + sqrt(b²-4ac)) / 2a
        where a=1, b=16*LcL*β_n^4, c=-β_n^4

    **Stable** (``LcL > 0``): solves the depressed cubic equation for β
    via Cardano's formula:

    .. code-block:: none

        5*LcL*β^3 + β - β_n = 0

    Error check: ``|β*φm(LcL*β²) - β_n| < 1e-6``.

    Args:
        beta_neutral: Neutral value of β = u*/u(h).
        LcL: Canopy density scale Lc divided by Obukhov length L.

    Returns:
        β value.
    """
    if LcL <= 0.0:  # Fortran lines 258-261: unstable quadratic
        aa = 1.0
        bb = 16.0 * LcL * beta_neutral**4
        cc = -(beta_neutral**4)
        beta = jnp.sqrt((-bb + jnp.sqrt(jnp.maximum(bb * bb - 4.0 * aa * cc, 0.0))) / (2.0 * aa))
    else:  # Fortran lines 263-272: stable cubic
        aa = 5.0 * LcL
        bb = 0.0
        cc = 1.0
        dd = -beta_neutral
        qq = (2.0 * bb**3 - 9.0 * aa * bb * cc + 27.0 * aa**2 * dd) ** 2 - 4.0 * (
            bb**2 - 3.0 * aa * cc
        ) ** 3
        qq = jnp.sqrt(jnp.maximum(qq, 0.0))
        rr = 0.5 * (qq + 2.0 * bb**3 - 9.0 * aa * bb * cc + 27.0 * aa**2 * dd)
        rr = rr ** (1.0 / 3.0)
        beta = -(bb + rr) / (3.0 * aa) - (bb**2 - 3.0 * aa * cc) / (3.0 * aa * rr)

    # Error check — Fortran lines 274-278
    y = LcL * beta**2
    fy = _phim_monin_obukhov(y)
    err = beta * fy - beta_neutral
    if abs(err) > 1.0e-6:
        endrun(msg=" ERROR: GetBeta: beta error")

    return beta


# ===========================================================================
# Private: Prandtl / Schmidt number
# ===========================================================================


def _GetPrSc(
    beta_neutral: float,
    beta_neutral_max: float,
    LcL: float,
) -> float:
    """
    Calculate the Prandtl (= Schmidt) number at the canopy top.

    Mirrors Fortran subroutine ``GetPrSc`` (lines 292-310).

    Reference: Bonan et al. (2018), eqs. (A25), (A34).

    .. code-block:: none

        PrSc = Pr0 + Pr1 * tanh(Pr2 * LcL)

    For sparse canopies (``sparse_canopy_type == 1``):

    .. code-block:: none

        PrSc = (1 - beta_n/beta_n_max) * 1.0
               + (beta_n/beta_n_max) * PrSc

    Args:
        beta_neutral: Neutral β.
        beta_neutral_max: Maximum allowed neutral β.
        LcL: Canopy density scale / Obukhov length.

    Returns:
        PrSc value.
    """
    PrSc = Pr0 + Pr1 * jnp.tanh(Pr2 * LcL)  # Fortran line 304

    if sparse_canopy_type == 1:  # Fortran lines 306-308
        f = beta_neutral / beta_neutral_max
        PrSc = (1.0 - f) * 1.0 + f * PrSc

    return PrSc


def _GetBeta_scalar(beta_neutral: float, LcL: float) -> float:
    """Scalar (math.*) version of _GetBeta for root-solver use."""
    if LcL <= 0.0:
        aa = 1.0
        bb = 16.0 * LcL * beta_neutral**4
        cc = -(beta_neutral**4)
        beta = math.sqrt((-bb + math.sqrt(max(bb * bb - 4.0 * aa * cc, 0.0))) / (2.0 * aa))
    else:
        aa = 5.0 * LcL
        bb = 0.0
        cc = 1.0
        dd = -beta_neutral
        qq = (2.0 * bb**3 - 9.0 * aa * bb * cc + 27.0 * aa**2 * dd) ** 2 - 4.0 * (
            bb**2 - 3.0 * aa * cc
        ) ** 3
        qq = math.sqrt(max(qq, 0.0))
        rr = 0.5 * (qq + 2.0 * bb**3 - 9.0 * aa * bb * cc + 27.0 * aa**2 * dd)
        rr = rr ** (1.0 / 3.0)
        beta = -(bb + rr) / (3.0 * aa) - (bb**2 - 3.0 * aa * cc) / (3.0 * aa * rr)

    y = LcL * beta**2
    fy = _phim_scalar(y)
    err = beta * fy - beta_neutral
    if abs(err) > 1.0e-6:
        endrun(msg=" ERROR: GetBeta: beta error")
    return beta


def _GetPrSc_scalar(
    beta_neutral: float,
    beta_neutral_max: float,
    LcL: float,
) -> float:
    """Scalar (math.*) version of _GetPrSc for root-solver use."""
    PrSc = Pr0 + Pr1 * math.tanh(Pr2 * LcL)
    if sparse_canopy_type == 1:
        f = beta_neutral / beta_neutral_max
        PrSc = (1.0 - f) * 1.0 + f * PrSc
    return PrSc


# ===========================================================================
# Private: RSL-modified psi functions
# ===========================================================================


def _GetPsiRSL(
    za: float,
    hc: float,
    disp: float,
    obu: float,
    beta: float,
    PrSc: float,
) -> Tuple[float, float, float, float]:
    """
    RSL-modified ψ stability functions for momentum and scalars.

    Mirrors Fortran subroutine ``GetPsiRSL`` (lines 312-398).

    Reference: Bonan et al. (2018), appendix A2.

    Computes the integrated stability corrections between ``hc`` and
    ``za`` including the roughness sublayer psihat modifications.

    **Momentum** (Bonan et al. 2018, eq. A16, A21, 19):

    .. code-block:: none

        phim = phim_monin_obukhov((hc-disp)/obu)
        c1   = (1 - vkc/(2*beta*phim)) * exp(0.5*c2)
        psim = -psim1 + psim2 + c1*psihat_m(za) - c1*psihat_m(hc) + vkc/beta

    **Scalars** (eqs. A19, A21, 20):

    .. code-block:: none

        phic = phic_monin_obukhov((hc-disp)/obu)
        c1   = (1 - PrSc*vkc/(2*beta*phic)) * exp(0.5*c2)
        psic = -psic1 + psic2 + c1*psihat_h(za) - c1*psihat_h(hc)

    Args:
        za: Atmospheric height (m).
        hc: Canopy top height (m).
        disp: Displacement height (m).
        obu: Obukhov length (m).
        beta: u*/u(h) at canopy top.
        PrSc: Prandtl/Schmidt number.

    Returns:
        Tuple ``(psim, psic, psim2, psim_hat2)`` where:
        - ``psim``: ψ for momentum including RSL (evaluated za→hc).
        - ``psic``: ψ for scalars including RSL.
        - ``psim2``: MO ψ for momentum evaluated at hc.
        - ``psim_hat2``: RSL psihat for momentum evaluated at hc.
    """
    dt = hc - disp  # Fortran line 358
    zdt_za = (za - hc) / dt  # normalised height at za
    zdt_hc = 0.0  # (hc - hc) / dt == 0 always
    dtL = dt / obu  # stability ratio

    # Momentum — Fortran lines 360-376
    phim = _phim_monin_obukhov((hc - disp) / obu)
    c1_m = (1.0 - vkc / (2.0 * beta * phim)) * jnp.exp(0.5 * c2)

    psim_hat1_raw = _LookupPsihatM(zdt_za, dtL)
    psim_hat2_raw = _LookupPsihatM(zdt_hc, dtL)
    psim_hat1 = psim_hat1_raw * c1_m
    psim_hat2 = psim_hat2_raw * c1_m

    psim1 = _psim_monin_obukhov((za - disp) / obu)
    psim2 = _psim_monin_obukhov((hc - disp) / obu)

    psim = -psim1 + psim2 + psim_hat1 - psim_hat2 + vkc / beta

    # Scalars — Fortran lines 378-393
    phic = _phic_monin_obukhov((hc - disp) / obu)
    c1_c = (1.0 - PrSc * vkc / (2.0 * beta * phic)) * jnp.exp(0.5 * c2)

    psic_hat1_raw = _LookupPsihatH(zdt_za, dtL)
    psic_hat2_raw = _LookupPsihatH(zdt_hc, dtL)
    psic_hat1 = psic_hat1_raw * c1_c
    psic_hat2 = psic_hat2_raw * c1_c

    psic1 = _psic_monin_obukhov((za - disp) / obu)
    psic2 = _psic_monin_obukhov((hc - disp) / obu)

    psic = -psic1 + psic2 + psic_hat1 - psic_hat2

    return psim, psic, psim2, psim_hat2


# ===========================================================================
# Private: ObuFunc callback (hybrid root-finder signature)
# ===========================================================================


def _ObuFunc(
    p: int,
    ic: int,
    il: int,
    mlcanopy_inst: mlcanopy_type,
    obu_val: float,
) -> Tuple[float, mlcanopy_type]:
    """
    Solve for the Obukhov length at a single patch.

    Mirrors Fortran subroutine ``ObuFunc`` (lines 178-244).

    Reference: Bonan et al. (2018), eqs. (A28)-(A33), (19)-(20).

    For a trial ``obu_val``:

    1. Clamp ``obu_val`` to avoid singularity near zero using
       ``LcL_min`` / ``LcL_max``.
    2. Compute neutral β and stability-corrected β via
       :func:`_GetBeta` (HF RSL and no-RSL versions), then blend.
    3. Compute displacement height ``zdisp`` (with optional sparse
       canopy correction).
    4. Compute PrSc via :func:`_GetPrSc`.
    5. Compute ψ functions via :func:`_GetPsiRSL`.
    6. Derive ``ustar``, ``tstar``, ``qstar``,
       ``gac_to_hc``, and the new Obukhov length.
    7. Return ``obu_dif = obu_new - obu_val``.

    Matches the :data:`MLMathToolsMod.FuncType` callback signature.

    Args:
        p: Patch index.
        ic: Unused (set to 0 by caller); required by callback contract.
        il: Unused (set to 0 by caller); required by callback contract.
        mlcanopy_inst: Canopy container; ``zdisp``, ``beta``, ``PrSc``,
            ``ustar``, ``gac_to_hc``, and ``obu`` are updated.
        obu_val: Trial Obukhov length (m).

    Returns:
        Tuple ``(obu_dif, mlcanopy_inst)``.
    """
    Lc_p = float(mlcanopy_inst.Lc_canopy[p])
    ztop_p = float(mlcanopy_inst.ztop_canopy[p])
    lai_p = float(mlcanopy_inst.lai_canopy[p])
    sai_p = float(mlcanopy_inst.sai_canopy[p])
    zref_p = float(mlcanopy_inst.zref_forcing[p])
    uref_p = float(mlcanopy_inst.uref_forcing[p])
    thref_p = float(mlcanopy_inst.thref_forcing[p])
    thvref_p = float(mlcanopy_inst.thvref_forcing[p])
    qref_p = float(mlcanopy_inst.qref_forcing[p])
    rhomol_p = float(mlcanopy_inst.rhomol_forcing[p])
    taf_p = float(mlcanopy_inst.taf_canopy[p])
    qaf_p = float(mlcanopy_inst.qaf_canopy[p])

    # Clamp obu_val away from zero — Fortran lines 195-202
    obu_min_stable = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    if obu_val >= 0.0:
        obu_cur = max(obu_val, obu_min_stable)
    else:
        obu_cur = min(obu_val, obu_max_unstable)
    LcL = Lc_p / obu_cur

    # Neutral beta — Fortran lines 204-206
    c1_n = (vkc / jnp.log((ztop_p + z0mg) / z0mg)) ** 2
    beta_neutral = min(jnp.sqrt(c1_n + cr * (lai_p + sai_p)), beta_neutral_max)

    # Stability-corrected beta (HF + no-RSL blend) — Fortran lines 208-215
    beta_HF = _GetBeta(beta_neutral, LcL)
    beta_norsl = _GetBeta(vkc / 2.0, LcL)
    if LcL > aH12[1]:  # Fortran: aH12(2) → 0-based index 1
        beta_val = beta_HF
    else:
        beta_val = beta_norsl + (beta_HF - beta_norsl) / (
            1.0 + aH12[0] * abs(LcL - aH12[1]) ** aH12[2]
        )

    # Displacement height — Fortran lines 217-224
    hc_minus_d = beta_val**2 * Lc_p
    if sparse_canopy_type == 1:
        hc_minus_d *= 1.0 - jnp.exp(-0.25 * (lai_p + sai_p) / beta_val**2)
    hc_minus_d = min(ztop_p, hc_minus_d)
    zdisp_val = ztop_p - hc_minus_d

    if (zref_p - zdisp_val) < 0.0:
        endrun(msg=" ERROR: ObuFunc: zdisp height > zref")

    # PrSc — Fortran line 228
    PrSc_val = _GetPrSc(beta_neutral, beta_neutral_max, LcL)

    # psi functions — Fortran lines 230-233
    psim, psic, _dum1, _dum2 = _GetPsiRSL(zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val)

    # Friction velocity, temperature and humidity scales — Fortran lines 235-243
    zlog = jnp.log((zref_p - zdisp_val) / (ztop_p - zdisp_val))
    ustar_val = uref_p * vkc / (zlog + psim)
    tstar = (thref_p - taf_p) * vkc / (zlog + psic)
    qstar = (qref_p - qaf_p) * vkc / (zlog + psic)
    gac_to_hc_v = rhomol_p * vkc * ustar_val / (zlog + psic)

    # New Obukhov length — Fortran lines 245-249
    tvstar = tstar + 0.61 * thref_p * qstar
    obu_new = ustar_val**2 * thvref_p / (vkc * grav * tvstar)
    obu_dif = obu_new - obu_val

    mlcanopy_inst = mlcanopy_inst._replace(
        zdisp_canopy=mlcanopy_inst.zdisp_canopy.at[p].set(zdisp_val),
        beta_canopy=mlcanopy_inst.beta_canopy.at[p].set(beta_val),
        PrSc_canopy=mlcanopy_inst.PrSc_canopy.at[p].set(PrSc_val),
        ustar_canopy=mlcanopy_inst.ustar_canopy.at[p].set(ustar_val),
        gac_to_hc_canopy=mlcanopy_inst.gac_to_hc_canopy.at[p].set(gac_to_hc_v),
        obu_canopy=mlcanopy_inst.obu_canopy.at[p].set(obu_cur),
    )
    return obu_dif, mlcanopy_inst


# ===========================================================================
# Private: ObuFuncPure — pure-scalar variant (no JAX reads/writes)
# ===========================================================================


def _ObuFuncPure(
    obu_val: float,
    *,
    Lc_p: float,
    ztop_p: float,
    lai_p: float,
    sai_p: float,
    zref_p: float,
    uref_p: float,
    thref_p: float,
    thvref_p: float,
    qref_p: float,
    rhomol_p: float,
    taf_p: float,
    qaf_p: float,
) -> float:
    """Pure-scalar version of _ObuFunc: no JAX reads/writes, returns obu_dif only.

    Uses math.* instead of jnp.* throughout — eliminates ~200+ JAX eager-dispatch
    calls per hybrid-solver iteration, giving ~50× speedup for _GetObu.
    """
    obu_min_stable = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    if obu_val >= 0.0:
        obu_cur = max(obu_val, obu_min_stable)
    else:
        obu_cur = min(obu_val, obu_max_unstable)
    LcL = Lc_p / obu_cur

    c1_n = (vkc / math.log((ztop_p + z0mg) / z0mg)) ** 2
    beta_neutral = min(math.sqrt(c1_n + cr * (lai_p + sai_p)), beta_neutral_max)

    beta_HF = _GetBeta_scalar(beta_neutral, LcL)
    beta_norsl = _GetBeta_scalar(vkc / 2.0, LcL)
    if LcL > _aH12_1:
        beta_val = beta_HF
    else:
        beta_val = beta_norsl + (beta_HF - beta_norsl) / (
            1.0 + _aH12_0 * abs(LcL - _aH12_1) ** _aH12_2
        )

    hc_minus_d = beta_val**2 * Lc_p
    if sparse_canopy_type == 1:
        hc_minus_d *= 1.0 - math.exp(-0.25 * (lai_p + sai_p) / beta_val**2)
    hc_minus_d = min(ztop_p, hc_minus_d)
    zdisp_val = ztop_p - hc_minus_d

    if (zref_p - zdisp_val) < 0.0:
        endrun(msg=" ERROR: ObuFunc: zdisp height > zref")

    PrSc_val = _GetPrSc_scalar(beta_neutral, beta_neutral_max, LcL)

    psim, psic, _dum1, _dum2 = _GetPsiRSL_scalar(
        zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val
    )

    zlog = math.log((zref_p - zdisp_val) / (ztop_p - zdisp_val))
    ustar_val = uref_p * vkc / (zlog + psim)
    tstar = (thref_p - taf_p) * vkc / (zlog + psic)
    qstar = (qref_p - qaf_p) * vkc / (zlog + psic)

    tvstar = tstar + 0.61 * thref_p * qstar
    obu_new = ustar_val**2 * thvref_p / (vkc * grav * tvstar)
    obu_dif = obu_new - obu_val
    return obu_dif


# ===========================================================================
# Private: JAX-traceable β solver (Task E)
# ===========================================================================


def _GetBeta_jax(beta_neutral, LcL):
    """
    JAX-traceable β = u*/u(h) solver.

    Mirrors :func:`_GetBeta` but replaces Python ``if/else`` on ``LcL``
    with ``jnp.where`` so the function is ``jax.jit``- and
    ``jax.grad``-compatible.  The error-check ``endrun`` is omitted since
    it uses Python ``if`` on a traced value.

    Both branches are always evaluated; the stable-branch denominator
    ``aa_s`` is guarded to ``1.0`` when ``LcL <= 0`` to prevent division
    by zero in the branch that will be discarded by ``jnp.where``.

    Args:
        beta_neutral: Neutral β — JAX scalar.
        LcL: Canopy density scale / Obukhov length — JAX scalar.

    Returns:
        β — JAX scalar.
    """
    # --- Unstable branch (LcL <= 0): solve β² = (-b + sqrt(b²-4ac)) / 2a ---
    bb_u = 16.0 * LcL * beta_neutral**4
    cc_u = -(beta_neutral**4)
    disc_u = jnp.maximum(bb_u * bb_u - 4.0 * cc_u, 1.0e-30)  # a=1
    beta_u = jnp.sqrt(jnp.maximum((-bb_u + jnp.sqrt(disc_u)) / 2.0, 1.0e-30))

    # --- Stable branch (LcL > 0): Cardano formula for 5*LcL*β³ + β = β_n ---
    # Guard aa_s != 0 when LcL==0 so no NaN before jnp.where selects branch.
    # jnp.maximum avoids select op → prevents XLA select_divide_fusion bug.
    # LcL ≥ 0 in the stable branch; when LcL < 0 the branch is discarded anyway.
    aa_s = jnp.maximum(5.0 * LcL, 1.0e-10)
    # bb_s == 0, cc_s == 1, dd_s == -beta_neutral
    dd_s = -beta_neutral
    qq_s = jnp.sqrt(
        jnp.maximum(
            (27.0 * aa_s**2 * dd_s) ** 2 + 108.0 * aa_s**3,
            1.0e-30,
        )
    )
    rr_s = jnp.maximum(
        0.5 * (qq_s + 27.0 * aa_s**2 * dd_s),
        1e-30,
    ) ** (1.0 / 3.0)
    beta_s = -rr_s / (3.0 * aa_s) + 1.0 / rr_s  # simplifies with bb=0,cc=1

    return jnp.where(LcL <= 0.0, beta_u, beta_s)


# ===========================================================================
# Private: JAX-traceable ObuFunc residual (Task E)
# ===========================================================================


def _ObuFuncPure_jax(
    obu_val,
    *,
    Lc_p: float,
    ztop_p: float,
    lai_p: float,
    sai_p: float,
    zref_p: float,
    uref_p: float,
    thref_p: float,
    thvref_p: float,
    qref_p: float,
    rhomol_p: float,
    taf_p: float,
    qaf_p: float,
):
    """
    JAX-traceable Obukhov-length residual.

    Mirrors :func:`_ObuFuncPure` but uses ``jnp.*`` throughout and
    replaces the Python ``if obu_val >= 0.0`` branch with ``jnp.where``.
    Calls :func:`_GetBeta_jax` (JAX-traceable β), :func:`_GetPrSc`
    (already uses ``jnp.tanh``), and :func:`_GetPsiRSL` (already uses
    ``jnp.searchsorted`` / ``jnp.exp``).

    The ``zref_p - zdisp_val < 0`` error check is dropped: it uses
    Python ``if`` on a traced value and would break JIT.

    Args:
        obu_val: Trial Obukhov length (m) — JAX scalar.
        Remaining keyword args: patch-level constants (Python floats).

    Returns:
        Residual ``obu_new - obu_val`` (m) — JAX scalar.
    """
    obu_min_stable = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    obu_cur = jnp.where(
        obu_val >= 0.0,
        jnp.maximum(obu_val, obu_min_stable),
        jnp.minimum(obu_val, obu_max_unstable),
    )
    _eps_obu = jnp.asarray(1e-30)
    # Use sign * max(|obu|, eps) to avoid select-as-denominator pattern
    # which triggers XLA select_divide_fusion compilation failure.
    _obu_abs_safe = jnp.maximum(jnp.abs(obu_cur), _eps_obu)
    _obu_sign = jnp.where(obu_cur < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    LcL_val = Lc_p * _obu_sign / _obu_abs_safe

    z0mg_safe = jnp.maximum(z0mg, _eps_obu)
    c1_n = (vkc / jnp.log((ztop_p + z0mg_safe) / z0mg_safe)) ** 2
    beta_neutral = jnp.minimum(
        jnp.sqrt(c1_n + cr * (lai_p + sai_p)),
        jnp.asarray(beta_neutral_max),
    )

    beta_HF = _GetBeta_jax(beta_neutral, LcL_val)
    beta_norsl = _GetBeta_jax(jnp.asarray(vkc / 2.0), LcL_val)
    beta_val = jnp.where(
        LcL_val > _aH12_1,
        beta_HF,
        (
            beta_norsl
            + (beta_HF - beta_norsl) / (1.0 + _aH12_0 * jnp.abs(LcL_val - _aH12_1) ** _aH12_2)
        ),
    )

    hc_minus_d = beta_val**2 * Lc_p
    if sparse_canopy_type == 1:  # static Python branch — baked in at trace time
        hc_minus_d = hc_minus_d * (1.0 - jnp.exp(-0.25 * (lai_p + sai_p) / beta_val**2))
    hc_minus_d = jnp.minimum(hc_minus_d, jnp.asarray(ztop_p))
    zdisp_val = ztop_p - hc_minus_d

    PrSc_val = _GetPrSc(beta_neutral, beta_neutral_max, LcL_val)

    psim, psic, _dum1, _dum2 = _GetPsiRSL(zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val)

    _hc_d = jnp.maximum(ztop_p - zdisp_val, _eps_obu)
    zlog = jnp.log(jnp.maximum((zref_p - zdisp_val) / _hc_d, _eps_obu))
    # sign * max(|denom|, eps) avoids select-as-denominator → no select_divide_fusion
    _dm_abs = jnp.maximum(jnp.abs(zlog + psim), _eps_obu)
    _dm_sign = jnp.where((zlog + psim) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    _dc_abs = jnp.maximum(jnp.abs(zlog + psic), _eps_obu)
    _dc_sign = jnp.where((zlog + psic) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    ustar_val = uref_p * vkc * _dm_sign / _dm_abs
    tstar = (thref_p - taf_p) * vkc * _dc_sign / _dc_abs
    qstar = (qref_p - qaf_p) * vkc * _dc_sign / _dc_abs

    tvstar = tstar + 0.61 * thref_p * qstar
    # Use sign * max(|tvstar|, eps) to avoid select-as-denominator pattern
    # which triggers XLA select_divide_fusion compilation failure.
    _tvstar_abs_safe = jnp.maximum(jnp.abs(tvstar), _eps_obu)
    _tvstar_sign = jnp.where(tvstar < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    obu_new = ustar_val**2 * thvref_p * _tvstar_sign / (vkc * grav * _tvstar_abs_safe)
    return obu_new - obu_val


# ===========================================================================
# Private: fixed-iteration Obukhov solver (Task E)
# ===========================================================================


def _obu_body_pure(i, carry):
    """Pure fori_loop body for the Obukhov secant solver.

    Defined at module level (not as a closure) so ``jax.lax.fori_loop``
    traces it once and reuses the compiled XLA program for every patch and
    every sub-step.  All patch-level constants are threaded through ``carry``
    as JAX arrays rather than closed-over Python scalars.

    Carry layout: (x0, x1, f0, f1,
                   Lc_p, ztop_p, lai_p, sai_p,
                   zref_p, uref_p, thref_p, thvref_p,
                   qref_p, rhomol_p, taf_p, qaf_p)
    """
    (
        x0,
        x1,
        f0,
        f1,
        Lc_p,
        ztop_p,
        lai_p,
        sai_p,
        zref_p,
        uref_p,
        thref_p,
        thvref_p,
        qref_p,
        rhomol_p,
        taf_p,
        qaf_p,
    ) = carry
    # sign * max(|df|, eps) avoids select-as-denominator → no select_divide_fusion
    _df = f1 - f0
    _df_abs_safe = jnp.maximum(jnp.abs(_df), jnp.asarray(1e-30))
    _df_sign = jnp.where(_df < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    dx = -f1 * (x1 - x0) * _df_sign / _df_abs_safe
    x_new = x1 + dx
    f_new = _ObuFuncPure_jax(
        x_new,
        Lc_p=Lc_p,
        ztop_p=ztop_p,
        lai_p=lai_p,
        sai_p=sai_p,
        zref_p=zref_p,
        uref_p=uref_p,
        thref_p=thref_p,
        thvref_p=thvref_p,
        qref_p=qref_p,
        rhomol_p=rhomol_p,
        taf_p=taf_p,
        qaf_p=qaf_p,
    )
    return (
        x1,
        x_new,
        f1,
        f_new,
        Lc_p,
        ztop_p,
        lai_p,
        sai_p,
        zref_p,
        uref_p,
        thref_p,
        thvref_p,
        qref_p,
        rhomol_p,
        taf_p,
        qaf_p,
    )


def _obu_fixed_iter(init_obu, kwargs, n_iter: int = 25):
    """
    Fixed-iteration Obukhov-length solver via ``jax.lax.fori_loop``.

    Replaces :func:`MLMathToolsMod.hybrid_scalar` (Python ``while``-loop)
    with a ``jax.lax.fori_loop`` of exactly ``n_iter`` secant steps so the
    solver is ``jax.jit``-compatible.

    The secant update is::

        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

    ``n_iter=25`` is conservative; Bonan et al. (2018) report typical
    convergence in < 10 iterations for moderate stability regimes.

    The body function is defined at module level (:func:`_obu_body_pure`) so
    ``jax.lax.fori_loop`` traces it **once** regardless of how many patches
    or sub-steps are run.  Patch constants are passed through the carry tuple
    as JAX scalars rather than being closed over as Python floats, which
    previously caused one XLA ``scan`` recompilation per patch per sub-step.

    Args:
        init_obu: Initial Obukhov length (m) — JAX scalar.
        kwargs:   Dict of patch-level constants passed to
                  :func:`_ObuFuncPure_jax`.  Values may be Python floats or
                  JAX scalars; they are converted to JAX arrays here.
        n_iter:   Number of fixed iterations (default 25).

    Returns:
        Converged Obukhov length estimate (m) — JAX scalar.
    """
    # Convert all kwargs to JAX scalars so they have stable abstract types
    # and are passed through the carry rather than baked into the trace.
    Lc_p = jnp.asarray(kwargs["Lc_p"])
    ztop_p = jnp.asarray(kwargs["ztop_p"])
    lai_p = jnp.asarray(kwargs["lai_p"])
    sai_p = jnp.asarray(kwargs["sai_p"])
    zref_p = jnp.asarray(kwargs["zref_p"])
    uref_p = jnp.asarray(kwargs["uref_p"])
    thref_p = jnp.asarray(kwargs["thref_p"])
    thvref_p = jnp.asarray(kwargs["thvref_p"])
    qref_p = jnp.asarray(kwargs["qref_p"])
    rhomol_p = jnp.asarray(kwargs["rhomol_p"])
    taf_p = jnp.asarray(kwargs["taf_p"])
    qaf_p = jnp.asarray(kwargs["qaf_p"])

    obu0 = init_obu
    obu1 = jnp.asarray(-100.0)

    f0 = _ObuFuncPure_jax(
        obu0,
        Lc_p=Lc_p,
        ztop_p=ztop_p,
        lai_p=lai_p,
        sai_p=sai_p,
        zref_p=zref_p,
        uref_p=uref_p,
        thref_p=thref_p,
        thvref_p=thvref_p,
        qref_p=qref_p,
        rhomol_p=rhomol_p,
        taf_p=taf_p,
        qaf_p=qaf_p,
    )
    f1 = _ObuFuncPure_jax(
        obu1,
        Lc_p=Lc_p,
        ztop_p=ztop_p,
        lai_p=lai_p,
        sai_p=sai_p,
        zref_p=zref_p,
        uref_p=uref_p,
        thref_p=thref_p,
        thvref_p=thvref_p,
        qref_p=qref_p,
        rhomol_p=rhomol_p,
        taf_p=taf_p,
        qaf_p=qaf_p,
    )

    init_carry = (
        obu0,
        obu1,
        f0,
        f1,
        Lc_p,
        ztop_p,
        lai_p,
        sai_p,
        zref_p,
        uref_p,
        thref_p,
        thvref_p,
        qref_p,
        rhomol_p,
        taf_p,
        qaf_p,
    )

    final_carry = jax.lax.fori_loop(0, n_iter, _obu_body_pure, init_carry)
    obu_final = final_carry[1]  # x1 from final iteration
    return obu_final


# ===========================================================================
# Private: GetObu driver
# ===========================================================================


def _GetObu(p: int, mlcanopy_inst: mlcanopy_type, diff_mode: bool = False) -> mlcanopy_type:
    """
    Solve for the Obukhov length using the hybrid root solver.

    Mirrors Fortran subroutine ``GetObu`` (lines 160-178).

    When ``diff_mode=True``, keeps values as JAX arrays (no ``float()``
    extraction) and writes back via ``_ObuFuncPure_jax`` side-products.

    Args:
        p: Patch index.
        mlcanopy_inst: Canopy container; Obukhov length and dependent
            quantities are updated via :func:`_ObuFunc`.
        diff_mode: If True, use JAX-traceable path (no float extraction).

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    if not diff_mode:
        # Original fast path — pre-extract scalars for speed
        _kwargs = dict(
            Lc_p=float(mlcanopy_inst.Lc_canopy[p]),
            ztop_p=float(mlcanopy_inst.ztop_canopy[p]),
            lai_p=float(mlcanopy_inst.lai_canopy[p]),
            sai_p=float(mlcanopy_inst.sai_canopy[p]),
            zref_p=float(mlcanopy_inst.zref_forcing[p]),
            uref_p=float(mlcanopy_inst.uref_forcing[p]),
            thref_p=float(mlcanopy_inst.thref_forcing[p]),
            thvref_p=float(mlcanopy_inst.thvref_forcing[p]),
            qref_p=float(mlcanopy_inst.qref_forcing[p]),
            rhomol_p=float(mlcanopy_inst.rhomol_forcing[p]),
            taf_p=float(mlcanopy_inst.taf_canopy[p]),
            qaf_p=float(mlcanopy_inst.qaf_canopy[p]),
        )
        obu_converged = float(_obu_fixed_iter(jnp.asarray(100.0), _kwargs, n_iter=25))
        # Use the JAX-traceable writeback instead of _ObuFunc to avoid
        # re-extracting the 12 scalar values already in _kwargs (saves
        # ~12 D→H syncs per sub-step in non-diff mode).
        mlcanopy_inst = _obu_writeback_jax(p, obu_converged, _kwargs, mlcanopy_inst)
        return mlcanopy_inst

    # --- Differentiable path: keep everything as JAX arrays ---
    _kwargs = dict(
        Lc_p=mlcanopy_inst.Lc_canopy[p],
        ztop_p=mlcanopy_inst.ztop_canopy[p],
        lai_p=mlcanopy_inst.lai_canopy[p],
        sai_p=mlcanopy_inst.sai_canopy[p],
        zref_p=mlcanopy_inst.zref_forcing[p],
        uref_p=mlcanopy_inst.uref_forcing[p],
        thref_p=mlcanopy_inst.thref_forcing[p],
        thvref_p=mlcanopy_inst.thvref_forcing[p],
        qref_p=mlcanopy_inst.qref_forcing[p],
        rhomol_p=mlcanopy_inst.rhomol_forcing[p],
        taf_p=mlcanopy_inst.taf_canopy[p],
        qaf_p=mlcanopy_inst.qaf_canopy[p],
    )

    # IFT-based gradient for the Obukhov-length fixed-point solver.
    #
    # Differentiating through the 25-iteration secant solver (_obu_fixed_iter)
    # gives exploding gradients: each secant step's Jacobian accumulates over
    # 25 iterations, producing |J|^25 amplification (observed: ~1e+144).
    #
    # Fix (same pattern as _bisect_gs_ift):
    #   obu_ift = obu0 - F(obu0; theta) / stop_grad(dF/dobu)
    # where F(obu) = _ObuFuncPure_jax(obu, theta) = obu_new(obu, theta) - obu.
    #
    # Forward: F(obu0) ≈ 0  ⟹  obu_ift ≈ obu0 = obu*  ✓
    # Backward: d(obu_ift)/dtheta = -∂F/∂theta / stop_grad(dF/dobu) = IFT  ✓
    #
    # dF/dobu = dobu_new/dobu - 1 < 0 for a converging fixed-point.

    # Forward solve (stop gradient through iterations)
    obu_raw = _obu_fixed_iter(jnp.asarray(100.0), _kwargs, n_iter=25)
    obu0 = jax.lax.stop_gradient(obu_raw)

    # Residual at obu0 — gradient flows through theta (thref_p, thvref_p, ...)
    f0 = _ObuFuncPure_jax(obu0, **_kwargs)

    # dF/dobu via central FD — stop_gradient (denominator only)
    _delta_obu = jnp.asarray(1.0, dtype=obu0.dtype)  # 1 m step (obu is O(100-10000 m))
    df_dobu = jax.lax.stop_gradient(
        (
            _ObuFuncPure_jax(obu0 + _delta_obu, **_kwargs)
            - _ObuFuncPure_jax(obu0 - _delta_obu, **_kwargs)
        )
        / (2.0 * _delta_obu)
    )

    # Gate Newton step on |df_dobu| > eps (IFT well-conditioned)
    _eps_ift = jnp.asarray(1e-6)
    apply = jnp.abs(df_dobu) > _eps_ift
    safe_f0 = jnp.where(apply, f0, jnp.zeros_like(f0))
    safe_denom = jnp.where(apply, df_dobu, -jnp.ones_like(df_dobu))
    # Default safe_denom = -1.0: when apply=False, safe_f0/safe_denom = 0 → obu_ift = obu0 ✓
    obu_ift = obu0 - safe_f0 / safe_denom

    # Recompute all derived quantities from the IFT-corrected obu (JAX-traceable)
    mlcanopy_inst = _obu_writeback_jax(p, obu_ift, _kwargs, mlcanopy_inst)
    return mlcanopy_inst


def _obu_writeback_jax(p, obu_val, kwargs, mlcanopy_inst):
    """JAX-traceable write-back of Obukhov-derived quantities.

    Recomputes beta, zdisp, PrSc, ustar, gac_to_hc from ``obu_val``
    using the same physics as ``_ObuFuncPure_jax``, then writes them
    back into ``mlcanopy_inst``.
    """
    Lc_p = kwargs["Lc_p"]
    ztop_p = kwargs["ztop_p"]
    lai_p = kwargs["lai_p"]
    sai_p = kwargs["sai_p"]
    zref_p = kwargs["zref_p"]
    uref_p = kwargs["uref_p"]
    rhomol_p = kwargs["rhomol_p"]

    obu_min_stable = Lc_p / LcL_max
    obu_max_unstable = Lc_p / LcL_min
    obu_cur = jnp.where(
        obu_val >= 0.0,
        jnp.maximum(obu_val, obu_min_stable),
        jnp.minimum(obu_val, obu_max_unstable),
    )
    _eps_obu = jnp.asarray(1e-30)
    # Use sign * max(|obu|, eps) to avoid select-as-denominator pattern
    # which triggers XLA select_divide_fusion compilation failure.
    _obu_abs_safe = jnp.maximum(jnp.abs(obu_cur), _eps_obu)
    _obu_sign = jnp.where(obu_cur < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    LcL_val = Lc_p * _obu_sign / _obu_abs_safe

    z0mg_safe = jnp.maximum(z0mg, _eps_obu)
    c1_n = (vkc / jnp.log((ztop_p + z0mg_safe) / z0mg_safe)) ** 2
    beta_neutral = jnp.minimum(
        jnp.sqrt(c1_n + cr * (lai_p + sai_p)),
        jnp.asarray(beta_neutral_max),
    )

    beta_HF = _GetBeta_jax(beta_neutral, LcL_val)
    beta_norsl = _GetBeta_jax(jnp.asarray(vkc / 2.0), LcL_val)
    beta_val = jnp.where(
        LcL_val > _aH12_1,
        beta_HF,
        (
            beta_norsl
            + (beta_HF - beta_norsl) / (1.0 + _aH12_0 * jnp.abs(LcL_val - _aH12_1) ** _aH12_2)
        ),
    )

    hc_minus_d = beta_val**2 * Lc_p
    if sparse_canopy_type == 1:
        hc_minus_d = hc_minus_d * (1.0 - jnp.exp(-0.25 * (lai_p + sai_p) / beta_val**2))
    hc_minus_d = jnp.minimum(hc_minus_d, ztop_p)
    zdisp_val = ztop_p - hc_minus_d

    PrSc_val = _GetPrSc(beta_neutral, beta_neutral_max, LcL_val)

    psim, psic, _, _ = _GetPsiRSL(zref_p, ztop_p, zdisp_val, obu_cur, beta_val, PrSc_val)

    _hc_d2 = jnp.maximum(ztop_p - zdisp_val, _eps_obu)
    zlog = jnp.log(jnp.maximum((zref_p - zdisp_val) / _hc_d2, _eps_obu))
    # Use sign * max(|denom|, eps) — avoids jnp.where-as-denominator NaN grads
    # (jnp.where differentiates both branches; 0/0 = NaN even when masked).
    _dm2_abs = jnp.maximum(jnp.abs(zlog + psim), _eps_obu)
    _dm2_sign = jnp.where((zlog + psim) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    _dc2_abs = jnp.maximum(jnp.abs(zlog + psic), _eps_obu)
    _dc2_sign = jnp.where((zlog + psic) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    ustar_val = uref_p * vkc * _dm2_sign / _dm2_abs
    gac_to_hc_v = rhomol_p * vkc * ustar_val * _dc2_sign / _dc2_abs

    return mlcanopy_inst._replace(
        zdisp_canopy=mlcanopy_inst.zdisp_canopy.at[p].set(zdisp_val),
        beta_canopy=mlcanopy_inst.beta_canopy.at[p].set(beta_val),
        PrSc_canopy=mlcanopy_inst.PrSc_canopy.at[p].set(PrSc_val),
        ustar_canopy=mlcanopy_inst.ustar_canopy.at[p].set(ustar_val),
        gac_to_hc_canopy=mlcanopy_inst.gac_to_hc_canopy.at[p].set(gac_to_hc_v),
        obu_canopy=mlcanopy_inst.obu_canopy.at[p].set(obu_cur),
    )


# ===========================================================================
# Private: RoughnessLength
# ===========================================================================


def _RoughnessLength(p: int, mlcanopy_inst: mlcanopy_type) -> mlcanopy_type:
    """
    Calculate the roughness length for momentum via inline bisection.

    Mirrors Fortran subroutine ``RoughnessLength`` (lines 454-520).

    Uses the self-consistent definition:

    .. code-block:: none

        z0m = (hc - d) * exp(-vkc/beta) * exp(-psim_hc + psim(z0m)) * exp(psim_hat_hc)

    where ``psim_hc`` and ``psim_hat_hc`` come from
    :func:`_GetPsiRSL` evaluated at the canopy top.
    Bisection between ``aval = ztop`` and ``bval = 0`` to tolerance
    0.001 m; maximum 20 iterations.

    Args:
        p: Patch index.
        mlcanopy_inst: Canopy container; ``z0m_canopy[p]`` is updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    ztop_p = float(mlcanopy_inst.ztop_canopy[p])
    zdisp_p = float(mlcanopy_inst.zdisp_canopy[p])
    obu_p = float(mlcanopy_inst.obu_canopy[p])
    beta_p = float(mlcanopy_inst.beta_canopy[p])
    PrSc_p = float(mlcanopy_inst.PrSc_canopy[p])
    zref_p = float(mlcanopy_inst.zref_forcing[p])

    # psi and psihat at canopy top — Fortran lines 475-476
    # Use scalar version: no JAX dispatch overhead in the bisection loop below
    _, _, psim_hc, psim_hat_hc = _GetPsiRSL_scalar(zref_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psim_hc = float(psim_hc)
    psim_hat_hc = float(psim_hat_hc)

    hc_minus_d = ztop_p - zdisp_p
    exp1 = math.exp(-vkc / beta_p)
    exp2 = math.exp(psim_hat_hc)

    def _z0m_func(trial: float) -> float:
        """Implicit equation: z0m_formula(trial) - trial.  Uses math.* (no JAX)."""
        psim_t = _psim_scalar(trial / obu_p)
        return hc_minus_d * exp1 * math.exp(-psim_hc + psim_t) * exp2 - trial

    aval: float = ztop_p
    bval: float = 0.0
    err: float = 0.001
    nmax: int = 20

    fa = float(_z0m_func(aval))
    fb = float(_z0m_func(bval))

    if fa * fb > 0.0:
        endrun(
            msg=" ERROR: RoughnessLength: bisection error - f(a) and f(b) do not have opposite signs"
        )

    cval: float = 0.0
    n: int = 1
    while abs(bval - aval) > err and n <= nmax:
        cval = (aval + bval) / 2.0
        fc = float(_z0m_func(cval))
        if fa * fc < 0.0:
            bval = cval
            fb = fc
        else:
            aval = cval
            fa = fc
        n += 1

    if n > nmax:
        endrun(msg=" ERROR: RoughnessLength: maximum iteration exceeded")

    return mlcanopy_inst._replace(z0m_canopy=mlcanopy_inst.z0m_canopy.at[p].set(cval))


def _RoughnessLength_jax(p: int, mlcanopy_inst: mlcanopy_type) -> mlcanopy_type:
    """JAX-traceable roughness length solver via ``jax.lax.fori_loop`` bisection.

    Replaces the Python ``while``-loop in :func:`_RoughnessLength` with a
    fixed 20-iteration ``fori_loop`` bisection so the function is
    ``jax.jit``- and ``jax.grad``-compatible.  Error checks are omitted.
    """
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    zdisp_p = mlcanopy_inst.zdisp_canopy[p]
    obu_p = mlcanopy_inst.obu_canopy[p]
    beta_p = mlcanopy_inst.beta_canopy[p]
    PrSc_p = mlcanopy_inst.PrSc_canopy[p]
    zref_p = mlcanopy_inst.zref_forcing[p]

    # psi and psihat at canopy top (JAX version)
    _, _, psim_hc, psim_hat_hc = _GetPsiRSL(zref_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)

    hc_minus_d = ztop_p - zdisp_p
    exp1 = jnp.exp(-vkc / beta_p)
    exp2 = jnp.exp(psim_hat_hc)

    def _z0m_func_jax(trial):
        psim_t = _psim_monin_obukhov(trial / obu_p)
        return hc_minus_d * exp1 * jnp.exp(-psim_hc + psim_t) * exp2 - trial

    aval = ztop_p
    bval = jnp.asarray(0.0)
    fa = _z0m_func_jax(aval)
    fb = _z0m_func_jax(bval)

    def bisect_body(i, carry):
        a, b, fa, fb = carry
        c = (a + b) / 2.0
        fc = _z0m_func_jax(c)
        # If fa*fc < 0 → root in [a,c], else root in [c,b]
        go_left = fa * fc < 0.0
        a_new = jnp.where(go_left, a, c)
        b_new = jnp.where(go_left, c, b)
        fa_new = jnp.where(go_left, fa, fc)
        fb_new = jnp.where(go_left, fc, fb)
        return (a_new, b_new, fa_new, fb_new)

    a_f, b_f, _, _ = jax.lax.fori_loop(0, 20, bisect_body, (aval, bval, fa, fb))
    z0m_val = (a_f + b_f) / 2.0

    return mlcanopy_inst._replace(z0m_canopy=mlcanopy_inst.z0m_canopy.at[p].set(z0m_val))


# ===========================================================================
# Private: WindProfile
# ===========================================================================


def _WindProfile(
    p: int,
    lm_over_beta: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Wind speed profile above and within the canopy.

    Mirrors Fortran subroutine ``WindProfile`` (lines 522-565).

    Reference: Bonan et al. (2018), eqs. (19) and (21).

    **Above canopy** (ic > ntop, at height zs):

    .. code-block:: none

        wind(ic) = ustar/vkc * (log((zs-d)/(ztop-d)) + psim)

    **At canopy top**:

    .. code-block:: none

        uaf = ustar / beta

    **Within canopy** (ic ≤ ntop, at height zs):

    .. code-block:: none

        wind(ic) = uaf * exp((zs - ztop) / (lm/beta))

    Args:
        p: Patch index.
        lm_over_beta: lm/β turbulence length scale (m).
        mlcanopy_inst: Canopy container; ``uaf_canopy`` and
            ``wind_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    _ntop = int(mlcanopy_inst.ntop_canopy[p])
    _ncan = int(mlcanopy_inst.ncan_canopy[p])
    ztop_p = float(mlcanopy_inst.ztop_canopy[p])
    zdisp_p = float(mlcanopy_inst.zdisp_canopy[p])
    obu_p = float(mlcanopy_inst.obu_canopy[p])
    beta_p = float(mlcanopy_inst.beta_canopy[p])
    PrSc_p = float(mlcanopy_inst.PrSc_canopy[p])
    ustar_p = float(mlcanopy_inst.ustar_canopy[p])

    wind = mlcanopy_inst.wind_profile
    uaf = mlcanopy_inst.uaf_canopy

    # Pre-extract zs_profile row to avoid per-layer JAX syncs
    _zs_p = np.asarray(mlcanopy_inst.zs_profile[p])

    # Accumulate wind values in numpy array; batch write-back at end
    _wind_new = np.zeros(_ncan + 2)

    # Above-canopy wind — Fortran lines 544-547
    # Use scalar _GetPsiRSL_scalar + math.log to avoid JAX dispatch per layer
    for ic in range(_ntop + 1, _ncan + 1):
        zs_ic = float(_zs_p[ic])
        psim, _, _, _ = _GetPsiRSL_scalar(zs_ic, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        zlog_m = math.log((zs_ic - zdisp_p) / (ztop_p - zdisp_p))
        _wind_new[ic] = ustar_p / vkc * (zlog_m + psim)

    # Wind at canopy top — Fortran line 549
    uaf_val = ustar_p / beta_p
    uaf = uaf.at[p].set(uaf_val)

    # Within-canopy wind — Fortran lines 551-553
    for ic in range(1, _ntop + 1):
        zs_ic = float(_zs_p[ic])
        _wind_new[ic] = uaf_val * math.exp((zs_ic - ztop_p) / lm_over_beta)

    # Batch write-back
    _sl = slice(1, _ncan + 1)
    wind = wind.at[p, _sl].set(jnp.array(_wind_new[_sl]))

    return mlcanopy_inst._replace(
        uaf_canopy=uaf,
        wind_profile=wind,
    )


def _WindProfile_jax(
    p: int,
    ntop: int,
    ncan: int,
    lm_over_beta,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """JAX-traceable wind speed profile using ``_GetPsiRSL`` and ``jnp.*``.

    Loop bounds ``ntop``, ``ncan`` are concrete Python ints from GridInfo.
    All intermediate values stay as JAX arrays.
    """
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    zdisp_p = mlcanopy_inst.zdisp_canopy[p]
    obu_p = mlcanopy_inst.obu_canopy[p]
    beta_p = mlcanopy_inst.beta_canopy[p]
    PrSc_p = mlcanopy_inst.PrSc_canopy[p]
    ustar_p = mlcanopy_inst.ustar_canopy[p]

    wind = mlcanopy_inst.wind_profile
    uaf = mlcanopy_inst.uaf_canopy
    zs_p = mlcanopy_inst.zs_profile[p]

    # Above-canopy wind
    for ic in range(ntop + 1, ncan + 1):
        zs_ic = zs_p[ic]
        psim, _, _, _ = _GetPsiRSL(zs_ic, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        _hcd_w = jnp.maximum(ztop_p - zdisp_p, 1e-30)
        zlog_m = jnp.log(jnp.maximum((zs_ic - zdisp_p) / _hcd_w, 1e-30))
        wind = wind.at[p, ic].set(ustar_p / vkc * (zlog_m + psim))

    # Wind at canopy top
    uaf_val = ustar_p / jnp.maximum(beta_p, 1e-30)
    uaf = uaf.at[p].set(uaf_val)

    # Within-canopy wind
    for ic in range(1, ntop + 1):
        zs_ic = zs_p[ic]
        wind = wind.at[p, ic].set(uaf_val * jnp.exp((zs_ic - ztop_p) / lm_over_beta))

    return mlcanopy_inst._replace(
        uaf_canopy=uaf,
        wind_profile=wind,
    )


# ===========================================================================
# Private: AerodynamicConductance
# ===========================================================================


def _AerodynamicConductance(
    p: int,
    lm_over_beta: float,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Aerodynamic conductances above and within the canopy.

    Mirrors Fortran subroutine ``AerodynamicConductance`` (lines 567-680).

    Reference: Bonan et al. (2018), eqs. (24), (26), (27).

    **Above canopy** (Fortran lines 591-620, eq. 24):

    .. code-block:: none

        gac(ic) = rhomol * vkc * ustar / (log((zs_{i+1}-d)/(zs_i-d)) + psic)

    Special cases: top layer uses ``zref`` as upper boundary; top foliage
    layer uses ``ztop`` as lower boundary and ``zs(ntop+1)`` as upper.
    A consistency check against ``gac_to_hc`` is applied.

    **Within canopy** (Fortran lines 622-641, eq. 26):

    .. code-block:: none

        res  = PrSc / (beta*ustar) * (exp(-zl/lm_beta) - exp(-zu/lm_beta))
        gac  = rhomol / res

    Top foliage layer combines within and above contributions in series.

    **Soil surface conductance** (Fortran lines 643-672):
    Selected by ``HF_extension_type``:

    - **1**: exponential profile extended to ``z0cg = 0.1*z0mg``.
    - **2**: logarithmic profile to ``z0mg`` with wind floor of 0.1 m/s.

    All conductances are capped so that the implied resistance does not
    exceed ``ra_max`` s/m.

    **Eddy diffusivity** (Fortran lines 674-678):

    .. code-block:: none

        kc_eddy(ic) = gac(ic) / rhomol * (zs_{ic+1} - zs_ic)

    Args:
        p: Patch index.
        lm_over_beta: lm/β turbulence length scale (m).
        mlcanopy_inst: Canopy container; ``gac0_soil``, ``gac_profile``,
            and ``kc_eddy_profile`` are updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    _ntop = int(mlcanopy_inst.ntop_canopy[p])
    _ncan = int(mlcanopy_inst.ncan_canopy[p])
    ztop_p = float(mlcanopy_inst.ztop_canopy[p])
    zdisp_p = float(mlcanopy_inst.zdisp_canopy[p])
    obu_p = float(mlcanopy_inst.obu_canopy[p])
    beta_p = float(mlcanopy_inst.beta_canopy[p])
    PrSc_p = float(mlcanopy_inst.PrSc_canopy[p])
    ustar_p = float(mlcanopy_inst.ustar_canopy[p])
    rhomol_p = float(mlcanopy_inst.rhomol_forcing[p])
    zref_p = float(mlcanopy_inst.zref_forcing[p])
    gac_to_hc_p = float(mlcanopy_inst.gac_to_hc_canopy[p])

    gac = mlcanopy_inst.gac_profile
    gac0 = mlcanopy_inst.gac0_soil
    kc_eddy = mlcanopy_inst.kc_eddy_profile

    # Pre-extract zs_profile row to avoid per-layer JAX syncs
    _zs_p = np.asarray(mlcanopy_inst.zs_profile[p])

    # Numpy accumulators for batch write-back
    _gac_new = np.zeros(_ncan + 2)
    _kc_eddy_new = np.zeros(_ncan + 2)

    # ------------------------------------------------------------------
    # Above-canopy conductances — Fortran lines 591-604
    # ------------------------------------------------------------------
    # Use scalar _GetPsiRSL_scalar to avoid ~120 JAX eager-dispatch calls per loop
    for ic in range(_ntop + 1, _ncan):  # Fortran: ntop+1 to ncan-1
        zs_lo = float(_zs_p[ic])
        zs_hi = float(_zs_p[ic + 1])
        _, psic1, _, _ = _GetPsiRSL_scalar(zs_lo, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        _, psic2, _, _ = _GetPsiRSL_scalar(zs_hi, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        psic = psic2 - psic1
        zlog_c = math.log((zs_hi - zdisp_p) / (zs_lo - zdisp_p))
        _gac_new[ic] = rhomol_p * vkc * ustar_p / (zlog_c + psic)

    # Top layer to reference height — Fortran lines 606-612
    ic = _ncan
    zs_lo = float(_zs_p[ic])
    _, psic1, _, _ = _GetPsiRSL_scalar(zs_lo, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    _, psic2, _, _ = _GetPsiRSL_scalar(zref_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psic = psic2 - psic1
    zlog_c = math.log((zref_p - zdisp_p) / (zs_lo - zdisp_p))
    _gac_new[ic] = rhomol_p * vkc * ustar_p / (zlog_c + psic)

    # ztop to zs(ntop+1) — Fortran lines 614-620
    ic = _ntop
    _, psic1, _, _ = _GetPsiRSL_scalar(ztop_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    _, psic2, _, _ = _GetPsiRSL_scalar(float(_zs_p[ic + 1]), ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psic = psic2 - psic1
    zlog_c = math.log((float(_zs_p[ic + 1]) - zdisp_p) / (ztop_p - zdisp_p))
    gac_above_foliage = rhomol_p * vkc * ustar_p / (zlog_c + psic)

    # Consistency check — Fortran lines 622-626
    sumres = 1.0 / gac_above_foliage
    for ic in range(_ntop + 1, _ncan + 1):
        sumres += 1.0 / _gac_new[ic]
    if abs(1.0 / sumres - gac_to_hc_p) > 1.0e-6:
        endrun(msg=" ERROR: AerodynamicConductance: above-canopy aerodynamic conductance error")

    # ------------------------------------------------------------------
    # Within-canopy conductances — Fortran lines 632-647
    # ------------------------------------------------------------------
    for ic in range(1, _ntop):  # Fortran: 1 to ntop-1
        zl = float(_zs_p[ic]) - ztop_p
        zu = float(_zs_p[ic + 1]) - ztop_p
        res = (
            PrSc_p
            / (beta_p * ustar_p)
            * (math.exp(-zl / lm_over_beta) - math.exp(-zu / lm_over_beta))
        )
        _gac_new[ic] = rhomol_p / res

    # Top foliage layer: combine below and above — Fortran lines 649-656
    ic = _ntop
    zl = float(_zs_p[ic]) - ztop_p
    zu = ztop_p - ztop_p  # = 0
    res = (
        PrSc_p / (beta_p * ustar_p) * (math.exp(-zl / lm_over_beta) - math.exp(-zu / lm_over_beta))
    )
    gac_below_foliage = rhomol_p / res
    _gac_new[ic] = 1.0 / (1.0 / gac_below_foliage + 1.0 / gac_above_foliage)

    # ------------------------------------------------------------------
    # Soil surface aerodynamic conductance — Fortran lines 658-686
    # ------------------------------------------------------------------
    z0cg = 0.1 * z0mg
    zs1 = float(_zs_p[1])
    if z0mg > zs1 or z0cg > zs1:
        endrun(msg=" ERROR: AerodynamicConductance: soil roughness error")

    if HF_extension_type == 1:  # Fortran lines 668-673
        zl = z0cg - ztop_p
        zu = zs1 - ztop_p
        res = (
            PrSc_p
            / (beta_p * ustar_p)
            * (math.exp(-zl / lm_over_beta) - math.exp(-zu / lm_over_beta))
        )
        _gac0_val = rhomol_p / res

    elif HF_extension_type == 2:  # Fortran lines 675-678
        zlog_m = math.log(zs1 / z0mg)
        # wind_profile[p,1] was just set by _WindProfile; re-read from mlcanopy_inst
        ustar_g = max(float(mlcanopy_inst.wind_profile[p, 1]), 0.1) * vkc / zlog_m
        _gac0_val = rhomol_p * vkc * ustar_g / zlog_m
    else:
        _gac0_val = float(gac0[p])

    # Resistance cap (ra_max) — Fortran lines 688-694
    _gac0_val = rhomol_p / min(rhomol_p / _gac0_val, ra_max)

    for ic in range(1, _ncan + 1):
        _gac_new[ic] = rhomol_p / min(rhomol_p / _gac_new[ic], ra_max)

    # Eddy diffusivity — Fortran lines 696-700
    for ic in range(1, _ncan + 1):
        if ic == _ncan:
            dz = zref_p - float(_zs_p[ic])
        else:
            dz = float(_zs_p[ic + 1]) - float(_zs_p[ic])
        _kc_eddy_new[ic] = _gac_new[ic] / rhomol_p * dz

    # Batch write-back — 3 bulk ops instead of ~20 per-element writes
    _sl = slice(1, _ncan + 1)
    return mlcanopy_inst._replace(
        gac0_soil=gac0.at[p].set(_gac0_val),
        gac_profile=gac.at[p, _sl].set(jnp.array(_gac_new[_sl])),
        kc_eddy_profile=kc_eddy.at[p, _sl].set(jnp.array(_kc_eddy_new[_sl])),
    )


def _AerodynamicConductance_jax(
    p: int,
    ntop: int,
    ncan: int,
    lm_over_beta,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """JAX-traceable aerodynamic conductances using ``_GetPsiRSL`` and ``jnp.*``.

    Loop bounds ``ntop``, ``ncan`` are concrete Python ints from GridInfo.
    Error checks and consistency assertions are omitted for differentiability.
    """
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    zdisp_p = mlcanopy_inst.zdisp_canopy[p]
    obu_p = mlcanopy_inst.obu_canopy[p]
    beta_p = mlcanopy_inst.beta_canopy[p]
    PrSc_p = mlcanopy_inst.PrSc_canopy[p]
    ustar_p = mlcanopy_inst.ustar_canopy[p]
    rhomol_p = mlcanopy_inst.rhomol_forcing[p]
    zref_p = mlcanopy_inst.zref_forcing[p]

    gac = mlcanopy_inst.gac_profile
    gac0 = mlcanopy_inst.gac0_soil
    kc_eddy = mlcanopy_inst.kc_eddy_profile
    zs_p = mlcanopy_inst.zs_profile[p]

    # --- Above-canopy conductances ---
    _eps_ac = jnp.asarray(1e-30)
    for ic in range(ntop + 1, ncan):
        zs_lo = zs_p[ic]
        zs_hi = zs_p[ic + 1]
        _, psic1, _, _ = _GetPsiRSL(zs_lo, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        _, psic2, _, _ = _GetPsiRSL(zs_hi, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
        psic_d = psic2 - psic1
        zlog_c = jnp.log(
            jnp.maximum((zs_hi - zdisp_p) / jnp.maximum(zs_lo - zdisp_p, _eps_ac), _eps_ac)
        )
        # sign * max(|x|, eps) avoids jnp.where-as-denominator NaN grads
        _dc_abs = jnp.maximum(jnp.abs(zlog_c + psic_d), _eps_ac)
        _dc_sign = jnp.where((zlog_c + psic_d) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
        gac = gac.at[p, ic].set(rhomol_p * vkc * ustar_p * _dc_sign / _dc_abs)

    # Top layer to reference height
    ic = ncan
    zs_lo = zs_p[ic]
    _, psic1, _, _ = _GetPsiRSL(zs_lo, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    _, psic2, _, _ = _GetPsiRSL(zref_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psic_d = psic2 - psic1
    zlog_c = jnp.log(
        jnp.maximum((zref_p - zdisp_p) / jnp.maximum(zs_lo - zdisp_p, _eps_ac), _eps_ac)
    )
    _dc_abs = jnp.maximum(jnp.abs(zlog_c + psic_d), _eps_ac)
    _dc_sign = jnp.where((zlog_c + psic_d) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    gac = gac.at[p, ic].set(rhomol_p * vkc * ustar_p * _dc_sign / _dc_abs)

    # ztop to zs(ntop+1)
    _, psic1, _, _ = _GetPsiRSL(ztop_p, ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    _, psic2, _, _ = _GetPsiRSL(zs_p[ntop + 1], ztop_p, zdisp_p, obu_p, beta_p, PrSc_p)
    psic_d = psic2 - psic1
    _hcd_ac = jnp.maximum(ztop_p - zdisp_p, _eps_ac)
    zlog_c = jnp.log(jnp.maximum((zs_p[ntop + 1] - zdisp_p) / _hcd_ac, _eps_ac))
    _dc_abs = jnp.maximum(jnp.abs(zlog_c + psic_d), _eps_ac)
    _dc_sign = jnp.where((zlog_c + psic_d) < 0.0, jnp.asarray(-1.0), jnp.asarray(1.0))
    gac_above_foliage = rhomol_p * vkc * ustar_p * _dc_sign / _dc_abs

    # --- Within-canopy conductances ---
    _bu_safe = jnp.maximum(beta_p * ustar_p, _eps_ac)
    for ic in range(1, ntop):
        zl = zs_p[ic] - ztop_p
        zu = zs_p[ic + 1] - ztop_p
        res = PrSc_p / _bu_safe * (jnp.exp(-zl / lm_over_beta) - jnp.exp(-zu / lm_over_beta))
        res_safe = jnp.maximum(jnp.abs(res), _eps_ac)
        gac = gac.at[p, ic].set(rhomol_p / res_safe)

    # Top foliage layer: combine below and above
    ic = ntop
    zl = zs_p[ic] - ztop_p
    res = PrSc_p / _bu_safe * (jnp.exp(-zl / lm_over_beta) - 1.0)  # exp(0)=1
    res_safe = jnp.maximum(jnp.abs(res), _eps_ac)
    gac_below_foliage = rhomol_p / res_safe
    gac = gac.at[p, ic].set(1.0 / (1.0 / gac_below_foliage + 1.0 / gac_above_foliage))

    # --- Soil surface aerodynamic conductance ---
    z0cg = 0.1 * z0mg
    zs1 = zs_p[1]
    z0mg_safe = jnp.maximum(z0mg, _eps_ac)

    if HF_extension_type == 1:  # static branch
        zl = z0cg - ztop_p
        zu = zs1 - ztop_p
        res = PrSc_p / _bu_safe * (jnp.exp(-zl / lm_over_beta) - jnp.exp(-zu / lm_over_beta))
        res_safe = jnp.maximum(jnp.abs(res), _eps_ac)
        _gac0_val = rhomol_p / res_safe
    elif HF_extension_type == 2:  # static branch
        zlog_m = jnp.log(jnp.maximum(zs1 / z0mg_safe, _eps_ac))
        zlog_m_safe = jnp.maximum(jnp.abs(zlog_m), _eps_ac)
        ustar_g = jnp.maximum(mlcanopy_inst.wind_profile[p, 1], 0.1) * vkc / zlog_m_safe
        _gac0_val = rhomol_p * vkc * ustar_g / zlog_m_safe
    else:
        _gac0_val = gac0[p]

    # Resistance cap
    _gac0_val = rhomol_p / jnp.minimum(rhomol_p / _gac0_val, ra_max)
    gac0 = gac0.at[p].set(_gac0_val)

    for ic in range(1, ncan + 1):
        gac = gac.at[p, ic].set(rhomol_p / jnp.minimum(rhomol_p / gac[p, ic], ra_max))

    # Eddy diffusivity
    for ic in range(1, ncan + 1):
        if ic == ncan:  # static branch (ic and ncan are Python ints)
            dz = zref_p - zs_p[ic]
        else:
            dz = zs_p[ic + 1] - zs_p[ic]
        kc_eddy = kc_eddy.at[p, ic].set(gac[p, ic] / rhomol_p * dz)

    return mlcanopy_inst._replace(
        gac0_soil=gac0,
        gac_profile=gac,
        kc_eddy_profile=kc_eddy,
    )


# ===========================================================================
# Private: HF2008 main solver
# ===========================================================================


def _HF2008(
    nstep_ml: int,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Canopy turbulence, wind profile, and aerodynamic conductances using
    the Harman and Finnigan (2008) roughness sublayer parameterisation.

    Mirrors Fortran subroutine ``HF2008`` (lines 67-160).

    References: Bonan et al. (2018) *Geosci. Model Dev.*, 11, 1467-1496;
    Bonan et al. (2025).

    For each patch (Fortran lines 108-156):

    1. Canopy density length scale ``Lc = ztop / (cd*(lai+sai))``.
    2. Temperature and specific humidity at canopy top for
       :func:`_ObuFunc`.
    3. Obukhov length via :func:`_GetObu`.
    4. Roughness length via :func:`_RoughnessLength`.
    5. Compute ``lm = 2*β³*Lc``, ``η = β/lm*ztop``, warn if ``η ≥ η_max``,
       then ``lm/β = ztop/η``.
    6. Wind profile via :func:`_WindProfile`.
    7. Momentum flux profile (Fortran lines 138-144):
       ``mflx(ic) = -ustar²`` above canopy; exponential decay within.
    8. Aerodynamic conductances via :func:`_AerodynamicConductance`.

    Args:
        nstep_ml: Current multilayer timestep counter (for warning messages).
        num_filter: Number of patches.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; all turbulence output fields
            updated.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    from legoesm.land.canopy.clm_ml_backend.clm_src_utils.clm_time_manager import get_nstep  # noqa: F401

    nstep = get_nstep()

    Lc = mlcanopy_inst.Lc_canopy
    taf = mlcanopy_inst.taf_canopy
    qaf = mlcanopy_inst.qaf_canopy
    mflx = mlcanopy_inst.mflx_profile

    for fp in range(1, num_filter + 1):
        p = int(filter_patch[fp - 1])

        pref_p = float(mlcanopy_inst.pref_forcing[p])
        ztop_p = float(mlcanopy_inst.ztop_canopy[p])
        lai_p = float(mlcanopy_inst.lai_canopy[p])
        sai_p = float(mlcanopy_inst.sai_canopy[p])
        _ntop = int(mlcanopy_inst.ntop_canopy[p])
        _ncan = int(mlcanopy_inst.ncan_canopy[p])

        # Canopy density length scale — Fortran line 110
        Lc_val = ztop_p / (cd * (lai_p + sai_p))
        Lc = Lc.at[p].set(Lc_val)
        mlcanopy_inst = mlcanopy_inst._replace(Lc_canopy=Lc)

        # Temperature and humidity at canopy top — Fortran lines 112-113
        tair_ntop = float(mlcanopy_inst.tair_profile[p, _ntop])
        eair_ntop = float(mlcanopy_inst.eair_profile[p, _ntop])
        taf_val = tair_ntop
        qaf_val = mmh2o / mmdry * eair_ntop / (pref_p - (1.0 - mmh2o / mmdry) * eair_ntop)
        taf = taf.at[p].set(taf_val)
        qaf = qaf.at[p].set(qaf_val)
        mlcanopy_inst = mlcanopy_inst._replace(taf_canopy=taf, qaf_canopy=qaf)

        # Obukhov length — Fortran line 115
        mlcanopy_inst = _GetObu(p, mlcanopy_inst)

        # Roughness length — Fortran line 118
        mlcanopy_inst = _RoughnessLength(p, mlcanopy_inst)

        # lm / beta — Fortran lines 126-130
        beta_p = float(mlcanopy_inst.beta_canopy[p])
        lm = 2.0 * beta_p**3 * Lc_val
        eta = beta_p / lm * ztop_p
        if eta >= eta_max:
            print(" Warning: HF2008: lm/beta error")
            print(f" nstep = {nstep}  nstep_ml = {nstep_ml}")
            print(f" eta = {eta}")
        lm_over_beta = ztop_p / eta

        # Wind profile — Fortran line 132
        mlcanopy_inst = _WindProfile(p, lm_over_beta, mlcanopy_inst)

        # Momentum flux profile — Fortran lines 134-140
        ustar_p = float(mlcanopy_inst.ustar_canopy[p])
        _zw_p = np.asarray(mlcanopy_inst.zw_profile[p])
        _mflx_new = np.zeros(_ncan + 2)
        for ic in range(1, _ncan + 1):
            zw_ic = float(_zw_p[ic])
            if zw_ic > ztop_p:
                _mflx_new[ic] = -(ustar_p**2)
            else:
                _mflx_new[ic] = -(ustar_p**2) * jnp.exp(2.0 * (zw_ic - ztop_p) / lm_over_beta)
        _sl_m = slice(1, _ncan + 1)
        mflx = mflx.at[p, _sl_m].set(jnp.array(_mflx_new[_sl_m]))
        mlcanopy_inst = mlcanopy_inst._replace(mflx_profile=mflx)

        # Aerodynamic conductances — Fortran line 142
        mlcanopy_inst = _AerodynamicConductance(p, lm_over_beta, mlcanopy_inst)

    return mlcanopy_inst


def _HF2008_diff(
    grid: GridInfo,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """JAX-traceable HF2008 turbulence driver.

    Replaces :func:`_HF2008` for differentiable mode.  Uses
    ``_GetObu(diff_mode=True)``, ``_RoughnessLength_jax``,
    ``_WindProfile_jax``, and ``_AerodynamicConductance_jax``.
    All intermediate values stay as JAX arrays; no ``float()`` or
    ``np.asarray()`` calls.
    """
    p = grid.p
    ntop = grid.ntop
    ncan = grid.ncan

    Lc = mlcanopy_inst.Lc_canopy
    taf = mlcanopy_inst.taf_canopy
    qaf = mlcanopy_inst.qaf_canopy
    mflx = mlcanopy_inst.mflx_profile

    pref_p = mlcanopy_inst.pref_forcing[p]
    ztop_p = mlcanopy_inst.ztop_canopy[p]
    lai_p = mlcanopy_inst.lai_canopy[p]
    sai_p = mlcanopy_inst.sai_canopy[p]

    # Canopy density length scale
    Lc_val = ztop_p / (cd * (lai_p + sai_p))
    Lc = Lc.at[p].set(Lc_val)
    mlcanopy_inst = mlcanopy_inst._replace(Lc_canopy=Lc)

    # Temperature and humidity at canopy top
    tair_ntop = mlcanopy_inst.tair_profile[p, ntop]
    eair_ntop = mlcanopy_inst.eair_profile[p, ntop]
    taf_val = tair_ntop
    qaf_val = mmh2o / mmdry * eair_ntop / (pref_p - (1.0 - mmh2o / mmdry) * eair_ntop)
    taf = taf.at[p].set(taf_val)
    qaf = qaf.at[p].set(qaf_val)
    mlcanopy_inst = mlcanopy_inst._replace(taf_canopy=taf, qaf_canopy=qaf)

    # Obukhov length (differentiable)
    mlcanopy_inst = _GetObu(p, mlcanopy_inst, diff_mode=True)

    # Roughness length (differentiable)
    mlcanopy_inst = _RoughnessLength_jax(p, mlcanopy_inst)

    # lm / beta
    beta_p = mlcanopy_inst.beta_canopy[p]
    lm = 2.0 * beta_p**3 * Lc_val
    eta = beta_p / lm * ztop_p
    # Skip eta warning in diff mode
    lm_over_beta = ztop_p / eta

    # Wind profile (differentiable)
    mlcanopy_inst = _WindProfile_jax(p, ntop, ncan, lm_over_beta, mlcanopy_inst)

    # Momentum flux profile — use jnp.where instead of if
    ustar_p = mlcanopy_inst.ustar_canopy[p]
    zw_p = mlcanopy_inst.zw_profile[p]
    for ic in range(1, ncan + 1):
        zw_ic = zw_p[ic]
        mflx_val = jnp.where(
            zw_ic > ztop_p,
            -(ustar_p**2),
            -(ustar_p**2) * jnp.exp(2.0 * (zw_ic - ztop_p) / lm_over_beta),
        )
        mflx = mflx.at[p, ic].set(mflx_val)
    mlcanopy_inst = mlcanopy_inst._replace(mflx_profile=mflx)

    # Aerodynamic conductances (differentiable)
    mlcanopy_inst = _AerodynamicConductance_jax(p, ntop, ncan, lm_over_beta, mlcanopy_inst)

    return mlcanopy_inst


# ===========================================================================
# Public: CanopyTurbulence driver
# ===========================================================================


def CanopyTurbulence(
    nstep_ml: int,
    num_filter: int,
    filter_patch: Sequence[int],
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,
) -> mlcanopy_type:
    """
    Main driver for the canopy turbulence parameterisation.

    Mirrors Fortran subroutine ``CanopyTurbulence`` (lines 43-67).

    When ``grid`` is provided (differentiable mode), uses the JAX-traceable
    ``_HF2008_diff`` path.

    Args:
        nstep_ml: Current multilayer timestep counter.
        num_filter: Number of patches in the filter.
        filter_patch: Patch index filter (1-based values).
        mlcanopy_inst: Canopy container; all turbulence output fields
            updated by :func:`_HF2008`.
        grid: If provided, use differentiable code path.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    if turb_type == 1:
        if grid is not None:
            return _HF2008_diff(grid, mlcanopy_inst)
        return _HF2008(nstep_ml, num_filter, filter_patch, mlcanopy_inst)
    else:
        endrun(msg=" ERROR: CanopyTurbulence: turb_type not valid")
        return mlcanopy_inst  # Unreachable


# ===========================================================================
# Public: LookupPsihatINI
# ===========================================================================


def LookupPsihatINI() -> None:
    """
    Initialise the RSL psihat look-up tables from a NetCDF file.

    Mirrors Fortran subroutine ``LookupPsihatINI`` (lines 455-530).

    Reads variables ``dtLgridM``, ``zdtgridM``, ``psigridM``,
    ``dtLgridH``, ``zdtgridH``, and ``psigridH`` from the file whose
    path is given by the module-level variable ``rslfile``.

    Fortran stores the grids as:

    .. code-block:: none

        zdtgridM(nZ, 1), dtLgridM(1, nL), psigridM(nZ, nL)

    while the NetCDF file has dimensions in the opposite order:

    .. code-block:: none

        psigridM_nc(nL, nZ)  →  psigridM[ii, jj] = psigridM_nc[jj, ii]

    The copy loop transposes the NetCDF arrays into the Fortran (and
    Python) convention.  After this routine the module-level arrays
    ``zdtgridM``, ``dtLgridM``, ``psigridM``, ``zdtgridH``,
    ``dtLgridH``, and ``psigridH`` are populated and ready for use
    by :func:`_LookupPsihat`.

    Raises:
        SystemExit: (via :func:`endrun`) if the NetCDF dimensions do
            not match the expected ``nZ`` and ``nL`` or if any variable
            cannot be read.
    """
    global dtLgridM, dtLgridH, zdtgridM, zdtgridH, psigridM, psigridH
    global _zdtgM_1d, _dtLgM_1d, _neg_zdtgM_1d
    global _zdtgH_1d, _dtLgH_1d, _neg_zdtgH_1d
    global _dtLgM_jax, _zdtgM_jax, _neg_zdtgM_jax, _psigridM_jax
    global _dtLgH_jax, _zdtgH_jax, _neg_zdtgH_jax, _psigridH_jax

    import netCDF4 as nc  # noqa: F401 — defer import

    print("Attempting to read RSL look-up table .....")

    with nc.Dataset(rslfile, "r") as ncid:

        # Check dimensions — Fortran lines 487-498
        nZ_nc = len(ncid.dimensions["nZ"])
        nL_nc = len(ncid.dimensions["nL"])
        if nZ_nc != nZ:
            endrun(msg=" ERROR: LookupPsihatINI: nZ does not equal expected value")
        if nL_nc != nL:
            endrun(msg=" ERROR: LookupPsihatINI: nL does not equal expected value")

        # Read 1-D coordinate arrays and 2-D psihat grids — Fortran lines 500-520
        def _read(name: str):
            if name not in ncid.variables:
                endrun(msg=f" ERROR: LookupPsihatINI: error reading {name}")
            return ncid.variables[name][:]

        dtLgridM_nc = np.asarray(_read("dtLgridM"), dtype=float)  # shape (nL,)
        zdtgridM_nc = np.asarray(_read("zdtgridM"), dtype=float)  # shape (nZ,)
        psigridM_nc = np.asarray(
            _read("psigridM"), dtype=float
        )  # Actually shape (nZ, nL) not (nL, nZ)!

        dtLgridH_nc = np.asarray(_read("dtLgridH"), dtype=float)
        zdtgridH_nc = np.asarray(_read("zdtgridH"), dtype=float)
        psigridH_nc = np.asarray(_read("psigridH"), dtype=float)

    print("Successfully read RSL look-up table")

    # Copy into module-level arrays with Fortran index conventions —
    # Fortran lines 522-530:
    #   dtLgridM(1, jj) = dtLgridM_nc(jj)  →  dtLgridM[0, jj]
    #   zdtgridM(ii, 1) = zdtgridM_nc(ii)  →  zdtgridM[ii, 0]
    #   psigridM(ii,jj) = psigridM_nc(jj,ii) →  psigridM[ii, jj]
    #
    # NOTE: psigridM_nc is already in shape (nZ, nL), so no transpose needed

    # Populate numpy arrays directly (no JAX immutable operations needed)
    dtLgridM[0, :] = dtLgridM_nc
    dtLgridH[0, :] = dtLgridH_nc
    zdtgridM[:, 0] = zdtgridM_nc
    zdtgridH[:, 0] = zdtgridH_nc
    psigridM[:, :] = psigridM_nc
    psigridH[:, :] = psigridH_nc

    # Populate cached 1D views — eliminates per-call slicing and negation in _LookupPsihatM/H
    _dtLgM_1d = dtLgridM[0].copy()
    _zdtgM_1d = zdtgridM[:, 0].copy()
    _neg_zdtgM_1d = -_zdtgM_1d
    _dtLgH_1d = dtLgridH[0].copy()
    _zdtgH_1d = zdtgridH[:, 0].copy()
    _neg_zdtgH_1d = -_zdtgH_1d

    # Pre-convert to JAX arrays once — avoids jnp.asarray() on every
    # _LookupPsihatM/H call (~10K times per diff-mode forward pass).
    _dtLgM_jax = jnp.asarray(_dtLgM_1d)
    _zdtgM_jax = jnp.asarray(_zdtgM_1d)
    _neg_zdtgM_jax = jnp.asarray(_neg_zdtgM_1d)
    _psigridM_jax = jnp.asarray(psigridM)
    _dtLgH_jax = jnp.asarray(_dtLgH_1d)
    _zdtgH_jax = jnp.asarray(_zdtgH_1d)
    _neg_zdtgH_jax = jnp.asarray(_neg_zdtgH_1d)
    _psigridH_jax = jnp.asarray(psigridH)
