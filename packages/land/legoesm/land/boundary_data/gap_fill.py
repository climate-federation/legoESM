"""Reconcile the surfdata with the driver's authoritative land-sea mask.

The driver's land-sea mask (sftlf / ERA5 lsm / topography) — not the surfdata —
decides which cells are land.  Wherever the mask says land but the surfdata has
no coverage (small islands, coast mismatch, HWSD / ice gaps), the land model
must still get *finite* parameters.  These helpers fill such cells (and any
residual NaN) with a bare-soil fallback, so every column is valid regardless
of the mask.

Two flavours of the fill:
  - :func:`fill_land_param_gaps` — public, host-side: computes the
    surfdata-covered mask from the gsd via :func:`surfdata_covered`, builds a
    per-scheme bare fallback, and ``jnp.where`` -fills the leaves.  Used at
    simulation start by :func:`~legoesm.land.boundary_data.init_land_surface_data`.
  - :func:`_gap_fill_tree` — internal, JAX-pure: takes a *precomputed* covered
    mask and bare-fallback tree, so it can run inside a ``lax.scan`` body
    without a host roundtrip.  Used by
    :func:`~legoesm.land.boundary_data.make_step_land_params_updater`.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.land.surface_params import (
    LandSurfaceParams,
    clm5_pft_table, array_to_params, PARAM_NAMES,
)
from legoesm.land.canopy.config import CanopyLandParams

from legoesm.land.boundary_data._internals import (
    _CI_DEFAULT, _KN_DEFAULT, _ALF_DEFAULT,
    _M_C3, _M_C4, _B0_C3, _B0_C4,
    _TGC_DEFAULT_C, _HC_MIN_M,
    _EMISS_BARE, _RZ0M_BARE,
    _ALB_VIS_BARE, _ALB_NIR_BARE,
)


def _bare_land_surface_params(ncol: int):
    """Bare-soil :class:`LandSurfaceParams` (CLM5 PFT 0 row) broadcast to ncol."""
    row = np.asarray(clm5_pft_table())[0]                    # bare_soil (12,)
    return array_to_params(jnp.broadcast_to(jnp.asarray(row), (ncol, row.shape[0])), PARAM_NAMES)


def _bare_canopy_params(ncol: int) -> CanopyLandParams:
    """Bare (no-vegetation) :class:`CanopyLandParams` broadcast to ncol."""
    full = lambda v: jnp.full(ncol, v)
    return CanopyLandParams(
        LAI=full(0.0), hc=full(_HC_MIN_M), fC4=full(0.0), FNonVeg=full(1.0),
        CI=full(_CI_DEFAULT), kn=full(_KN_DEFAULT),
        Vcmax25_C3_leaf=full(0.0), Vcmax25_C4_leaf=full(0.0),
        m_C3=full(_M_C3), m_C4=full(_M_C4), b0_C3=full(_B0_C3), b0_C4=full(_B0_C4),
        alf=full(_ALF_DEFAULT), TgC=full(_TGC_DEFAULT_C),
        ALB_VIS=full(_ALB_VIS_BARE), ALB_NIR=full(_ALB_NIR_BARE),
        emissivity=full(_EMISS_BARE), rz0m=full(_RZ0M_BARE), rd=full(0.0),
    )


def surfdata_covered(gsd) -> np.ndarray:
    """Boolean ``(ncol,)``: columns the surfdata actually covers (finite pft_frac)."""
    pft = np.asarray(gsd.pft_frac)
    pft0 = pft[0] if pft.ndim == 3 else pft                  # (ncol, npft)
    return np.isfinite(pft0.sum(axis=-1))


def fill_land_param_gaps(land_params, gsd):
    """Replace surfdata-uncovered (and any non-finite) columns with a bare fallback.

    Reconciles the surfdata with the driver's authoritative land mask: a column is
    kept only where the surfdata covers it *and* the value is finite, otherwise it
    falls back to bare soil.  Works for :class:`CanopyLandParams` or
    :class:`LandSurfaceParams`.
    """
    covered = jnp.asarray(surfdata_covered(gsd))            # (ncol,)
    ncol = covered.shape[0]
    fb = (_bare_canopy_params(ncol) if isinstance(land_params, CanopyLandParams)
          else _bare_land_surface_params(ncol))

    def _fill(v, f):
        v = jnp.asarray(v)
        keep = covered.reshape(covered.shape + (1,) * (v.ndim - 1))
        return jnp.where(keep & jnp.isfinite(v), v, jnp.asarray(f))

    return jax.tree.map(_fill, land_params, fb)


def _gap_fill_tree(land_params, fb, covered_jnp):
    """Mask-fill helper: same logic as :func:`fill_land_param_gaps` but with
    pre-computed (JAX-side) ``covered`` mask, so it runs inside a ``lax.scan``
    body without a host roundtrip."""
    def _fill(v, f):
        v = jnp.asarray(v)
        keep = covered_jnp.reshape(covered_jnp.shape + (1,) * (v.ndim - 1))
        return jnp.where(keep & jnp.isfinite(v), v, jnp.asarray(f))
    return jax.tree.map(_fill, land_params, fb)
