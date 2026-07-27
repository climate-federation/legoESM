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
  - :func:`gap_fill_tree` — internal, JAX-pure: takes a *precomputed* covered
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
    tuned_pft_root_arrays,
    CI_DEFAULT, KN_DEFAULT, ALF_DEFAULT,
    M_C3, M_C4, B0_C3, B0_C4,
    TGC_DEFAULT_C, HC_MIN_M,
    EMISS_BARE, RZ0M_BARE,
    ALB_VIS_BARE, ALB_NIR_BARE,
)


def bare_land_surface_params(ncol: int):
    """Bare-soil :class:`LandSurfaceParams` (CLM5 PFT 0 row) broadcast to ncol.

    ``fC4`` is set to 0 (bare soil is not C4) so this fallback's pytree structure
    matches the surfdata provider's output (which always populates ``fC4``); a
    ``None`` here would break the ``jax.tree.map`` gap-fill against a populated
    ``fC4`` leaf.
    """
    row = np.asarray(clm5_pft_table())[0]                    # bare_soil (12,)
    p = array_to_params(jnp.broadcast_to(jnp.asarray(row), (ncol, row.shape[0])), PARAM_NAMES)
    return p._replace(fC4=jnp.zeros(ncol))


# Bare ground is index 0 of the CLM5 17-PFT axis (CLM5_PFT_NAMES[0]).
_BARE_PFT_INDEX = 0


def bare_canopy_params(ncol: int, *, tuned_root_params: bool = False) -> CanopyLandParams:
    """Bare (no-vegetation) :class:`CanopyLandParams` broadcast to ncol.

    ``tuned_root_params`` also fills the optional per-column
    ``root_depth``/``theta_wp``/``theta_fc`` from the BARE-SOIL row (PFT index 0)
    of the calibrated tables.  This must MATCH the params being gap-filled: the
    fallback and the values are combined leaf-by-leaf through a pytree map, so a
    fallback carrying ``None`` where the params carry an array is a structure
    mismatch (``None is not a valid value for jnp.array``), not a silent default.
    """
    full = lambda v: jnp.full(ncol, v)
    _root_kw = {}
    if tuned_root_params:
        _rt = tuned_pft_root_arrays()
        _root_kw = {k: full(float(v[_BARE_PFT_INDEX])) for k, v in _rt.items()}
    return CanopyLandParams(
        LAI=full(0.0), hc=full(HC_MIN_M), fC4=full(0.0), FNonVeg=full(1.0),
        CI=full(CI_DEFAULT), kn=full(KN_DEFAULT),
        Vcmax25_C3_leaf=full(0.0), Vcmax25_C4_leaf=full(0.0),
        m_C3=full(M_C3), m_C4=full(M_C4), b0_C3=full(B0_C3), b0_C4=full(B0_C4),
        alf=full(ALF_DEFAULT), TgC=full(TGC_DEFAULT_C),
        ALB_VIS=full(ALB_VIS_BARE), ALB_NIR=full(ALB_NIR_BARE),
        emissivity=full(EMISS_BARE), rz0m=full(RZ0M_BARE), rd=full(0.0),
        **_root_kw,
    )


def surfdata_covered(gsd) -> np.ndarray:
    """Boolean ``(ncol,)``: columns the surfdata actually covers (finite pft_frac)."""
    pft = np.asarray(gsd.pft_frac)
    pft0 = pft[0] if pft.ndim == 3 else pft                  # (ncol, npft)
    return np.isfinite(pft0.sum(axis=-1))


def fill_land_param_gaps(land_params, gsd, f_land=None):
    """Replace surfdata-uncovered (and any non-finite) columns with a bare fallback.

    Reconciles the surfdata with the driver's authoritative land mask: a column is
    kept only where the surfdata covers it *and* the value is finite, otherwise it
    falls back to bare soil.  Works for :class:`CanopyLandParams` or
    :class:`LandSurfaceParams`.

    ``f_land`` (optional, ``(ncol,)`` in the model's column order, e.g. the
    driver's ``_f_land`` ravelled): when given, surfdata values are kept ONLY where
    the *driver* mask says land (``f_land > 0``) AND the surfdata covers the cell
    with finite values; ocean cells (``f_land == 0``) and mask-land cells the
    surfdata misses both fall back to bare soil.  This pins the land params to the
    authoritative mask so they carry surfdata only on driver-land cells and can
    never disagree with the ocean tile (weighted by ``1 - f_land``).  Land params
    are finite on every column either way; ``f_land`` only constrains *where*
    surfdata (vs bare) appears.  Without ``f_land`` the behaviour is unchanged.
    """
    covered = jnp.asarray(surfdata_covered(gsd))            # (ncol,)
    ncol = covered.shape[0]
    keep_col = covered
    if f_land is not None:
        f_land = jnp.asarray(f_land).reshape(-1)
        if f_land.shape[0] != ncol:
            raise ValueError(
                f"f_land has {f_land.shape[0]} columns but the land params have "
                f"{ncol}; ravel / grid-column mismatch."
            )
        keep_col = keep_col & (f_land > 0.0)                # surfdata only on driver-land
    fb = (bare_canopy_params(
              ncol, tuned_root_params=land_params.root_depth is not None)
          if isinstance(land_params, CanopyLandParams)
          else bare_land_surface_params(ncol))

    def _fill(v, f):
        v = jnp.asarray(v)
        keep = keep_col.reshape(keep_col.shape + (1,) * (v.ndim - 1))
        return jnp.where(keep & jnp.isfinite(v), v, jnp.asarray(f))

    return jax.tree.map(_fill, land_params, fb)


def gap_fill_tree(land_params, fb, covered_jnp):
    """Mask-fill helper: same logic as :func:`fill_land_param_gaps` but with
    pre-computed (JAX-side) ``covered`` mask, so it runs inside a ``lax.scan``
    body without a host roundtrip."""
    def _fill(v, f):
        v = jnp.asarray(v)
        keep = covered_jnp.reshape(covered_jnp.shape + (1,) * (v.ndim - 1))
        return jnp.where(keep & jnp.isfinite(v), v, jnp.asarray(f))
    return jax.tree.map(_fill, land_params, fb)
