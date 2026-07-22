"""
JAX/Python translation of the CLM soil state variables module.

Original Fortran module: SoilStateType
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval as nan
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsno, nlevsoi
from legoesm.land.canopy.clm_ml_backend.clm_src_main.decompMod import bounds_type

# ---------------------------------------------------------------------------
# soilstate_type
# ---------------------------------------------------------------------------


class soilstate_type(NamedTuple):
    """
    Soil state variables.

    Mirrors Fortran derived type ``soilstate_type`` (lines 22-43).

    All arrays are allocated over column (``c``) or patch (``p``)
    indices with 1-based soil layer indices, matching the Fortran
    ``begc:endc`` / ``begp:endp`` convention.  Index 0 is allocated
    but unused for column/patch dimensions; soil layer index 0 is
    unused except for ``thk_col``, which spans
    ``-nlevsno+1 : nlevgrnd`` (Fortran) → Python shape
    ``(endc+1, nlevsno+nlevgrnd)``.

    Attributes:
        cellorg_col:    Organic matter (kg/m3),
                        shape ``(endc+1, nlevsoi+1)``.
        cellsand_col:   Sand percent,
                        shape ``(endc+1, nlevsoi+1)``.
        cellclay_col:   Clay percent,
                        shape ``(endc+1, nlevsoi+1)``.
        hksat_col:      Hydraulic conductivity at saturation (mm H2O/s),
                        shape ``(endc+1, nlevgrnd+1)``.
        hk_l_col:       Hydraulic conductivity (mm H2O/s),
                        shape ``(endc+1, nlevgrnd+1)``.
        smp_l_col:      Soil matric potential (mm),
                        shape ``(endc+1, nlevgrnd+1)``.
        bsw_col:        Clapp and Hornberger "b" parameter,
                        shape ``(endc+1, nlevgrnd+1)``.
        watsat_col:     Volumetric soil water at saturation (porosity),
                        shape ``(endc+1, nlevgrnd+1)``.
        sucsat_col:     Minimum soil suction (mm),
                        shape ``(endc+1, nlevgrnd+1)``.
        dsl_col:        Dry surface layer thickness (mm),
                        shape ``(endc+1,)``.
        soilresis_col:  Soil evaporative resistance S&L14 (s/m),
                        shape ``(endc+1,)``.
        thk_col:        Thermal conductivity of each layer (W/m/K).
                        Fortran: ``(-nlevsno+1 : nlevgrnd)`` → Python
                        shape ``(endc+1, nlevsno+nlevgrnd+1)`` where
                        index ``k`` in Python corresponds to Fortran
                        index ``k - nlevsno + 1``, i.e. Python index
                        ``nlevsno-1`` = Fortran index 0.  In practice
                        the caller accesses this as
                        ``thk_col[c, snl+nlevsno]`` to obtain Fortran
                        ``thk_col(c, snl+1)``.
        tkmg_col:       Thermal conductivity, soil minerals (W/m/K),
                        shape ``(endc+1, nlevgrnd+1)``.
        tkdry_col:      Thermal conductivity, dry soil (W/m/K),
                        shape ``(endc+1, nlevgrnd+1)``.
        csol_col:       Heat capacity, soil solids (J/m³/K),
                        shape ``(endc+1, nlevgrnd+1)``.
        rootfr_patch:   Effective fraction of roots in each soil layer,
                        shape ``(endp+1, nlevgrnd+1)``.
    """

    # sand / clay / organic matter
    cellorg_col: jnp.ndarray  # (endc+1, nlevsoi+1)
    cellsand_col: jnp.ndarray  # (endc+1, nlevsoi+1)
    cellclay_col: jnp.ndarray  # (endc+1, nlevsoi+1)

    # hydraulic properties
    hksat_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    hk_l_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    smp_l_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    bsw_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    watsat_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    sucsat_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    dsl_col: jnp.ndarray  # (endc+1,)
    soilresis_col: jnp.ndarray  # (endc+1,)

    # thermal conductivity / heat capacity
    thk_col: jnp.ndarray  # (endc+1, nlevsno+nlevgrnd+1)
    tkmg_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    tkdry_col: jnp.ndarray  # (endc+1, nlevgrnd+1)
    csol_col: jnp.ndarray  # (endc+1, nlevgrnd+1)

    # roots
    rootfr_patch: jnp.ndarray  # (endp+1, nlevgrnd+1)


# ---------------------------------------------------------------------------
# create_soilstate  (replaces Fortran Init + InitAllocate)
# ---------------------------------------------------------------------------


def create_soilstate(bounds: bounds_type) -> soilstate_type:
    """
    Allocate and initialise all soil state arrays to ``nan`` (= ``spval``).

    Mirrors Fortran ``Init`` + ``InitAllocate`` (lines 47-80).

    **Array shapes** (all initialised to ``spval``):

    .. code-block:: none

        cellorg/sand/clay_col  : (endc+1, nlevsoi+1)     Fortran: (begc:endc, 1:nlevsoi)
        hksat/hk_l/smp_l/bsw/
        watsat/sucsat_col      : (endc+1, nlevgrnd+1)    Fortran: (begc:endc, 1:nlevgrnd)
        dsl/soilresis_col      : (endc+1,)               Fortran: (begc:endc)
        thk_col                : (endc+1, nlevsno+nlevgrnd+1)
                                                          Fortran: (begc:endc, -nlevsno+1:nlevgrnd)
        tkmg/tkdry/csol_col    : (endc+1, nlevgrnd+1)    Fortran: (begc:endc, 1:nlevgrnd)
        rootfr_patch           : (endp+1, nlevgrnd+1)    Fortran: (begp:endp, 1:nlevgrnd)

    The ``thk_col`` second dimension has ``nlevsno + nlevgrnd + 1``
    elements so that Python index ``j`` corresponds to Fortran index
    ``j - nlevsno``, preserving the Fortran ``-nlevsno+1 : nlevgrnd``
    range with 1-based soil layers at indices ``nlevsno+1 …
    nlevsno+nlevgrnd``.

    Args:
        bounds: Index bounds from :func:`decompMod.get_clump_bounds`.

    Returns:
        :class:`soilstate_type` with all fields filled with ``spval``.
    """
    endc = bounds.endc
    endp = bounds.endp

    def _rc(*shape: int) -> jnp.ndarray:
        return jnp.full(shape, nan, dtype=jnp.float64)

    nc = endc + 1  # column dimension
    ng = nlevgrnd + 1  # 1:nlevgrnd → index 0 unused
    ns = nlevsoi + 1  # 1:nlevsoi  → index 0 unused
    nth = nlevsno + nlevgrnd + 1  # -nlevsno+1:nlevgrnd
    np_ = endp + 1  # patch dimension

    return soilstate_type(
        # sand / clay / organic matter
        cellorg_col=_rc(nc, ns),
        cellsand_col=_rc(nc, ns),
        cellclay_col=_rc(nc, ns),
        # hydraulic properties
        hksat_col=_rc(nc, ng),
        hk_l_col=_rc(nc, ng),
        smp_l_col=_rc(nc, ng),
        bsw_col=_rc(nc, ng),
        watsat_col=_rc(nc, ng),
        sucsat_col=_rc(nc, ng),
        dsl_col=_rc(nc),
        soilresis_col=_rc(nc),
        # thermal conductivity / heat capacity
        thk_col=_rc(nc, nth),
        tkmg_col=_rc(nc, ng),
        tkdry_col=_rc(nc, ng),
        csol_col=_rc(nc, ng),
        # roots
        rootfr_patch=_rc(np_, ng),
    )
