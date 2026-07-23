"""
JAX translation of pftconMod Fortran module.

Vegetation (PFT) parameters for the CLM land surface model.
Provides data structures and initialization routines for leaf and stem
optical properties, rooting distribution, specific leaf area, and
photosynthetic pathway parameters used in radiative transfer and
carbon/water cycle calculations.

Original Fortran module: pftconMod
Fortran lines 1-260
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis, mxpft, numrad  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import pftcon_val  # noqa: F401

# ---------------------------------------------------------------------------
# PFT index reference (Fortran lines 75-130)
# ---------------------------------------------------------------------------
#    0  => not_vegetated
#    1  => needleleaf_evergreen_temperate_tree
#    2  => needleleaf_evergreen_boreal_tree
#    3  => needleleaf_deciduous_boreal_tree
#    4  => broadleaf_evergreen_tropical_tree
#    5  => broadleaf_evergreen_temperate_tree
#    6  => broadleaf_deciduous_tropical_tree
#    7  => broadleaf_deciduous_temperate_tree
#    8  => broadleaf_deciduous_boreal_tree
#    9  => broadleaf_evergreen_shrub
#   10  => broadleaf_deciduous_temperate_shrub
#   11  => broadleaf_deciduous_boreal_shrub
#   12  => c3_arctic_grass
#   13  => c3_non-arctic_grass
#   14  => c4_grass
#   15  => c3_crop
#   16  => c3_irrigated
#   17-78 => various crop PFTs (see Fortran source for full list)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


class pftcon_type(NamedTuple):
    """
    CLM vegetation (PFT) parameters mirroring Fortran ``pftcon_type``
    (lines 18-35).

    All 1-D arrays have shape ``(mxpft + 1,)`` matching the Fortran
    allocation ``(0:mxpft)``. All 2-D arrays have shape
    ``(mxpft + 1, numrad)`` matching ``(0:mxpft, numrad)``.
    Index 0 is ``not_vegetated``; indices 1–mxpft correspond to the
    PFTs listed above.

    Attributes:
        dleaf: Characteristic leaf dimension (m).
            Fortran: ``dleaf(0:mxpft)``.
        c3psn: Photosynthetic pathway: 0 = C4, 1 = C3.
            Fortran: ``c3psn(0:mxpft)``.
        xl: Leaf/stem orientation index (-).
            Fortran: ``xl(0:mxpft)``.
        rhol: Leaf reflectance by waveband (1 = vis, 2 = nir) (-).
            Fortran: ``rhol(0:mxpft, numrad)``.
        rhos: Stem reflectance by waveband (1 = vis, 2 = nir) (-).
            Fortran: ``rhos(0:mxpft, numrad)``.
        taul: Leaf transmittance by waveband (1 = vis, 2 = nir) (-).
            Fortran: ``taul(0:mxpft, numrad)``.
        taus: Stem transmittance by waveband (1 = vis, 2 = nir) (-).
            Fortran: ``taus(0:mxpft, numrad)``.
        rootprof_beta: Jackson (1996) rooting distribution parameter (-).
            Fortran: ``rootprof_beta(0:mxpft)``.
        slatop: Specific leaf area at top of canopy, projected area
            basis (m2/gC).
            Fortran: ``slatop(0:mxpft)``.
    """

    dleaf: Array  # (mxpft+1,)
    c3psn: Array  # (mxpft+1,)
    xl: Array  # (mxpft+1,)
    rhol: Array  # (mxpft+1, numrad)
    rhos: Array  # (mxpft+1, numrad)
    taul: Array  # (mxpft+1, numrad)
    taus: Array  # (mxpft+1, numrad)
    rootprof_beta: Array  # (mxpft+1,)
    slatop: Array  # (mxpft+1,)


# ---------------------------------------------------------------------------
# Initialization helpers (mirror Init / InitAllocate / InitRead)
# ---------------------------------------------------------------------------


def InitAllocate() -> pftcon_type:
    """
    Allocate a ``pftcon_type`` instance with all fields set to zero.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 54-68).
    Arrays span indices 0–mxpft (size ``mxpft + 1``) matching the
    Fortran ``allocate(field(0:mxpft))`` convention.

    Returns:
        A zero-initialised :class:`pftcon_type`.
    """
    n = mxpft + 1  # Fortran: allocate(field(0:mxpft))
    z1 = jnp.zeros(n, dtype=jnp.float64)
    z2 = jnp.zeros((n, numrad + 1), dtype=jnp.float64)
    return pftcon_type(
        dleaf=z1,
        c3psn=z1,
        xl=z1,
        rhol=z2,
        rhos=z2,
        taul=z2,
        taus=z2,
        rootprof_beta=z1,
        slatop=z1,
    )


def InitRead(this: pftcon_type) -> pftcon_type:
    """
    Read and initialize vegetation (PFT) parameters.

    Mirrors Fortran subroutine ``InitRead`` (lines 70-240).
    All array assignments follow the Fortran source exactly, including
    the sentinel value ``-999`` for uninitialised PFTs and the optional
    tower-site optical-property override block guarded by
    ``pftcon_val == 1``.

    Fortran 1-based slice ``field(i:j) = v`` maps to
    ``field.at[i:j+1].set(v)`` (Python slice is exclusive at the end).
    2-D Fortran slice ``field(i:j, ib)`` maps to
    ``field.at[i:j+1, ib].set(v)``.

    Args:
        this: Zero-allocated :class:`pftcon_type` from
            :func:`InitAllocate`.

    Returns:
        Fully populated :class:`pftcon_type`.
    """
    n = mxpft + 1

    # ------------------------------------------------------------------
    # Leaf dimension (m) — Fortran lines 133-134
    # ------------------------------------------------------------------
    dleaf = jnp.full(n, -999.0, dtype=jnp.float64)
    dleaf = dleaf.at[1:17].set(0.04)  # dleaf(1:16) = 0.04

    # ------------------------------------------------------------------
    # Photosynthetic pathway (0 = C4, 1 = C3) — Fortran lines 136-140
    # ------------------------------------------------------------------
    c3psn = jnp.full(n, -999.0, dtype=jnp.float64)
    c3psn = c3psn.at[1:14].set(1.0)  # c3psn(1:13)  = 1.
    c3psn = c3psn.at[14:15].set(0.0)  # c3psn(14:14) = 0.
    c3psn = c3psn.at[15:17].set(1.0)  # c3psn(15:16) = 1.

    # ------------------------------------------------------------------
    # Leaf/stem orientation index (-) — Fortran lines 142-151
    # ------------------------------------------------------------------
    xl = jnp.full(n, -999.0, dtype=jnp.float64)
    xl = xl.at[1:4].set(0.01)  # xl(1:3)   = 0.01
    xl = xl.at[4:6].set(0.10)  # xl(4:5)   = 0.10
    xl = xl.at[6:7].set(0.01)  # xl(6:6)   = 0.01
    xl = xl.at[7:9].set(0.25)  # xl(7:8)   = 0.25
    xl = xl.at[9:10].set(0.01)  # xl(9:9)   = 0.01
    xl = xl.at[10:12].set(0.25)  # xl(10:11) = 0.25
    xl = xl.at[12:17].set(-0.30)  # xl(12:16) = -0.30

    # ------------------------------------------------------------------
    # Leaf reflectance: vis and nir — Fortran lines 153-165
    # ------------------------------------------------------------------
    rhol = jnp.full((n, numrad + 1), -999.0, dtype=jnp.float64)
    rhol = rhol.at[1:4, ivis].set(0.07)  # rhol(1:3,  ivis) = 0.07
    rhol = rhol.at[4:9, ivis].set(0.10)  # rhol(4:8,  ivis) = 0.10
    rhol = rhol.at[9:10, ivis].set(0.07)  # rhol(9:9,  ivis) = 0.07
    rhol = rhol.at[10:12, ivis].set(0.10)  # rhol(10:11,ivis) = 0.10
    rhol = rhol.at[12:17, ivis].set(0.11)  # rhol(12:16,ivis) = 0.11

    rhol = rhol.at[1:4, inir].set(0.35)  # rhol(1:3,  inir) = 0.35
    rhol = rhol.at[4:9, inir].set(0.45)  # rhol(4:8,  inir) = 0.45
    rhol = rhol.at[9:10, inir].set(0.35)  # rhol(9:9,  inir) = 0.35
    rhol = rhol.at[10:12, inir].set(0.45)  # rhol(10:11,inir) = 0.45
    rhol = rhol.at[12:17, inir].set(0.35)  # rhol(12:16,inir) = 0.35

    # ------------------------------------------------------------------
    # Stem reflectance: vis and nir — Fortran lines 167-173
    # ------------------------------------------------------------------
    rhos = jnp.full((n, numrad + 1), -999.0, dtype=jnp.float64)
    rhos = rhos.at[1:12, ivis].set(0.16)  # rhos(1:11, ivis) = 0.16
    rhos = rhos.at[12:17, ivis].set(0.31)  # rhos(12:16,ivis) = 0.31

    rhos = rhos.at[1:12, inir].set(0.39)  # rhos(1:11, inir) = 0.39
    rhos = rhos.at[12:17, inir].set(0.53)  # rhos(12:16,inir) = 0.53

    # ------------------------------------------------------------------
    # Leaf transmittance: vis and nir — Fortran lines 175-184
    # ------------------------------------------------------------------
    taul = jnp.full((n, numrad + 1), -999.0, dtype=jnp.float64)
    taul = taul.at[1:17, ivis].set(0.05)  # taul(1:16, ivis) = 0.05

    taul = taul.at[1:4, inir].set(0.10)  # taul(1:3,  inir) = 0.10
    taul = taul.at[4:9, inir].set(0.25)  # taul(4:8,  inir) = 0.25
    taul = taul.at[9:10, inir].set(0.10)  # taul(9:9,  inir) = 0.10
    taul = taul.at[10:12, inir].set(0.25)  # taul(10:11,inir) = 0.25
    taul = taul.at[12:17, inir].set(0.34)  # taul(12:16,inir) = 0.34

    # ------------------------------------------------------------------
    # Stem transmittance: vis and nir — Fortran lines 186-192
    # ------------------------------------------------------------------
    taus = jnp.full((n, numrad + 1), -999.0, dtype=jnp.float64)
    taus = taus.at[1:12, ivis].set(0.001)  # taus(1:11, ivis) = 0.001
    taus = taus.at[12:17, ivis].set(0.12)  # taus(12:16,ivis) = 0.12

    taus = taus.at[1:12, inir].set(0.001)  # taus(1:11, inir) = 0.001
    taus = taus.at[12:17, inir].set(0.25)  # taus(12:16,inir) = 0.25

    # ------------------------------------------------------------------
    # Jackson (1996) rooting distribution parameter (-) — Fortran lines 194-206
    # ------------------------------------------------------------------
    rootprof_beta = jnp.full(n, -999.0, dtype=jnp.float64)
    rootprof_beta = rootprof_beta.at[1:2].set(0.976)  # rootprof_beta(1:1)   = 0.976
    rootprof_beta = rootprof_beta.at[2:4].set(0.943)  # rootprof_beta(2:3)   = 0.943
    rootprof_beta = rootprof_beta.at[4:5].set(0.993)  # rootprof_beta(4:4)   = 0.993
    rootprof_beta = rootprof_beta.at[5:6].set(0.966)  # rootprof_beta(5:5)   = 0.966
    rootprof_beta = rootprof_beta.at[6:7].set(0.993)  # rootprof_beta(6:6)   = 0.993
    rootprof_beta = rootprof_beta.at[7:8].set(0.966)  # rootprof_beta(7:7)   = 0.966
    rootprof_beta = rootprof_beta.at[8:9].set(0.943)  # rootprof_beta(8:8)   = 0.943
    rootprof_beta = rootprof_beta.at[9:11].set(0.964)  # rootprof_beta(9:10)  = 0.964
    rootprof_beta = rootprof_beta.at[11:13].set(0.914)  # rootprof_beta(11:12) = 0.914
    rootprof_beta = rootprof_beta.at[13:17].set(0.943)  # rootprof_beta(13:16) = 0.943

    # ------------------------------------------------------------------
    # Specific leaf area at top of canopy (m2/gC) — Fortran lines 208-225
    # ------------------------------------------------------------------
    slatop = jnp.full(n, -999.0, dtype=jnp.float64)
    slatop = slatop.at[1].set(0.010)
    slatop = slatop.at[2].set(0.008)
    slatop = slatop.at[3].set(0.024)
    slatop = slatop.at[4].set(0.012)
    slatop = slatop.at[5].set(0.012)
    slatop = slatop.at[6].set(0.030)
    slatop = slatop.at[7].set(0.030)
    slatop = slatop.at[8].set(0.030)
    slatop = slatop.at[9].set(0.012)
    slatop = slatop.at[10].set(0.030)
    slatop = slatop.at[11].set(0.030)
    slatop = slatop.at[12].set(0.030)
    slatop = slatop.at[13].set(0.030)
    slatop = slatop.at[14].set(0.030)
    slatop = slatop.at[15].set(0.030)
    slatop = slatop.at[16].set(0.030)

    # ------------------------------------------------------------------
    # Tower site adjustments (pftcon_val == 1) — Fortran lines 227-240
    # Optical property overrides for Majasalmi & Bright (2019)
    # ------------------------------------------------------------------
    if pftcon_val == 1:
        print(f"{iulog}: pftconMod ... using non-default values")

        # xl — Fortran lines 231-232
        #       xl = xl.at[7].set(0.59)                         # CHATS: BDT temperate
        xl = xl.at[7].set(0.53)  # CHATS: walnut

        # rhol ivis — Fortran lines 234-235
        #       rhol = rhol.at[7, ivis].set(0.08)               # CHATS: BDT temperate
        rhol = rhol.at[7, ivis].set(0.06)  # CHATS: walnut

        # rhol inir — Fortran lines 237-238
        #       rhol = rhol.at[7, inir].set(0.42)               # CHATS: BDT temperate
        rhol = rhol.at[7, inir].set(0.42)  # CHATS: walnut

        rhos = rhos.at[7, ivis].set(0.21)  # CHATS: deciduous bark
        rhos = rhos.at[7, inir].set(0.49)  # CHATS: deciduous bark

        # taul ivis — Fortran lines 243-244
        #       taul = taul.at[7, ivis].set(0.06)               # CHATS: BDT temperate
        taul = taul.at[7, ivis].set(0.04)  # CHATS: walnut

        # taul inir — Fortran lines 246-247
        #       taul = taul.at[7, inir].set(0.43)               # CHATS: BDT temperate
        taul = taul.at[7, inir].set(0.43)  # CHATS: walnut

    return this._replace(
        dleaf=dleaf,
        c3psn=c3psn,
        xl=xl,
        rhol=rhol,
        rhos=rhos,
        taul=taul,
        taus=taus,
        rootprof_beta=rootprof_beta,
        slatop=slatop,
    )


def Init() -> pftcon_type:
    """
    Initialize a ``pftcon_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 47-52), which is the
    public entry point that delegates to ``InitAllocate`` then
    ``InitRead``.

    Returns:
        A fully populated :class:`pftcon_type`.
    """
    this = InitAllocate()
    return InitRead(this)


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors Fortran: type(pftcon_type), public :: pftcon)
# ---------------------------------------------------------------------------
pftcon: pftcon_type = Init()
