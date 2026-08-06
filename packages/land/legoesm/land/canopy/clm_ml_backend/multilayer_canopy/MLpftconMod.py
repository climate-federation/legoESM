"""
JAX translation of MLpftconMod Fortran module.

Vegetation (PFT) parameters unique to the multilayer canopy model.
Provides data structures and initialization routines for PFT-level
parameters used in photosynthesis, stomatal conductance, plant
hydraulics, and radiative transfer calculations.

Original Fortran module: MLpftconMod
Fortran lines 1-260
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import mxpft  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_utils.spmdMod import masterproc  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import pftcon_val  # noqa: F401

# ---------------------------------------------------------------------------
# PFT index reference (Fortran lines 96-161)
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


class MLpftcon_type(NamedTuple):
    """
    Vegetation (PFT) parameters for the multilayer canopy model,
    mirroring Fortran ``MLpftcon_type`` (lines 19-44).

    All arrays have shape ``(mxpft + 1,)`` matching the Fortran
    allocation ``(0:mxpft)``, where index 0 is ``not_vegetated`` and
    indices 1–mxpft correspond to the PFTs listed above.

    Attributes:
        vcmaxpft: Maximum carboxylation rate at 25 °C (umol/m2/s).
            Fortran: ``vcmaxpft(0:mxpft)``.
        gplant_SPA: Stem (xylem-to-leaf) hydraulic conductance
            (mmol H2O/m2 leaf area/s/MPa).
            Fortran: ``gplant_SPA(0:mxpft)``.
        capac_SPA: Plant capacitance (mmol H2O/m2 leaf area/MPa).
            Fortran: ``capac_SPA(0:mxpft)``.
        iota_SPA: Stomatal water-use efficiency (umol CO2/mol H2O).
            Fortran: ``iota_SPA(0:mxpft)``.
        root_radius_SPA: Fine root radius (m).
            Fortran: ``root_radius_SPA(0:mxpft)``.
        root_density_SPA: Fine root density (g biomass/m3 root).
            Fortran: ``root_density_SPA(0:mxpft)``.
        root_resist_SPA: Hydraulic resistivity of root tissue
            (MPa·s·g/mmol H2O).
            Fortran: ``root_resist_SPA(0:mxpft)``.
        gsmin_SPA: Minimum stomatal conductance (mol H2O/m2/s).
            Fortran: ``gsmin_SPA(0:mxpft)``.
        g0_BB: Ball-Berry minimum leaf conductance (mol H2O/m2/s).
            Fortran: ``g0_BB(0:mxpft)``.
        g1_BB: Ball-Berry slope of conductance-photosynthesis
            relationship (-).
            Fortran: ``g1_BB(0:mxpft)``.
        g0_MED: Medlyn minimum leaf conductance (mol H2O/m2/s).
            Fortran: ``g0_MED(0:mxpft)``.
        g1_MED: Medlyn slope of conductance-photosynthesis
            relationship (-).
            Fortran: ``g1_MED(0:mxpft)``.
        psi50_gs: Leaf water potential at which 50 % of stomatal
            conductance is lost (MPa).
            Fortran: ``psi50_gs(0:mxpft)``.
        shape_gs: Shape parameter for stomatal conductance in relation
            to leaf water potential (-).
            Fortran: ``shape_gs(0:mxpft)``.
        emleaf: Leaf emissivity (-).
            Fortran: ``emleaf(0:mxpft)``.
        clump_fac: Foliage clumping index (-).
            Fortran: ``clump_fac(0:mxpft)``.
    """

    vcmaxpft: Array  # (mxpft+1,)
    gplant_SPA: Array  # (mxpft+1,)
    capac_SPA: Array  # (mxpft+1,)
    iota_SPA: Array  # (mxpft+1,)
    root_radius_SPA: Array  # (mxpft+1,)
    root_density_SPA: Array  # (mxpft+1,)
    root_resist_SPA: Array  # (mxpft+1,)
    gsmin_SPA: Array  # (mxpft+1,)
    g0_BB: Array  # (mxpft+1,)
    g1_BB: Array  # (mxpft+1,)
    g0_MED: Array  # (mxpft+1,)
    g1_MED: Array  # (mxpft+1,)
    psi50_gs: Array  # (mxpft+1,)
    shape_gs: Array  # (mxpft+1,)
    emleaf: Array  # (mxpft+1,)
    clump_fac: Array  # (mxpft+1,)


# ---------------------------------------------------------------------------
# Initialization helpers (mirror Init / InitAllocate / InitRead)
# ---------------------------------------------------------------------------


def InitAllocate() -> MLpftcon_type:
    """
    Allocate a ``MLpftcon_type`` instance with all fields set to zero.

    Mirrors Fortran subroutine ``InitAllocate`` (lines 63-83).
    Arrays span indices 0–mxpft (size ``mxpft + 1``) matching the
    Fortran ``allocate(field(0:mxpft))`` convention.

    Returns:
        A zero-initialised :class:`MLpftcon_type`.
    """
    n = mxpft + 1  # Fortran: allocate(field(0:mxpft))
    z = jnp.zeros(n, dtype=jnp.float64)
    return MLpftcon_type(
        vcmaxpft=z,
        gplant_SPA=z,
        capac_SPA=z,
        iota_SPA=z,
        root_radius_SPA=z,
        root_density_SPA=z,
        root_resist_SPA=z,
        gsmin_SPA=z,
        g0_BB=z,
        g1_BB=z,
        g0_MED=z,
        g1_MED=z,
        psi50_gs=z,
        shape_gs=z,
        emleaf=z,
        clump_fac=z,
    )


def InitRead(this: MLpftcon_type) -> MLpftcon_type:
    """
    Read and initialise vegetation (PFT) parameters.

    Mirrors Fortran subroutine ``InitRead`` (lines 85-245).
    All array assignments follow the Fortran source exactly, including
    the sentinel value ``-999`` for uninitialised PFTs and the optional
    tower-site override block guarded by ``pftcon_val == 1``.

    Fortran 1-based slice ``field(i:j) = v`` maps to
    ``field.at[i:j+1].set(v)`` (Python slice is exclusive at the end).

    Args:
        this: Zero-allocated :class:`MLpftcon_type` from
            :func:`InitAllocate`.

    Returns:
        Fully populated :class:`MLpftcon_type`.
    """
    if masterproc:
        print(f"{iulog}: Attempting to initialize MLpftcon .....")

    # ------------------------------------------------------------------
    # vcmaxpft: Maximum carboxylation rate at 25 C (umol/m2/s)
    # Fortran lines 169-186
    # ------------------------------------------------------------------
    vcmaxpft = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    vcmaxpft = vcmaxpft.at[1].set(62.5)
    vcmaxpft = vcmaxpft.at[2].set(62.5)
    vcmaxpft = vcmaxpft.at[3].set(39.1)
    vcmaxpft = vcmaxpft.at[4].set(41.0)
    vcmaxpft = vcmaxpft.at[5].set(61.4)
    vcmaxpft = vcmaxpft.at[6].set(41.0)
    vcmaxpft = vcmaxpft.at[7].set(57.7)
    vcmaxpft = vcmaxpft.at[8].set(57.7)
    vcmaxpft = vcmaxpft.at[9].set(61.7)
    vcmaxpft = vcmaxpft.at[10].set(54.0)
    vcmaxpft = vcmaxpft.at[11].set(54.0)
    vcmaxpft = vcmaxpft.at[12].set(78.2)
    vcmaxpft = vcmaxpft.at[13].set(78.2)
    vcmaxpft = vcmaxpft.at[14].set(51.6)
    vcmaxpft = vcmaxpft.at[15].set(100.7)
    vcmaxpft = vcmaxpft.at[16].set(100.7)

    # ------------------------------------------------------------------
    # Plant hydraulics
    # Fortran lines 188-196
    # ------------------------------------------------------------------
    gplant_SPA = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    gplant_SPA = gplant_SPA.at[1:17].set(4.0)  # gplant_SPA(1:16) = 4.

    capac_SPA = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    capac_SPA = capac_SPA.at[1:12].set(2500.0)  # capac_SPA(1:11) = 2500.
    capac_SPA = capac_SPA.at[12:17].set(500.0)  # capac_SPA(12:16) = 500.

    # ------------------------------------------------------------------
    # Stomatal optimization: iota_SPA (umol CO2/mol H2O)
    # Fortran lines 198-204
    # ------------------------------------------------------------------
    iota_SPA = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    iota_SPA = iota_SPA.at[1:2].set(750.0)  # iota_SPA(1:1)  = 750.
    iota_SPA = iota_SPA.at[2:4].set(1500.0)  # iota_SPA(2:3)  = 1500.
    iota_SPA = iota_SPA.at[4:5].set(500.0)  # iota_SPA(4:4)  = 500.
    iota_SPA = iota_SPA.at[5:17].set(750.0)  # iota_SPA(5:16) = 750.

    # ------------------------------------------------------------------
    # Root hydraulics
    # Fortran lines 206-213
    # ------------------------------------------------------------------
    root_radius_SPA = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    root_radius_SPA = root_radius_SPA.at[1:17].set(0.29e-3)  # root_radius_SPA(1:16) = 0.29e-03

    root_density_SPA = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    root_density_SPA = root_density_SPA.at[1:17].set(0.31e6)  # root_density_SPA(1:16) = 0.31e06

    root_resist_SPA = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    root_resist_SPA = root_resist_SPA.at[1:17].set(25.0)  # root_resist_SPA(1:16) = 25.

    # ------------------------------------------------------------------
    # Minimum stomatal conductance (mol H2O/m2/s)
    # Fortran lines 215-217
    # ------------------------------------------------------------------
    gsmin_SPA = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    gsmin_SPA = gsmin_SPA.at[1:17].set(0.002)  # gsmin_SPA(1:16) = 0.002

    # ------------------------------------------------------------------
    # Ball-Berry stomatal conductance parameters
    # Fortran lines 219-228
    # ------------------------------------------------------------------
    g0_BB = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    g0_BB = g0_BB.at[1:14].set(0.01)  # g0_BB(1:13)   = 0.01
    g0_BB = g0_BB.at[14:15].set(0.04)  # g0_BB(14:14)  = 0.04
    g0_BB = g0_BB.at[15:17].set(0.01)  # g0_BB(15:16)  = 0.01

    g1_BB = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    g1_BB = g1_BB.at[1:14].set(9.0)  # g1_BB(1:13)  = 9.
    g1_BB = g1_BB.at[14:15].set(4.0)  # g1_BB(14:14) = 4.
    g1_BB = g1_BB.at[15:17].set(9.0)  # g1_BB(15:16) = 9.

    # ------------------------------------------------------------------
    # Medlyn stomatal conductance parameters
    # Fortran lines 230-250
    # ------------------------------------------------------------------
    g0_MED = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    g0_MED = g0_MED.at[1:17].set(0.0001)  # g0_MED(1:16) = 0.0001

    g1_MED = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    g1_MED = g1_MED.at[1].set(2.35)
    g1_MED = g1_MED.at[2].set(2.35)
    g1_MED = g1_MED.at[3].set(2.35)
    g1_MED = g1_MED.at[4].set(4.12)
    g1_MED = g1_MED.at[5].set(4.12)
    g1_MED = g1_MED.at[6].set(4.45)
    g1_MED = g1_MED.at[7].set(4.45)
    g1_MED = g1_MED.at[8].set(4.45)
    g1_MED = g1_MED.at[9].set(4.70)
    g1_MED = g1_MED.at[10].set(4.70)
    g1_MED = g1_MED.at[11].set(4.70)
    g1_MED = g1_MED.at[12].set(2.22)
    g1_MED = g1_MED.at[13].set(5.25)
    g1_MED = g1_MED.at[14].set(1.62)
    g1_MED = g1_MED.at[15].set(5.79)
    g1_MED = g1_MED.at[16].set(5.79)

    # ------------------------------------------------------------------
    # Leaf water potential parameters
    # Fortran lines 252-257
    # ------------------------------------------------------------------
    psi50_gs = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    psi50_gs = psi50_gs.at[1:17].set(-2.3)  # psi50_gs(1:16) = -2.3

    shape_gs = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    shape_gs = shape_gs.at[1:17].set(40.0)  # shape_gs(1:16) = 40.

    # ------------------------------------------------------------------
    # Leaf emissivity (-)
    # Fortran lines 259-261
    # ------------------------------------------------------------------
    emleaf = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    emleaf = emleaf.at[1:17].set(0.98)  # emleaf(1:16) = 0.98

    # ------------------------------------------------------------------
    # Foliage clumping index (-)
    # Fortran lines 263-265
    # ------------------------------------------------------------------
    clump_fac = jnp.full(mxpft + 1, -999.0, dtype=jnp.float64)
    clump_fac = clump_fac.at[1:17].set(1.0)  # clump_fac(1:16) = 1.

    # ------------------------------------------------------------------
    # Tower site adjustments (pftcon_val == 1)
    # Fortran lines 267-280
    # ------------------------------------------------------------------
    if pftcon_val == 1:
        if masterproc:
            print(f"{iulog}: MLpftcon ... using non-default values")

        vcmaxpft = vcmaxpft.at[7].set(125.0)  # CHATS: Rosati et al. (2006)
        gplant_SPA = gplant_SPA.at[7].set(7.0)  # CHATS: Tyree et al. (1993)
        iota_SPA = iota_SPA.at[7].set(375.0)  # CHATS: Rosati et al. (2006)
        root_resist_SPA = root_resist_SPA.at[7].set(14.0)  # CHATS: Tyree et al. (1994)
        psi50_gs = psi50_gs.at[7].set(-1.60)  # CHATS: SPA-walnut
        shape_gs = shape_gs.at[7].set(40.0)  # CHATS: SPA-walnut

    if masterproc:
        print(f"{iulog}: Successfuly initialized MLpftcon")

    return this._replace(
        vcmaxpft=vcmaxpft,
        gplant_SPA=gplant_SPA,
        capac_SPA=capac_SPA,
        iota_SPA=iota_SPA,
        root_radius_SPA=root_radius_SPA,
        root_density_SPA=root_density_SPA,
        root_resist_SPA=root_resist_SPA,
        gsmin_SPA=gsmin_SPA,
        g0_BB=g0_BB,
        g1_BB=g1_BB,
        g0_MED=g0_MED,
        g1_MED=g1_MED,
        psi50_gs=psi50_gs,
        shape_gs=shape_gs,
        emleaf=emleaf,
        clump_fac=clump_fac,
    )


def Init() -> MLpftcon_type:
    """
    Initialise a ``MLpftcon_type`` instance.

    Mirrors Fortran subroutine ``Init`` (lines 57-62), which is the
    public entry point that delegates to ``InitAllocate`` then
    ``InitRead``.

    Returns:
        A fully populated :class:`MLpftcon_type`.
    """
    this = InitAllocate()
    return InitRead(this)


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors Fortran: type(MLpftcon_type), public :: MLpftcon)
# ---------------------------------------------------------------------------
MLpftcon: MLpftcon_type = Init()


# ---------------------------------------------------------------------------
# Parameter injection factory (for gradient-based optimization)
# ---------------------------------------------------------------------------


def make_pft_params(theta_dict: dict, base: MLpftcon_type | None = None) -> MLpftcon_type:
    """Build an MLpftcon_type with optimizable parameters injected from theta_dict.

    Intended use: parameter optimization experiments.  All fields not listed
    in theta_dict are taken from ``base`` (defaults to the module-level
    ``MLpftcon`` singleton).

    theta_dict keys are ``(field_name, pft_index)`` tuples or plain field names
    (applied to all PFTs, replacing the full array).

    Supported injection patterns::

        # Single PFT override (most common — CHATS7 PFT 7):
        make_pft_params({("vcmaxpft", 7): 125.0, ("iota_SPA", 7): 375.0})

        # Full-array replacement (replace all PFTs at once):
        make_pft_params({"vcmaxpft": jnp.full(mxpft+1, 100.0)})

        # Scale factor (multiply current value by alpha):
        # Use module-global mutation directly; see diags/optimize_params.py.

    Parameters
    ----------
    theta_dict : dict mapping (field, pft_idx) or field_name to new values
    base       : MLpftcon_type to start from (defaults to MLpftcon singleton)

    Returns
    -------
    New MLpftcon_type with injected values (fully differentiable if theta_dict
    values are JAX traced arrays; ``MLpftcon.vcmaxpft[pft]`` uses dynamic gather
    so vcmaxpft is traceable through CanopyNitrogenProfileMod).
    """
    if base is None:
        base = MLpftcon
    kwargs = {}
    for key, val in theta_dict.items():
        if isinstance(key, tuple):
            field_name, pft_idx = key
            current = getattr(base, field_name)
            kwargs[field_name] = current.at[pft_idx].set(val)
        else:
            kwargs[key] = val
    return base._replace(**kwargs)
