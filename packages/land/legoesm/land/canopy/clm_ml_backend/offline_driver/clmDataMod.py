"""
JAX translation of clmDataMod Fortran module.

Read CLM forcing data for tower sites. Provides routines to load leaf
and stem area indices, volumetric soil moisture, and optional soil
moisture adjustment factors from CLM netCDF history files, then derive
liquid water and ice content for each soil layer.

Original Fortran module: clmDataMod
Fortran lines 1-250
"""

import atexit
from typing import Dict

import jax.numpy as jnp
import netCDF4 as nc  # noqa: F401
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.CanopyStateType import canopystate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SoilStateType import soilstate_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.SurfaceAlbedoType import surfalb_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.WaterStateBulkType import waterstatebulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import handle_err  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import denh2o, spval  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import nlevgrnd, nlevsoi  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.ColumnType import col  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.clmSoilOptionMod import clm_phys, nlev_soil_adjust  # noqa: F401

# ---------------------------------------------------------------------------
# NetCDF dataset cache
# ---------------------------------------------------------------------------

_NC_DATASET_CACHE: Dict[str, nc.Dataset] = {}


def _get_cached_dataset(ncfilename: str) -> nc.Dataset:
    """Return an open netCDF dataset handle, opening it once per file path."""
    ds = _NC_DATASET_CACHE.get(ncfilename)
    if ds is None:
        ds = nc.Dataset(ncfilename, "r")
        _NC_DATASET_CACHE[ncfilename] = ds
    return ds


def close_cached_datasets() -> None:
    """Close and clear all cached netCDF datasets for this module."""
    for ds in _NC_DATASET_CACHE.values():
        try:
            ds.close()
        except Exception:
            pass
    _NC_DATASET_CACHE.clear()


atexit.register(close_cached_datasets)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def clmData(
    fin_clm: str,
    fin_soil_adjust: str,
    strt: int,
    begp: int,
    endp: int,
    begc: int,
    endc: int,
    soilstate_inst: soilstate_type,
    waterstatebulk_inst: waterstatebulk_type,
    canopystate_inst: canopystate_type,
    surfalb_inst: surfalb_type,
) -> tuple[waterstatebulk_type, canopystate_type, surfalb_type]:
    """
    Read variables from a CLM netCDF history file for the current time
    step and populate canopy state and bulk water state containers.

    Mirrors Fortran subroutine ``clmData`` (lines 33-115).

    The following quantities are derived:

    - ``elai_patch``, ``esai_patch``: leaf/stem area index broadcast
      to all patches from a single gridcell value.
    - ``h2osoi_vol_col``: volumetric soil water, read from the CLM
      history file and optionally multiplied by a moisture adjustment
      factor. CLM4.5 files provide ``nlevgrnd`` layers; CLM5.0 files
      provide ``nlevsoi`` layers (bedrock layers set to zero). For
      CLM5.0, values are capped at ``watsat`` before deriving liquid
      water.
    - ``h2osoi_liq_col``: liquid water ``= h2osoi_vol * dz * denh2o``.
    - ``h2osoi_ice_col``: set to zero for all layers.

    Args:
        fin_clm: Path to the CLM netCDF history file.
        fin_soil_adjust: Path to the soil moisture adjustment netCDF
            file (ignored when ``nlev_soil_adjust == 0``).
        strt: 1-based time slice index into the netCDF file.
        begp: First patch index.
        endp: Last patch index.
        begc: First column index.
        endc: Last column index.
        soilstate_inst: Soil state container supplying ``watsat_col``
            (read-only).
        waterstatebulk_inst: Bulk water state container; soil water
            arrays are updated and returned in a new instance.
        canopystate_inst: Canopy state container; ``elai_patch`` and
            ``esai_patch`` are updated and returned in a new instance.
        surfalb_inst: Surface albedo container (passed through
            unchanged; included for interface parity with Fortran).

    Returns:
        Tuple of updated ``(waterstatebulk_inst, canopystate_inst,
        surfalb_inst)``.
    """
    # Unpack inputs (Fortran associate block, lines 72-84)
    dz = col.dz  # Soil layer thickness (m)
    nbedrock = col.nbedrock  # Depth to bedrock index
    watsat = soilstate_inst.watsat_col  # Porosity

    elai = canopystate_inst.elai_patch  # Leaf area index (m2/m2)
    esai = canopystate_inst.esai_patch  # Stem area index (m2/m2)
    h2osoi_vol = waterstatebulk_inst.h2osoi_vol_col  # Volumetric soil water (m3/m3)
    h2osoi_liq = waterstatebulk_inst.h2osoi_liq_col  # Liquid water (kg H2O/m2)
    h2osoi_ice = waterstatebulk_inst.h2osoi_ice_col  # Ice lens (kg H2O/m2)

    # ------------------------------------------------------------------
    # Read CLM LAI and SAI for current time step — Fortran lines 87-91
    # ------------------------------------------------------------------
    elai_loc, esai_loc = readCLMveg(fin_clm, strt)

    # Broadcast single gridcell value to all patches — Fortran lines 89-91
    for p in range(begp, endp + 1):  # Fortran: do p = begp, endp
        elai = elai.at[p].set(float(elai_loc))
        esai = esai.at[p].set(float(esai_loc))

    # ------------------------------------------------------------------
    # Read volumetric soil water for current time step — Fortran lines 93-94
    # ------------------------------------------------------------------
    h2osoi_clm45, h2osoi_clm50 = readCLMsoil(fin_clm, strt)

    # ------------------------------------------------------------------
    # Distribute soil water to all columns — Fortran lines 96-113
    # ------------------------------------------------------------------
    for c in range(begc, endc + 1):  # Fortran: do c = begc, endc
        ci = c - begc  # 0-based column offset

        if clm_phys == "CLM4_5":
            # CLM4.5: nlevgrnd layers — Fortran lines 98-100
            for j in range(1, nlevgrnd + 1):
                h2osoi_vol = h2osoi_vol.at[ci, j].set(float(h2osoi_clm45[j - 1]))

        elif clm_phys == "CLM5_0":
            # CLM5.0: nlevsoi soil layers — Fortran lines 101-105
            for j in range(1, nlevsoi + 1):
                h2osoi_vol = h2osoi_vol.at[ci, j].set(float(h2osoi_clm50[j - 1]))
            # Bedrock layers set to zero — Fortran lines 106-108
            for j in range(nlevsoi + 1, nlevgrnd + 1):
                h2osoi_vol = h2osoi_vol.at[ci, j].set(0.0)

        # Overwrite CLM soil moisture with adjustment factor — Fortran lines 110-115
        if nlev_soil_adjust > 0:
            h2osoi_factor = readSoilWatFactor(fin_soil_adjust, strt)
            for j in range(1, nlev_soil_adjust + 1):
                h2osoi_vol = h2osoi_vol.at[ci, j].set(float(h2osoi_vol[ci, j]) * h2osoi_factor)

        # Cap soil moisture at porosity for CLM5.0 — Fortran lines 117-121
        if clm_phys == "CLM5_0":
            nb = int(nbedrock[c])
            for j in range(1, nb + 1):  # Fortran: do j = 1, nbedrock(c)
                h2osoi_vol = h2osoi_vol.at[ci, j].set(
                    jnp.minimum(float(h2osoi_vol[ci, j]), float(watsat[c, j]))
                )

        # Set liquid water and ice — Fortran lines 123-126
        for j in range(1, nlevgrnd + 1):  # Fortran: do j = 1, nlevgrnd
            h2osoi_liq = h2osoi_liq.at[ci, j - 1].set(
                float(h2osoi_vol[ci, j]) * float(dz[c, j]) * denh2o
            )
            h2osoi_ice = h2osoi_ice.at[ci, j - 1].set(0.0)

    return (
        waterstatebulk_inst._replace(
            h2osoi_vol_col=h2osoi_vol,
            h2osoi_liq_col=h2osoi_liq,
            h2osoi_ice_col=h2osoi_ice,
        ),
        canopystate_inst._replace(
            elai_patch=elai,
            esai_patch=esai,
        ),
        surfalb_inst,
    )


# ---------------------------------------------------------------------------
# Private: read LAI and SAI from CLM netCDF history file
# ---------------------------------------------------------------------------


def readCLMveg(
    ncfilename: str,
    strt: int,
) -> tuple[float, float]:
    """
    Read leaf and stem area index from a CLM netCDF history file.

    Mirrors Fortran subroutine ``readCLMveg`` (lines 117-165).

    Reads a single spatial gridcell at time slice ``strt`` from the
    ``ELAI`` and ``ESAI`` variables. In the Fortran the arrays are
    dimensioned ``(1,1,1)`` (lon=1, lat=1, time=1 for a tower site);
    here scalar floats are returned directly.

    The netCDF convention note from the Fortran source (lines 136-138)
    applies: Fortran column-major ``(lon, lat, time)`` maps to Python/
    C row-major ``(time, lat, lon)``; indexing is adjusted accordingly.

    Args:
        ncfilename: Path to the CLM netCDF history file.
        strt: 1-based time slice index.

    Returns:
        Tuple of ``(elai_mod, esai_mod)`` as scalar floats.
    """
    # Fortran: start2=(1, strt), count2=(1,1) — lon=1 gridcell, 1 timestep
    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    ds = _get_cached_dataset(ncfilename)

    # ELAI(nlndgrid, ntime) — Fortran lines 142-146
    elai_mod = float(ds.variables["ELAI"][t, 0])

    # ESAI(nlndgrid, ntime) — Fortran lines 148-152
    esai_mod = float(ds.variables["ESAI"][t, 0])

    return elai_mod, esai_mod


# ---------------------------------------------------------------------------
# Private: read volumetric soil water from CLM netCDF history file
# ---------------------------------------------------------------------------


def readCLMsoil(
    ncfilename: str,
    strt: int,
) -> tuple[Array, Array]:
    """
    Read volumetric soil water from a CLM netCDF history file.

    Mirrors Fortran subroutine ``readCLMsoil`` (lines 167-220).

    Reads the ``H2OSOI`` variable for the current time slice.
    Depending on ``clm_phys``:

    - ``'CLM4_5'``: reads ``nlevgrnd`` layers into ``h2osoi_clm45``
      (shape ``(nlevgrnd,)``); ``h2osoi_clm50`` is set to ``spval``.
    - ``'CLM5_0'``: reads ``nlevsoi`` layers into ``h2osoi_clm50``
      (shape ``(nlevsoi,)``); ``h2osoi_clm45`` is set to ``spval``.

    Both arrays are initialised to ``spval`` before reading, matching
    the Fortran ``h2osoi_clm45(:,:,:) = spval`` initialisation
    (Fortran lines 181-182).

    Args:
        ncfilename: Path to the CLM netCDF history file.
        strt: 1-based time slice index.

    Returns:
        Tuple of ``(h2osoi_clm45, h2osoi_clm50)`` as 1-D JAX arrays
        of shapes ``(nlevgrnd,)`` and ``(nlevsoi,)`` respectively.
    """
    # Initialize to spval — Fortran lines 181-182
    h2osoi_clm45 = jnp.full((nlevgrnd,), spval, dtype=jnp.float64)
    h2osoi_clm50 = jnp.full((nlevsoi,), spval, dtype=jnp.float64)

    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    ds = _get_cached_dataset(ncfilename)
    if clm_phys == "CLM4_5":
        # H2OSOI(nlndgrid, nlevgrnd, ntime) — Fortran lines 195-201
        # Fortran start3=(1,1,strt), count3=(1,nlevgrnd,1)
        data = ds.variables["H2OSOI"][t, :nlevgrnd, 0]  # (nlevgrnd,)
        h2osoi_clm45 = jnp.array(data, dtype=jnp.float64)

    elif clm_phys == "CLM5_0":
        # H2OSOI(nlndgrid, nlevsoi, ntime) — Fortran lines 205-211
        # Fortran start3=(1,1,strt), count3=(1,nlevsoi,1)
        data = ds.variables["H2OSOI"][t, :nlevsoi, 0]  # (nlevsoi,)
        h2osoi_clm50 = jnp.array(data, dtype=jnp.float64)

    return h2osoi_clm45, h2osoi_clm50


# ---------------------------------------------------------------------------
# Private: read soil moisture adjustment factor from netCDF file
# ---------------------------------------------------------------------------


def readSoilWatFactor(
    ncfilename: str,
    strt: int,
) -> float:
    """
    Read soil moisture adjustment factor from a netCDF file.

    Mirrors Fortran subroutine ``readSoilWatFactor`` (lines 222-248).

    Reads a single scalar ``FACTOR`` value at time slice ``strt``
    from a one-dimensional time series variable.

    Args:
        ncfilename: Path to the soil moisture adjustment netCDF file.
        strt: 1-based time slice index.

    Returns:
        Scalar soil moisture adjustment factor.
    """
    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    # factor(ntime) — Fortran lines 240-244
    ds = _get_cached_dataset(ncfilename)
    h2osoi_factor_loc = float(ds.variables["FACTOR"][t])

    return h2osoi_factor_loc
