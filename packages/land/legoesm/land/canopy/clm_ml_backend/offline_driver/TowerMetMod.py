"""
JAX translation of TowerMetMod Fortran module.

Read tower meteorology forcing for offline CLM simulations. Provides
routines to load atmospheric variables from tower netCDF files for the
current and next CLM time steps, derive missing quantities (specific
humidity, longwave radiation), partition solar radiation into direct
and diffuse components per waveband, and populate the CLM and CLMml
forcing containers.

Original Fortran module: TowerMetMod
Fortran lines 1-290
"""

import atexit
from typing import Dict, Tuple

import jax.numpy as jnp
import netCDF4 as nc  # noqa: F401
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_biogeophys.FrictionVelocityMod import frictionvel_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main import GridcellType  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun, handle_err  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.atm2lndType import atm2lnd_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import sb  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varpar import inir, ivis  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.PatchType import patch  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.wateratm2lndBulkType import wateratm2lndbulk_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmdry, mmh2o  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import SatVap  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.offline_driver.TowerDataMod import tower_ht, tower_lat, tower_lon  # noqa: F401

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
# Private: atmospheric CO2 concentration
# ---------------------------------------------------------------------------


def TowerMetCO2() -> float:
    """
    Return the atmospheric CO2 concentration (ppm).

    Mirrors Fortran function ``TowerMetCO2`` (lines 35-48).
    The commented-out alternative value of 367 ppm is preserved.

    Returns:
        Atmospheric CO2 concentration in ppm (umol/mol).
    """
    #   return 367.0    # Fortran line 44 (commented out)
    return 383.0  # Fortran line 45


# ---------------------------------------------------------------------------
# Private: solar radiation partitioning
# ---------------------------------------------------------------------------


def TowerMetSolarRad(
    fsds: float,
) -> Tuple[float, float, float, float]:
    """
    Partition total solar radiation into direct and diffuse components
    for visible and near-infrared wavebands.

    Mirrors Fortran subroutine ``TowerMetSolarRad`` (lines 50-100).

    Total solar radiation is split 50/50 between the visible and
    near-infrared wavebands (Fortran lines 82-83). Within each waveband
    the direct beam fraction is computed from a third-order polynomial
    in the waveband irradiance and clamped to ``[0.01, 0.99]``:

    .. code-block:: none

        fsds_vis = 0.5 * fsds
        rvis = a0 + fsds_vis*(a1 + fsds_vis*(a2 + fsds_vis*a3))
        rvis = clamp(rvis, 0.01, 0.99)

        fsds_nir = 0.5 * fsds
        rnir = b0 + fsds_nir*(b1 + fsds_nir*(b2 + fsds_nir*b3))
        rnir = clamp(rnir, 0.01, 0.99)

    Args:
        fsds: Total incident solar radiation (W/m2).

    Returns:
        Tuple of ``(forc_solad_vis, forc_solai_vis, forc_solad_nir,
        forc_solai_nir)`` in W/m2.
    """
    # Polynomial coefficients — Fortran lines 68-76
    a0 = 0.17639
    a1 = 0.00380
    a2 = -9.0039e-6
    a3 = 8.1351e-9
    b0 = 0.29548
    b1 = 0.00504
    b2 = -1.4957e-5
    b3 = 1.4881e-8

    # Visible waveband — Fortran lines 82-86
    fsds_vis = 0.5 * fsds
    rvis = a0 + fsds_vis * (a1 + fsds_vis * (a2 + fsds_vis * a3))
    rvis = float(jnp.maximum(0.01, jnp.minimum(0.99, rvis)))

    # Near-infrared waveband — Fortran lines 88-90
    fsds_nir = 0.5 * fsds
    rnir = b0 + fsds_nir * (b1 + fsds_nir * (b2 + fsds_nir * b3))
    rnir = float(jnp.maximum(0.01, jnp.minimum(0.99, rnir)))

    # Direct beam and diffuse per waveband — Fortran lines 92-95
    forc_solad_vis = fsds_vis * rvis
    forc_solai_vis = fsds_vis * (1.0 - rvis)
    forc_solad_nir = fsds_nir * rnir
    forc_solai_nir = fsds_nir * (1.0 - rnir)

    return forc_solad_vis, forc_solai_vis, forc_solad_nir, forc_solai_nir


# ---------------------------------------------------------------------------
# Private: atmospheric emissivity
# ---------------------------------------------------------------------------


def TowerMetEmiss(eair: float, tair: float) -> float:
    """
    Compute atmospheric emissivity from vapor pressure and temperature.

    Used when longwave radiation is missing from the tower forcing file.
    Mirrors Fortran function ``TowerMetEmiss`` (lines 102-120).

    .. code-block:: none

        emiss = 0.7 + 5.95e-5 * 0.01 * eair * exp(1500 / tair)

    Args:
        eair: Atmospheric vapor pressure (Pa).
        tair: Atmospheric temperature (K).

    Returns:
        Dimensionless atmospheric emissivity.
    """
    return 0.7 + 5.95e-5 * 0.01 * eair * float(jnp.exp(1500.0 / tair))


# ---------------------------------------------------------------------------
# Public: read forcing for the current CLM time step
# ---------------------------------------------------------------------------


def TowerMetCurr(
    ncfilename: str,
    strt: int,
    it: int,
    begp: int,
    endp: int,
    atm2lnd_inst: atm2lnd_type,
    wateratm2lndbulk_inst: wateratm2lndbulk_type,
    frictionvel_inst: frictionvel_type,
) -> Tuple[atm2lnd_type, wateratm2lndbulk_type, frictionvel_type]:
    """
    Read atmospheric forcing variables for the current CLM time step.

    Mirrors Fortran subroutine ``TowerMetCurr`` (lines 122-215).

    For each patch ``p`` in ``[begp, endp]`` the following CLM forcing
    fields are populated:

    - Grid-cell level: ``forc_u_grc``, ``forc_v_grc`` (= 0),
      ``forc_solai_grc``, ``forc_pco2_grc``, ``forc_po2_grc``.
    - Column level: ``forc_t_downscaled_col``,
      ``forc_q_downscaled_col`` (derived from RH or read directly),
      ``forc_pbot_downscaled_col`` (defaulted to 101325 Pa if missing),
      ``forc_lwrad_downscaled_col`` (computed from emissivity if
      missing), ``forc_rain_downscaled_col``,
      ``forc_snow_downscaled_col`` (= 0),
      ``forc_solad_downscaled_col``.
    - Patch level: ``forc_hgt_u_patch`` (overridden by ``tower_ht``;
      defaulted to 30 m if missing).

    Missing values are indicated by a rounded value of ``-999``
    (``nint(x) == -999``). Specific humidity is derived from relative
    humidity when available; longwave is derived from the Stefan-
    Boltzmann law when missing.

    CO2 and O2 partial pressures are set as:

    .. code-block:: none

        forc_pco2 = (CO2_ppm / 1e6) * forc_pbot
        forc_po2  = 0.209 * forc_pbot

    Tower latitude and longitude are written into ``grc.latdeg`` and
    ``grc.londeg``.

    Args:
        ncfilename: Path to the tower meteorology netCDF file.
        strt: 1-based time slice index into the netCDF file.
        it: Tower site index into ``TowerDataMod`` arrays.
        begp: First patch index.
        endp: Last patch index.
        atm2lnd_inst: Atmosphere-to-land forcing container; most fields
            are updated and returned in a new instance.
        wateratm2lndbulk_inst: Bulk atm-to-land water forcing container;
            ``forc_q``, ``forc_rain``, ``forc_snow`` updated.
        frictionvel_inst: Friction velocity container; ``forc_hgt_u``
            updated.

    Returns:
        Tuple of updated ``(atm2lnd_inst, wateratm2lndbulk_inst,
        frictionvel_inst)``.
    """
    # Unpack output arrays (Fortran associate block, lines 160-173)
    forc_u: Array = atm2lnd_inst.forc_u_grc
    forc_v: Array = atm2lnd_inst.forc_v_grc
    forc_solad: Array = atm2lnd_inst.forc_solad_downscaled_col
    forc_solai: Array = atm2lnd_inst.forc_solai_grc
    forc_pco2: Array = atm2lnd_inst.forc_pco2_grc
    forc_po2: Array = atm2lnd_inst.forc_po2_grc
    forc_t: Array = atm2lnd_inst.forc_t_downscaled_col
    forc_pbot: Array = atm2lnd_inst.forc_pbot_downscaled_col
    forc_lwrad: Array = atm2lnd_inst.forc_lwrad_downscaled_col
    forc_q: Array = wateratm2lndbulk_inst.forc_q_downscaled_col
    forc_rain: Array = wateratm2lndbulk_inst.forc_rain_downscaled_col
    forc_snow: Array = wateratm2lndbulk_inst.forc_snow_downscaled_col
    forc_hgt_u: Array = frictionvel_inst.forc_hgt_u_patch

    # Read raw tower meteorology — Fortran line 176
    zref, tref, rhref, qref, uref, fsds_raw, flds, pref, prect = readTowerMet(ncfilename, strt)

    for p in range(begp, endp + 1):  # Fortran: do p = begp, endp
        c = int(patch.column[p])
        g = int(patch.gridcell[p])
        ci = c  # column arrays indexed directly by c in Fortran

        # ----------------------------------------------------------------
        # Grid-cell level — Fortran lines 182-192
        # ----------------------------------------------------------------
        forc_u = forc_u.at[g].set(uref)
        forc_v = forc_v.at[g].set(0.0)

        # Solar radiation: partition into direct/diffuse per waveband
        # NOTE: forc_solad at column c; forc_solai at gridcell g
        # Fortran lines 188-192
        fsds = float(jnp.maximum(fsds_raw, 0.0))
        solad_vis, solai_vis, solad_nir, solai_nir = TowerMetSolarRad(fsds)
        forc_solad = forc_solad.at[ci, ivis].set(solad_vis)
        forc_solad = forc_solad.at[ci, inir].set(solad_nir)
        forc_solai = forc_solai.at[g, ivis].set(solai_vis)
        forc_solai = forc_solai.at[g, inir].set(solai_nir)

        # ----------------------------------------------------------------
        # Column level — Fortran lines 195-203
        # ----------------------------------------------------------------
        forc_t = forc_t.at[ci].set(tref)
        forc_q = forc_q.at[ci].set(qref)
        forc_pbot = forc_pbot.at[ci].set(pref)
        forc_lwrad = forc_lwrad.at[ci].set(flds)
        forc_rain = forc_rain.at[ci].set(prect)
        forc_snow = forc_snow.at[ci].set(0.0)

        # ----------------------------------------------------------------
        # Patch level: forcing height — Fortran lines 205-211
        # ----------------------------------------------------------------
        forc_hgt_u = forc_hgt_u.at[p].set(zref)
        forc_hgt_u = forc_hgt_u.at[p].set(float(tower_ht[it]))  # override with tower data

        # Default to 30 m if missing — Fortran line 211
        if round(float(forc_hgt_u[p])) == -999:
            forc_hgt_u = forc_hgt_u.at[p].set(30.0)

        # ----------------------------------------------------------------
        # Fill missing values — Fortran lines 214-240
        # ----------------------------------------------------------------

        # Atmospheric pressure default — Fortran lines 215-216
        if round(float(forc_pbot[ci])) == -999:
            forc_pbot = forc_pbot.at[ci].set(101325.0)

        # Specific humidity from RH or directly — Fortran lines 218-227
        forc_rh = rhref
        if round(forc_rh) != -999:
            esat, _ = SatVap(float(forc_t[ci]))
            eair = (forc_rh / 100.0) * esat
            q_val = mmh2o / mmdry * eair / (float(forc_pbot[ci]) - (1.0 - mmh2o / mmdry) * eair)
            forc_q = forc_q.at[ci].set(q_val)
        elif round(float(forc_q[ci])) != -999:
            eair = (
                float(forc_q[ci])
                * float(forc_pbot[ci])
                / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * float(forc_q[ci]))
            )
        else:
            endrun(msg=" TowerMet error: rhref and qref not valid")
            eair = 0.0  # Unreachable; satisfies type checker

        # Longwave radiation from emissivity if missing — Fortran lines 229-233
        if round(float(forc_lwrad[ci])) == -999:
            emiss = TowerMetEmiss(eair, float(forc_t[ci]))
            forc_lwrad = forc_lwrad.at[ci].set(emiss * sb * float(forc_t[ci]) ** 4)

        # CO2 and O2 partial pressures — Fortran lines 235-236
        forc_pco2 = forc_pco2.at[g].set((TowerMetCO2() / 1.0e6) * float(forc_pbot[ci]))
        forc_po2 = forc_po2.at[g].set(0.209 * float(forc_pbot[ci]))

        # ----------------------------------------------------------------
        # Latitude and longitude — Fortran lines 239-240
        # ----------------------------------------------------------------
        # Type narrowing: grc arrays must be initialized at this point
        assert GridcellType.grc.latdeg is not None
        assert GridcellType.grc.londeg is not None
        # Convert to JAX arrays if needed to use .at indexing
        latdeg_jax = jnp.asarray(GridcellType.grc.latdeg)
        londeg_jax = jnp.asarray(GridcellType.grc.londeg)
        GridcellType.grc = GridcellType.grc._replace(
            latdeg=latdeg_jax.at[g].set(float(tower_lat[it])),
            londeg=londeg_jax.at[g].set(float(tower_lon[it])),
        )

    return (
        atm2lnd_inst._replace(
            forc_u_grc=forc_u,
            forc_v_grc=forc_v,
            forc_solad_downscaled_col=forc_solad,
            forc_solai_grc=forc_solai,
            forc_pco2_grc=forc_pco2,
            forc_po2_grc=forc_po2,
            forc_t_downscaled_col=forc_t,
            forc_pbot_downscaled_col=forc_pbot,
            forc_lwrad_downscaled_col=forc_lwrad,
        ),
        wateratm2lndbulk_inst._replace(
            forc_q_downscaled_col=forc_q,
            forc_rain_downscaled_col=forc_rain,
            forc_snow_downscaled_col=forc_snow,
        ),
        frictionvel_inst._replace(
            forc_hgt_u_patch=forc_hgt_u,
        ),
    )


# ---------------------------------------------------------------------------
# Private: read raw meteorology from tower netCDF file
# ---------------------------------------------------------------------------


def readTowerMet(
    ncfilename: str,
    strt: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Read atmospheric forcing variables from a tower netCDF file.

    Mirrors Fortran subroutine ``readTowerMet`` (lines 218-280).

    All variables are read at a single spatial point (tower site) and
    single time slice ``strt``. Optional variables (``FLDS``, ``PSRF``,
    ``RH``, ``QBOT``, ``ZBOT``) are set to ``-999.0`` when absent from
    the file, matching the Fortran fallback assignments.

    Required variables (abort if missing): ``FSDS``, ``PRECTmms``,
    ``TBOT``, ``WIND``.

    Fortran dimension convention note (lines 233-235): Fortran column-
    major ``(lon, lat, time)`` maps to Python/C row-major
    ``(time, lat, lon)``; the single tower gridcell is at index
    ``[t, 0, 0]``.

    Args:
        ncfilename: Path to the tower meteorology netCDF file.
        strt: 1-based time slice index.

    Returns:
        Tuple of scalar floats
        ``(zbot, tbot, rhbot, qbot, ubot, fsdsbot, fldsbot, pbot,
        prect)`` matching the Fortran ``intent(out)`` arguments.
    """
    t = strt - 1  # Convert 1-based Fortran index to 0-based Python

    def _read_optional(ds: nc.Dataset, varname: str) -> float:
        """Return variable value or -999.0 if absent."""
        if varname in ds.variables:
            return float(ds.variables[varname][t, 0, 0])
        return -999.0

    ds = _get_cached_dataset(ncfilename)

    # Optional variables — Fortran lines 238-262
    fldsbot = _read_optional(ds, "FLDS")  # Longwave radiation (W/m2)
    pbot = _read_optional(ds, "PSRF")  # Atmospheric pressure (Pa)
    rhbot = _read_optional(ds, "RH")  # Relative humidity (%)
    qbot = _read_optional(ds, "QBOT")  # Specific humidity (kg/kg)
    zbot = _read_optional(ds, "ZBOT")  # Observational height (m)

    # Required variables — Fortran lines 248-278
    if "FSDS" not in ds.variables:
        handle_err(-1, "FSDS")
    fsdsbot = float(ds.variables["FSDS"][t, 0, 0])  # Solar radiation (W/m2)

    if "PRECTmms" not in ds.variables:
        handle_err(-1, "PRECTmms")
    prect = float(ds.variables["PRECTmms"][t, 0, 0])  # Precipitation (mm/s)

    if "TBOT" not in ds.variables:
        handle_err(-1, "TBOT")
    tbot = float(ds.variables["TBOT"][t, 0, 0])  # Air temperature (K)

    if "WIND" not in ds.variables:
        handle_err(-1, "WIND")
    ubot = float(ds.variables["WIND"][t, 0, 0])  # Wind speed (m/s)

    return zbot, tbot, rhbot, qbot, ubot, fsdsbot, fldsbot, pbot, prect


# ---------------------------------------------------------------------------
# Public: read forcing for the next CLM time step (CLMml 3-point interp)
# ---------------------------------------------------------------------------


def TowerMetNext(
    ncfilename: str,
    strt: int,
    begp: int,
    endp: int,
    mlcanopy_inst: mlcanopy_type,
) -> mlcanopy_type:
    """
    Read atmospheric forcing for the next CLM time step.

    Mirrors Fortran subroutine ``TowerMetNext`` (lines 216-280).

    Required by the 3-point time interpolation of atmospheric forcing
    from the CLM time step to the finer multilayer canopy time step
    (``met_type == 3``). Populates ``*_next_forcing`` fields in the
    ``mlcanopy_type`` container rather than the CLM forcing arrays.

    Missing-value handling is identical to :func:`TowerMetCurr`:
    pressure defaults to 101325 Pa, specific humidity is derived from
    RH when available, and longwave radiation is computed from
    atmospheric emissivity when missing.

    CO2 is stored as ppm (not Pa) in the multilayer canopy container,
    matching the CLMml convention (Fortran line 280).

    Args:
        ncfilename: Path to the tower meteorology netCDF file.
        strt: 1-based time slice index for the **next** CLM time step.
        begp: First patch index.
        endp: Last patch index.
        mlcanopy_inst: Multilayer canopy container; ``*_next_forcing``
            fields are updated and returned in a new instance.

    Returns:
        Updated :class:`mlcanopy_type`.
    """
    # Unpack next-timestep forcing arrays (Fortran associate block, lines 247-256)
    tref_next: Array = mlcanopy_inst.tref_next_forcing  # Air temperature (K)
    qref_next: Array = mlcanopy_inst.qref_next_forcing  # Specific humidity (kg/kg)
    uref_next: Array = mlcanopy_inst.uref_next_forcing  # Wind speed (m/s)
    pref_next: Array = mlcanopy_inst.pref_next_forcing  # Air pressure (Pa)
    co2ref_next: Array = mlcanopy_inst.co2ref_next_forcing  # CO2 (umol/mol = ppm)
    swskyb_next: Array = mlcanopy_inst.swskyb_next_forcing  # Direct beam solar (W/m2)
    swskyd_next: Array = mlcanopy_inst.swskyd_next_forcing  # Diffuse solar (W/m2)
    lwsky_next: Array = mlcanopy_inst.lwsky_next_forcing  # Longwave radiation (W/m2)

    # Read raw tower meteorology — Fortran line 259
    zref, tref, rhref, qref, uref, fsds_raw, flds, pref, prect = readTowerMet(ncfilename, strt)

    for p in range(begp, endp + 1):  # Fortran: do p = begp, endp

        # Direct assignments — Fortran lines 263-267
        uref_next = uref_next.at[p].set(uref)
        tref_next = tref_next.at[p].set(tref)
        qref_next = qref_next.at[p].set(qref)
        pref_next = pref_next.at[p].set(pref)
        lwsky_next = lwsky_next.at[p].set(flds)

        # Solar radiation partition — Fortran lines 269-270
        fsds = float(jnp.maximum(fsds_raw, 0.0))
        solad_vis, solai_vis, solad_nir, solai_nir = TowerMetSolarRad(fsds)
        swskyb_next = swskyb_next.at[p, ivis].set(solad_vis)
        swskyd_next = swskyd_next.at[p, ivis].set(solai_vis)
        swskyb_next = swskyb_next.at[p, inir].set(solad_nir)
        swskyd_next = swskyd_next.at[p, inir].set(solai_nir)

        # Missing value handling (same logic as TowerMetCurr) — Fortran lines 273-285
        if round(float(pref_next[p])) == -999:
            pref_next = pref_next.at[p].set(101325.0)

        forc_rh = rhref
        if round(forc_rh) != -999:
            esat, _ = SatVap(float(tref_next[p]))
            eair = (forc_rh / 100.0) * esat
            q_val = mmh2o / mmdry * eair / (float(pref_next[p]) - (1.0 - mmh2o / mmdry) * eair)
            qref_next = qref_next.at[p].set(q_val)
        elif round(float(qref_next[p])) != -999:
            eair = (
                float(qref_next[p])
                * float(pref_next[p])
                / (mmh2o / mmdry + (1.0 - mmh2o / mmdry) * float(qref_next[p]))
            )
        else:
            endrun(msg=" TowerMetNext error: rhref and qref not valid")
            eair = 0.0  # Unreachable; satisfies type checker

        if round(float(lwsky_next[p])) == -999:
            emiss = TowerMetEmiss(eair, float(tref_next[p]))
            lwsky_next = lwsky_next.at[p].set(emiss * sb * float(tref_next[p]) ** 4)

        # CO2 in ppm for multilayer canopy — Fortran line 280
        co2ref_next = co2ref_next.at[p].set(TowerMetCO2())

    return mlcanopy_inst._replace(
        tref_next_forcing=tref_next,
        qref_next_forcing=qref_next,
        uref_next_forcing=uref_next,
        pref_next_forcing=pref_next,
        co2ref_next_forcing=co2ref_next,
        swskyb_next_forcing=swskyb_next,
        swskyd_next_forcing=swskyd_next,
        lwsky_next_forcing=lwsky_next,
    )
