"""Interface between legoESM and the CLM-ML-JAX multilayer canopy model.

This module provides :func:`compute_clm_ml_canopy_fluxes`, which translates
legoESM's ``AtmToSurface`` forcing into the ``mlcanopy_type`` input
container, calls ``MLCanopyFluxes``, and maps the output back to a
``SurfaceFluxOutput``.

All imports of ``clm_ml_jax`` / ``multilayer_canopy`` are **lazy** (inside
function bodies) so that the rest of legoESM continues to import cleanly
without the optional ``canopy`` extra installed.

Variable-unit conventions
-------------------------
- legoESM ``psi_soil``: matric potential [m], negative for unsaturated
- CLM ``smp_l``:         matric potential [mm], negative for unsaturated
- legoESM ``K_unsat``:  hydraulic conductivity [m/s]
- CLM ``hk_l``:          hydraulic conductivity [mm/s]
- CLM ``swskyb``, ``swskyd``: direct/diffuse SW per waveband [W/m²]
- CLM patch/column/gridcell indices: 1-based (index 0 unused)
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.coupler.coupling_fields import AtmToSurface
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.canopy.state import CanopyState
from legoesm.land.surface_scheme import SurfaceFluxOutput
from legoesm.thermo import saturation_mixing_ratio

if TYPE_CHECKING:
    from legoesm.land.config import MultiLayerLandConfig

# ---------------------------------------------------------------------------
# Module-level CLM initialization guard
# ---------------------------------------------------------------------------

_CLM_INITIALIZED: bool = False
_CLM_STEP_COUNTER: int = 0  # incremented each call; feeds clm_time_manager.itim


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_clm_initialized() -> None:
    """Call CLM phase-1 initialization exactly once."""
    global _CLM_INITIALIZED
    if _CLM_INITIALIZED:
        return
    from clm_src_main.clm_varpar import clm_varpar_init
    from offline_driver import clmSoilOptionMod

    # Use CLM4.5 physics so nlevsoi=10 matches legoESM's default 10-layer soil.
    clmSoilOptionMod.clm_phys = "CLM4_5"
    clm_varpar_init()

    # Initialize MLpftcon and psihat look-up tables.
    from multilayer_canopy.MLCanopyTurbulenceMod import LookupPsihatINI
    from clm_src_main import pftconMod
    from multilayer_canopy import MLpftconMod

    pftconMod.pftcon = pftconMod.Init()
    MLpftconMod.MLpftcon = MLpftconMod.Init()
    LookupPsihatINI()

    # Initialize orbital parameters for year 2000.
    from clm_share.shr_orb_mod import shr_orb_params
    import clm_src_utils.clm_varorb as _varorb

    # shr_orb_params returns (eccen, obliq_deg, mvelp_deg, obliqr, lambm0, mvelpp)
    eccen, _obliq, _mvelp, obliqr, lambm0, mvelpp = shr_orb_params(2000)
    _varorb.eccen = float(eccen)
    _varorb.obliqr = float(obliqr)
    _varorb.mvelpp = float(mvelpp)
    _varorb.lambm0 = float(lambm0)

    _CLM_INITIALIZED = True


def _setup_clm_topology(
    ncol: int,
    lat_deg: np.ndarray,
    dz_soil: np.ndarray,
    z_soil: np.ndarray,
    z_ref: float,
) -> None:
    """Set up CLM module-level topology singletons for ``ncol`` columns.

    Uses a 1:1 mapping: patch index ``p`` = column index ``c`` = gridcell
    index ``g`` = legoESM column ``i + 1`` (1-based).

    Parameters
    ----------
    ncol : int
        Number of legoESM columns.
    lat_deg : np.ndarray
        Latitude of each column [degrees], shape ``(ncol,)``.
    dz_soil : np.ndarray
        Soil layer thicknesses [m], shape ``(n_layers,)``.  Used to
        populate ``col.dz``, ``col.z``, ``col.zi``.
    z_soil : np.ndarray
        Soil layer mid-point depths from surface [m], shape ``(n_layers,)``.
    z_ref : float
        Atmospheric reference height [m] (e.g. ``land_config.z_ref``).
    """
    from clm_src_main import ColumnType as _col_mod
    from clm_src_main import GridcellType as _grc_mod
    from clm_src_main.ColumnType import column_type
    from clm_src_main.PatchType import patch
    from clm_src_main.clm_varpar import nlevsno, nlevgrnd
    from clm_src_main.clm_varcon import ispval

    n_layers = len(dz_soil)

    # ---- patch ----
    # patch arrays: 1-based, index 0 unused
    col_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    gc_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    itype_arr = np.full(ncol + 1, ispval, dtype=np.int32)
    for i in range(ncol):
        p = i + 1  # 1-based patch index
        col_arr[p] = p  # column = patch (1:1)
        gc_arr[p] = p   # gridcell = patch (1:1)
        itype_arr[p] = 13  # C3 non-arctic grass (PFT 13)

    patch.column = jnp.array(col_arr, dtype=jnp.int32)
    patch.gridcell = jnp.array(gc_arr, dtype=jnp.int32)
    patch.itype = jnp.array(itype_arr, dtype=jnp.int32)

    # ---- col ----
    # Shape: (ncol+1, nlevsno+nlevgrnd+1)
    nth = nlevsno + nlevgrnd + 1
    snl_np = np.zeros(ncol + 1, dtype=np.int32)  # no snow
    snl_np[0] = ispval
    dz_np = np.full((ncol + 1, nth), np.nan)
    z_np = np.full((ncol + 1, nth), np.nan)
    zi_np = np.full((ncol + 1, nth), np.nan)
    nbedrock_np = np.full(ncol + 1, ispval, dtype=np.int32)

    # Soil layers in CLM convention:
    # Python index j (1-based) → Fortran layer j → depth from surface
    # nlevsno=5 in CLM4.5, so Python j=6..15 are soil layers 1..10 for CLM4.5
    # BUT in standalone mode snl=0, so Fortran layer 1 = Python index nlevsno+1 = 6
    # _GetCLMVar: j = int(snl[c]) + 1 = 1 → Python index 1 (no snow offset applied)
    # This seems inconsistent... let me use the simpler j=1..n_layers directly.
    # _GetCLMVar line: j = int(_snl_np[c]) + 1 = 1, then t_soisno_col[c, 1]
    # So the convention in standalone CLM-ML-JAX is: j=1 is the top soil layer,
    # regardless of the snow layer allocation. The snl offset is not applied here.

    for i in range(ncol):
        c = i + 1  # 1-based column index
        for j in range(1, n_layers + 1):
            dz_np[c, j] = float(dz_soil[j - 1])
            z_np[c, j] = float(z_soil[j - 1])
        # Interface depths: zi[c, 0] = 0 (surface), zi[c, j] = z[c, j] + dz[c, j]/2
        zi_np[c, 0] = 0.0
        cum = 0.0
        for j in range(1, n_layers + 1):
            cum += float(dz_soil[j - 1])
            zi_np[c, j] = cum
        nbedrock_np[c] = n_layers

    _col_mod.col = column_type(
        snl=jnp.array(snl_np, dtype=jnp.int32),
        dz=jnp.array(dz_np, dtype=jnp.float64),
        z=jnp.array(z_np, dtype=jnp.float64),
        zi=jnp.array(zi_np, dtype=jnp.float64),
        nbedrock=jnp.array(nbedrock_np, dtype=jnp.int32),
    )

    # ---- grc ----
    from clm_src_main.GridcellType import GridcellType as gridcell_type
    latdeg_np = np.full(ncol + 1, 0.0, dtype=np.float64)
    londeg_np = np.full(ncol + 1, 0.0, dtype=np.float64)
    for i in range(ncol):
        g = i + 1
        latdeg_np[g] = float(lat_deg[i]) if i < len(lat_deg) else 0.0
    _grc_mod.grc = gridcell_type(
        latdeg=jnp.array(latdeg_np, dtype=jnp.float64),
        londeg=jnp.array(londeg_np, dtype=jnp.float64),
    )


def _setup_clm_time(dt: float, doy: float, step_count: int) -> None:
    """Configure the CLM time manager for the current legoESM timestep."""
    import clm_src_utils.clm_time_manager as _tm

    _tm.dtstep = int(dt)
    _tm.itim = max(1, step_count)
    # Use year 2000, day-1 as reference; itim * dtstep gives elapsed seconds.
    # get_curr_calday(offset=0) = start_calday + elapsed_days
    # Start on Jan 1 → calday 1. doy=0 → calday 1.
    _tm.start_date_ymd = 20000101
    _tm.start_date_tod = 0
    # curr_date_ymd is updated by get_curr_date(); pre-set as fallback.
    _tm.curr_date_ymd = 20000101
    _tm.curr_date_tod = int((step_count * dt) % 86400)


def _sw_partition(
    sw_down: jnp.ndarray,
    f_vis: float = 0.5,
    f_dir: float = 0.5,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Partition total downwelling SW into direct/diffuse × VIS/NIR bands.

    Parameters
    ----------
    sw_down : jnp.ndarray
        Total downwelling shortwave [W/m²], shape ``(ncol,)``.
    f_vis : float
        Fraction of total SW in the visible band (0.4–0.7 µm).
        Default 0.5 (climatological mean).
    f_dir : float
        Fraction of total SW that is direct beam.
        Default 0.5 (approximate global mean; reduce for overcast).

    Returns
    -------
    swskyb_vis, swskyb_nir : jnp.ndarray
        Direct beam SW in VIS and NIR bands [W/m²].
    swskyd_vis, swskyd_nir : jnp.ndarray
        Diffuse SW in VIS and NIR bands [W/m²].
    """
    vis = f_vis * sw_down
    nir = (1.0 - f_vis) * sw_down
    swskyb_vis = f_dir * vis
    swskyb_nir = f_dir * nir
    swskyd_vis = (1.0 - f_dir) * vis
    swskyd_nir = (1.0 - f_dir) * nir
    return swskyb_vis, swskyb_nir, swskyd_vis, swskyd_nir


def _build_stubs(
    ncol: int,
    forcing: AtmToSurface,
    canopy_config: CLMMLCanopyConfig,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    T_soil_top: jnp.ndarray,
    psi_soil: jnp.ndarray | None,
    theta_soil: jnp.ndarray | None,
    dz_soil: np.ndarray,
    soil_hydraulics: Any | None,
) -> dict[str, Any]:
    """Build minimal CLM input stub objects from legoESM state.

    All stubs are simple Python namespaces; only the fields accessed by
    ``_GetCLMVar``, ``initVerticalStructure``, ``SoilResistance``, and the
    soil relative humidity block are populated.

    Returns
    -------
    dict with keys: atm2lnd, wateratm2lndbulk, surfalb, soilstate,
                    waterstatebulk, temperature, frictionvel, canopystate,
                    energyflux, waterfluxbulk, solarabs, waterdiagnosticbulk
    """
    from clm_src_main.clm_varpar import nlevsoi, ivis, inir, nlevgrnd, nlevsno
    # NOTE: ivis=1, inir=2 in CLM

    np_ = ncol + 1  # 1-based patch dimension

    # ---- SW partitioning ----
    swskyb_vis, swskyb_nir, swskyd_vis, swskyd_nir = _sw_partition(
        forcing.sw_down)

    # shape (np_, numrad+1) = (np_, 3); CLM uses ivis=1, inir=2
    numrad = 2
    swskyb_col = np.zeros((np_, numrad + 1))
    swskyd_grc = np.zeros((np_, numrad + 1))
    swskyb_col_jax = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    swskyd_grc_jax = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)

    for i in range(ncol):
        p = i + 1  # 1-based
        swskyb_col_jax = swskyb_col_jax.at[p, ivis].set(swskyb_vis[i])
        swskyb_col_jax = swskyb_col_jax.at[p, inir].set(swskyb_nir[i])
        swskyd_grc_jax = swskyd_grc_jax.at[p, ivis].set(swskyd_vis[i])
        swskyd_grc_jax = swskyd_grc_jax.at[p, inir].set(swskyd_nir[i])

    # ---- CO2 and O2 partial pressures ----
    # forc_pco2[g] = CO2 partial pressure [Pa]
    # co2_ppmv × p_surface × 1e-6 → Pa
    # In _GetCLMVar: co2ref_cur = forc_pco2[g] / pbot * 1e6 → umol/mol (round-trips)
    co2_pa = forcing.co2_ppmv * forcing.p_surface * 1.0e-6  # Pa
    o2_pa = canopy_config.o2ref * 1.0e-3 * forcing.p_surface  # Pa (mmol/mol × 1e-3)

    # ---- 1-D forcing arrays (1-based) ----
    def _pad1(arr):
        """Pad array to shape (ncol+1,) with 0 at index 0."""
        out = jnp.zeros(np_, dtype=jnp.float64)
        return out.at[1:].set(arr.astype(jnp.float64))

    forc_u = _pad1(forcing.u_lowest)
    forc_v = _pad1(forcing.v_lowest)
    forc_pco2 = _pad1(co2_pa)
    forc_po2 = _pad1(o2_pa)
    forc_solad_col = swskyb_col_jax   # (np_, 3)  direct SW
    forc_solai_grc = swskyd_grc_jax   # (np_, 3)  diffuse SW
    forc_t = _pad1(forcing.T_lowest)
    forc_pbot = _pad1(forcing.p_surface)
    forc_lwrad = _pad1(forcing.lw_down)
    forc_q = _pad1(forcing.q_lowest)
    forc_rain = _pad1(forcing.precip_total - forcing.precip_snow)
    forc_snow = _pad1(forcing.precip_snow)

    # ---- Ground albedo ----
    # Use land_params or config fallback
    if land_params is not None and hasattr(land_params, "albedo_veg") and land_params.albedo_veg is not None:
        alb_arr = land_params.albedo_veg
    else:
        alb_arr = jnp.full(ncol, float(land_config.albedo_land))

    albgrd_col = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    albgri_col = jnp.zeros((np_, numrad + 1), dtype=jnp.float64)
    for i in range(ncol):
        c = i + 1
        albgrd_col = albgrd_col.at[c, ivis].set(float(alb_arr[i]) * 0.9)  # VIS slightly less
        albgrd_col = albgrd_col.at[c, inir].set(float(alb_arr[i]) * 1.1)  # NIR slightly more
        albgri_col = albgri_col.at[c, ivis].set(float(alb_arr[i]) * 0.9)
        albgri_col = albgri_col.at[c, inir].set(float(alb_arr[i]) * 1.1)

    # ---- Soil state ----
    n_layers = len(dz_soil)
    smp_l_col = jnp.zeros((np_, nlevsoi + 1), dtype=jnp.float64)  # [mm]
    hk_l_col = jnp.zeros((np_, nlevsoi + 1), dtype=jnp.float64)   # [mm/s]
    rootfr_patch = jnp.zeros((np_, nlevsoi + 1), dtype=jnp.float64)
    h2osoi_ice = jnp.zeros((np_, nlevsoi + 1), dtype=jnp.float64)  # no ice
    thk_col = jnp.full((np_, nlevsoi + 1), 0.5, dtype=jnp.float64)  # ~0.5 W/m/K

    root_depth = float(land_config.root_depth) if hasattr(land_config, "root_depth") else 1.0
    z_centers = np.array([float(z) for z in dz_soil], dtype=np.float64).cumsum() - dz_soil / 2

    # Root fraction exponential profile (same as legoESM multilayer_land.py)
    root_frac_np = np.exp(-z_centers / root_depth)
    root_frac_np = root_frac_np / root_frac_np.sum()

    for i in range(ncol):
        p = i + 1  # 1-based patch = column
        for j in range(1, min(n_layers, nlevsoi) + 1):
            jl = j - 1  # 0-based legoESM index
            if psi_soil is not None:
                # psi [m] → smp_l [mm]; preserve sign (negative for unsaturated)
                smp_l_col = smp_l_col.at[p, j].set(psi_soil[i, jl] * 1000.0)
            else:
                smp_l_col = smp_l_col.at[p, j].set(-1000.0)  # ~10 kPa suction default

            if soil_hydraulics is not None and psi_soil is not None and theta_soil is not None:
                from legoesm.land.soil_hydraulics import hydraulic_conductivity
                K = hydraulic_conductivity(psi_soil[i, jl], theta_soil[i, jl], soil_hydraulics)
                hk_l_col = hk_l_col.at[p, j].set(float(K) * 1000.0)  # m/s → mm/s
            else:
                hk_l_col = hk_l_col.at[p, j].set(1.0e-4)  # 0.1 mm/s default

            rootfr_patch = rootfr_patch.at[p, j].set(float(root_frac_np[jl]))

    # ---- Soil resistance (bare-ground aerodynamic, rough estimate) ----
    # beta_soil → soil evaporative resistance ~ 100 / beta s/m
    soilresis_col = jnp.full(np_, 100.0, dtype=jnp.float64)

    # ---- Temperature ----
    t_a10_patch = _pad1(forcing.T_lowest)   # 10-day mean ≈ current T (fallback)
    t_soisno_col = jnp.zeros((np_, nlevsoi + 1), dtype=jnp.float64)
    for i in range(ncol):
        c = i + 1
        t_soisno_col = t_soisno_col.at[c, 1].set(float(T_soil_top[i]))

    # ---- canopystate ----
    htop_patch = jnp.zeros(np_, dtype=jnp.float64)
    elai_patch = jnp.zeros(np_, dtype=jnp.float64)
    esai_patch = jnp.zeros(np_, dtype=jnp.float64)
    for i in range(ncol):
        p = i + 1
        htop_v = (float(land_params.htop[i]) if land_params is not None and land_params.htop is not None
                  else 5.0)  # 5 m default canopy height
        lai_v  = (float(land_params.LAI[i]) if land_params is not None and land_params.LAI is not None
                  else 2.0)  # LAI=2 default
        sai_v  = (float(land_params.SAI[i]) if land_params is not None and land_params.SAI is not None
                  else 0.5)
        htop_patch = htop_patch.at[p].set(htop_v)
        elai_patch = elai_patch.at[p].set(lai_v)
        esai_patch = esai_patch.at[p].set(sai_v)

    # ---- frictionvel ----
    forc_hgt_u_patch = jnp.full(np_, float(land_config.z_ref) if hasattr(land_config, "z_ref") else 10.0,
                                  dtype=jnp.float64)

    # ---- Build stub namespaces ----
    atm2lnd = SimpleNamespace(
        forc_u_grc=forc_u,
        forc_v_grc=forc_v,
        forc_pco2_grc=forc_pco2,
        forc_po2_grc=forc_po2,
        forc_solad_downscaled_col=forc_solad_col,
        forc_solai_grc=forc_solai_grc,
        forc_t_downscaled_col=forc_t,
        forc_pbot_downscaled_col=forc_pbot,
        forc_lwrad_downscaled_col=forc_lwrad,
    )
    wateratm2lndbulk = SimpleNamespace(
        forc_q_downscaled_col=forc_q,
        forc_rain_downscaled_col=forc_rain,
        forc_snow_downscaled_col=forc_snow,
    )
    surfalb = SimpleNamespace(
        albgrd_col=albgrd_col,
        albgri_col=albgri_col,
    )
    soilstate = SimpleNamespace(
        smp_l_col=smp_l_col,
        hk_l_col=hk_l_col,
        rootfr_patch=rootfr_patch,
        soilresis_col=soilresis_col,
        thk_col=thk_col,
    )
    waterstatebulk = SimpleNamespace(
        h2osoi_ice_col=h2osoi_ice,
    )
    temperature = SimpleNamespace(
        t_a10_patch=t_a10_patch,
        t_soisno_col=t_soisno_col,
    )
    frictionvel = SimpleNamespace(
        forc_hgt_u_patch=forc_hgt_u_patch,
    )
    canopystate = SimpleNamespace(
        htop_patch=htop_patch,
        elai_patch=elai_patch,
        esai_patch=esai_patch,
    )
    # Output receiver stubs (MLCanopyFluxes writes to these only when mlcan_to_clm=1)
    energyflux = SimpleNamespace(
        taux_patch=jnp.zeros(np_),
        tauy_patch=jnp.zeros(np_),
        eflx_lh_tot_patch=jnp.zeros(np_),
        eflx_sh_tot_patch=jnp.zeros(np_),
        eflx_lwrad_out_patch=jnp.zeros(np_),
    )
    waterfluxbulk = SimpleNamespace(
        qflx_evap_tot_patch=jnp.zeros(np_),
    )
    solarabs = SimpleNamespace(
        fsa_patch=jnp.zeros(np_),
    )
    waterdiagnosticbulk = SimpleNamespace(
        q_ref2m_patch=jnp.zeros(np_),
    )

    return dict(
        atm2lnd=atm2lnd,
        wateratm2lndbulk=wateratm2lndbulk,
        surfalb=surfalb,
        soilstate=soilstate,
        waterstatebulk=waterstatebulk,
        temperature=temperature,
        frictionvel=frictionvel,
        canopystate=canopystate,
        energyflux=energyflux,
        waterfluxbulk=waterfluxbulk,
        solarabs=solarabs,
        waterdiagnosticbulk=waterdiagnosticbulk,
    )


def _init_mlcanopy(ncol: int, stubs: dict, canopy_config: CLMMLCanopyConfig) -> Any:
    """Allocate and cold-start a fresh ``mlcanopy_type`` instance.

    Called only on the first legoESM step (``canopy_state.mlcanopy is None``).
    Sets ``htop``, ``hbot``, and beta-distribution shape parameters for the
    vertical PAD profile, then calls ``init_cold`` for leaf water potential
    and intercepted water initialisation.
    """
    from multilayer_canopy.MLCanopyFluxesType import create_mlcanopy, init_cold
    from multilayer_canopy.MLclm_varpar import nlevmlcan
    from clm_src_main.clm_varcon import spval

    mlcanopy = create_mlcanopy(1, ncol)

    # Set canopy geometry for each patch.
    ztop = mlcanopy.ztop_canopy
    zbot = mlcanopy.zbot_canopy
    # getPADparameters assigns PFT-default beta params only when pbeta < 0.
    # create_mlcanopy initializes them to spval (1e36), so force them negative
    # to trigger the PFT lookup in getPADparameters.
    pbeta_lai = jnp.full_like(mlcanopy.pbeta_lai_canopy, -1.0)
    pbeta_sai = jnp.full_like(mlcanopy.pbeta_sai_canopy, -1.0)

    canopy = stubs["canopystate"]
    for i in range(ncol):
        p = i + 1
        htop_v = float(canopy.htop_patch[p])
        hbot_v = 0.1 * htop_v  # hbot ≈ 10% of htop by default

        ztop = ztop.at[p].set(htop_v)
        zbot = zbot.at[p].set(hbot_v)

    mlcanopy = mlcanopy._replace(
        ztop_canopy=ztop,
        zbot_canopy=zbot,
        pbeta_lai_canopy=pbeta_lai,
        pbeta_sai_canopy=pbeta_sai,
    )

    # Call init_cold to set initial leaf water potential and intercepted water.
    # Signature: init_cold(mlcanopy_inst, begp, endp)
    mlcanopy = init_cold(mlcanopy, 1, ncol)

    return mlcanopy


def _extract_surface_fluxes(
    mlcanopy: Any,
    ncol: int,
    forcing: AtmToSurface,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
) -> SurfaceFluxOutput:
    """Map ``mlcanopy_type`` output fields to a ``SurfaceFluxOutput``.

    Sign conventions
    ----------------
    - ``shflx_canopy``: sensible heat flux [W/m²], positive into atmosphere
    - ``lhflx_canopy``: latent heat flux [W/m²], positive into atmosphere
    - ``gsoi_soil``:    ground heat flux [W/m²], positive into soil (CLM)
    - ``lwup_canopy``:  upwelling LW [W/m²]
    - ``gppveg_canopy``: GPP [µmol CO₂/m²/s]
    """
    from clm_src_main.clm_varpar import ivis, inir

    # Per-column arrays → (ncol,)
    shflx = jnp.stack([mlcanopy.shflx_canopy[i + 1] for i in range(ncol)])
    lhflx = jnp.stack([mlcanopy.lhflx_canopy[i + 1] for i in range(ncol)])
    G_soil = jnp.stack([mlcanopy.gsoi_soil[i + 1] for i in range(ncol)])
    lw_up = jnp.stack([mlcanopy.lwup_canopy[i + 1] for i in range(ncol)])
    # GPP: µmol CO2/m²/s → gC/m²/s (1 µmol CO2 = 12e-6 gC)
    gpp_umol = jnp.stack([mlcanopy.gppveg_canopy[i + 1] for i in range(ncol)])
    gpp = gpp_umol * 12.0e-6  # µmol/m²/s → gC/m²/s

    # Friction velocity → momentum flux components
    ustar = jnp.stack([mlcanopy.ustar_canopy[i + 1] for i in range(ncol)])
    rho_a = forcing.p_surface / (287.058 * forcing.T_lowest)  # kg/m³
    tau_total = rho_a * ustar ** 2  # [N/m²]
    # Split tau between x and y proportional to wind components
    u = forcing.u_lowest
    v = forcing.v_lowest
    ws = jnp.sqrt(u ** 2 + v ** 2 + 1.0e-6)
    tau_x = tau_total * u / ws
    tau_y = tau_total * v / ws

    # Net radiation: absorbed SW + net LW
    # rnet_canopy is the full-column net radiation (canopy + soil).
    # swveg_canopy and swsoi_soil are indexed (patch, band) where band
    # indices ivis=1, inir=2; index 0 holds spval and must be skipped.
    rnet = jnp.stack([mlcanopy.rnet_canopy[i + 1] for i in range(ncol)])
    sw_net = jnp.stack(
        [
            mlcanopy.swveg_canopy[i + 1, ivis]
            + mlcanopy.swveg_canopy[i + 1, inir]
            + mlcanopy.swsoi_soil[i + 1, ivis]
            + mlcanopy.swsoi_soil[i + 1, inir]
            for i in range(ncol)
        ]
    )
    lw_net = rnet - sw_net

    # Canopy-mean albedo: 1 - SW_absorbed / SW_down
    sw_down = forcing.sw_down
    albedo = jnp.where(sw_down > 1.0, 1.0 - sw_net / (sw_down + 1.0e-6), 0.15)

    # Surface temperature: use soil surface T (updated by CLM-ML)
    T_surface = jnp.stack([mlcanopy.tg_soil[i + 1] for i in range(ncol)])

    # q_surface: saturation at soil surface T
    q_surface = saturation_mixing_ratio(T_surface, forcing.p_surface)

    # Emissivity and roughness from canopy
    emissivity = jnp.full(ncol, float(land_config.emissivity_land)
                          if hasattr(land_config, "emissivity_land") else 0.96)
    z0 = jnp.stack([mlcanopy.z0m_canopy[i + 1] for i in range(ncol)])

    return SurfaceFluxOutput(
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        sw_net=sw_net,
        lw_net=lw_net,
        lw_up=lw_up,
        G_soil=G_soil,
        T_surface=T_surface,
        q_surface=q_surface,
        albedo=albedo,
        emissivity=emissivity,
        z0=z0,
        gpp=gpp,
        # CLM-ML doesn't return stomatal ratio in the same way; set to 1
        stomatal_ratio=jnp.ones(ncol),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def compute_clm_ml_canopy_fluxes(
    T_soil_top: jnp.ndarray,
    forcing: AtmToSurface,
    canopy_config: CLMMLCanopyConfig,
    land_config: "MultiLayerLandConfig",
    land_params: Any | None,
    w_frac_rz: jnp.ndarray,
    wind_speed: jnp.ndarray,
    canopy_state: CanopyState | None,
    dt: float,
    T_soil: jnp.ndarray | None = None,
    psi_soil: jnp.ndarray | None = None,
    theta_soil: jnp.ndarray | None = None,
    lat: jnp.ndarray | None = None,
    doy: float = 0.0,
) -> tuple[SurfaceFluxOutput, CanopyState]:
    """Compute canopy fluxes via the CLM-ML-JAX multilayer canopy model.

    Parameters
    ----------
    T_soil_top:
        Soil surface temperature [K], shape ``(ncol,)``.
    forcing:
        Atmospheric forcing from the coupler.
    canopy_config:
        Static CLM-ML-JAX configuration (``CLMMLCanopyConfig``).
    land_config:
        Parent ``MultiLayerLandConfig`` (for fallback soil parameters).
    land_params:
        Per-column ``LandSurfaceParams`` (or ``None`` for config defaults).
        Must supply ``LAI``, ``SAI``, ``htop`` for the canopy scheme.
    w_frac_rz:
        Root-zone soil-moisture stress fraction [0–1], shape ``(ncol,)``.
    wind_speed:
        Scalar wind speed [m/s], shape ``(ncol,)``.
    canopy_state:
        Previous-step canopy state.  ``None`` or ``mlcanopy is None``
        triggers cold-start allocation.
    dt:
        Timestep [s].
    T_soil:
        Full soil temperature profile [K], shape ``(ncol, n_layers)``.
    psi_soil:
        Soil matric potential [m], shape ``(ncol, n_layers)``.
    theta_soil:
        Volumetric water content [m³/m³], shape ``(ncol, n_layers)``.
    lat:
        Latitude [degrees], shape ``(ncol,)`` or ``None``.
    doy:
        Day of year (0-based float).

    Returns
    -------
    surface_out : SurfaceFluxOutput
        Canopy surface energy balance fluxes for the legoESM coupler.
    new_canopy_state : CanopyState
        Updated prognostic state to carry forward to the next step.
    """
    global _CLM_STEP_COUNTER

    # ---- Phase-1 CLM initialization (once) ----
    _ensure_clm_initialized()

    # Lazy imports of CLM-ML-JAX entry point
    from multilayer_canopy.MLCanopyFluxesMod import MLCanopyFluxes
    from clm_src_main.decompMod import bounds_type
    from legoesm.land.soil_grid import make_soil_grid

    ncol = T_soil_top.shape[0]
    _CLM_STEP_COUNTER += 1

    # ---- Build soil grid data ----
    grid = make_soil_grid(land_config.soil_grid)
    dz_soil = np.array(grid.dz, dtype=np.float64)     # (n_layers,)
    z_soil = np.array(grid.z_node, dtype=np.float64)   # (n_layers,)
    n_layers = len(dz_soil)

    # ---- Latitude array ----
    if lat is not None:
        lat_deg = np.array(lat, dtype=np.float64)
    else:
        lat_deg = np.zeros(ncol, dtype=np.float64)

    # ---- CLM global state setup ----
    _setup_clm_topology(ncol, lat_deg, dz_soil, z_soil,
                        float(land_config.z_ref) if hasattr(land_config, "z_ref") else 10.0)
    _setup_clm_time(dt, doy, _CLM_STEP_COUNTER)

    # ---- Build stub CLM instances ----
    soil_hyd = getattr(land_config, "hydraulics", None)
    stubs = _build_stubs(
        ncol, forcing, canopy_config, land_config, land_params,
        T_soil_top, psi_soil, theta_soil, dz_soil, soil_hyd,
    )

    # ---- Allocate / retrieve mlcanopy_type ----
    if canopy_state is None or canopy_state.mlcanopy is None:
        mlcanopy = _init_mlcanopy(ncol, stubs, canopy_config)
    else:
        mlcanopy = canopy_state.mlcanopy

    # ---- Decomposition bounds ----
    bounds = bounds_type(begg=1, endg=ncol, begl=1, endl=ncol,
                         begc=1, endc=ncol, begp=1, endp=ncol)

    # ---- filter_exposedvegp: all columns (1-based) ----
    filter_exposedvegp = list(range(1, ncol + 1))
    num_exposedvegp = ncol

    # ---- Set MLclm_varctl global settings ----
    import multilayer_canopy.MLclm_varctl as _ml_ctl
    _ml_ctl.runge_kutta_type = canopy_config.runge_kutta_type
    _ml_ctl.met_type = canopy_config.met_type
    # dtime_ml: sub-step length. Must divide dt evenly.
    _ml_ctl.dtime_ml = dt / max(1, canopy_config.num_ml_steps)
    _ml_ctl.mlcan_to_clm = 0  # we read output directly from mlcanopy_type

    # ---- Call MLCanopyFluxes ----
    mlcanopy_new = MLCanopyFluxes(
        bounds=bounds,
        num_exposedvegp=num_exposedvegp,
        filter_exposedvegp=filter_exposedvegp,
        atm2lnd_inst=stubs["atm2lnd"],
        canopystate_inst=stubs["canopystate"],
        soilstate_inst=stubs["soilstate"],
        temperature_inst=stubs["temperature"],
        waterstatebulk_inst=stubs["waterstatebulk"],
        waterfluxbulk_inst=stubs["waterfluxbulk"],
        energyflux_inst=stubs["energyflux"],
        frictionvel_inst=stubs["frictionvel"],
        surfalb_inst=stubs["surfalb"],
        solarabs_inst=stubs["solarabs"],
        mlcanopy_inst=mlcanopy,
        wateratm2lndbulk_inst=stubs["wateratm2lndbulk"],
        waterdiagnosticbulk_inst=stubs["waterdiagnosticbulk"],
        _o2ref_py=float(canopy_config.o2ref),
    )

    # ---- Extract SurfaceFluxOutput ----
    surface_out = _extract_surface_fluxes(
        mlcanopy_new, ncol, forcing, land_config, land_params)

    return surface_out, CanopyState(mlcanopy=mlcanopy_new)
