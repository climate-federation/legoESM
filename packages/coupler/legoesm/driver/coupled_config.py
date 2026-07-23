"""Configuration for the fully coupled Earth System Model.

Defines ``CoupledConfig`` which selects ocean, land, and carbon cycle
modes, and preset factories for standard configurations.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm.ocean.simple_ocean import SimpleOceanConfig
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.land.config import LandConfig, MultiLayerLandConfig
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.richards import RichardsConfig  # noqa: F401 (used in docstring)


class CoupledConfig(NamedTuple):
    """Configuration for the coupled ESM driver.

    Selects ocean mode, land complexity, and carbon cycle options.
    Each field maps to an existing component — no new physics is
    introduced; this is purely wiring configuration.

    Parameters
    ----------
    ocean_mode : str
        ``"slab"`` (default), ``"two_layer"``, or ``"dynamic"`` (3D ocean).
    ocean_config : SimpleOceanConfig
        Slab/two-layer ocean parameters (ignored when ``ocean_mode="dynamic"``).
    land_mode : str
        ``"none"`` (aquaplanet), ``"slab"`` (bucket), or ``"multilayer"``
        (Richards' equation soil hydrology).
    land_config : LandConfig or MultiLayerLandConfig
        Land parameters.  Type should match ``land_mode``:
        ``LandConfig`` for ``"slab"``, ``MultiLayerLandConfig`` for
        ``"multilayer"``.  Ignored when ``land_mode="none"``.
    use_pft : bool
        Use PFT-weighted land parameters instead of scalar defaults.
    carbon_active : bool
        Master switch for the carbon cycle.
    carbon_land : str
        Land carbon scheme: ``"none"``, ``"differland"`` (DALEC 6-pool),
        ``"seasonal"`` (prescribed NEE cycle).
    carbon_ocean : bool
        Enable ocean biogeochemistry CO2 exchange.
    co2_tracer : bool
        Enable prognostic atmospheric CO2 transport.
    co2_ppmv_init : float
        Initial atmospheric CO2 concentration [ppmv].
    f_land_mode : str
        ``"zero"`` (aquaplanet), ``"analytical"`` (default land mask),
        ``"file"`` (external land mask).
    coupling_dt : float
        Surface coupling interval [s].
    """
    # Ocean.  ``ocean_mode`` ∈ {"slab", "two_layer", "fixed", "dynamic"}.
    #   slab/two_layer/fixed  -> thermodynamic SimpleOceanConfig (make_ocean).
    #   dynamic               -> the prognostic 3D LatLonCGridOceanModel
    #     (T,S,u,v,eta; KPP/implicit vertical mixing, split-explicit barotropic,
    #     EOS, baroclinic dynamics) stepped by the coupler on a SHARED lat-lon
    #     grid with the atmosphere (no cross-grid remap).  Pair with a
    #     LatLonCGridOceanConfig and the ocean_nlev/ocean_dt_s/ocean_H_max_m
    #     fields below; the coupled cold-start uses the OMIP-validated stable
    #     stack (rk3 momentum + implicit_cn barotropic + implicit vmix + smc03
    #     PGF), forced in CoupledESMDriver._init_ocean.  See
    #     docs/ocean/coupled_3d_ocean_plan.md.
    ocean_mode: str = "slab"
    ocean_config: SimpleOceanConfig | LatLonCGridOceanConfig = SimpleOceanConfig(
        mode="slab")
    # Dynamic (3D) ocean knobs — consumed only when ocean_mode=="dynamic"
    # (defaults keep slab/two_layer runs byte-identical).
    ocean_nlev: int = 20            # vertical levels (20 = OMIP production)
    ocean_dt_s: float = 300.0       # ocean SUBSTEP dt [s]; the coupler substeps
    # the 3D ocean at this dt within each coupling_dt (NEVER step the ocean at
    # coupling_dt=3600 s — it violates the legoESM ocean CFL; OMIP runs 300 s
    # at 1°).
    ocean_H_max_m: float = 5500.0   # max ocean depth [m]
    # Dynamic-ocean initial condition (ocean_mode=="dynamic" only).
    #   "rest" (default): idealized rest state (exponential T, uniform S,
    #           all-ocean aquaplanet bathy) — byte-identical Phase-1 behaviour.
    #   "woa":  WOA18 reanalysis T/S climatology (observed stratification,
    #           avoids the long cold-start spin-up) + a WOA-derived realistic
    #           land mask (continents).  Requires woa_t_path / woa_s_path.
    ocean_ic: str = "rest"
    woa_t_path: str | None = None   # WOA18 temperature file (ocean_ic=="woa")
    woa_s_path: str | None = None   # WOA18 salinity    file (ocean_ic=="woa")
    # NEMO eORCA tripole mesh_mask file. When the dynamic-ocean grid is a
    # tripole (active-fold) geometry — a DIFFERENT grid from the lat-lon
    # atmosphere, coupled via the Phase-2 cross-grid conservative remap — the
    # land mask + bathymetry are read from THIS file (the same file the tripole
    # geometry was built from), via the OMIP-validated cold-start recipe
    # (adcroft PGF + implicit_cn barotropic + implicit vmix + C_smag_lap=0.33).
    # Required when ocean_mode=="dynamic" and the ocean grid is tripole.
    tripole_mesh_path: str | None = None
    # Land
    land_mode: str = "slab"
    land_config: LandConfig | MultiLayerLandConfig = LandConfig()
    use_pft: bool = False
    # Source of the spatial land parameters when use_pft: "analytical" (latitude
    # bands) or "clm" (the CLM reference surfdata: real PFT map + reference soil).
    land_param_source: str = "analytical"
    # Transient land-use cover (opt-in; requires land_param_source="clm").  When
    # True with land_cover_surfdata set, the CLM path's PFT-weighted VEGETATION
    # params re-weight each segment from the transient legoesm_surfdata cover
    # (LUH2/HYDE/...); the CLM soil (hydraulics/thermal) stays frozen.  Empty
    # land_cover_surfdata → static cover (byte-identical to a normal run).
    transient_land_cover: bool = False
    land_cover_surfdata: str = ""
    # Coupled DIURNAL surface model for multilayer land (default ON): use the
    # physical Monin-Obukhov surface exchange + Farquhar photosynthesis-stomata
    # coupling (vs a constant bulk coefficient + soil-only beta).  Well-posed in the
    # coupled model because the atmosphere supplies a resolved diurnal cycle.
    land_diurnal_surface: bool = True
    # Sub-grid elevation-band snow for multilayer land (default OFF): re-partition
    # precipitation phase and melt over sub-grid elevation bands from the CLM
    # ``STD_ELEV`` map, so a warm cell keeps bright snow on its cold high fractions
    # (fixes the high-elevation / perennial-snow warm-albedo bias).  Requires
    # ``land_mode="multilayer"`` and ``land_param_source="clm"`` (the band elevations
    # come from the CLM surface map).  See ``legoesm.land.snow_bands``.
    land_elev_bands: bool = False
    # Carbon cycle
    carbon_active: bool = False
    carbon_land: str = "none"
    carbon_ocean: bool = False
    co2_tracer: bool = False
    co2_ppmv_init: float = 415.0
    # Spun-up land carbon IC (finidat).  Path to a ``global_carbon_ic.npz`` built
    # by ``scripts/data/build_global_carbon_ic.py``.  When set (and the land is
    # multilayer + differland carbon), the coupled run INGESTS the seeded 8-pool
    # per-cell CarbonState + the per-cell permafrost ``phi`` INSTEAD of cold-
    # starting carbon: it starts at the mapped equilibrium and MAINTAINS the
    # seeded permafrost SOC (``phi`` -> ``make_coupler`` permafrost protection).
    # ``""`` (default) => cold-start (byte-identical).  Requires a run grid
    # matching the finidat (STRICT grid-match; see
    # ``legoesm.land.carbon.global_init.load_finidat_carbon_ic``).
    carbon_ic_path: str = ""
    # Tile fractions.  f_land_mode ∈ {"zero", "analytical", "from_ocean"}.
    #   "from_ocean": f_land = 1 - (dynamic-ocean WOA-derived ocean mask), so
    #   the atmosphere land fraction and the 3D-ocean wet mask come from one
    #   source on the shared grid (no flux leak).  Requires ocean_mode=
    #   "dynamic" + ocean_ic="woa".
    f_land_mode: str = "analytical"
    # Coupling
    coupling_dt: float = 3600.0
    # Dynamic surface → radiation feedback.  When True, the coupler's
    # tile-blended surface albedo and skin temperature (which carry the
    # sea-ice albedo feedback, zenith ocean albedo, snow brightening, and the
    # ice/land prognostic skin temperature) are fed back to the atmosphere's
    # radiation each segment instead of the frozen config scalars.  Default
    # False keeps existing coupled runs byte-identical; enable it together with
    # the dynamic surface schemes (IceConfig.temp_dependent_albedo,
    # OceanAlbedoConfig.method='zenith', LandConfig.snow_albedo_feedback), whose
    # effect on radiation is otherwise silently dropped.
    couple_surface_radiation: bool = False

    # SHARED air-sea surface fluxes (close the air-sea heat+water budget).
    # When True, the coupler's tile-blended surface sensible / latent heat flux
    # (computed with ITS bulk scheme, q_sfc = 0.98*q_sat mixing ratio, and the
    # ocean-tile C_H/C_E) is fed back to the ATMOSPHERE's surface tendency each
    # segment, so the heat + water leaving the atmosphere EQUALS what the
    # coupler feeds the ocean -- the model becomes flux-coupled, not just
    # SST-coupled, and the air-sea budget closes.  Default False keeps existing
    # coupled runs byte-identical (the atmosphere computes its own bulk surface
    # fluxes, independent of the ocean-driving fluxes); recommended ON after a
    # coupled validation run.  Incompatible with a turbulence / unified-physics
    # scheme that owns surface exchange (the pipeline raises) -- use the bulk-BL
    # surface path.  Lagged one coupling segment (explicit coupling), exactly
    # like ``couple_surface_radiation``.
    couple_surface_fluxes: bool = False

    # Warm-start the land soil at the atmosphere's lat-structured near-surface
    # air temperature (t=0) instead of the uniform 280 K default.  The uniform
    # default makes tropical land soil start ~18 K too cold, and the (slow)
    # multilayer soil takes months to spin up — dragging the global near-surface
    # air temperature down through the long transient.  Default False keeps
    # existing coupled runs byte-identical; recommended ON for a faster, more
    # realistic land spin-up.
    warm_start_soil: bool = False

    # 3D-ocean WOA restoring (Newtonian / Haney) timescales [days].  Consumed
    # ONLY when ocean_mode=="dynamic" + ocean_ic=="woa": the surface-layer T/S
    # are relaxed toward the WOA-climatology initial state with these timescales
    # after each coupling step.  A short coupled spin-up from realistic IC will
    # NOT reach equilibrium (the dry cold-start atmosphere radiates away the
    # warm WOA ocean's heat -> a cold collapse a free 3D ocean cannot buffer
    # like a slab); restoring anchors the surface near observed climatology
    # while the atmosphere equilibrates (the standard coupled spin-up protocol).
    # The restoring flux represents the missing ocean heat transport / flux
    # correction during spin-up.  0.0 => OFF (no relaxation) => byte-identical,
    # so every existing dynamic-ocean run is unchanged.  The convective-gustiness
    # air-sea fix (CouplerConfig.gustiness_w_zi) is necessary but NOT sufficient
    # alone (measured: 15d still drifts -92 K/yr); restoring is the complement.
    ocean_restore_sst_tau_days: float = 0.0   # surface T relaxation [days]
    ocean_restore_sss_tau_days: float = 0.0   # surface S relaxation [days]


# ============================================================================
# Preset factories
# ============================================================================

def preset_aquaplanet(**overrides) -> CoupledConfig:
    """Slab ocean everywhere, no land, no carbon."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="none",
        f_land_mode="zero",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_simple(**overrides) -> CoupledConfig:
    """Slab ocean + slab bucket land, no carbon."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="slab",
        land_config=LandConfig(),
        f_land_mode="analytical",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


#: CoupledConfig.ocean_mode selects the coupled driver's ocean BRANCH
#: (thermodynamic vs dynamic); within the thermodynamic branch the physics run
#: is make_ocean(ocean_config), so the two fields must agree through this map:
#: each simple-ocean mode onto the ocean_mode domain {slab, two_layer}.
#: (The old comment called ocean_mode "never dispatched" — false: it gates
#: dynamic-vs-thermodynamic in CoupledESMDriver._init_ocean.)
_OCEAN_MODE_LABEL = {"fixed": "slab", "slab": "slab", "two_layer": "two_layer"}


def ocean_mode_label(simple_mode: str) -> str | None:
    """Public accessor: the ``CoupledConfig.ocean_mode`` value that a
    ``SimpleOceanConfig.mode`` maps onto (None for unknown modes).  Used by
    the coupled driver's consistency guard — do not import the private map."""
    return _OCEAN_MODE_LABEL.get(simple_mode)


def preset_complexity(level, **overrides) -> CoupledConfig:
    """Build a ``CoupledConfig`` from a model-wide ``ModelComplexity`` level.

    The model-wide complexity dial reaching the coupled driver: a user writes
    ``preset_complexity("idealized")`` instead of hand-assembling
    ``ocean_config`` + ``land_mode`` + ``land_config``.  Maps the level's ocean
    and land rungs (``components.model_complexity_rungs``) onto the coupled
    config:

    ===============  =================================  =========================
    level            ocean                              land
    ===============  =================================  =========================
    ``idealized``    fixed-SST slab                     slab bucket
    ``intermediate`` slab mixed layer                   multi-layer Richards column
    ===============  =================================  =========================

    Scoped to the simple-ocean levels: the coupled driver's ocean is ALWAYS the
    simple ocean (``_init_ocean`` calls ``make_ocean(ocean_config)``;
    ``CoupledConfig.ocean_config`` is a ``SimpleOceanConfig``), so ``full`` (a
    prognostic 3-D ``OceanModel``) is NOT representable here and raises — build a
    full-3-D ocean run via ``driver.component_factory.resolve_model_complexity``
    + the ocean component factory instead.  ``overrides`` are applied last (e.g.
    ``carbon_active=True``, ``f_land_mode=``).
    """
    from legoesm.components import (
        LandComplexity,
        ModelComplexity,
        OceanComplexity,
        model_complexity_rungs,
        ocean_simple_mode,
    )

    rungs = model_complexity_rungs(level)
    if rungs.ocean is OceanComplexity.FULL_3D:
        raise ValueError(
            f"{ModelComplexity(level)!s} complexity has a full-3D ocean that is not "
            "representable in CoupledConfig (which holds a SimpleOceanConfig; the "
            "coupled driver's ocean is always make_ocean).  Build a full-3D ocean "
            "run via driver.component_factory.resolve_model_complexity + the ocean "
            "component factory instead."
        )

    ocean_mode = ocean_simple_mode(rungs.ocean)  # "fixed" | "slab" | "two_layer"
    ocean_config = SimpleOceanConfig(mode=ocean_mode)
    if rungs.land is LandComplexity.MULTILAYER:
        land_mode, land_config = "multilayer", MultiLayerLandConfig()
    else:
        land_mode, land_config = "slab", LandConfig()

    defaults = dict(
        ocean_mode=_OCEAN_MODE_LABEL[ocean_mode],
        ocean_config=ocean_config,
        land_mode=land_mode,
        land_config=land_config,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_pft(**overrides) -> CoupledConfig:
    """Slab ocean + slab land with PFT-weighted parameters."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="slab",
        land_config=LandConfig(),
        use_pft=True,
        f_land_mode="analytical",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_richards(**overrides) -> CoupledConfig:
    """Slab ocean + multi-layer Richards' equation land, no carbon."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="multilayer",
        land_config=MultiLayerLandConfig(
            # Richards equation is always active in multilayer mode
        ),
        f_land_mode="analytical",
        carbon_active=False,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_slab_carbon(**overrides) -> CoupledConfig:
    """Slab ocean + Richards' land + DifferLand carbon + atm CO2 tracer."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="multilayer",
        land_config=MultiLayerLandConfig(
            # Richards equation is always active in multilayer mode
            carbon=CarbonConfig(scheme="differland"),
        ),
        f_land_mode="analytical",
        carbon_active=True,
        carbon_land="differland",
        co2_tracer=True,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


def preset_full_coupled(**overrides) -> CoupledConfig:
    """Slab ocean + Richards' land + land & ocean carbon + atm CO2."""
    defaults = dict(
        ocean_mode="slab",
        ocean_config=SimpleOceanConfig(mode="slab", h_mix=50.0),
        land_mode="multilayer",
        land_config=MultiLayerLandConfig(
            # Richards equation is always active in multilayer mode
            carbon=CarbonConfig(scheme="differland"),
        ),
        f_land_mode="analytical",
        carbon_active=True,
        carbon_land="differland",
        carbon_ocean=True,
        co2_tracer=True,
    )
    defaults.update(overrides)
    return CoupledConfig(**defaults)


PRESETS = {
    "aquaplanet": preset_aquaplanet,
    "slab_simple": preset_slab_simple,
    "slab_pft": preset_slab_pft,
    "slab_richards": preset_slab_richards,
    "slab_carbon": preset_slab_carbon,
    "full_coupled": preset_full_coupled,
}
