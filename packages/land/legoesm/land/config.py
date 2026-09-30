"""Land model configuration."""

from __future__ import annotations

from typing import Any, NamedTuple

from legoesm import constants
from legoesm.land.carbon.config import CarbonConfig
from legoesm.land.stomata import StomataConfig
from legoesm.land.snow_bands import ElevationSnowBandConfig
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.soil_thermal import SoilThermalConfig
from legoesm.land.topmodel_runoff import TopmodelConfig
from legoesm.land.richards import RichardsConfig
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig
from legoesm.surface_albedo import LandAlbedoConfig


__param_spec__ = {
    "LandConfig": {
        "scheme_key": "land.slab",
        "excluded": {
            "beta_min": "numerics: soil-water stress floor",
            "snow_melt_rate": "unreachable: degree-day melt coefficient read ONLY in the Q_net-is-None fallback of update_snow (snow_budget.py). The two callers that thread THIS config field (slab_land, multilayer_land) both pass an energy-limited Q_net, so jax.grad w.r.t. the config field is identically zero. (The coupler snow-albedo-feedback diagnostic in physics_pipeline.py does call update_snow with Q_net=None, but with the FUNCTION DEFAULT snow_melt_rate, not this config field, so it does not make the field trainable.)",
            "z_ref": "convention: reference height [m]",
        },
        "params": {
            "C_soil": {"units": "J m-3 K-1", "bounds": (660000.0, 6000000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "Cd_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "Ch_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "W_max": {"units": "kg m-2", "bounds": (49.5, 450.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "legoESM land surface", "shape": None},
            "K_infiltration": {"units": "m/s", "bounds": (1e-7, 1e-4), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "Rawls Brakensiek Miller 1983 (Green-Ampt K_s)", "shape": None},
            "infil_suction_boost": {"units": "1", "bounds": (0.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Mein & Larson 1973 (Green-Ampt psi_f/L_f)", "shape": None},
            "albedo_land": {"units": "1", "bounds": (0.066, 0.6), "tunable_tier": 1, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "d_soil": {"units": "m", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "emissivity_land": {"units": "1", "bounds": (0.3168, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "z0_land": {"units": "m", "bounds": (0.0165, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
        },
    },
    "MultiLayerLandConfig": {
        "scheme_key": "land.multilayer",
        "excluded": {
            "beta_min": "numerics: soil-water stress floor",
            "snow_melt_rate": "unreachable: degree-day melt coefficient read ONLY in the Q_net-is-None fallback of update_snow (snow_budget.py). The two callers that thread THIS config field (slab_land, multilayer_land) both pass an energy-limited Q_net, so jax.grad w.r.t. the config field is identically zero. (The coupler snow-albedo-feedback diagnostic in physics_pipeline.py does call update_snow with Q_net=None, but with the FUNCTION DEFAULT snow_melt_rate, not this config field, so it does not make the field trainable.)",
            "z_ref": "convention: reference height [m]",
        },
        "params": {
            "Cd_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "Ch_land": {"units": "1", "bounds": (0.00099, 0.009), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
            "albedo_land": {"units": "1", "bounds": (0.066, 0.6), "tunable_tier": 1, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "emissivity_land": {"units": "1", "bounds": (0.3168, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "radiation", "reference": "legoESM land surface", "shape": None},
            "root_depth": {"units": "m", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "legoESM land surface", "shape": None},
            "soil_evap_resistance_exp": {"units": "1", "bounds": (0.0, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Sellers 1992 / Lee & Pielke 1992 (bare-soil evap resistance)", "shape": None},
            "soil_evap_litter_resistance_s_m": {"units": "s m-1", "bounds": (0.0, 400.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Sakaguchi & Zeng 2009 (forest-floor litter resistance)", "shape": None},
            "theta_fc": {"units": "1", "bounds": (0.099, 0.9), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "theta_wp": {"units": "1", "bounds": (0.0495, 0.45), "tunable_tier": 2, "transform": "sigmoid", "category": "material", "reference": "legoESM land surface", "shape": None},
            "z0_land": {"units": "m", "bounds": (0.0165, 0.15), "tunable_tier": 2, "transform": "sigmoid", "category": "surface", "reference": "legoESM land surface", "shape": None},
        },
    },
}


class LandConfig(NamedTuple):
    """Slab land + bucket hydrology configuration."""
    C_soil: float = 2.0e6       # Soil heat capacity [J/m3/K]
    d_soil: float = 1.0         # Slab soil depth [m]
    W_max: float = 150.0        # Bucket capacity [kg/m2]
    # Bucket runoff partition (Green-Ampt infiltration excess + saturation excess)
    K_infiltration: float = 1.0e-5     # Saturated infiltration capacity K_s [m/s]
    infil_suction_boost: float = 2.0   # Green-Ampt suction enhancement psi_f/L_f [-]
    infiltration_excess: bool = True   # Enable Hortonian infiltration-excess runoff
    albedo_land: float = 0.2    # Fallback constant albedo
    emissivity_land: float = constants.emissivity_land
    z0_land: float = 0.05       # Roughness length [m]
    Cd_land: float = 3.0e-3     # Land drag coefficient (constant scheme)
    Ch_land: float = 3.0e-3     # Land heat transfer coefficient (constant)
    beta_min: float = 0.1       # Minimum moisture availability (dry soil)
    # DEFAULT: Monin-Obukhov, the exchange law the coupled land tile runs;
    # "constant" is a fixed coefficient for idealized work.
    bulk_scheme: str = "most"  # "most" or "constant"
    z_ref: float = 10.0           # Observed forcing height [m]; model uses z_lowest
    bulk_n_iter: int = 5          # MOST iterations
    # Snow/albedo
    snow_albedo_feedback: bool = False  # Enable snow albedo feedback
    land_albedo: LandAlbedoConfig = LandAlbedoConfig()
    T_snow_melt: float = constants.T_freeze
    snow_melt_rate: float = 5.0e-6  # Snowmelt rate [kg/m2/s/K above T_melt]
    # Carbon cycle
    carbon: CarbonConfig = CarbonConfig()
    # Stomatal conductance / plant physiology
    stomata: StomataConfig = StomataConfig()
    # Surface scheme: ``SimpleSEBConfig`` (default) or ``TwoLeafCanopyConfig``.
    # Type hint is ``Any`` because NamedTuple does not support Unions well;
    # dispatch is done via ``isinstance`` inside ``step_land``.
    # DEFAULT: the two-leaf canopy with Monin-Obukhov surface exchange.
    #
    # ``SimpleSEBConfig`` is for ACADEMIC / SIMPLIFIED TESTS ONLY and must not be
    # a production default (user directive 2026-08-20).  Measured on a
    # well-watered column, one day, realistic forcing, latent heat in W/m2:
    #
    #     simple_seb, stomata off :  405 forest / 283 grass / 194 bare
    #     simple_seb, stomata on  :  129 forest /   4 grass /   2 bare
    #
    # Both are unusable and for opposite reasons.  With stomata off nothing
    # limits evaporation, so it runs at potential over a soil whose top layer
    # barely drains.  With stomata on, the Jarvis conductance has NO leaf-area
    # dependence, so BARE GROUND is throttled by stomata it does not have.  The
    # two-leaf canopy partitions the cell into canopy and soil and gives each its
    # own resistance, which is why the LMIP simulations built on it reproduce
    # observed latent heat and photosynthesis.
    surface_scheme: Any = TwoLeafCanopyConfig()
    # Runoff scheme (appended for positional-ABI stability): "bucket" (default,
    # Green-Ampt Hortonian + Dunne saturation-excess, byte-identical) or
    # "topmodel" (SIMTOP sub-grid saturated fraction + topographic baseflow,
    # Niu 2005 / CLM4.5).  Unknown -> ValueError at dispatch.
    runoff_scheme: str = "bucket"
    topmodel: TopmodelConfig = TopmodelConfig()


class MultiLayerLandConfig(NamedTuple):
    """Multi-layer soil model configuration (Task 8)."""
    # Surface properties
    albedo_land: float = 0.2    # Fallback constant albedo
    emissivity_land: float = constants.emissivity_land
    z0_land: float = 0.05
    # Bulk flux
    Cd_land: float = 3.0e-3
    Ch_land: float = 3.0e-3
    beta_min: float = 0.1
    # Surface soil resistance to BARE-SOIL evaporation.  The top soil layer dries
    # into a high-resistance crust far faster than the root-zone mean, so bare-soil
    # evaporation is throttled by the TOP-layer effective saturation S_top via a
    # beta-method efficiency S_top**soil_evap_resistance_exp (Sellers 1992 / Lee &
    # Pielke 1992 magnitude at exp=2).  exp=0 disables it (legacy: bare-soil evap
    # limited only by the root-zone beta + whole-column water supply -> over-strong
    # soil evaporation and a too-fast surface dry-down).
    soil_evap_resistance_exp: float = 2.0
    # Below-canopy soil-evaporation SERIES-RESISTANCE formulation (canopy path).
    # When True (default), the top-layer moisture control on soil evaporation is a
    # Sellers-1992 surface resistance r_ss = exp(8.206 - 4.255 theta_1/theta_sat)
    # PLUS a Sakaguchi-Zeng-2009 forest-floor litter resistance, added in SERIES
    # with the below-canopy aerodynamic resistance in the two-leaf soil energy
    # balance (energy_balance.soil_surface_evap_resistance); the beta efficiency
    # S_top**soil_evap_resistance_exp is then bypassed (mutually exclusive — same
    # Sellers-1992 physics).  The aerodynamic-only path (r_ss omitted) let a wet
    # forest floor evaporate at near-potential rate; the resistance form throttles
    # a WET surface and self-scales with canopy density across biomes.  Static
    # Python bool (feature gate).  False -> legacy beta efficiency (SimpleSEB path
    # is unaffected either way).
    soil_evap_series_resistance: bool = True
    # Reference forest-floor litter resistance [s/m] (Sakaguchi & Zeng 2009),
    # scaled by litter cover 1 - exp(-0.5 LAI); ~0 for bare soil, saturating for a
    # closed canopy.  Only used when soil_evap_series_resistance is True.  Default
    # 300 s/m calibrated on the FLUXNET EC-site validation (4 biomes: US-MMS DBF,
    # DE-Obe ENF, US-Ton savanna, US-Var grassland) to give a physical closed-
    # canopy soil-evaporation fraction (JJA ~0.2 forest, ~0 Mediterranean-summer)
    # and unbiased LE vs energy-closure-corrected obs; well within the published
    # forest-floor range (~1e2-1e3 s/m).
    soil_evap_litter_resistance_s_m: float = 300.0
    bulk_scheme: str = "most"       # see LandConfig.bulk_scheme
    z_ref: float = 10.0           # Observed forcing height [m]; model uses z_lowest
    bulk_n_iter: int = 5
    # Snow/albedo
    snow_albedo_feedback: bool = False  # Enable snow albedo feedback
    land_albedo: LandAlbedoConfig = LandAlbedoConfig()
    T_snow_melt: float = constants.T_freeze
    snow_melt_rate: float = 5.0e-6
    # Sub-grid elevation-band snow (VIC snow bands / CESM MEC); ``None`` (default)
    # runs the single cell-mean snowpack.  See ``legoesm.land.snow_bands``.
    elev_bands: ElevationSnowBandConfig | None = None
    # Root water uptake
    root_depth: float = 1.0       # Root e-folding depth [m]
    # ``theta_wp`` is the SOIL wilting point [m3/m3] — the moisture reference for
    # the soil column (initial-condition seeding, bare-soil references).  The
    # PLANT wilting point (the θ below which ROOT-ZONE transpiration and its
    # GPP/stomatal stress shut off) is ``theta_wp_plant``; ``None`` (default)
    # makes it equal to ``theta_wp`` so behaviour is unchanged, but the two can
    # be set separately — e.g. a phreatophyte (deep-rooted oak) extracts water to
    # a LOWER θ than the soil-evaporation cutoff, so ``theta_wp_plant <
    # theta_wp`` lets transpiration continue where the soil is otherwise "dry".
    theta_wp: float = 0.15        # SOIL wilting point volumetric water content
    theta_wp_plant: float | None = None  # PLANT wilting point (None => theta_wp)
    theta_fc: float = 0.30        # Field capacity volumetric water content
    # Sub-configs
    soil_grid: SoilGridConfig = SoilGridConfig()
    hydraulics: SoilHydraulicsConfig = SoilHydraulicsConfig()
    thermal: SoilThermalConfig = SoilThermalConfig()
    richards: RichardsConfig = RichardsConfig(fc_drain_saturation=0.5)
    # Carbon cycle
    carbon: CarbonConfig = CarbonConfig()
    # Stomatal conductance / plant physiology
    stomata: StomataConfig = StomataConfig()
    # Surface scheme: ``SimpleSEBConfig`` (default), ``TwoLeafCanopyConfig``,
    # or ``CLMMLCanopyConfig``.  Runtime dispatch via ``isinstance`` inside
    # ``step_multilayer_land``.
    # DEFAULT: the two-leaf canopy with Monin-Obukhov surface exchange.
    #
    # ``SimpleSEBConfig`` is for ACADEMIC / SIMPLIFIED TESTS ONLY and must not be
    # a production default (user directive 2026-08-20).  Measured on a
    # well-watered column, one day, realistic forcing, latent heat in W/m2:
    #
    #     simple_seb, stomata off :  405 forest / 283 grass / 194 bare
    #     simple_seb, stomata on  :  129 forest /   4 grass /   2 bare
    #
    # Both are unusable and for opposite reasons.  With stomata off nothing
    # limits evaporation, so it runs at potential over a soil whose top layer
    # barely drains.  With stomata on, the Jarvis conductance has NO leaf-area
    # dependence, so BARE GROUND is throttled by stomata it does not have.  The
    # two-leaf canopy partitions the cell into canopy and soil and gives each its
    # own resistance, which is why the LMIP simulations built on it reproduce
    # observed latent heat and photosynthesis.
    surface_scheme: Any = TwoLeafCanopyConfig()
    # Canopy-water interception (shared CLM-ML formulation, land/canopy/
    # interception.py).  ``None`` (default) = off (rain infiltrates directly).
    # When set, the two-leaf path intercepts rain into a prognostic ``W_canopy``
    # store, drips the excess as throughfall, and evaporates the wet leaf —
    # reducing soil infiltration and re-partitioning the canopy latent flux.
    # SimpleSEB has no canopy latent stream and ignores it; the CLM-ML canopy
    # has its OWN internal interception and ignores it too.
    interception: Any | None = None


def resolve_land_config(land_mode: str, land_config=None):
    """Return the land config object matching ``land_mode``.

    Single source of truth for the ``land_mode`` -> config-type mapping used by
    the coupled driver and the ``run_lmip_smoke`` driver: ``"multilayer"`` ->
    :class:`MultiLayerLandConfig`, ``"slab"``/``"none"`` -> :class:`LandConfig`.
    A ``land_config`` of the wrong type for the mode is replaced with the
    mode's default (so the runtime type always matches the selected model).
    """
    if land_mode == "multilayer":
        return land_config if isinstance(land_config, MultiLayerLandConfig) else MultiLayerLandConfig()
    if land_mode == "none":
        return LandConfig()
    return land_config if isinstance(land_config, LandConfig) else LandConfig()


def inactive_land_param_names(config: MultiLayerLandConfig) -> frozenset:
    """Registry-qualified land tunables that carry no gradient in ``config``.

    A trainable-parameter collector over the multilayer land must drop these
    (no inert parameters): the ice impedance exponent is read only when soil
    freeze/thaw is on.
    """
    if config.thermal.enable_freeze_thaw:
        return frozenset()
    return frozenset({"land.richards.ice_impedance_exponent"})


LAND_MODELS = ("none", "slab", "multilayer")


def land_model_switches(land_model: str) -> dict:
    """``land_model`` name -> the AMIP driver's land switches.

    ONE VOCABULARY FOR ONE CONCEPT.  The coupled driver and ``run_lmip_smoke``
    have always selected the land surface by name through
    ``CoupledESMConfig.land_mode`` and :func:`resolve_land_config`, using
    exactly these three values.  The AMIP driver instead exposed two
    independent booleans, ``slab_land_active`` and ``use_multilayer_land``,
    whose FOUR combinations encode only THREE states — and whose all-false
    combination means "no land surface at all", which is not a simpler land
    model but the absence of one.  That ambiguity has cost this project real
    campaign time before.  This function is the bridge, so a run can name its
    land model the same way everywhere.

    STRICTLY AN ALIAS.  Each name sets exactly the switch the existing flag
    sets, and nothing else:

        none       -> slab_land_active=False, use_multilayer_land=False
        slab       -> slab_land_active=True
        multilayer -> use_multilayer_land=True

    In particular "multilayer" does NOT also set ``slab_land_active``, even
    though the multilayer soil is layered on top of the slab's surface energy
    balance and needs an active land tile.  Setting both would make
    ``--land-model multilayer`` a DIFFERENT run from today's
    ``--use-multilayer-land`` for any deck that relies on topography or a
    land-mask file to activate the tile: the driver already turns the runtime
    tile on when a mask is present (``model_driver`` activates on
    ``land_mask_path or slab_land_active``), so forcing the config flag would
    change the one case where neither is set.  Whether a tile ends up active is
    left exactly where it already lives — with the topography, the mask, and
    the driver's own validation, which refuses a requested land tile over an
    idealized topography with a clear error.

    Raises on an unknown name rather than falling back to a default, so a typo
    cannot silently select a different land surface.
    """
    if land_model not in LAND_MODELS:
        raise ValueError(
            f"unknown land_model {land_model!r}; expected one of "
            f"{LAND_MODELS}. Note that 'none' means NO land surface model — "
            "land temperature falls back to the neighbouring prescribed SST "
            "minus a lapse rate — and is not a simplified land model."
        )
    return {
        "none": dict(slab_land_active=False, use_multilayer_land=False),
        "slab": dict(slab_land_active=True, use_multilayer_land=False),
        "multilayer": dict(slab_land_active=False, use_multilayer_land=True),
    }[land_model]


def describe_land_model(*, slab_land_active: bool, use_multilayer_land: bool,
                        has_land_mask: bool = False) -> str:
    """The land model a set of switches RESOLVES to, as one of :data:`LAND_MODELS`.

    The inverse of :func:`land_model_switches`, for printing what a run is
    actually about to do instead of trusting what its deck says it does.  A
    land-mask file activates the tile on its own, so it is part of the answer.
    """
    if use_multilayer_land:
        return "multilayer"
    if slab_land_active or has_land_mask:
        return "slab"
    return "none"


# ===========================================================================
# The land model the baked _TUNED_*_MULTILAYER tables were calibrated under
# ===========================================================================
# ``scripts/run/train_multilayer_land_era5.py`` fits EVERY baked multilayer table
# (albedo, emissivity, snow, soil thermal/hydraulic scales, and the canopy
# conductance Vc_max25/g1/LCMA) inside this exact configuration.  Both the
# calibrator and the coupled driver build their tile through this one function,
# so a run can no longer deploy the tables under different land physics.
#
# WHY IT EXISTS (2026-08-19): the coupled AMIP tile ran with ``stomata.enabled``
# False and no carbon state, so ``compute_effective_beta`` took its third branch
# (beta = beta_soil) and the baked canopy conductance was INERT — parameters
# fitted against a stomatal resistance the coupled run did not have.  Flipping
# ``land_stomatal_beta`` alone was not enough either: without the differland
# carbon scheme AND a prescribed carbon state the same dispatch falls to Jarvis,
# a different stomatal model from the Farquhar one the tables were fitted under.
_CALIB_ML_N_LAYERS = 8
_CALIB_ML_SOIL_DEPTH_M = 3.0      # total soil-column depth [m]
_CALIB_ML_SOIL_GROWTH = 1.5       # geometric layer-thickness growth factor [-]


def calibrated_multilayer_setup() -> dict:
    """Mode settings of the multilayer land model the baked tables were fit under.

    Returns a FRESH dict of :class:`MultiLayerLandConfig` field values (never a
    shared mutable singleton) covering only the non-spatial *mode* choices:
    surface-exchange law, stomatal/photosynthesis path, soil-column geometry and
    surface scheme.  The per-cell hydraulics / thermal / albedo maps are NOT here
    — those already come from the one shared ``clm_multilayer_setup`` bake, which
    preserves every field returned by this function.

    ``carbon.scheme == "differland"`` selects the Farquhar branch of
    ``stomata_utils.compute_effective_beta``, but that branch ALSO needs a
    non-None ``carbon_state`` at the call site; a consumer that applies this
    setup must seed one (see ``init_carbon_state``), exactly as the calibrator
    and ``train_coupled_land_era5.py`` do.
    """
    return dict(
        soil_grid=SoilGridConfig(n_layers=_CALIB_ML_N_LAYERS,
                                 total_depth=_CALIB_ML_SOIL_DEPTH_M,
                                 growth_factor=_CALIB_ML_SOIL_GROWTH),
        bulk_scheme="most",
        stomata=StomataConfig(enabled=True),
        carbon=CarbonConfig(scheme="differland"),
        snow_albedo_feedback=True,
        # simple_seb, because that is what the tables were FITTED under — not
        # because it is good.  It is structurally broken (its humidity gradient
        # self-extinguishes) and is no longer the library default.  These tables
        # must be re-fitted against the two-leaf canopy; until then this records
        # the scheme they belong to rather than pretending otherwise.
        surface_scheme=SimpleSEBConfig(),
    )


def apply_calibrated_multilayer(config: MultiLayerLandConfig) -> MultiLayerLandConfig:
    """Return ``config`` with the calibration land model applied.

    Every other field (per-cell hydraulics, thermal, albedo, Richards, runoff,
    interception, ...) is carried through untouched.
    """
    if not isinstance(config, MultiLayerLandConfig):
        raise TypeError(
            "apply_calibrated_multilayer expects a MultiLayerLandConfig, got "
            f"{type(config).__name__}; the calibrated tables are a MULTILAYER "
            "bake and have no slab equivalent."
        )
    return config._replace(**calibrated_multilayer_setup())


def biophysics_lmip_two_leaf_setup() -> dict:
    """Mode settings of the biophysics LMIP calibration (two-leaf canopy).

    Fitted by ``templates/land/biophysics/lmip_biophys_2deg.yaml`` — a 10-year,
    2-degree offline CRU-JRA run scored against observed fluxes. This is the
    configuration those parameters BELONG to; running them under another surface
    scheme is the mismatch the selector exists to prevent.

    Covers the non-spatial MODE choices only, exactly like
    :func:`calibrated_multilayer_setup`. Per-column hydraulics / thermal /
    albedo maps still come from the shared ``clm_multilayer_setup`` bake, and
    the calibration's SCALAR albedo settings are applied on top by
    :func:`apply_biophysics_lmip_two_leaf`, which needs the existing config and
    so cannot express them in this dict.
    """
    from legoesm.land.clm_surface_map import (
        BIOPHYS_LMIP_N_LAYERS, BIOPHYS_LMIP_SOIL_DEPTH_M,
        BIOPHYS_LMIP_SOIL_GROWTH)
    return dict(
        soil_grid=SoilGridConfig(n_layers=BIOPHYS_LMIP_N_LAYERS,
                                 total_depth=BIOPHYS_LMIP_SOIL_DEPTH_M,
                                 growth_factor=BIOPHYS_LMIP_SOIL_GROWTH),
        bulk_scheme="most",
        # The two-leaf canopy carries its OWN Ball-Berry stomata, so the separate
        # stomatal-beta throttle stays OFF: switching it on down-regulates the
        # same conductance twice. The offline calibration ran with
        # ``stomata_enabled: false`` for exactly that reason.
        stomata=StomataConfig(enabled=False),
        # The offline LMIP calibration ran with the differland prognostic carbon
        # cycle active: the canopy carries its own Farquhar PHOTOSYNTHESIS (a GPP
        # flux), but the carbon POOLS/respiration are differland's, and the
        # Vc_max25/g1/LCMA tables were fitted with it on. Deploying them under a
        # "none" carbon scheme runs a model the parameters were not fitted to.
        carbon=CarbonConfig(scheme="differland"),
        snow_albedo_feedback=True,
        surface_scheme=TwoLeafCanopyConfig(),
    )


def apply_biophysics_lmip_two_leaf(config: MultiLayerLandConfig) -> MultiLayerLandConfig:
    """Apply the biophysics LMIP calibration, including its albedo scalars."""
    from legoesm.land.clm_surface_map import biophysics_lmip_albedo_scalars
    cfg = config._replace(**biophysics_lmip_two_leaf_setup())
    # SCALARS ONLY on the existing albedo config: it also carries the per-column
    # snow_cover_scale map built from the surfdata, and replacing the whole
    # object would discard it.
    return cfg._replace(
        land_albedo=cfg.land_albedo._replace(**biophysics_lmip_albedo_scalars()))
