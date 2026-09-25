"""Sea ice model configuration."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.surface_albedo import IceAlbedoConfig
from legoesm.ice.constants_config import (
    NEMO_SI3_CONSTANTS_CONFIG,
    IceConstantsConfig,
)

__param_spec__ = {
    "SI3ThermoConfig": {
        "scheme_key": "ice.si3_thermo",
        "excluded": {
            "new_ice_salinity_fraction": "oracle identity: rn_sinew=0.75 is fixed by the ORCA1 deck",
        },
        "params": {},
    },
    "SnowConfig": {
        "scheme_key": "ice.snow",
        "excluded": {
            "h_snow_min": "numerics: min snow depth for active conductivity",
            "sublim_partition": "physics: sublimation source partition (1.0 = snow-first mode)",
        },
        "params": {},
    },
    "BrineConfig": {
        "scheme_key": "ice.brine",
        "excluded": {
            "S_ice_min": "numerics: salinity floor",
            "S_ice_max": "numerics: salinity cap for stability",
            # S_ocean_ref stays fixed: an environmental reference (ocean
            # salinity), not a tunable ice closure; defaults to constants.
        },
        "params": {
            # Salinity of newly-frozen lead/basal ice — the brine-rejection
            # closure that sets the ice<->ocean salt flux. Defaults to
            # constants.S_ice_bulk_default (4 PSU) but is a calibratable physical
            # closure, so it is exposed (eligible despite the constants default).
            "S_ice_new": {
                "units": "PSU", "bounds": (1.0, 12.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "bulk salinity of newly frozen sea ice (Notz/CICE ~4 PSU)",
                "shape": None,
            },
        },
    },
    "RidgingConfig": {
        "scheme_key": "ice.ridging",
        "excluded": {"closing_rate_max": "numerics: convergence-rate sanity cap"},
        "params": {
            "e_star": {
                "units": "m", "bounds": (0.1, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Lipscomb (2007) participation e-folding thickness", "shape": None,
            },
            "mu_rdg": {
                "units": "1", "bounds": (2.0, 8.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Lipscomb (2007) ridge-thickness multiplier", "shape": None,
            },
            "H_star": {
                "units": "m", "bounds": (50.0, 200.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Hibler/Lipscomb max ridge thickness scale", "shape": None,
            },
            "snow_fraction_retained": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "CICE donor-snow retention fraction in ridges", "shape": None,
            },
            "cs_shear_ridging": {
                "units": "1", "bounds": (0.0, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Rothrock (1975) / CICE shear-ridging fraction Cs", "shape": None,
            },
        },
    },
    "MeltPondConfig": {
        "scheme_key": "ice.ponds",
        "excluded": {
            "refreeze_width_K": "numerics: refreeze-ramp smoothing half-width",
        },
        "params": {
            "drainage_timescale_s": {
                "units": "s", "bounds": (3600.0, 864000.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "CESM melt-pond drainage e-folding time", "shape": None,
                "legacy_name": "drainage_timescale",
            },
            "pond_to_ice_max_area": {
                "units": "1", "bounds": (0.2, 0.9), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "CESM cap on pond area fraction per category", "shape": None,
            },
            "depth_to_area_ratio": {
                "units": "1", "bounds": (0.3, 1.5), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "CICE pond volume->area conversion", "shape": None,
            },
            "snow_block_threshold": {
                "units": "m", "bounds": (1.0e-3, 2.0e-2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "CICE min snow depth blocking pond formation (~5 mm)", "shape": None,
            },
        },
    },
    "SeaIceConfig": {
        "scheme_key": "ice.sea_ice",
        "excluded": {
            "h_ice_min": "numerics: min ice thickness for smooth ops",
            "T_ice_min": "numerics: lower temperature bound for stability",
            "z_ref": "convention: MOST reference height (10 m standard)",
            "Delta_min": "numerics: deformation-rate regulariser floor",
            "alpha_mevp": "numerics: mEVP stress relaxation (stability-coupled to N_mevp)",
            "beta_mevp": "numerics: mEVP velocity relaxation (stability-coupled to N_mevp)",
            "T_evp": "numerics: EVP damping ratio coupled to the N_evp subcycle count (E_factor = 1/(2*T_evp*N_evp))",
            "sw_transmittance_const": "constant-scheme SW transmittance, default 0.0 (off — delta_eddington computes its own) = the physical floor; not a well-posed sigmoid tunable (default on the bound)",
        },
        "params": {
            "albedo_ice": {
                "units": "1", "bounds": (0.4, 0.85), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "bare-ice broadband albedo (fallback constant)", "shape": None,
            },
            "albedo_ocean": {
                "units": "1", "bounds": (0.03, 0.15), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "open-water albedo for freezing calc", "shape": None,
            },
            "emissivity_ice": {
                "units": "1", "bounds": (0.9, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "sea-ice longwave emissivity", "shape": None,
            },
            # (sw_transmittance_const: excluded — default 0.0 (off) on the bound.)
            "z0_ice": {
                "units": "m", "bounds": (1.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ice aerodynamic roughness length", "shape": None,
            },
            "Cd_ice": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ice-atmosphere momentum drag coefficient", "shape": None,
            },
            "Ch_ice": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ice-atmosphere heat transfer coefficient", "shape": None,
            },
            "drag_ocean": {
                "units": "1", "bounds": (1.0e-3, 1.0e-2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ocean-ice drag coefficient", "shape": None,
            },
            "drag_atm": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "air-ice drag coefficient", "shape": None,
            },
            "ocean_heat_transfer_coeff": {
                "units": "W/m^2/K", "bounds": (5.0, 50.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "ocean-ice sensible heat transfer", "shape": None,
            },
            "h_new_ice": {
                "units": "m", "bounds": (0.01, 0.2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "thickness of newly-formed lead ice", "shape": None,
            },
            "e_yield": {
                "units": "1", "bounds": (1.5, 2.5), "tunable_tier": 2,
                "transform": "sigmoid", "category": "rheology",
                "reference": "Hibler VP yield-curve eccentricity (=2)", "shape": None,
            },
            "P_star": {
                "units": "N/m^2", "bounds": (1.0e4, 5.0e4), "tunable_tier": 1,
                "transform": "sigmoid", "category": "rheology",
                "reference": "Hunke & Dukowicz (1997) ice strength parameter", "shape": None,
            },
            "C_strength": {
                "units": "1", "bounds": (10.0, 30.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "rheology",
                "reference": "ice-strength concentration decay constant", "shape": None,
            },
        },
    },
}


class SnowConfig(NamedTuple):
    """Snow-on-ice layer configuration.

    Snow is tracked as a single bulk layer per ice category with
    constant density.  Conductive heat flux through the combined
    snow-ice column uses harmonic averaging of conductivities.
    Snow-ice (white-ice) formation by Archimedes flooding is enabled
    when ``flooding`` is True.
    """
    enabled: bool = False              # Master gate (False = no snow tracking)
    rho_snow: float = constants.rho_snow
    k_snow: float = constants.k_snow
    c_snow: float = constants.c_snow
    h_snow_min: float = 1.0e-4         # Min snow depth for active conductivity [m]
    flooding: bool = True              # Enable snow-ice (white-ice) flooding
    sublim_partition: float = 1.0      # Fraction of sublimation mass drawn from snow
                                        # (1.0 = sublimate snow first, then ice)


class BrineConfig(NamedTuple):
    """Sea-ice bulk salinity + brine-rejection configuration.

    Tracks a single bulk ice salinity ``S_ice`` per category and routes
    the salt budget into the ocean coupler channel.

    - During freezing of new ice in a lead: a fraction
      ``(S_ocean - S_ice_new) / S_ocean`` of the salt mass stays in the
      ocean (brine rejection); ``S_ice_new`` is captured by the ice as
      bulk salinity.
    - During basal/surface melt: ice salt is returned to the ocean.
    - Snow-ice (flooding) consolidates salty seawater into the ice
      column at ocean salinity (Notz / Maykut Untersteiner convention).
    """
    enabled: bool = False              # Master gate (False = freshwater convention)
    S_ice_new: float = constants.S_ice_bulk_default  # Salinity of newly frozen ice [PSU]
    S_ocean_ref: float = constants.S_ocean_ref       # Reference ocean salinity [PSU]
    S_ice_min: float = 0.0
    S_ice_max: float = 12.0            # Cap for numerical stability


class RidgingConfig(NamedTuple):
    """Mechanical ridging configuration (Lipscomb 2007).

    Convergence-driven thickness redistribution: thin-ice categories
    "donate" area + volume to ridged ice that is added to thicker
    categories.  Participation by category follows an exponential
    function of category mean thickness; ridge thickness range follows
    the Hibler / Lipscomb formula  H_min = 2*h_part,  H_max =
    min(mu_rdg * sqrt(h_part), H_star).
    """
    enabled: bool = False
    e_star: float = 0.36               # Participation e-folding thickness [m]
                                        # (Lipscomb 2007 default e* = 0.36 m)
    mu_rdg: float = 4.0                # Ridge-thickness multiplier
                                        # (H_max = mu * sqrt(h_part))
    H_star: float = 100.0              # Maximum ridge thickness scale [m]
    snow_fraction_retained: float = 0.5  # Fraction of donor snow retained
                                          # in ridges (remainder enters ocean
                                          # via runoff, CICE convention).
    cs_shear_ridging: float = 0.25     # Shear-ridging participation fraction
                                        # Cs (Rothrock 1975 / CICE): fraction of
                                        # shear deformation that drives ridging
                                        # in addition to pure convergence.
    closing_rate_max: float = 1.0      # Cap on convergence rate [1/s]
                                        # (sanity bound; physical Δ rarely
                                        # exceeds 1e-5)
    # "strain": existing convergence + shear closure. "convergence":
    # convergence only, also available on the tripole C-grid via its existing
    # fold-aware transport divergence. Does not enable rheology or landfast.
    closing_scheme: str = "strain"


class MeltPondConfig(NamedTuple):
    """CESM-style melt-pond configuration.

    Tracks pond area fraction (relative to ice area) and mean depth
    per category.  Pond water is sourced from surface melt and
    rainfall on bare ice; drains via simple linear Darcy parameter to
    the ocean.  Pond freezing returns water to bulk ice volume.
    """
    enabled: bool = False
    refreeze_threshold: float = constants.T_freeze  # Air T below which ponds refreeze [K]
    # Smoothing half-width [K] for the refreeze ramp around the
    # threshold (keeps the refreeze response differentiable — see
    # step_ponds).  Smaller → sharper, CICE-like cutoff.
    refreeze_width_K: float = 0.5
    drainage_timescale_s: float = 86400.0  # Drainage e-folding time [s] (1 day)
    pond_to_ice_max_area: float = 0.6    # Cap pond fraction per category
    depth_to_area_ratio: float = 0.8     # Volume → area conversion (CICE)
    # Minimum snow depth that blocks pond formation [m].  Snow
    # thinner than this is treated as transparent to surface melt
    # water so deposition / patchy snow does not lock out pond
    # capture.  CICE uses ~5 mm.
    snow_block_threshold: float = 5.0e-3


class SI3ThermoConfig(NamedTuple):
    """The only supported SI3 thermodynamic identity in this campaign.

    Selector provenance: BL99/P07 is ``icethd_zdf_bl99.F90:248-275``;
    option-2 salinity is ``icethd_sal.F90:204-249``; the top-level ordering is
    ``icethd.F90:148-183``.  Values are exposed for receipts and validation,
    but :func:`validate_si3_thermo_config` rejects every other combination.
    """

    n_ice_layers: int = 3
    n_snow_layers: int = 3
    conductivity: str = "p07"
    salinity_scheme: int = 2
    new_ice_salinity_fraction: float = 0.75
    drainage: bool = True
    flushing: bool = True
    ponds: bool = False
    lateral_melt: bool = False


def validate_si3_thermo_config(config: "SeaIceConfig") -> None:
    """Reject unsupported SI3 selector mixtures before any tendency runs."""

    if config.thermo_scheme != "si3_bl99":
        return
    wanted = SI3ThermoConfig()
    if config.si3 != wanted:
        raise ValueError(
            "thermo_scheme='si3_bl99' supports only the ORCA1-resolved "
            f"identity {wanted!r}; got {config.si3!r}"
        )
    if config.ice_constants != NEMO_SI3_CONSTANTS_CONFIG:
        raise ValueError(
            "thermo_scheme='si3_bl99' requires the NEMO 5.0.2 phycst/EOS "
            "constant set; mixing canonical legoESM constants into this "
            "oracle identity is unsupported"
        )
    conflicts = []
    if config.n_categories != 1:
        conflicts.append("n_categories must be 1 (HFN single category; namitd)")
    if config.ponds.enabled:
        conflicts.append("ponds.enabled must be False (ln_pnd=.false.)")
    if config.ridging.enabled:
        conflicts.append("ridging is inert in the C1D thermodynamic column")
    if config.dynamics != "none" or config.transport != "none":
        conflicts.append("dynamics and transport must both be 'none' in C1D")
    if conflicts:
        raise ValueError("invalid SI3 C1D identity: " + "; ".join(conflicts))


def validate_si3_bulk_config(config: "SeaIceConfig") -> None:
    """Reject mixtures outside the ORCA1 constant-coefficient ice identity.

    Selector provenance is ``sbcblk.F90:335-350,1085-1168`` and the ORCA1
    overlay ``namelist_cfg:129-142``.  The shipped C1D ECMWF arm is not this
    selector and remains uncertified.
    """
    if config.bulk_scheme != "nemo_si3_constant":
        return
    expected = constants.bulk_transfer_ice_orca1
    conflicts = []
    if (config.Cd_ice, config.Ch_ice, config.Ce_ice) != (
        expected, expected, expected,
    ):
        conflicts.append("Cd_ice=Ch_ice=Ce_ice must be ORCA1's 1e-3")
    if config.emissivity_ice != constants.emissivity_ice_nemo:
        conflicts.append("emissivity_ice must be NEMO sbc_phy emiss_i")
    if config.ice_constants != NEMO_SI3_CONSTANTS_CONFIG:
        conflicts.append("NEMO SI3 physical-constant set is required")
    if config.n_categories != 1 or config.ponds.enabled:
        conflicts.append("only jpl=1 with ln_pnd=.false. is certified")
    if conflicts:
        raise ValueError("invalid NEMO SI3 bulk identity: " + "; ".join(conflicts))


class SeaIceConfig(NamedTuple):
    """Thermodynamic slab + optional dynamics sea ice configuration.

    Dynamics modes:
    - ``"none"``: Slab thermodynamics only (diagnostic free-drift).
    - ``"free_drift"``: Free-drift velocity with tracer advection.
    - ``"evp"``: Elastic-Viscous-Plastic rheology (Hunke & Dukowicz 1997).
    - ``"mevp"``: Modified-EVP pseudo-time relaxation toward implicit VP
      (Bouillon 2013 / Kimmritz 2015).

    Multi-category ice:
    - ``n_categories=1``: Single-category slab (default, backward compatible).
    - ``n_categories=5``: 5-category CICE-standard ITD.

    New-physics gates (all default off for backward compatibility):
    - ``snow.enabled``: track snow-on-ice
    - ``brine.enabled``: track bulk ice salinity + route salt flux
    - ``ridging.enabled``: Lipscomb 2007 mechanical ridging
    - ``ponds.enabled``: CESM-style melt ponds
    - ``shortwave_scheme``: "constant" (default) | "maykut_untersteiner" |
      "delta_eddington"
    """
    rho_ice: float = constants.rho_ice
    c_ice: float = constants.c_pi
    k_ice: float = constants.k_ice_default
    L_f: float = constants.L_f
    h_ice_min: float = 0.01         # Min ice thickness for smooth ops [m]
    albedo_ice: float = 0.65        # Fallback constant albedo
    albedo_ocean: float = 0.06      # Ocean albedo for open-water freezing calc
    emissivity_ice: float = constants.emissivity_ice
    z0_ice: float = 5e-4            # Ice roughness length [m]
    Cd_ice: float = constants.bulk_transfer_ice_default  # Ice-atmosphere drag coefficient
    Ch_ice: float = constants.bulk_transfer_ice_default  # Ice-atmosphere heat transfer coefficient
    # Transport
    drag_ocean: float = 5.5e-3      # Ocean-ice drag coefficient
    drag_atm: float = 1.3e-3        # Air-ice drag coefficient
    rho_air_ref: float = constants.rho_air
    rho_ocean_ref: float = constants.rho_ocean
    T_freeze_ocean: float = constants.T_freeze_ocean
    # Surface (snow / upper-ice) melt point.  The ice/snow TOP is fresh,
    # so it melts at 0 C = constants.T_freeze (273.15 K) — distinct from
    # the saline basal/ocean freezing point T_freeze_ocean (271.35 K).
    # Used as the surface skin-temperature clamp ceiling and the
    # surface-melt trigger; the basal conductive gradient, open-water
    # surface temperature, and lead-freeze ocean exchange keep using
    # T_freeze_ocean.
    T_melt_surface: float = constants.T_freeze
    T_ice_min: float = 180.0        # Lower bound for numerical stability [K]
    ocean_heat_transfer_coeff: float = 20.0  # Ocean-ice heat transfer [W/m^2/K]
    # SKIN-ONLY gate.  The response latent (TileResponse.lhflx) and surface mass
    # flux ALWAYS report the realized (over-ablation-capped) values -- that is
    # the latent the atmosphere actually receives (the coupler blends the ice
    # latent as L_s * surface_mass_flux), so the atmosphere energy<->water budget
    # closes regardless of this flag.  This flag only controls whether the
    # implicit SKIN TEMPERATURE T_new is RE-SOLVED with that realized latent so
    # the returned skin temp is consistent with it.  True (default) re-solves;
    # False leaves the skin cooled by the uncapped bulk latent (a bounded thin-
    # ice residual, <= L_s*rho_ice*h/dt).  In practice reachable clamp cells are
    # MELTING -> T pinned at the melt point -> the re-solve is a no-op there; its
    # only active effect is sub-freezing clamp cells (h < h_ice_min regime).
    # Static feature gate (Python ``if``, not a traced ``where``).
    latent_skin_resolve: bool = True
    # Concentration dynamics
    h_new_ice: float = 0.05         # Thickness for new ice formation [m]
    # Bulk flux algorithm
    bulk_scheme: str = "constant"   # "constant" or "most"
    # Stable-regime (zeta>0) MOST similarity functions for the MOST-family
    # bulk schemes: "dyer1974" (default -5*zeta) | "beljaars_holtslag1991" |
    # "grachev2007_sheba" (SHEBA — the Arctic sea-ice reference) |
    # "gryanik2020".  Unknown -> ValueError at dispatch.
    stability_scheme: str = "dyer1974"
    z_ref: float = 10.0             # Reference height for MOST [m]
    bulk_n_iter: int = 5            # MOST iterations
    # Temperature-dependent albedo (Task 10)
    temp_dependent_albedo: bool = False
    ice_albedo: IceAlbedoConfig = IceAlbedoConfig()
    # --- Dynamics ---
    dynamics: str = "none"          # "none", "free_drift", "evp", or "mevp"
    differentiable_dynamics: bool = False  # scan vs fori_loop for EVP
    # --- EVP rheology parameters ---
    N_evp: int = 120                # EVP subcycle count
    e_yield: float = 2.0            # Yield curve eccentricity
    P_star: float = 2.75e4          # Ice strength parameter [N/m^2]
    C_strength: float = 20.0        # Strength exponential decay constant
    Delta_min: float = 2.0e-9       # Minimum deformation rate [1/s]
    T_evp: float = 0.36             # EVP damping timescale ratio
    # --- mEVP rheology parameters ---
    N_mevp: int = 120               # mEVP pseudo-time iteration count
    alpha_mevp: float = 500.0       # mEVP stress relaxation parameter
    beta_mevp: float = 500.0        # mEVP velocity relaxation parameter
    # --- Multi-category ice ---
    n_categories: int = 1           # 1=single-category (backward compat), 5=CICE ITD
    # --- Tracer transport ---
    transport: str = "none"         # "none" or "advect"
    # PPM transport is monotone only when |u·dt/dx| ≤ 1.  Default 1
    # substep covers the typical sea-ice regime (u ~ 0.1 m/s, dx ~
    # 50 km, dt ~ 1 h → C ~ 0.007).  Raise for storm / fine-grid
    # conditions where the Courant number can exceed 1.
    transport_subcycles: int = 1
    # --- ITD remap scheme ---
    # "simple" preserves legacy behaviour (volume-conserving local rescale).
    # "lipscomb2001" uses the piecewise-linear g(h) remapping (Lipscomb 2001).
    itd_remap: str = "simple"
    # --- New-physics modules (each gated off by default) ---
    snow: SnowConfig = SnowConfig()
    brine: BrineConfig = BrineConfig()
    ridging: RidgingConfig = RidgingConfig()
    ponds: MeltPondConfig = MeltPondConfig()
    # --- Shortwave / albedo scheme ---
    # "constant"            → uniform ``albedo_ice``
    # "maykut_untersteiner" → α(T_sfc, h_ice) thin-ice ramp
    # "delta_eddington"     → Briegleb & Light 2007 two-band albedo with
    #                         snow / pond modifications + interior SW
    #                         penetration coefficient ``i0_vis``.
    shortwave_scheme: str = "constant"
    # SW transmittance through ice+snow to the ocean for the CONSTANT
    # shortwave scheme (fraction of INCIDENT sw_down; bounded by the
    # non-reflected column input).  Debited from the ice surface-absorbed SW
    # and delivered to the ocean via the existing sw_penetrated ->
    # ocean_heat_extraction channel, closing the energy budget the old
    # ocean-side A*tau*swd surrogate left open (codex).  0.0 = bit-identical
    # legacy (no penetration); the OMIP runner passes --ice-thermo-sw-trans.
    # delta_eddington computes its own transmittance and ignores this.
    # APPENDED at the tail (after every pre-existing field) so positional
    # SeaIceConfig(...) constructors keep their meaning (codex L1-r1 #3).
    sw_transmittance_const: float = 0.0
    # Lead / open-water ice formation source (new-physics path only):
    #  * "ice_skin" (legacy): lead ice grows from the ICE-skin atmospheric
    #    deficit max(-Q_sfc, 0) x (1-A), gated on SST <= T_freeze_ocean, and
    #    the ocean is DEBITED its latent heat.  Under a prognostic ocean that
    #    separately receives the open-water flux (1-A)*q_open this double-
    #    counts the cooling and nothing caps the ocean at the freezing point:
    #    measured 2026-09-02 on both OMIP grids, the polar top cell reaches
    #    -3 C (min -8 C) by day 105 while NEMO never drops below freezing.
    #  * "nemo_qlead": NEMO SI3 lead heat budget (icesbc.F90:357-405).
    #    zqld = (1-A)*q_open*dt is the open-water cooling; zqfr = rho0*cp*
    #    dz_top*(Tf-SST) the energy that brings the top cell to freezing;
    #    qlead = min(0, zqld - zqfr) forms ice, and its latent heat is
    #    RETURNED to the ocean (extraction -L_f*rho_i*dV/dt), so the ocean
    #    cools at most to Tf and a supercooled cell (zqfr > 0) freezes back
    #    to Tf (frazil).  Needs q_open_top + ocean_dz_top_m from the
    #    caller (step_sea_ice kwargs); the OMIP driver passes them.
    lead_freeze_source: str = "ice_skin"
    # Layered SI3 is an option inside this existing model.
    # ORDER, decided at the 2026-09-25 merge: both sides appended fields at
    # the tail claiming to preserve positional SeaIceConfig constructors, and
    # only one of the two claims can survive.  main's lead_freeze_source keeps
    # its own index (immediately after sw_transmittance_const) because main is
    # the shared tree; the four fidelity-lane fields below move one place
    # later.  No positional SeaIceConfig(...) call exists anywhere in the
    # tree -- every call site is keyword-only -- so nothing is broken either
    # way, and this comment replaces two claims that could not both be true.
    thermo_scheme: str = "zero_layer"
    si3: SI3ThermoConfig = SI3ThermoConfig()
    ice_constants: IceConstantsConfig = IceConstantsConfig()
    # NEMO distinguishes the Dalton (Ce) and Stanton (Ch) coefficients even
    # when ORCA1 makes them equal.
    Ce_ice: float = constants.bulk_transfer_ice_default
