"""Coupler configuration."""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm.surface_albedo import OceanAlbedoConfig

__param_spec__ = {
    "CouplerConfig": {
        "scheme_key": "coupler.surface",
        "excluded": {
            "U_min": "numerics: minimum wind-speed floor for bulk fluxes [m/s]",
            "co2_ppmv_default": "forcing: default atmospheric CO2 [ppmv]",
            "coupling_dt": "numerics: coupling timestep [s]",
            "z_q_atm": "convention: humidity measurement reference height [m]",
            "z_ref": "convention: reference height [m]",
            "z_t_atm": "convention: temperature measurement reference height [m]",
        },
        "params": {
            "Cd_ocean": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ocean-atmosphere momentum drag coefficient", "shape": None,
            },
            "Ch_ocean": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ocean-atmosphere heat transfer coefficient", "shape": None,
            },
            "ocean_albedo": {
                "units": "1", "bounds": (0.03, 0.15), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "open-ocean broadband albedo", "shape": None,
            },
            "ocean_emissivity": {
                "units": "1", "bounds": (0.9, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "open-ocean longwave emissivity", "shape": None,
            },
            "ocean_z0": {
                "units": "m", "bounds": (1.0e-5, 1.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "open-ocean aerodynamic roughness length", "shape": None,
            },
            "gustiness": {
                "units": "m s-1", "bounds": (0.0, 10.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "sub-grid convective gustiness floor (Wing 2018 RCEMIP1 / Beljaars 1995)", "shape": None,
            },
        },
    },
}


class TileConfig(NamedTuple):
    """Surface tile fraction configuration.

    f_land and f_lake are static masks from config/bathymetry, shape (6, n, n).
    Water fraction = 1 - f_land - f_lake is split into ocean and ice
    prognostically.
    """
    f_land: jax.Array      # Land fraction [0-1], shape (6, n, n)
    f_lake: jax.Array      # Lake fraction [0-1], shape (6, n, n)


class CouplerConfig(NamedTuple):
    """Configuration for the surface coupler."""
    coupling_dt: float = 3600.0       # Coupling interval [s]
    U_min: float = 1.0                # Numerical wind speed floor [m/s]
    # Sub-grid convective gustiness floor [m/s] for the air-sea bulk fluxes:
    # |U|_eff = sqrt(|U|^2 + gustiness^2).  DEFAULT 1.0: in an INTERACTIVE
    # coupled ocean equilibrium evaporation is ENERGY-limited, so a large
    # gustiness over-cools the slab (E transiently up -> SST cools -> q_sat down
    # -> E settles lower; measured gustiness=5 -> SST drift -20 K/yr).  ~5 m/s
    # (Wing 2018) is correct only for PRESCRIBED-SST (SCM/AMIP).  See the CAM
    # surface-energy audit.
    gustiness: float = 1.0
    # COARE 3.0 free-convection gustiness on the air-sea tile flux: the
    # boundary-layer depth z_i [m] that sets the convective velocity scale
    # w* = (g·z_i·<w'θv'>/θv)^(1/3) -> U_eff = sqrt(|U|^2 + (β·w*)^2).
    # None => SCHEME-NATIVE (bulk_flux.resolve_gustiness_w_zi: 600 m for
    # coare3, off for every other scheme) — the SAME nullable semantics as
    # the atmosphere surface layer (SurfaceLayerConfig.gustiness_w_zi) and
    # the slab ocean (SimpleOceanConfig.gustiness_w_zi), so the DEFAULT
    # coare3 interface no longer splits (atm 600 m vs tile off — the strict
    # xfail this closes).  0.0 => explicitly OFF.  Under the default
    # 'constant' closure the field is never read, so the OMIP / default
    # coupled run is byte-identical.  See the cmip_air_sea_decoupling fix:
    # a calm warm tropical ocean barely evaporates without w* (hfls ~45 vs
    # ~120 W/m²), drying the atmosphere -> weak greenhouse -> the warm SST
    # radiates to space -> 3D-ocean cold collapse.
    gustiness_w_zi: float | None = None  # Gustiness BL depth z_i [m]; None=scheme-native
    ocean_albedo: float = 0.06        # Fallback constant ocean albedo
    ocean_emissivity: float = 0.97    # Default ocean emissivity
    ocean_z0: float = 1e-4            # Ocean roughness length [m]
    co2_ppmv_default: float = 400.0   # Default CO2 concentration
    Cd_ocean: float = 1.5e-3          # Ocean drag coefficient
    Ch_ocean: float = 1.5e-3          # Ocean heat transfer coefficient
    bulk_scheme: str = "constant"     # "constant", "most", "coare3", "large_yeager"
    # Thermodynamic constants set for the ocean-tile most/coare3/large_yeager
    # fluxes (#762): "legoesm" (default) = constant L_v / dry c_pd;
    # "aerobulk" = NEMO/AeroBulk/COARE parity (SST-dependent L_vap(T_sfc),
    # moist cp_air(q_atm)).  Kept consistent with the atmosphere
    # SurfaceLayerConfig.thermo_convention by run_coupled.
    thermo_convention: str = "legoesm"
    # Stable-regime (zeta>0) MOST similarity functions for the MOST-family
    # bulk schemes ("most"/"coare3"/"large_yeager"): "dyer1974" (default,
    # historical -5*zeta), "beljaars_holtslag1991", "grachev2007_sheba"
    # (Arctic/SHEBA), "gryanik2020".  Unstable branch is Businger-Dyer for
    # every choice; validated at dispatch (unknown -> ValueError).
    stability_scheme: str = "dyer1974"
    z_ref: float = 10.0               # Wind reference height [m]
    # Air temperature / specific humidity reference heights. Default to
    # z_ref for legacy single-height callers (lake, idealized adapter,
    # AMIP-style runs that read from the lowest atm level). For OMIP /
    # JRA55-do, set both to 2.0 — the reanalysis delivers ``tas`` and
    # ``huss`` at 2 m while ``uas, vas`` are at 10 m.
    z_t_atm: float = 10.0             # Air-temperature reference height [m]
    z_q_atm: float = 10.0             # Specific-humidity reference height [m]
    bulk_n_iter: int = 5              # MOST iterations (coare3/large_yeager)
    # Zenith-dependent ocean albedo (Task 10)
    ocean_albedo_config: OceanAlbedoConfig = OceanAlbedoConfig()
