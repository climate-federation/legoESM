"""Sea-ice-scoped physical constants with an explicit NEMO/SI3 preset."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants

# These values define a selectable reference-model constant set, not calibration
# levers.  The structural parameter gate nevertheless requires every float in a
# ``*Config`` NamedTuple to be classified explicitly.
__param_spec__ = {
    "IceConstantsConfig": {
        "scheme_key": "ice.constants",
        "excluded": {
            "T0": "physical constant",
            "rho_ice": "physical constant",
            "rho_snow": "physical constant",
            "rho_ocean": "reference-model physical constant",
            "c_ice": "physical constant",
            "c_ocean": "reference-model physical constant",
            "latent_fusion": "physical constant",
            "latent_sublimation": "physical constant",
            "k_ice": "reference-model physical constant",
            "k_snow": "reference-model physical constant",
            "liquidus_slope": "physical constant",
            "S_ocean_ref": "environmental reference constant",
            "lead_albedo": "reference-model surface constant",
        },
        "params": {},
    },
}


class IceConstantsConfig(NamedTuple):
    """Constants consumed by layered sea-ice thermodynamics.

    Defaults remain the canonical legoESM values.  Reference-model values are
    selected through a config object, never by mutating ``legoesm.constants``.
    """

    T0: float = constants.T_freeze
    rho_ice: float = constants.rho_ice
    rho_snow: float = constants.rho_snow
    rho_ocean: float = constants.rho_ocean
    c_ice: float = constants.c_pi
    c_ocean: float = constants.c_sw
    latent_fusion: float = constants.L_f
    latent_sublimation: float = constants.L_s
    k_ice: float = constants.k_ice_default
    k_snow: float = constants.k_snow
    liquidus_slope: float = constants.mu_ice_freeze
    S_ocean_ref: float = constants.S_ocean_ref
    lead_albedo: float = 0.06  # canonical legoESM open-water albedo


# NEMO 5.0.2 values printed by the accepted C1D_OMIP_L3 run and defined at
# src/OCE/DOM/phycst.F90:39,57-66; rho0/rcp are the live TEOS-10 values from
# src/OCE/TRA/eosbn2.F90:1898-1899.  Values already represented by canonical
# constants are referenced directly; the remaining literals are SI3-specific.
NEMO_SI3_CONSTANTS_CONFIG = IceConstantsConfig(
    T0=constants.T_freeze,
    rho_ice=917.0,
    rho_snow=330.0,
    rho_ocean=constants.rho_ocean_nemo,
    c_ice=2096.7,
    c_ocean=3991.86795711963,
    latent_fusion=constants.L_fus_nemo,
    latent_sublimation=2.8344e6,
    k_ice=2.034396,
    k_snow=0.5,
    liquidus_slope=0.054,
    S_ocean_ref=34.7,
    lead_albedo=0.066,
)


__all__ = ("IceConstantsConfig", "NEMO_SI3_CONSTANTS_CONFIG")
