"""Lake model configuration."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants

__param_spec__ = {
    "LakeConfig": {
        "scheme_key": "coupler.lake",
        "excluded": {"z_ref": "convention: reference height [m]"},
        "params": {
            "h_epi": {
                "units": "m", "bounds": (1.0, 20.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "two-layer lake epilimnion depth", "shape": None,
            },
            "h_hypo": {
                "units": "m", "bounds": (5.0, 100.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "two-layer lake hypolimnion depth", "shape": None,
            },
            "k_mix": {
                "units": "m^2/s", "bounds": (1.0e-3, 1.0e-1), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "lake epi-hypo turbulent mixing coefficient", "shape": None,
            },
            "wind_mix_alpha": {
                "units": "1", "bounds": (0.01, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "lake wind-driven mixing efficiency", "shape": None,
            },
            "Cd_lake": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "lake-atmosphere momentum drag coefficient", "shape": None,
            },
            "Ch_lake": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "lake-atmosphere heat transfer coefficient", "shape": None,
            },
            "albedo_lake": {
                "units": "1", "bounds": (0.03, 0.2), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "lake surface broadband albedo", "shape": None,
            },
            "albedo_lake_ice": {
                "units": "1", "bounds": (0.3, 0.7), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "frozen-lake (ice/snow) broadband albedo", "shape": None,
            },
            "emissivity_lake": {
                "units": "1", "bounds": (0.9, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "lake surface longwave emissivity", "shape": None,
            },
            "z0_lake": {
                "units": "m", "bounds": (1.0e-5, 1.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "lake aerodynamic roughness length", "shape": None,
            },
        },
    },
}


class LakeConfig(NamedTuple):
    """Two-layer lake (epilimnion + hypolimnion) configuration."""
    h_epi: float = 5.0              # Epilimnion depth [m]
    h_hypo: float = 20.0            # Hypolimnion depth [m]
    rho_water: float = constants.rho_water
    c_water_mass: float = constants.c_pw
    k_mix: float = 1.0e-2           # Vertical mixing coefficient [m2/s]
    wind_mix_alpha: float = 0.1     # Wind-driven mixing enhancement factor
    albedo_lake: float = 0.08
    albedo_lake_ice: float = 0.6    # frozen-lake ice/snow broadband albedo
    emissivity_lake: float = 0.97
    z0_lake: float = 1e-4           # Roughness length [m]
    Cd_lake: float = 1.5e-3         # Drag coefficient (constant scheme)
    Ch_lake: float = 1.5e-3         # Heat transfer coefficient (constant)
    T_freeze: float = constants.T_freeze
    # Freshwater EOS parameters (T_freshwater_max_density and
    # rho_freshwater_curvature) live in legoesm.constants and are not
    # repeated here — physical constants belong in the constants
    # module per CLAUDE.md.
    bulk_scheme: str = "constant"   # "constant" or "most"
    z_ref: float = 10.0             # Reference height for MOST [m]
    bulk_n_iter: int = 5            # MOST iterations
