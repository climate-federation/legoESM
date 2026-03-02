"""Lake model configuration."""

from __future__ import annotations

from typing import NamedTuple


class LakeConfig(NamedTuple):
    """Two-layer lake (epilimnion + hypolimnion) configuration."""
    h_epi: float = 5.0              # Epilimnion depth [m]
    h_hypo: float = 20.0            # Hypolimnion depth [m]
    rho_water: float = 1000.0       # Water density [kg/m3]
    c_water: float = 4218.0         # Specific heat [J/kg/K]
    k_mix: float = 1.0e-2           # Vertical mixing coefficient [m2/s]
    wind_mix_alpha: float = 0.1     # Wind-driven mixing enhancement factor
    albedo_lake: float = 0.08
    emissivity_lake: float = 0.97
    z0_lake: float = 1e-4           # Roughness length [m]
    Cd_lake: float = 1.5e-3         # Drag coefficient
    Ch_lake: float = 1.5e-3         # Heat transfer coefficient
    T_freeze: float = 273.15        # Freezing point [K]
