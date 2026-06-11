"""Configuration for the fast-SBM bin microphysics scheme.

Defaults are the WRF ``module_mp_fast_sbm.F`` oracle values (CGS → SI).
Where the oracle's reference coefficient differs from the repo-wide
``legoesm.constants`` value (different reference temperature), the oracle
value is kept here as an explicit scheme parameter so the port stays
faithful without monkey-patching shared constants.
"""

from __future__ import annotations

from typing import NamedTuple


class FastSBMConfig(NamedTuple):
    """Tunable parameters of the fast-SBM scheme (oracle defaults).

    Attributes:
        d_vapor_ref_m2s: water-vapor diffusivity at the reference state
            [m²/s] (oracle ``D_MYIN = 0.211`` cm²/s; ``legoesm.constants.
            D_vapor = 2.21e-5`` uses a different reference temperature).
        nu_air_ref_m2s: kinematic viscosity of air used in the ventilation
            Reynolds number [m²/s] (oracle ``COEFF_VISCOUS = 0.13`` cm²/s).
        diffusivity_T_exponent: temperature exponent of the diffusivity law
            ``D = D_ref (p₀/p)(T/T₀)^a`` (oracle ``1.94``).
        ventilation_max: cap on the ventilation factor (oracle
            ``VENTPL_MAX = 5.0``).
    """

    d_vapor_ref_m2s: float = 2.11e-5
    nu_air_ref_m2s: float = 1.3e-5
    diffusivity_T_exponent: float = 1.94
    ventilation_max: float = 5.0
