"""Configuration for ocean surface forcing schemes."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants


class PrescribedForcingConfig(NamedTuple):
    """Fixed wind stress and heat/freshwater fluxes."""
    tau_x: float = 0.0           # Zonal wind stress [N/m^2]
    tau_y: float = 0.0           # Meridional wind stress [N/m^2]
    Q_net: float = 0.0           # Net surface heat flux [W/m^2] (+ into ocean)
    E_minus_P: float = 0.0       # Evaporation minus precipitation [m/s]
    wind_profile: str = "constant"   # "constant", "cosine_latitude", "single_gyre", "double_gyre", "double_gyre_sin2", "double_gyre_tapered", or "global_wind"
    tau_max: float = 0.1         # Max wind stress for wind profiles [N/m^2]
    tropical_wind_scale: float = 1.0  # Scale factor for wind stress within
                                       # ±tropical_wind_lat_deg of equator.
                                       # 1.0 = no change (default).
                                       # 0.5 = halve tropical winds.
                                       # Tapers smoothly via Gaussian to
                                       # preserve continuity at the edges.
    tropical_wind_lat_deg: float = 15.0  # Half-width of tropical reduction
                                          # band [degrees].
    lat_south_deg: float = 15.0  # Southern basin boundary [degrees]
    lat_north_deg: float = 75.0  # Northern basin boundary [degrees]
    wind_buffer_deg: float = 0.0  # Buffer zone width [degrees] where wind tapers to zero at basin edges


class RestoringConfig(NamedTuple):
    """SST/SSS restoring to target profiles.

    Three modes for the restoring target, selected per tracer:

    1. **Built-in cosine** (default): ``T_star = T_eq − (T_eq − T_pole)·sin²(lat)``,
       ``S_star = S_star (constant)``. Set ``T_profile="cosine"`` and leave
       ``T_star_array`` / ``S_star_array`` as None.
    2. **Built-in constant**: ``T_star = T_eq`` everywhere. Set
       ``T_profile="constant"``.
    3. **Arbitrary user-provided 2D arrays**: pass ``T_star_array`` and/or
       ``S_star_array`` (matching grid horizontal shape). Used directly
       as the target. Lets any experiment supply paper-equation profiles
       (e.g., DINO eq B1/B2 with equatorial Gaussian dip), observed
       climatologies, etc. When provided, ``T_profile`` is ignored.

    Two stability options, both default off for back-compat:

    - **``subtract_qsr=True``**: subtract ``OceanSurfaceForcing.sw_down``
      from the restoring tendency in the surface layer. Implements the
      standard NEMO/DINO eq 8 non-solar split (paper:
      ``q_ns = A_θ(T*-T) − Q_sr``).
    - **``implicit=True``**: integrate restoring with analytical implicit
      Euler instead of forward Euler. Stable for any dt — needed for
      strong restoring + strong vertical mixing combinations
      (DINO-like). Returns an effective tendency
      ``(T_new - T_old) / dt`` so the dycore's existing
      forward-Euler tracer update gives the right answer.
    """
    tau_T: float = 2592000.0     # Temperature restoring timescale [s] (30 days)
    tau_S: float = 2592000.0     # Salinity restoring timescale [s] (30 days)
    T_star_eq: float = 25.0      # Target equatorial SST [degC] (cosine mode)
    T_star_pole: float = 0.0     # Target polar SST [degC] (cosine mode)
    S_star: float = 35.0         # Target SSS [PSU] (constant mode)
    T_profile: str = "cosine"    # "cosine" or "constant" (ignored if T_star_array given)
    # Optional arbitrary 2D targets — when provided override the built-in formulas
    T_star_array: object = None  # jax.Array | None, shape matching grid horizontal
    S_star_array: object = None  # jax.Array | None, shape matching grid horizontal
    # Stability options (back-compat defaults)
    subtract_qsr: bool = False   # eq 8 non-solar split — see docstring
    implicit: bool = False       # analytical implicit-Euler integration


def tau_from_flux_coefficient(A: float, rho_0: float, c_p_or_one: float,
                               dz_0: float) -> float:
    """Convert a paper-style heat- or salt-flux coefficient to a restoring
    timescale (seconds) the way NEMO derives ``τ_T = ρ₀·c_p·Δz_0/A_Θ``
    and ``τ_S = ρ₀·Δz_0/A_S``.

    For temperature: ``A`` is in W/m²/K, pass ``c_p_or_one = c_p``.
    For salinity:    ``A`` is in kg/m²/s, pass ``c_p_or_one = 1.0``.
    """
    return rho_0 * c_p_or_one * dz_0 / A


class BulkFormulaConfig(NamedTuple):
    """COARE-like air-sea flux formulation."""
    C_D: float = 1.5e-3     # Drag coefficient (constant scheme)
    C_H: float = 1.5e-3     # Sensible heat transfer coefficient (constant)
    C_E: float = 1.5e-3     # Latent heat transfer coefficient (constant)
    rho_a: float = constants.rho_air
    c_pa: float = constants.c_pd
    L_v: float = constants.L_v
    T_a: float = 280.0      # Air temperature [K]
    U_a: float = 5.0        # Wind speed [m/s]
    q_a: float = 0.005      # Air specific humidity [kg/kg]
    SW_down: float = 200.0  # Downward shortwave [W/m^2]
    LW_down: float = 300.0  # Downward longwave [W/m^2]
    bulk_scheme: str = "constant"  # "constant", "coare3", "large_yeager"
    z_ref: float = 10.0     # Reference height for MOST [m]
    z0: float = 1e-4        # Roughness length for MOST [m]
    bulk_n_iter: int = 5    # MOST iterations
    emissivity: float = 0.97  # Surface longwave emissivity


class SurfaceForcingConfig(NamedTuple):
    """Top-level surface forcing configuration.

    Use ``scheme="combined"`` to apply prescribed wind stress together
    with temperature/salinity restoring — needed for realistic
    baroclinic gyre experiments.
    """
    scheme: str = "none"  # "prescribed", "restoring", "combined", "bulk_formulas", "none"
    prescribed: PrescribedForcingConfig = PrescribedForcingConfig()
    restoring: RestoringConfig = RestoringConfig()
    bulk_formulas: BulkFormulaConfig = BulkFormulaConfig()
