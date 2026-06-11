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
    wind_profile: str = "constant"   # "constant", "cosine_latitude", "single_gyre", "double_gyre", "double_gyre_sin2", "double_gyre_tapered", "channel_sine", "global_wind", or "two_belt"
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
    # Minimum cell thickness [m] used to discriminate ocean vs land cells
    # for the inv_rho_dz / inv_rho_csw_dz mask.  1 mm is well below any
    # real partial-cell thickness (typical floors are O(m)) but well above
    # numerical noise from jacobian * dz_ref.
    min_wet_cell_thickness_m: float = 1.0e-3


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


class FluxFeedbackConfig(NamedTuple):
    """Veros-style "flux + feedback" surface forcing (global_4deg transfer).

    Surface-layer tendencies computed from per-step traced
    ``OceanSurfaceForcing`` channels (``q_prescribed`` [W/m²], ``q_feedback``
    [W/m²/K], ``T_feedback_target`` [°C], ``S_restore_target`` [PSU] —
    monthly interpolation lives in the driver/harness, NOT here):

        dT/dt = (q_prescribed + q_feedback·(T_target − T_surf))
                / (rho_0 · c_sw · dz_0)                       [K/s]
        dS/dt = (S_target − S_surf) / tau_restore_s           [PSU/s]

    with a simple sea-ice mask zeroing BOTH where the surface is below
    freezing AND the net heat flux is cooling (Veros global_4deg
    ``set_forcing_kernel``).

    Heat-capacity note: ``c_sw`` defaults to ``constants.c_sw`` (3994.0
    J/(kg·K), Gill 1982) which is NOT the value the Veros global_4deg setup
    kernel hardcodes (``cp_0 = 3991.86795711963``, a 0.05% mismatch). A
    Veros-faithful recipe must pass the Veros value explicitly; it is a
    setup-kernel literal, not a ``legoesm.constants`` candidate.

    Ice-threshold note: the default is the constants-derived
    ``T_freeze_ocean − T_freeze`` = −1.7999999999999545 °C, equal to Veros's
    literal ``−1.8`` only to ~4.6e-14 (float representation of the
    subtraction). It selects a comparison branch, so the gap is physically
    inert; documented for bit-level oracle work.

    Penetrative shortwave (Veros global_flexible / global_1deg ``qsol``):
    with ``penetrative_shortwave=True`` the scheme consumes the optional
    ``OceanSurfaceForcing.q_solar`` channel [W/m²] and deposits it through
    the water column via the SHARED two-band Jerlov kernel
    (``shortwave_penetration_tendency``; ``shortwave_water_type="I"`` is
    exactly the Veros literals R=0.58, ζ1=0.35 m, ζ2=23.0 m), converted with
    THIS config's ``c_sw``/``rho_0`` (Veros cp_0 ownership — not the eos
    module global).  Heat-ownership contract + ice gating are documented on
    ``OceanSurfaceForcing.q_solar``.  Passing ``q_solar`` while
    ``penetrative_shortwave=False`` raises (never a silent top-cell fallback
    or a silently ignored channel).
    """
    c_sw: float = constants.c_sw        # seawater specific heat [J/(kg·K)] — see note
    rho_0: float = constants.rho_ocean  # Boussinesq reference density [kg/m³]
    tau_restore_s: float = 2592000.0    # SSS restoring timescale [s] (Veros t_rest = 30 d)
    ice_mask: bool = True               # apply the simple sea-ice mask
    # Freezing threshold [°C] for the ice mask (Veros literal −1.8).
    ice_threshold_C: float = constants.T_freeze_ocean - constants.T_freeze
    # Minimum top-cell thickness [m] discriminating ocean vs land columns
    # (same convention as PrescribedForcingConfig).
    min_wet_cell_thickness_m: float = 1.0e-3
    # Penetrative-shortwave channel (Veros global_flexible/global_1deg qsol):
    # consume OceanSurfaceForcing.q_solar through the shared two-band Jerlov
    # column.  OFF by default ⇒ bit-identical existing paths.
    penetrative_shortwave: bool = False
    # Jerlov water type for the q_solar column ("I" ≡ the Veros setup
    # literals 0.58/0.35/23.0; see shortwave_penetration.JERLOV_TYPES).
    shortwave_water_type: str = "I"


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
    baroclinic gyre experiments.  Use ``scheme="external"`` to apply a
    coupler-provided ``OceanSurfaceForcing`` (tau / q_net / freshwater / salt)
    through the physics path — the cubed-sphere two-way coupling route.  NOTE
    ``external`` uses the ATMOSPHERE tau convention (ocean reaction = -tau),
    OPPOSITE the ``prescribed`` scheme (on-ocean +tau).  (MPAS uses a separate
    physics factory and does not yet dispatch ``external``.)  Use
    ``scheme="flux_feedback"`` for the Veros-style prescribed-flux +
    SST-feedback + SSS-restoring tracer forcing driven by traced
    ``OceanSurfaceForcing`` channels (wind stress still flows through the
    ``tau_x``/``tau_y`` channels in the dynamics seam, explicit).
    """
    scheme: str = "none"  # "prescribed","restoring","combined","bulk_formulas","external","flux_feedback","none"
    prescribed: PrescribedForcingConfig = PrescribedForcingConfig()
    restoring: RestoringConfig = RestoringConfig()
    bulk_formulas: BulkFormulaConfig = BulkFormulaConfig()
    flux_feedback: FluxFeedbackConfig = FluxFeedbackConfig()
