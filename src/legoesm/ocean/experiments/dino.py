"""DINO (DIabatic Neverworld Ocean) — replication of Kamm et al. (2025).

Reference
---------
Kamm, D., Deshayes, J., & Madec, G. (2025). DINO: a diabatic model of
pole-to-pole ocean dynamics to assess subgrid parameterizations across
horizontal scales. *Geosci. Model Dev.*, 18, 8091-8107.
https://doi.org/10.5194/gmd-18-8091-2025

NEMO reference implementation: https://github.com/vopikamm/DINO/tree/v0.2.0
Zenodo deposit: https://doi.org/10.5281/zenodo.15016824

Scope
-----
Replicates the DINO 1° (R1) configuration on lat-lon (Mercator) and
MPAS regional grids. The domain is a 50°-wide Atlantic-sector basin
spanning ~70°S to 70°N with a re-entrant channel between 45° and 65°S.

Phase 2A — DINOConfig dataclass.

Other phases live in this module: bathymetry (Phase 2B), surface
forcing (Phase 2C), initial conditions (Phase 2D), vertical grid
(Phase 2E), and dispatch wiring (Phase 2F).

See ``docs/ocean_experiments/dino_replication_plan.md`` for the full
plan, decisions log, and audit findings.
"""

from __future__ import annotations

from dataclasses import dataclass

from legoesm import constants


@dataclass
class DINOConfig:
    """Configuration for the DINO experiment (1° R1).

    All numeric defaults reflect the paper (Tables 1-2, Appendices A-D)
    and were cross-checked against the upstream NEMO namelist
    (vopikamm/DINO@v0.2.0, EXPREF/namelist_cfg, fetched 2026-05-14).
    See ``docs/ocean_experiments/dino_replication_plan.md`` for the
    decisions log.
    """

    # ------------------------------------------------------------------
    # Domain (Sect 2.2, Table 2)
    # ------------------------------------------------------------------
    lon_west_deg: float = -50.0    # western basin boundary
    lon_east_deg: float = 0.0      # eastern basin boundary
    lat_max_deg: float = 70.0      # symmetric N/S truncation latitude
    channel_lat_south_deg: float = -65.0  # southern edge of re-entrant channel
    channel_lat_north_deg: float = -45.0  # northern edge of re-entrant channel
    channel_width_deg: float = 20.0       # Δφ_c in eq A4 (= |lat_n - lat_s|)

    # ------------------------------------------------------------------
    # Reference physical constants used by the paper (Table 1)
    # Different from project defaults; kept as scheme parameters for
    # paper-fidelity replication (rho_0 differs from constants.rho_ocean
    # by ~0.1%, c_p differs from constants.c_sw by ~0.05%).
    # ------------------------------------------------------------------
    rho_0: float = 1026.0          # Boussinesq reference density [kg/m³]
    c_p: float = 3991.86           # Specific heat capacity [J/(kg·K)]

    # ------------------------------------------------------------------
    # Bathymetry (Appendix A, Zenodo namelist)
    # b = g_φ · g_λ · (H_deep - H_shallow) + H_shallow  per the
    # implementation note in the plan (deep at interior, shallow at
    # boundary). The naming here is intentionally clearer than the
    # paper's "H_max"/"H_min".
    # ------------------------------------------------------------------
    H_deep: float = 4000.0         # max depth (deep basin) [m]; namelist rn_H
    H_shallow: float = 2000.0      # min depth (at coast) [m]; namelist rn_hborder
    s_lambda_inv_deg: float = 1.0 / 3.0   # zonal slope param 1/rn_distLam = 1/3 deg⁻¹
    channel_wall_slope: float = 1.5       # rn_slp_cha (deg⁻¹) — channel-boundary slope

    # Sill (Scotia Ridge analog), eq. A5 + namelist
    sill_lon_m_deg: float = -50.0  # λ_m, anchored at western wall (Drake Passage)
    sill_lat_m_deg: float = -55.0  # φ_m, mid-channel latitude
    sill_gaussian_width_s: float = 4.0    # rn_ds_width [degrees]
    H_sill: float = 2500.0                # rn_ds_depth [m]

    # No Mid-Atlantic Ridge (paper Sect 2.2 explicitly excludes; namelist
    # rn_mr_* are test scaffolding — see decisions log 2026-05-14).

    # ------------------------------------------------------------------
    # Vertical grid (Appendix C, eq. C3)
    # z(k) = a₂ + a₁·k + a₀·a_cr·ln(cosh((k - k_th)/a_cr))
    # ------------------------------------------------------------------
    n_levels: int = 36             # K, number of full levels
    dz_min: float = 10.0           # Δz_min, top-layer thickness [m]
    k_th: int = 35                 # k_th, inflection level index
    a_cr: float = 10.5             # stretching parameter

    # ------------------------------------------------------------------
    # Surface forcing — wind (Sect 2.3, eq 7; piecewise cubic / PCHIP)
    # tau_u(φ) interpolated through the (lat, tau) knots below.
    # ------------------------------------------------------------------
    wind_tau_lats_deg: tuple = (-70.0, -45.0, -15.0, 0.0, 15.0, 45.0, 70.0)
    wind_tau_values: tuple = (0.0, 0.2, -0.1, -0.02, -0.1, 0.1, 0.0)  # [N/m²]

    # ------------------------------------------------------------------
    # Surface forcing — restoring (Sect 2.3, eqs 8-9; Appendix B eqs B1-B2)
    # Paper uses heat-flux coefficients (W/m²/K and kg/m²/s); the DINO
    # module converts these to timescales internally per layer thickness.
    # ------------------------------------------------------------------
    A_theta: float = 40.0          # T-restoring heat flux coefficient [W/m²/K]
    A_S: float = 3.858e-3          # S-restoring salt flux coefficient [kg/m²/s]
    L_phi_deg: float = 140.0       # meridional extent for cosine profile [deg]

    # Restoring profile values (annual-mean equivalents of eqs B1-B4)
    T_star_eq: float = 27.0        # equatorial target T [°C]
    T_star_n_mean: float = 5.0     # northern boundary target T (annual mean of B3)
    T_star_s_mean: float = -0.5    # southern boundary target T (annual mean of B4)
    S_star_eq: float = 37.25       # equatorial target S [g/kg]
    S_star_n: float = 35.0         # northern boundary target S [g/kg]
    S_star_s: float = 35.1         # southern boundary target S [g/kg]
    S_star_eq_dip_amp: float = 1.25       # equatorial Gaussian dip amplitude (eq B2)
    S_star_eq_dip_sigma_deg: float = 7.5  # equatorial Gaussian dip width [deg]

    # ------------------------------------------------------------------
    # Solar radiation (eq B5; annual-mean used in Phase 1, full seasonal
    # form deferred to Phase 5)
    # ------------------------------------------------------------------
    Q_sr_amp: float = 230.0                 # peak insolation [W/m²]
    solar_declination_amp_deg: float = 23.5  # declination amplitude (used by
                                            # full B5; annual mean integrates
                                            # over this internally)

    # Jerlov type I shortwave penetration (Sect 2.3 of paper). The
    # ShortwavePenetrationConfig in legoESM has matching defaults —
    # we just request water_type="I" downstream.
    jerlov_water_type: str = "I"

    # ------------------------------------------------------------------
    # Vertical mixing — KPP + enhanced-diffusion convection.
    # Paper uses TKE; we use KPP (decisions log 2026-05-14).
    # Background values match paper namelist (rn_avm0, rn_avt0).
    # ------------------------------------------------------------------
    A_v_bg: float = 1.2e-4         # background vertical viscosity [m²/s]
    K_v_bg: float = 1.2e-5         # background vertical diffusivity [m²/s]
    K_conv: float = 100.0          # enhanced-diffusion convective K [m²/s] (rn_evd)

    # ------------------------------------------------------------------
    # GM/Redi mesoscale eddy parameterization (Visbeck 1997; decisions
    # log: stay with Visbeck, do not implement Tréguier 1997).
    # ------------------------------------------------------------------
    use_gm_redi: bool = True
    visbeck_alpha: float = 0.015   # Visbeck dimensionless prefactor
    visbeck_kappa_min: float = 200.0      # κ_GM floor [m²/s]
    visbeck_kappa_max: float = 2000.0     # κ_GM ceiling [m²/s]
    redi_S_max: float = 0.005             # Redi slope tapering threshold
    gm_redi_slope_scheme: str = "triads"  # Griffies 1998 triads (matches NEMO iso-neutral)

    # ------------------------------------------------------------------
    # Lateral mixing of momentum (geopotential / iso-level Laplacian;
    # confirmed match with NEMO via Zenodo namelist 2026-05-14)
    # ------------------------------------------------------------------
    U_M: float = 0.27              # viscous velocity scale [m/s] (rn_Uv)
    # Tracer iso-neutral diffusion velocity scale (R1 only; eq below
    # Table 2). At R1 our GM/Redi handles this — we keep U_T for
    # reference but do not apply a separate harmonic tracer diffusion.
    U_T: float = 0.027             # tracer diffusivity velocity scale [m/s] (rn_Ut)

    # ------------------------------------------------------------------
    # Bottom drag (paper Sect 2.1 "nonlinear friction term"; namelist
    # rn_drag = 1e-3 quadratic).
    # ------------------------------------------------------------------
    C_d_bottom: float = 1.0e-3     # quadratic drag coefficient

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    dt: float = 2700.0             # baroclinic timestep [s] (45 min, Table 2)

    # ------------------------------------------------------------------
    # Numerical scheme selections (decisions log 2026-05-14):
    # - PGF: adcroft (closest to NEMO ln_hpg_sco standard Jacobian)
    # - Barotropic solver: implicit_cn (avoid checkerboard noise)
    # - Tracer advection: tvd (start simple; sensitivity study deferred)
    # - hi_precision_pressure: True
    # - Free-slip lateral BC: legoESM default (matches NEMO)
    # - Vector-invariant + AL81 EEN PV-flux: legoESM default
    # ------------------------------------------------------------------
    pgf_scheme: str = "adcroft"
    barotropic_solver: str = "implicit_cn"
    barotropic_implicit_theta_eta: float = 0.55
    tracer_advection: str = "tvd"
    hi_precision_pressure: bool = True

    # ------------------------------------------------------------------
    # Diagnostics (paper Figs 5-6: MOC and σ_2 referenced to 2000 m)
    # ------------------------------------------------------------------
    sigma_2_ref_depth: float = 2000.0     # reference depth for σ_2 [m]
    rho_ref_z0: float = 1026.0            # ρ_ref(z=0) [kg/m³]
    rho_ref_z2000: float = 1035.0         # ρ_ref(z=2000) [kg/m³]


# Convenience: surface-layer restoring timescales derived from heat-flux
# coefficients (eq 8 of paper). Useful for sanity printouts.
def restoring_timescale_T_days(cfg: DINOConfig) -> float:
    """τ_T = ρ₀ · c_p · Δz₀ / A_Θ, in days (≈ 11.85 d for default)."""
    seconds = cfg.rho_0 * cfg.c_p * cfg.dz_min / cfg.A_theta
    return seconds / 86400.0


def restoring_timescale_S_days(cfg: DINOConfig) -> float:
    """τ_S = ρ₀ · Δz₀ / A_S, in days (≈ 30.8 d for default)."""
    seconds = cfg.rho_0 * cfg.dz_min / cfg.A_S
    return seconds / 86400.0
