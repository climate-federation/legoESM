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

import math
from dataclasses import dataclass

import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.vertical import OceanZStarCoordinate


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


# ---------------------------------------------------------------------
# Phase 2B — Analytical bathymetry (Appendix A; ported from
# vopikamm/DINO@v0.2.0 MY_SRC/usrdef_zgr.F90)
# ---------------------------------------------------------------------

def _smooth_step(x, a: float, b: float):
    """Quintic smooth step: 6t⁵ - 15t⁴ + 10t³ with t = (x-a)/(b-a) clipped.

    Paper eq A2; matches Zenodo `smooth_step` (zgr_lib lines 823-825).
    Returns 0 for x ≤ a, 1 for x ≥ b, smooth in between.
    """
    t = jnp.clip((x - a) / (b - a), 0.0, 1.0)
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def _exp_bathy(x, x_left: float, x_right: float, width: float,
               dist_lam: float, dist_taper: float):
    """Tapered exponential — paper eq A1, matches Zenodo `exp_bathy`.

    Three-piece piecewise function: 0 outside [x_left, x_right], rises
    via tapered exponential to 1 in the interior.
    """
    znorm = 1.0 + jnp.exp(-width / dist_lam)

    # Left taper region: x ∈ [x_left, x_left + dist_taper]
    s_left = _smooth_step(x, x_left, x_left + dist_taper)
    val_left = (1.0 - jnp.exp(-(x - x_left) / dist_lam) / znorm) * (1.0 - s_left) + s_left

    # Right taper region: x ∈ [x_right - dist_taper, x_right]
    s_right = 1.0 - _smooth_step(x, x_right - dist_taper, x_right)
    val_right = (1.0 - jnp.exp((x - x_right) / dist_lam) / znorm) * (1.0 - s_right) + s_right

    in_left = (x >= x_left) & (x < x_left + dist_taper)
    in_right = (x > x_right - dist_taper) & (x <= x_right)
    in_interior = (x >= x_left + dist_taper) & (x <= x_right - dist_taper)

    return jnp.where(
        in_left, val_left,
        jnp.where(in_right, val_right,
                  jnp.where(in_interior, jnp.ones_like(x),
                            jnp.zeros_like(x))),
    )


def _gauss_ring(lon, lat, lon0: float, lat0: float, ring_radius: float,
                dist_lam: float, depth_top: float, depth_bot):
    """Gaussian ring centered at (lon0, lat0) with given radius.

    Matches Zenodo `gauss_ring`:
      arg = (-x² - y² + 2·rad·sqrt(x²+y²) - rad²) / dist_lam²
          = -(sqrt(x²+y²) - rad)² / dist_lam²

    Returns depth_bot away from the ring, depth_top on the ring.
    `depth_bot` may be a scalar or an array (e.g., the underlying
    bathymetry, so the ring can only shoal — never deepen — a column).
    """
    x = lon - lon0
    y = lat - lat0
    r = jnp.sqrt(x**2 + y**2)
    arg = -(r - ring_radius) ** 2 / dist_lam ** 2
    return (depth_top - depth_bot) * jnp.exp(arg) + depth_bot


def dino_bathymetry(lon_deg, lat_deg, cfg: DINOConfig | None = None):
    """Build DINO basin bathymetry at the given (lon, lat) points.

    Ports the analytical bathymetry construction from Kamm et al. 2025
    Appendix A (eqs A1-A5) and the upstream NEMO source
    (vopikamm/DINO@v0.2.0 MY_SRC/usrdef_zgr.F90, nn_botcase=1, the
    "bowl_cosh" basin).

    The Mid-Atlantic Ridge is intentionally omitted (paper Sect 2.2;
    `ln_mid_ridge=.false.` in the upstream namelist).

    Sign convention: ``H_bathy`` is positive (depth below sea level),
    bounded between ``H_shallow`` (at coasts) and ``H_deep`` (in the
    deep interior), and ``H_sill`` along the Drake-passage sill ring.

    Parameters
    ----------
    lon_deg, lat_deg : array
        Same-shape arrays of longitudes/latitudes in degrees. Can be
        any rank (will broadcast).
    cfg : DINOConfig, optional
        DINO configuration; defaults to ``DINOConfig()``.

    Returns
    -------
    H_bathy : array
        Same shape as ``lon_deg``. Depth in meters (positive).
    """
    if cfg is None:
        cfg = DINOConfig()

    lon_deg = jnp.asarray(lon_deg)
    lat_deg = jnp.asarray(lat_deg)

    # Domain extents
    lon_min = cfg.lon_west_deg
    lon_max = cfg.lon_east_deg
    lat_min = -cfg.lat_max_deg
    lat_max = cfg.lat_max_deg

    width_lon = lon_max - lon_min     # = 50°
    cha_min = cfg.channel_lat_south_deg
    cha_max = cfg.channel_lat_north_deg
    cha_width = cfg.channel_width_deg  # = 20°

    # Length scales (degrees) — note the Mercator correction on the
    # meridional scale matches the Zenodo code, NOT the paper text.
    # The code computes dist_phi = cos(rad·phi_max) · rn_distLam, which
    # SHRINKS the meridional taper length at high latitudes. Paper text
    # writes s_phi = cos(rad·phi_max) · s_lambda, which is the inverse
    # convention. Trusting the code (produces Fig 1).
    dist_lam_deg = 1.0 / cfg.s_lambda_inv_deg                              # 3°
    dist_phi_deg = math.cos(math.radians(cfg.lat_max_deg)) * dist_lam_deg  # ~1.03°

    # ------------------------------------------------------------------
    # Channel modification (paper eq A4): inside the channel band, the
    # zonal walls vanish so the flow is re-entrant.
    # ------------------------------------------------------------------
    g_phi_cha = _exp_bathy(
        lat_deg, cha_min, cha_max,
        width=width_lon, dist_lam=dist_lam_deg, dist_taper=cha_width / 2.0,
    )
    g_lambda = _exp_bathy(
        lon_deg, lon_min, lon_max,
        width=width_lon, dist_lam=dist_lam_deg, dist_taper=cha_width / 2.0,
    )
    # In the channel band, force g_lambda = 1 (no zonal walls)
    g_lambda = g_lambda * (1.0 - g_phi_cha) + g_phi_cha

    # Meridional taper for the full N/S extent (paper eq A3)
    g_phi = _exp_bathy(
        lat_deg, lat_min, lat_max,
        width=width_lon, dist_lam=dist_phi_deg, dist_taper=cha_width / 2.0,
    )

    # Bathymetry: deep at interior (g=1), shallow at boundaries (g=0)
    bathy = g_lambda * g_phi * (cfg.H_deep - cfg.H_shallow) + cfg.H_shallow

    # ------------------------------------------------------------------
    # Drake-passage sill (paper eq A5): Gaussian ring anchored at the
    # western wall, extending east into the channel. Only allowed to
    # SHOAL the column (depth_bot=bathy means it never deepens).
    # ------------------------------------------------------------------
    sill_taper = _smooth_step(
        lon_deg,
        cfg.sill_lon_m_deg,
        cfg.sill_lon_m_deg + cfg.sill_gaussian_width_s,
    )
    ring_radius = cha_width / 2.0  # zrad in Zenodo code
    sill = _gauss_ring(
        lon_deg, lat_deg,
        lon0=cfg.sill_lon_m_deg,
        lat0=cfg.sill_lat_m_deg,
        ring_radius=ring_radius,
        dist_lam=cfg.sill_gaussian_width_s,
        depth_top=cfg.H_sill,
        depth_bot=bathy,
    )
    bathy = sill_taper * sill + (1.0 - sill_taper) * bathy

    return bathy


# ---------------------------------------------------------------------
# Phase 2E — Vertical grid (Appendix C, eq C3)
# ---------------------------------------------------------------------

def _dino_stretching_coefficients(
    K_formula: int,
    H: float,
    dz_min: float,
    k_th: float,
    a_cr: float,
) -> tuple[float, float, float]:
    """Compute (a₀, a₁, a₂) for the Lévy 2010 / DINO tanh+ln(cosh) stretching.

    Paper eq C3 in the variable-name convention of NEMO's mi96_1d
    (vopikamm/DINO@v0.2.0 MY_SRC/zgr_lib.F90). The paper text says
    "K = 36 levels" (cell count) but writes K-1 in the formula
    denominators; the NEMO source clarifies that K in the formula is
    the *interface* count = n_levels + 1.

    Constraints:
      z(k=1)        = 0   (surface interface)
      z(k=K_formula) = H  (bottom interface)
      z(k=2) - z(k=1) ≈ dz_min  (top layer thickness)

    where z(k) = a₂ + a₁·k + a₀·a_cr·ln(cosh((k-k_th)/a_cr)).
    """
    Km1 = K_formula - 1
    th = math.tanh((1 - k_th) / a_cr)
    log_cosh_K = math.log(math.cosh((K_formula - k_th) / a_cr))
    log_cosh_1 = math.log(math.cosh((1 - k_th) / a_cr))
    denom = th - (a_cr / Km1) * (log_cosh_K - log_cosh_1)
    a0 = (dz_min - H / Km1) / denom
    a1 = dz_min - a0 * th
    a2 = -a1 - a0 * a_cr * log_cosh_1
    return a0, a1, a2


def _dino_depth_at_k(
    k: float, a0: float, a1: float, a2: float, k_th: float, a_cr: float,
) -> float:
    """Evaluate z(k) = a₂ + a₁·k + a₀·a_cr·ln(cosh((k-k_th)/a_cr))."""
    return a2 + a1 * k + a0 * a_cr * math.log(math.cosh((k - k_th) / a_cr))


def create_dino_z_star(cfg: DINOConfig | None = None) -> OceanZStarCoordinate:
    """36-level Lévy 2010 stretched z* grid for DINO (Appendix C).

    Top layer thickness ≈ ``dz_min`` (10 m); bottom layer is ~600 m.
    Surface interface is exactly 0; bottom interface is snapped to
    exactly ``-H_deep`` to absorb sub-meter rounding from the formula.

    Returns
    -------
    OceanZStarCoordinate
        With n_levels = cfg.n_levels (36 by default), H_max = cfg.H_deep.
    """
    if cfg is None:
        cfg = DINOConfig()

    n_levels = cfg.n_levels
    H = cfg.H_deep
    K_formula = n_levels + 1  # interface count (NEMO jpk convention)

    a0, a1, a2 = _dino_stretching_coefficients(
        K_formula=K_formula,
        H=H,
        dz_min=cfg.dz_min,
        k_th=float(cfg.k_th),
        a_cr=cfg.a_cr,
    )

    # Interfaces at integer k = 1, 2, ..., K_formula → n_levels+1 interfaces
    k_half = list(range(1, K_formula + 1))
    z_half_pos = [
        _dino_depth_at_k(float(k), a0, a1, a2, float(cfg.k_th), cfg.a_cr)
        for k in k_half
    ]

    # legoESM convention: z negative below surface, surface at index 0
    z_half_list = [-z for z in z_half_pos]
    z_half_list[0] = 0.0   # snap surface (formula gives ~1e-12 residue)
    z_half_list[-1] = -H   # snap bottom (formula gives sub-meter residue)
    z_half_ref = jnp.asarray(z_half_list)

    # Layer thicknesses (positive, surface-to-bottom)
    dz_ref = z_half_ref[:-1] - z_half_ref[1:]

    # Cell centers as midpoints of adjacent interfaces (matches legoESM
    # convention; paper formula evaluated at k+0.5 gives nearly the same
    # values but using midpoints keeps z_full_ref consistent with dz_ref).
    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])

    # Distance between adjacent full levels
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]

    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
    )


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
