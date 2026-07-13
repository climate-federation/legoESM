"""Ice-shelf basal-melt + cavity FW source (Holland & Jenkins 1999).

Antarctic ice shelves account for ~50 % of the continent's mass loss
through basal melting at the ice-ocean interface.  This module ports
the standard 3-equation thermodynamic boundary-layer parameterisation
(Hellmer & Olbers 1989, Jenkins 1991, Holland & Jenkins 1999) and a
simpler 1-equation linear scheme (Beckmann & Goosse 2003) for
configurations that prefer speed over accuracy.

Three-equation system at the ice-ocean interface:

    heat:        ρ_i · L_f · ṁ = ρ_w · c_w · γ_T · (T_a − T_b)
    salt:        ρ_i · S_b · ṁ = ρ_w · γ_S · (S_a − S_b)        (ice S = 0)
    freezing pt: T_b = a · S_b + b + c · p_ice

with unknowns (ṁ, T_b, S_b) given ambient (T_a, S_a) and ice-base
pressure p_ice.  Substituting the freezing-point + salt eqns into
the heat eqn yields a quadratic in ṁ that we solve in closed form.

Sign convention
---------------
``ṁ > 0`` = MELT (ice → water); ``ṁ < 0`` = FREEZE-ON of marine ice.
Freshwater into the ocean equals ``ρ_i · ṁ`` [kg/m²/s, positive INTO
ocean].  Heat extracted from the ocean equals
``ρ_w · c_w · γ_T · (T_a − T_b)`` [W/m², positive = ocean LOSES energy
to the cavity], matching the ice-tile convention in
``TileResponse.ocean_heat_extraction``.

References
----------
* Hellmer, H. H., & Olbers, D. J. (1989). A two-dimensional model
  for the thermohaline circulation under an ice shelf. *Antarctic
  Science*, 1(4), 325–336.
* Jenkins, A. (1991). A one-dimensional model of ice shelf-ocean
  interaction. *J. Geophys. Res.*, 96(C11), 20671–20677.
* Holland, D. M., & Jenkins, A. (1999). Modeling thermodynamic ice-
  ocean interactions at the base of an ice shelf. *J. Phys.
  Oceanogr.*, 29(8), 1787–1800.
* Beckmann, A., & Goosse, H. (2003). A parameterization of
  ice-shelf-ocean interaction for climate models. *Ocean Modelling*,
  5(2), 157–170.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants


__physics_contract__ = {
    "summary": (
        "Ice-shelf basal melt at the ice-ocean interface (Holland & Jenkins "
        "1999 three-equation thermodynamic boundary layer, or Beckmann-Goosse "
        "2003 one-equation): compute the melt rate, interface T/S, and the "
        "freshwater + heat fluxes it drives into the ocean cavity."
    ),
    "inputs": {
        "T_amb_C": "degC", "S_amb_PSU": "psu", "p_ice_dbar": "dbar",
        "config.gamma_T": "m/s", "config.gamma_S": "m/s",
    },
    "outputs": {
        "m_dot_m_s": "m/s", "T_b_C": "degC", "S_b_PSU": "psu",
        "freshwater_to_ocean": "kg/m^2/s", "heat_extracted_from_ocean": "W/m^2",
    },
    "sign_convention": (
        "m_dot > 0 = MELT (ice -> water), < 0 = freeze-on; melt is a freshwater "
        "SOURCE into the ocean (freshwater_to_ocean = rho_i*m_dot > 0) that "
        "dilutes salinity, and a latent-heat SINK "
        "(heat_extracted_from_ocean = rho_w*c_w*gamma_T*(T_a - T_b) > 0 = the "
        "ocean LOSES heat to the cavity); a cavity boundary source/sink, not "
        "interior-conservative; depths positive downward."
    ),
    # Adds freshwater + extracts latent heat at the cavity: a boundary
    # source/sink, not a conservative interior operator.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Holland, D. M. & Jenkins, A. (1999), JPO 29, 1787-1800; Jenkins "
        "(1991) JGR 96, 20671-20677; Beckmann & Goosse (2003), Ocean Modelling "
        "5, 157-170"
    ),
    "idealized_test": (
        "tests/unit/test_ice_shelf.py + "
        "tests/ocean/unit/test_isf_prescribed_melt.py — warm ambient water "
        "(T_a > T_freeze) melts (m_dot>0, freshwater in, heat out); T_a at the "
        "freezing point gives ~zero melt; freeze-on flips the signs."
    ),
}


__param_spec__ = {
    "IceShelfConfig": {
        "scheme_key": "ocean.ice_shelf",
        "excluded": {
            "freeze_a": "material: linear freezing-point salinity coeff (fixed)",
            "freeze_b": "material: linear freezing-point offset (fixed)",
            "freeze_c": "material: linear freezing-point depth coeff (fixed)",
        },
        "params": {
            "gamma_S": {"units": "1", "bounds": (1.6665e-07, 1.515e-06), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "three-equation ice-shelf melt (Gade/Holland-Jenkins)", "shape": None},
            "gamma_T": {"units": "1", "bounds": (3.3e-05, 0.0003), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "three-equation ice-shelf melt (Gade/Holland-Jenkins)", "shape": None},
            "melt_factor_linear": {"units": "1", "bounds": (6.6e-08, 6e-07), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "three-equation ice-shelf melt (Gade/Holland-Jenkins)", "shape": None},
        },
    },
}


class IceShelfConfig(NamedTuple):
    """Configuration for the ice-shelf cavity FW source.

    Defaults match Holland-Jenkins 1999 reference values.
    """
    # --- Scheme dispatch ---
    enabled: bool = False
    scheme: str = "three_equation"   # "three_equation" or "linear"

    # --- Turbulent exchange velocities ---
    # ``gamma_T`` and ``gamma_S`` carry units of m/s.  The standard
    # production choice is to derive them from a velocity-dependent
    # u* via ``γ = Γ · u*``, but for many idealised cavity studies a
    # constant value is used.  Defaults below match Holland-Jenkins
    # 1999 for an ambient current ~ 0.1 m/s + C_d = 2.5e-3.
    gamma_T: float = 1.0e-4          # thermal exchange velocity [m/s]
    # Provenance: Holland-Jenkins 1999 / ISOMIP+ use gamma_T/gamma_S ~= 35
    # (gamma_S ~= 2.9e-6 for this gamma_T). This default gives a ratio ~= 198
    # (gamma_S lower than the H-J reference, so the ratio sits higher); melt is
    # only weakly sensitive to gamma_S, so it is left as-is (within param_spec
    # bounds, possibly a deliberate calibration) — a future retune has the
    # reference ratio here.
    gamma_S: float = 5.05e-7         # salt exchange velocity [m/s]

    # --- Freezing-point linear coefficients (Jenkins 1991) ---
    # T_f(S, p) = a·S + b + c·p
    freeze_a: float = -5.73e-2       # [°C / PSU]
    freeze_b: float = 8.32e-2        # [°C]
    freeze_c: float = -7.61e-4       # [°C / dbar]

    # --- Densities + thermodynamics ---
    rho_ice: float = constants.rho_ice    # [kg/m³]
    rho_w: float = constants.rho_ocean    # [kg/m³]
    c_w: float = constants.c_sw           # [J/(kg·K)] seawater
    L_f: float = constants.L_f            # [J/kg] latent heat fusion

    # --- Linear-scheme parameter (Beckmann-Goosse 2003) ---
    # ``melt_factor_linear`` is the lumped ``γ_T · ρ_w · c_w / (ρ_i · L_f)``
    # — produces ṁ [m/s] from the thermal driving θ [°C].  Default
    # gives ~6 m/yr per K driving, consistent with B-G 2003.
    melt_factor_linear: float = 2.0e-7   # [m/(s·K)]


# ==============================================================================
# Freezing-point and pressure helpers
# ==============================================================================

def freezing_point_C(
    S_PSU: jnp.ndarray,
    p_dbar: jnp.ndarray,
    *,
    config: IceShelfConfig = IceShelfConfig(),
) -> jnp.ndarray:
    """In-situ freezing-point of seawater [°C] (Jenkins 1991 linear).

    ``T_f(S, p) = a · S + b + c · p`` with ``a = -5.73e-2 °C/PSU``,
    ``b = 8.32e-2 °C``, ``c = -7.61e-4 °C/dbar``.

    Pressure is in dbar (= ~depth in m for seawater).
    """
    return config.freeze_a * S_PSU + config.freeze_b + config.freeze_c * p_dbar


def ice_base_pressure_dbar(
    depth_m: jnp.ndarray,
) -> jnp.ndarray:
    """Hydrostatic pressure at the ice-shelf base [dbar].

    Approximation ``p ≈ 1 dbar/m`` (seawater density ≈ 1025 kg/m³,
    g = 9.81 m/s² → 1.0055 dbar/m).  Caller passes the ice-shelf
    draft in metres (positive downward).
    """
    return jnp.asarray(depth_m)


# ==============================================================================
# NEMO 'spe' prescribed melt (parametrised cavity, Mathiot et al. 2017)
# ==============================================================================

def isf_prescribed_melt_tendencies(
    S: jnp.ndarray,
    dz_live: jnp.ndarray,
    wet_cell: jnp.ndarray,
    fwf_kg_m2_s: jnp.ndarray,
    zmin_m: jnp.ndarray,
    zmax_m: jnp.ndarray,
    *,
    rho_0: float,
    c_sw: float,
    L_fus: float,
    config: IceShelfConfig = IceShelfConfig(),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NEMO ISF 'spe' prescribed-melt tendencies (isfparmlt.F90 spe case).

    Coordinate convention: depths positive DOWN, ``dz_live`` positive
    thickness, fluxes positive INTO the ocean.  Per column with melt
    ``fwf > 0`` [kg/m²/s], NEMO deposits over the depth band
    ``[zmin, zmax]`` (parametrised cavity):

    * ``pqoce = −fwf · L_fus`` — latent heat drawn from the band
      (COOLING: melting consumes heat; a heat sink, negative into-ocean).
    * ``pqhc  = +fwf · c_sw · T_frz`` — heat content of the melt water
      entering AT the in-situ freezing temperature (°C; a negative
      ``T_frz`` also cools).
    * salinity: melt water is fresh — virtual-salt dilution
      ``dS = −S · fwf/(ρ0·h)`` per band cell.
    * volume: ``+fwf/ρ0`` [m/s] free-surface source (sea level RISES).

    Each TBL cell takes the NEMO full+fraction share of the column
    fluxes, so the column budget closes exactly:
    ``ρ0·c_sw·Σ dT_k·h_k = pqoce + pqhc`` and
    ``ρ0·Σ dS_k·h_k = −fwf·⟨S⟩_w`` (locked by tests).

    TBL construction (EXACT NEMO isftbl semantics — isfpar.F90 +
    isftbl.F90, codex r3 #3):

    1. ``zmax`` is clamped to the column depth (isfpar: MIN(ztblmax,
       bathy)); ``ktop`` is the cell CONTAINING ``zmin`` (isf_tbl_ktop:
       last level whose top interface depth <= zmin) and ``zmin`` is
       SNAPPED UP to that cell's top interface.
    2. TBL thickness ``htbl = max(min(zmax − zmin_snap, depth below
       zmin_snap), e3t(ktop))`` — at least the top cell, at most the
       water below.
    3. ``kbot`` = first level where the cumulative thickness from
       ``ktop`` reaches ``htbl``; the bottom cell enters with the
       FRACTION ``(htbl − Σe3(ktop..kbot−1))/e3(kbot)`` (isf_tbl_lvl).
       Cells ``ktop..kbot−1`` enter FULL — no top/bottom geometric
       overlap weighting.
    4. The freezing point is evaluated PER LEVEL with the NEMO TEOS-10
       ``eos_fzp`` at the cell-centre depth and TBL-averaged with the
       same full+fraction weights over ``htbl`` (isf_tbl_avg), exactly
       NEMO's ``ztfrz3d`` → ``ztfrz`` construction.

    Returns ``(dT_dt [°C/s], dS_dt [PSU/s], eta_dot [m/s])``; all zero
    where ``fwf <= 0`` or the column is dry.
    """
    from legoesm.ocean.eos import nemo_eos_fzp

    S = jnp.asarray(S)
    dz = jnp.asarray(dz_live) * jnp.asarray(wet_cell)
    nlev = dz.shape[-1]
    fwf = jnp.maximum(jnp.asarray(fwf_kg_m2_s), 0.0)
    zmin = jnp.asarray(zmin_m)
    zmax = jnp.asarray(zmax_m)

    # Interface depths (positive down) from the live thicknesses.
    z_bot = jnp.cumsum(dz, axis=-1)
    z_top = z_bot - dz
    col_depth = z_bot[..., -1]

    # (1) clamp zmax to bathy; ktop = cell containing zmin; snap zmin.
    zmax_c = jnp.minimum(zmax, col_depth)
    # count wet interfaces at-or-above zmin among WET cells only (dry
    # cells have z_top == col_depth; exclude them so ktop stays wet).
    is_wet = dz > 0.0
    below = (z_top <= zmin[..., jnp.newaxis]) & is_wet
    ktop = jnp.maximum(jnp.sum(below, axis=-1) - 1, 0)
    ktop_idx = ktop[..., jnp.newaxis]
    zmin_snap = jnp.take_along_axis(z_top, ktop_idx, axis=-1)[..., 0]
    e3_ktop = jnp.take_along_axis(dz, ktop_idx, axis=-1)[..., 0]

    # (2) TBL thickness bounds (isfpar rhisf_tbl construction).
    htbl = jnp.maximum(
        jnp.minimum(zmax_c - zmin_snap, col_depth - zmin_snap), e3_ktop)

    # (3) kbot + bottom fraction (isf_tbl_lvl).  For k >= ktop the
    # cumulative thickness from ktop is z_bot_k − zmin_snap.
    idx = jnp.arange(nlev)
    cum = z_bot - zmin_snap[..., jnp.newaxis]
    reach = (cum >= htbl[..., jnp.newaxis] - 1.0e-9) & (idx >= ktop_idx)
    # first reaching level; fall back to the last wet cell if roundoff
    # leaves none (htbl <= water below zmin_snap by construction).
    kbot = jnp.where(jnp.any(reach, axis=-1),
                     jnp.argmax(reach, axis=-1),
                     jnp.sum(is_wet, axis=-1) - 1)
    kbot = jnp.maximum(kbot, ktop)
    kbot_idx = kbot[..., jnp.newaxis]
    e3_kbot = jnp.take_along_axis(dz, kbot_idx, axis=-1)[..., 0]
    cum_above = jnp.take_along_axis(
        z_top, kbot_idx, axis=-1)[..., 0] - zmin_snap
    frac = jnp.where(e3_kbot > 0.0, (htbl - cum_above)
                     / jnp.where(e3_kbot > 0.0, e3_kbot, 1.0), 0.0)
    frac = jnp.clip(frac, 0.0, 1.0)

    full = (idx >= ktop_idx) & (idx < kbot_idx)
    at_bot = idx == kbot_idx
    w = dz * full + frac[..., jnp.newaxis] * dz * at_bot   # Σw = htbl

    has_melt = (fwf > 0.0) & (col_depth > 0.0)
    htbl_safe = jnp.where(htbl > 0.0, htbl, 1.0)
    w_norm = jnp.where(has_melt[..., jnp.newaxis], w / htbl_safe[..., jnp.newaxis], 0.0)

    # (4) per-level NEMO eos_fzp at cell-centre depth, TBL-averaged.
    tfrz_lvl = nemo_eos_fzp(S, 0.5 * (z_top + z_bot))
    T_frz = jnp.sum(tfrz_lvl * w_norm, axis=-1)

    # Fluxes (isfparmlt spe): qoce = −fwf·L, qhc = +fwf·cp·tfrz.
    q_net = fwf * (c_sw * T_frz - L_fus)
    dz_safe = jnp.maximum(dz, 1.0e-10)
    dT_dt = q_net[..., jnp.newaxis] * w_norm / (rho_0 * c_sw) / dz_safe
    dS_dt = -S * fwf[..., jnp.newaxis] * w_norm / rho_0 / dz_safe
    eta_dot = jnp.where(has_melt, fwf / rho_0, 0.0)
    return dT_dt, dS_dt, eta_dot


# ==============================================================================
# Three-equation system (Holland & Jenkins 1999)
# ==============================================================================

class IceShelfMeltResult(NamedTuple):
    """Outputs of an ice-shelf basal-melt computation.

    ``m_dot`` is the rate of ice-thickness loss in metres/second
    (positive = melt, negative = freeze-on).  Multiply by ``ρ_ice``
    to obtain the equivalent freshwater mass flux into the ocean
    [kg/m²/s].
    """
    m_dot_m_s: jnp.ndarray            # [m/s] melt rate, sign-bearing
    T_b_C: jnp.ndarray                # [°C] interface temperature
    S_b_PSU: jnp.ndarray              # [PSU] interface salinity
    freshwater_to_ocean: jnp.ndarray  # [kg/m²/s, positive INTO ocean]
    heat_extracted_from_ocean: jnp.ndarray  # [W/m², positive = ocean LOSES heat]


def three_equation_melt(
    T_amb_C: jnp.ndarray,
    S_amb_PSU: jnp.ndarray,
    p_ice_dbar: jnp.ndarray,
    *,
    config: IceShelfConfig = IceShelfConfig(),
) -> IceShelfMeltResult:
    """Holland & Jenkins (1999) three-equation basal-melt parameterisation.

    Closed-form solution of the quadratic in ṁ obtained by
    substituting the freezing-point relation + the salt equation
    into the heat equation.  Assumes the ice has zero salinity and
    neglects the (small) sensible-heat ice-conduction term.

    Parameters
    ----------
    T_amb_C : array
        Ambient (cavity-water) temperature [°C].
    S_amb_PSU : array
        Ambient salinity [PSU].
    p_ice_dbar : array
        Pressure at the ice-shelf base [dbar].
    config : :class:`IceShelfConfig`

    Returns
    -------
    :class:`IceShelfMeltResult`
    """
    alpha = config.rho_w * config.c_w * config.gamma_T     # [W/(m²·K)]
    beta = config.rho_w * config.gamma_S                    # [kg/(m²·s)]
    gamma = config.rho_ice * config.L_f                     # [J/m³]
    delta = config.rho_ice                                  # [kg/m³]

    T_freeze_amb = freezing_point_C(S_amb_PSU, p_ice_dbar, config=config)
    theta = T_amb_C - T_freeze_amb                          # [°C]
    T_minus_bcp = T_amb_C - config.freeze_b - config.freeze_c * p_ice_dbar

    # Quadratic A·m² + B·m + C = 0 in ``m = m_dot``.
    A = gamma * delta
    B = gamma * beta - alpha * delta * T_minus_bcp
    C = -alpha * beta * theta

    # Standard quadratic root.  The physical root is ``(-B + √Δ) /
    # (2A)`` — it reduces to ``+α·β·θ / (γ·β) = α·θ / γ`` (i.e.
    # γ_T·c_w·ρ_w·θ / (ρ_i·L_f)) in the small-melt limit, matching
    # the Beckmann-Goosse linearisation.
    disc = B * B - 4.0 * A * C
    # AD-safe sqrt of the discriminant: ``d/dx sqrt(x) = 1/(2 sqrt(x))`` is
    # +inf at x=0, so ``sqrt(max(disc, 0))`` produces a NaN GRADIENT whenever
    # disc <= 0 (physical roots, freeze-on edge cases).  The double-``where``
    # keeps the primal BIT-IDENTICAL to ``sqrt(max(disc, 0))`` (sqrt(disc) for
    # disc > 0, exactly 0 for disc <= 0) while making the reverse pass finite.
    sqrt_disc = jnp.where(disc > 0.0, jnp.sqrt(jnp.where(disc > 0.0, disc, 1.0)), 0.0)
    m_dot = (-B + sqrt_disc) / (2.0 * A)

    # Salt balance: S_b = β·S_a / (δ·m + β).
    denom = delta * m_dot + beta
    denom_safe = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
    S_b = jnp.where(
        jnp.abs(denom) > 1e-30, beta * S_amb_PSU / denom_safe, S_amb_PSU,
    )
    T_b = freezing_point_C(S_b, p_ice_dbar, config=config)

    # Heat extracted from ocean by the cavity [W/m²].
    Q = alpha * (T_amb_C - T_b)
    # Freshwater INTO ocean [kg/m²/s].
    fw = config.rho_ice * m_dot

    return IceShelfMeltResult(
        m_dot_m_s=m_dot,
        T_b_C=T_b,
        S_b_PSU=S_b,
        freshwater_to_ocean=fw,
        heat_extracted_from_ocean=Q,
    )


# ==============================================================================
# Linear melt (Beckmann-Goosse 2003)
# ==============================================================================

def linear_melt(
    T_amb_C: jnp.ndarray,
    S_amb_PSU: jnp.ndarray,
    p_ice_dbar: jnp.ndarray,
    *,
    config: IceShelfConfig = IceShelfConfig(),
) -> IceShelfMeltResult:
    """Linear 1-equation basal-melt scheme (Beckmann-Goosse 2003).

    ``ṁ = melt_factor_linear · (T_amb − T_freeze(S_amb, p))``

    The lumped ``melt_factor_linear`` absorbs ``γ_T · c_w · ρ_w /
    (ρ_i · L_f)`` so the formula directly produces ṁ [m/s] from the
    thermal driving θ [°C].  Default ~6 m/yr/K (Beckmann-Goosse
    2003).  ``T_b`` is approximated as ``T_freeze(S_amb, p)`` (no
    interface salinity feedback) and ``S_b = S_amb`` (consistent
    with the no-salt-feedback assumption).

    Outputs match the 3-equation ``IceShelfMeltResult`` signature so
    callers can switch schemes via ``config.scheme`` without
    branching on the return type.
    """
    T_freeze_amb = freezing_point_C(S_amb_PSU, p_ice_dbar, config=config)
    theta = T_amb_C - T_freeze_amb
    m_dot = config.melt_factor_linear * theta
    fw = config.rho_ice * m_dot
    # Linearised heat budget: latent heat alone, no diffusive flux.
    Q = config.rho_ice * config.L_f * m_dot
    return IceShelfMeltResult(
        m_dot_m_s=m_dot,
        T_b_C=T_freeze_amb,
        S_b_PSU=S_amb_PSU,
        freshwater_to_ocean=fw,
        heat_extracted_from_ocean=Q,
    )


def compute_basal_melt(
    T_amb_C: jnp.ndarray,
    S_amb_PSU: jnp.ndarray,
    p_ice_dbar: jnp.ndarray,
    *,
    config: IceShelfConfig,
) -> IceShelfMeltResult:
    """Dispatch entry: route to the configured scheme.

    Disabled config (``enabled=False``) still returns a valid
    :class:`IceShelfMeltResult` with all fields zero so callers can
    add the freshwater + heat flux unconditionally.
    """
    if not config.enabled:
        zero = jnp.zeros_like(jnp.asarray(T_amb_C))
        return IceShelfMeltResult(
            m_dot_m_s=zero,
            T_b_C=freezing_point_C(S_amb_PSU, p_ice_dbar, config=config),
            S_b_PSU=jnp.asarray(S_amb_PSU),
            freshwater_to_ocean=zero,
            heat_extracted_from_ocean=zero,
        )
    if config.scheme == "three_equation":
        return three_equation_melt(
            T_amb_C, S_amb_PSU, p_ice_dbar, config=config,
        )
    if config.scheme == "linear":
        return linear_melt(
            T_amb_C, S_amb_PSU, p_ice_dbar, config=config,
        )
    raise ValueError(
        f"IceShelfConfig.scheme={config.scheme!r} not recognised; "
        "expected 'three_equation' or 'linear'."
    )
