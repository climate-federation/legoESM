"""Stomatal conductance models and coupled A-gs solver.

Three empirical / semi-empirical stomatal conductance models:

- ``ball_berry_gs``: Ball, Woodrow & Berry (1987) — RH-based.
- ``medlyn_gs``:     Medlyn et al. (2011) — VPD / optimality.
- ``jarvis_gs``:     Jarvis (1976) multiplicative — CO2-independent,
                     used when no Farquhar photosynthesis is active.

A fixed-point coupled Farquhar-stomata solver (``coupled_farquhar_stomata``)
provides the SimpleSEB path with a canopy-integrated GPP estimate using the
Leuning C3/C4 Farquhar model from ``canopy/photosynthesis.py``.  Phase 2
replaces the fixed-point iteration with a Newton solver to improve
convergence at high VPD.

All models return ``gs`` in ``mol H2O / m^2 / s`` at the leaf boundary.
``Ball-Berry`` and ``Medlyn`` take explicit ``m``/``g1`` and ``b0``/``g0``
parameters (not a config) so the two-leaf canopy scheme can pass
per-leaf ``(m_C3, b0_C3)`` values without constructing a new StomataConfig
per column.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure


# =====================================================================
# Unit conversion factors  (local constants for readability)
# =====================================================================

_PAR_FRAC = 0.48      # Fraction of shortwave that is PAR
_PAR_CONV = 4.6       # umol photons per J of PAR (roughly)
_MC = 12.0e-6         # g C per umol CO2


# =====================================================================
# Empirical stomatal conductance models
# =====================================================================

def ball_berry_gs(
    An: jax.Array,
    RH: jax.Array,
    Cs: jax.Array,
    m: jax.Array | float,
    b0: jax.Array | float,
) -> jax.Array:
    """Ball-Berry (1987) stomatal conductance.

    gs = b0 + m * max(An, 0) * RH / Cs   [mol H2O / m^2 / s]

    Parameters
    ----------
    An : net assimilation rate [μmol CO2 / m^2 / s]
    RH : relative humidity at the leaf surface [-]
    Cs : CO2 concentration at the leaf surface [μmol / mol]
    m  : Ball-Berry slope [-]; typically 9 for C3, 4 for C4
    b0 : Ball-Berry intercept / residual conductance [mol / m^2 / s];
         typically 0.01 for C3, 0.04 for C4

    Returns
    -------
    gs : stomatal conductance, lower-bounded by ``b0`` [mol H2O / m^2 / s]
    """
    A_pos = jnp.maximum(An, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    return jnp.maximum(b0 + m * A_pos * RH / Cs_safe, b0)


def medlyn_gs(
    An: jax.Array,
    VPD_kPa: jax.Array,
    Cs: jax.Array,
    g1: jax.Array | float,
    g0: jax.Array | float,
) -> jax.Array:
    """Medlyn et al. (2011) optimal (USO) stomatal conductance.

    gs = g0 + 1.6 * (1 + g1 / sqrt(VPD)) * max(An, 0) / Cs   [mol H2O / m^2 / s]

    Parameters
    ----------
    An      : net assimilation rate [μmol CO2 / m^2 / s]
    VPD_kPa : vapour pressure deficit at the leaf surface [kPa]
    Cs      : CO2 concentration at the leaf surface [μmol / mol]
    g1      : Medlyn slope [kPa^0.5]; typically 4 for C3, 1-2 for C4
    g0      : residual conductance [mol / m^2 / s]

    Returns
    -------
    gs : stomatal conductance, lower-bounded by ``g0`` [mol H2O / m^2 / s]
    """
    A_pos = jnp.maximum(An, 0.0)
    Cs_safe = jnp.maximum(Cs, 1.0)
    # Floor VPD to avoid divergence of 1/sqrt(VPD) as VPD → 0.
    VPD_safe = jnp.maximum(VPD_kPa, 0.05)
    return jnp.maximum(
        g0 + 1.6 * (1.0 + g1 / jnp.sqrt(VPD_safe)) * A_pos / Cs_safe,
        g0,
    )


def jarvis_gs(
    T: jax.Array,
    sw_down: jax.Array,
    q_air: jax.Array,
    p_surface: jax.Array,
    beta_soil: jax.Array,
    config,
) -> jax.Array:
    """Jarvis (1976) multiplicative stomatal conductance.

    ``gs = gs_max * f(PAR) * f(T) * f(VPD) * f(soil moisture)``

    Independent of CO2.  Used as the stomatal path when no Farquhar
    photosynthesis is active (carbon cycle off OR canopy EB off).

    ``config`` is a ``StomataConfig`` NamedTuple — Jarvis uses its scalar
    parameters directly because the slab+SEB path does not carry per-column
    Jarvis coefficients.
    """
    PAR = _PAR_FRAC * sw_down
    f_PAR = PAR / (PAR + config.K_PAR + 1e-10)

    T_C = T - 273.15
    dT = (T_C - config.T_opt_jarvis) / config.T_range_jarvis
    f_T = jnp.maximum(1.0 - dT ** 2, 0.0)

    e_sat = saturation_vapor_pressure(T)
    e_air = q_air * p_surface / (constants.epsilon + q_air)
    VPD_hPa = jnp.maximum(e_sat - e_air, 0.0) / 100.0
    f_VPD = jnp.clip(1.0 - config.a_vpd * VPD_hPa, 0.01, 1.0)

    f_soil = jnp.clip(beta_soil, 0.0, 1.0)
    return config.gs_max * f_PAR * f_T * f_VPD * f_soil


# =====================================================================
# Effective beta for latent heat coupling
# =====================================================================

def compute_stomatal_beta(
    gs: jax.Array,
    LAI: jax.Array | None,
    beta_soil: jax.Array,
    config,
) -> jax.Array:
    """Convert stomatal conductance to an effective moisture factor.

    Blends bare-soil evaporation (limited by ``beta_soil``) with canopy
    transpiration (limited by ``gs``).  When ``LAI`` is available, a
    Beer-law canopy fraction weights the two components.  When ``LAI``
    is ``None``, the result is the more limiting of soil and stomata.
    """
    beta_canopy = jnp.clip(gs / config.gs_ref, 0.0, 1.0)

    if LAI is not None:
        f_canopy = 1.0 - jnp.exp(-config.k_ext * LAI)
        beta_eff = (1.0 - f_canopy) * beta_soil + f_canopy * beta_canopy
    else:
        beta_eff = jnp.minimum(beta_soil, beta_canopy)

    return jnp.clip(beta_eff, 0.0, 1.0)


# =====================================================================
# Coupled Farquhar (Leuning) + stomata solver — damped Newton
# =====================================================================

def coupled_farquhar_stomata(
    T_leaf: jax.Array,
    sw_down: jax.Array,
    co2_ppmv: jax.Array | float,
    q_air: jax.Array,
    p_surface: jax.Array,
    LAI: jax.Array,
    beta_soil: jax.Array,
    config,
    TgC: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Solve the coupled Leuning Farquhar + Ball-Berry/Medlyn system.

    Uses the **Leuning C3 + Q10 C4** Farquhar model from
    ``canopy/photosynthesis.py`` with a continuous ``fC4`` fraction for
    mixed C3/C4 canopies.  Solves the scalar-per-column diffusion
    constraint

        F(Ci) = Ci - (Ca - 1.6 · max(A_n(Ci), 0) / max(gs(A_n(Ci)), g0)) = 0

    via damped Newton iteration with the element-wise derivative
    ``dF/dCi`` extracted by a single ``jax.jvp`` call per step
    (``F`` is element-wise, so a unit-tangent JVP gives the diagonal of
    the Jacobian in O(n)).  Replaces the earlier fixed-point iteration
    which stalls or oscillates at high VPD because the Medlyn
    ``1/sqrt(VPD)`` term amplifies small Ci changes into large gs
    changes.

    Soil moisture stress is applied as a multiplicative down-regulation
    of ``Vcmax25`` (CLM / Bonan et al. 2011).

    Parameters
    ----------
    T_leaf    : leaf / surface temperature [K]
    sw_down   : downward shortwave radiation [W / m^2]
    co2_ppmv  : atmospheric CO2 [μmol / mol]
    q_air     : near-surface specific humidity [kg / kg]
    p_surface : surface pressure [Pa]
    LAI       : leaf area index [m^2 / m^2]
    beta_soil : soil moisture availability factor [0-1]
    config    : StomataConfig (provides ``stomata_model``, ``n_iter_ags``,
                ``Vcmax25_C3/C4``, ``alf_C3``, ``fC4``, ``g0``, ``g1_bb``,
                ``g1_med``, ``TgC_default``, ``k_ext``)
    TgC       : 30-day mean growth temperature [°C].  If ``None``,
                ``config.TgC_default`` is broadcast to ``T_leaf.shape``.

    Returns
    -------
    gs : canopy stomatal conductance [mol H2O / m^2 / s]
    gpp: gross primary production [gC / m^2 / s]
    """
    # Deferred import avoids a circular dependency: photosynthesis.py is a
    # sibling module under canopy/.
    from legoesm.land.canopy.photosynthesis import photosynthesis as _leuning_mixed

    PAR_umol = _PAR_FRAC * sw_down * _PAR_CONV
    fAPAR = 1.0 - jnp.exp(-config.k_ext * LAI)
    APAR_umol = fAPAR * PAR_umol

    Ca = jnp.broadcast_to(
        jnp.asarray(co2_ppmv, dtype=T_leaf.dtype), T_leaf.shape)

    e_sat = saturation_vapor_pressure(T_leaf)
    e_air = q_air * p_surface / (constants.epsilon + q_air)
    VPD_kPa = jnp.maximum(e_sat - e_air, 0.0) / 1000.0
    RH = jnp.clip(e_air / jnp.maximum(e_sat, 1.0), 0.0, 1.0)

    if TgC is None:
        TgC = jnp.broadcast_to(
            jnp.asarray(config.TgC_default, dtype=T_leaf.dtype), T_leaf.shape)

    beta_safe = jnp.clip(beta_soil, 0.01, 1.0)
    Vcmax25_C3_eff = jnp.broadcast_to(
        config.Vcmax25_C3 * beta_safe, T_leaf.shape)
    Vcmax25_C4_eff = jnp.broadcast_to(
        config.Vcmax25_C4 * beta_safe, T_leaf.shape)

    fC4 = jnp.broadcast_to(
        jnp.asarray(config.fC4, dtype=T_leaf.dtype), T_leaf.shape)
    alf = jnp.broadcast_to(
        jnp.asarray(config.alf_C3, dtype=T_leaf.dtype), T_leaf.shape)

    def _An(Ci):
        return _leuning_mixed(
            T_leaf, Ci, APAR_umol,
            Vcmax25_C3_eff, Vcmax25_C4_eff,
            fC4, p_surface, alf, TgC,
        )

    def _gs_from_An(An):
        if config.stomata_model == "medlyn":
            return medlyn_gs(An, VPD_kPa, Ca, config.g1_med, config.g0)
        return ball_berry_gs(An, RH, Ca, config.g1_bb, config.g0)

    def _F(Ci):
        """Residual F(Ci) = Ci - (Ca - 1.6 · max(A_n,0) / max(gs, g0)).

        Element-wise in ``Ci`` because ``_An`` and ``_gs_from_An`` are
        element-wise, so a unit-tangent ``jax.jvp`` returns the diagonal
        of the Jacobian in a single forward-mode pass.
        """
        A_net = _An(Ci)
        gs_local = _gs_from_An(A_net)
        gs_safe = jnp.maximum(gs_local, config.g0)
        A_pos = jnp.maximum(A_net, 0.0)
        return Ci - (Ca - 1.6 * A_pos / gs_safe)

    # Initial guess: chi = Ci/Ca ≈ 0.7 for C3, 0.4 for pure C4.
    chi0 = 0.7 - 0.3 * fC4
    Ci = chi0 * Ca

    # Damped Newton.  dF/dCi > 0 always for this residual (the diffusion
    # constraint monotonically increases with Ci), so we can safely floor
    # the derivative at a small positive value.  Damping factor 0.8 trades
    # quadratic convergence for global robustness at radiation-collapse
    # transitions and high-VPD regimes.
    DAMP = 0.8
    for _ in range(config.n_iter_ags):
        F_val, dF_dCi = jax.jvp(_F, (Ci,), (jnp.ones_like(Ci),))
        dF_safe = jnp.maximum(dF_dCi, 1e-3)
        Ci = Ci - DAMP * F_val / dF_safe
        Ci = jnp.clip(Ci, 1.0, 0.99 * jnp.maximum(Ca, 1.0))

    # Final evaluation at converged Ci.
    A_net = _An(Ci)
    gs = _gs_from_An(A_net)

    # Leuning ``photosynthesis`` returns An clipped at 0 (net assimilation),
    # so use it directly as a GPP proxy.  Strictly speaking GPP should be
    # gross of respiration, but the canopy two-leaf path uses the same
    # convention, so both SEB and canopy surfaces report consistent
    # ``co2_flux`` magnitudes to the carbon cycle.
    gpp = jnp.maximum(A_net, 0.0) * _MC
    return gs, gpp
