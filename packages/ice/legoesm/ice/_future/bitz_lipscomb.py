"""Bitz & Lipscomb (1999) energy-conserving multi-layer sea-ice thermodynamics.

**NOT YET INTEGRATED** — this module is the validated thermodynamic *core*
for replacing the single-skin-node (Semtner-0) ice base in
``ice/sea_ice.py`` (finding F-ICE-1).  It is not yet imported by any
production path; wire it in by:

1. adding a prognostic per-layer enthalpy field ``q_ice_layers`` (shape
   ``(..., n_ice_layers)``) to :class:`legoesm.ice.state.SeaIceState`
   (with a default so existing constructors are unaffected);
2. gating it in ``_thermo_single`` behind a ``config.thermo_scheme ==
   "bitz_lipscomb"`` branch — replacing the single-node
   ``skin_cap`` / constant-``k`` ``F_cond`` with
   :func:`bitz_lipscomb_conduction_step`, the surface energy balance
   supplying ``F_top`` and the ocean freezing point supplying
   ``T_bottom``;
3. advecting / ITD-remapping the layer enthalpy alongside the existing
   ``E = T·h·a`` sensible enthalpy in ``transport.py`` / ``itd.py``;
4. extending the restart I/O for the new field.

The distinctive BL99 physics versus Semtner-0:

* **Brine enthalpy.** Sea ice stores latent heat in brine pockets.  The
  enthalpy ``q(T,S) = −ρ_i[c_0(T_m−T) + L_0(1−T_m/T) − c_w T_m]`` (T, T_m
  in °C, melting temperature ``T_m = −μS``) makes the effective heat
  capacity ``c_i = c_0 + L_0 μ S / T²`` rise sharply near the melting
  point — internal heat storage a zero-layer skin cannot represent.
* **Salinity-dependent conductivity** ``k = k_0 + βS/T`` (Untersteiner
  1964): brine lowers conductivity, so warmer/saltier ice insulates
  more.
* **Exact energy conservation.** The enthalpy is updated by the implicit
  conductive-flux divergence, so the column enthalpy change equals the
  net boundary flux to machine precision.

References
----------
* Bitz, C. M. & Lipscomb, W. H. (1999). An energy-conserving
  thermodynamic model of sea ice. *J. Geophys. Res.*, 104, 15669–15677.
* Untersteiner, N. (1964). Calculations of temperature regime and heat
  budget of sea ice in the Central Arctic. *J. Geophys. Res.*, 69,
  4755–4766.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.timestepping.tridiagonal import thomas_solve

__all__ = (
    "freezing_temperature",
    "ice_enthalpy",
    "ice_temperature_from_enthalpy",
    "ice_thermal_conductivity",
    "ice_specific_heat",
    "bitz_lipscomb_conduction_step",
)

# Ice is always at or below its (salinity-depressed) melting point, so the
# internal temperature in °C is strictly negative.  Floor its magnitude
# away from zero before any ``1/T`` (brine enthalpy, conductivity, heat
# capacity) so those terms stay finite and differentiable as T → 0⁻.
_T_FLOOR_C = 1.0e-2          # [°C]
_K_MIN = 0.1                 # conductivity floor [W/m/K]
# Quadratic-discriminant floor. 0.0 is the exact degenerate double-root limit
# (sqrt(0)=0, root -> vertex -B/2c0) and absorbs any roundoff-negative
# discriminant; a POSITIVE floor (was 1.0) perturbs a valid near-degenerate root
# by forcing sqrt(disc) >= 1, so it is wrong — keep the floor at 0.
_DISC_FLOOR = 0.0


def _neg_temp_c(T_K):
    """Temperature in °C clamped strictly negative (``≤ −_T_FLOOR_C``).

    Keeps the brine ``1/T`` terms finite/differentiable for ice at or
    just below the melting point.
    """
    T_c = T_K - constants.T_freeze
    return -jnp.maximum(-T_c, _T_FLOOR_C)


def freezing_temperature(S):
    """Salinity-depressed melting temperature ``T_m = −μS`` [°C]."""
    return -constants.mu_ice_freeze * S


def ice_enthalpy(T_K, S):
    """BL99 sea-ice enthalpy ``q(T,S)`` [J/m³] (negative for ice).

    ``q = −ρ_i [ c_0 (T_m − T) + L_0 (1 − T_m/T) − c_w T_m ]`` with all
    temperatures in °C and ``T_m = −μS``.  The ``L_0(1 − T_m/T)`` term is
    the brine latent-heat storage that warms toward zero as ``T → T_m``.
    """
    c0 = constants.c_pi
    cw = constants.c_pw
    L0 = constants.L_f
    Tm = freezing_temperature(S)
    Tc = _neg_temp_c(T_K)
    return -constants.rho_ice * (
        c0 * (Tm - Tc) + L0 * (1.0 - Tm / Tc) - cw * Tm
    )


def ice_temperature_from_enthalpy(q, S):
    """Invert :func:`ice_enthalpy` for temperature [K] (BL99 quadratic).

    Solves ``c_0 T² + B T + C = 0`` with ``B = (c_w − c_0)T_m − L_0 −
    q/ρ_i`` and ``C = L_0 T_m`` for the physical root ``T ≤ T_m``.
    """
    c0 = constants.c_pi
    cw = constants.c_pw
    L0 = constants.L_f
    Tm = freezing_temperature(S)
    q_spec = q / constants.rho_ice                 # [J/kg]
    B = (cw - c0) * Tm - L0 - q_spec
    C = L0 * Tm
    disc = jnp.maximum(B * B - 4.0 * c0 * C, _DISC_FLOOR)
    T_c = (-B - jnp.sqrt(disc)) / (2.0 * c0)        # physical (lower) root
    return T_c + constants.T_freeze


def ice_thermal_conductivity(T_K, S):
    """Brine-dependent conductivity ``k = k_0 + βS/T`` [W/m/K] (Untersteiner).

    With ``T < 0`` the brine term is negative, so saltier/warmer ice
    conducts less; clamped to a positive floor for stability.
    """
    Tc = _neg_temp_c(T_K)
    k = constants.k_ice_default + constants.beta_ice_cond * S / Tc
    return jnp.maximum(k, _K_MIN)


def ice_specific_heat(T_K, S):
    """Effective specific heat ``c_i = c_0 + L_0 μ S / T²`` [J/kg/K].

    The ``dq/dT`` of :func:`ice_enthalpy`; the brine term diverges near
    the melting point (large near-surface heat storage).
    """
    Tc = _neg_temp_c(T_K)
    return constants.c_pi + constants.L_f * constants.mu_ice_freeze * S / (Tc * Tc)


def bitz_lipscomb_conduction_step(q_layers, h, S, F_top, T_bottom_K, dt):
    """One energy-conserving implicit conduction step over the ice column.

    Parameters
    ----------
    q_layers : (..., n_layers)
        Per-layer enthalpy [J/m³], top layer first.
    h : (...,)
        Ice thickness [m] (split into ``n_layers`` equal layers).
    S : (...,) or (..., n_layers)
        Bulk ice salinity [PSU].
    F_top : (...,)
        Net heat flux into the ice top [W/m²], positive downward
        (surface energy balance residual conducted into the ice).
    T_bottom_K : (...,)
        Ice-base temperature [K] (ocean freezing point — Dirichlet).
    dt : float
        Time step [s].

    Returns
    -------
    q_new : (..., n_layers)
        Updated layer enthalpy [J/m³] (energy-conserving update).
    T_new_K : (..., n_layers)
        Updated layer temperature [K].
    F_bottom : (...,)
        Conductive heat flux out of the base into the ocean [W/m²],
        positive downward.  ``Σ(q_new−q_old)·dz = (F_top − F_bottom)·dt``
        holds to machine precision.
    """
    n = q_layers.shape[-1]
    S_b = jnp.broadcast_to(S[..., None] if S.ndim == h.ndim else S, q_layers.shape)
    dz = jnp.maximum(h, 1e-3) / n  # coeff-ok: min ice-thickness floor for layer division [m]
    dz_e = dz[..., None]

    T_old = ice_temperature_from_enthalpy(q_layers, S_b)   # (..., n) [K]
    c_i = ice_specific_heat(T_old, S_b)                    # frozen at T_old
    k_c = ice_thermal_conductivity(T_old, S_b)             # layer-centre k

    # Harmonic-mean conductivity at the n−1 interior interfaces.
    k_iface = 2.0 * k_c[..., :-1] * k_c[..., 1:] / (
        k_c[..., :-1] + k_c[..., 1:]
    )                                                      # (..., n-1)
    heat_cap = constants.rho_ice * c_i * dz_e / dt         # (..., n) [W/m²/K]

    # Tridiagonal conductances g = k/dz between interior layers, plus a
    # half-cell bottom-boundary conductance to the Dirichlet T_bottom.
    g_int = k_iface / dz_e                                 # (..., n-1)
    g_bot = 2.0 * k_c[..., -1] / dz                        # (...,) base half-cell

    zero = jnp.zeros_like(heat_cap[..., :1])
    # Sub-diagonal a (coupling to layer above): 0 at top.
    a = jnp.concatenate([zero, -g_int], axis=-1)           # (..., n)
    # Super-diagonal c (coupling to layer below): 0 at bottom.
    c = jnp.concatenate([-g_int, zero], axis=-1)           # (..., n)
    # Main diagonal: heat capacity + sum of adjacent conductances.
    g_up = jnp.concatenate([zero, g_int], axis=-1)         # conductance above
    g_dn = jnp.concatenate([g_int, zero], axis=-1)         # conductance below
    b = heat_cap + g_up + g_dn
    b = b.at[..., -1].add(g_bot)                           # bottom Dirichlet link

    # RHS: explicit heat-capacity term + boundary fluxes.  The whole
    # system is in W/m² (``heat_cap`` and the conductances ``g`` carry
    # the 1/dz), so the surface flux enters directly (not divided by dz).
    d = heat_cap * T_old
    d = d.at[..., 0].add(F_top)                            # top flux BC [W/m²]
    d = d.at[..., -1].add(g_bot * T_bottom_K)              # bottom Dirichlet

    T_new = thomas_solve(a, b, c, d)                       # (..., n) [K]

    # Energy-conserving enthalpy update (flux form): q_new = q_old +
    # ρ c_i ΔT.  Summed over layers this telescopes to the net boundary
    # flux, so column energy is conserved to machine precision.
    q_new = q_layers + constants.rho_ice * c_i * (T_new - T_old)
    # Diagnostic basal conductive flux (positive downward, into ocean):
    # g_bot·(T_{n-1} − T_bottom) [W/m²].  With the flux-form enthalpy
    # update the interior fluxes telescope, leaving
    # Σ(q_new−q_old)·dz = (F_top − F_bottom)·dt exactly.
    F_bottom = g_bot * (T_new[..., -1] - T_bottom_K)
    return q_new, T_new, F_bottom

# NOTE (basal/surface growth-melt — deliberately NOT implemented here):
# The enthalpy-conserving freeze/melt at the ice base is the natural next
# BL99 process, but the enthalpy assigned to *newly-frozen* congelation
# ice is convention-dependent and changes the growth rate by a large
# factor: forming ice at the freezing point T_f (mostly brine, near-zero
# latent → fast growth, with the latent realized later via conduction)
# vs. an effective latent ``L_eff = L_f·(1 + μS/(T_f−T)) > L_f`` (slow
# growth).  An iter-37 prototype reproduced the fresh-ice Stefan rate
# exactly but gave *faster*-than-fresh growth for saline ice under the
# T_f convention — physically defensible but the opposite of the
# effective-latent convention.  Resolving which BL99/CICE uses needs the
# reference + a coupled ice-growth benchmark, so it is left for the
# integration step rather than shipped as uncertain physics.
