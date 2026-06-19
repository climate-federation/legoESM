"""CATKE column physics — diffusivities, dissipation, surface TKE flux.

Pure, differentiable column functions implementing the CATKE
(Convective-Adjustment Turbulent-Kinetic-Energy) vertical-mixing closure of
Wagner et al. (2025, JAMES, doi:10.1029/2024MS004522), faithful to the
Oceananigans ``CATKEVerticalDiffusivity`` reference implementation.

CATKE is a one-equation PROGNOSTIC-TKE closure (distinct from the Gaspar/
Burchard :mod:`...tke`): given a TKE field ``e``, it derives eddy diffusivities
``K_X = l_X * w*`` (``w* = sqrt(max(e, e_min))``) with a DYNAMIC convective
mixing length (Deardorff ``w*^3/Jb`` scaling — predicting both convective-layer
depth and timescale) and a Richardson-number stability function that blends
high / low / negative-Ri regimes with SEPARATE coefficients for momentum (u),
tracers (c) and TKE (e).

This module is the grid-agnostic NUMERICAL core: every input is supplied at the
vertical "interface" level where diffusivities live (between cell centres),
shape ``(..., n_iface)``, plus the per-column depth ``H`` and surface buoyancy
flux ``Jb``.  The prognostic TKE evolution (the implicit ``e`` solve + surface
flux boundary condition + carrying ``e`` on the ocean state) is wired in the
dispatch layer (a separate step), so these functions can be unit-tested on
synthetic profiles with NO grid / EOS coupling.  ``N2_above`` (the buoyancy
frequency of the SHALLOWER neighbour interface, used by the entrainment branch)
is an explicit argument so this core makes no assumption about the caller's
vertical level ordering.

All coefficients come from :class:`CATKEConfig`; the only bare literals are
exempt math constants / safety floors.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig

__physics_contract__ = {
    "units": {
        "e": "m^2/s^2", "N2": "1/s^2", "S2": "1/s^2", "Jb": "m^2/s^3",
        "K_u": "m^2/s", "K_c": "m^2/s", "K_e": "m^2/s",
        "dissipation_rate": "1/s", "surface_tke_flux": "m^3/s^3",
    },
    "signs": (
        "diffusivities K_X >= 0; convective columns (Jb>0 & N2<0) enhance K via "
        "the Deardorff length; dissipation_rate >= 0; surface_tke_flux <= 0 "
        "(a downward TKE flux INTO the column)."
    ),
    "conserves": "none (computes diffusivities/dissipation; the TKE budget is "
                 "closed by the prognostic solve in the dispatch layer)",
    "differentiable": True,
    "reference": "Wagner et al. (2025), JAMES, doi:10.1029/2024MS004522; "
                 "Oceananigans CATKEVerticalDiffusivity",
    "idealized_test": "tests/ocean/unit/test_catke.py — stratified column gives "
                      "near-background K; convective column gives enhanced K; "
                      "differentiable wrt e/N2.",
}

# Safety floors (exempt numerics — NOT tunable).
_EPS_S2 = 1.0e-20      # shear floor so Ri = N2/S2 is finite at rest
_EPS_LEN = 1.0e-20     # length floor so 1/l is finite


def _turbulent_velocity(e, minimum_tke):
    """w* = sqrt(max(e, e_min)).  The floor gives a background mixing rate."""
    return jnp.sqrt(jnp.maximum(e, minimum_tke))


def _step(x, c, w):
    """Piecewise-linear ramp: 0 for x<c, 1 for x-c>w, linear between."""
    return jnp.clip((x - c) / w, 0.0, 1.0)


def _stability_function(Ri, c_un, c_lo, c_hi, c_ri_lower, c_ri_width):
    """Ri-dependent stability function blending negative/low/high-Ri regimes.

    Ri < 0  -> c_un;  Ri >= 0 -> c_lo + (c_hi - c_lo) * step(Ri, c_ri_lower,
    c_ri_width).  (Oceananigans ``scale``.)
    """
    sigma_pos = c_lo + (c_hi - c_lo) * _step(Ri, c_ri_lower, c_ri_width)
    return jnp.where(Ri < 0.0, c_un, sigma_pos)


def _stable_length(w_star, N2, depth, height_above_bottom, c_surface, c_bottom):
    """Stable (shear) length: min(c_s*depth, c_b*hab, w*/sqrt(N2+))."""
    d = jnp.minimum(c_surface * depth, c_bottom * height_above_bottom)
    N2p = jnp.maximum(N2, 0.0)
    # Stratification length w*/sqrt(N2); unbounded (-> d) where N2<=0.
    ell_N = jnp.where(N2p > 0.0, w_star / jnp.sqrt(N2p + _EPS_LEN), jnp.inf)
    return jnp.minimum(d, ell_N)


def _convective_length(w_star, N2, N2_above, S2, depth, Jb,
                       c_conv, c_entr, c_sheared, jb_eps):
    """Dynamic convective / entrainment length (Deardorff scaling).

    Convecting (Jb>jb_eps & N2<0): l = c_conv*w*^3/(Jb+jb_eps)*(1 - c_sheared*Ri_f),
    Ri_f = depth*S2*w*/(Jb+jb_eps).  Entraining (Jb>jb_eps & N2>0 & N2_above<0):
    l = c_entr*Jb/(w* N2 + jb_eps).  Otherwise 0.
    """
    convecting = (Jb > jb_eps) & (N2 < 0.0)
    entraining = (Jb > jb_eps) & (N2 > 0.0) & (N2_above < 0.0)

    # Branch-safe denominators: JAX evaluates BOTH where-branches for the
    # gradient, so the masked-out branch must never divide by ~0 (Jb+jb_eps -> 0
    # for stabilising flux Jb ~ -jb_eps; w*·N2+jb_eps -> 0 for N2 ~ -jb_eps/w*).
    # Substituting 1.0 where the branch is inactive keeps the value finite and
    # the gradient clean (codex review).
    conv_den = jnp.where(convecting, Jb + jb_eps, 1.0)
    Ri_f = depth * S2 * w_star / conv_den
    ell_c = jnp.maximum(
        c_conv * w_star ** 3 / conv_den * (1.0 - c_sheared * Ri_f), 0.0)

    ent_den = jnp.where(entraining, w_star * N2 + jb_eps, 1.0)
    ell_e = c_entr * Jb / ent_den

    return jnp.where(convecting, ell_c, jnp.where(entraining, ell_e, 0.0))


def _variable_mixing_length(w_star, Ri, N2, N2_above, S2, depth,
                            height_above_bottom, H, Jb,
                            c_un, c_lo, c_hi, c_conv, c_entr, cfg):
    """Per-variable mixing length: min(H, max(sigma*l_stable, l_convective))."""
    sigma = _stability_function(Ri, c_un, c_lo, c_hi,
                                cfg.c_ri_lower, cfg.c_ri_width)
    ell_star = sigma * _stable_length(
        w_star, N2, depth, height_above_bottom,
        cfg.c_surface_shear, cfg.c_bottom_shear)
    ell_h = _convective_length(
        w_star, N2, N2_above, S2, depth, Jb,
        c_conv, c_entr, cfg.c_sheared_plume,
        cfg.minimum_convective_buoyancy_flux)
    ell = jnp.maximum(ell_star, ell_h)
    return jnp.minimum(H, ell)


def _richardson(N2, S2):
    """Gradient Richardson number Ri = N2 / S2 (shear-floored; sign = sign N2)."""
    return N2 / jnp.maximum(S2, _EPS_S2)


def catke_diffusivities(e, N2, N2_above, S2, depth, height_above_bottom, H, Jb,
                        cfg: CATKEConfig):
    """CATKE eddy viscosity / diffusivities at interfaces from a TKE field.

    Parameters (all ``(..., n_iface)`` unless noted)
    ------------------------------------------------
    e : turbulent kinetic energy [m^2/s^2].
    N2 : squared buoyancy frequency [1/s^2] (signed; <0 = unstable).
    N2_above : ``N2`` of the SHALLOWER neighbour interface (entrainment branch).
    S2 : squared vertical shear |du/dz|^2+|dv/dz|^2 [1/s^2].
    depth : distance below the surface [m] (>=0).
    height_above_bottom : distance above the sea floor [m] (>=0).
    H : column depth [m] (broadcastable; per-column scalar).
    Jb : surface buoyancy flux [m^2/s^3] (>0 destabilising; per-column scalar).
    cfg : CATKEConfig.

    Returns
    -------
    (K_u, K_c, K_e) : each ``(..., n_iface)`` [m^2/s] — momentum viscosity,
        tracer diffusivity, TKE diffusivity.  Each clamped at its config cap.
    """
    w_star = _turbulent_velocity(e, cfg.minimum_tke)
    Ri = _richardson(N2, S2)

    ell_u = _variable_mixing_length(
        w_star, Ri, N2, N2_above, S2, depth, height_above_bottom, H, Jb,
        cfg.c_un_u, cfg.c_lo_u, cfg.c_hi_u, cfg.c_conv_u, cfg.c_entr_u, cfg)
    ell_c = _variable_mixing_length(
        w_star, Ri, N2, N2_above, S2, depth, height_above_bottom, H, Jb,
        cfg.c_un_c, cfg.c_lo_c, cfg.c_hi_c, cfg.c_conv_c, cfg.c_entr_c, cfg)
    ell_e = _variable_mixing_length(
        w_star, Ri, N2, N2_above, S2, depth, height_above_bottom, H, Jb,
        cfg.c_un_e, cfg.c_lo_e, cfg.c_hi_e, cfg.c_conv_e, cfg.c_entr_e, cfg)

    K_u = jnp.minimum(ell_u * w_star, cfg.maximum_viscosity)
    K_c = jnp.minimum(ell_c * w_star, cfg.maximum_tracer_diffusivity)
    K_e = jnp.minimum(ell_e * w_star, cfg.maximum_tke_diffusivity)
    return K_u, K_c, K_e


def catke_dissipation_rate(e, N2, N2_above, S2, depth, height_above_bottom, H,
                           Jb, cfg: CATKEConfig):
    """TKE linear dissipation rate ``omega`` such that ``eps = omega * e``.

    omega = sqrt(|e|) / l_D where l_D = min(H, max(l_stable/sigma_D, l_conv_D))
    with the ``c_*_diss`` regime coefficients.  For e<0 (oscillatory advection
    undershoot) returns the numerical damping rate 1/negative_tke_damping.
    """
    w_star = _turbulent_velocity(e, cfg.minimum_tke)
    Ri = _richardson(N2, S2)

    sigma_D = _stability_function(
        Ri, cfg.c_un_diss, cfg.c_lo_diss, cfg.c_hi_diss,
        cfg.c_ri_lower, cfg.c_ri_width)
    ell_star = _stable_length(
        w_star, N2, depth, height_above_bottom,
        cfg.c_surface_shear, cfg.c_bottom_shear)
    # Dissipation stable length is DIVIDED by the stability function.
    ell_star = ell_star / jnp.maximum(sigma_D, _EPS_LEN)
    ell_h = _convective_length(
        w_star, N2, N2_above, S2, depth, Jb,
        cfg.c_conv_diss, cfg.c_entr_diss, cfg.c_sheared_plume,
        cfg.minimum_convective_buoyancy_flux)
    ell_D = jnp.minimum(H, jnp.maximum(ell_star, ell_h))

    omega_physical = jnp.sqrt(jnp.abs(e)) / jnp.maximum(ell_D, _EPS_LEN)
    omega_numerical = 1.0 / cfg.negative_tke_damping_time_s
    return jnp.where(e < 0.0, omega_numerical, omega_physical)


def catke_surface_tke_flux(u_star, w_convective_cubed, cfg: CATKEConfig):
    """Surface TKE flux (downward, into the column): -c_w_ustar*u*^3 - c_w_conv*wConv^3.

    Parameters
    ----------
    u_star : friction velocity [m/s] (from the surface momentum stress).
    w_convective_cubed : cubed convective velocity scale [m^3/s^3]
        (``(Jb * mixed-layer depth)`` style; 0 in stably-forced columns).
    """
    return -cfg.c_w_ustar * u_star ** 3 - cfg.c_w_conv * w_convective_cubed
