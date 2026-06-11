"""Diffusional growth (condensation/evaporation) of super-droplets.

Faithful JAX port of the ERF super-droplet growth ODE
(``ERF_SuperDropletPCMassChange.H`` ``dRsqdt`` + ``MassChange.cpp`` coefficient
setup). Each droplet's wet radius evolves by vapor diffusion (Maxwell-Mason),
modified by the Kelvin curvature and Raoult solute terms (Köhler theory) and a
Fukuta-Walter transitional (Knudsen) correction to the diffusivity.

The ODE is integrated in ``u = R²`` (radius squared), which is the natural
variable: for a pure droplet far from activation it grows linearly,
``d(R²)/dt = 2(S-1)/(F_k+F_d)``.

.. math::

    \\frac{d R^2}{dt}
       = \\frac{2(S-1)}{F_k+F_d}
       - \\frac{2\\,(a/T)}{F_k+F_d}\\,\\frac{1}{R}
       + \\frac{2\\,b\\,N_s}{F_k+F_d}\\,\\frac{1}{R^3}

with

* ``F_k = (L/(R_v T) - 1)·(L ρ_l)/(K T)``     (latent-heat conduction term),
* ``F_d = (ρ_l R_v T)/(d_cf·D·e_s)``          (vapor-diffusion term),
* ``d_cf = (1+Kn)/(1+2Kn(1+Kn))``, ``Kn = λ_v/R``, ``λ_v = 2D/√(8 T R_v/π)``,
* ``a = 2σ/(R_v ρ_l)``                         (Kelvin curvature coeff),
* ``b = (3/4π)(M_w/ρ_l)``                      (Raoult solute coeff),
* ``N_s = m_s · i / M_s``                       (effective solute moles).

All physical constants come from :mod:`legoesm.constants`; the saturation
vapor pressure ``e_s(T)`` comes from :mod:`legoesm.thermo` (audit-mandated:
no re-implemented saturation curve). This module is JIT- and grad-compatible.

References
----------
* Shima et al. (2009) QJRMS 135:1307-1320.
* Rogers & Yau (1989) *A Short Course in Cloud Physics*.
* Pruppacher & Klett (1997) *Microphysics of Clouds and Precipitation*.
* ERF ``Source/Particles/ERF_SuperDropletPCMassChange.H`` (oracle).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.particles import SuperDropletState

__physics_contract__ = {
    "summary": (
        "Super-droplet diffusional growth: vapor-diffusion (Maxwell-Mason) "
        "condensation/evaporation with Kelvin curvature and Raoult solute "
        "(Köhler) terms and a Knudsen diffusivity correction."
    ),
    "inputs": {
        "radius": "m",
        "S": "1",
        "T": "K",
        "solute_mass": "kg",
        "dt": "s",
    },
    "outputs": {"radius": "m"},
    "sign_convention": (
        "S>1 (supersaturated) grows the droplet (dR>0); S<1 (subsaturated) "
        "evaporates it (dR<0). d(R^2)/dt has the sign of the net of the "
        "(S-1) supersaturation, Kelvin (negative), and solute (positive) terms."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Shima et al. (2009) QJRMS 135:1307; Rogers & Yau (1989); ERF SuperDropletPCMassChange (dRsqdt)",
    "idealized_test": (
        "Fixed S>1, no curvature/solute, R~15um: d(R^2)/dt = 2(S-1)/(F_k+F_d) "
        "(constant) reproduces the analytic Maxwell growth to machine precision; "
        "S<1 shrinks the droplet; a soluble droplet at S<=1 has a finite "
        "Köhler equilibrium radius (dR/dt = 0 fixed point)."
    ),
}

# Minimum radius [m] used to floor R² so the 1/R and 1/R³ terms and √ stay
# finite for (near-)evaporated droplets — a safety floor, not a tunable
# (well below any aerosol dry radius ~1e-9 m).
_R_FLOOR = 1.0e-9
_R_SQ_FLOOR = _R_FLOOR * _R_FLOOR

# Water molar mass in kg/mol (constants.M_H2O is g/mol; g->kg is a unit factor).
_M_H2O_KG = constants.M_H2O * 1.0e-3


def drsq_dt(
    r_sq: jax.Array,
    S: jax.Array,
    T: jax.Array,
    e_s: jax.Array,
    N_s: jax.Array,
    include_curvature: bool = True,
    include_solute: bool = True,
) -> jax.Array:
    """Right-hand side ``d(R²)/dt`` [m²/s] of the diffusional-growth ODE.

    Elementwise over arrays; ``S, T, e_s`` may be scalars (broadcast over the
    droplet ensemble) or per-droplet arrays. ``include_curvature`` and
    ``include_solute`` are static booleans (feature gating — Python ``if`` on a
    config flag, not a traced ``where``, so disabled terms are not traced).

    Parameters
    ----------
    r_sq : jax.Array
        Radius squared R² [m²].
    S : jax.Array
        Saturation ratio (= RH, ``q_v/q_sat``) [-].
    T : jax.Array
        Temperature [K].
    e_s : jax.Array
        Saturation vapor pressure [Pa].
    N_s : jax.Array
        Effective solute moles per droplet [mol] (van't Hoff ``m_s·i/M_s``).
    include_curvature, include_solute : bool
        Toggle the Kelvin and Raoult terms.
    """
    D = constants.D_vapor
    K = constants.k_air
    Rv = constants.R_v
    rho_l = constants.rho_water
    L = constants.L_v

    r_sq = jnp.maximum(r_sq, _R_SQ_FLOOR)
    R = jnp.sqrt(r_sq)

    # Fukuta-Walter transitional correction to the diffusivity (Knudsen).
    lambda_v = 2.0 * D / jnp.sqrt(8.0 * T * Rv / jnp.pi)
    Kn = lambda_v / R
    d_cf = (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn))

    F_k = (L / (Rv * T) - 1.0) * (L * rho_l) / (K * T)
    F_d = (rho_l * Rv * T) / (d_cf * D * e_s)
    denom = F_k + F_d

    out = 2.0 * (S - 1.0) / denom

    if include_curvature:
        a = 2.0 * constants.sigma_water / (Rv * rho_l)   # Kelvin coeff [m·K]
        out = out - 2.0 * (a / T) / denom / R

    if include_solute:
        b = (3.0 / (4.0 * jnp.pi)) * (_M_H2O_KG / rho_l)  # Raoult coeff [m³/mol]
        out = out + 2.0 * b * N_s / denom / (r_sq * R)    # / R³

    return out


def _rk4_step(r_sq, h, S, T, e_s, N_s, include_curvature, include_solute):
    k1 = drsq_dt(r_sq, S, T, e_s, N_s, include_curvature, include_solute)
    k2 = drsq_dt(r_sq + 0.5 * h * k1, S, T, e_s, N_s, include_curvature, include_solute)
    k3 = drsq_dt(r_sq + 0.5 * h * k2, S, T, e_s, N_s, include_curvature, include_solute)
    k4 = drsq_dt(r_sq + h * k3, S, T, e_s, N_s, include_curvature, include_solute)
    return r_sq + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _euler_step(r_sq, h, S, T, e_s, N_s, include_curvature, include_solute):
    return r_sq + h * drsq_dt(
        r_sq, S, T, e_s, N_s, include_curvature, include_solute
    )


def integrate_radius(
    state: SuperDropletState,
    S: jax.Array,
    T: jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
) -> SuperDropletState:
    """Advance every droplet's radius by diffusional growth over ``dt``.

    The growth ODE is integrated in R² with ``cfg.n_substeps_condensation``
    equal sub-steps using the configured integrator. Only ``active`` droplets
    change; inactive slots keep their radius. Returns a new
    :class:`SuperDropletState` with the updated ``radius``.

    Parameters
    ----------
    state : SuperDropletState
        Droplet ensemble.
    S, T : jax.Array
        Ambient saturation ratio [-] and temperature [K] (scalar or per-droplet).
    dt : float
        Physics step [s].
    cfg : SDMConfig
        Scheme configuration.
    """
    if cfg.condensation_integrator == "rk4":
        step_fn = _rk4_step
    elif cfg.condensation_integrator == "euler":
        step_fn = _euler_step
    else:
        raise ValueError(
            f"Unknown SDM condensation_integrator: {cfg.condensation_integrator!r} "
            "(expected 'rk4' or 'euler')"
        )

    n_sub = int(cfg.n_substeps_condensation)
    if n_sub < 1:
        raise ValueError(
            f"n_substeps_condensation must be >= 1, got {cfg.n_substeps_condensation!r}"
        )

    e_s = saturation_vapor_pressure(jnp.asarray(T, dtype=state.radius.dtype))
    if cfg.include_solute:
        N_s = state.solute_mass * cfg.solute_ionization / cfg.solute_molar_mass
    else:
        N_s = jnp.zeros_like(state.radius)

    h = jnp.asarray(dt, dtype=state.radius.dtype) / n_sub
    include_curvature = cfg.include_curvature
    include_solute = cfg.include_solute

    def body(_, r_sq):
        r_sq = step_fn(r_sq, h, S, T, e_s, N_s, include_curvature, include_solute)
        return jnp.maximum(r_sq, _R_SQ_FLOOR)

    r_sq = lax.fori_loop(0, n_sub, body, state.radius**2)
    radius_new = jnp.sqrt(jnp.maximum(r_sq, _R_SQ_FLOOR))
    # Inactive droplets do not grow.
    radius_new = jnp.where(state.active > 0, radius_new, state.radius)
    return state._replace(radius=radius_new)
