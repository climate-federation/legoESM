"""swirl-lm's Monin-Obukhov surface layer, ported for CloudBench column runs.

The CloudBench LES (swirl-lm 98f96f80, ``boundary_condition/
monin_obukhov_similarity_theory.py`` with ``numerics/diffusion.py`` and
``equations/utils.py``) closes its lower boundary with this law; a column driven
like those runs uses the same law so every closure sees the LES's surface physics.

At the evaluation height ``z`` (the LES used its first node, ``dz/2``):

* bulk Richardson number ``Rb = g z (θ − θ_s) / ((u² + v²) θ)`` (no gustiness;
  0 when the wind is exactly zero);
* ``ζ = z/L`` from ten Newton steps (finite-difference slope, as swirl-lm) on
  ``Rb = ζ (ln(z/z0) − ψ_h) / (ln(z/z0) − ψ_m)²`` from ``ζ = 0``;
* ψ by the sign of ``b = θ − θ_s``: unstable Businger-Dyer (``γ``, the
  ``α = 1`` swirl-lm hard-codes), stable ``−β ζ``, neutral 0;
* scalar flux (+up) ``c_h |U_g| (φ_s − φ_1)``, ``c_h = κ²/((ln − ψ_h)(ln − ψ_m))``,
  ``|U_g| = √(g_u² + u² + v²)`` (``z_t`` is not used by swirl-lm);
* momentum flux ``u'w' = −κ²/(ln − ψ_m)² u |U_g|``.

The stability ``θ`` differs by field, as in swirl-lm: the potential temperature for
the scalar fluxes, the liquid-water potential temperature for the stress.

KNOWN DIFFERENCES / switches (repo RULE 5):

* The stable/unstable/neutral choice depends on the traced ``b``. The default
  blends the two branches with a logistic weight of width ``stability_width_K``
  (inside the Newton solve and in the fluxes alike); ``stability_width_K = 0``
  is swirl-lm's exact three-way switch. The per-case gap is pinned in the tests.
* On the smooth path the blend evaluates the stable branch at the unstable
  iterates too, where ``−βζ`` grows without bound (ζ ≪ 0 in calm air) and gives
  the Newton solve poles. There the stable branch reads ``−β ζ₊`` with the smooth
  positive part ``ζ₊ = ζ σ(ζ/δ)`` (``δ = stable_continuation_width``): ``ζ₊ → ζ``
  for ζ ≫ δ, so the clearly stable law is exact, and ``ζ₊ ≥ −0.28 δ`` for ζ < 0. Measured 2026-11-03 (z 70 m, z0 1e-3, wind
  0.05-5 m/s, b −0.05..0.05 K): with it the solve converges wherever the hard
  switch does; without it, it did not at wind ≤ 0.3 m/s. The hard path is untouched.
* INHERITED LIMITATION of swirl-lm: in strongly stable calm air (Rb above the
  linear law's critical ``β_h/β_m²``) the Richardson equation has no root and the
  ten Newton steps return an unconverged ζ, in the reference and in this port.
* The unstable branch clips ``1 − γζ`` at 0 as swirl-lm does; on the smooth
  path the clip floor is ``_UNSTABLE_BASE_FLOOR`` so its slope stays finite. It
  only binds at ``ζ ≥ 1/γ``, outside the unstable regime.
* ``density_free_drag``: swirl-lm divides its drag coefficient by density and
  writes the result into a dynamic-stress slot, so the LES's surface stress is
  the kinematic value in Pa, i.e. ``1/ρ`` of the dimensionally consistent one.
  ``True`` reproduces that (kinematic ``u'w'`` divided by ``rho``); ``False`` is
  the consistent law.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "swirl-lm (CloudBench LES) Monin-Obukhov surface layer: bulk-Richardson "
        "Newton solve for z/L, Businger-Dyer / linear stability functions, "
        "kinematic surface fluxes of theta_l, q_t and momentum."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "theta": "K", "theta_l": "K", "q_t": "kg/kg",
        "rho": "kg/m^3", "z": "m", "theta_l_s": "K", "q_t_s": "kg/kg",
    },
    "outputs": {
        "w_theta_l": "K m/s", "w_q_t": "m/s", "uw": "m^2/s^2", "vw": "m^2/s^2",
    },
    "sign_convention": (
        "Fluxes are POSITIVE UPWARD (out of the surface): w_theta_l > 0 when the "
        "surface theta_l exceeds the air's. The momentum flux opposes the wind "
        "(uw = -C u |U_g|), so rho*uw is the stress ON THE ATMOSPHERE. A surface "
        "flux is a source for the column; nothing is conserved here."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "swirl-lm 98f96f80 boundary_condition/monin_obukhov_similarity_theory.py; "
        "Stoll & Porte-Agel (2009) J. Atmos. Sci. 66, 412-431"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_swirl_most.py: hard switch matches a "
        "NumPy transcription of swirl-lm to 1e-12; smooth default gap pinned per "
        "case; check_grads through the Newton solve; zero fluxes at equilibrium."
    ),
}

__param_spec__ = {
    "SwirlMOSTConfig": {
        "scheme_key": "atm.turb.SwirlMOSTConfig",
        "excluded": {
            "kappa": "physical constant (von Karman), the value swirl-lm uses",
            "g_m_s2": "physical constant, set to the reference model's value",
            "stability_width_K": "numerics: smoothing width of the stability switch",
            "stable_continuation_width": "numerics: smoothing width of the stable branch's continuation to zeta < 0",
        },
        "params": {
            "z0_m": {"units": "m", "bounds": (1e-04, 1e-02), "tunable_tier": 2, "transform": "sigmoid", "category": "surface_exchange", "reference": "swirl-lm MOST momentum roughness z_0 (used for scalars too)", "shape": None},
            "gamma_m": {"units": "1", "bounds": (8.0, 28.0), "tunable_tier": 2, "transform": "sigmoid", "category": "monin_obukhov", "reference": "Businger-Dyer unstable momentum coefficient (swirl-lm proto default 15)", "shape": None},
            "gamma_h": {"units": "1", "bounds": (8.0, 28.0), "tunable_tier": 2, "transform": "sigmoid", "category": "monin_obukhov", "reference": "Businger-Dyer unstable heat coefficient (swirl-lm proto default 15)", "shape": None},
            "beta_m": {"units": "1", "bounds": (2.0, 10.0), "tunable_tier": 2, "transform": "sigmoid", "category": "monin_obukhov", "reference": "Stoll & Porte-Agel (2009) stable momentum coefficient (CloudBench 4.8)", "shape": None},
            "beta_h": {"units": "1", "bounds": (2.0, 12.0), "tunable_tier": 2, "transform": "sigmoid", "category": "monin_obukhov", "reference": "Stoll & Porte-Agel (2009) stable heat coefficient (CloudBench 7.8)", "shape": None},
            "gustiness_m_s": {"units": "m/s", "bounds": (0.0, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "surface_exchange", "reference": "swirl-lm surface_gustiness added in quadrature to the flux wind speed", "shape": None},
        },
    },
}

#: swirl-lm root_finder.newton_method: fixed iteration count, the slope perturbation
#: used where the iterate is exactly zero, and the resolution multiplier of the
#: relative perturbation.
_N_NEWTON = 10
_NEWTON_ZERO_STEP = 1.0e-4
_NEWTON_EPS_RESOLUTION_FACTOR = 10.0
#: Floor of ``1 − γζ`` on the smooth path (see module docstring).
_UNSTABLE_BASE_FLOOR = 1.0e-30


class SwirlMOSTConfig(NamedTuple):
    """swirl-lm MOST settings (CloudBench run values are set by its builder)."""

    z0_m: float = 1.0e-3
    gamma_m: float = 15.0
    gamma_h: float = 15.0
    beta_m: float = 4.8
    beta_h: float = 7.8
    gustiness_m_s: float = 1.0
    kappa: float = constants.kappa_von_karman
    g_m_s2: float = constants.g
    stability_width_K: float = 0.01
    stable_continuation_width: float = 1.0e-3
    density_free_drag: bool = False


class SwirlMOSTFluxes(NamedTuple):
    """Kinematic surface fluxes, positive upward."""

    w_theta_l: jax.Array
    w_q_t: jax.Array
    uw: jax.Array
    vw: jax.Array


def validate_swirl_most(cfg: SwirlMOSTConfig) -> None:
    if not cfg.stability_width_K >= 0.0:
        raise ValueError(f"stability_width_K must be >= 0, got {cfg.stability_width_K}")
    if cfg.stability_width_K > 0.0 and not cfg.stable_continuation_width > 0.0:
        raise ValueError("stable_continuation_width must be > 0 on the smooth path, got "
                         f"{cfg.stable_continuation_width}")
    if not cfg.z0_m > 0.0:
        raise ValueError(f"z0_m must be > 0, got {cfg.z0_m}")


def _divide_no_nan(num, den):
    safe = jnp.where(den == 0.0, 1.0, den)
    return jnp.where(den == 0.0, 0.0, num / safe)


def stability_functions(zeta, b, cfg: SwirlMOSTConfig):
    """``(ψ_m, ψ_h)`` at ``ζ`` for buoyancy ``b = θ − θ_s`` (see module docstring)."""
    hard = cfg.stability_width_K == 0.0
    floor = 0.0 if hard else _UNSTABLE_BASE_FLOOR
    x = jnp.maximum(1.0 - cfg.gamma_m * zeta, floor) ** 0.25
    psi_m_u = (2.0 * jnp.log(0.5 * (1.0 + x)) + jnp.log(0.5 * (1.0 + x ** 2))
               - 2.0 * jnp.arctan(x) + 0.5 * jnp.pi)
    y = jnp.maximum(1.0 - cfg.gamma_h * zeta, floor) ** 0.5
    psi_h_u = 2.0 * jnp.log(0.5 * (1.0 + y))
    psi_m_s = -cfg.beta_m * zeta
    psi_h_s = -cfg.beta_h * zeta
    if hard:
        zero = jnp.zeros_like(zeta)
        psi_m = jnp.where(b < 0.0, psi_m_u, jnp.where(b > 0.0, psi_m_s, zero))
        psi_h = jnp.where(b < 0.0, psi_h_u, jnp.where(b > 0.0, psi_h_s, zero))
        return psi_m, psi_h
    delta = cfg.stable_continuation_width
    zeta_pos = zeta * jax.nn.sigmoid(zeta / delta)
    psi_m_s, psi_h_s = -cfg.beta_m * zeta_pos, -cfg.beta_h * zeta_pos
    w_u = jax.nn.sigmoid(-b / cfg.stability_width_K)
    return w_u * psi_m_u + (1.0 - w_u) * psi_m_s, w_u * psi_h_u + (1.0 - w_u) * psi_h_s


def bulk_richardson(theta, theta_s, u, v, z, cfg: SwirlMOSTConfig):
    return cfg.g_m_s2 * z * _divide_no_nan(theta - theta_s, (u ** 2 + v ** 2) * theta)


def normalized_height(theta, theta_s, u, v, z, cfg: SwirlMOSTConfig):
    """``ζ = z/L`` by swirl-lm's fixed ten-step finite-difference Newton solve."""
    ln_z = jnp.log(z / cfg.z0_m)
    r_b = bulk_richardson(theta, theta_s, u, v, z, cfg)
    b = theta - theta_s

    def err(zeta):
        psi_m, psi_h = stability_functions(zeta, b, cfg)
        return r_b - zeta * (ln_z - psi_h) / (ln_z - psi_m) ** 2

    dtype = jnp.result_type(r_b)
    res = np.finfo(dtype).resolution
    eps = float(np.power(2.0, np.ceil(np.log(_NEWTON_EPS_RESOLUTION_FACTOR * res)
                                      / np.log(2.0))))
    zeta = jnp.zeros_like(r_b)
    for _ in range(_N_NEWTON):
        dx = eps * jnp.abs(zeta)
        dx = jnp.where(dx == 0.0, _NEWTON_ZERO_STEP, dx)
        slope = (err(zeta + 0.5 * dx) - err(zeta - 0.5 * dx)) / dx
        zeta = zeta - _divide_no_nan(err(zeta), slope)
    return zeta


def swirl_most_fluxes(u, v, theta, theta_l, q_t, rho, z, theta_l_s, q_t_s,
                      cfg: SwirlMOSTConfig) -> SwirlMOSTFluxes:
    """Kinematic surface fluxes from the state at height ``z`` (see module docstring).

    ``theta``: potential temperature (scalar-flux stability); ``theta_l``: liquid-water
    potential temperature (stress stability and the heat-flux scalar); ``q_t``: the
    total water the surface exchanges with; ``rho``: air density at ``z`` (read only
    by ``density_free_drag``). ``cfg`` is validated by the caller
    (:func:`validate_swirl_most`), so its float fields may be traced."""
    ln_z = jnp.log(z / cfg.z0_m)
    u_g = jnp.sqrt(cfg.gustiness_m_s ** 2 + u ** 2 + v ** 2)
    zeta_s = normalized_height(theta, theta_l_s, u, v, z, cfg)
    psi_m_s, psi_h_s = stability_functions(zeta_s, theta - theta_l_s, cfg)
    c_h = _divide_no_nan(cfg.kappa ** 2, (ln_z - psi_h_s) * (ln_z - psi_m_s))
    zeta_m = normalized_height(theta_l, theta_l_s, u, v, z, cfg)
    psi_m_m, _ = stability_functions(zeta_m, theta_l - theta_l_s, cfg)
    c_m = cfg.kappa ** 2 / (ln_z - psi_m_m) ** 2
    if cfg.density_free_drag:
        c_m = c_m / rho
    return SwirlMOSTFluxes(
        w_theta_l=c_h * u_g * (theta_l_s - theta_l),
        w_q_t=c_h * u_g * (q_t_s - q_t),
        uw=-c_m * u * u_g,
        vw=-c_m * v * u_g)
