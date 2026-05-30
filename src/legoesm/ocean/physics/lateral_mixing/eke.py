"""Prognostic eddy-kinetic-energy (EKE) closure — Eden & Greatbatch (2008).

A 2-D (depth-integrated) eddy-energy field ``E`` whose budget is

    dE/dt + advection(E) = iso-diffusion(E) + P - eps

with a Visbeck-style **prognostic** GM coefficient that replaces a constant one:

    kappa_GM = c_k * L * sqrt(E)                                   (>= 0)
    P        = kappa_GM * sigma^2     (GM mean-APE -> EKE conversion; sigma = N|S|)
    eps      = c_eps * E^{3/2} / L    (Eden-Greatbatch dissipation, >= 0)

``E`` is 2-D to match the 2-D ``kappa_GM`` the GM/Redi tendency already accepts
(the Visbeck path). ``sigma`` (depth-averaged Eady growth rate ``<N|S|>_z``) and the
mixing length ``L`` (first-baroclinic Rossby radius, floored at ``l_min``) come from
the SHARED GM/Redi Visbeck machinery (``compute_visbeck_kappa_gm`` internals) — this
module does NOT recompute N^2/slopes/L (no duplicate numerics). The closure functions
here are pure and take ``E``, ``sigma``, ``L`` as inputs; the advection + isopycnal
diffusion of ``E`` reuse the tracer-transport machinery; the state-field threading +
GM/Redi coupling are wired separately (build-spec gates E2, E6).

The mixing length ``L`` is selectable (``EKEConfig.mixing_length_scheme``):
``"rossby"`` (default) uses the Visbeck first-baroclinic length ``max(L_rossby,
l_min)`` — legoESM's pre-``eke_len`` behaviour; ``"rhines"`` reproduces Veros's
Rhines-limited ``eke_len = max(l_min, min(eke_cross·L_rossby, eke_crhin·L_rhines))``
from the deformation radius (``eke_deformation_radius``) and the eddy-energy Rhines
scale (``eke_rhines_length``). See ``docs/ocean_fidelity/eke_len_build_spec.md``.

Defaults match Veros ACC (``eke_c_k=0.4``, ``eke_c_eps=0.5``, ``eke_lmin=100``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class EKEConfig(NamedTuple):
    """Prognostic EKE closure parameters (Eden & Greatbatch 2008)."""

    c_k: float = 0.4          # kappa_GM = c_k * L * sqrt(E)   (Veros eke_c_k)
    c_eps: float = 0.5        # dissipation eps = c_eps * E^{3/2}/L (Veros eke_c_eps)
    l_min: float = 100.0      # mixing-length floor [m]        (Veros eke_lmin)
    k_iso: float = 1000.0     # isopycnal diffusivity for E [m^2/s]
    advection_scheme: str = "superbee"  # E advection (reuses the tracer dispatch)
    e_min: float = 1.0e-8     # positivity floor on E [m^2/s^2]
    kappa_gm_max: float = 1.0e4  # safety ceiling on the prognostic kappa_GM [m^2/s]
    # Mixing-length scheme for kappa_GM = c_k·L·√E:
    #   "rossby" (default) — L = max(L_rossby, l_min), the Visbeck first-baroclinic
    #     length (N̄·H/|f|). legoESM's pre-eke_len behaviour (eke_cross/eke_crhin unused).
    #   "rhines" — Veros eke_len = max(l_min, min(eke_cross·L_rossby, eke_crhin·L_rhines))
    #     from the deformation radius c₁/|f| (c₁=∫N dz/π, equatorial-limited) and the
    #     eddy-energy Rhines scale √(√E/β).
    mixing_length_scheme: str = "rossby"
    eke_cross: float = 1.0    # deformation-radius weight in eke_len (Veros eke_cross)
    eke_crhin: float = 1.0    # Rhines-scale weight in eke_len      (Veros eke_crhin)


def eke_mixing_length(L_rossby: jnp.ndarray, cfg: EKEConfig) -> jnp.ndarray:
    """Mixing length L = max(L_rossby, l_min) — the Rossby-radius length from the
    Visbeck machinery, floored so kappa_GM/dissipation stay well-defined. This is
    the ``mixing_length_scheme="rossby"`` length (legoESM's pre-eke_len default)."""
    return jnp.maximum(L_rossby, cfg.l_min)


# Denominator safety floor for the |f| and β reciprocals in the Rhines-limited
# mixing length — a pure numerical floor (matches Veros eke_len's ``max(·, 1e-16)``),
# exempt from the named-constant rule like the other eps floors in this module.
_DENOM_FLOOR = 1.0e-16


def eke_rhines_length(
    E: jnp.ndarray, beta: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """Eddy-energy **Rhines scale** ``L_rhines = sqrt(sqrt(E) / beta)`` [m] (Veros
    ``L_rhines``, ``veros/core/eke.py:62``).

    ``sqrt(E)`` is the eddy velocity scale [m/s] (``E`` is the specific eddy energy
    [m^2/s^2]); ``beta = df/dy`` [1/(m·s)]; ``sqrt(E)/beta`` has units m^2, so the
    result is a length. ``E`` is floored at 0 and regularised by ``+1e-30`` before
    the sqrt (finite gradient at ``E=0``); ``beta`` is floored at ``_DENOM_FLOOR``
    (matching Veros, which assumes ``beta > 0`` — true on the sphere where
    ``f = 2Ω sinφ`` ⇒ ``β = 2Ω cosφ/R ≥ 0``).
    """
    sqrt_E = jnp.sqrt(jnp.maximum(E, 0.0) + 1.0e-30)
    beta_safe = jnp.maximum(beta, _DENOM_FLOOR)
    return jnp.sqrt(sqrt_E / beta_safe)


def eke_deformation_radius(
    int_N_dz: jnp.ndarray, f_coriolis: jnp.ndarray, beta: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """First-baroclinic **Rossby deformation radius**, equatorially limited (Veros
    ``L_rossby``, ``veros/core/eke.py:54``):

        c1       = int_N_dz / pi                      # 1st-baroclinic phase speed [m/s]
        L_rossby = min( c1/|f|,  sqrt(c1/(2·beta)) )  # [m]

    ``int_N_dz = ∫N dz`` [m/s] is the column buoyancy-frequency integral (Veros's
    ``Σ √(max(0,N²))·dzw·maskW``); the ``1/pi`` is the WKB first-baroclinic factor.
    The midlatitude branch ``c1/|f|`` is the deformation radius; the equatorial
    branch ``sqrt(c1/2β)`` caps it as ``|f|→0``. ``|f|`` and ``β`` denominators are
    floored at ``_DENOM_FLOOR``; the equatorial sqrt is regularised by ``+1e-30`` for
    a finite gradient at ``c1=0`` (where the midlatitude branch is exactly 0 and wins
    the min, so the forward value is unaffected).
    """
    c1 = jnp.maximum(int_N_dz, 0.0) / jnp.pi
    f_safe = jnp.maximum(jnp.abs(f_coriolis), _DENOM_FLOOR)
    beta_safe = jnp.maximum(beta, _DENOM_FLOOR)
    L_mid = c1 / f_safe
    L_eq = jnp.sqrt(c1 / (2.0 * beta_safe) + 1.0e-30)
    return jnp.minimum(L_mid, L_eq)


def eke_len_composite(
    L_def: jnp.ndarray, L_rhines: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """Veros ``eke_len`` composite mixing length [m] (``veros/core/eke.py:63``):

        eke_len = max( l_min, min(eke_cross·L_def, eke_crhin·L_rhines) )

    The ``min`` lets the eddy-energy Rhines scale ``L_rhines`` limit the (larger)
    deformation radius ``L_def`` where eddies are weak/small; ``l_min`` floors the
    result. All inputs are lengths [m] ≥ 0.
    """
    inner = jnp.minimum(cfg.eke_cross * L_def, cfg.eke_crhin * L_rhines)
    return jnp.maximum(cfg.l_min, inner)


def eke_kappa_gm(E: jnp.ndarray, L: jnp.ndarray, cfg: EKEConfig) -> jnp.ndarray:
    """Prognostic GM coefficient ``kappa_GM = c_k * L * sqrt(E)`` (clamped >= 0 and
    <= kappa_gm_max). E is floored at 0 before the sqrt (sqrt of a tiny positive is
    used at E=0 so the gradient stays finite)."""
    E_pos = jnp.maximum(E, 0.0)
    kappa = cfg.c_k * L * jnp.sqrt(E_pos + 1.0e-30)
    return jnp.clip(kappa, 0.0, cfg.kappa_gm_max)


def eke_local_tendency(
    E: jnp.ndarray, sigma: jnp.ndarray, L: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """Local EKE source minus sink: ``P - eps`` [m^2/s^3].

    ``P = kappa_GM * sigma^2`` is the rate the GM flux converts mean available
    potential energy into eddy energy (``sigma = N|S|`` is the Eady growth rate, so
    ``kappa_GM * sigma^2 = kappa_GM * M^4/N^2``). ``eps = c_eps * E^{3/2}/L`` is the
    Eden-Greatbatch dissipation. Advection + isopycnal diffusion of E are applied
    separately (tracer machinery), so this returns ONLY the local source/sink.

    Parameters
    ----------
    E : array — eddy kinetic energy [m^2/s^2], 2-D (n_lat, n_lon) or any shape.
    sigma : array — depth-averaged Eady growth rate <N|S|>_z [1/s], same shape.
    L : array — mixing length [m] (already floored, see ``eke_mixing_length``).
    cfg : EKEConfig.
    """
    E_pos = jnp.maximum(E, 0.0)
    kappa = eke_kappa_gm(E_pos, L, cfg)
    production = kappa * sigma ** 2
    dissipation = cfg.c_eps * E_pos ** 1.5 / jnp.maximum(L, cfg.l_min)
    return production - dissipation


def eke_apply_local_source(
    E: jnp.ndarray, sigma: jnp.ndarray, L: jnp.ndarray, cfg: EKEConfig, dt: float,
) -> jnp.ndarray:
    """One step of the local EKE source/sink with **semi-implicit dissipation** —
    unconditionally positivity-preserving (``E_{n+1} >= 0``) with NO clipping/mask.

    Production is explicit (``P = kappa_GM(E_n)·sigma^2 >= 0``); dissipation is
    linearised implicitly (``eps = c_eps·√E_n·E_{n+1}/L``), giving

        E_{n+1} = (E_n + dt·P) / (1 + dt·c_eps·√E_n / L)

    whose numerator is >= 0 (E_n >= 0, P >= 0) and denominator >= 1, so the result
    is >= 0 by construction (not by a floor). Advection + isopycnal diffusion of E
    are applied separately by the step (also positivity-preserving). This is the
    standard stable treatment of the quadratic-in-magnitude EKE dissipation
    (Eden-Greatbatch / Veros).
    """
    E_pos = jnp.maximum(E, 0.0)
    production = eke_kappa_gm(E_pos, L, cfg) * sigma ** 2
    diss_rate = cfg.c_eps * jnp.sqrt(E_pos + 1.0e-30) / jnp.maximum(L, cfg.l_min)
    return (E_pos + dt * production) / (1.0 + dt * diss_rate)


def validate_eke_config(cfg: EKEConfig) -> None:
    """Fail-fast validation of EKE parameters (dispatch discipline). Raises
    ``ValueError`` on non-physical values."""
    if cfg.c_k <= 0.0:
        raise ValueError(f"EKEConfig.c_k must be > 0, got {cfg.c_k!r}")
    if cfg.c_eps <= 0.0:
        raise ValueError(f"EKEConfig.c_eps must be > 0, got {cfg.c_eps!r}")
    if cfg.l_min <= 0.0:
        raise ValueError(f"EKEConfig.l_min must be > 0, got {cfg.l_min!r}")
    if cfg.k_iso < 0.0:
        raise ValueError(f"EKEConfig.k_iso must be >= 0, got {cfg.k_iso!r}")
    if cfg.e_min < 0.0:
        raise ValueError(f"EKEConfig.e_min must be >= 0, got {cfg.e_min!r}")
    if cfg.kappa_gm_max <= 0.0:
        raise ValueError(
            f"EKEConfig.kappa_gm_max must be > 0, got {cfg.kappa_gm_max!r}"
        )
    if cfg.mixing_length_scheme not in ("rossby", "rhines"):
        raise ValueError(
            "EKEConfig.mixing_length_scheme must be 'rossby' or 'rhines', got "
            f"{cfg.mixing_length_scheme!r}"
        )
    if cfg.eke_cross <= 0.0:
        raise ValueError(f"EKEConfig.eke_cross must be > 0, got {cfg.eke_cross!r}")
    if cfg.eke_crhin <= 0.0:
        raise ValueError(f"EKEConfig.eke_crhin must be > 0, got {cfg.eke_crhin!r}")


__all__ = [
    "EKEConfig",
    "eke_mixing_length",
    "eke_rhines_length",
    "eke_deformation_radius",
    "eke_len_composite",
    "eke_kappa_gm",
    "eke_local_tendency",
    "eke_apply_local_source",
    "validate_eke_config",
]
