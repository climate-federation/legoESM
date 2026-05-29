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


def eke_mixing_length(L_rossby: jnp.ndarray, cfg: EKEConfig) -> jnp.ndarray:
    """Mixing length L = max(L_rossby, l_min) — the Rossby-radius length from the
    Visbeck machinery, floored so kappa_GM/dissipation stay well-defined."""
    return jnp.maximum(L_rossby, cfg.l_min)


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


__all__ = ["EKEConfig", "eke_mixing_length", "eke_kappa_gm", "eke_local_tendency"]
