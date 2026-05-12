"""Pacanowski & Philander (1981) Richardson-number dependent mixing.

POP / E3SM Omega convention:
    ν (momentum) = ν₀ / (1 + α·Ri)^n + ν_b
    κ (tracer)   = ν / (1 + α·Ri) + κ_b

The tracer division gives an EFFECTIVE Prandtl-Ri scaling Pr = ν/κ
that grows with Ri.  ``cfg.Pr_t`` is retained for API compatibility
but is now a no-op — non-default values trigger a UserWarning at
function-call time so calibration sweeps notice the field is dead.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import compute_buoyancy_frequency
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing.config import RichardsonVerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def richardson_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: RichardsonVerticalMixingConfig,
    apply_diffusion: bool = True,
) -> VerticalMixingOutput:
    """Apply Richardson-number dependent vertical mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
        In-situ density.
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : RichardsonVerticalMixingConfig

    Returns
    -------
    VerticalMixingOutput
    """
    eps = _EPS

    # Pr_t is now a no-op (the canonical Pr-Ri scaling is built into
    # the K_v formula).  Warn calibration users that adjusting it has
    # no effect — silent dead-config is a known calibration trap.
    if cfg.Pr_t != 10.0:
        warnings.warn(
            "RichardsonVerticalMixingConfig.Pr_t is no longer used "
            "after the iter-47 PP81 fix; the canonical POP/E3SM "
            "scaling κ = ν/(1+αRi) + κ_b makes Pr a function of Ri "
            "automatically.  Setting Pr_t to a non-default value has "
            "no effect on the diffusivities.",
            UserWarning,
            stacklevel=2,
        )

    # N^2 at interfaces
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)

    # Shear^2 at interfaces
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u[..., :-1] - u[..., 1:]
    dv = v[..., :-1] - v[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)

    # Richardson number
    Ri = N2 / jnp.maximum(S2, eps)
    Ri = jnp.maximum(Ri, 0.0)  # Clip negative Ri (unstable → max mixing)

    # Pacanowski & Philander (1981, JPO 11, p.1448, Eq. 1) — the
    # POP / E3SM Omega convention is:
    #   ν (momentum) = ν₀ / (1 + α·Ri)^n + ν_b      [n = 2]
    #   κ (tracer)   = ν / (1 + α·Ri) + κ_b
    # The full ν (including ν_b background) enters the tracer
    # division; the tracer division produces an EFFECTIVE
    # Prandtl-Ri scaling Pr = ν/κ that grows with Ri, which is
    # physically essential because in stable shear momentum mixes
    # more efficiently than tracer.
    #
    # The previous formulation computed K_v with (1+αRi)^n decay
    # (momentum's form) and assigned momentum A_v = K_v · const_Pr_t —
    # which (a) used the momentum decay rate for the tracer field, and
    # (b) discarded the canonical Prandtl-Ri dependence (constant 10×
    # Prandtl instead of Ri-growing).
    one_plus_aRi = 1.0 + cfg.alpha * Ri
    A_v = cfg.K_0 / one_plus_aRi ** cfg.n + cfg.A_bg              # momentum
    K_v = A_v / one_plus_aRi + cfg.K_bg                            # tracer

    # Apply variable-K vertical diffusion.  When ``apply_diffusion`` is
    # False, return zero tendencies; the caller will apply K_v/A_v via an
    # unconditionally-stable backward-Euler solver after the explicit step.
    if apply_diffusion:
        vel = jnp.stack([u, v], axis=0)
        vel_tend = jax.vmap(
            lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, A_v),
            in_axes=0, out_axes=0,
        )(vel)

        tracers = jnp.stack([T, S], axis=0)
        tr_tend = jax.vmap(
            lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K_v),
            in_axes=0, out_axes=0,
        )(tracers)
    else:
        zero_uv = jnp.zeros_like(u)
        vel_tend = jnp.stack([zero_uv, zero_uv], axis=0)
        zero_T = jnp.zeros_like(T)
        tr_tend = jnp.stack([zero_T, zero_T], axis=0)

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        K_v=K_v,
        A_v=A_v,
    )
