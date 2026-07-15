"""Rayleigh friction gravity wave drag.

Simplest GWD parameterization: applies linear drag proportional to wind
speed in the boundary layer and an upper-atmosphere sponge layer.

Follows the same pattern as held_suarez.py for sigma-based drag profiles.

Faithfulness to Held & Suarez (1994)
------------------------------------
FAITHFUL to HS94 — the boundary-layer MOMENTUM tendency:
  * ``k_bl = k_max · max(0, (σ − σ_b)/(1 − σ_b))`` applied to BOTH wind
    components (``du_dt = −k_bl·u``, ``dv_dt = −k_bl·v``) is EXACTLY HS94's
    low-level Rayleigh friction ``k_v = k_f · max(0, (σ − σ_b)/(1 − σ_b))``
    (their momentum-drag eq.); it matches the model's own ``held_suarez.py``
    forcing term. The DEFAULT constants are the HS94 values: ``k_max = 1/day =
    k_f`` and ``σ_b = 0.7`` (== ``held_suarez.K_F`` / ``held_suarez.SIGMA_B``).
ADDITIONS / DEPARTURES from HS94:
  * **the frictional HEATING is NOT HS94**: this scheme returns the KE removed
    by the FULL ``k_drag`` (BL drag + sponge) as heat ``dT_dt = −(u·du_dt +
    v·dv_dt)/c_pd``. HS94 has NO KE-compensating heating term — its temperature
    forcing is a SEPARATE Newtonian relaxation ``−k_T·(T − T_eq)`` while the
    velocity drag does not heat. So the FAITHFUL claim is scoped to the MOMENTUM
    tendency; the heating (which makes the scheme energy-conserving) is an
    addition;
  * the upper **sin² SPONGE is NOT HS94** (HS94 has no sponge layer): a purely
    numerical sponge-layer Rayleigh damping ``k_sponge = sponge_k · sin²(½π ·
    clip((sponge_top − σ)/sponge_top, 0, 1))`` that ramps from 0 at
    ``σ = sponge_top`` to ``sponge_k`` at the model top (``σ → 0``), to absorb
    upward-propagating gravity waves and prevent spurious reflection. Standard
    dynamical-core device, not a HS94 physics term;
  * the sponge **HEATS the removed KE** (``dT_dt`` is computed from the FULL
    ``k_drag = k_bl + k_sponge``), so the scheme is energy-CONSERVING — UNLIKE a
    physical radiation sponge, which removes gravity-wave energy from the domain
    (radiated to space) rather than converting it to local heat. A design choice
    that keeps the column energy budget closed; not how a physical sponge acts;
  * ``conserves = ["energy"]`` is CORRECT here (unlike the launched-wave schemes
    ``hines.py``/``lindzen.py`` = ["none"]): there is NO external wave source —
    the drag + sponge convert RESOLVED mean-flow KE directly to heat, both
    tracked, so the resolved-KE→heat closure ``c_pd·Σρ·dT·dz == eps_gwd`` IS the
    full energy budget (it remains definitional, but there is no unbudgeted
    external source to make it incomplete). Momentum is NOT conserved (a sink).
NUMERICS: the ``k_bl`` UPPER clip at 1 is a redundant guard (``(σ−σ_b)/(1−σ_b)
≤ 1`` for any physical ``σ ≤ 1``; bites only if ``σ > 1``, i.e. ``p_full >
p_sfc``); the ``sponge_arg`` clip to [0, 1]; the ``p_sfc`` floor at 1 Pa.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_rayleigh_gwd_faithful.py``.

References
----------
- Held, I. M., & Suarez, M. J. (1994). A proposal for the intercomparison of
  the dynamical cores of atmospheric general circulation models. Bull. Amer.
  Meteor. Soc., 75, 1825-1830 (the σ_b/k_f low-level Rayleigh friction).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.config import RayleighConfig
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py and
# docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md). The architect pins
# units/signs/conservation/reference; the body must honour it.
__physics_contract__ = {
    "summary": (
        "Rayleigh-friction gravity-wave drag: a linear momentum sink in the "
        "boundary layer plus a sin^2 sponge near the model top."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "rho": "kg/m^3", "lat": "rad", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "eps_gwd": "W/m^2",
    },
    "sign_convention": (
        "Drag opposes the wind: du_dt = -k(sigma)*u (and v), so du_dt has the "
        "opposite sign to u; eps_gwd (frictional dissipation -> heating) is "
        "positive-definite."
    ),
    # Momentum is NOT conserved (a drag/sponge removes momentum to the surface /
    # absorbs it at the top), but total ENERGY is: the kinetic energy lost by the
    # mean flow is returned as frictional heating, dT_dt = -(u*du_dt + v*dv_dt)/c_pd
    # (eps_gwd = column-integrated KE loss). Unlike the launched-wave GWD schemes
    # (hines.py/lindzen.py = ["none"]) there is NO external wave source here — the
    # drag + sponge convert RESOLVED mean-flow KE directly to heat (both tracked),
    # so the resolved-KE->heat closure IS the full energy budget. (The sponge
    # heating the removed KE is a design choice; a physical radiation sponge would
    # remove the GW energy from the domain instead.) So conserved quantity = energy.
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Held & Suarez (1994) low-level sigma-drag Rayleigh friction "
        "(k_f, sigma_b); the upper sin^2 sponge is a numerical addition, not HS94."
    ),
    "idealized_test": (
        "rest state (u=v=0) -> zero tendency; for u>0 the boundary-layer drag is "
        "non-positive and its magnitude ramps from 0 at sigma=sigma_b to k_max at "
        "the surface."
    ),
}


def rayleigh_gwd(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    rho: jax.Array,
    lat: jax.Array,
    dt: float,
    config: RayleighConfig,
) -> GWDOutput:
    """Compute Rayleigh friction GWD tendencies.

    Parameters
    ----------
    u, v : jax.Array
        Wind components [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full, z_half : jax.Array
        Heights [m], shapes (ncol, nlev) and (ncol, nlev+1).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    lat : jax.Array
        Latitude [rad], shape (ncol,).
    dt : float
        Time step [s].
    config : RayleighConfig

    Returns
    -------
    GWDOutput
    """
    ncol, nlev = u.shape

    # Sigma coordinate
    p_sfc = p_half[:, -1:]  # (ncol, 1)
    sigma = p_full / jnp.clip(p_sfc, 1.0, None)

    # Boundary layer drag: ramps from 0 at sigma_b to k_max at surface
    k_bl = config.k_max * jnp.clip(
        (sigma - config.sigma_b) / (1.0 - config.sigma_b), 0.0, 1.0
    )

    # Upper sponge: sin^2 taper near model top
    sponge_arg = jnp.clip(
        (config.sponge_top - sigma) / config.sponge_top, 0.0, 1.0
    )
    k_sponge = config.sponge_k * jnp.sin(0.5 * jnp.pi * sponge_arg) ** 2

    # Combined drag coefficient
    k_drag = k_bl + k_sponge

    # Tendencies
    du_dt = -k_drag * u
    dv_dt = -k_drag * v

    # Frictional heating: dT/dt = -(u*du/dt + v*dv/dt) / c_pd
    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd

    # Column dissipation (positive-definite: KE lost by the mean flow)
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)

    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)
