"""Gravitational sedimentation and surface rain accumulation for SDM.

Faithful JAX port of the gravity part of ERF ``SuperDropletPC::AdvectParticles``
(``z -= v_terminal·dt``; no flow advection in a quiescent column) plus
``SuperDropletsMoist::rainAccumulation`` (the downward represented-mass flux
through the surface accumulates as precipitation,
``rain_accum += max(0, -flux)·dt/ρ_w`` in metres, ×1000 for mm).

Super-droplets carry a height ``z`` [m] *alongside* the
:class:`~...particles.SuperDropletState` (position is a driver concern, not a
core particle attribute). Each step every active droplet falls at its
configured terminal velocity; a droplet whose new height crosses the surface
(``z <= 0``) deposits its entire represented water mass ``ξ·m`` into the
surface precipitation bucket and is deactivated (ERF deactivates/recycles
particles leaving the domain).

**Invariant:** airborne represented water + accumulated precipitation is
conserved exactly (sedimentation neither grows nor evaporates droplets).
Pure function of its inputs; ``jit``/``scan``-friendly. Not differentiable
through the surface-crossing indicator (a step function — same status as the
coalescence Monte-Carlo; the user waived differentiability for these paths).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.kernels import terminal_velocity
from legoesm.atmosphere.physics.microphysics.sdm.particles import (
    SuperDropletState,
    water_mass_per_droplet,
)

__physics_contract__ = {
    "summary": (
        "SDM gravitational sedimentation: droplets fall at their terminal "
        "velocity; represented mass crossing the surface accumulates as "
        "precipitation and the droplet deactivates."
    ),
    "inputs": {
        "multiplicity": "1",
        "radius": "m",
        "z": "m",
        "rho": "kg/m^3",
        "p": "Pa",
        "T": "K",
        "dt": "s",
        "area": "m^2",
    },
    "outputs": {"z": "m", "precip": "kg/m^2", "active": "1"},
    "sign_convention": (
        "Terminal velocity is a positive fall speed; z decreases monotonically "
        "for active droplets. Precipitation is non-negative and accumulates; "
        "airborne represented water + accumulated precip·area is conserved."
    ),
    "conserves": ["mass"],
    "differentiable": False,
    "reference": "Shima et al. (2009) QJRMS 135:1307; ERF SuperDropletPC AdvectParticles / rainAccumulation",
    "idealized_test": (
        "Quiescent column: every droplet reaches the surface in z0/v_t(R) "
        "(larger drops first); final precip·area equals the initial airborne "
        "represented water exactly; conservation holds at every step."
    ),
}


def sediment_step(
    state: SuperDropletState,
    z: jax.Array,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
    dt: float | jax.Array,
    area: float | jax.Array,
    cfg: SDMConfig,
) -> tuple[SuperDropletState, jax.Array, jax.Array]:
    """Advance droplet heights by one sedimentation step.

    Parameters
    ----------
    state : SuperDropletState
        Droplet ensemble (``multiplicity`` = represented count in the column).
    z : jax.Array
        Droplet heights [m] above the surface, shape ``(n_sd,)``.
    rho, p, T : float
        Ambient air density [kg/m³], pressure [Pa], temperature [K] — consumed
        by the ``cloud_rain_shima`` terminal velocity; the simpler laws ignore
        them.
    dt : float
        Time step [s].
    area : float
        Horizontal area [m²] the column represents (converts the deposited
        represented mass [kg] to precipitation [kg/m²]).
    cfg : SDMConfig
        Selects the terminal-velocity law (static argument).

    Returns
    -------
    (state_new, z_new, dprecip)
        Updated ensemble (crossed droplets deactivated), new heights (floored
        at 0), and this step's surface precipitation increment [kg/m²].
    """
    dtype = state.radius.dtype
    dt = jnp.asarray(dt, dtype=dtype)
    rho_a = jnp.asarray(rho, dtype=dtype)
    p_a = jnp.asarray(p, dtype=dtype)
    T_a = jnp.asarray(T, dtype=dtype)

    v_t = terminal_velocity(state.radius, rho_a, p_a, T_a, cfg)
    z_new = z - v_t * dt

    was_active = state.active > 0.0
    crossed = was_active & (z_new <= 0.0)

    m_rep = state.multiplicity * water_mass_per_droplet(state)  # [kg] each
    dprecip = jnp.sum(jnp.where(crossed, m_rep, 0.0)) / area    # [kg/m^2]

    active_new = jnp.where(crossed, 0.0, state.active)
    z_new = jnp.where(was_active, jnp.maximum(z_new, 0.0), z)

    return state._replace(active=active_new), z_new, dprecip


def column_rainout(
    state0: SuperDropletState,
    z0: jax.Array,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
    dt: float | jax.Array,
    n_steps: int,
    area: float | jax.Array,
    cfg: SDMConfig,
) -> tuple[SuperDropletState, jax.Array, jax.Array, dict]:
    """Rain out a quiescent column for ``n_steps`` (``lax.scan`` driver).

    Returns ``(state, z, precip_total, history)`` where ``precip_total`` is the
    accumulated surface precipitation [kg/m²] and ``history`` stacks per-step
    ``precip`` (cumulative, kg/m²), ``airborne`` (represented water still
    airborne, kg), and ``n_active``. Precipitation in mm of liquid water is
    ``precip_total / constants.rho_water * 1000`` (the ERF unit conversion).

    Wrap in ``jax.jit`` with ``static_argnames=("n_steps", "cfg")``.
    """
    def body(carry, _):
        state, z, precip = carry
        state, z, dp = sediment_step(state, z, rho, p, T, dt, area, cfg)
        precip = precip + dp
        airborne = jnp.sum(state.active * state.multiplicity
                           * water_mass_per_droplet(state))
        diag = {
            "precip": precip,
            "airborne": airborne,
            "n_active": jnp.sum(state.active),
        }
        return (state, z, precip), diag

    z0 = jnp.asarray(z0, dtype=state0.radius.dtype)
    precip0 = jnp.zeros((), dtype=state0.radius.dtype)
    (state, z, precip), history = lax.scan(
        body, (state0, z0, precip0), xs=None, length=n_steps)
    return state, z, precip, history
