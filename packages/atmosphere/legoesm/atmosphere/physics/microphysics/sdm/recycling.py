"""Recycling of deactivated super-droplets (ERF ``SuperDropletPC::Recycle``).

A super-droplet whose multiplicity reached zero (fully consumed by
coalescence) or that rained out of the column (sedimentation) is *recycled*:
reset to a fresh dry-aerosol particle and re-injected, exactly as the ERF
oracle —

* aerosol (solute) mass drawn from the configured initialization mode (here
  the truncated log-normal of :func:`...init.sample_lognormal_radius`);
* water content reset to a tiny seed droplet of radius ``1e-15 m`` (the
  oracle's literal value — effectively dry; the wet radius is then the dry
  aerosol radius and the condensation solver re-equilibrates it);
* multiplicity set to the constant average ``ξ = n_total/n_sd`` (the ERF
  constant-multiplicity branch; the sampled-importance branch is not ported);
* height re-drawn uniformly in ``[z_min, z_max]`` (the recycle bounds);
* the droplet is re-activated.

Recycling INJECTS particles by design — it conserves nothing (the new aerosol
mass and the ~1e-45 kg water seed are sources); pair it with the surface
precipitation sink for a statistically steady rain column. Pure function of
an explicit PRNG key; not differentiable (resampling), like the coalescence MC.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm.init import sample_lognormal_radius
from legoesm.atmosphere.physics.microphysics.sdm.particles import SuperDropletState

__physics_contract__ = {
    "summary": (
        "Recycle deactivated super-droplets as fresh dry aerosol: lognormal "
        "solute mass, tiny water seed (R=1e-15 m, the oracle's value), "
        "constant average multiplicity, uniform re-injection height, "
        "re-activated."
    ),
    "inputs": {
        "multiplicity": "1",
        "radius": "m",
        "solute_mass": "kg",
        "active": "1",
        "z": "m",
        "key": "1 (jax.random PRNG key)",
        "xi_recycle": "1",
        "r_dry_median": "m",
        "geom_std": "1",
        "solute_density": "kg/m^3",
        "z_min": "m",
        "z_max": "m",
    },
    "outputs": {"multiplicity": "1", "radius": "m", "solute_mass": "kg",
                "active": "1", "z": "m"},
    "sign_convention": (
        "Only inactive slots change; active droplets pass through untouched. "
        "Recycling is a SOURCE (new aerosol + negligible water seed) — it "
        "deliberately conserves nothing; the represented number increases by "
        "xi_recycle per recycled slot."
    ),
    "conserves": ["none"],
    "differentiable": False,
    "reference": "Shima et al. (2009) QJRMS 135:1307; ERF SuperDropletPCRecycle",
    "idealized_test": (
        "All-active ensemble passes through bit-identically; an inactive slot "
        "becomes active with xi_recycle, a lognormal dry radius (wet == dry), "
        "the 1e-15 m water-mass seed implied radius bound, and z in "
        "[z_min, z_max]; deterministic for a fixed key."
    ),
}

# ERF's literal water-seed radius for recycled (dry) particles [m].
_WATER_SEED_RADIUS = 1.0e-15


def recycle_inactive(
    state: SuperDropletState,
    z: jax.Array,
    key: jax.Array,
    xi_recycle: float | jax.Array,
    r_dry_median: float,
    geom_std: float,
    solute_density: float,
    z_min: float | jax.Array,
    z_max: float | jax.Array,
) -> tuple[SuperDropletState, jax.Array]:
    """Recycle every inactive slot as a fresh dry-aerosol super-droplet.

    Parameters
    ----------
    state : SuperDropletState
        Droplet ensemble; slots with ``active == 0`` are recycled.
    z : jax.Array
        Droplet heights [m], shape ``(n_sd,)``.
    key : jax.Array
        PRNG key (consumed; split a fresh one per call).
    xi_recycle : float
        Multiplicity given to each recycled droplet (the ERF constant-mode
        average, e.g. ``n_total/n_sd`` of the initialization).
    r_dry_median, geom_std, solute_density : float
        Log-normal dry-aerosol mode (median radius [m], geometric std, solute
        density [kg/m³]) — the recycled solute mass is ``(4/3)π ρ_s r_dry³``.
    z_min, z_max : float
        Re-injection height bounds [m] (ERF ``recyc_zmin/zmax``).

    Returns
    -------
    (state_new, z_new)
        Ensemble with inactive slots recycled + their new heights. Active
        slots are untouched (bit-identical pass-through).
    """
    n_sd = state.radius.shape[0]
    dtype = state.radius.dtype
    k_r, k_z = random.split(key)

    inactive = state.active <= 0.0

    r_dry = sample_lognormal_radius(k_r, n_sd, r_dry_median, geom_std,
                                    dtype=dtype)
    m_solute = (4.0 / 3.0) * jnp.pi * solute_density * r_dry**3
    # ERF resets the water mass to a 1e-15 m seed; the effective wet radius of
    # the recycled (essentially dry) droplet is its dry aerosol radius.
    z_new_draw = z_min + random.uniform(k_z, (n_sd,), dtype=dtype) * (z_max - z_min)

    xi = jnp.where(inactive, jnp.asarray(xi_recycle, dtype), state.multiplicity)
    radius = jnp.where(inactive, r_dry, state.radius)
    solute = jnp.where(inactive, m_solute, state.solute_mass)
    active = jnp.where(inactive, 1.0, state.active).astype(dtype)
    z_out = jnp.where(inactive, z_new_draw, z)

    return (state._replace(multiplicity=xi, radius=radius,
                           solute_mass=solute, active=active),
            z_out)
