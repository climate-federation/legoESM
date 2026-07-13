"""Constant-coefficient vertical mixing.

Wraps the existing vertical_diffusion from mixing.py with constant
viscosity A_v and diffusivity K_v.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.mixing import vertical_diffusion
from legoesm.ocean.physics.vertical_mixing._shared import vmap_vertical_diffusion
from legoesm.ocean.physics.vertical_mixing.config import ConstantVerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate

__physics_contract__ = {
    "summary": (
        "Constant-coefficient vertical mixing: apply a uniform vertical "
        "viscosity A_v to (u, v) and diffusivity K_v to (T, S) as "
        "down-gradient diffusion with no-flux top/bottom boundaries."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "degC", "S": "psu",
        "jacobian": "1 (z-star dimensionless)",
        "cfg.A_v": "m^2/s", "cfg.K_v": "m^2/s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "K_v": "m^2/s", "A_v": "m^2/s",
    },
    "sign_convention": (
        "A_v, K_v >= 0; Fickian down-gradient flux F = -K dq/dz; z positive up; "
        "no-flux top and bottom BC, so the dz-weighted column integral of each "
        "diffused field is invariant; surface fluxes applied separately. With "
        "apply_diffusion=False the tendencies are zero and only K_v/A_v are "
        "returned for the implicit solver."
    ),
    # Flux-form diffusion with no-flux BC conserves the column integral of the
    # diffused heat (energy), salt and momentum.
    "conserves": ["energy", "salt", "momentum"],
    "differentiable": True,
    "reference": (
        "Fickian vertical diffusion; Griffies (2004) Fundamentals of Ocean "
        "Climate Models (constant-coefficient vertical mixing)"
    ),
    "idealized_test": (
        "rest / vertically uniform column -> zero tendency; a two-layer step "
        "relaxes toward the column mean while the dz-weighted column integral "
        "of T, S, u, v is conserved (no-flux BC)."
    ),
}


def constant_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: ConstantVerticalMixingConfig,
    apply_diffusion: bool = True,
    dt: float | None = None,
) -> VerticalMixingOutput:
    """Apply constant-coefficient vertical diffusion to u, v, T, S.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
        Velocity components.
    T, S : array (6, n, n, nlev)
        Temperature and salinity.
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : ConstantVerticalMixingConfig

    Returns
    -------
    VerticalMixingOutput
    """
    # The latitude-dependent background (Gregg 2003 / CVMix) needs the column
    # latitude + N^2 and is applied ONLY on the implicit vertical-mixing path
    # (k_profiles.compute_vertical_K_profiles).  Fail loud rather than silently
    # returning a spatially-constant field the caller did not ask for.
    if getattr(cfg, "lat_dependent", False):
        raise ValueError(
            "ConstantVerticalMixingConfig.lat_dependent=True is only honoured "
            "on the IMPLICIT vertical-mixing path "
            "(k_profiles.compute_vertical_K_profiles); the explicit "
            "constant_vertical_mixing tendency path cannot apply the latitude "
            "field.  Use implicit vertical mixing, or set lat_dependent=False.")
    nlev = u.shape[-1]

    # Velocities with viscosity A_v / tracers with diffusivity K_v.
    # When ``apply_diffusion`` is False, the diffusion is deferred to a
    # backward-Euler implicit solve in the dynamics step.  dt (when provided)
    # is passed so the explicit-Euler CFL cap fires inside ``vertical_diffusion``.
    vel_tend, tr_tend = vmap_vertical_diffusion(
        u, v, T, S, cfg.A_v, cfg.K_v,
        lambda q, c: vertical_diffusion(q, z_coord, jacobian, c, dt=dt),
        apply_diffusion,
    )

    # Constant K/A diagnostics at interfaces
    K_diag = jnp.full((*u.shape[:-1], nlev - 1), cfg.K_v, dtype=u.dtype)
    A_diag = jnp.full((*u.shape[:-1], nlev - 1), cfg.A_v, dtype=u.dtype)

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0],
        dS_dt=tr_tend[1],
        K_v=K_diag,
        A_v=A_diag,
    )
