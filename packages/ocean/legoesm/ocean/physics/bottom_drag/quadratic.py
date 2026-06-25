"""Quadratic bottom drag: tau = -C_d * |u| * u."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import QuadraticDragConfig
from legoesm.ocean.physics.bottom_drag.output import bottom_level_drag_output
from legoesm.ocean.vertical import OceanZStarCoordinate

__physics_contract__ = {
    "summary": (
        "Quadratic (drag-law) bottom drag: a momentum sink applied at the "
        "deepest wet level, (du/dt, dv/dt) = -C_d |u| (u, v) / dz_bottom."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s",
        "jacobian": "1 (z-star dimensionless)",
        "z_coord.dz_ref": "m",
        "cfg.C_d": "1 (dimensionless drag coefficient)",
    },
    "outputs": {"du_dt": "m/s^2", "dv_dt": "m/s^2"},
    "sign_convention": (
        "Drag opposes near-bottom velocity: du_dt = -C_d |u| u / dz_bottom, "
        "anti-parallel to (u, v); for each component its drag has the opposite "
        "sign to that component. Non-zero only at the bottom level (zeros above)."
    ),
    # Momentum sink into the solid bottom — not conserved within the fluid.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "MITgcm bottomDragQuadratic; bottom stress tau = rho_0 C_d |u| u, "
        "C_d ~ 1e-3 (dimensionless)."
    ),
    "idealized_test": (
        "u=0 -> zero drag; a constant near-bottom speed spins down as the "
        "quadratic ODE du/dt = -(C_d/dz_bottom) |u| u (algebraic 1/(1+t/tau) "
        "decay, NOT exponential), monotonically toward zero."
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def quadratic_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: QuadraticDragConfig,
) -> BottomDragOutput:
    """Apply quadratic bottom drag at the deepest level.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : QuadraticDragConfig

    Returns
    -------
    BottomDragOutput
    """
    eps = _EPS

    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (6, n, n)

    # Speed at bottom level
    u_bot = u[..., -1]
    v_bot = v[..., -1]
    speed = jnp.sqrt(u_bot**2 + v_bot**2 + eps)

    # Drag: -C_d * |u| * u / dz_bottom
    inv_dz = 1.0 / jnp.maximum(dz_bottom, eps)
    drag_u = -cfg.C_d * speed * u_bot * inv_dz
    drag_v = -cfg.C_d * speed * v_bot * inv_dz

    # Place the bottom-level drag at the deepest level, zeros above (shared).
    return bottom_level_drag_output(drag_u, drag_v, u.shape[-1])
