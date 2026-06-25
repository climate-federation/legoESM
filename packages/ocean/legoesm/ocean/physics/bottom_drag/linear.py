"""Linear bottom drag: du/dt = -r * u / dz_bottom.

The coefficient r has units [m/s] so that the bottom stress
tau = rho_0 * r * u is independent of vertical resolution.
This matches MITgcm's ``bottomDragLinear`` convention and is
consistent with the quadratic drag implementation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.physics.bottom_drag.config import LinearDragConfig
from legoesm.ocean.physics.bottom_drag.output import bottom_level_drag_output
from legoesm.ocean.vertical import OceanZStarCoordinate

# Machine-checked scheme contract (see tests/test_physics_contracts.py and
# docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md).
__physics_contract__ = {
    "summary": (
        "Linear bottom drag: a momentum sink applied at the deepest wet level, "
        "du/dt = -r u / dz_bottom."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s",
        "jacobian": "1 (z-star dimensionless)",
        "z_coord.dz_ref": "m",
        "cfg.r": "m/s",
    },
    "outputs": {"du_dt": "m/s^2", "dv_dt": "m/s^2"},
    "sign_convention": (
        "Drag opposes near-bottom velocity: du_dt = -r u / dz_bottom <= 0 for "
        "u>0; non-zero only at the bottom level (zeros above)."
    ),
    # Momentum sink into the solid bottom — not conserved within the fluid.
    "conserves": ["none"],
    "differentiable": True,
    "reference": "MITgcm bottomDragLinear; bottom stress tau = rho_0 r u, r in [m/s].",
    "idealized_test": (
        "u=0 -> zero drag; a constant near-bottom u spins down exponentially with "
        "e-folding time dz_bottom / r."
    ),
}

_EPS = float(jnp.finfo(jnp.float32).eps)


def linear_bottom_drag(
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: LinearDragConfig,
) -> BottomDragOutput:
    """Apply linear bottom drag at the deepest level.

    du/dt = -r * u / dz_bottom   [m/s^2]

    where dz_bottom = dz_ref[-1] * jacobian is the physical thickness
    of the bottom layer.

    Parameters
    ----------
    u, v : array (..., nlev)
        Velocity components.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides reference layer thicknesses).
    jacobian : array (...)
        z-star Jacobian (eta + H_bathy) / H_max at cell centers.
    cfg : LinearDragConfig
        Configuration with r in [m/s].

    Returns
    -------
    BottomDragOutput
    """
    # Bottom layer thickness
    dz_bottom = z_coord.dz_ref[-1] * jacobian  # (...)
    inv_dz = 1.0 / jnp.maximum(dz_bottom, _EPS)

    drag_u = -cfg.r * u[..., -1] * inv_dz
    drag_v = -cfg.r * v[..., -1] * inv_dz
    # Place the bottom-level drag at the deepest level, zeros above (shared).
    return bottom_level_drag_output(drag_u, drag_v, u.shape[-1])
