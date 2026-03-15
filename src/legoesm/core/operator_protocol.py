"""Operator backend protocol for horizontal differential operators.

Defines a common interface for 2D horizontal differential operators,
enabling physics code to be agnostic about whether it uses centered,
FC-Gram, or FV numerics.

This is an additive module — no existing code is modified.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class HorizontalOperators(Protocol):
    """Protocol for 2D horizontal differential operators on any grid."""

    def gradient_x(self, q: jax.Array) -> jax.Array: ...
    def gradient_y(self, q: jax.Array) -> jax.Array: ...
    def divergence(self, u: jax.Array, v: jax.Array) -> jax.Array: ...
    def curl_z(self, u: jax.Array, v: jax.Array) -> jax.Array: ...
    def laplacian(self, q: jax.Array) -> jax.Array: ...
    def hyperdiffusion(self, q: jax.Array, coeff: float) -> jax.Array: ...


class FCOperatorBackend:
    """Wraps operators_fc.py functions into the HorizontalOperators protocol."""

    def __init__(self, grid, fc_config):
        self.grid = grid
        self.fc_config = fc_config

    def gradient_x(self, q):
        from legoesm.core.operators_fc import fc_gradient_x
        return fc_gradient_x(q, self.grid, self.fc_config)

    def gradient_y(self, q):
        from legoesm.core.operators_fc import fc_gradient_y
        return fc_gradient_y(q, self.grid, self.fc_config)

    def divergence(self, u, v):
        from legoesm.core.operators_fc import fc_divergence
        return fc_divergence(u, v, self.grid, self.fc_config)

    def curl_z(self, u, v):
        from legoesm.core.operators_fc import fc_curl_z
        return fc_curl_z(u, v, self.grid, self.fc_config)

    def laplacian(self, q):
        from legoesm.core.operators_fc import fc_laplacian
        return fc_laplacian(q, self.grid, self.fc_config)

    def hyperdiffusion(self, q, coeff):
        from legoesm.core.operators_fc import fc_hyperdiffusion
        return fc_hyperdiffusion(q, self.grid, self.fc_config, coeff)


class FVOperatorBackend:
    """Wraps operators_fv.py functions into the HorizontalOperators protocol."""

    def __init__(self, grid):
        self.grid = grid

    def gradient_x(self, q):
        from legoesm.core.operators_fv import fv_gradient_x
        return fv_gradient_x(q, self.grid)

    def gradient_y(self, q):
        from legoesm.core.operators_fv import fv_gradient_y
        return fv_gradient_y(q, self.grid)

    def divergence(self, u, v):
        from legoesm.core.operators_fv import fv_divergence
        return fv_divergence(u, v, self.grid)

    def curl_z(self, u, v):
        from legoesm.core.operators import curl_z as _curl_z
        from legoesm.core.field import Field
        u_f = Field(data=u, name="u", dims=("face", "x", "y"), units="m/s")
        v_f = Field(data=v, name="v", dims=("face", "x", "y"), units="m/s")
        return _curl_z(u_f, v_f, self.grid).data

    def laplacian(self, q):
        from legoesm.core.operators import laplacian as _lap
        from legoesm.core.field import Field
        f = Field(data=q, name="q", dims=("face", "x", "y"), units="")
        return _lap(f, self.grid).data

    def hyperdiffusion(self, q, coeff):
        from legoesm.core.operators import hyperdiffusion as _hyp
        from legoesm.core.field import Field
        f = Field(data=q, name="q", dims=("face", "x", "y"), units="")
        return _hyp(f, self.grid, coeff).data
