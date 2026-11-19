"""Test helpers for calling Zhang-McFarlane at CAM's cold-start PBL height.

ZM refuses a call without the carried PBL height ``pblh`` (oracle row 50:
CAM always hands zm_convr the pbuf ``pblh``, convect_deep.F90:266).  Tests
that exercise ZM outside a run pass CAM's initial value instead: ``pblh = 0``
(physpkg.F90:419-424), i.e. the launch-level search starts at the lowest
level only (zm_conv.F90:738-741).
"""
from __future__ import annotations

import math

import jax.numpy as jnp

from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection as _zm,
)


def zm_cold_start(*args, **kwargs):
    """``zhang_mcfarlane_convection`` with ``pblh`` defaulted to zeros."""
    T = kwargs["T"] if "T" in kwargs else args[0]
    kwargs.setdefault("pblh", jnp.zeros(jnp.shape(T)[0], jnp.asarray(T).dtype))
    return _zm(*args, **kwargs)


def cold_phys_state(T_native):
    """A fresh ``PhysicsState`` (``pblh = 0``) for a bridge call on a state
    whose temperature field is ``T_native`` (any column layout, levels last)."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    shape = jnp.shape(T_native)
    return init_physics_state(math.prod(shape[:-1]), shape[-1], PhysicsConfig(),
                              dtype=jnp.asarray(T_native).dtype)
