"""The seeded physics carry has no weak-typed leaf, so a jitted step does not compile twice.

``jnp.full(shape, <python float>, dtype=None)`` is weak-typed; the physics hands the same field
back strongly typed, so a weak seed changes the jitted step's argument signature after the first
step and forces a second full compile (``relvar``: 296 s and 17 GB at MPAS level 1 with isotope
passengers, enso_attribution docs/85 13.36).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.physics.physics_state import init_physics_state


def _cfg(scheme):
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    return PhysicsConfig(turbulence=TurbulenceConfig(scheme=scheme))


def _weak(ps):
    return [jax.tree_util.keystr(p) for p, x in jax.tree_util.tree_flatten_with_path(ps)[0]
            if getattr(x, "weak_type", False)]


@pytest.mark.parametrize("scheme", ["tke", "clubb"])
def test_seed_has_no_weak_leaf(scheme):
    assert _weak(init_physics_state(4, 3, _cfg(scheme))) == []


def test_step_with_strong_output_compiles_once():
    """A jitted step that returns every carry field strongly typed (as the physics does) hits
    its cache on the second call."""
    ps = init_physics_state(4, 3, _cfg("clubb"))
    step = jax.jit(lambda s: jax.tree_util.tree_map(
        lambda x: x * jnp.ones((), x.dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, s))
    ps2 = step(ps)
    step(ps2)
    assert step._cache_size() == 1
