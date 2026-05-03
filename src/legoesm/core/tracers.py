"""Unified tracer management for legoESM.

Provides a ``TracerRegistry`` that maps tracer names to metadata, and
helper functions for creating, stacking, and transporting tracer arrays.

All tracers are stored in a ``dict[str, jax.Array]`` which is a native
JAX pytree, enabling automatic differentiation, ``jit``, and ``vmap``
over the full tracer set.

Example
-------
>>> registry = make_moisture_registry()
>>> tracers = init_tracers(registry, shape_3d=(6, 48, 48, 40))
>>> tracers["q_v"] = tracers["q_v"].at[...].set(initial_q_v)
>>> # Transport all tracers at once:
>>> stacked = stack_tracers(tracers, registry)  # (6, 48, 48, 40, n_tracers)
>>> unstacked = unstack_tracers(stacked, registry)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.core.precision import get_policy


class TracerInfo(NamedTuple):
    """Metadata for a single tracer species."""
    name: str
    units: str = "kg/kg"
    positive_definite: bool = True
    transported: bool = True
    long_name: str = ""


class TracerRegistry(NamedTuple):
    """Ordered registry of tracer species.

    ``names`` is the canonical ordering — tracer index ``i`` corresponds
    to ``names[i]``.  The ordering is stable across the simulation.
    """
    tracers: tuple[TracerInfo, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tracers)

    @property
    def n_tracers(self) -> int:
        return len(self.tracers)

    def index(self, name: str) -> int:
        """Return the integer index for a named tracer."""
        for i, t in enumerate(self.tracers):
            if t.name == name:
                return i
        raise KeyError(f"Tracer {name!r} not in registry: {self.names}")

    def has(self, name: str) -> bool:
        return name in self.names


def make_moisture_registry() -> TracerRegistry:
    """Standard water-species tracer set (vapor + cloud + rain)."""
    return TracerRegistry(tracers=(
        TracerInfo("q_v", "kg/kg", True, True, "specific humidity"),
        TracerInfo("q_c", "kg/kg", True, True, "cloud liquid water"),
        TracerInfo("q_r", "kg/kg", True, True, "rain water"),
    ))


def make_full_moisture_registry() -> TracerRegistry:
    """Full microphysics tracer set (6 species + 3 number concentrations)."""
    return TracerRegistry(tracers=(
        TracerInfo("q_v", "kg/kg", True, True, "specific humidity"),
        TracerInfo("q_c", "kg/kg", True, True, "cloud liquid water"),
        TracerInfo("q_r", "kg/kg", True, True, "rain water"),
        TracerInfo("q_i", "kg/kg", True, True, "cloud ice"),
        TracerInfo("q_s", "kg/kg", True, True, "snow"),
        TracerInfo("q_g", "kg/kg", True, True, "graupel"),
        TracerInfo("N_c", "1/kg", True, True, "cloud droplet number"),
        TracerInfo("N_r", "1/kg", True, True, "rain droplet number"),
        TracerInfo("N_i", "1/kg", True, True, "ice crystal number"),
    ))


def init_tracers(
    registry: TracerRegistry,
    shape_3d: tuple,
) -> dict[str, jnp.ndarray]:
    """Allocate zero-initialized tracer arrays.

    Parameters
    ----------
    registry : TracerRegistry
    shape_3d : tuple
        Shape of 3D atmospheric fields, e.g. ``(6, n, n, nlev)``.

    Returns
    -------
    dict[str, jax.Array]
        Mapping from tracer name to zero-filled array of *shape_3d*.
    """
    _sd = get_policy().storage
    return {info.name: jnp.zeros(shape_3d, dtype=_sd) for info in registry.tracers}


def stack_tracers(
    tracers: dict[str, jnp.ndarray],
    registry: TracerRegistry,
) -> jnp.ndarray:
    """Stack tracer dict into a single array with trailing tracer dim.

    Parameters
    ----------
    tracers : dict[str, jax.Array]
        Each value has shape ``(..., nlev)``.
    registry : TracerRegistry
        Defines the stacking order.

    Returns
    -------
    jax.Array, shape ``(..., nlev, n_tracers)``
    """
    arrays = [tracers[name] for name in registry.names]
    return jnp.stack(arrays, axis=-1)


def unstack_tracers(
    stacked: jnp.ndarray,
    registry: TracerRegistry,
) -> dict[str, jnp.ndarray]:
    """Unstack array with trailing tracer dim into tracer dict.

    Parameters
    ----------
    stacked : jax.Array, shape ``(..., nlev, n_tracers)``
    registry : TracerRegistry

    Returns
    -------
    dict[str, jax.Array]
    """
    return {
        name: stacked[..., i]
        for i, name in enumerate(registry.names)
    }


def clip_positive_definite(
    tracers: dict[str, jnp.ndarray],
    registry: TracerRegistry,
) -> dict[str, jnp.ndarray]:
    """Clip positive-definite tracers to zero floor.

    Preserves non-positive-definite tracers unchanged.
    """
    result = {}
    for info in registry.tracers:
        q = tracers[info.name]
        if info.positive_definite:
            q = jnp.maximum(q, 0.0)
        result[info.name] = q
    return result
