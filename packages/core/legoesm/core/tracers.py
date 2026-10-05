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
        # UNIT CONVENTION — all three numbers are STORED AND TRANSPORTED PER
        # MASS [#/kg], so a dycore's mass-mixing-ratio advection is the right
        # operator for them and the ratio q/N that sets particle size is
        # transport-invariant.  The microphysics formulas and the radiation
        # r_eff coupling still work in PER-VOLUME [#/m³] for N_c and N_r
        # (x_c=q_c·ρ/N_c, dN_r_au∝ρ/x*; see _warm_rain.effective_Nc +
        # morrison/seifert_beheng); the conversion happens at the physics
        # bridge, where ρ is in hand — microphysics/integration.py,
        # ``number_per_mass_to_per_volume``.  N_i was always per-mass.
        # Storing N_c/N_r per volume instead, as this did until 2026-08-14,
        # let them drift against their own mass by the density change along a
        # trajectory.  A wrong label here injects a ρ-factor error into
        # restarts and any diagnostic that keys off the registry units.
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


