"""Dycore resolution: registry/plugin first, built-in atmosphere solvers second.

Split out of :mod:`legoesm.registry` so the registry substrate itself stays free
of any component import.  ``create_dycore`` falls back to the atmosphere built-in
solvers, so it (not the registry) carries the deferred atmosphere dependency —
keeping ``legoesm.registry`` importable by every component (e.g. the ocean
barotropic resolving an SW core from ``SW_BAROTROPIC_REGISTRY``) without
transitively reaching the atmosphere.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from legoesm.registry import DYCORE_REGISTRY, UnknownRegistryEntryError


def create_dycore(name: str, *args: Any, **kwargs: Any):
    """Resolve a dycore by *name*, instantiate it, and validate the contract.

    ``create_dycore("cdgrid_shallow_water", grid, config)`` -> a validated dycore.
    Extra args/kwargs are forwarded to the resolved factory (``Model(grid,
    config)``).

    Resolution order: an explicitly-registered factory or entry-point plugin
    first (so a plugin can override or add a dycore), then the **built-in**
    atmosphere solvers.  The built-in lookup is a deferred import.  Raises
    ``ValueError`` for an unknown name and ``TypeError`` for a factory whose
    product is not a valid dycore.
    """
    from legoesm.components import validate_dycore

    try:
        factory: Callable = DYCORE_REGISTRY.get(name)
    except UnknownRegistryEntryError:
        # Only a genuine registry MISS falls back to the built-ins.  A ValueError
        # raised while *loading* an entry-point plugin is NOT caught here — it
        # propagates, so a broken plugin override is never silently replaced by
        # the built-in.
        from legoesm.atmosphere.dynamics import get_solver_class

        factory = get_solver_class(name)  # raises ValueError if unknown there too
    dycore = factory(*args, **kwargs)
    validate_dycore(dycore)
    return dycore
