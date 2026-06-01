"""Component / dycore registry — discovery + validation (master plan P2 / B4).

A small name -> factory registry with three properties the architecture needs:

* **raise on unknown** — a typo'd scheme fails here with the list of valid names,
  never silently falls through to a default (the dispatch-discipline rule);
* **plugin discovery** — third-party dycores/components register via
  ``importlib.metadata`` entry points (group ``legoesm.<kind>s``), so
  ``pip install legoesm-mycore`` makes ``create_dycore("mycore", ...)`` work with
  no edit to legoESM;
* **contract validation** — a resolved dycore is checked with
  :func:`legoesm.components.validate_dycore` (``isinstance(.., DycoreProtocol)``
  alone is not enough — see that function), so a malformed plugin fails at
  resolution, not deep in a rollout.

This is the discovery mechanism the existing hardcoded dispatch tables
(``_SOLVER_TO_CLASS`` etc.) migrate onto; that rewiring is a follow-up — this
module is additive and changes no existing dispatch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class UnknownRegistryEntryError(ValueError):
    """Raised by :meth:`Registry.get` when *name* is not registered.

    A ``ValueError`` subclass so existing ``except ValueError`` /
    ``pytest.raises(ValueError)`` callers keep working — but a *distinct* type so
    a true miss can be told apart from a ``ValueError`` raised while *loading* an
    entry-point plugin (which must NOT be swallowed as a miss).
    """


class Registry:
    """A ``name -> factory`` registry with raise-on-unknown + entry-point plugins.

    Parameters
    ----------
    kind:
        What is registered (``"dycore"``, ``"component"``); used in error
        messages and the entry-point group name ``legoesm.<kind>s``.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entry_group = f"legoesm.{kind}s"
        self._entries: dict[str, Callable] = {}
        # name -> the exception its entry-point raised at load(); a broken plugin
        # is remembered so EVERY lookup of that name re-raises it (never silently
        # masked by a built-in fallback on a later call).
        self._failed_entry_points: dict[str, Exception] = {}
        self._loaded_entry_points = False

    def register(self, name: str, factory: Callable, *, overwrite: bool = False) -> Callable:
        """Register *factory* under *name*.  Returns *factory* (decorator-friendly)."""
        if not overwrite and name in self._entries:
            raise ValueError(
                f"{self._kind} {name!r} is already registered; pass overwrite=True "
                f"to replace it"
            )
        self._entries[name] = factory
        return factory

    def get(self, name: str) -> Callable:
        """Return the factory for *name*, raising ``ValueError`` if unknown.

        On a miss, entry-point plugins are loaded once (lazily) before failing.
        """
        if name not in self._entries and not self._loaded_entry_points:
            self._load_entry_points()
        if name in self._entries:
            return self._entries[name]
        if name in self._failed_entry_points:
            # A plugin claimed this name but failed to load — re-raise that error
            # on every lookup so a broken override is never silently bypassed.
            raise self._failed_entry_points[name]
        avail = ", ".join(self.available()) or "(none)"
        raise UnknownRegistryEntryError(
            f"Unknown {self._kind} {name!r}. Available: {avail}"
        )

    def available(self) -> list[str]:
        """Sorted list of registered names (loads entry-point plugins first)."""
        if not self._loaded_entry_points:
            self._load_entry_points()
        return sorted(self._entries)

    def _load_entry_points(self) -> None:
        self._loaded_entry_points = True
        from importlib.metadata import entry_points

        try:
            eps = entry_points(group=self._entry_group)
        except TypeError:  # Python <3.10 importlib.metadata API
            eps = entry_points().get(self._entry_group, [])  # type: ignore[call-arg]
        for ep in eps:
            # In-process registrations win over plugins of the same name.
            if ep.name in self._entries:
                continue
            try:
                self._entries[ep.name] = ep.load()
            except Exception as exc:  # one broken plugin must not abort discovery
                self._failed_entry_points[ep.name] = exc


# The dynamical-core registry.  Built-in solvers migrate onto this in a
# follow-up; today it carries plugin dycores + anything explicitly registered.
DYCORE_REGISTRY = Registry("dycore")


def create_dycore(name: str, *args: Any, **kwargs: Any):
    """Resolve a dycore by *name*, instantiate it, and validate the contract.

    ``create_dycore("cdgrid_shallow_water", grid, config)`` -> a validated dycore.
    Extra args/kwargs are forwarded to the resolved factory (``Model(grid,
    config)``).

    Resolution order: an explicitly-registered factory or entry-point plugin
    first (so a plugin can override or add a dycore), then the **built-in**
    atmosphere solvers.  The built-in lookup is a deferred import, so the registry
    substrate stays importable without the atmosphere component (a pure-core
    install resolves plugins only).  Raises ``ValueError`` for an unknown name and
    ``TypeError`` for a factory whose product is not a valid dycore.
    """
    from legoesm.components import validate_dycore

    try:
        factory: Callable = DYCORE_REGISTRY.get(name)
    except UnknownRegistryEntryError:
        # Only a genuine registry MISS falls back to the built-ins.  A ValueError
        # raised while *loading* an entry-point plugin is NOT caught here — it
        # propagates, so a broken plugin override is never silently replaced by
        # the built-in.  Deferred import keeps legoesm.registry decoupled from the
        # atmosphere component.
        from legoesm.atmosphere.dynamics import get_solver_class

        factory = get_solver_class(name)  # raises ValueError if unknown there too
    dycore = factory(*args, **kwargs)
    validate_dycore(dycore)
    return dycore
