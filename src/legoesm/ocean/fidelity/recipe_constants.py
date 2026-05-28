"""Recipe-level constants override mechanism for Phase G.

Veros's canonical constants differ from legoESM's at the 0.02-0.1% level:

==========  ==========  ==========
Constant    legoESM     Veros
==========  ==========  ==========
g           9.80616     9.81
R_earth     6.371229e6  6.370e6
rho_ocean   1025.0      1024.0
Omega       7.292e-5    7.292115e-5
==========  ==========  ==========

For Phase G tier-2 tendency comparison this drift is large enough to
contaminate the comparison before scheme-level differences enter. The
override mechanism temporarily pins legoESM's constants to Veros's values
for the duration of a recipe run.

Design notes
------------

Many ocean modules snapshot ``legoesm.constants.X`` at import time
(``rho_0 = constants.rho_ocean`` in ``ocean/eos.py``), and a further
fan-out of modules ``from legoesm.ocean.eos import rho_0 as _RHO_0``
captures the snapshot a second time. Mutating ``legoesm.constants.X``
alone does NOT update either layer — they are bound at import time.

The override mechanism here keeps an explicit registry of every snapshot
site so that ``override_constants(rho_ocean=1024.0)`` propagates through
both ``legoesm.constants.rho_ocean`` and every ``X.rho_0`` /
``Y._RHO_0`` / ``Z.rho_0_ref`` alias used in the ocean stack. The
registry is the **only** source of truth for which constants are
overridable and where their shadows live; adding a new constant to the
override surface requires updating ``_CONSTANT_SHADOWS`` below.

This is the **v1** approach. The **v2** refactor (plumb a
``ConstantsConfig`` NamedTuple through every leaf function that reads a
constant, removing all import-time snapshots) is tracked under
Phase G.4 in ``docs/ocean_fidelity/phase_g_veros_recipe_audit.md``.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Any, Iterator

from legoesm import constants


# Registry of constant → list of (module path, attribute name) snapshot
# sites. Every line is a module-level binding observed via
# ``grep -rE "^[a-zA-Z_]+ = constants\.X\b"`` plus the fan-out of
# ``from legoesm.ocean.eos import X as Y``. Function-local imports are
# safe (they read at call time, so an override of ``constants.X``
# propagates automatically) and are NOT listed.
#
# Adding a new constant to the override surface requires:
#  1. Adding it to this dict (with at minimum the ``("legoesm.constants",
#     X)`` entry so the override touches the canonical site).
#  2. Grepping for any ``Y = constants.X`` and ``from legoesm.ocean.eos
#     import X`` and adding each module-level snapshot here.
#  3. Adding a unit test that override + read-after-import propagates.
_CONSTANT_SHADOWS: dict[str, list[tuple[str, str]]] = {
    "g": [
        ("legoesm.coupler.bulk_flux", "G"),
    ],
    "Omega": [],
    "R_earth": [],
    "rho_ocean": [
        ("legoesm.ocean.eos", "rho_0"),
        # Aliased imports of eos.rho_0:
        ("legoesm.ocean.diagnostics", "_RHO_0"),
        ("legoesm.ocean.physics.shortwave_penetration", "_RHO_0_DEFAULT"),
        ("legoesm.ocean.physics.mpas_physics", "rho_0_ref"),
        ("legoesm.ocean.physics.lateral_mixing._gm_redi_common", "_RHO_0_DEFAULT"),
        ("legoesm.ocean.physics.lateral_mixing.gm_redi_mpas", "_RHO_0"),
        ("legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid", "_RHO_0"),
        ("legoesm.ocean.physics.surface_forcing.prescribed", "rho_0_ref"),
        ("legoesm.ocean.physics.surface_forcing.bulk_formulas", "rho_0_ref"),
    ],
    "c_sw": [
        ("legoesm.ocean.eos", "c_sw"),
        # Aliased imports of eos.c_sw:
        ("legoesm.ocean.physics.shortwave_penetration", "_C_SW_DEFAULT"),
        ("legoesm.ocean.physics.surface_forcing.prescribed", "c_sw"),
        ("legoesm.ocean.physics.surface_forcing.bulk_formulas", "c_sw"),
    ],
}


# Veros's canonical constants, exposed as a named dict so callers can
# build a recipe override directly:
#
#     with override_constants(**VEROS_CONSTANTS):
#         ...
VEROS_CONSTANTS: dict[str, float] = {
    "g": 9.81,
    "Omega": 7.292115e-5,
    "R_earth": 6.370e6,
    "rho_ocean": 1024.0,
    # c_sw is not a Veros recipe knob (Veros linear EOS does not use it
    # directly), but we expose the canonical Veros value here for
    # completeness.
    "c_sw": 3994.0,
}


class UnknownConstantError(ValueError):
    """Raised when ``override_constants`` is called with a constant that
    is not in the override registry."""


def _resolve_targets(name: str) -> list[tuple[Any, str]]:
    """Return [(module_object, attr_name), ...] including
    ``legoesm.constants.<name>`` itself."""
    if name not in _CONSTANT_SHADOWS:
        raise UnknownConstantError(
            f"Constant {name!r} is not in the override registry. "
            f"Known constants: {sorted(_CONSTANT_SHADOWS)}. To add a new "
            f"constant, update _CONSTANT_SHADOWS in this module and add "
            f"a unit test exercising the override + import-snapshot path."
        )
    targets: list[tuple[Any, str]] = [(constants, name)]
    for module_path, attr in _CONSTANT_SHADOWS[name]:
        targets.append((importlib.import_module(module_path), attr))
    return targets


@contextlib.contextmanager
def override_constants(**overrides: float) -> Iterator[None]:
    """Temporarily override legoESM constants and their snapshot sites.

    Parameters
    ----------
    **overrides : float
        Constant name → override value. Each name must appear in
        ``_CONSTANT_SHADOWS``; unknown names raise
        :class:`UnknownConstantError` before any mutation happens.

    Yields
    ------
    None
        Within the ``with`` block, ``legoesm.constants.X`` and every
        snapshot site of ``X`` (per the registry) are set to the
        override value. On exit the original values are restored, even
        if the body raises.

    Notes
    -----
    Mutation is global to the Python process. The mechanism is not
    re-entrant — nested ``override_constants`` calls for the *same*
    constant produce undefined behavior on the original-value stack.
    """
    if not overrides:
        yield
        return

    # Resolve every (module, attr) pair up front so a malformed override
    # (unknown constant) raises before any mutation.
    plans: list[tuple[Any, str, Any, Any]] = []
    for name, new_value in overrides.items():
        targets = _resolve_targets(name)
        for module, attr in targets:
            old_value = getattr(module, attr)
            plans.append((module, attr, old_value, new_value))

    try:
        for module, attr, _old, new in plans:
            setattr(module, attr, new)
        yield
    finally:
        # Restore in reverse order so any computed-from-other-constants
        # state lands back consistent.
        for module, attr, old, _new in reversed(plans):
            setattr(module, attr, old)


def assert_registry_consistent() -> list[str]:
    """Return a list of registry inconsistencies; empty list = OK.

    Useful as a one-line CI check that ``_CONSTANT_SHADOWS`` has not
    silently drifted from the actual constants module. Each registered
    constant must exist in ``legoesm.constants``; each shadow site must
    have a module attribute matching the registered name.
    """
    errors: list[str] = []
    for name, shadows in _CONSTANT_SHADOWS.items():
        if not hasattr(constants, name):
            errors.append(
                f"Constant {name!r} is registered but missing from "
                f"legoesm.constants."
            )
        for module_path, attr in shadows:
            try:
                mod = importlib.import_module(module_path)
            except ImportError as exc:
                errors.append(
                    f"Cannot import shadow module {module_path!r} for "
                    f"constant {name!r}: {exc}"
                )
                continue
            if not hasattr(mod, attr):
                errors.append(
                    f"Shadow site {module_path}.{attr} (for constant "
                    f"{name!r}) does not exist."
                )
    return errors


__all__ = (
    "UnknownConstantError",
    "VEROS_CONSTANTS",
    "assert_registry_consistent",
    "override_constants",
)
