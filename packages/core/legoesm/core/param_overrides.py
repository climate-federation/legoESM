"""Splice trained parameter leaves into a scheme ``*Config`` NamedTuple.

Lives in ``legoesm.core`` rather than ``legoesm.training`` because the
consumers are the COMPONENTS (land carbon calibration, ocean GM/Redi, the
atmosphere physics configs), and a component reaching into the training /
orchestration layer to get a 15-line ``_replace`` wrapper breaks the
"Earth-system components are independent of each other" import contract —
the edge dragged in ``training -> tuning -> driver.config -> atmosphere``.

Doctrine (CLAUDE.md): trained values inject via the config pytree, never via
new function signatures, and the splice happens INSIDE the loss so the
substituted leaves are traced.  Production keeps static Python-float leaves so
they constant-fold and do not trigger a retrace.
"""

from __future__ import annotations

import jax

__all__ = ("apply_param_overrides",)


def apply_param_overrides(config_obj, field_values: dict[str, jax.Array]):
    """Return ``config_obj`` with ``field_values`` spliced in via ``_replace``.

    ``config_obj`` is the scheme's ``*Config`` NamedTuple; ``field_values`` is one
    scheme's slice of :meth:`TrainablePhysicsParams.to_overrides`. Unknown fields
    raise ``ValueError`` (never a silent no-op). Applied inside the training loss
    so the substituted leaves are traced."""
    if not field_values:
        return config_obj
    fields = getattr(config_obj, "_fields", None)
    if fields is None:
        raise ValueError(f"{type(config_obj).__name__} is not a NamedTuple config")
    bad = [k for k in field_values if k not in fields]
    if bad:
        raise ValueError(
            f"{type(config_obj).__name__} has no field(s) {bad}; known: {list(fields)}"
        )
    return config_obj._replace(**field_values)
