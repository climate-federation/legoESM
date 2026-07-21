"""Field-level seed baseline for the param-spec gate's TODO modules.

``PARAM_SPEC_TODO`` (in ``tests/test_param_specs.py``) allowlists whole *modules*
that do not yet carry a ``__param_spec__``. Without a field-level floor, a NEW
``: float`` default added to one of those pre-existing modules would slip through
unclassified (the module is already allowlisted) — the param-hygiene invariant
could regress silently in exactly the ~30 modules where most physics configs
live. This baseline closes that gap: it records the set of spec-required
``Class.field`` names present in each TODO module at seed time, and the gate
asserts a TODO module's CURRENT float-default fields are a SUBSET (removals fine,
additions red — a new tunable must be specced/excluded or the whole module
graduated out of TODO).

SHRINK-ONLY, in lockstep with ``PARAM_SPEC_TODO``: when a module graduates (gains
a complete ``__param_spec__`` and leaves ``PARAM_SPEC_TODO``), delete its entry
here too. Regenerate with ``scripts/tmp/_seed_param_spec_baseline.py``.

Empty since 2026-07-11 (CLUBB graduated — clubb.py ships a full __param_spec__),
in lockstep with the now-empty ``PARAM_SPEC_TODO``.
"""

from __future__ import annotations

PARAM_SPEC_FIELD_BASELINE: dict[str, frozenset[str]] = {}
