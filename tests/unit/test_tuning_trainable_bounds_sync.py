"""Registry<->trainable bound synchronization (#595 / PR10, codex review).

``trainable_params.py`` declares its bounds "must match tuning.py validated
ranges".  Pin it so widening one catalog without the other can't silently
drift (the PR10 regression that prompted this): for every parameter name that
appears in BOTH the tuning catalog (``TUNING_PARAMETERS``) and a trainable
``ParamConstraint``, the ``(min_val, max_val)`` bounds must agree.
"""
from __future__ import annotations

import legoesm.training.trainable_params as tp
from legoesm.tuning import TUNING_PARAMETERS


def _all_trainable_constraints() -> dict[str, tp.ParamConstraint]:
    """Collect every ParamConstraint defined across the trainable tier lists
    (DEFAULT_TRAINABLE + the per-scheme tiers), keyed by name.  Tiers that
    repeat a name must carry identical bounds (asserted below)."""
    seen: dict[str, tp.ParamConstraint] = {}
    for value in vars(tp).values():
        if not isinstance(value, list):
            continue
        for c in value:
            if isinstance(c, tp.ParamConstraint):
                if c.name in seen:
                    assert (c.min_val, c.max_val) == (
                        seen[c.name].min_val, seen[c.name].max_val), (
                        f"{c.name}: trainable tiers disagree on bounds "
                        f"({c.min_val},{c.max_val}) vs "
                        f"({seen[c.name].min_val},{seen[c.name].max_val})")
                seen[c.name] = c
    return seen


def test_trainable_bounds_match_tuning_catalog():
    constraints = _all_trainable_constraints()
    assert constraints, "no ParamConstraint instances found in trainable_params"
    mismatches = []
    for name, c in constraints.items():
        if name not in TUNING_PARAMETERS:
            continue
        t = TUNING_PARAMETERS[name]
        if (c.min_val, c.max_val) != (t.min_val, t.max_val):
            mismatches.append(
                f"  {name}: trainable ({c.min_val}, {c.max_val}) != "
                f"tuning ({t.min_val}, {t.max_val})")
    assert not mismatches, (
        "trainable<->tuning bound drift (trainable_params.py bounds must match "
        "tuning.py validated ranges):\n" + "\n".join(mismatches))


def test_shared_names_exist():
    """At least the known shared scalar knobs are in both catalogs (guards the
    test from silently passing if an import/refactor empties one side)."""
    constraints = _all_trainable_constraints()
    shared = set(constraints) & set(TUNING_PARAMETERS)
    assert {"sbm_tau_c", "albedo_ocean"} <= shared, (
        f"expected sbm_tau_c + albedo_ocean shared; got {sorted(shared)[:10]}")
