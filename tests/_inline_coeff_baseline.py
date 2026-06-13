"""Seed baseline for the inline-physics-coefficient ratchet.

``COEFF_BUDGET`` is the per-file allowance of sanctioned inline empirical-float
(and large-int) sites measured from the tree when the gate
(``tests/test_no_inline_physics_coeffs``) was introduced. Format:
``{repo_rel_path: {normalized_source_line: count}}``. EXACT / shrink-only: a
migrated file must drop (or reduce) its entry in the same commit; a file absent
here gets a zero budget (new files born clean); ``set(COEFF_BUDGET) <=
_SEED_FILES`` is asserted so a new file can never be budgeted. Re-seed with
``scripts/tmp/_seed_coeff_baseline.py``.
"""

from __future__ import annotations

COEFF_BUDGET: dict[str, dict[str, int]] = {
}

# Baseline fully burned down: every physics roster file is clean. The gate is now
# a pure new-code-clean tripwire (a new inline coefficient fails immediately).
# IMMUTABLE EMPTY (not derived from COEFF_BUDGET): so ``test_budget_only_shrinks``
# stays non-vacuous — any future commit that re-adds a file allowance to
# COEFF_BUDGET is now caught (set(COEFF_BUDGET) <= _SEED_FILES would fail).
_SEED_FILES: frozenset[str] = frozenset()
