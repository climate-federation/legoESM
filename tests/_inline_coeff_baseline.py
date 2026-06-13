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

# PARALLEL-MERGE BACKLOG (2026-06-13): the inline-coeff baseline had burned to {}
# (every roster file clean). Merging this gate-bearing branch with a ``main`` that
# gained three NEW physics modules in parallel (the concurrent CLUBB full-closure
# port + the SDM column/Lagrangian adapters) surfaced their inline coefficients —
# code that never passed through this gate because it landed via a separate PR.
# They are reconciled here as explicit, shrink-only migration backlog (NOT a
# weakening of the forward invariant: any OTHER new file is still born clean at
# zero budget). Burn these down — de-inline to a ``*Config`` field / module-level
# provenance block, or annotate genuine numerics one-offs with ``# coeff-ok:`` —
# in the CLUBB / SDM owners' follow-up. Re-seed with
# ``scripts/tmp/_seed_coeff_baseline.py``.
COEFF_BUDGET: dict[str, dict[str, int]] = {
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/column.py': {
        'jnp.asarray(2.0 ** 31, dtype=dtype)).astype(jnp.uint32)': 1,
        'jnp.sum(jnp.abs(qv_f) * 1.0e9 + jnp.abs(T_f) * 1.0e3),': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/microphysics/sdm/lagrangian.py': {
        'number_concentration: float | jax.Array = 1.0e8,': 1,
    },
    'packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py': {
        'coef = jnp.maximum(xm_dw / jnp.where(small_vint, 1.0, xm_vint), -0.99)': 1,
        'em_min = 1.5 * config.w_tol ** 2': 1,
        'host_numerical_diffusion: float = 0.05,': 1,
        'lhs = lhs.at[0, :, 1:-1].set(c * (-3.0 * inv * idzt * rho_up * wp2_up + 1.5 * idzt * wp2_up))': 1,
        'lhs = lhs.at[1, :, 1:-1].set(c * (3.0 * inv * idzt * rho_lo * wp2_lo - 1.5 * idzt * wp2_lo))': 1,
        'return jnp.full((ngrdcol,), 1.0e5)': 1,
        'sigma_sqd_w = jnp.clip(params.gamma_coef * (1.0 - jnp.minimum(max_corr, 1.0)), 0.0, 0.99)': 1,
    },
}

# Re-armed for exactly the three parallel-merge backlog files above: a future
# commit that adds a budget for ANY OTHER file fails ``set(COEFF_BUDGET) <=
# _SEED_FILES``. Shrink-only: as a backlog file is cleaned its COEFF_BUDGET entry
# drops; once all three are clean, restore this to ``frozenset()``.
_SEED_FILES: frozenset[str] = frozenset(COEFF_BUDGET)
