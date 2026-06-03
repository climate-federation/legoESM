# Ocean physics — codex /adversarial-review follow-up (iter-3)

After the physics-validator agent's two internal codex rounds, an
independent `/codex:adversarial-review --scope working-tree` pass was
run on the same working tree.  Codex returned three findings.  This
document records the triage and what was done.

## Finding #1 (high) — bottom drag double-applied in explicit barotropic

**Codex claim.**  ``implicit_bottom_drag_factor`` is applied at every
explicit barotropic substep on top of the depth-mean bottom drag that
``F_slow_u`` already carries (because the 3D PE solver added bottom
drag to ``du_dt`` before depth-averaging).  Effective barotropic
drag is ``≈ 2·r/H`` rather than ``r/H``.  The implicit-CN barotropic
path does not apply this factor, so it is internally consistent.

**Verification.**  Confirmed by reading:
- ``src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:1265-1277``
  adds ``-r·u[..., -1]/dz_bot`` to ``du_dt`` at the bottom layer.
- ``src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:422``
  computes ``F_slow_u = depth_mean(du_dt)`` — depth-mean of the bottom
  drag is ``-r·u_bot/H``.
- ``src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:340-345``
  multiplies ``U_bar_new`` by ``implicit_bottom_drag_factor`` — second
  application.
- ``src/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py``
  has no analogous multiplier.

**Pre-existing scope.**  The double-application predates the recent
fix.  The validator's iter-2 change from ``1 - dt·r/H`` to
``1/(1 + dt·r/H)`` swapped formulas of equal first-order magnitude;
both forms double-apply identically to leading order in ``dt·r/H``.
The recent change is justified for shallow-water safety (no velocity
sign flip when ``dt·r/H ≥ 1``) and is **not** the source of the
double-application bug.

**Action.**  *Documentation only.*  Updated the
``implicit_bottom_drag_factor`` docstring at
``src/legoesm/ocean/dynamics/ocean_tendency_common.py:250`` to record
the limitation and the architectural ownership question.  A behavior
fix (single-owner drag) requires either:

- subtract the depth-mean bottom drag from ``F_slow_u`` in
  ``ocean_model_latlon_cgrid.py`` and ``ocean_model_mpas.py`` before
  passing to the barotropic substep, OR
- skip adding bottom drag in the 3D ``du_dt`` path that feeds
  ``F_slow_u``.

Either path is a multi-file change with measurable production impact
on ocean kinetic-energy decay (drag timescale doubles).  Validation
requires AMIP comparison runs.  Out of scope for a unit-test triage;
flagged as architectural debt for a paired follow-up PR.

## Finding #2 (high) — KPP reads forcings outside the ocean budget

**Codex claim.**  ``_make_kpp`` reads ``Q_net``, ``Q_S``, ``tau`` from
``surface_forcing`` directly to drive KPP, but the configured
prescribed/bulk surface-forcing wrapper discards its diagnostic
fluxes before they reach the column budget.  KPP can therefore react
to fluxes that never enter the heat / salt / free-surface tendencies,
or fall back to proxies while a different scheme heats/salts the top
layer.

**Status.**  Pre-existing architectural inconsistency, already listed
in the validator's iter-2 residuals (#1).  Codex re-confirmed.  Fix
requires routing one surface-forcing diagnostic object through the
combined physics pipeline so KPP and the budget application share a
single source of truth.

**Action.**  No code change.  Keeps the existing residual entry in
the validator report.

## Finding #3 (medium) — plume convection is non-conservative

**Codex claim.**  ``plume_convection`` (``src/legoesm/ocean/physics/
convection/plume.py``) detrains heat / salt to environment levels
``k ≥ 1`` without a compensating sink at the source layer, leaving
the column-integrated tendency non-zero.

**Verification.**  Probe in
``.physics-validator/ocean-revalidation/`` with a static-instability
column (cold dense surface above warm interior) gave
``Σ_k dT_dt[k]·dz[k] = -1.88 × 10⁻³ K·m/s`` — clearly non-zero.  The
salt sum was zero only because the test happened to use a uniform
``S`` profile.

**Action.**  *Fix applied.*  Added a column-integral correction at
the surface layer:

```python
column_dT = jnp.sum(dT_dt * dz_actual, axis=-1)
column_dS = jnp.sum(dS_dt * dz_actual, axis=-1)
dT_dt = dT_dt.at[..., 0].add(-column_dT / dz_top)
dS_dt = dS_dt.at[..., 0].add(-column_dS / dz_top)
```

This treats the surface mixed layer as the plume's source: detrained
heat / salt at depth is balanced by an equivalent removal at ``k=0``.
Mass is unchanged (redistribution within the column, no flux through
boundaries).

**Test.**  Added
``tests/ocean/unit/test_plume_convection.py::test_plume_conserves_column_heat_and_salt``
which exercises a column with non-uniform ``T`` *and* non-uniform
``S`` and asserts ``max |Σ dT·dz|, max |Σ dS·dz| < 1e-12``.  The
existing ``test_plume_outputs_finite_and_signs_consistent`` was
updated to drop the now-incorrect assertion that ``dT_dt[..., 0]
== 0`` (the surface row is precisely where the conservation
correction lives).

## Files changed by this follow-up

| File | Change |
| --- | --- |
| ``src/legoesm/ocean/physics/convection/plume.py`` | Added column-integral conservation correction (~10 lines). |
| ``src/legoesm/ocean/dynamics/ocean_tendency_common.py`` | Expanded docstring on ``implicit_bottom_drag_factor`` to document the duplicate-application architecture. |
| ``tests/ocean/unit/test_plume_convection.py`` | New ``test_plume_conserves_column_heat_and_salt``; existing test loosened to drop the surface-row-zero assertion. |

## Open architectural debt (carried forward)

1. **Single-owner bottom drag** between ``F_slow_u`` and the
   barotropic substep multiplier (codex #1 — multi-file change).
2. **KPP / surface-forcing flux plumbing** (codex #2 — multi-file
   change).
3. **Function-default constants** in
   ``shortwave_penetration.py``, ``biogeochemistry/{gas_exchange,
   carbonate}.py`` (CLAUDE.md-forbidden literal defaults — pre-existing
   tech debt).
4. **``KPPConfig.Ri_conv``** named like a Richardson threshold but
   compared against ``N²`` (API rename).
5. **``test_longrun_conservation_with_fixer``** failure (heat drift
   3.46e-7 vs 1e-8 tolerance — conservation-fixer algorithmic
   limitation, not introduced by these rounds).
