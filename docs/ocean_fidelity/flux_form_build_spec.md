# Build Spec — flux-form horizontal momentum advection (lat-lon C-grid)

> Autonomous build contract (Ralph). Re-read IN FULL each iteration. Branch:
> `matching_Veros_oracle`. Companions: `flux_form_momentum_scope.md` (the design),
> `oracle_recipe_strategy.md` (doctrine rule H + apples-to-apples), and the progress log
> `flux_form_build_progress.md` (the loop's memory — update every iteration).

## Mission
Add a **flux-form** horizontal momentum advection option to the canonical lat-lon C-grid dycore,
config-selectable, so legoESM can assemble the Veros-ACC-active block set (apples-to-apples).
This is doctrine rule H "add the missing method to the canonical module" — NOT a `veros_*` clone.
Done = every gate below is honestly green. Emit `<promise>FLUXFORMDONE</promise>` only then.

## Operating rules (every iteration)
- **CLAUDE.md is absolute**: verification-first, no duplicate numerics (reuse the existing
  face-interp/limiter helpers + `divergence_cgrid`, do not reinvent), dispatch `raise ValueError`
  on unknown literals, every new `.py`/symbol gets a direct test, `JAX_ENABLE_X64=1` for science.
- **ANTI-GAMING**: never weaken a test, loosen a tolerance, clip/mask numerics, or narrow a check
  to manufacture green. A gate that can't be met honestly is an escalation, not a license to relax.
- **Truth-tiers-first**: this scheme is trusted because it passes conservation / analytic /
  equivariance / stability gates — NOT because it matches Veros. Oracle match is the LAST, lowest-
  trust check.
- **Minimal diffs, incremental commits**: commit each green increment (explicit paths only). The
  function must never be left broken at a commit boundary.
- **Locate yourself each iteration**: read the progress log, `git log --oneline -15`, run the gates.

## Where it plugs in
`ocean/dynamics/ocean_pe_latlon_cgrid.py`: `_mom_adv = config.momentum_advection` (~line 2101);
dispatch at the `_bc_pv_flux(...)` call (~line 2146). Add a `flux_form` branch calling a new
substage `_bc_horizontal_momentum_advection_flux_form(du_dt, dv_dt, u, v, h_u, h_v, h_k,
u_mask_3d, v_mask_3d, mask, grid, config, z_coord)`; else keep `_bc_pv_flux`. Config in
`state.py LatLonCGridOceanConfig`: add `"flux_form"` to the `momentum_advection` doc + a
`momentum_flux_scheme: str = "upwind"` field. Reuse `divergence_cgrid`, `_upwind_to_u/v_points`,
`_tvd_to_u/v_points`, `min_cell_to_uface/vface`, `compute_face_masks_3d`, `_neumann_fill_cgrid`.

## Acceptance gates (ALL must be honestly green)

- **F1 — existing paths bit-identical.** `tests/ocean/unit/test_baroclinic_decomposition.py`
  stays green: adding the `flux_form` branch must not change `vector_invariant`/`weno5`/`weno7`
  output at all.
- **F2 — dispatch discipline.** `momentum_advection` validated against
  `{vector_invariant, weno5, weno7, flux_form}` with `ValueError` on unknown (fix the current
  silent fallthrough) in `_validate_config`; `momentum_flux_scheme` validated against its set.
  A dispatch test in `test_config_footguns.py` covers both (valid pass, unknown raises).
- **F3 — zero-velocity ⇒ zero.** With `u=v=0`, the flux-form horizontal-advection contribution
  is exactly 0 (a direct substage test).
- **F4 — uniform-flow analytic.** Uniform `u=const, v=0` on a flat doubly-periodic domain ⇒
  advective tendency ≈ 0 to round-off (∇·(uu) of constant flux = 0). Direct test.
- **F5 — momentum conservation.** On a flat doubly-periodic domain the domain-integrated
  flux-form advective tendency (Σ over wet cells) is at machine-ε level for u and v (advection
  redistributes momentum, does not create it). Direct test. (Upwind is *dissipative* in energy —
  that is correct, NOT a violation; do not assert energy conservation for upwind.)
- **F6 — differentiability.** `jax.grad` of a scalar loss through the flux-form path is finite
  and nonzero (add to `test_ocean_differentiability.py`).
- **F7 — idealized-gyre stability.** A Munk/Stommel gyre (`experiments/munk_gyre.py` or
  `baroclinic_gyre.py`) run with `momentum_advection="flux_form"` (upwind) for a modest horizon
  stays finite (no NaN/Inf) and KE plateaus (does not blow up). Compare it RUNS against the
  vector-invariant baseline (different numbers expected; both stable).
- **F8 — regression lock.** Once F1–F7 pass, add a `flux_form` case to the
  `test_baroclinic_decomposition.py` golden (regenerate) so the new path is bit-identical-locked
  going forward.
- **F9 — oracle confirmation (LAST, informational).** Re-run the ACC tier-2 comparison with
  `momentum_advection="flux_form"`; record whether `du_adv` corr improves toward Veros in the §8
  ledger. Not a pass/fail gate (Veros↔legoESM still differ in grid/EOS-arg details); document.

## Verify-before-commit
Run the narrowest relevant gate(s) after each edit + the F1 bit-identical gate for any orchestrator
touch. Full pytest suite is NOT runnable here (exit-143) — use targeted gates + the bit-identical
gate + the grad/conservation/analytic tests; record what ran. No new failure introduced by this
work (verify pre-existing failures are pre-existing).

## Completion / stop
Emit `<promise>FLUXFORMDONE</promise>` ONLY when F1–F8 are honestly green (F9 documented), the new
substage + tests are committed with explicit paths, and the progress log records the final state.
Emit a BLOCKED note instead if a gate cannot be met honestly after ≥3 genuine iterations (e.g.
conservation fails and the discretization needs a design decision) — write the blocker + options
to the log; do not relax a gate.
