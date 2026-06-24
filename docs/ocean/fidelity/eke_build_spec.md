# Build Spec — prognostic EKE (eddy kinetic energy) closure

> Autonomous build contract (Ralph). Re-read IN FULL each iteration. Branch:
> `matching_Veros_oracle`. Companions: `eke_scope.md` (the design), `oracle_recipe_strategy.md`
> (doctrine rule H + apples-to-apples), `flux_form_build_spec.md` (the sibling build, done — same
> discipline), and the progress log `eke_build_progress.md` (the loop's memory — update every
> iteration). This is the SECOND and last ACC apples-to-apples must-build block (flux-form
> momentum was the first; done + adopted).

## Mission
Add a prognostic Eden & Greatbatch (2008) EKE closure to the canonical ocean dycore, so the GM
coefficient becomes prognostic (`kappa_GM = c_k·L·√E`) — config-selectable, default OFF. Then
adopt it in the ACC recipe (Veros ACC runs `enable_eke=True`). Done = every gate below honestly
green. Emit `<promise>EKEDONE</promise>` only then.

## Operating rules (every iteration)
- **CLAUDE.md is absolute**: verification-first; reuse shared blocks (tracer advection for E,
  the GM/Redi isopycnal-diffusion machinery, `compute_visbeck_kappa_gm`, `compute_buoyancy_frequency`)
  — NO duplicate numerics; dispatch `raise ValueError` on unknown literals; every new `.py`/symbol
  gets a direct test; `JAX_ENABLE_X64=1` for science. **SegmentCarry discipline**: adding a state
  field is cross-cutting — update the NamedTuple def, `pack_carry`/`unpack_carry`, every
  `SegmentCarry(...)` constructor + the per-step ref loops in `test_compiled_segments.py`,
  `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, and any direct constructors in tests.
- **ANTI-GAMING**: never weaken a test, loosen a tolerance, clip/mask numerics to fake a gate
  (a justified positivity floor is allowed ONLY if documented + physically standard, e.g. E ≥ E_min).
- **Truth-tiers-first**: trust the closure because it passes positivity / budget-closure /
  channel-spinup / differentiability — NOT because it matches Veros. Oracle is the LAST, lowest-trust.
- **Minimal diffs, incremental commits**; never leave the step/state broken at a commit boundary.
- **Locate yourself each iteration**: read the progress log, `git log --oneline -15`, run the gates.

## Build order (the gates are ordered so each builds on the prior)
The closure is built + verified as a PURE module first (E1-E5), THEN threaded into the state/step
(E6, the cross-cutting risk), THEN integrated/validated (E7-E9). This keeps the hot loop unbroken
until the closure is trusted.

## Acceptance gates (ALL must be honestly green; E9 informational)
- **E1 — EKE closure module (pure).** New `ocean/physics/lateral_mixing/eke.py`:
  `EKEConfig` NamedTuple (`c_k`, `c_eps`, `l_min`, iso-diffusivity, advection scheme; defaults =
  Veros ACC `eke_c_k=0.4`, `eke_c_eps=0.5`, `eke_lmin=100.0`) + a pure tendency function
  (production from the GM eddy buoyancy flux, dissipation `c_eps·E^{3/2}/L`, mixing length L≥l_min)
  + the `kappa_GM = c_k·L·√E` diagnosis. Direct unit tests: output shapes, all-finite,
  dissipation ≤ 0 and ∝ E^{3/2}, kappa_GM monotone in E, kappa_GM ≥ 0.
- **E2 — dispatch.** GM/Redi gains a prognostic-`kappa_GM` mode (vs constant/visbeck), selected by
  config (e.g. `GMRediConfig.kappa_gm_scheme="eke"` or an `eke` config presence) with
  `ValueError` on unknown literal (single VALID set). EKEConfig validated fail-fast.
- **E3 — positivity.** Under the closure + the (positivity-preserving) advection/diffusion of E,
  `E ≥ 0` is preserved over repeated application (or a documented `E_min` floor, standard practice).
  Direct test: start E ≥ 0, apply N steps of the E update, assert min(E) ≥ 0 (or ≥ E_min).
- **E4 — budget closure (tier 0).** Advection + isopycnal diffusion of E CONSERVE ∫E in a closed/
  periodic domain to machine-eps (only production − dissipation change ∫E). Direct test (reuse the
  flux-telescoping pattern from the flux-form build).
- **E5 — differentiability (tier 1).** `jax.grad` through the EKE tendency + the kappa_GM coupling
  is finite and nonzero.
- **E6 — state threading + zero-behaviour-when-OFF.** Add `eke` to `LatLonCGridOceanState`
  (+ init `rest_state_*`, the step loop integrating E alongside T/S, `SegmentCarry`, restart I/O,
  channel-packing). DEFAULT OFF (no EKE config) ⇒ **bit-identical existing behaviour**: the
  baroclinic-decomposition gate + the ocean unit suite stay green; the per-step ref loops are
  updated. Gate: existing gates green with EKE off + a test that EKE-on integrates an `eke` field.
- **E7 — idealized channel (tier 2).** A short baroclinic-channel (Eady/ACC-like) integration with
  EKE on stays finite, `E` spins up to a BOUNDED level (does not blow up / go negative), and
  `kappa_GM` responds to `E` (varies in space/time, ≥ 0). Compare it runs vs EKE-off (both stable).
- **E8 — regression lock.** Add an EKE-active case to a committed golden (decomposition gate or a
  dedicated bit-identical regression) so the EKE path is locked going forward.
- **E9 — oracle confirmation (informational).** ACC tier-2 comparison with EKE active vs Veros's
  `eke` field + `kappa_GM` (developed-flow Veros run). Record whether legoESM's E / kappa_GM
  correlate with Veros in the §8 ledger; if validated, ADOPT EKE in `build_acc_model_config`
  (Veros ACC is `enable_eke=True`). Not pass/fail.

## Verify-before-commit
Run the narrowest relevant gate(s) after each edit + the baroclinic-decomposition bit-identical
gate for any orchestrator/state/step touch. Full pytest suite is NOT runnable here (exit-143) —
use targeted gates + zero-behaviour-by-construction (EKE off ⇒ identical) + the truth-tier tests;
record what ran. No new failure introduced (verify pre-existing failures are pre-existing).

## Completion / stop
Emit `<promise>EKEDONE</promise>` ONLY when E1-E8 are honestly green and committed (E9 documented),
the closure + state threading + tests are committed with explicit paths, and the progress log
records the final state. Emit a clear BLOCKED note instead if a gate cannot be met honestly after
≥3 genuine iterations (e.g. positivity or coupling stability needs a design decision) — write the
blocker + options to the log; do not relax a gate.
