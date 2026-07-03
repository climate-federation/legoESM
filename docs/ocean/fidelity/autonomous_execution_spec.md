# Autonomous Execution Spec — legoESM Ocean Verification Queue (Ralph)

> Re-read this file IN FULL every iteration. It is the contract for the autonomous run.
> Branch: `matching_Veros_oracle`. Companion: `oracle_recipe_strategy.md` (the why),
> `phase_g_recipe_fidelity_plan.md`, `phase_g_veros_recipe_audit.md`. Veros is installed at
> `/home/dbalwada/veros` (in the venv) so the tier-3 comparison can run locally.

## Mission
Execute the full oracle-recipe verification queue (Q1–Q8 below) to completion, autonomously.
Come back to the user ONLY when every task's acceptance gate is honestly green, or when
genuinely blocked. Scope chosen by the user: **everything in the strategy doc.**

## Operating rules — READ EVERY ITERATION
- **CLAUDE.md is absolute**: verification-first, constant discipline (no literal physical
  constants), no duplicate numerics, dispatch raises `ValueError` on unknown literals,
  import discipline, every new `.py` gets a direct test, `JAX_ENABLE_X64=1` for scientific
  validation.
- **ANTI-GAMING (non-negotiable).** NEVER weaken/skip/`xfail` a test, loosen a tolerance,
  mask/clip/coerce numerics, narrow a comparison region, or delete a check to manufacture a
  green gate. "Done" = a *physically-justified change that passes the MEANINGFUL gate*. If a
  gate cannot be met honestly, that is an **escalation**, not a license to relax it.
- **Minimal local diffs.** No unrelated refactors inside a task. No new abstractions beyond
  what the task needs.
- **Commit per verified task.** Explicit paths only — never `git add -A`/`.`; never stage
  anything under `docs/references/`. End commit messages with the CLAUDE.md
  `Co-Authored-By` line.
- **Verify before commit**: run the narrowest relevant test(s) AND a no-regression check on
  the existing ocean suite for files you touched. Record the result.
- **Locate yourself each iteration**: read the PROGRESS LOG, then `git log --oneline -20`,
  then run the relevant tests. Work the lowest-numbered UNFINISHED task. Do not redo
  finished, committed, gate-green tasks.

## Progress log — maintain `docs/ocean/fidelity/autonomous_progress.md`
One dated entry per task attempt: task id, what changed, the exact gate command + result,
commit hash, and every micro-decision made (so it is auditable). This is the loop's memory.

## Escalation protocol — HIGH BAR (the user does not want micro-decisions)
- **Micro-decision** (naming, field grouping, a default value, file placement, which of two
  equally-valid impls): pick the CLAUDE.md/doctrine-consistent default, LOG it, proceed.
  Do NOT stop.
- **STOP + emit a BLOCKED promise** only for: a genuine fork the strategy doc does not
  resolve; a numerics/physics tradeoff with no defensible default; or ≥3 honest iterations
  failing to meet a gate. On stop, write a clear blocker summary + options to the log.

---

## The queue (in order). Each task: DO, then GATE, then NON-GOALS/ESCALATE.

### Q1 — Phase G density residual: diagnose-then-fix (prime suspect: wall-row padding)
Evidence (oracle-free probes, recorded in strategy §7 + phase-g memory): cumsum/flip order →
2.5e-15 kg/m³ (exonerated); actual-ρ vs geometric-depth → ≈0.05 kg/m³ (too small); EOS at
T=0,S=0 (zero-padded wall rows from `create_regional_latlon_grid`) → −27 kg/m³/cell.
**DO (diagnose FIRST, do not assume):**
1. Determine whether the *interior* region metric in `build_region_masks` actually includes
   any wall / T=0,S=0 / wall-adjacent cells. If the interior L2=5.1 came from wall cells
   leaking into the "interior" mask → that is the bug: fix the mask AND keep the EOS off
   T=0,S=0 (mask wall rows in `land_mask` via `replace_land_mask`/`land_mask_override`, or
   interpolate them from adjacent Veros cells).
2. If interior was already clean of walls (so walls do NOT explain 5.1), pivot to cause (a)
   time-level alignment: test `vs.rho`/`vs.temp` τ vs τ±1 in the bridge for a consistent
   (T,S,ρ) triple.
**GATE:** (a) a test asserts the bridged ACC state has ZERO wet cells with T==0 & S==0;
(b) `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python scripts/validate/ocean_fidelity/compare_tendencies_acc.py --runlen-s 864000`
→ interior density L2 < 0.1 kg/m³ AND pattern-corr > 0.999; (c) full ocean unit suite green.
**NON-GOALS:** bitwise; driving the genuine ρ-vs-depth (~0.05) or float diffs to zero.
**ESCALATE:** if after an honest wall fix + the single time-level test the residual is still
unexplained (> 0.1 and no identifiable bug), STOP and report the updated partition. Do NOT
chase indefinitely (doctrine §3 rule G).

### Q2 — Per-process tendency comparison (face-stagger interpolation)
**DO:** interpolate legoESM face-staggered momentum tendencies to cell centres (or pad Veros
u/v faces to the n_lon+1 wrap convention) so `du_*, dv_*, dtemp_*, dsalt_*` compare
per-process — replace the "shape mismatch / aggregate; deferred" rows.
**GATE:** the driver emits per-process metrics for all momentum + tracer processes; interior
pattern-corr > 0.95 AND sign-match > 0.95 per *active* process. A process that cannot reach
this due to a DOCUMENTED structural/discretization delta (not a bug) is logged in the §8
ledger with explicit justification and accepted (NON-GOAL to force to zero).

### Q3 — Register fidelity modules (test discovery)
**DO:** register `recipe_constants`, `veros_acc_recipe`, `veros_state_bridge`,
`tendency_probe` in `ocean/fidelity/__init__.py __all__` + lazy loader.
**GATE:** importable; a direct construction test for `build_acc_recipe()` is green.

### Q4 — `ConstantsConfig` (strategy §6, G-C1…G-C5)
**DO, in order, each step ending green:** G-C1 define `ConstantsConfig` (5 base constants:
g, Omega, R_earth, rho_0, c_sw; defaults reference `legoesm.constants`) + add to
`LatLonCGridOceanConfig` (+ thread to `OceanPhysicsConfig`) → zero behavior change. G-C2
de-mirror the `ocean/` rho_0/c_sw module mirror bindings (the `_CONSTANT_SHADOWS` targets) +
inline `constants.X` tendency reads in physics. G-C3 thread to `ocean/coupler/*` +
`forcing/*`. G-C4 migrate the recipe to `ConstantsConfig(**VEROS_CONSTANTS)` injection;
delete the `override_constants` monkey-patch. G-C5 audit-guard test (fails on reintroduced
direct reads / mirror bindings).
**GATE per step:** full suite green; G-C1 numerically identical to baseline; G-C5 audit
test red on a deliberately-reintroduced violation, green otherwise. EOS coeffs stay in EOS
configs (NOT in `ConstantsConfig`). The cross-boundary `legoesm.coupler.bulk_flux.G` leak is
out of ocean scope — wrap at the ocean seam or LOG as deferred.

### Q5 — CI clarity guard (deterministic)
**DO:** `tests/ocean/unit/test_clarity_guards.py` (ast-based): function-LOC ceiling (~400,
allow-list seeded with current offenders so green on day 1), two-source-of-truth param
detector, deprecated-but-live config detector.
**GATE:** green on the current tree (allow-list baseline); confirm ONCE it goes red on a
deliberately-introduced violation, then revert the violation.

### Q6 — Equivariance tier expansion (strategy §4/§7)
**DO:** add the remaining φ-transforms to `tests/ocean/unit/test_equivariance.py`: bridge
round-trip (`transpose ∘ reverse_z ∘ strip_halo` preserves the physical field — subsumes
Q1's no-T=0/S=0 assertion), axis-transpose equivariance, halo-width invariance. Generalize
the `test_tripole_fold.py` round-trip helpers — do not duplicate.
**GATE:** all new tests green; the bridge round-trip test passes (post-Q1).

### Q7 — Config regrouping + footguns
**DO:** group/section the 45-field `LatLonCGridOceanConfig` with section comments (+ a class
docstring with a minimal-vs-production example). Resolve the 3 footguns: single source of
truth for `A_h` (dynamics vs physics config); remove/clearly-gate the deprecated
physics-level bottom drag; validate `eos`/`eos_linear` coupling (raise `ValueError` per
dispatch discipline). Do not change numerics.
**GATE:** full suite green; the clarity-guard two-source check passes for `A_h`; a dispatch
test covers the `eos`/`eos_linear` validation.

### Q8 — Decompose `latlon_cgrid_ocean_baroclinic_tendencies` (RISKIEST — LAST)
**DO:** extract the 1299-LOC function into ~12 named pure substage functions (operators
unchanged; SAME numerics — pure extraction, same op order). Then remove its allow-list entry
from the Q5 clarity guard. Do the same for `_step_impl` if time permits.
**GATE (stricter, numerics-preserving):** a regression test proves the decomposed tendency is
BIT-IDENTICAL (or within 1e-12 rel in x64) to the pre-decomposition output on a frozen ACC
state; full suite green; the clarity guard's LOC ceiling now passes WITHOUT the
`baroclinic_tendencies` allow-list entry.
**ESCALATE:** if bit-identity cannot be achieved (unavoidable reassociation), STOP and
report — do NOT accept a silent numeric change to claim the gate.

---

## Completion / stop
There is ONE stop tag: `<promise>RALPHDONE</promise>`. Emit it (and nothing should follow that
needs doing) in EITHER of these two cases, after writing a final summary to the progress log:

- **COMPLETE** — every gate Q1–Q8 is honestly green AND the relevant existing suites pass
  (ocean unit suite always; atm/ocean matrix only if touched code warrants it). State
  "STATUS: COMPLETE" in your message and the log.
- **BLOCKED** — a genuine fork/blocker the spec does not resolve, or ≥3 honest iterations
  failing a gate. State "STATUS: BLOCKED — <one-line reason> — <options>" in your message and
  the log.

Do not emit the stop tag for any other reason. Micro-decisions are made with a logged default
and the loop continues.
