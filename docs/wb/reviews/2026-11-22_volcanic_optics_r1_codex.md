Reading additional input from stdin...
OpenAI Codex v0.166.0
--------
workdir: /glade/derecho/scratch/pg2328/wt_volc
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: medium
reasoning summaries: none
session id: 01a2055f-2da9-7f63-b31f-9c933165ad3d
--------
user
# Adversarial review r1: volcanic SW aerosol with the CMIP6 file's own per-band optics
READ-ONLY review of the uncommitted diff in /glade/derecho/scratch/pg2328/wt_volc (git diff vs HEAD,
plus new files tests/unit/test_volcanic_sw_optics.py, scripts/validate/probe_volcanic_sw_optics.py). Do not edit.

What changed: volcanic stratospheric SW aerosol used to be a solar-weighted GRAY extinction added
into `aerosol_od` and solved with the grey tropospheric SSA 0.93 / g 0.70. Now
`_load_volcanic_sw_optics_profile` (forcing/external.py) builds per-file-layer, per-band moments
(tau, tau*omega, tau*omega*g) from ext_sun/omega_sun/g_sun, bands sorted ascending wavenumber and
validated 1:1 against RRTMGP SW band limits (one documented 2600/2680 cm-1 edge difference);
`aerosol_layer_fields` (forcing/surface_utils.py) remaps each moment with the existing conservative
log-p overlap remap x tropopause weight to (ncol,nlev,3,14) `aerosol_sw_volc`, keeping volcanic OUT
of `aerosol_od`; new optional field threaded model_driver -> SegmentForcing/pack_forcing ->
physics_pipeline (step_unified kwarg captured by closure, compute_radiation_core, rrtmgp radiation_fn;
gray/zero radiation raise) and integration.py (forcing dict key `aerosol_sw_volc`) -> RRTMGP.solve_columns
(`aerosol_volcanic_moments_sw`, halos, overhead layer, subcolumn expansion key, column chunking)
-> two_stream `_mix_sw_aerosol_and_clip` adds the g-point's band moments to tau / tau*ssa / tau*ssa*g.
Grey ext_sun-only SW profile path deleted (a file with ext_sun but no omega/g now raises).
Voted detail choices: Q1 sorted 1:1 + edge validation (author A, codex C, GLM B); Q2 raise (codex, GLM);
Q3 float32 cache; Q4 None when absent; Q5 non-RRTMGP lanes raise; Q6 non-finite raises (moot: 0 NaN in
production file); Q7 rtol 1e-10 equivalence, closure |res| <= 1e-10*incident + 1e-10.

Check hard: (1) band mapping/order correctness incl. g_point_to_bnd; (2) every radiation call path that
receives aerosol_od also receives the new field (MPAS, cube, lat-lon, spectral, compiled, per-step,
operator-split SPMD, chunked, subcolumns, MC optics, clear-sky) -- name any that silently drop it;
(3) energy: SW budget closure, no double counting of volcanic in aerosol_od, CCN unchanged;
(4) AD/JIT: pytree stability (None vs array across segments), dtype under compute_fp32, gradient;
(5) tests vacuous? (6) memory of (ncol,nsrc=70,3,14) host interpolation at res6; (7) any silent
behaviour change for runs WITHOUT a volcanic file or with legacy column `aod` files.
Reply: numbered findings with severity (BLOCKER/MAJOR/MINOR), file:line, failure scenario; then VERDICT.
codex
I’ll read the repository rules, trace the new field through the radiation call paths, and check the tests. I won’t edit any files.
exec
/usr/bin/bash -lc 'cat CLAUDE.md' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
# legoESM Claude Memory

## RULE 0 — CONSERVATION OF MASS, ENERGY AND MOMENTUM. THE STRICTEST RULE.
User directive 2026-10-11: *"Mass, energy and momentum conservations should be
the strictest rules."* Outranks every other rule in this file, including speed,
brevity and convergence.

- **No change may create or destroy mass, energy or momentum** — no cap, clip,
  floor, limiter, fallback or "hold" that removes or adds a quantity without
  routing it to an explicit sink/source in the same budget. A capped water
  amount goes somewhere (dew, runoff, store); its latent heat goes somewhere.
- **Every physics/numerics change states its budget and proves it closes**:
  name the conserved quantities it touches, and add a test asserting
  `in - out - d(storage) = 0` to round-off (`JAX_ENABLE_X64=1`), shown to FAIL
  when the fix is reverted.
- **Exchanged fluxes carry the same value and sign at both ends of a coupling.**
- **A known non-conserving path (fallback, held column, clamp) is a defect**:
  report it with its measured size the moment it is found, never absorb it.
- Speed or convergence fixes that touch a solver must show the conserved
  totals are unchanged (to round-off) before and after.

## RULE 1 — CONCISE AND CLEAR. THE DEFAULT, EVERY REPLY.
User directive 2026-08-21: *"Be concise and clear — make this a default."*
Eleventh ask. Default, not mode.

**2026-09-08, NAMED AT OPUS 5 (and every Mythos/Fable-class model):
*"Force opus 5 to be concise and clear as default."*** Twelfth ask. The
long-session drift is the failure mode: replies start terse and grow back
into paragraphs after an hour of tool calls. So the cap is not a mood, it
is checked EVERY reply, and re-read whenever a session runs long.

GATE (RULE 2 demands one, so here it is, mechanical, self-applied before
send): count the rendered lines of the reply. **>5 lines, or any table
the user did not ask to compare, or any header in a reply under ~500
words → DELETE and rewrite.** Verdict first line; then at most 3 short
bullets; then stop. A status report on background jobs is ONE line per
job, no preamble. If it will not fit, the content wanted a table — keep
the table, delete the prose.

- **Verdict first line.** Then max 3 short bullets. Stop.
- **~60 words.** Longer only if report/walkthrough asked.
- **Plain words.** No file:line, function names, config keys, job ids, PR
  numbers unless asked — those go in commit. Say THING, not SYMBOL.
- **One idea per line.** Clause explaining clause = second bullet, or cut.
- **No headers, no tables, no bold-label blocks** in normal reply. Read as
  structure, same length. Table only if >=3 things on >=2 axes AND user must
  compare.
- **Own error in one sentence, at top.** Confession not exemption from
  brevity — narrative belong in commit, which already have it.
- **Decision needed = one question, numbered options, your pick named.**
- Cut any line user not act on. Tool calls already visible.

Full rationale + ten prior callouts: `## Response Style` at end of file. That
section disagree with this one → this one win.

## RULE 2 — RULES DO NOT ENFORCE THEMSELVES. CHECK THE GATE, NOT THE TEXT.
User 2026-08-21: *"how do we ensure the rules are not being violated?"* Honest
answer: MORE TEXT DOES NOT. Both rules broken in the session that prompted this
were ALREADY in this file. Adding emphasis to a rule that is being violated
changes nothing; adding a MECHANICAL GATE does.

**So: before claiming a rule was followed, name the gate that checked it.**
"I was careful" is not compliance. If no gate exists, say NO GATE out loud in
the status — do not let silence imply a check happened.

| rule | gate | where |
|---|---|---|
| no banned constants / saturation re-impl | PreToolUse hook + CI ratchet | `.claude/hooks/check_banned_literals.py`, `tests/test_no_hardcoded_constants.py` |
| codex review on uncommitted numerics | Stop hook, once/session, ADVISORY | `.claude/hooks/require_review_artifact.py` |
| scheme dispatch raise on unknown | CI ratchet, grow-only | `tests/test_dispatch_hardening.py` |
| every scheme field validated | CI ratchet | `tests/test_validate_strict_coverage.py` |
| physics contract per module | CI ratchet, shrink-only TODO | `tests/test_physics_contracts.py` |
| tunable reachable from a driver | CI audit, shrink-only baseline | `tests/unit/test_params_reachability_audit.py` |
| one parameter, one legal range | CI gate, shrink-only | `tests/unit/test_param_bounds_agree.py` |
| no private cross-module imports | CI ratchet, allowlist EMPTY | `tests/test_no_private_cross_imports.py` |
| no inline physics coefficients | CI ratchet + baseline | `tests/test_no_inline_physics_coeffs.py` |

**NO GATE EXISTS for these — they are honour-system, so state compliance
explicitly every time:**
- **DUAL review (codex AND GLM).** The Stop hook only sees UNCOMMITTED work and
  only mentions codex. A merge commits instantly, so it never fires on one —
  which is exactly how a 14-conflict merge shipped unreviewed on 2026-08-20.
- **NO UNASKED CHOICES / no hidden config choices.** Nothing can detect a
  plausible decision made silently.
- **Controlled comparison** (one variable, eval protocol held fixed).
- **Instrument validated before its number is quoted.**
- **Non-vacuity**: every new test shown to FAIL when its fix is reverted.

**When a gate does not exist, build one before adding more prose.** Two of the
rows above were written the day the rule they enforce was broken.

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` / `batch_allreduce_mpi(op="sum")` / `global_sum_if_distributed` backward = allreduce SUM of the cotangent (correct for sums reused inside the computation; #1814). The reduction that PRODUCES the loss every rank evaluates and differentiates identically (normalised or not, e.g. a global mean) takes `global_sum_mpi(..., final_loss=True)` (identity backward). Forward mode (jvp) on both paths = allreduce SUM of the tangent (the default is a linear primitive whose transpose is also allreduce SUM). MPI halo: `_sendrecv_vjp` custom_vjp. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.
- **NO INERT PARAMETERS EVER (STRICT, user 2026-08-17).** Every leaf of trainable pytree must carry loss gradient. Mode-inactive params frozen OUT of trainable set (`_inactive_keys` pattern); first training step gates rest via `assert_no_inert` (`train_land_params_era5.py`) — zero-gradient leaf aborts run. Any new trainer/calibrator adopt both pieces; param "wired in but off" without freeze-out = defect.

## RULE 4 — SEARCH BEFORE YOU BUILD, AND PONYTAIL FULL IS THE DEFAULT.
User directive 2026-08-23, after a nearest-neighbour-on-the-sphere regridder
was written into a data script while the coupler ALREADY had a grid-remap
module (`coupler/grid_remap.py`) and core had another (`grids/regridding.py`):
*"always check if anything already exists before implementing"*.

- **Before writing ANY new function, class, script, or numerical method: grep
  the repo for an existing one.** Name in the status what was searched and what
  was found — "searched X, found nothing" is a claim and gets the grep pasted.
  This extends the existing pre-impl-search rule from numerics to EVERYTHING,
  and it applies to scripts and probes, not just packages.
- Found something close? EXTEND it or move it to the right home (generic
  methods live in the shared package, e.g. cross-grid machinery in the
  coupler) — never a second implementation "for now".
- NO GATE exists for this; it is honour-system, so compliance is stated
  explicitly every time (RULE 2).
- **Ponytail FULL is the default, every session** (user 2026-08-20 and again
  2026-08-23): laziest working solution, reuse over rebuild, delete before
  add, shortest diff that is actually correct. Not a mode to be asked for.

## RULE 5 — DIFFERENTIABILITY IS NEVER TRADED FOR ORACLE FIDELITY.
User directive 2026-10-20, STRICT, on the Kessler port (the smooth saturation
ramp misses the reference by up to 100 % and lets cloud water go slightly
negative; the reference's exact step would match it): *"keep the current
smoothing - differentiability cannot be traded off!"*

- **The default is always the differentiable form.** A switch in the
  reference whose branch depends on TRACED model state (a saturation step,
  a threshold `max`/`where` on a prognostic, a clip, a cutoff on the
  column's traced pressure) is ported as a smooth
  function whose sharpness/width is a config field; the hard limit is
  reachable only by setting that field, never as the default.
- **Out of scope (state-independent):** selections that do not depend on
  traced state and keep JAX gradients intact — fixed land/sea masks, domain boundaries, index
  picks, static scheme dispatch. Smoothing those changes the domain or
  breaks conservation; port them exactly.
- **Never propose moving a default to the hard limit to close an oracle gap.**
  Not as an ASK, not as a "pick". The gap is recorded instead.
- **Record the gap as a KNOWN DIFFERENCE** where the code lives (docstring)
  and in the PR body: what the reference does, what the port does, the
  measured size at the production setting, which field recovers the
  reference, and any invariant the smoothing breaks (e.g. how negative
  cloud water goes). Never loosen the tolerance.
- **Every such switch gets three tests:** one AT the hard limit,
  matching the reference to the oracle tolerance; one AT the smooth default
  asserting the recorded gap PER CASE (each fixture column's measured
  mismatch, stored with the test, to 1e-6 relative with a 1e-12 absolute
  floor); and a gradient test at the default (finite, non-zero derivative
  through the switch, `jax.test_util.check_grads`). Never a blanket xfail,
  never one worst-case bound. A change to the smooth form OR its default
  width re-measures and updates the stored gaps in the same PR, with the
  reason.
- RULE 3 is waived only when all four hold: the smooth default is kept for
  differentiability, the hard limit is reachable via its field and matches
  the reference, the gap is recorded as above, and all three tests exist and pass.
  Missing any one, RULE 3 applies.
- Applies to every oracle translation (CAM, IFS, WRF, ESCOMP, NEMO, FV3, ...).

## RULE 3 — A FIX WHOSE DEFAULT KEEPS THE BUG IS NOT A FIX. IT IS A BUG WITH A KNOB.
User directive 2026-08-22, on finding non-orographic gravity waves still being
launched at the SURFACE months after that exact defect was "fixed": *"Why does
gravity-wave drag now launch at the surface? This is absurd and we should never
ever make arbitrary choices like this without my supervision and agreement."*

WHAT ACTUALLY HAPPENED, because the shape repeats: #1394 measured the defect
(55% of the wave's momentum deposited below 1 km), added `hines_launch_p` as the
cure, and left its default at `0.0` — which the field's own comment documents as
"unset = legacy SURFACE launch". One campaign config set 70000. The production
config never did. Every production run since has carried the defect the fix was
written to remove, and the tropical ocean lost a third of its evaporation.
NOBODY CHOSE THIS. That is precisely why it survived.

STRICT, MECHANICAL:
1. **Landing a fix behind a default that preserves the old behaviour is
   FORBIDDEN unless the user is asked, in that PR, in one line: "default stays
   broken (X) or moves to the fixed value (Y)?"** No exceptions for "callers can
   override", "it is only a default", or ABI/positional-tuple preservation —
   those explain the mechanism, never the choice.
2. **A knob whose default is documented as "unset", "legacy", "off", or
   "0.0 = old behaviour" is a DEFECT REPORT, not a feature.** Grep for that
   wording before any campaign; each hit is an open bug in every run that does
   not set it.
3. **The fix's PR must edit the PRODUCTION config, not only a campaign deck.**
   The existing rule "new knob ships with a committed config that selects it"
   was satisfied by a side deck here and still shipped the bug — so the
   production config is now named explicitly.
4. **Before any production campaign, diff the run's RESOLVED config against the
   best previous run's resolved config and read every differing row out loud.**
   Not the deck, the resolved config. This regression was one row in such a
   diff and would have been caught before the GPU-hours, not after.
5. A default that disagrees with the best known configuration is reported as a
   FINDING the moment it is noticed — never silently carried into a run.

## ABSOLUTE RULE — NO UNASKED CHOICES. ASK FIRST, EVERY TIME.
User directive 2026-08-21, stated ABSOLUTE and repeated: *"Every choice such
as changing options should be checked with me before being changed"* and *"Do
not make random choices like this. This is a major error from Opus 5 in
particular."* Model near operationalization: unasked choice make every
downstream number non-comparable to what came before, and user cannot see it
inside diff full of other work.

THE FAILURE MODE, NAMED: fill gap in task with plausible decision instead of
question. Decision usually defensible — exactly why it go unchallenged and
unnoticed. Defensible choice made silently still unasked choice, and that the
error prohibited here — not bad taste, not wrong values. **Plausibility is not
authorisation.**

**2026-08-21 EXTENSION — HIDDEN CHOICES IN SIMULATION CONFIGURATION.** User:
*"add a strict rule not to take hidden choices such as for configurations of
simulations. We now have many bugs related to incorrect choices."* The rule
above bans choices YOU make. This bans choices the CODE makes for you and
nobody records. Same bug, different author.

A HIDDEN CHOICE is any place a scientific decision get made without appearing
in the run's configuration:
- **a default that disagrees with the run.** `g` defaulted to library constant
  while card pinned another (#1627); density fed into that pressure built on
  defaults too. Fix at last call site give DIFFERENTLY wrong answer.
- **a fallback that substitutes physics.** `use_multilayer_land: false` on MPAS
  ran NO land model — land T was nearest ocean SST minus lapse rate. Three
  waves of convection tuning against it.
- **a default that sits on domain geometry.** Lock-exchange front defaulted to
  prime meridian; channel spanned 0–0.576°, so front sat on WESTERN WALL. 1
  cold column of 66. Gates certified uniform box.
- **two bounds for one parameter.** `cloud_q_c_diagnostic` spec say 5e-5–1.5e-3,
  driver validate 1e-6–1e-3. Proposed value legal by one, illegal by other.
- **a lever nothing selects.** Neural drag flag no config set. Persistent-D
  behind env var, default off, while it is the leading fix for #1028.
  `cape_relaxation_sink` declared, read by nothing, A/B "in flight" against it.

MECHANICAL RULES, not judgement calls:
1. **Every scientific choice appear in the run's config, not in a default.**
   Building a run: name grid, scheme, resolution, timestep, physics switches
   EXPLICITLY even where default already correct. A default is not a record.
2. **A defaulted physical constant is a defect.** If caller can know run value,
   pass it. If it cannot, that is tracked debt with signature change named —
   never silent.
3. **Before running: enumerate what the config RESOLVED to, from resolved
   config and lane's code path** — not deck's comments. Prose is pointer, never
   citable fact. No prognostic state → say so out loud; usually that is finding.
4. **`false` on a component switch does not mean "simpler version".** May mean
   NOTHING. Check what lane fall back to.
5. **New knob ship with a committed config that selects it**, or it is not
   shipped. Same PR.
6. **Two declared ranges for one parameter = neither is the range.** Fix before
   using either.

STOP AND ASK before changing or selecting any of:
- a default value of a config field, CLI flag, or scheme selection
- which scheme/parameterization a driver or harness selects
- a tunable parameter's value, bounds, or tier
- what a case, deck, or experiment is forced with, and which cases exist at all
- a threshold, cadence, resolution, timestep, or analysis window
- which quantities are scored, and how they are weighted
- anything that changes model STATE a run carries — adding or removing a
  prognostic variable or tracer counts, EVEN when the added field is zero
- where data comes from (source, version, pinning) and whether a file is
  fetched, vendored, or generated
- whether a previously-tolerated condition becomes a hard error

Ask ONE line per item: current value -> proposed value, and why. Batch them; no
drip-feed. Work cannot proceed without answer → say so and stop. Never proceed
on assumption then flag after.

NO EXEMPTION FOR ANY OF THESE, all used as excuses:
- "the change was needed to make the requested thing work"
- "the old value was wrong / a bug"
- "a reviewer recommended it"
- "it is obviously an improvement"
- "it is only a default, callers can override"
- "it only adds a field, and the field is zero"

Fix bug; then ASK before default move. Fix and choice entangled → land fix,
leave choice pending.

NOT COVERED, proceed normally: what user asked in that message; pure additions
changing no existing behaviour; tests; comments; docs; error text.

**2026-10-16 EXTENSION — NO SIDE IMPLEMENTATION DECISIONS (STRICT).** User:
*"No side implementation decisions as strict Claude.md rule."* Approving a
change does NOT approve the decisions made while implementing it.

A SIDE DECISION = any value, behaviour, or method that changes what the code
computes, accepts, rejects, or reads, and is NOT dictated by the user's answer,
the oracle being ported, or existing repo code reused unchanged. Each one gets
ASKED (one line, current -> proposed, batched) before the code embodying it;
one found mid-implementation → stop and ask then. Classes, not exhaustive:
- tolerances and acceptance thresholds (incl. checks on input files, and the
  tolerance an oracle-match test certifies)
- guards, floors, caps, clamp vs extrapolate, and their fp32-vs-fp64 behaviour
- edge cases: zero/empty/missing input, absent file, old checkpoint
- mappings between representations (bands, grids, levels, units)
- algorithm and order: interpolation, aggregation, update/operation order
- where data or files live; names and defaults of user-facing switches
- which lanes/grids get it; error vs warning vs fallback there

PRECEDENCE: this OVERRIDES the "NOT COVERED" line above for pure additions and
tests whenever they carry a side decision. Still exempt: error MESSAGE wording,
comments, docs. Ponytail/minimal-diff governs how much code, never licenses
picking a value. "Formatting only", "minor", "internal", "implementation
detail", "numerically negligible" are NOT exemptions. Subagent briefs must
state this rule. Any item on an end-of-task UNASKED list is a violation of this
rule (target: empty), not a disclosure that discharges it. FAILURE 2026-10-16:
radiation stage 1 shipped a band-list check tolerance and an fp32 tiny-path
guard behaviour, both labelled "formatting only", neither asked.

**2026-11-03 — WHO ANSWERS: DETAIL CHOICES GO TO A VOTE, NOT THE USER.** User:
*"For such detailed question do not ask me but check with [GLM] and codex and
use the majority of votes. Just ask me big picture / important questions such
as on design or main directions."* Every side decision above is still never
made silently, but a DETAIL choice (tolerance, guard, edge case, test value,
cache key, refusal scope, error-vs-fallback) is put to GLM
(`mcp__zai__ask_glm`) and codex, batched in one prompt each, plus the author's
own vote; majority of the three decides. Record each choice with its three
votes in the commit/PR body. The USER is asked only about design, main
direction, scope, and anything they named as their call. Precedence: this
replaces "ask the user" for detail choices everywhere in this section and in
the STOP-AND-ASK list; for design/direction the user rule stands unchanged.

MECHANICAL CHECK, end of every task: print list of choices made, each one line,
marked ASKED (user), VOTED (GLM/codex/author majority, votes shown) or UNASKED. Empty UNASKED list = target. Anything on UNASKED list
offered for revert in same message. Worked example, five from 2026-08-21 that
prompted this rule: routing each case's droplet number into single-column arm;
allocating zero graupel field in every column running microphysics; which
campaign cases got registered and using ARM cumulus case as stand-in for another
campaign's regime; pinning forcing downloads to fixed versions and checksums;
turning missing variable name into hard error. Every one defensible. None asked.
That the error.

**2026-11-03 REFINEMENT — DETAIL DECISIONS GO TO A VOTE, NOT THE USER.** User,
repeated three times in one day: *"For detailed questions do not ask me but
check with him [GLM] and codex and use the majority of votes. Just ask me big
picture / important questions such as on design or main directions."*
- **Detail side decision** (names, tolerances, guards, edge cases, file
  locations, minor scope trims, which diagnostic next): author's pick + GLM vote
  + codex vote, majority of three wins. Never silent: record each voter's pick
  in the commit, and mark the item VOTED on the end-of-task list (VOTED counts
  as discharged, like ASKED).
- **Still ASK the user:** model design, main direction, production defaults and
  config changes (RULE 3), which experiments/compute to launch, data source,
  what is trained or scored, anything changing comparability of results.
- Unsure which class → big picture → ask.
- Subagent briefs must state this rule.

## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
User directive 2026-08-21. Any time two runs / configs / templates / lanes are
compared, give a table: one row per field that DIFFERS, one column per
configuration, named in the header. Omit fields that agree — it is a diff, not
an inventory. Prose comparisons hide the one line that matters; the land
coupling week was lost to a surface-scheme mismatch a five-row table exposes at
a glance. This is also the required answer to "why did A work and B not".
If the table has more than one row, no single output difference is attributable
yet — say so.

## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
User directive 2026-08-21, after a week was spent coupling the WRONG land model
and then chasing two wrong causes in a row: *"we focus now on accuracy and we
must check with adversarial reviews our claims, especially when they imply
coding decisions."*

The existing dual-review rule covers DIFFS. This one covers the CLAIM that
justifies the diff, and it comes FIRST:

- **State the claim, then get it reviewed, THEN write code.** If a sentence of
  the form "X causes Y, so I will change Z" is about to become an edit, it goes
  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
  only after the code is written is a claim defended, not tested.
- **Say which parts are MEASURED and which are READ OFF THE CODE.** A scaling
  argument, a mechanism, a "this is the only consumer" — each is a separate
  claim with its own evidence.
- **Quote the number's PROVENANCE next to it.** "62 K leaf temperature" from a
  synthetic sweep that deliberately includes unreachable forcings is not the
  same statement as 62 K in a run. Report the reachable-subset number and the
  full-box number separately, and say which is which. A frightening number with
  no provenance wastes the reader's attention and can send the work sideways.
- **A number a well-posed scheme could never produce is a BUG IN THE
  INSTRUMENT until proven otherwise.** Check the probe before reporting the
  physics.
- **When a working configuration exists, diff against it FIRST.** See
  [[reuse-the-working-path-dont-rebuild]]: the offline land driver was healthy
  the whole time; the coupled lane simply called a different parameter provider.
  The first question on any coupling defect is "what does the working lane do
  differently", not "what could be wrong with the physics".
- **Reviewer disagreement is the signal to MEASURE, not to average.** Name the
  discriminating check and run it; it is usually a grep.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep installed source in site-packages, read actual signature.
No reconstruct from memory. Applies to argument names, argument order,
keyword-only args, return arity.

Symbol under `jax.experimental.*` → assume API changed since training data.
Verify or search. No guessing module paths.

### Report uncertainty explicitly
End any non-trivial code response with `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

Empty block = valid answer. Missing block = not.

My premise wrong — bug misdiagnosed, or thing asked for won't work — say so
before writing code.

### Diagnose before patching
Something fail: state candidate causes and how to discriminate between them,
then test. No going straight to fix. No agreeing with cause I suggested unless
evidence support it.


**Applies to THIS repo's own API too** — large enough that memory unreliable.
FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)` (really
`reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost full probe round-trip.
Worse, `RRTMGPConfig()` default `include_clouds=False`, so offline harness
omitting it return CLEAR-SKY fluxes and EVERY cloud gradient exactly 0.0 —
silently wrong number, not error. Read NamedTuple `_fields` / `_field_defaults`
before constructing config.

User's premise wrong — bug misdiagnosed, or thing asked for won't work — say so
BEFORE writing code.
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model near operational. Every rule below mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming file:line as cause
  require showing that line run in configuration under test: print ENCLOSING
  FUNCTION (`awk` nearest `def` above it), confirm active lane/driver call it.
  FAILURE: blamed positivity clamps at `model_driver.py:10923` for century's
  water source; they live in `_run_per_step` while century run `_run_column`,
  which contain no moisture clamp at all. Fix nearly written for lane the run
  never touch. Same class as reading entry point instead of full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of reference's guards/scope conditions you kept and
  which dropped, with reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so
  column-conserving borrow applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing fake "accelerating dry bias" reported before being caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when feature removed.** `inspect.getsource(X)` assertion where X
  is delegating wrapper pass while proving nothing. FAILURE: asserted against
  `MPASPrimitiveEquationModel.step`; floors are in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read output tail.** Exit code without tool's
  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
  kills and timeouts can surface as success. FAILURE: reported regression suite
  green on exit-0 that was actually `Out Of Memory` mid-run. Quote decisive line
  when claiming suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify in code before
  writing it.** Plausible-sounding reason permanently hide real defect. FAILURE:
  classified `convective_buoyancy_death_memory` as "carried in SegmentCarry" (it
  not — leaf `BechtoldConfig.buoyancy_death_memory` exist and nothing map to
  it), asserted `SBMConfig.precip_efficiency` leaf that not exist, and credited
  `micro_substeps` to consumer that read `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print window next to number.** Differing spans = confound, not result.
  FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr"; matched
  windows gave +13.4. Extends existing controlled-comparison rule to derived
  rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV at
  0.1 kg/m2 over 8 days resolve only ~±4.6 kg/m2/yr — no reporting trend inside
  one quantum. Prefer fp64 from model state (checkpoints) over parsed log lines.
  Same class as throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** fp32 mismatch NOT a
  failure until re-run with x64; and *new* failure not yours until reproduced
  with your change stashed. Do both before reporting regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For controlled pair, COPY pinned checkpoint into each arm dir; never
  use `PREV_CKPT_DIR`-style newest-wins pointer while another run advancing.
  FAILURE (twice): arms exited instantly at "Already at/past target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF THEM.**
  Verify resolved paths in run log, not the flag you passed. FAILURE:
  `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016 SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** Code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" naming a limitation, guard, or
  missing feature MUST be re-verified in CURRENT code at point of use before
  repeated as finding — this repo routinely fix things without updating prose.
  Comment name module imposing guard → OPEN THAT MODULE. FAILURES (2026-07-30,
  three in one day): quoted `_run_column` "turbulent surface fluxes are
  intentionally NOT applied" comment and `AMIP.md` Known #3 to claim MPAS has no
  turbulence — MPAS turbulence path exist (`turbulence/integration.py`, Perot
  edge->cell) and run resolve `turbulence=louis` + `surface_bulk_scheme=coare3`;
  and doubted a cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; real clamp to
  lookup-table range is one call deeper in `rrtmgp/optics/cloud_optics.py`. Same
  class as blaming a line without proving its enclosing function runs.
- **BEFORE ATTRIBUTING A BIAS TO A COMPONENT, PROVE THE COMPONENT EXISTS AND
  RUNS IN THE CONFIG UNDER TEST.** ABSENT component and BADLY-TUNED one give
  SAME symptom, and every tuning arm against absent component return null.
  Enumerate, from RESOLVED config and LANE'S code path (not deck's comments):
  the scheme, its PROGNOSTIC STATE, its inputs. No prognostic state -> say so
  out loud; that usually the finding.
  Corollaries, each earned:
  (a) **A `false` on a component switch does not mean "the simpler version" —
  it may mean NOTHING.** Check what lane actually fall back to.
  (b) **A DECK COMMENT NAMING A FALLBACK IS A POINTER, NOT A FACT** (prose rule,
  applied to configs): grep named symbol in that lane's setup.
  (c) **When N different schemes for component A all fail to move a bias, STOP
  TUNING A.** Limiter upstream, or A's inputs wrong.
  (d) **A parameter wrong at BOTH ENDS of a physical range is a MISSING
  DEPENDENCE, not a mistuning** — thing it should depend on not read.
  FAILURE 2026-08-15, most expensive of campaign: MPAS AMIP deck set
  `use_multilayer_land: false` with comment "this deck runs the slab". There is
  NO slab land on MPAS lane (`slab_land_active` wired only on cube/general
  path), so runs had **no land surface model at all** — land T_sfc was nearest
  OCEAN's prescribed SST minus 6.5 K/km, land evaporation fixed 0.6 of
  saturation, no soil, no water store, no runoff, no stomata, no surface energy
  balance. Tropical deserts evaporated 3.5x observed and rainforest 0.73x FROM
  SAME CONSTANT (corollary d, unnoticed); sensible heat ~25 W/m2 on Sahara and
  Amazon alike. Waves 9-11 spent GPU-hours swapping FIVE convection schemes and
  dozen trigger knobs against tropical-land rain deficit no convection scheme
  could ever fix (corollary c, unnoticed). Interactive soil moisture, bucket
  hydrology, stomatal control are DEFAULTS for AMIP case, not options.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from unweighted `p_s` mean; AREA-WEIGHTED mass was invariant at
  983.493 hPa to 6 digits, i.e. defect did not exist. Habits carried from
  quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT IS
  CITED AS THE CAUSE.** X claimed to drive Y → change X by known factor, check Y
  respond as mechanism predict. FAILURE: proposed "damp-to-rest pumps mass
  convergence" (predict ~linear in sponge coefficient); quartering coefficient
  slowed growth only 1.5x, refuting it — fix would have shipped on false
  mechanism. Label every uncaught claim PLAUSIBLE; honest "cause unknown" is
  cheap, confident wrong cause buy a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below from
defect shipped in ONE session that prompted it, and every one caught by
adversarial reviewer rather than by me — i.e. each avoidable by reading two more
lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace data
  flow before writing it.** FAILURE: wrote "dropping this call now makes the
  test fail" about `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` not consumed until `d_sw5`. Sentence written
  from intent, not from call chain. Before asserting "X is covered by this
  test", name consumer of X and confirm it execute inside test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` give unconditional copy; it is
  copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of code
  with two aliasing behaviours, asserted as safe. Extends *Never infer an API*
  to STRENGTH of a guarantee, not just signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET A
  GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called interim helpers directly. Went
  into COMMIT MESSAGE, which cannot be edited after. Weaken to what you verified
  ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY IS
  NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` mean duogrid⇒bounded, and I wrote converse pair "unreachable" —
  inverting it and mislabelling legitimate regional/nested category as
  impossible. Quote the line next to restatement so direction checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print baseline value the
  perturbation multiplies BEFORE trusting result.** FAILURE: divg-corner probe
  scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0, so control was no-op
  and proved nothing; claim actually rested on predicted==actual identity. Say
  which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts differing in grid
  conventions AND in whether ext bundle existed, while asserting it isolated the
  exchange. Applies to fixtures, not just runs — controlled-comparison rule
  cover `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT, NOT
  ONLY THE TEST YOU WROTE.** Fixture edit = change to every test in the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout
  --`/`git restore`, `cp -f`, `rm -f`; never bare `cp`/`mv` for revert. FAILURE:
  bare `cp` hit interactive overwrite prompt and "revert" never applied — one
  `git status` short of reporting clean tree that was not clean. VERIFY EVERY
  REVERT with `git status --porcelain`.
- **A LOG-SCRAPING GUARD MUST EXCLUDE THE PROMPT/COMMAND IT ECHOES.** FAILURE:
  review wrapper grepped whole log for failure token its own prompt contained,
  so it "retried" every time and never printed exit status. Same class as test
  that cannot fail.
- **BUDGET THE REVIEW LOOP INTO THE WORK.** Adversarial reviewer found real
  defects in EACH of three rounds here; rounds 2 and 3 not ceremony. No
  presenting first-round implementation as finished, no treating "tests pass" as
  terminal condition — round-1 diff passed 401 tests while still containing
  silent fallback and false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
Gates above stop wrong CLAIMS about the model; these stop cheaper, more frequent
error — being wrong about CODE AND DATA IN FRONT OF YOU. Every rule below
mechanical, each cost full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** "Never infer
  an API" rule cover signatures; it also cover array SHAPE, AXIS ORDER, index
  base, padding convention. Before first index of any mesh/state field, print
  its `.shape`. FAILURE: indexed `mesh.cellsOnCell[c, k]` assuming
  `(nCells, 6)`; it is `(6, nCells)`, so probe raised `IndexError` on every
  method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. Real number produced by specific code path →
  GET IT BY CALLING THAT PATH.** Re-deriving lookalike from first principles
  silently answer different question. FAILURE: computed `max_degree` from 1-ring
  `cellsOnEdge` adjacency and reported it as "ppermute round count" — real
  exchange is halo-depth-aware, so proxy said 8 rounds for every method/rank
  count while real schedule said 12→14. Proxy could not even reproduce known
  answer — that the tell: **run proxy against a case whose real value you
  already know BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state which
  quantity it minimizes, confirm that quantity is measured bottleneck.**
  FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS minimize EDGE
  CUT, bottleneck is MAX_DEGREE, and measured they move OPPOSITELY (metis 19
  rounds @64 vs sfc 14, while metis has lower cut). "Better partitioner" is not
  a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
  against raw source first. FAILURE: regex over multi-line series lists attached
  subdiv-9's `9.60/11.48 ms` to `s8` label, and that mislabelled pair went into
  codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing test, grep
  for one already asserting same invariant. FAILURE: added two tests counting
  PCG reduction sites duplicating existing `test_reduction_count_halved`; codex
  had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG, AND
  SAY WHICH.** Assertion you just wrote is claim with no more standing than the
  code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant OFF; resolver is
  `!= "0"`, so `""` means ON — test was wrong, not the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep driver for
  decomposition/sharding the run would need.** FAILURE: scoped 128-GPU coupled
  ladder before finding `CoupledESM` contain three occurrences of "shard", all
  in comments — coupled ocean is replicated, so ladder would have measured
  unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  serial-vs-SPMD gap as "cadence PHASE disagreement"; predicates agree with ZERO
  offset and real cause was serial short-tail fallback. One `grep` of two
  predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* Gates above stop wrong CLAIMS; these stop wrong RUNS.
Simulation launched on hypothesis no measurement can refute = pure waste, and it
also cost WALL-CLOCK of queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job costing
  >1 GPU-hour, write down three things: (a) exact number the run will produce,
  (b) value that CONFIRMS and value that REFUTES, (c) why a cheaper offline/CPU
  test on an EXISTING checkpoint cannot answer it. Cannot fill all three -> DO
  NOT SUBMIT; run cheap test first.** Nearly every question asked so far
  (fluxes, tendencies, radii, momentum budgets, cloud optics) was answerable
  offline from saved checkpoint in minutes. FAILURE 2026-07-30: submitted two
  5-YEAR full-physics runs (8 h walltime each) while model had KNOWN unfixed +56
  W/m2 albedo error and no low-level circulation — five simulated years of
  broken climate, answering no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run full
  scorecard FIRST, work LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of session on hfls deficit (-40 W/m2) while dominant
  error was rsut (+56 W/m2) — and scorecard naming it was already sitting in run
  directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer ZERO
  questions, because no output attributable to any one of them. Multi-change
  config legitimate ONLY as deliberate new BASELINE labelled as such and never
  compared term-by-term against the old one. FAILURE 2026-07-30: one launch
  flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state largest outstanding scorecard term
  and why run still worth its GPU-hours. Fix big term, then extend. Short
  validation windows (days) are for "does it run and is new physics behaving";
  multi-year windows are for model that already pass.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** Hand-computed ratio that
  "matches" observed ratio is not evidence when calculation omit factors code
  actually applies. State it as arithmetic, or don't state it. FAILURE
  2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored sub-grid
  inhomogeneity factor radiation applies to cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing probe, check run
  directory for scorecard/manifest/diagnostic answering the question, and
  `scripts/validate/` for validator. Reading existing artifact cost seconds; new
  probe cost an hour and need its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule mechanical, from specific
failure in that one session. Root cause turned out to be one-line IC defect
findable offline in seconds; arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING ON
  IT — one line of arithmetic, before the conclusion.** Convert to quantity
  whose plausible span you know, compare. FAILURE: PGF probe returned 0.02
  m/s^2 and I built whole attribution on it; that value imply ~14 kg/m^3 density
  difference between ADJACENT 1-degree cells when entire ocean span ~6 kg/m^3.
  Check take seconds, number sat unused for hours, and running it immediately
  would have exposed real defect (cells initialised to T=0/S=0) before a single
  GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State instrument's resolution, confirm reported step is not
  merely its first sample. FAILURE: five one-variable arms all reported "BLOWUP
  at step 36" and I read that config-invariance as physics. `run_omip` set
  `block_size = max(1, diag_every)` and only test at block boundaries, so 36 was
  FIRST LOOK. At `--diag-every 1` answer was step 1. Corollary: arms run at
  coarse diagnostic stride prove only "this setting alone does not fix it",
  NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  same step can fail for different reasons. FAILURE: declared open North Pole
  "refuted" because capping did not move blowup step; states showed capping HAD
  removed that mode (|u|max 19.27 -> 0.0137 m/s) and merely uncovered second,
  unrelated one at same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read signature of every function a probe call,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported NaN "conservation defect"
  in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** Sentinel that is valid number of right dtype pass every finite/NaN
  guard downstream. FAILURE: `_interp_profile_to_z_coord` returned `0.0` for
  no-data column; `0.0` is not NaN, so caller's `where(isnan, fill, x)` never
  fired and 39032 wet cells entered model as fresh water at 0 degC. TWO
  COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one calling it "documented degenerate
  fallback" and one commenting "still no NaN leaks" — exact inversion of truth.
  Degenerate branch must return something → return NaN (or raise) so downstream
  guard can see it, and treat any test asserting magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run closure/redistribution check FIRST — it name the operator class
  instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted hypothesis I was about to spend a GPU arm on (dipole
  whose two-cell sum GREW is not a diffusion instability) and localised
  config-invariant source to single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance (git
  SHA + inputs + flags) stamped on every run. Inline heredoc probes not citable,
  cannot be re-run against changed model, and hide own bugs — both defects above
  came from uncommitted ones, and committing closure probe immediately surfaced
  two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In blown-up field, "worst
  cells are coastal / shallow / polar" describe where damage LANDED, usually
  downstream of single upstream defect. FAILURE: measured P(bad|coastal)=33% vs
  0.67% interior, real 50x enrichment, and treated it as coastal process; those
  were simply columns the IC defect had corrupted. Before citing spatial
  pattern, confirm snapshot is ORIGIN state (`snapshot_final.npz` stamp
  `_step = n_steps` regardless of when run stopped — verify with dedicated
  1-step run).

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe (via `global_sum_mpi`; `final_loss=True` only for a sum seeded as the loss); `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never bespoke `veros_*` solver). Oracle-matching numerics go in canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in fidelity harness, never model. Test: "would a user with different goal ever select this?" No → harness.
- Conventions handled only in bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" changing wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for block that also clear truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert pair exists) so `--test/--only =<name> --grid <grid>` never select nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never silent no-op (dispatch-hardening). MPI-safe: key guard off global pre-slice match count; empty-slice ranks fall through to barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (template test importing class directly does NOT cover dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup N/A there; do not invent one to match pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: human dictate *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fill *body*; every declared invariant checked **mechanically** so violations fail LOUDLY. Each gate = **tripwire, not proof**, ships synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — NEW physics file must ship contract or be classified. Author contract + acceptance test BEFORE body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
- **Best coding practices = use existing system**, not parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets unit test, slopbuster sweeps, iterate-with-codex on substantial changes.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick bucket by what script DOES. New bucket need real category, not dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact. EXCEPTION: `.physics-validator/` is tracked on purpose — review logs, probes and banked oracle results cited as evidence by docs, tests and validation scripts (user 2026-10-24); only its error captures are ignored.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). New subpackage → update `tests/test_federation_plan.py` same PR.
- **Do NOT "flatten" `packages/<pkg>/legoesm/<subpkg>/` → `packages/<pkg>/` — double name is LOAD-BEARING, not redundant.** Editable installs (dev/test workflow) require on-disk path == import path, and hatchling REFUSES prefix-*adding* `sources` remap in dev mode (`ValueError: Dev mode installations are unsupported when any path rewrite in the sources option changes a prefix rather than removes it`; editables#20). So `packages/atmosphere/legoesm/atmosphere/` is only layout where `import legoesm.atmosphere` resolve under `pip install -e`. Flattening to keep `legoesm.*` build wheel but break editable dev; flattening to bare `import atmosphere` mean rewriting ~23k import sites + dropping namespace. Verified 2026-07-13 — don't re-attempt. `src/legoesm/` is SANCTIONED root meta-package (the `legoesm.*` TOP-LEVEL modules: `constants`, `config`, `cli`, `registry`, `dycore_factory`, …), same load-bearing reason — NOT stray `src/` to remove. "Never add source at `src/`" mean don't create NEW `src/` trees for component code; existing `src/legoesm/` stays.
- **Tests mirror package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catch stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- **New user-tunable config field → same-PR CLI flag in every affected run script.** Rule applies to `ExperimentConfig`/`DycoreConfig`/`OutputConfig` fields in `run_amip.py`, `OceanExperimentConfig`/`KPPConfig`/`VerticalMixingConfig` in `run_omip.py`, `MultiLayerLandConfig` sub-configs in `run_lmip.py`. "User-tunable" = any float/int/str/bool field user would reasonably change for sensitivity run (excludes: internal-only flags such as `debug_precision`/`diagnostics_perf_mode`, unimplemented stubs like `carbon_cycle` in AMIP, parallel-milestone-gated fields like `distributed_mode=spmd`). Each new flag also needs: (1) wiring in `build_config_from_args` / equivalent config-builder, (2) round-trip test in corresponding `tests/unit/test_run_*_cli.py`, (3) `validate_strict` membership check if it is a scheme Literal. FLOAT tunables with `__param_spec__` need no flag: all four MIP drivers take `--config` (dests, choices-validated) + `--params` (registry-qualified `scheme_key.field`, bounds-validated; atm via flattened-scalar map). Reachability machine-audited by `tests/unit/test_params_reachability_audit.py` against shrink-only `_params_reachability_baseline.py` (new spec'd tunable unreachable in its component's drivers go red) — the #691 "CLI flag gaps" debt closure.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Mutable singletons (`grids.halo._halo_backend`/`_mpi_topology`, `cubesphere_exchange._spmd_mesh`): use accessors (`get_halo_backend`/`get_mpi_topology`/`get_spmd_mesh`), never import global. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/ packages/ | grep -v " as _"` = 0 (line-grep miss multi-line/function-scope/`_UPPER` imports); CI ratchet `tests/test_no_private_cross_imports.py` (AST, incl. function-scope; dunder-exempt; shrink-only allowlist EMPTY since 2026-06-10 — keep it empty).
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean/experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
Two CI tripwires enforce this (extend, never weaken; baselines shrink-only): `tests/test_no_inline_physics_coeffs.py` (+`_inline_coeff_baseline.py`) and `tests/test_param_specs.py` (+`_param_spec_baseline.py`).
- **No inline empirical coefficient in physics function body OR signature default.** Across `*/physics/*`, `land/`, `ice/`, `coupler/`: every float literal (and int `|v|>16`) inside function scope flagged. Move to (a) scheme `*Config` NamedTuple field if tunable/scheme-defining, (b) module-level `_UPPER_SNAKE` constant/table block (published fits/tables: Sutherland, Hall-Pruppacher, Morel-Berthon, Jerlov, KK2000) with provenance comment if fixed published constant, or (c) real `# coeff-ok: <reason>` (reason REQUIRED) for genuine numerics one-off. Exempt: math {0,0.25,0.5,1,2,3,4,6}, exact conversions, `|v|<=1e-6` floors, `|v|>=1e20` guards, subscript indices, Pow exponents, small ints (indices/counts).
- **Per-file layout (consistent placement):** docstring → imports → `__physics_contract__` → `__param_spec__` → fixed `_UPPER_SNAKE` constant/table blocks (under `# --- <category> (<reference>) ---` comments) → `*Config` NamedTuple(s) with fields grouped by same category comments (snake_case + unit suffixes) → functions whose bodies read ONLY `cfg.<field>` / `constants.*` / exempt math. Kwarg-default literals → reference module constant or config field (Name in signature, not literal).
- **Every physics scheme `*Config` declares `__param_spec__`** (module-level pure dict literal next to NamedTuple). Per float field: `units`, `bounds (lo,hi)`, `tunable_tier`, `transform` (sigmoid/softplus/none), `category`, `reference`, `shape` (None or dim key like `n_pft` for variable-size array params), optional `legacy_name`. NEW config module must ship spec or be classified in `PARAM_SPEC_TODO` (shrink-only). Inclusion computed: only `:float`-annotated fields spec-eligible.
- **Tunable/fixed split (the continuum), classify SAME way every time:**
  - `tunable_tier 1` (**core**) = params already trained in practice / well-posed (surface albedos, ice strength `P_star`, gray optical depths, bulk exchange `C_H`/`C_E`).
  - `tunable_tier 2` (**extended**) = clear closure knobs with literature bounds (relaxation timescales τ, entrainment/drag/autoconv rate coefficients, thresholds, emissivity, roughness).
  - `tunable_tier 0` / `excluded` (with reason string) = NOT trainable: numerics floors/caps/regularisers, smoothing widths, measurement conventions (e.g. MOST 10 m), and anything **iteration-coupled** (mEVP `alpha`/`beta`, EVP `T_evp` couple to subcycle count).
  - Collector select tiers `1..N` (`build_trainable_params(config, tier="core"/"extended"/"aggressive", include=, exclude=)`); flip a param's status with 1-line `tunable_tier` edit. See [[param-hygiene-spec-effort]].
- **Loop-iteration COUNTS are never config/trainable** → module constant (e.g. `_N_EVP_DEFAULT = 120`), not config field, not kwarg-default literal. Structurally guaranteed: ints not spec-eligible, so iteration count can never reach trainable collector. See [[loop-counts-never-trainable]].
- **A tunable closure whose default is a `constants.X` reference** (e.g. `S_ice_new = constants.S_ice_bulk_default`) is *eligible* (may be `__param_spec__` param with explicit bounds + tier) though not *required* (AST gate won't force it). Expose genuine calibratable closures; keep environmental references (ocean salinity) fixed/excluded.
- **Trained values inject via config pytree, not new signatures:** `params.to_overrides()` → `legoesm.core.param_overrides.apply_param_overrides(physics_config, overrides)` (`NamedTuple._replace`) INSIDE the loss so leaves TRACED (SegmentForcing doctrine); production keeps static Python-float leaves (constant-folded, no retrace). Register newly-specced module in `param_collector.SPEC_MODULES` (drift-tested).

## Naming
- Surface T = `T_sfc` everywhere. No new `T_surface`/`Ts`.
- Driver/config schema field names match runtime field. New tunable: same name in `driver/config.py`, scheme config NamedTuple, YAML schema (consistent `hyperdiff_coeff`).
- **Same name + different units = bug magnet (audit).** Unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New C-vs-K args MUST carry `_C`/`_K`. Config fields sharing base (`tau_*`, `T_*`, `c_*`, `C_*`) MUST have consistent unit suffixes OR distinct names.
- **snake_case all NamedTuple fields**, even capitalized symbols (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.
- **Renaming a tunable `__param_spec__` field = rename field AND its spec key together + add `legacy_name: <old>`.** Spec key must equal float field name (`test_param_specs`), so half-rename go red; `legacy_name` document prior name for any externally-saved tuned JSON. (2026-06-22: `CarbonConfig.T_opt→T_opt_C`/`T_width→T_width_C`, `StomataConfig.T_opt_jarvis→T_opt_jarvis_C`/`T_range_jarvis→T_range_jarvis_C` — Celsius fields gaining `_C` suffix.)
- **`tau_`/`τ` prefix valid for wind/wave STRESS [Pa], not timescales only.** Ocean wind stress and GWD launch stress (`tau_0`, `tau_max`) = standard GFD τ notation — do NOT auto-rename to `stress_*`. Real anti-pattern = MIXED units under one `tau_*` family inside single config without unit suffixes; fix by suffixing timescale members (`_s`), not stresses.

### Open naming debt
- `T_sfc`(368)/`T_surface`(59)/`Ts`(~6): coupler+`land/{multilayer_land,snow_budget,stomata_utils,slab_land}.py`, `ice/sea_ice.py`, `coupler/{accumulator,lake/two_layer_lake}.py` still `T_surface`; 3 files MIX BOTH — `driver/coupled_esm_driver.py`, `ice/sea_ice.py`, `driver/earth_system_driver.py`. Unify cleanup PR.
- `nlev`(4119)/`n_levels`(199)/`nz`(33): `nlev` dominates. Cleanup PR.
- `tau_relax`: RESOLVED 2026-05-29 → `KuoConfig.tau_relax_s`[s], `PhillipsTwoLayerConfig.tau_relax_days`[days] (matches `backscatter.tau_relax_days`). Keep unit suffix on any new relaxation-timescale field.
- `C_water`/`c_water`: RESOLVED 2026-05-29 → `SoilThermalConfig.C_water_vol`[J/m³/K], `LakeConfig.c_water_mass`[J/kg/K]. Keep `_vol`/`_mass` on new heat-capacity fields.
- `n_layers` overloaded: soil=`n_soil_layers`, ML=`n_hidden_layers`, reserve `n_layers` for atm/ocean vert.

## Dispatch (audit)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown.** Silent `else: <default>` mask typos+dead branches. Historical: `cloud_fraction.compute_cloud_properties` ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` zero CO2; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV typos → enstrophy in `{compressible_euler_mpas,primitive_eq_mpas,shallow_water_mpas,ocean_pe_mpas}.py`; bulk-scheme typos → constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz. HARDENED 2026-05-29 (now `raise ValueError`, validated at fn entry on static config): `carbon_cycle.py:step_carbon`, `coupler.py:ocean_tile_response`, `ice/sea_ice.py:_bulk_flux_dispatch`. HARDENED later — all now raise: `slab_land.py`, `multilayer_land.py`, `coupler/lake/two_layer_lake.py`, `bulk_formulas.py` (via `core/bulk_flux.py::validate_bulk_scheme`). HARDENED 2026-06-22 (atm MPAS `pv_scheme` typo→energy + thompson `snow_scheme` typo→bulk): `compressible_euler_mpas.py`, `primitive_eq_mpas.py`, `shallow_water_mpas.py` (now `energy`/`enstrophy` else-`raise`), `microphysics/thompson.py` (fn-entry `snow_scheme` guard). These are nested scheme-Config fields, NOT `ExperimentConfig` literals → factory/fn-entry raise is the defense, not `validate_strict`.
- Dispatch in `lax.fori_loop`/`lax.cond` (`coupler/bulk_flux.py:222`): validate at fn entry on static Python val, not traced body.
- **Nested scheme-Config dispatch raises too.** A `scheme`/`pv_scheme`/`snow_scheme`-style field in leaf `*Config` NamedTuple (NOT `ExperimentConfig` literal, so `validate_strict` never see it) MUST still `raise ValueError` on unknown — validated at fn/factory entry on static config value (matches `ocean_pe_mpas` sibling). Bare `else: <default>` here silently run different physics on typo. Lock each new guard in `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` (grow-only) so it can't be silently deleted.
- Add membership-set assertions in `ExperimentConfig.validate_strict` for new scheme literals. RESOLVED 2026-06-09: `convection`/`turbulence`/`gravity_wave_drag` now HAVE validate_strict membership checks (`config.py:430-455`). Guarded + enforced going forward by `tests/test_validate_strict_coverage.py` (every scheme-like config field must be membership-validated or in its `KNOWN_UNVALIDATED` loader-validated allow-list; removing guard or adding unvalidated scheme field → red). Companion factory-level guard: `tests/test_dispatch_hardening.py` (existing unknown-scheme `raise` may not be silently deleted; 76-entry grow-only baseline).

## Common Mistakes
**NamedTuple fields**: verify actual field. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrade. Adding field to `SegmentCarry`: update every call site. `grep -rn "SegmentCarry(" --include="*.py"`.

**Land/face masks (latlon C-grid)**:
- Never `state._replace(land_mask=...)` on `LatLonCGridOceanState` without updating `u_mask`+`v_mask`. Stale face masks → mass flux through walls → silent leak.
- Preferred: `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- Post-construction: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomic update 3 masks.
- `_assert_runtime_invariants` (gated `enable_runtime_checks`) catches inconsistencies.

**JIT/compilation**:
- Never build closures in training loops. `build_segment_fn` create new fn per call → inside `for epoch`/`_loss_fn` recompile each iter. Build once outside; pass changing vals as args.
- Helper fns inside `lax.scan` body: Python defs in `_single_step` recreated each trace. Move to module scope.
- Dead code from iteration: when refactoring, grep for vars assigned never used.

**SegmentCarry**: canonical hot-loop state. Adding field cross-cutting: NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, direct `SegmentCarry(...)` in validation tests. New diagnostic fields (`max_cfl`) reset to zero at segment start, not accumulated.

## Assets
Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability.

## Response Style

### RULE -3 — BREVITY IS THE DEFAULT, LENGTH IS OPT-IN (2026-08-18, FIFTH callout)
User: *"be succinct and clear - make this a default."* Rules -2, -1 and 0 all
framed brevity as a CAP to check before sending, and I wrote to the cap every
time. Inverted:

**DEFAULT REPLY = <=3 LINES. NO HEADER. NO TABLE. NO BULLETS.**

Length is earned, not assumed. It is unlocked ONLY by an explicit ask
("report", "walkthrough", "long version", "what is left") or by numbers the
user will genuinely compare (a table of >=3 rows AND >=2 columns). Bold
headers, per-item verdicts, before/after lists and caveat lines are
REQUEST-ONLY formats.

Mechanical check: write the reply, delete everything after line 3, and send it
if it still answers. If it does not, the content wanted a table — keep the
table and delete the prose.

Background-task notifications get NO reply at all unless a result is ready, a
claim was refuted, or a decision is needed.

### RULE -2 — MECHANICAL LENGTH LIMITS (2026-08-14, THIRD callout)
User: *"be more succinct and clearer - make this a rule."* RULE -1 and RULE 0
below already said this and I still shipped multi-section replies with tables
and bold headers on routine results. Prose rules did not work, so these are
COUNTABLE. Check them before sending.

- **5 lines of prose, hard.** Not 5 sentences — 5 rendered lines.
- **ONE bold header per reply, or none.** Multiple `**Headers**` = a report,
  and reports are only for when one was asked for.
- **A table needs >=3 rows AND >=2 columns of real data.** Two numbers go in a
  sentence. A table of one comparison is decoration.
- **No line explaining method, discipline, or what a control proved.** The
  result only. "Pre-registered X, got Y" is one clause, not a paragraph.
- **No restating a caveat already in the commit.** The commit is the record.
- Status/job IDs: one line total, at the end, no formatting.

If the content genuinely needs more, say "long version?" and stop.

### RULE -1 — THE CAP APPLIES TO GOOD NEWS AND BAD NEWS ALIKE (2026-08-12)
User, again, after a session of correct-but-long replies: *"be succinct and
clear."* The rule below was being followed for status and ignored for
findings — a retraction, a root cause, or a self-caught instrument bug is
NOT a licence to write six paragraphs. Length is not proof of rigor.

- **A finding is ONE line: the corrected fact.** Not the discovery story,
  not what it means for three other workstreams, not a list of what still
  stands. "X was wrong; the real number is Y" and stop.
- **Never re-explain a thing already said once in the same reply.** If a
  number appears in the verdict, it does not reappear in the evidence.
- **Cut every sentence that exists to show diligence.** "I checked", "the
  control caught it", "this is the Nth time" — delete. The tool calls are
  visible; the user is not grading effort.
- **Detail goes in the commit, the memory file, or the peer message —
  never the reply.** That is what those artifacts are for.
- Default reply: **3-5 bullets, under ~60 words.** A long reply must be
  ASKED for ("report", "walkthrough", "what is left").
**STRICT RULE, EVERY SESSION, EVERY MODEL (Opus included): be succinct AND
clear.** Not style preference, not default that decay over long session.
Succinct = answer first, ~60 words, bullets not paragraphs (RULE 0). Clear =
plain words a colleague outside this repo can act on, no bare identifiers (RULE
-1). Both must hold; reply failing either one = failed reply.

### RULE -1 — CLARITY IS THE HARD RULE. IF THE USER CANNOT FOLLOW IT, IT FAILED.
User, 2026-08-12 (and 2026-08-07, 2026-08-11 — same complaint every time):
*"Ensure you are clearer — I have no clue what you are saying most of the
time."* Outranks brevity: short reply nobody understand worse than no reply.
Brevity already being followed when this said; defect is UNEXPLAINED INTERNAL
DETAIL, not length.

MECHANICAL TEST, apply to every sentence before sending: could a colleague who
know climate modelling but never opened this repo act on it? Need file name,
function name, job id, or flag to make sense → REWRITE IT.

- Say the THING, not the SYMBOL. "the model runs radiation 18 times more often
  than intended", not "split_rad=False makes rad_update_interval_steps inert".
- NO identifiers in a reply: no file:line, no function names, no config keys, no
  job ids, no PR numbers, unless user asked for that exact thing. They belong in
  commit message. Number user should act on is fine.
- One idea per line. Line has clause explaining clause → split it.
- State CONSEQUENCE first, cause second, stop. Not mechanism, not how it was
  found, not who found it.
- Never write sentence whose subject is piece of code. Subject = model, run,
  campaign, number, or user's decision.
- Something wrong → lead with what now false and what to do. Not narrative of
  discovery.
- Question to user = numbered options in plain words, with your pick.

FAILURE PATTERN, all three callouts: long autonomous stretches. Each status
reply drifted back into repo-internal vocabulary ("the pin", "the waiver", "the
factory", "the skeleton") meaning nothing outside this session. Re-read this
section whenever session run long, and before every status report.

### RULE 0 — HARD CAP ~60 WORDS. ANSWER FIRST. STANDING ORDER, ALL SESSIONS.
Five callouts in two days (2026-08-07/08). Not style preference — a cap. Lead
with result. 3-5 bullets. Stop. Job IDs, caveats, file:line and reasoning go in
COMMIT, never the reply. Retraction = ONE sentence plus corrected fact. Long
replies only when report explicitly requested.

User, 2026-08-07, after repeated callouts in single session ("you are too
verbose", "I have no idea what you are saying", "just laser-focused summary"):
**"Make being succinct, clear and to the point a strict rule for all future
sessions."** Outranks every other formatting instinct.

Mechanical test before sending — line fail → cut it:
- Would user act differently without this line? No → delete.
- Is it method, process, or what I tried? → delete. Tool calls visible.
- Is it caveat nobody would act on? → delete (put in commit).
- Is it re-explaining something already said once? → delete.

Shape: **verdict first**, then only evidence that change the verdict, then ONE
question with numbered options if decision needed. Default length few lines.
Long reply must earn it by being asked for (report, walkthrough, per-phase
notes).

Jargon banned unless sentence also say what it mean in plain words. Say "how
much communication the split costs", not "the ppermute round count".

### PLAIN LANGUAGE FIRST — the rule that finally worked (2026-08-07)
THIRD callout: *"I really don't understand when you talk to me. Things are
incredibly unclear."* Brevity alone did NOT fix it — earlier rules were being
followed. Real defect = **unexplained jargon** and leaving user to infer the
decision. What worked, now required shape:

- **Write for colleague, not reviewer.** Short sentences. Ordinary words.
- **NO unexplained domain jargon.** Terms like *ppermute, max_degree, halo fill,
  production-exact, VJP, chromatic bound* banned unless sentence also say what
  they mean in plain words. Prefer plain phrase outright: "how much
  communication the split costs", not "the round count of the comm graph".
- **Use short LABELLED blocks**, bolded, 1-3 lines each:
  **What I built / What broke / What I was wrong about / What's safe /
  What I need from you.** Labels do the navigating so user not.
- **End with ONE explicit question and numbered options** when decision needed.
  Say which you would pick and single deciding factor.
- **Asked to compare options → answer the comparison** — what each buy, what it
  cost, and question that decide it. No re-describing options.
- **Own errors in one plain sentence, at top.** "I was wrong earlier: X is not
  guaranteed." No narration of how discovered.
- **No tables, no file:line, no job ids in reply** unless asked. Those go in
  commit message. Reply is for the decision.

### Earlier callouts (same session) — still in force
**SECOND callout (2026-08-06), because rule below was written then ignored:
_"stop being verbose. It is really hard to understand. be more direct, to the
point, clear about issues. Bullets summarizing."_ Plus: _"aren't you using
caveman?"_ — terse mode was ACTIVE and I still wrote essays.**
- **BULLETS BY DEFAULT.** Prose paragraphs = failure mode. One line per fact.
- **Lead with the issue.** Not how it was found.
- **Delete every sentence not changing what user does next.**
- **Never re-explain caveat already stated once.**
- **No near-miss stories.** "I almost got X wrong" is not finding. State
  corrected number, move on.
- Terse mode (caveman/ponytail) active → APPLIES TO WHOLE REPLY — including
  findings, status, caveats. Length not substitute for rigor.

User callout 2026-08-06 (earlier): *"you are quite unclear... provide more succinct,
clear summary, clear choice, do not make many but targeted and verified
assumptions."* Evidence reader has to assemble into conclusion is not a report.
Structure, this order, then stop:

1. **VERDICT first, <=2 lines.** What true / what happened. Never open with
   method, caveats, or narration of what was run.
2. **THE DECISION, if any: ONE recommendation.** Name option you would take and
   why, one line. Menu of options with balanced caveats push work back onto
   user — only list alternatives when they genuinely must choose, and even then
   say which you'd pick.
3. **EVIDENCE: only what change the verdict.** Decisive number, file:line, or
   measurement. Not everything checked.

- **AT MOST ONE unverified claim per response, explicitly labelled PLAUSIBLE.**
  Everything else verified before stated. No enumerating candidate causes — pick
  the one you tested, report it. Untested hypotheses = clutter reading as
  findings.
- **Do not narrate the process.** Tool calls already visible. Report outcome,
  not itinerary.
- **Table only for >=3 things compared on >=2 axes.** Otherwise a sentence.
- **Retract in one line, move on.** No re-litigating superseded claim.
- **Caveats: only those changing what user should DO.** Limitation nobody would
  act on belong in commit message, not reply.
- Numerics change -> state effect on stability, accuracy, conservation,
  differentiability. No guesses as facts.
- **No Read images** unless user asks; report path.

**TERSE. Caveman register (user, 2026-08-06: "You speak too much... no need to waste tokens").** Fragments OK. Drop articles/filler/hedging/pleasantries. No narrating what you about to do, no restating request, no re-explaining finding already stated. Prose is for FINDINGS, not for process.
- **ALWAYS end with findings summary** — table or bullets: what measured, the number, CONFIRMED vs PLAUSIBLE, what still open. That summary = deliverable; rest is scaffolding.
- Long verbatim tool output → quote only DECISIVE line (`N passed`, failing assert, peak value).
- Commits/PRs/code comments/security warnings stay full English.

# iterate-with-codex agent
1. Implement change
2. `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged
5. `/codex:review --wait` again
6. Issues remain → 4
7. Stop when clean or after 30 iter
exec
/usr/bin/bash -lc 'git status --short && git diff --stat && git worktree list' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 288ms:
 M packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
 M packages/coupler/legoesm/driver/compiled_segments.py
 M packages/coupler/legoesm/driver/model_driver.py
 M packages/coupler/legoesm/driver/physics_pipeline.py
 M packages/ml/legoesm/training/scale_build.py
 M packages/tools/legoesm/forcing/external.py
 M packages/tools/legoesm/forcing/surface_utils.py
 A scripts/validate/probe_volcanic_sw_optics.py
 M tests/unit/test_aerosol_band_ccn.py
 M tests/unit/test_cross_lane_forcing.py
 A tests/unit/test_volcanic_sw_optics.py
 M tests/unit/test_wb_amip_deck_params.py
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_prompt.md
 .../atmosphere/physics/clouds/subcolumns.py        |   2 +-
 .../atmosphere/physics/radiation/integration.py    |  18 ++
 .../atmosphere/physics/radiation/rrtmgp/rrtmgp.py  |  35 +++
 .../physics/radiation/rrtmgp/rte/two_stream.py     |  31 ++-
 .../coupler/legoesm/driver/compiled_segments.py    |  18 +-
 packages/coupler/legoesm/driver/model_driver.py    |  14 +-
 .../coupler/legoesm/driver/physics_pipeline.py     |  21 +-
 packages/ml/legoesm/training/scale_build.py        |   9 +-
 packages/tools/legoesm/forcing/external.py         | 206 ++++++++++++++----
 packages/tools/legoesm/forcing/surface_utils.py    |  51 +++--
 scripts/validate/probe_volcanic_sw_optics.py       |  89 ++++++++
 tests/unit/test_aerosol_band_ccn.py                |  48 ++--
 tests/unit/test_cross_lane_forcing.py              |   2 +-
 tests/unit/test_volcanic_sw_optics.py              | 242 +++++++++++++++++++++
 tests/unit/test_wb_amip_deck_params.py             |   2 +-
 15 files changed, 689 insertions(+), 99 deletions(-)
/glade/u/home/pg2328/legoesm                                                                                                                        66bd2dbeb8 [perf/amip-cam6-derecho-2026-09-26]
/glade/derecho/scratch/pg2328/ab_land_t22/wt_base                                                                                                   fe487244ad (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_bisect                                                                                                 bb74138b60 (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_fc                                                                                                     fc6a65ca38 (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_parent                                                                                                 efa650f75e (detached HEAD)
/glade/derecho/scratch/pg2328/ab_vertical_form/t2_code                                                                                              3425b5f102 (detached HEAD)
/glade/derecho/scratch/pg2328/c96floor_wt                                                                                                           73e01e5f20 (detached HEAD)
/glade/derecho/scratch/pg2328/cosci/round2/wt_cross                                                                                                 c58502c2d6 [les/cosci-cross]
/glade/derecho/scratch/pg2328/cosci/round2/wt_gridprobe                                                                                             6dfcfe27ba [les/cosci-gridprobe]
/glade/derecho/scratch/pg2328/cosci/round2/wt_ladder_de2ce5bd6e                                                                                     de2ce5bd6e (detached HEAD)
/glade/derecho/scratch/pg2328/cpu_ladder_wt                                                                                                         dc8cce9256 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_ctrl_wt                                                                                                   6a7ffb8758 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_meas96_wt                                                                                                 a6fbb9cfab (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_meas_wt                                                                                                   d51e978df6 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_measctrl_wt                                                                                               a48c4ba94c (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_wt                                                                                                        2bf9345ea4 [perf/duo-face-barrier-exchange]
/glade/derecho/scratch/pg2328/duo_d1_wt                                                                                                             034039ebe3 [perf/duo-tracer-subcycle-gate]
/glade/derecho/scratch/pg2328/duo_ladder_wt                                                                                                         471c6f9215 (detached HEAD)
/glade/derecho/scratch/pg2328/les_suite/code_cloudbench_3c91d57fd9                                                                                  3c91d57fd9 (detached HEAD)
/glade/derecho/scratch/pg2328/les_suite/code_cloudbench_53fde89130                                                                                  53fde89130 (detached HEAD)
/glade/derecho/scratch/pg2328/mpi_coll_wt                                                                                                           e78cd6a664 [perf/cpu-mpi-collectives]
/glade/derecho/scratch/pg2328/mpi_main_wt                                                                                                           3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/snow_soot_wt                                                                                                          489f43e214 [probe/snow-soot-snicar-2026-11-22]
/glade/derecho/scratch/pg2328/t15_baseline                                                                                                          8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t15_final_controls                                                                                                    8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t15_mutations                                                                                                         8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t18_baseline                                                                                                          8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t18_controls                                                                                                          dfb0992da3 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ab                   e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ab0                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_bare                 e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_bare0                e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_base                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_c2                   a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_c87                  c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_chtrace              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_combo                adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ctr                  c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_dew                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_dewonly              a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fix                  c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_ra              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rb              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rc              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rd              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixc                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixc_rd              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixd                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixe                 bc2f61141d [fix/land-richards-convergence-2026-10-14]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h                    9f33b405d8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h2                   35aa6b6cd4 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h3                   b8b47d0e21 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h4                   c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_head                 c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ml                   e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ml0                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mr                   35aa6b6cd4 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mrb                  bc2f61141d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mvg                  adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_netroute             a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nodq                 a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_noheat               a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nv1                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nv2                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_p                    a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pre                  ff34b84c51 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_preretry             9f33b405d8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs                  adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs2                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs_trace            adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_retry                a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_rev                  a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_trace                c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_both                 defc338105 [les/cosci-r1b]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_df_1870              1870d992f5 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_df_4c42              4c42123a49 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_fp                   95603e44f3 [les/ckpt-fingerprint-fix]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_fv2                  18aca6dd00 [les/cosci-forcing-v2]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_head                 457dbea5f8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_parent               457dbea5f8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1b_8addcec          8addcec381 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score              2497892e0c (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_3d964ebbca   3d964ebbca (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_6bbfd92070   6bbfd92070 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_a463         a463de65db (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_c1           0e3e6724d2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_c4           a5a3454ae0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_961               961fe29cb6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_abl               d5cc841a18 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_c1xc3_e2ce05e30e  e2ce05e30e (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_fix               1b5b6ef3fa [les/cosci-r2-fixes]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_full_1b5b6ef3fa   1b5b6ef3fa (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_probe             bb863dcae2 [les/cosci-r2-probe]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_xgtp              961fe29cb6 [les/cosci-r2-xgtp]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/14b68afe-2408-4e1c-8792-e211c88e60b0/scratchpad/wt_amipcam6             8293a3945d [merge/amip-cam6-into-main] prunable
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_C                    3364be400a [perf/turb-ncar-no-recompile]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_Cbase                3148c95a76 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_atm                  9d6f8a95d9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_b4                   848d249168 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_base                 a06629f58e (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_base2                2887b6bafc (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_baseD                81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_c6                   394e5a50de [config/derecho-cam6-aerosol-dir]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo                  695cecbe69 [perf/fv3duo-land-per-face]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo_base             1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo_mut              1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_dyn                  f16195c2f1 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_f1                   54e808a511 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fin                  bcf0f56516 [fix/mpas-checkpoint-finite-agree]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fix                  c212eaa477 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fm                   d02f922724 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fm d02f922724        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fs                   068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fs 068a9fb621        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fx                   8c3f8c1cc4 [fix/mpas-host-collective-wait]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gap                  f7ed885dea (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gc                   b3fe12c19d [perf/mpas-gc-freeze]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcM                  81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_early            81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_everyrun         81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_nocollect        81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_hx_rev               37915678bb (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_land                 068a9fb621 [perf/mpas-land-imbalance]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_landdiag             1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll                   e50f77c3e6 [perf/latlon-moisture-fix-jit]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll_base              1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll_rev               1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_main                 1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_os                   068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_os 068a9fb621        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_own                  7090752ccf [perf/mpas-land-owned-only]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ownD                 81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ownMut               2887b6bafc (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_roof                 22fe7bb447 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_sc                   4d77ce7e4d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_scan                 c2bc2cf8b9 [fix/mpas-rrtmgp-use-scan]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_scanB                d1ba7bdfe1 [config/mpas-rrtmgp-scan-cpu]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_rico                 e841a23a53 [les/rico-launch]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_t_mpi                3d3537422b (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_t_stage2             2ebbd557ab (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp                   01a98593a4 [les/cosci-bench-ports]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_3729d071a9     3729d071a9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_3c40954272     3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_4591e98b41     4591e98b41 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_base              3729d071a9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_ms061             061f3e4be8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_pre               4591e98b41 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_mm                   3729d071a9 [les/cosci-forcing-v2-mainmerge]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms                   dcb57c7d5d [les/cosci-mainsync]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms_main              3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms_pre               a9d7024976 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/xg/wt_old               961fe29cb6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_3arm                 893bfffea0 [run/duo-3arm-2026-11-01]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_adfc4faba6           adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_bc2f61141d           bc2f61141d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_bounds               9c2788b7f9 [bounds/duo-ladder-2026-10-29]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_branch               c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_c3955855c0           c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_duo                  f1e4fa1324 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_fix                  a7fc7d945f [fix/early-init-single-host-spmd]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_inst                 9f43c442e0 [perf/scaling-limit-instruments]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ladder               83e3e383d7 [run/duo-ladder-2026-10-29]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_lim                  9f2018f11a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_main2                7e3b4a5f1a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mainref              64945d3ee2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_merge                8f51732771 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mesh                 476fc1da49 [fix/duo-physics-single-mesh]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mesh_rev             7e3b4a5f1a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_neigh                08c2c867d6 [feat/duo-neighbour-exchange]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_nofix                dede738487 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_nv                   2cf23f5c1d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ps0                  83e3e383d7 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ps1                  41d9104c77 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_run                  dc2c649648 [perf/duo-phase-timer-window]
/glade/derecho/scratch/pg2328/tmp/cs_baseline_wt                                                                                                    7dc7d63d2b (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/t28/base_tree                                                                                                     d19572042d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/testjob/base                                                                                                      e92d8855b3 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-1919-main                                                                                                      32e3e47022 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-1919-ours                                                                                                      7554620fc2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-bisect                                                                                                         00ee327568 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-land-followups                                                                                                 a11aab5780 [fix/land-followups]
/glade/derecho/scratch/pg2328/tmp/wt-main-base                                                                                                      10fd8fb64d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-merge-main-1919                                                                                                1be756e016 [tmp/merge-main-into-1919]
/glade/derecho/scratch/pg2328/tmp/wt-merge-richards                                                                                                 dca7e7bdf6 [tmp/merge-richards-into-amip]
/glade/derecho/scratch/pg2328/tmp/wt-readme-logo                                                                                                    6854a08be6 [docs/remove-readme-logo]
/glade/derecho/scratch/pg2328/tmp/wt-richards                                                                                                       2a97cb90ac [fix/land-richards-convergence]
/glade/derecho/scratch/pg2328/tmp/wt-richards-prev                                                                                                  99742ec406 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-testfix                                                                                                        f27b52c8b8 [fix/canopy-safeguard-gpu-test]
/glade/derecho/scratch/pg2328/tmp/wt_base_ae                                                                                                        be361fa60f (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt_cloudbench1m                                                                                                   78ce575587 [les/cloudbench-1m]
/glade/derecho/scratch/pg2328/tmp/wt_mut_ae                                                                                                         be361fa60f (detached HEAD)
/glade/derecho/scratch/pg2328/trimpas_prof_wt                                                                                                       890e837200 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_ctrl                                                                                                            3f07d5b866 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_treat                                                                                                           ed2c8ead3a (detached HEAD)
/glade/derecho/scratch/pg2328/wt_latlon_tracer                                                                                                      4cd050d083 [fix/latlon-tracer-vertical-conservation]
/glade/derecho/scratch/pg2328/wt_ocean_vadv                                                                                                         a8bef2521d [fix/ocean-vertical-tracer-conservation]
/glade/derecho/scratch/pg2328/wt_spec_main                                                                                                          5b56a56a79 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_volc                                                                                                               a87671b16c [feat/volcanic-sw-file-optics]
/glade/derecho/scratch/pg2328/wt_volc_base                                                                                                          a87671b16c (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/ab_base                                                                                              6d4bc64109 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193                                                                              28ac4d9c5a [fix/land-tas-t-ref2m] locked
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a1d4c1231007d5d0b                                                                              8232c3c49d [les/campaign-stage1]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a318fc5ce2a972ffc                                                                              8d09705873 [crm-mpi-port]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a4cf45c870727d40b                                                                              da10384494 [les/cosci-harness]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a62b01bad95c93693                                                                              a8f5ac7d2f [les/gabls3]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a874b7fc139d79083                                                                              8b49890830 [worktree-agent-a874b7fc139d79083]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-abcd9a518a3f50e9a                                                                              27b522e5dc [worktree-agent-abcd9a518a3f50e9a]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ad0003e1ed1298b1e                                                                              3d3537422b [les/campaign-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ae58ae57a5e97ee45                                                                              12367e6ef2 [diag/crm-les-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-afbef113e02d249fe                                                                              75bf36b1a7 [les/campaign-stage3b]
/glade/u/home/pg2328/legoesm/.claude/worktrees/canopy-snow                                                                                          1e691ac238 [feat/clm5-canopy-snow]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-fp32-tracers                                                                                     26dc651af2 [perf/crm-fp32-tracers]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-les-scaling                                                                                      43d9f3a14d [perf/crm-les-mpi-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-mpi-gaps                                                                                         f3068db97e [fix/crm-mpi-gaps]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-rad-blocking                                                                                     8669277fd4 [perf/crm-rad-blocking]
/glade/u/home/pg2328/legoesm/.claude/worktrees/final2                                                                                               be5b35cdfa (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/final3                                                                                               26dc651af2 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_base                                                                                              7759a33780 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_fuse                                                                                              b70807a971 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2                                                                                             e80dfe7f5a [perf/les-plane-fft]
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2-run                                                                                         e80dfe7f5a (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-scaling2                                                                                         2e94c07691 [perf/les-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/mpi-ad2                                                                                              d2cce031e9 [fix/mpi-second-order-ad]
/glade/u/home/pg2328/legoesm_generic_sw                                                                                                             0eb8c973d8 [feat/generic-sw-neighbour-layer]
/glade/u/home/pg2328/legoesm_main_base                                                                                                              efa650f75e (detached HEAD)
/glade/u/home/pg2328/legoesm_ocean_base                                                                                                             3d27e99599 (detached HEAD)
/glade/u/home/pg2328/legoesm_ocean_pe                                                                                                               929cc50e69 [feat/generic-ocean-pe-pilot]
/glade/u/home/pg2328/legoesm_pe_pilot                                                                                                               8b03018311 [feat/generic-pe-pilot]
/glade/u/home/pg2328/legoesm_ppm_fix                                                                                                                67d8dd0fdd [fix/latlon-ppm-seam-conservation]
/glade/u/home/pg2328/legoesm_rbffd                                                                                                                  39339d84d4 [feat/rbffd-grid]
/glade/u/home/pg2328/legoesm_sw_wired                                                                                                               9a92761845 [feat/generic-sw-wired]
/glade/u/home/pg2328/wt_main                                                                                                                        9923fc1d4b (detached HEAD)
/glade/work/pg2328/legoesm-base0918                                                                                                                 14e0296b30 (detached HEAD)
/glade/work/pg2328/legoesm-cpuperf                                                                                                                  a06629f58e (detached HEAD)
/glade/work/pg2328/legoesm-duofix                                                                                                                   7f90aee33c [merge/duo-physics-single-mesh]
/glade/work/pg2328/legoesm-ladder                                                                                                                   9923fc1d4b (detached HEAD)
/glade/work/pg2328/legoesm-license                                                                                                                  8f1f55534d [docs/getting-started-redirect]
/glade/work/pg2328/legoesm-thomas                                                                                                                   89d84371af [perf/eager-halo-exchange-jit]

exec
/usr/bin/bash -lc 'git diff HEAD -- packages/tools/legoesm/forcing/external.py packages/tools/legoesm/forcing/surface_utils.py' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
diff --git a/packages/tools/legoesm/forcing/external.py b/packages/tools/legoesm/forcing/external.py
index c4ac00c97e..6f5e4ae027 100644
--- a/packages/tools/legoesm/forcing/external.py
+++ b/packages/tools/legoesm/forcing/external.py
@@ -824,11 +824,8 @@ def _load_volcanic_profile(
 ) -> tuple[np.ndarray, object, np.ndarray, np.ndarray, np.ndarray]:
     """CMIP6 volcanic LW aerosol as a PRESSURE-RESOLVED absorption-OD profile.
 
-    ``kind="sw"`` returns the SHORTWAVE ``ext_sun`` EXTINCTION profile instead
-    (the SW slot takes extinction; SSA/g come from the radiation config), with
-    the solar bands collapsed solar-weighted via :func:`_collapse_sw_bands`
-    (flat mean if the file has no ``wl1_sun``/``wl2_sun``).  The B1/B3 notes
-    below are LW-only; B2 (stratospheric placement) applies to both.
+    The shortwave counterpart, with the file's own per-band optics, is
+    :func:`_load_volcanic_sw_optics_profile`.
 
     Fixes three defects in feeding ``bc_aeropt_cmip6_volc_lw_b16`` to the
     RRTMGP longwave ABSORPTION optical-depth slot:
@@ -865,10 +862,12 @@ def _load_volcanic_profile(
       = top / lowest pressure), aligned so ``aod_profile[..., j]`` occupies
       ``[p_edges[j], p_edges[j+1]]``.
 
-    Raises ``ValueError`` if the file lacks ``ext_earth`` (``ext_sun`` for
-    SW; caller treats as no profile source and stays byte-identical).
+    Raises ``ValueError`` if the file lacks ``ext_earth`` (caller treats as no
+    profile source and stays byte-identical).
     """
-    ext_name = "ext_sun" if kind == "sw" else "ext_earth"
+    if kind != "lw":
+        raise ValueError(f"_load_volcanic_profile: unknown kind {kind!r}")
+    ext_name = "ext_earth"
     ds = _open_forcing_dataset(path)
     if ext_name not in ds.data_vars:
         ds.close()
@@ -880,9 +879,7 @@ def _load_volcanic_profile(
     dims = list(var.dims)
 
     # B1: single-scattering albedo -> absorption fraction (1-omega) per band.
-    if kind == "sw":
-        omega = 0.0  # SW slot wants extinction, not absorption
-    elif "omega_earth" in ds.data_vars:
+    if "omega_earth" in ds.data_vars:
         omega = np.clip(
             np.asarray(ds["omega_earth"].values, dtype=np.float64), 0.0, 1.0,
         )
@@ -926,12 +923,7 @@ def _load_volcanic_profile(
          if d not in (time_name, lat_name, alt_name)),
         None,
     )
-    if band_name is not None and kind == "sw":
-        collapsed = _collapse_sw_bands(ds, abs_ext, dims, band_name)
-        abs_ext = (np.nanmean(abs_ext, axis=dims.index(band_name))
-                   if collapsed is None else collapsed)
-        dims = [d for d in dims if d != band_name]
-    elif band_name is not None:
+    if band_name is not None:
         band_axis = dims.index(band_name)
         if "wl1_earth" in ds and "wl2_earth" in ds:
             weights = _planck_band_weights(
@@ -952,22 +944,135 @@ def _load_volcanic_profile(
     order = [dims.index(time_name), dims.index(lat_name), dims.index(alt_name)]
     prof = np.transpose(abs_ext, order)  # (ntime, nlat, nalt) [1/km]
 
-    # Altitude layer edges (midpoints, outer half-steps) and thickness [km].
+    dz_km, p_edges = _volcanic_layer_geometry(alt_km)
+    # Per-layer AOD [-] = absorption extinction [1/km] * dz [km]; layer axis
+    # flipped to ASCENDING pressure (top first) to match ``p_edges``.
+    aod_profile = (prof * dz_km[None, None, :])[..., ::-1]
+    return mid_days, first_date, lat, aod_profile, p_edges
+
+
+def _volcanic_layer_geometry(alt_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
+    """Layer thickness ``dz_km`` (nalt,) [km] in file (altitude-ascending)
+    order, and the US-Std-Atm-1976 layer pressure edges (nalt+1,) [Pa] in
+    ASCENDING pressure (top first) -- the caller flips its layer axis to
+    match.  Edges are altitude midpoints with outer half-steps."""
     edges_km = np.empty(alt_km.size + 1, dtype=np.float64)
     edges_km[1:-1] = 0.5 * (alt_km[:-1] + alt_km[1:])
     edges_km[0] = alt_km[0] - 0.5 * (alt_km[1] - alt_km[0])
     edges_km[-1] = alt_km[-1] + 0.5 * (alt_km[-1] - alt_km[-2])
-    dz_km = np.abs(np.diff(edges_km))  # (nalt,)
+    dz_km = np.abs(np.diff(edges_km))
+    p_edges = _ussa1976_pressure(edges_km * 1000.0)[::-1]
+    return dz_km, p_edges
+
+
+# Solar-band optics variables of the CMIP6 volcanic file that give the
+# stratospheric aerosol its OWN shortwave single-scattering properties.
+_VOLC_SW_OPTICS_VARS = ("ext_sun", "omega_sun", "g_sun", "wl1_sun", "wl2_sun")
+# RRTMGP shortwave band limits [cm-1], ascending (``bnd_limits_wavenumber`` of
+# rrtmgp-gas-sw-g112/g224; a test pins this table to both files).  The volcanic
+# file's bands are mapped 1:1 onto these in ascending-wavenumber order.
+_RRTMGP_SW_BAND_LIMITS_CM1 = (
+    (820.0, 2680.0), (2680.0, 3250.0), (3250.0, 4000.0), (4000.0, 4650.0),
+    (4650.0, 5150.0), (5150.0, 6150.0), (6150.0, 7700.0), (7700.0, 8050.0),
+    (8050.0, 12850.0), (12850.0, 16000.0), (16000.0, 22650.0),
+    (22650.0, 29000.0), (29000.0, 38000.0), (38000.0, 50000.0),
+)
+# The CMIP6 file is on the RRTMG-SW grid, whose one differing edge is
+# 2600 cm-1 where RRTMGP has 2680 cm-1 (RRTMG band 29 is 820-2600, band 16
+# 2600-3250); every other edge must agree to this relative tolerance.
+_RRTMG_RRTMGP_EDGE_DIFF_CM1 = ((2680.0, 2600.0),)
+_VOLC_SW_EDGE_RTOL = 1.0e-3
 
-    # Per-layer AOD [-] = absorption extinction [1/km] * dz [km].
-    aod_profile = prof * dz_km[None, None, :]
-    p_edges = _ussa1976_pressure(edges_km * 1000.0)  # [Pa]
 
-    # Return ASCENDING pressure (top first): altitude ascending -> pressure
-    # descending, so flip the layer axis and the edges together.
-    aod_profile = aod_profile[..., ::-1]
-    p_edges = p_edges[::-1]
-    return mid_days, first_date, lat, aod_profile, p_edges
+@lru_cache(maxsize=4)
+def _load_volcanic_sw_optics_profile(path: str):
+    """CMIP6 volcanic SHORTWAVE optics as per-band scattering MOMENTS.
+
+    Returns ``(mid_days, first_date, lat, moments, p_edges, band_wn)``:
+
+    * ``moments`` ``(ntime, nlat, nlayer, 3, nband)`` float32, per file layer
+      and solar band: ``(tau, tau*omega, tau*omega*g)`` with
+      ``tau = ext_sun*dz``.  Moments (not tau/omega/g) because the layer remap
+      and the tropopause weight are linear in them, so remapping each moment
+      and mixing afterwards is exact.
+    * layer axis ASCENDING pressure (top first) aligned with ``p_edges``
+      ``(nlayer+1,)`` [Pa] (same geometry as the LW profile);
+    * bands sorted to ASCENDING wavenumber (the file ships RRTMG/E3SM order,
+      820-2600 cm-1 last; RRTMGP orders 820-2680 cm-1 first), with
+      ``band_wn`` ``(nband, 2)`` [cm-1] the sorted band edges so the radiation
+      can check the 1:1 band match against its own gas-optics table.
+
+    Returns None when the file has no ``ext_sun`` (a legacy column ``aod``
+    file, which stays in the column path).  A file with ``ext_sun`` but
+    without the other ``_VOLC_SW_OPTICS_VARS``, with bands that do not match
+    :data:`_RRTMGP_SW_BAND_LIMITS_CM1`, with non-finite values, or with a
+    negative asymmetry factor (the layer remap is for non-negative optical
+    depths) raises.
+    """
+    ds = _open_forcing_dataset(path)
+    if "ext_sun" not in ds:
+        ds.close()
+        return None
+    missing = [v for v in _VOLC_SW_OPTICS_VARS if v not in ds]
+    if missing:
+        ds.close()
+        raise ValueError(
+            f"Volcanic file {path!r} has ext_sun but no {missing}: the "
+            "stratospheric aerosol needs its own solar-band optics.")
+    var = ds["ext_sun"]
+    dims = list(var.dims)
+    alt_name = next((v for v in ("altitude", "alt", "height", "lev") if v in dims),
+                    None)
+    lat_name = next((v for v in ("lat", "latitude") if v in dims), None)
+    time_name = next((v for v in ("time", "month") if v in dims), None)
+    band_name = ds["wl1_sun"].dims[0]
+    if None in (alt_name, lat_name, time_name) or band_name not in dims:
+        ds.close()
+        raise ValueError(
+            f"CMIP6 volcanic file {path!r}: ext_sun dims {dims} lack an "
+            "altitude/latitude/time/solar-band axis.")
+    order = [dims.index(n) for n in (time_name, lat_name, alt_name, band_name)]
+    ext, omega, g = (
+        np.transpose(np.asarray(ds[v].values, dtype=np.float64), order)
+        for v in ("ext_sun", "omega_sun", "g_sun"))
+    for name, a in (("ext_sun", ext), ("omega_sun", omega), ("g_sun", g)):
+        if not np.all(np.isfinite(a)):
+            ds.close()
+            raise ValueError(f"Non-finite {name} in volcanic file {path!r}.")
+    if np.any(g < 0.0) or np.any(ext < 0.0):
+        ds.close()
+        raise ValueError(f"Negative ext_sun or g_sun in volcanic file {path!r}.")
+    if time_name == "time":
+        mid_days, first_date = _read_time_axis(ds)
+    else:
+        mid_days = np.array([15.5 + 30.4375 * m for m in range(ext.shape[0])])
+        first_date = None
+    lat = np.asarray(ds[lat_name].values, dtype=np.float64)
+    alt_km = np.asarray(ds[alt_name].values, dtype=np.float64)
+    wl1 = np.asarray(ds["wl1_sun"].values, dtype=np.float64)
+    wl2 = np.asarray(ds["wl2_sun"].values, dtype=np.float64)
+    ds.close()
+
+    dz_km, p_edges = _volcanic_layer_geometry(alt_km)
+    tau = ext * dz_km[None, None, :, None]
+    ssa = np.clip(omega, 0.0, 1.0)
+    moments = np.stack([tau, tau * ssa, tau * ssa * np.clip(g, -1.0, 1.0)],
+                       axis=-2)                       # (t, lat, alt, 3, band)
+    # um -> cm-1: wn = 1e4 / wl.
+    band_wn = np.stack([1.0e4 / np.maximum(wl1, wl2),
+                        1.0e4 / np.minimum(wl1, wl2)], axis=-1)
+    srt = np.argsort(band_wn[:, 0])
+    band_wn = band_wn[srt]
+    ref = np.asarray(_RRTMGP_SW_BAND_LIMITS_CM1)
+    for a, b in _RRTMG_RRTMGP_EDGE_DIFF_CM1:
+        ref = np.where(ref == a, b, ref)
+    if band_wn.shape != ref.shape or not np.allclose(
+            band_wn, ref, rtol=_VOLC_SW_EDGE_RTOL, atol=0.0):
+        raise ValueError(
+            f"Volcanic file {path!r} solar bands {band_wn.tolist()} [cm-1] do "
+            "not map 1:1 onto the RRTMGP shortwave bands.")
+    moments = moments[..., srt][:, :, ::-1].astype(np.float32)
+    return mid_days, first_date, lat, moments, p_edges, band_wn
 
 
 @lru_cache(maxsize=16)
@@ -2341,29 +2446,40 @@ def get_aerosol_lw_at_time(config: AerosolConfig, day: float,
     return _volcanic_profile_at_time(config, day, lat_grid, "lw")
 
 
-def get_aerosol_sw_volcanic_at_time(config: AerosolConfig, day: float,
-                                    lat_grid: jnp.ndarray | None = None):
-    """SHORTWAVE volcanic extinction as a pressure-resolved profile.
+def get_aerosol_sw_volcanic_optics_at_time(config: AerosolConfig, day: float,
+                                           lat_grid: jnp.ndarray):
+    """SHORTWAVE volcanic aerosol with its OWN per-band optics from the file.
 
-    Same contract as :func:`get_aerosol_lw_at_time` but for ``ext_sun``
-    (solar-weighted gray band collapse), gated on ``volcanic_enabled``.  None
-    for a legacy column ``aod(time, lat)`` volcanic file (no profile) -- the
-    caller then keeps that column inside :func:`get_aerosol_at_time`.
+    Returns ``(moments_col, p_edges, band_wn)``: ``moments_col``
+    ``(ncol, nsrc, 3, nband)`` per-file-layer ``(tau, tau*omega,
+    tau*omega*g)`` (see :func:`_load_volcanic_sw_optics_profile`), ``p_edges``
+    ``(nsrc+1,)`` [Pa] ascending, ``band_wn`` ``(nband, 2)`` [cm-1].  None
+    when volcanic forcing is off or the file carries no ``omega_sun`` /
+    ``g_sun`` / band edges (the caller then keeps the gray path).
     """
     if not config.volcanic_enabled or not config.volcanic_path:
         return None
-    return _volcanic_profile_at_time(config, day, lat_grid, "sw")
+    loaded = _load_volcanic_sw_optics_profile(config.volcanic_path)
+    if loaded is None:
+        return None
+    out = _volcanic_profile_at_time(config, day, lat_grid, "sw_optics",
+                                    loaded=loaded[:4] + (loaded[4],))
+    return out[0], out[1], loaded[5]
 
 
-def _volcanic_profile_at_time(config, day, lat_grid, kind):
-    """Shared body of the LW / SW volcanic profile samplers."""
-    try:
-        mid_days_v, first_date_v, lat_v, prof_v, p_edges = (
-            _load_volcanic_profile(config.volcanic_path, kind)
-        )
-    except ValueError:
-        # File has no ext_earth / ext_sun band — treat as "no source".
-        return None
+def _volcanic_profile_at_time(config, day, lat_grid, kind, loaded=None):
+    """Shared body of the LW / SW volcanic profile samplers.  Trailing axes
+    after the layer axis (the SW-optics moment/band axes) are carried along."""
+    if loaded is not None:
+        mid_days_v, first_date_v, lat_v, prof_v, p_edges = loaded
+    else:
+        try:
+            mid_days_v, first_date_v, lat_v, prof_v, p_edges = (
+                _load_volcanic_profile(config.volcanic_path, kind)
+            )
+        except ValueError:
+            # File has no ext_earth / ext_sun band — treat as "no source".
+            return None
 
     # Interpolate the (ntime, nlat, nlayer) profile in time -> (nlat, nlayer).
     if len(mid_days_v) > 12:
@@ -2385,7 +2501,7 @@ def _volcanic_profile_at_time(config, day, lat_grid, kind):
         # intentionally NOT summed away (unlike the gray SW/column path).
         prof_col = _interp_zonal_to_grid(lat_v, prof_t, lat_grid)
         prof_col = jnp.reshape(
-            jnp.clip(prof_col, 0.0, None), (-1, prof_t.shape[-1]),
+            jnp.clip(prof_col, 0.0, None), (-1, *prof_t.shape[1:]),
         )
         return prof_col, jnp.asarray(p_edges)
 
diff --git a/packages/tools/legoesm/forcing/surface_utils.py b/packages/tools/legoesm/forcing/surface_utils.py
index a763e6724d..8a44d65791 100644
--- a/packages/tools/legoesm/forcing/surface_utils.py
+++ b/packages/tools/legoesm/forcing/surface_utils.py
@@ -808,40 +808,53 @@ def stratospheric_layer_weight(p_half_col: jnp.ndarray,
 def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
                          active, lw_active, dtype, lon_col=None):
     """Per-layer aerosol forcing for one day: ``(aerosol_od, ccn_aod,
-    aerosol_lw_od)``, as the AMIP driver feeds radiation and the aerosol-CCN
-    droplet number.  ``ccn_aod`` is None when ``active`` is off; the two
-    optical-depth fields are zeros (ncol, nlev) when their switch is off.
-    ``lon_col`` [rad] reads the SW and CCN aerosol at each column's longitude;
-    None reads their zonal mean."""
+    aerosol_lw_od, aerosol_sw_volc)``, as the AMIP driver feeds radiation and
+    the aerosol-CCN droplet number.  ``ccn_aod`` is None when ``active`` is
+    off; the two optical-depth fields are zeros (ncol, nlev) when their switch
+    is off.  ``aerosol_sw_volc`` is the volcanic SW aerosol with its own
+    per-band optics, ``(ncol, nlev, 3, nband)`` moments ``(tau, tau*omega,
+    tau*omega*g)`` in ascending-wavenumber band order, kept out of
+    ``aerosol_od``; None when volcanic forcing is off or the volcanic file is a
+    legacy column ``aod`` file (which then stays in ``aerosol_od`` with the
+    grey aerosol SSA/g).  ``lon_col`` [rad] reads the
+    SW and CCN aerosol at each column's longitude; None reads their zonal
+    mean."""
     from legoesm.forcing.external import (
         get_aerosol_at_time,
         get_aerosol_lw_at_time,
-        get_aerosol_sw_volcanic_at_time,
+        get_aerosol_sw_volcanic_optics_at_time,
     )
 
     ncol, nlev = p_half_col.shape[0], p_half_col.shape[1] - 1
     ccn_aod = None
+    aerosol_sw_volc = None
     aerosol_od = jnp.zeros((ncol, nlev), dtype=dtype)
     if active:
         _acfg = acfg
-        # Volcanic SW (CMIP6 ``ext_sun``) is placed at its true
-        # stratospheric pressure like the LW profile below, NOT spread
-        # with the tropospheric column by pressure mass.  A legacy
-        # column-only volcanic file (profile None) stays in the column.
-        volc_sw = get_aerosol_sw_volcanic_at_time(_acfg, day, lat_grid=lat_col)
+        # Volcanic SW with the file's own per-band extinction / SSA / g:
+        # each moment is placed at its true stratospheric pressure (linear
+        # remap + tropopause weight, so mixing the remapped moments is exact)
+        # and kept OUT of the grey ``aerosol_od``.
+        volc_opt = get_aerosol_sw_volcanic_optics_at_time(_acfg, day, lat_col)
+        if volc_opt is not None:
+            mom, p_edges_v, _band_wn = volc_opt          # (ncol, nsrc, 3, nb)
+            nsrc, n_m, n_b = mom.shape[1:]
+            flat = jnp.moveaxis(jnp.asarray(mom).reshape(ncol, nsrc, n_m * n_b),
+                                -1, 0)                   # (3*nb, ncol, nsrc)
+            placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
+                              in_axes=(0, None, None))(
+                flat, jnp.asarray(p_edges_v), p_half_col) * w_strat
+            aerosol_sw_volc = jnp.moveaxis(placed, 0, -1).reshape(
+                ncol, nlev, n_m, n_b).astype(dtype)
+        # A legacy column-only volcanic file (no ext_sun) stays in the column.
         _trop_cfg = _acfg._replace(volcanic_enabled=False)
         aerosol_col = get_aerosol_at_time(
-            _acfg if volc_sw is None else _trop_cfg, day, lat_grid=lat_col,
-            lon_grid=lon_col,
+            _acfg if volc_opt is None else _trop_cfg,
+            day, lat_grid=lat_col, lon_grid=lon_col,
         )
         aerosol_od = distribute_column_aod_to_layers(
             jnp.asarray(aerosol_col), p_half_col,
         )
-        if volc_sw is not None:
-            aerosol_od = aerosol_od + (place_stratospheric_aod_profile_to_layers(
-                jnp.asarray(volc_sw[0]), jnp.asarray(volc_sw[1]), p_half_col,
-            ) * w_strat
-            ).astype(aerosol_od.dtype)
         # AOD->CCN input: TROPOSPHERIC column in the visible (550 nm)
         # band, volcanic excluded, from ``ccn_path`` if set.  The Andreae 2009
         # fit is local total AOT: a zonal mean gives polluted E China and the
@@ -872,4 +885,4 @@ def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
                 jnp.asarray(prof_col), jnp.asarray(p_edges), p_half_col,
             ) * w_strat
             ).astype(dtype)
-    return aerosol_od, ccn_aod, aerosol_lw_od
+    return aerosol_od, ccn_aod, aerosol_lw_od, aerosol_sw_volc

exec
/usr/bin/bash -lc 'git diff HEAD -- packages/atmosphere packages/coupler packages/ml' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 7ms:
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
index f0a808310b..448d9d718f 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
@@ -413,7 +413,7 @@ PER_COLUMN_SOLVER_KEYS = frozenset({
     "T", "p_full", "p_half", "sfc_temperature", "q_v", "cos_zenith",
     "sfc_albedo", "sfc_albedo_direct", "sfc_emissivity", "o3_vmr", "o3_top_vmr",
     "aerosol_optical_depth", "aerosol_absorption_optical_depth_lw",
-    "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
+    "aerosol_volcanic_moments_sw", "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
     "cloud_fraction",
 })
 
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
index c87b5dd24a..3a4e20fefa 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
@@ -977,6 +977,7 @@ def _call_radiation_backend(
     o3_top_vmr: jnp.ndarray | None = None,
     aerosol_od: jnp.ndarray | None = None,
     aerosol_lw_od: jnp.ndarray | None = None,
+    aerosol_sw_volc: jnp.ndarray | None = None,
     solar_spectral_fraction: jnp.ndarray | None = None,
     eccf: float | jnp.ndarray = 1.0,
     tsi: jnp.ndarray | None = None,
@@ -1045,6 +1046,10 @@ def _call_radiation_backend(
         ``aerosol_absorption_optical_depth_lw``.  ``None`` (default) is a
         no-op in the solver — byte-identical to no volcanic LW aerosol.
         Ignored by gray radiation.
+    aerosol_sw_volc : jnp.ndarray or None
+        Volcanic SHORTWAVE aerosol with its own per-band optics from the
+        CMIP6 file, (ncol, nlev, 3, n_bnd_sw) moments (tau, tau*ssa,
+        tau*ssa*g), passed to the solver as ``aerosol_volcanic_moments_sw``.
     solar_spectral_fraction : jnp.ndarray or None
         Per-g-point solar weights for spectral solar-cycle forcing,
         passed through to ``solve_columns``.
@@ -1225,6 +1230,8 @@ def _call_radiation_backend(
         solar_spectral_fraction=solar_spectral_fraction,
         **cloud_kwargs,
     )
+    if aerosol_sw_volc is not None:
+        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
     if cloud_config is not None:
         # One bulk ice density for the PSD radius and the optics size (CAM6).
         _rad_kwargs["cloud_rho_ice"] = cloud_config.rho_cloud_ice
@@ -1814,6 +1821,9 @@ def _make_hydrostatic_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
         # Tile-blended SURFACE SHORTWAVE ALBEDO as a per-step TRACED forcing
         # (same channel as T_sfc / o3 / ghg above, so the JIT'd step never
@@ -2088,6 +2098,8 @@ def _make_hydrostatic_radiation(
                 _aer_ext = shard_columns(_aer_ext, column_mesh)
             if _aer_lw_ext is not None:
                 _aer_lw_ext = shard_columns(_aer_lw_ext, column_mesh)
+            if _aer_volc_ext is not None:
+                _aer_volc_ext = shard_columns(_aer_volc_ext, column_mesh)
             if _cf_ovr is not None:
                 _cf_ovr = shard_columns(_cf_ovr, column_mesh)
             if q_snow_col is not None:
@@ -2133,6 +2145,7 @@ def _make_hydrostatic_radiation(
             o3_top_vmr=_o3_top_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_precip=_conv_precip_col,
@@ -2193,6 +2206,7 @@ def _make_hydrostatic_radiation(
                 o3_top_vmr=_o3_top_ext,
                 aerosol_od=_aer_ext,
                 aerosol_lw_od=_aer_lw_ext,
+                aerosol_sw_volc=_aer_volc_ext,
                 ghg_vmr_override=_ghg_ext,
                 cloud_fraction_override=None,
                 conv_precip=None,
@@ -3264,6 +3278,9 @@ def _make_spectral_pe_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
 
         # Surface albedo / emissivity overrides: a per-step TRACED
@@ -3427,6 +3444,7 @@ def _make_spectral_pe_radiation(
             tsi=_tsi_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_mass_flux_up=_conv_mf_col,
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
index a85a6f5555..1b40686a37 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
@@ -617,6 +617,7 @@ class RRTMGP:
       cloud_d_eff_snow: jnp.ndarray | None = None,
       snow_optics=None,
       cloud_snow_weight: jnp.ndarray | None = None,
+      aerosol_volcanic_moments_sw: jnp.ndarray | None = None,
   ):
       """Compute radiation for legoESM column arrays.
 
@@ -678,6 +679,12 @@ class RRTMGP:
           byte-identical.
       solar_spectral_fraction : jnp.ndarray | None
           Per-g-point solar source weights (ngpt_sw,).
+      aerosol_volcanic_moments_sw : jnp.ndarray | None
+          Shortwave aerosol with its OWN per-band optics (CMIP6 volcanic
+          sulfate), ``(ncol, nlev, 3, n_bnd_sw)`` per-layer moments
+          ``(tau, tau*ssa, tau*ssa*g)``, bands in ascending wavenumber (the
+          SW gas-optics band order).  Mixed with the grey
+          ``aerosol_optical_depth`` as a second component.  ``None`` = absent.
       ghg_vmr_override : dict | None
           Runtime GHG VMR overrides (e.g. ``{"co2": 4.15e-4}``).
       mcica_cloud_fraction : jnp.ndarray | None
@@ -757,6 +764,9 @@ class RRTMGP:
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_absorption_optical_depth_lw = jnp.asarray(
               aerosol_absorption_optical_depth_lw).astype(_table_dtype)
+      if aerosol_volcanic_moments_sw is not None:
+          aerosol_volcanic_moments_sw = jnp.asarray(
+              aerosol_volcanic_moments_sw).astype(_table_dtype)
       if cloud_path_liq is not None:
           cloud_path_liq = jnp.asarray(cloud_path_liq).astype(_table_dtype)
       if cloud_path_ice is not None:
@@ -868,6 +878,10 @@ class RRTMGP:
           aerosol_optical_depth = _zero(aerosol_optical_depth)
           aerosol_absorption_optical_depth_lw = _zero(
               aerosol_absorption_optical_depth_lw)
+          if aerosol_volcanic_moments_sw is not None:
+              aerosol_volcanic_moments_sw = jnp.concatenate(
+                  [jnp.zeros_like(aerosol_volcanic_moments_sw[:, :1]),
+                   aerosol_volcanic_moments_sw], axis=1)
           if cloud_r_eff_liq is not None:
               cloud_r_eff_liq = _top(cloud_r_eff_liq, cloud_r_eff_liq[:, :1])
           if cloud_r_eff_ice is not None:
@@ -1117,6 +1131,25 @@ class RRTMGP:
       # full SW solve so the per-band selection is defined in exactly one place.
       aer_ssa, aer_g = _resolve_aerosol_sw_optics(config, optics_lib)
 
+      # Per-band aerosol component (volcanic): (ncol, nlev, 3, n_bnd) ->
+      # (3, n_bnd, ncol, 1, nlev+2), the same halo / bottom-first layout as
+      # the grey aerosol OD; the moments are linear in OD, so the linear halo
+      # extrapolation + clip is the treatment the grey OD gets.
+      if aerosol_volcanic_moments_sw is not None:
+          n_bnd = int(optics_lib.gas_optics_sw.n_bnd)
+          if aerosol_volcanic_moments_sw.shape[-2:] != (3, n_bnd):
+              raise ValueError(
+                  "aerosol_volcanic_moments_sw trailing shape "
+                  f"{aerosol_volcanic_moments_sw.shape[-2:]}, expected (3, "
+                  f"{n_bnd}) for this SW gas-optics table.")
+          _nc, _nl = aerosol_volcanic_moments_sw.shape[:2]
+          _vm = jnp.moveaxis(aerosol_volcanic_moments_sw, (2, 3), (0, 1))
+          _vm = _add_halos(_vm[..., ::-1].reshape(3 * n_bnd * _nc, 1, _nl))
+          volc_moments_3d = jnp.clip(
+              _vm.reshape(3, n_bnd, _nc, 1, _nl + 2), 0.0, None)
+      else:
+          volc_moments_3d = None
+
       # Optional aerosol optical depth (longwave, pure absorber)
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_od_lw_3d = jnp.clip(
@@ -1157,6 +1190,7 @@ class RRTMGP:
               aerosol_optical_depth=aerosol_od_3d,
               aerosol_single_scattering_albedo=aer_ssa,
               aerosol_asymmetry_factor=aer_g,
+              aerosol_volcanic_moments=volc_moments_3d,
           )
           # Strip the singleton Y axis + vertical halos, flip back to the
           # legoESM TOA-first convention -> (n_gpt, ncol, nlev).
@@ -1257,6 +1291,7 @@ class RRTMGP:
           aerosol_optical_depth=aerosol_od_3d,
           aerosol_single_scattering_albedo=aer_ssa,
           aerosol_asymmetry_factor=aer_g,
+          aerosol_volcanic_moments=volc_moments_3d,
           solar_fraction_by_gpt=solar_weights,
           use_scan=config.use_scan,
           gpoint_batch_size=getattr(config, "gpoint_batch_size", 0),
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
index 44aa2ef1ab..1920aede7b 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
@@ -630,6 +630,7 @@ def compute_sw_optical_props_gpt(
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Per-g-point shortwave optical properties (optical_depth / ssa / asymmetry).
 
@@ -653,7 +654,8 @@ def compute_sw_optical_props_gpt(
   )
   return _mix_sw_aerosol_and_clip(
       igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
-      aerosol_single_scattering_albedo, aerosol_asymmetry_factor)
+      aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+      aerosol_volcanic_moments)
 
 
 def _mix_sw_aerosol_and_clip(
@@ -663,8 +665,18 @@ def _mix_sw_aerosol_and_clip(
     aerosol_optical_depth: Array | None,
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
-  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g."""
+  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g.
+
+  ``aerosol_volcanic_moments`` ``(3, n_bnd_sw, *layer_shape)``: a second
+  aerosol component with its OWN per-band optics, given as scattering
+  moments ``(tau, tau*ssa, tau*ssa*g)`` (stratospheric volcanic sulfate from
+  the CMIP6 file); added to the same tau / tau*ssa / tau*ssa*g sums as the
+  grey aerosol, so the mix is the exact optical-depth-weighted combination.
+  """
+  if aerosol_optical_depth is None and aerosol_volcanic_moments is not None:
+    aerosol_optical_depth = jnp.zeros_like(sw_optical_props['optical_depth'])
   if aerosol_optical_depth is not None:
     # Per-band aerosol optics: ``aerosol_single_scattering_albedo`` /
     # ``aerosol_asymmetry_factor`` may be a SCALAR (grey aerosol, historical
@@ -687,6 +699,10 @@ def _mix_sw_aerosol_and_clip(
     w_bg = sw_optical_props['ssa']
     g_bg = sw_optical_props['asymmetry_factor']
     w_num = tau_bg * w_bg + tau_aer * ssa_aer
+    if aerosol_volcanic_moments is not None:
+      volc = aerosol_volcanic_moments[:, optics_lib.gas_optics_sw.g_point_to_bnd[igpt]]
+      tau_tot = tau_tot + volc[0]
+      w_num = w_num + volc[1]
     # AD-safe SW optical-property mixing (restored from commit 59407953
     # after AIMIP-#312 merge reverted it).  ``a / jnp.maximum(b, eps)``
     # has a ``-a/b**2`` VJP that overflows when ``b`` is at the floor —
@@ -705,6 +721,8 @@ def _mix_sw_aerosol_and_clip(
         tau_bg * w_bg * g_bg
         + tau_aer * ssa_aer * g_aer
     )
+    if aerosol_volcanic_moments is not None:
+      g_num = g_num + volc[2]
     g_denom = tau_tot * jnp.maximum(w_tot, 1.0e-12)
     g_tot = jnp.clip(
         safe_divide(g_num, g_denom, eps=1.0e-12, fill=0.0),
@@ -748,6 +766,7 @@ def compute_sw_optical_field(
     aerosol_optical_depth: Array | None = None,
     aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
     aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Stack per-g-point shortwave optical fields for the 3D MC ray tracer.
 
@@ -770,6 +789,7 @@ def compute_sw_optical_field(
         cloud_r_eff_liq, cloud_path_liq, cloud_r_eff_ice, cloud_path_ice,
         cloud_fraction, aerosol_optical_depth,
         aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+        aerosol_volcanic_moments=aerosol_volcanic_moments,
     )
     # Gas Rayleigh scattering optical depth (the gas scatter is ALL Rayleigh) as
     # a fraction of TOTAL scattering (gas + cloud + aerosol) -> the per-cell
@@ -866,6 +886,7 @@ def solve_sw(
     cloud_path_fn=None,
     clear_sky: bool = False,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Solves the two-stream radiative transfer equation for shortwave.
 
@@ -899,6 +920,9 @@ def solve_sw(
       shape as `temperature`), added to SW extinction.
     aerosol_single_scattering_albedo: Bulk aerosol single-scattering albedo.
     aerosol_asymmetry_factor: Bulk aerosol asymmetry factor.
+    aerosol_volcanic_moments: Optional second aerosol component with per-band
+      optics, ``(3, n_bnd_sw, *temperature.shape)`` moments
+      ``(tau, tau*ssa, tau*ssa*g)`` (see ``_mix_sw_aerosol_and_clip``).
     solar_fraction_by_gpt: Optional external spectral solar weights by g-point.
     use_scan: Whether to use scan or for loops for the recurrent operation.
 
@@ -997,6 +1021,7 @@ def solve_sw(
           cloud_r_eff_liq, liq_g, cloud_r_eff_ice, ice_g,
           cloud_fraction, aerosol_optical_depth,
           aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+          aerosol_volcanic_moments=aerosol_volcanic_moments,
           **snow_kw,
       )
       sw_fluxes = _sw_gpoint_fluxes(igpt, sw_optical_props)
@@ -1009,7 +1034,7 @@ def solve_sw(
           igpt, gas_props, False, cloud_r_eff_liq, liq_g,
           cloud_r_eff_ice, ice_g, cloud_fraction=cloud_fraction, **snow_kw)
       _aer = (aerosol_optical_depth, aerosol_single_scattering_albedo,
-              aerosol_asymmetry_factor)
+              aerosol_asymmetry_factor, aerosol_volcanic_moments)
       sw_fluxes = _sw_gpoint_fluxes(
           igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, cloudy_props, *_aer))
       sw_fluxes_clr = _sw_gpoint_fluxes(
diff --git a/packages/coupler/legoesm/driver/compiled_segments.py b/packages/coupler/legoesm/driver/compiled_segments.py
index c08a7d8667..9043592c7e 100644
--- a/packages/coupler/legoesm/driver/compiled_segments.py
+++ b/packages/coupler/legoesm/driver/compiled_segments.py
@@ -883,6 +883,10 @@ class SegmentForcing(NamedTuple):
     # Column-mean ozone VMR above the model top (ncol,) for the RRTMGP
     # overhead layer.  ``None`` => the solver falls back to the top-layer o3.
     o3_top_vmr: jax.Array | None = None
+    # Volcanic SHORTWAVE aerosol with the CMIP6 file's own per-band optics,
+    # (ncol, nlev, 3, n_bnd_sw) moments (tau, tau*ssa, tau*ssa*g).  ``None`` =>
+    # no such source (the volcanic SW, if any, is inside ``aerosol_od``).
+    aerosol_sw_volc: jax.Array | None = None
 
 
 # Canonical GHG species ordering for the ghg_vmr array.  The three halogens
@@ -940,6 +944,7 @@ def pack_forcing(
     land_ml_params=None,
     aerosol_ccn_aod=None,
     o3_top_vmr=None,
+    aerosol_sw_volc=None,
 ) -> SegmentForcing:
     """Pack per-segment forcing into a SegmentForcing pytree.
 
@@ -1066,6 +1071,9 @@ def pack_forcing(
         o3_top_vmr=(
             None if o3_top_vmr is None else jnp.asarray(o3_top_vmr)
         ),
+        aerosol_sw_volc=(
+            None if aerosol_sw_volc is None else jnp.asarray(aerosol_sw_volc)
+        ),
     )
 
 
@@ -1081,7 +1089,7 @@ GRID_SHAPED_FORCING_FIELDS = (
     "sfc_shflx_override", "sfc_lhflx_override", "sfc_evap_override",
     "sfc_taux_override", "sfc_tauy_override",
     "sfc_lw_up", "sfc_sw_up", "sfc_sw_down", "land_frac", "aerosol_ccn_aod",
-    "o3_top_vmr",
+    "o3_top_vmr", "aerosol_sw_volc",
 )
 
 
@@ -1603,6 +1611,7 @@ def split_physics_single_rank(carry, T_new, u_new, v_new, p_s_new,
         albedo_ice=statics.albedo_ice, albedo_ocean=statics.albedo_ocean,
         ghg_vmr_override=statics.ghg_vmr_override,
         aerosol_lw_od=statics.forcing.aerosol_lw_od,
+        aerosol_sw_volc=statics.forcing.aerosol_sw_volc,
         aerosol_ccn_aod=statics.forcing.aerosol_ccn_aod,
         o3_top_vmr=statics.forcing.o3_top_vmr,
         sfc_albedo_override=statics.forcing.sfc_albedo_override,
@@ -1658,6 +1667,7 @@ def split_physics_single_rank(carry, T_new, u_new, v_new, p_s_new,
                 statics.forcing.solar_weights, statics.forcing.s_0,
                 statics.forcing.o3_vmr, statics.forcing.aerosol_od,
                 aerosol_lw_od_precomputed=statics.forcing.aerosol_lw_od,
+                aerosol_sw_volc_precomputed=statics.forcing.aerosol_sw_volc,
                 aerosol_ccn_aod=statics.forcing.aerosol_ccn_aod,
                 o3_top_vmr=statics.forcing.o3_top_vmr,
                 tau_equator=statics.tau_equator, tau_pole=statics.tau_pole,
@@ -2592,6 +2602,7 @@ def build_segment_fn(
                     albedo_ice=_albedo_ice, albedo_ocean=_albedo_ocean,
                     ghg_vmr_override=_ghg_vmr_override,
                     aerosol_lw_od=forcing.aerosol_lw_od,
+                    aerosol_sw_volc=forcing.aerosol_sw_volc,
                     aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                     o3_top_vmr=forcing.o3_top_vmr,
                     sfc_albedo_override=forcing.sfc_albedo_override,
@@ -2634,6 +2645,7 @@ def build_segment_fn(
                             forcing.solar_weights, forcing.s_0,
                             forcing.o3_vmr, forcing.aerosol_od,
                             aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                            aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                             aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                             o3_top_vmr=forcing.o3_top_vmr,
                             tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3073,6 +3085,7 @@ def build_segment_fn(
                     forcing.solar_weights, forcing.s_0,
                     forcing.o3_vmr, forcing.aerosol_od,
                     aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                    aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                     aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                     o3_top_vmr=forcing.o3_top_vmr,
                     tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3106,6 +3119,7 @@ def build_segment_fn(
                         forcing.solar_weights, forcing.s_0,
                         forcing.o3_vmr, forcing.aerosol_od,
                         aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                        aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                         aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                         o3_top_vmr=forcing.o3_top_vmr,
                         tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3160,6 +3174,7 @@ def build_segment_fn(
                     forcing.solar_weights, forcing.s_0,
                     forcing.o3_vmr, forcing.aerosol_od,
                     aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                    aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                     aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                     o3_top_vmr=forcing.o3_top_vmr,
                     tau_equator=_tau_equator, tau_pole=_tau_pole,
@@ -3193,6 +3208,7 @@ def build_segment_fn(
                         forcing.solar_weights, forcing.s_0,
                         forcing.o3_vmr, forcing.aerosol_od,
                         aerosol_lw_od_precomputed=forcing.aerosol_lw_od,
+                        aerosol_sw_volc_precomputed=forcing.aerosol_sw_volc,
                         aerosol_ccn_aod=forcing.aerosol_ccn_aod,
                         o3_top_vmr=forcing.o3_top_vmr,
                         tau_equator=_tau_equator, tau_pole=_tau_pole,
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 8290a8f0a3..8df35bb65a 100644
--- a/packages/coupler/legoesm/driver/model_driver.py
+++ b/packages/coupler/legoesm/driver/model_driver.py
@@ -5848,6 +5848,7 @@ class ModelDriver:
             and bool(cfg.volcanic_aerosol_file)
         )
         self._aerosol_lw_od = None
+        self._aerosol_sw_volc = None
         self._o3_top_vmr = None
         self._aerosol_ccn_aod = None
         self._aerosol_config = AerosolConfig(
@@ -6155,7 +6156,8 @@ class ModelDriver:
         # (tracer_data.F90 bilinear lon/lat lininterp).
         lon_col = (jnp.asarray(self._owned_lon() if lon is None else lon)
                    .reshape(ncol) if self._aerosol_active else None)
-        aerosol_od, _ccn_aod, self._aerosol_lw_od = aerosol_layer_fields(
+        (aerosol_od, _ccn_aod, self._aerosol_lw_od,
+         self._aerosol_sw_volc) = aerosol_layer_fields(
             self._aerosol_config, day, lat_col, p_half_col, w_strat,
             active=self._aerosol_active, lw_active=self._aerosol_lw_active,
             dtype=p_s.dtype, lon_col=lon_col)
@@ -16479,6 +16481,10 @@ class ModelDriver:
                         _aer_lw = getattr(self, "_aerosol_lw_od", None)
                         if self._aerosol_lw_active and _aer_lw is not None:
                             _forcing_daily["aerosol_lw_od"] = _aer_lw
+                        # Volcanic SW with its own per-band optics (None =>
+                        # absent; the volcanic SW is then in aerosol_od).
+                        if getattr(self, "_aerosol_sw_volc", None) is not None:
+                            _forcing_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
                         # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
                         if getattr(self, "_aerosol_ccn_aod", None) is not None:
                             _forcing_daily["aerosol_ccn_aod"] = self._aerosol_ccn_aod
@@ -18463,6 +18469,8 @@ class ModelDriver:
                         _aer_lw = getattr(self, "_aerosol_lw_od", None)
                         if self._aerosol_lw_active and _aer_lw is not None:
                             _ext_daily["aerosol_lw_od"] = _aer_lw
+                        if getattr(self, "_aerosol_sw_volc", None) is not None:
+                            _ext_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
                         # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
                         if getattr(self, "_aerosol_ccn_aod", None) is not None:
                             _ext_daily["aerosol_ccn_aod"] = self._aerosol_ccn_aod
@@ -19436,6 +19444,7 @@ class ModelDriver:
                 solar_weights=solar_weights, s_0=current_s_0,
                 o3_vmr=o3_vmr, aerosol_od=aerosol_od,
                 aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                 aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                 o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                 ghg_vmr=ghg_vmr,
@@ -20651,6 +20660,7 @@ class ModelDriver:
                 solar_weights=solar_weights, s_0=current_s_0,
                 o3_vmr=o3_vmr, aerosol_od=aerosol_od,
                 aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                 aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                 o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                 ghg_vmr=ghg_vmr,
@@ -21540,6 +21550,7 @@ class ModelDriver:
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                 ghg_vmr_override=ghg_vmr,
                 aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                 aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                 o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                 land_ml=(self._land_ml_state
@@ -21685,6 +21696,7 @@ class ModelDriver:
                     held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                     ghg_vmr_override=ghg_vmr,
                     aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
+                    aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
                     aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
                     o3_top_vmr=getattr(self, "_o3_top_vmr", None),
                     land_ml=(self._land_ml_state
diff --git a/packages/coupler/legoesm/driver/physics_pipeline.py b/packages/coupler/legoesm/driver/physics_pipeline.py
index 650a26ad92..95a4f44f65 100644
--- a/packages/coupler/legoesm/driver/physics_pipeline.py
+++ b/packages/coupler/legoesm/driver/physics_pipeline.py
@@ -2896,6 +2896,7 @@ class PhysicsPipeline:
                                solar_weights, s_0,
                                o3_vmr_precomputed, aerosol_od_precomputed,
                                aerosol_lw_od_precomputed=None,
+                               aerosol_sw_volc_precomputed=None,
                                aerosol_ccn_aod=None,
                                o3_top_vmr=None,
                                tau_equator=None, tau_pole=None,
@@ -3318,6 +3319,7 @@ class PhysicsPipeline:
             o3_vmr_precomputed = _shard(o3_vmr_precomputed)
             aerosol_od_precomputed = _shard(aerosol_od_precomputed)
             aerosol_lw_od_precomputed = _shard(aerosol_lw_od_precomputed)
+            aerosol_sw_volc_precomputed = _shard(aerosol_sw_volc_precomputed)
             o3_top_vmr = _shard(o3_top_vmr)
             if cloud_kwargs:
                 cloud_kwargs = {k: (v if k == "cloud_rho_ice" else _shard(v))
@@ -3334,6 +3336,8 @@ class PhysicsPipeline:
             ghg_vmr_override=ghg_vmr_override,
             aerosol_lw_od_col=aerosol_lw_od_precomputed,
             o3_top_vmr_col=o3_top_vmr,
+            **({} if aerosol_sw_volc_precomputed is None
+               else {"aerosol_sw_volc_col": aerosol_sw_volc_precomputed}),
             # only the rrtmgp radiation_fn takes it (cesm_ocean_albedo)
             **({} if open_ocean_frac_col is None
                else {"open_ocean_frac_col": open_ocean_frac_col}),
@@ -3545,6 +3549,7 @@ class PhysicsPipeline:
                          albedo_ocean=pipeline.albedo_ocean,
                          ghg_vmr_override=None,
                          aerosol_lw_od=None,
+                         aerosol_sw_volc=None,
                          aerosol_ccn_aod=None,
                          o3_top_vmr=None,
                          T_land=None,
@@ -3626,6 +3631,7 @@ class PhysicsPipeline:
                         day_of_year, seconds_of_day,
                         solar_weights, s_0, o3_vmr, aerosol_od,
                         aerosol_lw_od_precomputed=aerosol_lw_od,
+                        aerosol_sw_volc_precomputed=aerosol_sw_volc,
                         aerosol_ccn_aod=aerosol_ccn_aod,
                         o3_top_vmr=o3_top_vmr,
                         tau_equator=tau_equator, tau_pole=tau_pole,
@@ -3841,8 +3847,11 @@ def _build_none_radiation_fn(config):
                      cloud_path_liq_lw=None, cloud_path_ice_lw=None,
                      cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                      cloud_rho_ice=None,
-                     cloud_fraction=None):
+                     cloud_fraction=None, aerosol_sw_volc_col=None):
         del aerosol_lw_od_col, o3_top_vmr_col  # zero-radiation: no-ops
+        if aerosol_sw_volc_col is not None:
+            raise ValueError("zero radiation cannot consume the volcanic SW "
+                             "per-band aerosol optics; use rrtmgp radiation.")
         ncol, nlev = T_col.shape
         z_full = jnp.zeros((ncol, nlev), dtype=T_col.dtype)
         z_half = jnp.zeros((ncol, nlev + 1), dtype=T_col.dtype)
@@ -3896,9 +3905,12 @@ def _build_gray_radiation_fn(config):
                      cloud_path_liq_lw=None, cloud_path_ice_lw=None,
                      cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                      cloud_rho_ice=None,
-                     cloud_fraction=None):
+                     cloud_fraction=None, aerosol_sw_volc_col=None):
         del ghg_vmr_override  # gray radiation does not use GHG concentrations
         del aerosol_lw_od_col, o3_top_vmr_col  # gray: no aerosol LW od / ozone
+        if aerosol_sw_volc_col is not None:
+            raise ValueError("gray radiation cannot consume the volcanic SW "
+                             "per-band aerosol optics; use rrtmgp radiation.")
         del cloud_path_liq, cloud_path_ice, cloud_r_eff_liq, cloud_r_eff_ice, cloud_fraction
         del cloud_rho_ice
         del cloud_path_liq_lw, cloud_path_ice_lw  # gray: no cloud optics
@@ -4026,7 +4038,8 @@ def _build_rrtmgp_radiation_fn(config):
                      cloud_path_liq_lw=None, cloud_path_ice_lw=None,
                      cloud_r_eff_liq=None, cloud_r_eff_ice=None,
                      cloud_rho_ice=None,
-                     cloud_fraction=None, open_ocean_frac_col=None):
+                     cloud_fraction=None, open_ocean_frac_col=None,
+                     aerosol_sw_volc_col=None):
         del tau_equator, tau_pole  # RRTMGP does not use gray optical depth
         _sw_scale = None
         # Eccentricity scales the incident SW *flux* by (a/r)^2; the orbital
@@ -4092,6 +4105,8 @@ def _build_rrtmgp_radiation_fn(config):
             o3_vmr=o3_vmr_col,
             aerosol_optical_depth=aerosol_od_col,
             aerosol_absorption_optical_depth_lw=aerosol_lw_od_col,
+            **({} if aerosol_sw_volc_col is None
+               else {"aerosol_volcanic_moments_sw": aerosol_sw_volc_col}),
             solar_spectral_fraction=solar_weights if solar_weights.size > 0 else None,
             ghg_vmr_override=ghg_vmr_override,
             cloud_path_liq=cloud_path_liq,
diff --git a/packages/ml/legoesm/training/scale_build.py b/packages/ml/legoesm/training/scale_build.py
index 7ab2138bbc..ce21b01893 100644
--- a/packages/ml/legoesm/training/scale_build.py
+++ b/packages/ml/legoesm/training/scale_build.py
@@ -1607,8 +1607,13 @@ def wb_aerosol_forcing(spec, grid, sigma, carry, time_ns):
     per_year = [aerosol_layer_fields(
         acfg._replace(start_year=y), sim_day, lat_col, p_half, w_strat,
         active=True, lw_active=lw_active, dtype=p_s.dtype) for y in years]
-    od, ccn, lw = (jnp.mean(jnp.stack(f), axis=0) for f in zip(*per_year))
-    return {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
+    od, ccn, lw = (jnp.mean(jnp.stack(f), axis=0)
+                   for f in list(zip(*per_year))[:3])
+    out = {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
+    volc = [f[3] for f in per_year]
+    if volc[0] is not None:
+        out["aerosol_sw_volc"] = jnp.mean(jnp.stack(volc), axis=0)
+    return out
 
 
 def build_spectral_forcing(sl, grid, time_ns, year, *, aerosol=None,

exec
/usr/bin/bash -lc 'git diff HEAD -- packages/atmosphere; cat tests/unit/test_volcanic_sw_optics.py; cat scripts/validate/probe_volcanic_sw_optics.py' in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
index f0a808310b..448d9d718f 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
@@ -413,7 +413,7 @@ PER_COLUMN_SOLVER_KEYS = frozenset({
     "T", "p_full", "p_half", "sfc_temperature", "q_v", "cos_zenith",
     "sfc_albedo", "sfc_albedo_direct", "sfc_emissivity", "o3_vmr", "o3_top_vmr",
     "aerosol_optical_depth", "aerosol_absorption_optical_depth_lw",
-    "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
+    "aerosol_volcanic_moments_sw", "cloud_path_liq", "cloud_path_ice", "cloud_r_eff_liq", "cloud_r_eff_ice",
     "cloud_fraction",
 })
 
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
index c87b5dd24a..3a4e20fefa 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
@@ -977,6 +977,7 @@ def _call_radiation_backend(
     o3_top_vmr: jnp.ndarray | None = None,
     aerosol_od: jnp.ndarray | None = None,
     aerosol_lw_od: jnp.ndarray | None = None,
+    aerosol_sw_volc: jnp.ndarray | None = None,
     solar_spectral_fraction: jnp.ndarray | None = None,
     eccf: float | jnp.ndarray = 1.0,
     tsi: jnp.ndarray | None = None,
@@ -1045,6 +1046,10 @@ def _call_radiation_backend(
         ``aerosol_absorption_optical_depth_lw``.  ``None`` (default) is a
         no-op in the solver — byte-identical to no volcanic LW aerosol.
         Ignored by gray radiation.
+    aerosol_sw_volc : jnp.ndarray or None
+        Volcanic SHORTWAVE aerosol with its own per-band optics from the
+        CMIP6 file, (ncol, nlev, 3, n_bnd_sw) moments (tau, tau*ssa,
+        tau*ssa*g), passed to the solver as ``aerosol_volcanic_moments_sw``.
     solar_spectral_fraction : jnp.ndarray or None
         Per-g-point solar weights for spectral solar-cycle forcing,
         passed through to ``solve_columns``.
@@ -1225,6 +1230,8 @@ def _call_radiation_backend(
         solar_spectral_fraction=solar_spectral_fraction,
         **cloud_kwargs,
     )
+    if aerosol_sw_volc is not None:
+        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
     if cloud_config is not None:
         # One bulk ice density for the PSD radius and the optics size (CAM6).
         _rad_kwargs["cloud_rho_ice"] = cloud_config.rho_cloud_ice
@@ -1814,6 +1821,9 @@ def _make_hydrostatic_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
         # Tile-blended SURFACE SHORTWAVE ALBEDO as a per-step TRACED forcing
         # (same channel as T_sfc / o3 / ghg above, so the JIT'd step never
@@ -2088,6 +2098,8 @@ def _make_hydrostatic_radiation(
                 _aer_ext = shard_columns(_aer_ext, column_mesh)
             if _aer_lw_ext is not None:
                 _aer_lw_ext = shard_columns(_aer_lw_ext, column_mesh)
+            if _aer_volc_ext is not None:
+                _aer_volc_ext = shard_columns(_aer_volc_ext, column_mesh)
             if _cf_ovr is not None:
                 _cf_ovr = shard_columns(_cf_ovr, column_mesh)
             if q_snow_col is not None:
@@ -2133,6 +2145,7 @@ def _make_hydrostatic_radiation(
             o3_top_vmr=_o3_top_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_precip=_conv_precip_col,
@@ -2193,6 +2206,7 @@ def _make_hydrostatic_radiation(
                 o3_top_vmr=_o3_top_ext,
                 aerosol_od=_aer_ext,
                 aerosol_lw_od=_aer_lw_ext,
+                aerosol_sw_volc=_aer_volc_ext,
                 ghg_vmr_override=_ghg_ext,
                 cloud_fraction_override=None,
                 conv_precip=None,
@@ -3264,6 +3278,9 @@ def _make_spectral_pe_radiation(
         _aer_lw_ext = (
             forcing.get("aerosol_lw_od") if forcing is not None else None
         )
+        _aer_volc_ext = (
+            forcing.get("aerosol_sw_volc") if forcing is not None else None
+        )
         _ghg_ext = forcing.get("ghg_vmr") if forcing is not None else None
 
         # Surface albedo / emissivity overrides: a per-step TRACED
@@ -3427,6 +3444,7 @@ def _make_spectral_pe_radiation(
             tsi=_tsi_ext,
             aerosol_od=_aer_ext,
             aerosol_lw_od=_aer_lw_ext,
+            aerosol_sw_volc=_aer_volc_ext,
             ghg_vmr_override=_ghg_ext,
             cloud_fraction_override=_cf_ovr,
             conv_mass_flux_up=_conv_mf_col,
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
index a85a6f5555..1b40686a37 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
@@ -617,6 +617,7 @@ class RRTMGP:
       cloud_d_eff_snow: jnp.ndarray | None = None,
       snow_optics=None,
       cloud_snow_weight: jnp.ndarray | None = None,
+      aerosol_volcanic_moments_sw: jnp.ndarray | None = None,
   ):
       """Compute radiation for legoESM column arrays.
 
@@ -678,6 +679,12 @@ class RRTMGP:
           byte-identical.
       solar_spectral_fraction : jnp.ndarray | None
           Per-g-point solar source weights (ngpt_sw,).
+      aerosol_volcanic_moments_sw : jnp.ndarray | None
+          Shortwave aerosol with its OWN per-band optics (CMIP6 volcanic
+          sulfate), ``(ncol, nlev, 3, n_bnd_sw)`` per-layer moments
+          ``(tau, tau*ssa, tau*ssa*g)``, bands in ascending wavenumber (the
+          SW gas-optics band order).  Mixed with the grey
+          ``aerosol_optical_depth`` as a second component.  ``None`` = absent.
       ghg_vmr_override : dict | None
           Runtime GHG VMR overrides (e.g. ``{"co2": 4.15e-4}``).
       mcica_cloud_fraction : jnp.ndarray | None
@@ -757,6 +764,9 @@ class RRTMGP:
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_absorption_optical_depth_lw = jnp.asarray(
               aerosol_absorption_optical_depth_lw).astype(_table_dtype)
+      if aerosol_volcanic_moments_sw is not None:
+          aerosol_volcanic_moments_sw = jnp.asarray(
+              aerosol_volcanic_moments_sw).astype(_table_dtype)
       if cloud_path_liq is not None:
           cloud_path_liq = jnp.asarray(cloud_path_liq).astype(_table_dtype)
       if cloud_path_ice is not None:
@@ -868,6 +878,10 @@ class RRTMGP:
           aerosol_optical_depth = _zero(aerosol_optical_depth)
           aerosol_absorption_optical_depth_lw = _zero(
               aerosol_absorption_optical_depth_lw)
+          if aerosol_volcanic_moments_sw is not None:
+              aerosol_volcanic_moments_sw = jnp.concatenate(
+                  [jnp.zeros_like(aerosol_volcanic_moments_sw[:, :1]),
+                   aerosol_volcanic_moments_sw], axis=1)
           if cloud_r_eff_liq is not None:
               cloud_r_eff_liq = _top(cloud_r_eff_liq, cloud_r_eff_liq[:, :1])
           if cloud_r_eff_ice is not None:
@@ -1117,6 +1131,25 @@ class RRTMGP:
       # full SW solve so the per-band selection is defined in exactly one place.
       aer_ssa, aer_g = _resolve_aerosol_sw_optics(config, optics_lib)
 
+      # Per-band aerosol component (volcanic): (ncol, nlev, 3, n_bnd) ->
+      # (3, n_bnd, ncol, 1, nlev+2), the same halo / bottom-first layout as
+      # the grey aerosol OD; the moments are linear in OD, so the linear halo
+      # extrapolation + clip is the treatment the grey OD gets.
+      if aerosol_volcanic_moments_sw is not None:
+          n_bnd = int(optics_lib.gas_optics_sw.n_bnd)
+          if aerosol_volcanic_moments_sw.shape[-2:] != (3, n_bnd):
+              raise ValueError(
+                  "aerosol_volcanic_moments_sw trailing shape "
+                  f"{aerosol_volcanic_moments_sw.shape[-2:]}, expected (3, "
+                  f"{n_bnd}) for this SW gas-optics table.")
+          _nc, _nl = aerosol_volcanic_moments_sw.shape[:2]
+          _vm = jnp.moveaxis(aerosol_volcanic_moments_sw, (2, 3), (0, 1))
+          _vm = _add_halos(_vm[..., ::-1].reshape(3 * n_bnd * _nc, 1, _nl))
+          volc_moments_3d = jnp.clip(
+              _vm.reshape(3, n_bnd, _nc, 1, _nl + 2), 0.0, None)
+      else:
+          volc_moments_3d = None
+
       # Optional aerosol optical depth (longwave, pure absorber)
       if aerosol_absorption_optical_depth_lw is not None:
           aerosol_od_lw_3d = jnp.clip(
@@ -1157,6 +1190,7 @@ class RRTMGP:
               aerosol_optical_depth=aerosol_od_3d,
               aerosol_single_scattering_albedo=aer_ssa,
               aerosol_asymmetry_factor=aer_g,
+              aerosol_volcanic_moments=volc_moments_3d,
           )
           # Strip the singleton Y axis + vertical halos, flip back to the
           # legoESM TOA-first convention -> (n_gpt, ncol, nlev).
@@ -1257,6 +1291,7 @@ class RRTMGP:
           aerosol_optical_depth=aerosol_od_3d,
           aerosol_single_scattering_albedo=aer_ssa,
           aerosol_asymmetry_factor=aer_g,
+          aerosol_volcanic_moments=volc_moments_3d,
           solar_fraction_by_gpt=solar_weights,
           use_scan=config.use_scan,
           gpoint_batch_size=getattr(config, "gpoint_batch_size", 0),
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
index 44aa2ef1ab..1920aede7b 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
@@ -630,6 +630,7 @@ def compute_sw_optical_props_gpt(
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Per-g-point shortwave optical properties (optical_depth / ssa / asymmetry).
 
@@ -653,7 +654,8 @@ def compute_sw_optical_props_gpt(
   )
   return _mix_sw_aerosol_and_clip(
       igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
-      aerosol_single_scattering_albedo, aerosol_asymmetry_factor)
+      aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+      aerosol_volcanic_moments)
 
 
 def _mix_sw_aerosol_and_clip(
@@ -663,8 +665,18 @@ def _mix_sw_aerosol_and_clip(
     aerosol_optical_depth: Array | None,
     aerosol_single_scattering_albedo: float | Array,
     aerosol_asymmetry_factor: float | Array,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
-  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g."""
+  """Mix aerosol into one g-point's SW gas(+cloud) optics, then clip ssa/g.
+
+  ``aerosol_volcanic_moments`` ``(3, n_bnd_sw, *layer_shape)``: a second
+  aerosol component with its OWN per-band optics, given as scattering
+  moments ``(tau, tau*ssa, tau*ssa*g)`` (stratospheric volcanic sulfate from
+  the CMIP6 file); added to the same tau / tau*ssa / tau*ssa*g sums as the
+  grey aerosol, so the mix is the exact optical-depth-weighted combination.
+  """
+  if aerosol_optical_depth is None and aerosol_volcanic_moments is not None:
+    aerosol_optical_depth = jnp.zeros_like(sw_optical_props['optical_depth'])
   if aerosol_optical_depth is not None:
     # Per-band aerosol optics: ``aerosol_single_scattering_albedo`` /
     # ``aerosol_asymmetry_factor`` may be a SCALAR (grey aerosol, historical
@@ -687,6 +699,10 @@ def _mix_sw_aerosol_and_clip(
     w_bg = sw_optical_props['ssa']
     g_bg = sw_optical_props['asymmetry_factor']
     w_num = tau_bg * w_bg + tau_aer * ssa_aer
+    if aerosol_volcanic_moments is not None:
+      volc = aerosol_volcanic_moments[:, optics_lib.gas_optics_sw.g_point_to_bnd[igpt]]
+      tau_tot = tau_tot + volc[0]
+      w_num = w_num + volc[1]
     # AD-safe SW optical-property mixing (restored from commit 59407953
     # after AIMIP-#312 merge reverted it).  ``a / jnp.maximum(b, eps)``
     # has a ``-a/b**2`` VJP that overflows when ``b`` is at the floor —
@@ -705,6 +721,8 @@ def _mix_sw_aerosol_and_clip(
         tau_bg * w_bg * g_bg
         + tau_aer * ssa_aer * g_aer
     )
+    if aerosol_volcanic_moments is not None:
+      g_num = g_num + volc[2]
     g_denom = tau_tot * jnp.maximum(w_tot, 1.0e-12)
     g_tot = jnp.clip(
         safe_divide(g_num, g_denom, eps=1.0e-12, fill=0.0),
@@ -748,6 +766,7 @@ def compute_sw_optical_field(
     aerosol_optical_depth: Array | None = None,
     aerosol_single_scattering_albedo: float | Array = _AEROSOL_SSA_DEFAULT,
     aerosol_asymmetry_factor: float | Array = _AEROSOL_ASYM_DEFAULT,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Stack per-g-point shortwave optical fields for the 3D MC ray tracer.
 
@@ -770,6 +789,7 @@ def compute_sw_optical_field(
         cloud_r_eff_liq, cloud_path_liq, cloud_r_eff_ice, cloud_path_ice,
         cloud_fraction, aerosol_optical_depth,
         aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+        aerosol_volcanic_moments=aerosol_volcanic_moments,
     )
     # Gas Rayleigh scattering optical depth (the gas scatter is ALL Rayleigh) as
     # a fraction of TOTAL scattering (gas + cloud + aerosol) -> the per-cell
@@ -866,6 +886,7 @@ def solve_sw(
     cloud_path_fn=None,
     clear_sky: bool = False,
     cloud_snow=None,
+    aerosol_volcanic_moments: Array | None = None,
 ) -> dict[str, Array]:
   """Solves the two-stream radiative transfer equation for shortwave.
 
@@ -899,6 +920,9 @@ def solve_sw(
       shape as `temperature`), added to SW extinction.
     aerosol_single_scattering_albedo: Bulk aerosol single-scattering albedo.
     aerosol_asymmetry_factor: Bulk aerosol asymmetry factor.
+    aerosol_volcanic_moments: Optional second aerosol component with per-band
+      optics, ``(3, n_bnd_sw, *temperature.shape)`` moments
+      ``(tau, tau*ssa, tau*ssa*g)`` (see ``_mix_sw_aerosol_and_clip``).
     solar_fraction_by_gpt: Optional external spectral solar weights by g-point.
     use_scan: Whether to use scan or for loops for the recurrent operation.
 
@@ -997,6 +1021,7 @@ def solve_sw(
           cloud_r_eff_liq, liq_g, cloud_r_eff_ice, ice_g,
           cloud_fraction, aerosol_optical_depth,
           aerosol_single_scattering_albedo, aerosol_asymmetry_factor,
+          aerosol_volcanic_moments=aerosol_volcanic_moments,
           **snow_kw,
       )
       sw_fluxes = _sw_gpoint_fluxes(igpt, sw_optical_props)
@@ -1009,7 +1034,7 @@ def solve_sw(
           igpt, gas_props, False, cloud_r_eff_liq, liq_g,
           cloud_r_eff_ice, ice_g, cloud_fraction=cloud_fraction, **snow_kw)
       _aer = (aerosol_optical_depth, aerosol_single_scattering_albedo,
-              aerosol_asymmetry_factor)
+              aerosol_asymmetry_factor, aerosol_volcanic_moments)
       sw_fluxes = _sw_gpoint_fluxes(
           igpt, _mix_sw_aerosol_and_clip(igpt, optics_lib, cloudy_props, *_aer))
       sw_fluxes_clr = _sw_gpoint_fluxes(
"""Volcanic stratospheric aerosol in the SHORTWAVE with the CMIP6 file's own
per-band extinction / single-scattering albedo / asymmetry.

- loader: bands sorted to RRTMGP order and checked against its band limits,
  moments (tau, tau*ssa, tau*ssa*g), refusals (no SSA/g, wrong bands, g<0);
- the band-limit table equals both RRTMGP SW gas-optics files;
- RRTMGP: moments carrying the grey ssa/g reproduce the grey path; the file's
  non-absorbing sulfate absorbs less than the grey 0.93; SW flux closure;
  jit parity; finite non-zero gradient;
- gray / zero radiation refuse the field.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
# RRTMG-SW / CMIP6-file band edges [um], in the file's own (E3SM) order.
_WL1 = np.array([3.07692, 2.5, 2.15054, 1.94175, 1.62602, 1.2987, 1.24224,
                 0.77821, 0.625, 0.441501, 0.344828, 0.263158, 0.2, 3.84615])
_WL2 = np.array([3.84615, 3.07692, 2.5, 2.15054, 1.94175, 1.62602, 1.2987,
                 1.24224, 0.77821, 0.625, 0.441501, 0.344828, 0.263158, 12.1951])
_LAT = np.linspace(-90.0, 90.0, 7)
_PROD_FILE = "/glade/work/pg2328/amip_inputs/volcanic_cmip6_from_cesm_1978-2014.nc"


def volc_sw_file(path, ext=0.01, omega=None, g=None, alt=(18.0, 20.0, 22.0, 24.0),
                 wl=(_WL1, _WL2), drop=()):
    """CMIP6-shaped volcanic file: ``ext_sun/omega_sun/g_sun(solar_bands,
    lat, altitude, month)``; per-band ``omega``/``g`` (file band order)."""
    nb = len(wl[0])
    omega = np.full(nb, 0.99) if omega is None else np.asarray(omega, float)
    g = np.full(nb, 0.7) if g is None else np.asarray(g, float)
    shape = (nb, _LAT.size, len(alt), 12)
    with netCDF4.Dataset(path, "w") as ds:
        for d, n in (("solar_bands", nb), ("lat", _LAT.size),
                     ("altitude", len(alt)), ("month", 12)):
            ds.createDimension(d, n)
        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
        ds.createVariable("altitude", "f8", ("altitude",))[:] = np.asarray(alt)
        dims = ("solar_bands", "lat", "altitude", "month")
        fields = {"ext_sun": np.broadcast_to(np.asarray(ext, float), shape),
                  "omega_sun": np.broadcast_to(omega[:, None, None, None], shape),
                  "g_sun": np.broadcast_to(g[:, None, None, None], shape)}
        for k, v in fields.items():
            if k not in drop:
                ds.createVariable(k, "f8", dims)[:] = v
        ds.createVariable("wl1_sun", "f8", ("solar_bands",))[:] = wl[0]
        ds.createVariable("wl2_sun", "f8", ("solar_bands",))[:] = wl[1]
    return str(path)


def _load(path):
    from legoesm.forcing.external import _load_volcanic_sw_optics_profile
    _load_volcanic_sw_optics_profile.cache_clear()
    return _load_volcanic_sw_optics_profile(path)


def test_loader_sorts_bands_and_builds_moments(tmp_path):
    omega = np.linspace(0.2, 1.0, 14)
    g = np.linspace(0.3, 0.8, 14)
    _, _, lat, mom, p_edges, band_wn = _load(
        volc_sw_file(tmp_path / "v.nc", omega=omega, g=g))
    assert mom.shape == (12, _LAT.size, 4, 3, 14) and mom.dtype == np.float32
    assert np.all(np.diff(p_edges) > 0)           # top first
    assert np.all(np.diff(band_wn[:, 0]) > 0)     # ascending wavenumber
    srt = np.argsort(1.0e4 / np.maximum(_WL1, _WL2))
    assert srt[0] == 13                           # 820-2600 cm-1 band first
    tau = 0.01 * 2.0                               # ext [1/km] * dz 2 km
    np.testing.assert_allclose(mom[0, 0, 0, 0], tau, rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 1], tau * omega[srt], rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 2], tau * omega[srt] * g[srt], rtol=1e-6)


@pytest.mark.parametrize("kw, match", [
    ({"drop": ("omega_sun",)}, "omega_sun"),
    ({"drop": ("g_sun",)}, "g_sun"),
    ({"g": np.full(14, -0.1)}, "Negative"),
    ({"wl": (_WL1[:13], _WL2[:13])}, "map 1:1"),
    ({"wl": (_WL1 * 1.05, _WL2 * 1.05)}, "map 1:1"),
])
def test_loader_refuses(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        _load(volc_sw_file(tmp_path / "v.nc", **kw))


def test_legacy_column_file_is_not_an_optics_source(tmp_path):
    p = str(tmp_path / "legacy.nc")
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", 3)
        ds.createVariable("lat", "f8", ("lat",))[:] = [-60.0, 0.0, 60.0]
        ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01
    assert _load(p) is None


@pytest.mark.parametrize("table", ["rrtmgp-gas-sw-g112.nc", "rrtmgp-gas-sw-g224.nc"])
def test_band_limit_table_equals_rrtmgp_files(table):
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        _default_data_path,
    )
    from legoesm.forcing.external import _RRTMGP_SW_BAND_LIMITS_CM1
    from legoesm.forcing.external import _open_forcing_dataset
    ds = _open_forcing_dataset(_default_data_path(table))
    lims = np.asarray(ds["bnd_limits_wavenumber"].values)
    ds.close()
    np.testing.assert_array_equal(lims, np.asarray(_RRTMGP_SW_BAND_LIMITS_CM1))


@pytest.mark.skipif(not os.path.exists(_PROD_FILE), reason="production file not staged")
def test_production_file_maps_and_sulfate_scatters():
    _, _, _, mom, _, band_wn = _load(_PROD_FILE)
    assert band_wn.shape == (14, 2)
    tau = mom[..., 0, :].astype(np.float64).sum(axis=2)   # (t, lat, band)
    ssa = mom[..., 1, :].astype(np.float64).sum(axis=2) / tau
    # Visible bands (ascending index 9-11 = 12850-29000 cm-1): non-absorbing.
    assert ssa[..., 9:12].min() > 0.999


def _columns(ncol=2, nlev=10):
    T = jnp.broadcast_to(jnp.linspace(210.0, 290.0, nlev)[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to(jnp.geomspace(5.0e2, 1.0e5, nlev + 1)[None, :],
                              (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    return dict(T=T, p_full=p_full, p_half=p_half,
                sfc_temperature=jnp.full((ncol,), 295.0),
                q_v=jnp.full((ncol, nlev), 2.0e-3),
                cos_zenith=jnp.array([0.6, 0.3])[:ncol],
                aerosol_optical_depth=jnp.full((ncol, nlev), 0.01))


def _volc_moments(kw, tau_b, ssa_b, g_b, layers=slice(1, 4)):
    ncol, nlev = kw["T"].shape
    tau = jnp.zeros((ncol, nlev, 14)).at[:, layers, :].set(jnp.asarray(tau_b))
    return jnp.stack([tau, tau * jnp.asarray(ssa_b),
                      tau * jnp.asarray(ssa_b) * jnp.asarray(g_b)], axis=2)


def _solver():
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    cfg = RRTMGPConfig()
    return RRTMGP.from_legoesm_config(cfg), cfg


def test_grey_moments_reproduce_grey_path():
    """Volcanic tau with the grey ssa/g == the same tau in aerosol_od."""
    solver, cfg = _solver()
    kw = _columns()
    volc = _volc_moments(kw, 0.05, cfg.aerosol_ssa, cfg.aerosol_g)
    a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
    kw_grey = {**kw, "aerosol_optical_depth":
               kw["aerosol_optical_depth"] + volc[:, :, 0, 0]}
    b = solver.solve_columns(**kw_grey)
    for f in ("sw_flux_up", "sw_flux_down", "sw_heating_rate"):
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=1e-10,
                                   atol=1e-10)


def test_file_optics_change_sw_and_close_budget():
    """Non-absorbing volcanic sulfate (ssa 1) absorbs less SW and reflects
    more than the same optical depth at the grey ssa 0.93; both close the SW
    column budget (TOA net - surface net = sum of layer heating * cp dp / g)."""
    from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rc
    solver, cfg = _solver()
    kw = _columns()
    grey = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=_volc_moments(
        kw, 0.1, cfg.aerosol_ssa, cfg.aerosol_g))
    file_ = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=_volc_moments(
        kw, 0.1, 1.0, 0.72))
    dp = kw["p_half"][:, 1:] - kw["p_half"][:, :-1]

    def absorbed(o):
        net = o.sw_flux_down - o.sw_flux_up
        return net[:, 0] - net[:, -1]

    for o in (grey, file_):
        col = jnp.sum(o.sw_heating_rate * dp, axis=1) * rc.CP_D / rc.G
        incident = o.sw_flux_down[:, 0]
        assert np.all(np.abs(np.asarray(absorbed(o) - col))
                      <= 1e-10 * np.asarray(incident) + 1e-10)
    assert np.all(np.asarray(absorbed(grey) - absorbed(file_)) > 0.5)
    assert np.all(np.asarray(file_.sw_flux_up[:, 0] - grey.sw_flux_up[:, 0]) > 0.5)


def test_jit_parity_and_gradient():
    solver, _ = _solver()
    kw = _columns()

    def toa_up(scale):
        volc = _volc_moments(kw, 0.08 * scale, 0.99, 0.7)
        return jnp.sum(solver.solve_columns(
            **kw, aerosol_volcanic_moments_sw=volc).sw_flux_up[:, 0])

    eager = toa_up(1.0)
    np.testing.assert_allclose(jax.jit(toa_up)(1.0), eager, rtol=1e-12)
    g = jax.grad(toa_up)(1.0)
    assert np.isfinite(g) and g > 0.0
    fd = (toa_up(1.0 + 1e-5) - toa_up(1.0 - 1e-5)) / 2e-5
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_wrong_band_count_raises():
    solver, _ = _solver()
    kw = _columns()
    with pytest.raises(ValueError, match="trailing shape"):
        solver.solve_columns(**kw, aerosol_volcanic_moments_sw=jnp.zeros(
            kw["T"].shape + (3, 13)))


@pytest.mark.parametrize("builder", ["_build_gray_radiation_fn",
                                     "_build_none_radiation_fn"])
def test_non_rrtmgp_radiation_refuses(builder):
    from legoesm.driver import physics_pipeline as pp
    from legoesm.driver.config import ExperimentConfig
    fn, _ = getattr(pp, builder)(ExperimentConfig(radiation="gray"))
    kw = _columns()
    ncol, nlev = kw["T"].shape
    args = (kw["T"], kw["p_full"], kw["p_half"], kw["q_v"],
            kw["sfc_temperature"], jnp.zeros(ncol), jnp.zeros(ncol), 180.0,
            43200.0, jnp.full((ncol,), 0.1), jnp.ones(ncol),
            jnp.zeros((ncol, nlev)), jnp.zeros((ncol, nlev)), jnp.zeros(0))
    with pytest.raises(ValueError, match="volcanic SW"):
        fn(*args, aerosol_sw_volc_col=jnp.zeros((ncol, nlev, 3, 14)))


def test_probe_runs_on_synthetic_file(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe", _REPO / "scripts/validate/probe_volcanic_sw_optics.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.main(volc_sw_file(tmp_path / "v.nc", omega=np.full(14, 1.0)), 1979, 1)
    out = capsys.readouterr().out
    assert "file-grey" in out and "ssa 1.0000" in out
"""Probe: SW effect of giving CMIP6 volcanic aerosol its own per-band optics.

One clear-sky equatorial column (CAM6 L32, p_s 1000 hPa, US-Std-like T,
no tropospheric aerosol), mid-month of a given year/month, RRTMGP full
tables.  Arms, same column / same placement / same tropopause weight:
  grey: OLD path -- ext_sun collapsed solar-weighted (5778 K Planck) to one
        gray tau, solved with the grey aerosol ssa/g (RRTMGPConfig 0.93/0.70);
  file: NEW path -- per-band (tau, tau*ssa, tau*ssa*g) from the file.
Prints TOA upward SW, surface downward SW and SW absorbed above 100 hPa
[W/m2] at cos(zenith) = 0.5 and the daily-mean-equivalent (x insolation
weight not applied; instantaneous).  Usage:
  python scripts/validate/probe_volcanic_sw_optics.py FILE YEAR MONTH
"""
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main(path, year, month):
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        RRTMGP, gas_table_files)
    from legoesm.forcing.external import (
        AerosolConfig, _SOLAR_PLANCK_TEMP_K, _planck_band_weights,
        get_aerosol_sw_volcanic_optics_at_time)
    from legoesm.forcing.surface_utils import (
        place_stratospheric_aod_profile_to_layers, stratospheric_layer_weight)
    from legoesm.grids.vertical import make_cam6_l32_levels

    sig = make_cam6_l32_levels()
    p_s = jnp.array([1.0e5])
    p_half = jnp.asarray(sig.pressure_at_half(p_s)).reshape(1, -1)
    p_full = jnp.asarray(sig.pressure_at_full(p_s)).reshape(1, -1)
    T = jnp.maximum(300.0 * (p_full / 1.0e5) ** 0.19, 200.0)
    q_v = jnp.clip(0.018 * (p_full / 1.0e5) ** 3, 3.0e-6, None)
    lat = jnp.array([0.0])
    day = 365.0 * (year - 1979) + 30.4 * (month - 1) + 15.0
    acfg = AerosolConfig(enabled=True, volcanic_enabled=True,
                         volcanic_path=path, start_year=1979)
    mom, p_edges, band_wn = get_aerosol_sw_volcanic_optics_at_time(acfg, day, lat)
    w = stratospheric_layer_weight(p_half, lat, p_full, T)
    nsrc = mom.shape[1]
    flat = jnp.moveaxis(jnp.asarray(mom, jnp.float64).reshape(1, nsrc, 42), -1, 0)
    placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
                      in_axes=(0, None, None))(flat, jnp.asarray(p_edges), p_half) * w
    volc = jnp.moveaxis(placed, 0, -1).reshape(1, -1, 3, 14)
    wl = 1.0e4 / band_wn                       # cm-1 -> um, (14, 2)
    wts = _planck_band_weights(wl[:, 1], wl[:, 0], _SOLAR_PLANCK_TEMP_K)
    tau_gray = jnp.tensordot(volc[:, :, 0, :], jnp.asarray(wts / wts.sum()),
                             axes=(2, 0))

    lw, sw = gas_table_files("full")
    cfg = RRTMGPConfig(lw_gas_file=lw, sw_gas_file=sw)
    solver = RRTMGP.from_legoesm_config(cfg)
    kw = dict(T=T, p_full=p_full, p_half=p_half,
              sfc_temperature=jnp.array([300.0]), q_v=q_v,
              cos_zenith=jnp.array([0.5]), sfc_albedo=jnp.array([0.07]))
    arms = {
        "none": solver.solve_columns(**kw),
        "grey": solver.solve_columns(**kw, aerosol_optical_depth=tau_gray),
        "file": solver.solve_columns(**kw, aerosol_optical_depth=jnp.zeros_like(tau_gray),
                                     aerosol_volcanic_moments_sw=volc),
    }
    k100 = int(jnp.argmin(jnp.abs(p_half[0] - 1.0e4)))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"file={path} {year}-{month:02d} lat=0 cosz=0.5 code={sha}")
    tau_vis = float(volc[0, :, 0, 10].sum())
    ssa_vis = float(volc[0, :, 1, 10].sum() / max(tau_vis, 1e-30))
    print(f"volcanic column tau 16000-22650 cm-1 {tau_vis:.4f} ssa {ssa_vis:.4f};"
          f" gray tau {float(tau_gray.sum()):.4f}")
    for k, o in arms.items():
        net = o.sw_flux_down[0] - o.sw_flux_up[0]
        print(f"{k:5s} toa_up {float(o.sw_flux_up[0, 0]):8.3f}  sfc_down "
              f"{float(o.sw_flux_down[0, -1]):8.3f}  abs_above_100hPa "
              f"{float(net[0] - net[k100]):7.3f}  abs_column "
              f"{float(net[0] - net[-1]):8.3f}")
    g, f = arms["grey"], arms["file"]
    print(f"file-grey: toa_up {float(f.sw_flux_up[0,0]-g.sw_flux_up[0,0]):+.3f} "
          f"sfc_down {float(f.sw_flux_down[0,-1]-g.sw_flux_down[0,-1]):+.3f} W/m2")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

exec
/usr/bin/bash -lc "rg -n 'aerosol_od|aerosol_sw_volc|aerosol_lw_od' packages/coupler/legoesm/driver/model_driver.py packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py packages/ml/legoesm/training/scale_build.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
packages/ml/legoesm/training/scale_build.py:1582:    """``aerosol_od`` / ``aerosol_ccn_aod`` / ``aerosol_lw_od`` for one sample:
packages/ml/legoesm/training/scale_build.py:1612:    out = {"aerosol_od": od, "aerosol_ccn_aod": ccn, "aerosol_lw_od": lw}
packages/ml/legoesm/training/scale_build.py:1615:        out["aerosol_sw_volc"] = jnp.mean(jnp.stack(volc), axis=0)
packages/ml/legoesm/training/scale_build.py:1856:            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:978:    aerosol_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:979:    aerosol_lw_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:980:    aerosol_sw_volc: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1037:    aerosol_od : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1042:    aerosol_lw_od : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1049:    aerosol_sw_volc : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1228:        aerosol_optical_depth=aerosol_od,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1229:        aerosol_absorption_optical_depth_lw=aerosol_lw_od,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1233:    if aerosol_sw_volc is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1234:        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1695:    (2009) AOD->CCN diagnostic (``forcing["aerosol_od"]``) so the radiation
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1814:        #   aerosol_od  : (ncol, nlev) per-layer aerosol optical depth
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1820:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1822:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1825:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1969:            # missing-aerosol_od raise is proxy-only, mirroring the
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1979:                    "nc_from_aerosol=True but no 'aerosol_od' was passed to "
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1988:                aerosol_od=(None if _aer_ext is None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2146:            aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2147:            aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2148:            aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2207:                aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2208:                aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2209:                aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3271:        # (see _make_hydrostatic_radiation): o3_vmr / aerosol_od are
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3277:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3279:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3282:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3445:            aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3446:            aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3447:            aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3501:    # o3_vmr / aerosol_od / ghg_vmr) — the spectral combined dispatcher
packages/coupler/legoesm/driver/model_driver.py:5850:        self._aerosol_lw_od = None
packages/coupler/legoesm/driver/model_driver.py:5851:        self._aerosol_sw_volc = None
packages/coupler/legoesm/driver/model_driver.py:6159:        (aerosol_od, _ccn_aod, self._aerosol_lw_od,
packages/coupler/legoesm/driver/model_driver.py:6160:         self._aerosol_sw_volc) = aerosol_layer_fields(
packages/coupler/legoesm/driver/model_driver.py:6168:        return o3_vmr, aerosol_od, ghg_vmr
packages/coupler/legoesm/driver/model_driver.py:13395:        Aerosol-CCN specified-Nc fill needs ``forcing["aerosol_od"]`` in the
packages/coupler/legoesm/driver/model_driver.py:13399:        never thread aerosol_od and the microphysics fill would raise its
packages/coupler/legoesm/driver/model_driver.py:13628:        # prescribed ``forcing["aerosol_od"]`` and the warm-rain closures match
packages/coupler/legoesm/driver/model_driver.py:13632:        # and by the factories' fail-fast (no aerosol_od => raise).
packages/coupler/legoesm/driver/model_driver.py:15543:        # ``forcing["o3_vmr"|"aerosol_od"|"ghg_vmr"]`` and forwards them
packages/coupler/legoesm/driver/model_driver.py:16476:                        _forcing_daily["aerosol_od"] = _aer
packages/coupler/legoesm/driver/model_driver.py:16479:                        # ``forcing.get("aerosol_lw_od")`` is None ⇒ RRTMGP
packages/coupler/legoesm/driver/model_driver.py:16481:                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
packages/coupler/legoesm/driver/model_driver.py:16483:                            _forcing_daily["aerosol_lw_od"] = _aer_lw
packages/coupler/legoesm/driver/model_driver.py:16485:                        # absent; the volcanic SW is then in aerosol_od).
packages/coupler/legoesm/driver/model_driver.py:16486:                        if getattr(self, "_aerosol_sw_volc", None) is not None:
packages/coupler/legoesm/driver/model_driver.py:16487:                            _forcing_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
packages/coupler/legoesm/driver/model_driver.py:16488:                        # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
packages/coupler/legoesm/driver/model_driver.py:18466:                        _ext_daily = {"o3_vmr": _o3, "aerosol_od": _aer}
packages/coupler/legoesm/driver/model_driver.py:18469:                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
packages/coupler/legoesm/driver/model_driver.py:18471:                            _ext_daily["aerosol_lw_od"] = _aer_lw
packages/coupler/legoesm/driver/model_driver.py:18472:                        if getattr(self, "_aerosol_sw_volc", None) is not None:
packages/coupler/legoesm/driver/model_driver.py:18473:                            _ext_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
packages/coupler/legoesm/driver/model_driver.py:18474:                        # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
packages/coupler/legoesm/driver/model_driver.py:19409:        o3_vmr, aerosol_od = ctx["o3_vmr"], ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:19432:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:19445:                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:19446:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:19447:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:20017:        o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:20052:            "o3_vmr": o3_vmr, "aerosol_od": aerosol_od, "ghg_vmr": ghg_vmr,
packages/coupler/legoesm/driver/model_driver.py:20284:            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
packages/coupler/legoesm/driver/model_driver.py:20366:        aerosol_od = ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:20614:            # (s_0/ghg_vmr/o3_vmr/aerosol_od/solar_weights) are traced arrays of
packages/coupler/legoesm/driver/model_driver.py:20635:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:20661:                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:20662:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:20663:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:20927:                        aerosol_od=_g(forcing.aerosol_od),
packages/coupler/legoesm/driver/model_driver.py:21426:        aerosol_od = ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:21503:            o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:21548:                o3_vmr, aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:21552:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:21553:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:21666:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:21694:                    o3_vmr, aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:21698:                    aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:21699:                    aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),

exec
/usr/bin/bash -lc "sed -n '20900,20950p' packages/coupler/legoesm/driver/model_driver.py; sed -n '1830,1875p' packages/ml/legoesm/training/scale_build.py; sed -n '20265,20305p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'chunk|PER_COLUMN|solve_columns' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; sed -n '1080,1160p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; rg -n 'def _interp_zonal_to_grid' packages/tools/legoesm/forcing/external.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
                    _h(held_lw_up_toa))

            # BUG-B state dump: when the held TOA SW goes super-physical, save the
            # full radiation-input state so it can be replayed offline through
            # compute_radiation_core (which produced this garbage) and bisected.
            _dump_path = os.environ.get("LEGOESM_DUMP_RAD")
            if _dump_path and not getattr(self, "_rad_dumped", False):
                import numpy as _np
                if float(_np.max(_np.asarray(held_sw_down_toa))) > 600.0:
                    self._rad_dumped = True
                    _tr = self.tracers if isinstance(self.tracers, dict) else {}
                    _g = lambda x: (None if x is None else _np.asarray(x))
                    _np.savez_compressed(
                        _dump_path,
                        T=_g(self.state.T.data), p_s=_g(self.state.p_s.data),
                        u=_g(self.cell_centre_state().u.data),
                        v=_g(self.cell_centre_state().v.data),
                        q_v=_g(self.q_v), q_c=_g(self.q_c), q_r=_g(self.q_r),
                        q_i=_g(_tr.get("q_i")), N_c=_g(_tr.get("N_c")),
                        N_i=_g(_tr.get("N_i")),
                        sst=_g(sst), sic=_g(sic),
                        lat=_g(_seg_lat), lon=_g(_seg_lon),
                        day_of_year=_np.asarray(forcing.day_of_year),
                        seconds_of_day=_np.asarray(forcing.seconds_of_day),
                        s_0=_np.asarray(forcing.s_0),
                        solar_weights=_g(forcing.solar_weights),
                        o3_vmr=_g(forcing.o3_vmr),
                        aerosol_od=_g(forcing.aerosol_od),
                        T_land=_g(getattr(carry, "T_land", None)),
                        held_sw_down_toa=_g(held_sw_down_toa),
                        held_sw_up_toa=_g(held_sw_up_toa),
                        held_lw_up_toa=_g(held_lw_up_toa),
                        day=_np.asarray(day),
                    )
                    logger.info("BUG-B: dumped corrupt-radiation state to %s "
                                "at day %.1f (sw_down_toa max=%.1f)",
                                _dump_path, day,
                                float(_np.max(_np.asarray(held_sw_down_toa))))

            # Keep carry auxiliary fields for checkpoint persistence and coupling
            self._carry_aux = {
                "held_dT_rad": held_dT_rad,
                "held_sw_net_sfc": held_sw_net_sfc,
                "held_lw_net_sfc": held_lw_net_sfc,
                "held_sw_up_toa": held_sw_up_toa,
                "held_lw_up_toa": held_lw_up_toa,
                "held_sw_down_toa": held_sw_down_toa,
                # Clear-sky held fields (#843) persisted for checkpoint restart.
                "held_sw_up_toa_clr": held_sw_up_toa_clr,
                "held_lw_up_toa_clr": held_lw_up_toa_clr,
                "conv_prog": conv_prog,
                                  convection=config.convection)
        target = era5_to_latlon_carry(s_tg, grid, sigma, target_phis=tp,
                                      microphysics=_micro,
                                      ice_skin_prognostic=_skin,
                                      convection=config.convection)
        sst = regrid_2d_to_gaussian(sst_src.sst, sst_src.lat, sst_src.lon, grid)
        # Prescribed surface planes the slice carries (spectral names),
        # regridded exactly like sst, then renamed to the driver-path
        # override names for pack_forcing.
        planes = prescribed_surface_planes(sst_src, grid)
        if "sfc_shf" in planes and not _WARNED_ERA5_FLUX_HOLD_FIXED:
            _WARNED_ERA5_FLUX_HOLD_FIXED = True
            logger.warning(
                "era5_surface_fluxes: hourly interpolation of the ERA5 surface "
                "fluxes is not implemented — every flux plane is HELD FIXED at "
                "its sample-start value for the whole training window "
                "(owner decision 2026-09-15).")
        plane_kwargs = {
            _spec_to_driver.get(_key, _key): _plane
            for _key, _plane in planes.items()}
        doy = float((times[i_ic] - np.datetime64(f"{year}-01-01"))
                    / np.timedelta64(1, "D"))
        forcing = pack_forcing(
            sst=sst, sic=jnp.zeros_like(sst),
            day_of_year=jnp.asarray(doy), seconds_of_day=jnp.asarray(0.0),
            solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
            o3_top_vmr=getattr(driver, "_o3_top_vmr", None),
            **plane_kwargs)
        return (ic, target, forcing)

    mine = list(_sharded_indices(
        cfg, yml, times, snaps_per_day, stride, rank, nproc))
    src = (ds, cloud_ds, flux_ds, int)
    local = _wb_local_stores(cfg, yml, era5_cfg, sst_cfg, times,
                             snaps_per_day, stride, rank, nproc)
    if local is not None and mine:
        _, i0, t0 = mine[0]
        _assert_same_sample(
            _sample_inputs(era5_cfg, sst_cfg, i0, t0, *src),
            _sample_inputs(era5_cfg, sst_cfg, i0, t0, *local))
        src = local
    samples = []
    for year, i_ic, i_tg in mine:
        sample = build(year, i_ic, i_tg, *src)
        samples.append(_sample_to_host(sample) if host_resident else sample)
            land_ml=self._land_ml_state,
            tke=ctx["tke"], qke=ctx["qke"], gwd_spectrum=ctx["gwd_spectrum"],
            **(
                {k: self.tracers.get(k)
                 for k in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")}
                if isinstance(self.tracers, dict) else {}
            ),
            **tiedtke_lag_seeds(cfg.convection, self.q_v, self.state.p_s),
        )

        if self.get_sst_sic is not None:
            sst, sic = self.get_sst_sic(day)
        else:
            sst = jnp.full(ctx["shape_2d"], 290.0, dtype=_sd)
            sic = jnp.zeros(ctx["shape_2d"], dtype=_sd)
        forcing = pack_forcing(
            sst=jnp.asarray(sst), sic=jnp.asarray(sic),
            day_of_year=day_of_year, seconds_of_day=seconds_of_day,
            solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
            aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
            o3_top_vmr=getattr(self, "_o3_top_vmr", None),
            ghg_vmr=ctx["ghg_vmr"],
            # NOTE: transient cover is deliberately NOT injected here.  The training
            # segment returns run_segment.raw (un-jitted), so the land calibration
            # differentiates w.r.t. the ``pipe.land_ml_params`` ATTRIBUTE (read fresh
            # each trace — see test_build_training_segment_land_gradient); a non-None
            # land_ml_params forcing leaf would SHADOW that attribute and break the
            # gradient.  The un-jitted path never had the closure-bake problem the
            # traced arg fixes (that is a production-only, jitted _run_compiled fix).
        )
        # SPMD: commit grid-shaped forcing leaves to the state's sharding
        # (no-op single-device / mpi4jax-distributed) — see shard_forcing.
        from legoesm.driver.compiled_segments import shard_forcing
        forcing = shard_forcing(forcing, self._device_config)
        return run_segment.raw, carry0, forcing

    # ==================================================================
    # Compiled segment execution path
    # ==================================================================

42:        "RRTMGP correlated-k radiation solver (solve_columns): builds the gas/"
347:  ``solve_columns(...)``.  The swirl_jatmos-style ``__init__`` and
384:      the *result* of ``solve_columns`` (or its compile-time graph)
395:        are NOT overridable per-call via ``solve_columns(...)`` kwargs
553:      # in ``solve_columns`` carries the actual zenith / albedo /
585:  def solve_columns(
1347:  def solve_columns_chunked(
1350:      column_chunk_size: int,
1359:      """Column-chunked wrapper over :meth:`solve_columns`.
1361:      Splits the leading column axis into ``ncol // column_chunk_size``
1362:      fixed-size blocks and maps :meth:`solve_columns` over them with
1366:      unchunked call, not an approximation.  The point is COMPILE TIME: XLA
1367:      lowers the per-block body ONCE at ``column_chunk_size`` columns, so the
1372:      LEVELS are NOT chunked: the vertical is coupled through the two-stream
1380:      column_chunk_size : int
1382:          disables chunking (a plain :meth:`solve_columns` call).  Must
1386:          :meth:`solve_columns`.
1388:          Any optional :meth:`solve_columns` keyword.  A leaf whose leading
1389:          axis equals ``ncol`` is treated as a per-column array and chunked
1398:          Reassembled over ``ncol`` — identical to :meth:`solve_columns`.
1402:      # chunking-specific check so a disabled call is a TRUE passthrough:
1404:      # solve_columns returns (no reshape overhead, no contract change).
1405:      if (not column_chunk_size or column_chunk_size <= 0
1406:              or ncol <= column_chunk_size):
1407:          return self.solve_columns(
1412:      # --- chunking is ACTIVE from here down ---
1414:      # layout ((n_gpt, ncol, nlev)) carries no leading (n_block, chunk)
1419:              "solve_columns_chunked supports only the RadiationOutput path; "
1421:              "per-g-point tuple incompatible with column chunking — call "
1422:              "solve_columns directly for the optics-only tracers."
1424:      if ncol % column_chunk_size != 0:
1426:              f"column_chunk_size {column_chunk_size} must divide ncol "
1433:      n_block = ncol // column_chunk_size
1436:      # sfc_albedo_direct) are read at GLOBAL ncol inside solve_columns via
1439:      # are chunked here, so a per-column ``(ncol, ...)`` config fallback would
1440:      # be read at global ncol inside each chunk-local solve → size mismatch
1442:      # pass the field explicitly (which IS chunked); the production AMIP path
1452:                  f"solve_columns_chunked: config fallback '{_sfc_name}' is a "
1455:                  "global ncol inside each chunk block (via "
1456:                  "_resolve_surface_field) → size mismatch under chunking. "
1458:                  "chunked) instead of relying on the per-column config field."
1462:      # (n_block, column_chunk_size, ...).  This row-major reshape is
1470:          return x.reshape(n_block, column_chunk_size, *x.shape[1:])
1481:      # (splitting it → each block sees a short weight vector → solve_columns
1493:      # the chunking buys compile time but no memory.  With it the backward
1495:      # ``column_chunk_size`` columns instead of ``ncol`` -- the AMIP WB arm
1507:      def _one(chunk):
1508:          return self.solve_columns(**chunk, **static)
1510:      out = jax.lax.map(_one, mapped)  # leaves: (n_block, chunk, ...)
1511:      # Merge (n_block, chunk) back to ncol — the exact inverse of _split.
            p_half=p_half,
            sfc_temperature=sfc_temperature,
            lat=lat,
            q_v=q_v,
            insolation=insolation,
            config=scheme_config,
            # Honour a per-column surface albedo here too (``None`` ->
            # ``GrayRadiationConfig.sfc_albedo``, byte-identical).  This used
            # to be dropped silently, so a tile-blended land albedo was
            # discarded on every gray hydrostatic/MPAS run — the same class of
            # defect as the RRTMGP path below.
            sfc_albedo=sfc_albedo_override,
        )

    # RRTMGP: use actual cos_sza if available (diurnal cycle), else derive
    # from daily-mean insolation using the daytime-effective zenith angle.
    # The eccentricity (a/r)^2 factor scales the SW *flux* (folded into
    # _sw_scale), NOT the optical-path cosine — so divide ``eccf`` out of any
    # cos_sza derived from the (eccf-folded) ``insolation``.  Gated on a static
    # orbit flag so the circular-orbit path is bit-for-bit unchanged.
    _orbit_on = getattr(radiation_config, "orbit", None) is not None
    _sw_scale = None
    # Transient TSI: the caller's ``insolation`` is already scaled by
    # tsi/S_0, but the solver runs at its configured S_0.  So derive the
    # optical-path cosine with ``tsi`` (stays geometric) and put tsi/S_0 on
    # the SW flux below.  ``None`` -> byte-identical to the static-S_0 path.
    if cos_sza is None:
        S_0 = radiation_config.rrtmgp.S_0 if tsi is None else tsi
        if f_day is not None:
            # Use daytime-effective cos(SZA): insol = (a/r)^2·S_0·f_day·<cos>_day
            # so <cos>_day = insol / (eccf·S_0·f_day).  The solver sees the
            # geometric daytime optical path; SW output is rescaled by
            # f_day·eccf afterward to recover daily-mean energy + distance.
            f_day_safe = jnp.maximum(f_day, 1.0e-6)
            cos_sza = jnp.clip(
                insolation / (eccf * S_0 * f_day_safe), 0.0, 1.0,
            )
            _sw_scale = f_day * eccf if _orbit_on else f_day
        else:
            cos_sza = jnp.clip(
                insolation / (eccf * jnp.clip(S_0, 1.0e-6, None)),
                0.0,
                1.0,
            )
            if _orbit_on:
                _sw_scale = jnp.full((cos_sza.shape[0],), eccf,
                                     dtype=cos_sza.dtype)
    elif _orbit_on:
        # Diurnal path: cos_sza is already geometric; apply the distance factor
        # to the SW flux (the solver runs with S_0, not S_0·eccf).
        _sw_scale = jnp.full((cos_sza.shape[0],), eccf, dtype=cos_sza.dtype)
    if tsi is not None:
        _tsi_ratio = jnp.broadcast_to(
            tsi / radiation_config.rrtmgp.S_0, (cos_sza.shape[0],)
        ).astype(cos_sza.dtype)
        _sw_scale = _tsi_ratio if _sw_scale is None else _sw_scale * _tsi_ratio
    # CESM2.1 ocean albedo split: the direct beam sees the open-ocean albedo
    # at THIS call's cosine (the one the solver uses), the diffuse beam keeps
    # the caller's blend (open ocean at OCEAN_ALBEDO_DIFFUSE_CESM).  The blend
    # is linear in the open-ocean fraction, so swapping its ocean term is exact.
    sfc_albedo_direct = None
    if open_ocean_frac is not None:
        _alb_dif = (sfc_albedo_override if sfc_albedo_override is not None
                    else radiation_config.rrtmgp.sfc_albedo)
        sfc_albedo_direct = cesm_direct_surface_albedo(
            _alb_dif, open_ocean_frac, cos_sza)
    q_v_safe = q_v if q_v is not None else jnp.zeros_like(T)

    # Compute ozone VMR based on config — unless the caller supplied a
    # pre-computed external (CMIP6 file) ozone column, which wins.
    if o3_vmr_override is not None:
        o3_vmr = o3_vmr_override
    else:
        o3_vmr = _compute_ozone_vmr(
            p_full, lat, radiation_config.ozone,
            T=T, lon=lon, ml_ozone_coefs=ml_ozone_coefs, p_half=p_half,
        )

    # Compute cloud properties if cloud scheme is active.
    cloud_kwargs = {}
    # Bound unconditionally: the subcolumn-overlap block below reads both, and
1600:def _interp_zonal_to_grid(lat_src: np.ndarray, field: np.ndarray,

exec
/usr/bin/bash -lc "sed -n '340,520p' CLAUDE.md; sed -n '65,145p' tests/unit/test_volcanic_sw_optics.py; git diff HEAD -- tests/unit/test_aerosol_band_ccn.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:

**2026-11-03 REFINEMENT — DETAIL DECISIONS GO TO A VOTE, NOT THE USER.** User,
repeated three times in one day: *"For detailed questions do not ask me but
check with him [GLM] and codex and use the majority of votes. Just ask me big
picture / important questions such as on design or main directions."*
- **Detail side decision** (names, tolerances, guards, edge cases, file
  locations, minor scope trims, which diagnostic next): author's pick + GLM vote
  + codex vote, majority of three wins. Never silent: record each voter's pick
  in the commit, and mark the item VOTED on the end-of-task list (VOTED counts
  as discharged, like ASKED).
- **Still ASK the user:** model design, main direction, production defaults and
  config changes (RULE 3), which experiments/compute to launch, data source,
  what is trained or scored, anything changing comparability of results.
- Unsure which class → big picture → ask.
- Subagent briefs must state this rule.

## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
User directive 2026-08-21. Any time two runs / configs / templates / lanes are
compared, give a table: one row per field that DIFFERS, one column per
configuration, named in the header. Omit fields that agree — it is a diff, not
an inventory. Prose comparisons hide the one line that matters; the land
coupling week was lost to a surface-scheme mismatch a five-row table exposes at
a glance. This is also the required answer to "why did A work and B not".
If the table has more than one row, no single output difference is attributable
yet — say so.

## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
User directive 2026-08-21, after a week was spent coupling the WRONG land model
and then chasing two wrong causes in a row: *"we focus now on accuracy and we
must check with adversarial reviews our claims, especially when they imply
coding decisions."*

The existing dual-review rule covers DIFFS. This one covers the CLAIM that
justifies the diff, and it comes FIRST:

- **State the claim, then get it reviewed, THEN write code.** If a sentence of
  the form "X causes Y, so I will change Z" is about to become an edit, it goes
  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
  only after the code is written is a claim defended, not tested.
- **Say which parts are MEASURED and which are READ OFF THE CODE.** A scaling
  argument, a mechanism, a "this is the only consumer" — each is a separate
  claim with its own evidence.
- **Quote the number's PROVENANCE next to it.** "62 K leaf temperature" from a
  synthetic sweep that deliberately includes unreachable forcings is not the
  same statement as 62 K in a run. Report the reachable-subset number and the
  full-box number separately, and say which is which. A frightening number with
  no provenance wastes the reader's attention and can send the work sideways.
- **A number a well-posed scheme could never produce is a BUG IN THE
  INSTRUMENT until proven otherwise.** Check the probe before reporting the
  physics.
- **When a working configuration exists, diff against it FIRST.** See
  [[reuse-the-working-path-dont-rebuild]]: the offline land driver was healthy
  the whole time; the coupled lane simply called a different parameter provider.
  The first question on any coupling defect is "what does the working lane do
  differently", not "what could be wrong with the physics".
- **Reviewer disagreement is the signal to MEASURE, not to average.** Name the
  discriminating check and run it; it is usually a grep.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep installed source in site-packages, read actual signature.
No reconstruct from memory. Applies to argument names, argument order,
keyword-only args, return arity.

Symbol under `jax.experimental.*` → assume API changed since training data.
Verify or search. No guessing module paths.

### Report uncertainty explicitly
End any non-trivial code response with `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

Empty block = valid answer. Missing block = not.

My premise wrong — bug misdiagnosed, or thing asked for won't work — say so
before writing code.

### Diagnose before patching
Something fail: state candidate causes and how to discriminate between them,
then test. No going straight to fix. No agreeing with cause I suggested unless
evidence support it.


**Applies to THIS repo's own API too** — large enough that memory unreliable.
FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)` (really
`reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost full probe round-trip.
Worse, `RRTMGPConfig()` default `include_clouds=False`, so offline harness
omitting it return CLEAR-SKY fluxes and EVERY cloud gradient exactly 0.0 —
silently wrong number, not error. Read NamedTuple `_fields` / `_field_defaults`
before constructing config.

User's premise wrong — bug misdiagnosed, or thing asked for won't work — say so
BEFORE writing code.
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)

def test_loader_sorts_bands_and_builds_moments(tmp_path):
    omega = np.linspace(0.2, 1.0, 14)
    g = np.linspace(0.3, 0.8, 14)
    _, _, lat, mom, p_edges, band_wn = _load(
        volc_sw_file(tmp_path / "v.nc", omega=omega, g=g))
    assert mom.shape == (12, _LAT.size, 4, 3, 14) and mom.dtype == np.float32
    assert np.all(np.diff(p_edges) > 0)           # top first
    assert np.all(np.diff(band_wn[:, 0]) > 0)     # ascending wavenumber
    srt = np.argsort(1.0e4 / np.maximum(_WL1, _WL2))
    assert srt[0] == 13                           # 820-2600 cm-1 band first
    tau = 0.01 * 2.0                               # ext [1/km] * dz 2 km
    np.testing.assert_allclose(mom[0, 0, 0, 0], tau, rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 1], tau * omega[srt], rtol=1e-6)
    np.testing.assert_allclose(mom[0, 0, 0, 2], tau * omega[srt] * g[srt], rtol=1e-6)


@pytest.mark.parametrize("kw, match", [
    ({"drop": ("omega_sun",)}, "omega_sun"),
    ({"drop": ("g_sun",)}, "g_sun"),
    ({"g": np.full(14, -0.1)}, "Negative"),
    ({"wl": (_WL1[:13], _WL2[:13])}, "map 1:1"),
    ({"wl": (_WL1 * 1.05, _WL2 * 1.05)}, "map 1:1"),
])
def test_loader_refuses(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        _load(volc_sw_file(tmp_path / "v.nc", **kw))


def test_legacy_column_file_is_not_an_optics_source(tmp_path):
    p = str(tmp_path / "legacy.nc")
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", 12)
        ds.createDimension("lat", 3)
        ds.createVariable("lat", "f8", ("lat",))[:] = [-60.0, 0.0, 60.0]
        ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01
    assert _load(p) is None


@pytest.mark.parametrize("table", ["rrtmgp-gas-sw-g112.nc", "rrtmgp-gas-sw-g224.nc"])
def test_band_limit_table_equals_rrtmgp_files(table):
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        _default_data_path,
    )
    from legoesm.forcing.external import _RRTMGP_SW_BAND_LIMITS_CM1
    from legoesm.forcing.external import _open_forcing_dataset
    ds = _open_forcing_dataset(_default_data_path(table))
    lims = np.asarray(ds["bnd_limits_wavenumber"].values)
    ds.close()
    np.testing.assert_array_equal(lims, np.asarray(_RRTMGP_SW_BAND_LIMITS_CM1))


@pytest.mark.skipif(not os.path.exists(_PROD_FILE), reason="production file not staged")
def test_production_file_maps_and_sulfate_scatters():
    _, _, _, mom, _, band_wn = _load(_PROD_FILE)
    assert band_wn.shape == (14, 2)
    tau = mom[..., 0, :].astype(np.float64).sum(axis=2)   # (t, lat, band)
    ssa = mom[..., 1, :].astype(np.float64).sum(axis=2) / tau
    # Visible bands (ascending index 9-11 = 12850-29000 cm-1): non-absorbing.
    assert ssa[..., 9:12].min() > 0.999


def _columns(ncol=2, nlev=10):
    T = jnp.broadcast_to(jnp.linspace(210.0, 290.0, nlev)[None, :], (ncol, nlev))
    p_half = jnp.broadcast_to(jnp.geomspace(5.0e2, 1.0e5, nlev + 1)[None, :],
                              (ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    return dict(T=T, p_full=p_full, p_half=p_half,
                sfc_temperature=jnp.full((ncol,), 295.0),
                q_v=jnp.full((ncol, nlev), 2.0e-3),
                cos_zenith=jnp.array([0.6, 0.3])[:ncol],
                aerosol_optical_depth=jnp.full((ncol, nlev), 0.01))


def _volc_moments(kw, tau_b, ssa_b, g_b, layers=slice(1, 4)):
    ncol, nlev = kw["T"].shape
    tau = jnp.zeros((ncol, nlev, 14)).at[:, layers, :].set(jnp.asarray(tau_b))
    return jnp.stack([tau, tau * jnp.asarray(ssa_b),
                      tau * jnp.asarray(ssa_b) * jnp.asarray(g_b)], axis=2)


diff --git a/tests/unit/test_aerosol_band_ccn.py b/tests/unit/test_aerosol_band_ccn.py
index f7d141cac1..58cd451d91 100644
--- a/tests/unit/test_aerosol_band_ccn.py
+++ b/tests/unit/test_aerosol_band_ccn.py
@@ -2,8 +2,9 @@
 
 - solar-weighted gray / 550 nm band from the file's own ``wl1_sun``/``wl2_sun``
   (order-independent; no edges -> old flat mean);
-- volcanic SW (``ext_sun``) placed at its stratospheric pressure, not spread
-  through the troposphere, and excluded from ``aerosol_ccn_aod``;
+- volcanic SW (``ext_sun`` + own optics) placed at its stratospheric pressure,
+  not spread through the troposphere, kept out of the grey ``aerosol_od`` and
+  excluded from ``aerosol_ccn_aod``;
 - ``aerosol_ccn_aod`` drives the AOD->CCN proxy when present, byte-identical
   sum(aerosol_od) when absent;
 - ``--aerosol-ccn-file`` -> ExperimentConfig -> AerosolConfig.ccn_path -> CCN.
@@ -59,16 +60,10 @@ def _kinne_file(path, band_aod, edges=True, lon=False):
 
 
 def _volc_file(path, ext=0.01, alt=(18.0, 20.0, 22.0, 24.0)):
-    """CMIP6-shaped ``ext_sun(solar_bands, lat, altitude, month)`` [1/km]."""
-    with netCDF4.Dataset(path, "w") as ds:
-        for d, n in (("solar_bands", 3), ("lat", _LAT.size), ("altitude", len(alt)), ("month", 12)):
-            ds.createDimension(d, n)
-        ds.createVariable("lat", "f8", ("lat",))[:] = _LAT
-        ds.createVariable("altitude", "f8", ("altitude",))[:] = np.asarray(alt)
-        ds.createVariable("ext_sun", "f8", ("solar_bands", "lat", "altitude", "month"))[:] = ext
-        ds.createVariable("wl1_sun", "f8", ("solar_bands",))[:] = _WL1
-        ds.createVariable("wl2_sun", "f8", ("solar_bands",))[:] = _WL2
-    return str(path)
+    """CMIP6-shaped volcanic file with its own solar-band optics (14 RRTMG
+    bands, see ``test_volcanic_sw_optics.volc_sw_file``)."""
+    from tests.unit.test_volcanic_sw_optics import volc_sw_file
+    return volc_sw_file(path, ext=ext, alt=alt)
 
 
 def test_solar_weighted_and_visible_band_collapse(tmp_path):
@@ -139,9 +134,14 @@ def test_volcanic_sw_above_tropopause_and_out_of_ccn(tmp_path):
     volc = _volc_file(tmp_path / "v.nc")  # 17-25 km, column ~0.08
     f0, aer0, _ = _driver_forcing(_cfg(aerosol_file=trop))
     f1, aer1, p_full = _driver_forcing(_cfg(aerosol_file=trop, volcanic_aerosol_file=volc))
-    dv = aer1 - aer0
-    np.testing.assert_allclose(dv.sum(-1), 0.01 * 8.0, rtol=1e-6)  # all of it lands
-    assert np.abs(dv[p_full > 3.0e4]).max() < 1e-12  # none below 300 hPa
+    assert f0._aerosol_sw_volc is None
+    # Volcanic SW is carried with its own optics, NOT in the grey field.
+    np.testing.assert_array_equal(aer1, aer0)
+    mom = np.asarray(f1._aerosol_sw_volc)            # (ncol, nlev, 3, 14)
+    assert mom.shape == aer0.shape + (3, 14)
+    np.testing.assert_allclose(mom[..., 0, :].sum(1), 0.01 * 8.0, rtol=1e-6)  # all lands
+    np.testing.assert_allclose(mom[..., 1, :], 0.99 * mom[..., 0, :], rtol=1e-6)
+    assert np.abs(mom[p_full > 3.0e4]).max() < 1e-12  # none below 300 hPa
     # CCN: tropospheric 550 nm column, volcanic excluded.
     np.testing.assert_allclose(np.asarray(f1._aerosol_ccn_aod), 0.20, rtol=1e-12)
     np.testing.assert_array_equal(np.asarray(f1._aerosol_ccn_aod),
@@ -268,19 +268,23 @@ def test_driver_volcanic_cut_at_state_tropopause(tmp_path):
     volc = _volc_file(tmp_path / "v.nc")
     trop = _kinne_file(tmp_path / "t.nc", [0.05, 0.20, 0.40], lon=True)
     cfg = _cfg(aerosol_file=trop, volcanic_aerosol_file=volc)
-    _, aer0, _ = _driver_forcing(_cfg(aerosol_file=trop))
-    _, aer_c, _ = _driver_forcing(cfg)
-    _, aer_w, p_full = _driver_forcing(
+
+    def vtau(f):  # per-layer volcanic tau of one band (all bands equal here)
+        return np.asarray(f._aerosol_sw_volc)[..., 0, 0]
+
+    f_c, _, _ = _driver_forcing(cfg)
+    f_w, _, p_full = _driver_forcing(
         cfg, temp_fn=lambda p: _std_temp(p, z_tp_m=19.0e3, t0=300.0)[0])
-    np.testing.assert_allclose((aer_c - aer0).sum(-1), 0.01 * 8.0, rtol=1e-6)
-    dv = aer_w - aer0
+    np.testing.assert_allclose(vtau(f_c).sum(-1), 0.01 * 8.0, rtol=1e-6)
+    dv = vtau(f_w)
     assert (dv.sum(-1) < 0.99 * 0.08).all() and (dv.sum(-1) > 0.0).all()
     p_tp = _std_temp(1.0e5, z_tp_m=19.0e3, t0=300.0)[1]
     assert np.abs(dv[p_full > 2.0 * p_tp]).max() < 1e-12
     # Explicit ``temp=`` (spectral caller) == state T; off-grid shape -> climatology.
     tp = _std_temp(jnp.asarray(p_full), z_tp_m=19.0e3, t0=300.0)[0]
-    np.testing.assert_array_equal(_driver_forcing(cfg, temp=tp)[1], aer_w)
-    np.testing.assert_array_equal(_driver_forcing(cfg, temp=tp[:, :-1])[1], aer_c)
+    np.testing.assert_array_equal(vtau(_driver_forcing(cfg, temp=tp)[0]), dv)
+    np.testing.assert_array_equal(
+        vtau(_driver_forcing(cfg, temp=tp[:, :-1])[0]), vtau(f_c))
 
 
 def test_owned_temp_selects_owned_faces():

codex
Two training forcing packers appear to omit the volcanic field after preprocessing removes it from the grey AOD. I’m checking those paths, the non-RRTMGP dispatches, and whether the tests can detect a band-order error.
exec
/usr/bin/bash -lc "sed -n '1600,1675p' packages/tools/legoesm/forcing/external.py; sed -n '1055,1090p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; rg -n 'aerosol_od|aerosol_lw_od|aerosol_sw_volc' packages --glob '*.py' | tail -140; ls -l .venv/bin/python; rg -n 'g_point_to_bnd|bnd_limits' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/gas_optics*" in /glade/derecho/scratch/pg2328/wt_volc
 exited 2 in 0ms:
def _interp_zonal_to_grid(lat_src: np.ndarray, field: np.ndarray,
                           lat_grid: jnp.ndarray) -> jnp.ndarray:
    """Interpolate a zonal-mean field to model grid latitudes.

    Parameters
    ----------
    lat_src : (nlat_src,)
        Source latitudes [degrees].  May be ascending (``-90 → +90``)
        or descending (``+89.5 → -89.5``, as in Kinne aerosol files);
        this helper detects a descending axis and flips both ``lat_src``
        and ``field`` along the leading axis before calling
        :func:`numpy.interp` (which requires strictly ascending ``xp``).
        Without this guard ``np.interp`` would silently extrapolate to
        the endpoint and produce a near-constant field at every model
        latitude, not a NaN / error (issue #207 bug 6).
    field : (nlat_src,) or (nlat_src, nlev) or (nlat_src, nlev, nband)
        Zonal-mean field at source latitudes.  Handles arbitrary trailing
        dimensions (e.g., 3D Kinne aerosol files with a band axis, #178).
    lat_grid : jax array, any shape
        Model grid latitudes [radians]. Will be converted to degrees.

    Returns
    -------
    jax array with shape (*lat_grid.shape,) or (*lat_grid.shape, *trailing)
    """
    lat_deg = np.asarray(jnp.degrees(lat_grid)).ravel()

    lat_src = np.asarray(lat_src)
    if lat_src.size > 1 and lat_src[0] > lat_src[-1]:
        # Descending source axis (e.g. Kinne aerosol files): flip so
        # ``np.interp``'s strictly-ascending-xp contract is satisfied.
        lat_src = lat_src[::-1]
        field = field[::-1]

    if field.ndim == 1:
        result = np.interp(lat_deg, lat_src, field)
        return jnp.array(result).reshape(lat_grid.shape)

    # Vectorised lerp: locate the bracketing source latitude once, then
    # gather both endpoints and combine.  Replaces the previous
    # ``for k in range(n_cols): np.interp(...)`` loop, which scaled
    # quadratically on T170+ (n_cols ~ 10⁴) at every external-forcing
    # update.
    nsrc = lat_src.shape[0]
    if nsrc == 1:
        result_flat = np.broadcast_to(
            field.reshape(1, -1), (lat_deg.size, np.prod(field.shape[1:])),
        ).copy()
    else:
        idx = np.clip(np.searchsorted(lat_src, lat_deg) - 1, 0, nsrc - 2)
        x0 = lat_src[idx]
        x1 = lat_src[idx + 1]
        # Avoid division by zero on duplicated source points.
        dx = np.where(x1 > x0, x1 - x0, 1.0)
        w = np.where(x1 > x0, (lat_deg - x0) / dx, 0.0)
        # Clamp at the endpoints (matches np.interp's flat extrapolation).
        w = np.clip(w, 0.0, 1.0)
        # Reshape ``field`` to (nsrc, n_cols) and gather along leading.
        trailing_shape = field.shape[1:]
        n_cols = int(np.prod(trailing_shape))
        field_flat = np.asarray(field).reshape(nsrc, n_cols)
        f0 = field_flat[idx]                    # (n_lat, n_cols)
        f1 = field_flat[idx + 1]                # (n_lat, n_cols)
        result_flat = (1.0 - w[:, None]) * f0 + w[:, None] * f1
    return jnp.array(result_flat).reshape((*lat_grid.shape, *field.shape[1:]))


def _interp_vertical(field_plev: jnp.ndarray, plev_src: np.ndarray,
                      p_target: jnp.ndarray) -> jnp.ndarray:
    """Interpolate vertically in log-pressure space.

    Parameters
    ----------
    field_plev : (..., nlev_src)
        Field on source pressure levels.
    plev_src : (nlev_src,)
        passed through to ``solve_columns``.
    clear_sky_toa : bool
        RRTMGP only: also return the clear-sky TOA upward fluxes from the
        same solve (``RadiationOutput.*_flux_up_toa_clr``).
    """
    if clear_sky_toa and radiation_config.scheme in ("simple_lw", "gray"):
        raise ValueError(
            f"clear_sky_toa is an RRTMGP option; scheme "
            f"{radiation_config.scheme!r} has no clear-sky solve")
    if open_ocean_frac is not None and radiation_config.scheme != "rrtmgp":
        raise ValueError(
            "open_ocean_frac (direct/diffuse ocean albedo split) is wired for "
            f"the rrtmgp solver only; scheme {radiation_config.scheme!r} has a "
            "single surface albedo and would silently drop the direct beam.")
    if radiation_config.scheme == "simple_lw":
        return _simple_lw_radiation(
            T=T, p_half=p_half, q_v=q_v, q_cloud=q_cloud, q_ice=q_ice,
            config=radiation_config.simple_lw,
        )

    if radiation_config.scheme == "gray":
        radiation_fn, scheme_config = _get_radiation_fn(radiation_config)
        return radiation_fn(
            T=T,
            p_full=p_full,
            p_half=p_half,
            sfc_temperature=sfc_temperature,
            lat=lat,
            q_v=q_v,
            insolation=insolation,
            config=scheme_config,
            # Honour a per-column surface albedo here too (``None`` ->
            # ``GrayRadiationConfig.sfc_albedo``, byte-identical).  This used
            # to be dropped silently, so a tile-blended land albedo was
            # discarded on every gray hydrostatic/MPAS run — the same class of
            # defect as the RRTMGP path below.
packages/coupler/legoesm/driver/physics_pipeline.py:3850:                     cloud_fraction=None, aerosol_sw_volc_col=None):
packages/coupler/legoesm/driver/physics_pipeline.py:3851:        del aerosol_lw_od_col, o3_top_vmr_col  # zero-radiation: no-ops
packages/coupler/legoesm/driver/physics_pipeline.py:3852:        if aerosol_sw_volc_col is not None:
packages/coupler/legoesm/driver/physics_pipeline.py:3898:                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
packages/coupler/legoesm/driver/physics_pipeline.py:3902:                     aerosol_lw_od_col=None,
packages/coupler/legoesm/driver/physics_pipeline.py:3908:                     cloud_fraction=None, aerosol_sw_volc_col=None):
packages/coupler/legoesm/driver/physics_pipeline.py:3910:        del aerosol_lw_od_col, o3_top_vmr_col  # gray: no aerosol LW od / ozone
packages/coupler/legoesm/driver/physics_pipeline.py:3911:        if aerosol_sw_volc_col is not None:
packages/coupler/legoesm/driver/physics_pipeline.py:4031:                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
packages/coupler/legoesm/driver/physics_pipeline.py:4035:                     aerosol_lw_od_col=None,
packages/coupler/legoesm/driver/physics_pipeline.py:4042:                     aerosol_sw_volc_col=None):
packages/coupler/legoesm/driver/physics_pipeline.py:4106:            aerosol_optical_depth=aerosol_od_col,
packages/coupler/legoesm/driver/physics_pipeline.py:4107:            aerosol_absorption_optical_depth_lw=aerosol_lw_od_col,
packages/coupler/legoesm/driver/physics_pipeline.py:4108:            **({} if aerosol_sw_volc_col is None
packages/coupler/legoesm/driver/physics_pipeline.py:4109:               else {"aerosol_volcanic_moments_sw": aerosol_sw_volc_col}),
packages/coupler/legoesm/driver/model_driver.py:5850:        self._aerosol_lw_od = None
packages/coupler/legoesm/driver/model_driver.py:5851:        self._aerosol_sw_volc = None
packages/coupler/legoesm/driver/model_driver.py:6159:        (aerosol_od, _ccn_aod, self._aerosol_lw_od,
packages/coupler/legoesm/driver/model_driver.py:6160:         self._aerosol_sw_volc) = aerosol_layer_fields(
packages/coupler/legoesm/driver/model_driver.py:6168:        return o3_vmr, aerosol_od, ghg_vmr
packages/coupler/legoesm/driver/model_driver.py:13395:        Aerosol-CCN specified-Nc fill needs ``forcing["aerosol_od"]`` in the
packages/coupler/legoesm/driver/model_driver.py:13399:        never thread aerosol_od and the microphysics fill would raise its
packages/coupler/legoesm/driver/model_driver.py:13628:        # prescribed ``forcing["aerosol_od"]`` and the warm-rain closures match
packages/coupler/legoesm/driver/model_driver.py:13632:        # and by the factories' fail-fast (no aerosol_od => raise).
packages/coupler/legoesm/driver/model_driver.py:15543:        # ``forcing["o3_vmr"|"aerosol_od"|"ghg_vmr"]`` and forwards them
packages/coupler/legoesm/driver/model_driver.py:16476:                        _forcing_daily["aerosol_od"] = _aer
packages/coupler/legoesm/driver/model_driver.py:16479:                        # ``forcing.get("aerosol_lw_od")`` is None ⇒ RRTMGP
packages/coupler/legoesm/driver/model_driver.py:16481:                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
packages/coupler/legoesm/driver/model_driver.py:16483:                            _forcing_daily["aerosol_lw_od"] = _aer_lw
packages/coupler/legoesm/driver/model_driver.py:16485:                        # absent; the volcanic SW is then in aerosol_od).
packages/coupler/legoesm/driver/model_driver.py:16486:                        if getattr(self, "_aerosol_sw_volc", None) is not None:
packages/coupler/legoesm/driver/model_driver.py:16487:                            _forcing_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
packages/coupler/legoesm/driver/model_driver.py:16488:                        # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
packages/coupler/legoesm/driver/model_driver.py:18466:                        _ext_daily = {"o3_vmr": _o3, "aerosol_od": _aer}
packages/coupler/legoesm/driver/model_driver.py:18469:                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
packages/coupler/legoesm/driver/model_driver.py:18471:                            _ext_daily["aerosol_lw_od"] = _aer_lw
packages/coupler/legoesm/driver/model_driver.py:18472:                        if getattr(self, "_aerosol_sw_volc", None) is not None:
packages/coupler/legoesm/driver/model_driver.py:18473:                            _ext_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
packages/coupler/legoesm/driver/model_driver.py:18474:                        # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
packages/coupler/legoesm/driver/model_driver.py:19409:        o3_vmr, aerosol_od = ctx["o3_vmr"], ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:19432:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:19445:                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:19446:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:19447:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:20017:        o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:20052:            "o3_vmr": o3_vmr, "aerosol_od": aerosol_od, "ghg_vmr": ghg_vmr,
packages/coupler/legoesm/driver/model_driver.py:20284:            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
packages/coupler/legoesm/driver/model_driver.py:20366:        aerosol_od = ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:20614:            # (s_0/ghg_vmr/o3_vmr/aerosol_od/solar_weights) are traced arrays of
packages/coupler/legoesm/driver/model_driver.py:20635:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:20661:                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:20662:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:20663:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:20927:                        aerosol_od=_g(forcing.aerosol_od),
packages/coupler/legoesm/driver/model_driver.py:21426:        aerosol_od = ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:21503:            o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:21548:                o3_vmr, aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:21552:                aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:21553:                aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/coupler/legoesm/driver/model_driver.py:21666:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:21694:                    o3_vmr, aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:21698:                    aerosol_lw_od=getattr(self, "_aerosol_lw_od", None),
packages/coupler/legoesm/driver/model_driver.py:21699:                    aerosol_sw_volc=getattr(self, "_aerosol_sw_volc", None),
packages/atmosphere/legoesm/atmosphere/physics/combined.py:530:    microphysics fill (both diagnose N_c from ``forcing["aerosol_od"]``).
packages/atmosphere/legoesm/atmosphere/physics/combined.py:2228:    # the spectral_pe radiation factory: T_sfc/o3_vmr/aerosol_od/ghg_vmr).
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:589:    # per-step traced ``forcing["aerosol_od"]`` fills ``hydrometeors.N_c``
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:783:            _aer_od = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:792:                    "nc_from_aerosol=True but no 'aerosol_od' was passed to "
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:800:                    aerosol_od=None if _aer_od is None else jnp.asarray(_aer_od),
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:986:    # (it reads ``forcing["aerosol_od"]``).  Left unset otherwise so a run
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:139:    aerosol_od: jnp.ndarray,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:156:    aerosol_od : jnp.ndarray
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:165:        Column AOD ``(ncol,)`` to use instead of the ``aerosol_od`` sum (the
packages/atmosphere/legoesm/atmosphere/physics/microphysics/aerosol_activation.py:173:    aod_col = jnp.sum(aerosol_od, axis=-1) if ccn_aod is None else ccn_aod
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:978:    aerosol_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:979:    aerosol_lw_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:980:    aerosol_sw_volc: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1037:    aerosol_od : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1042:    aerosol_lw_od : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1049:    aerosol_sw_volc : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1228:        aerosol_optical_depth=aerosol_od,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1229:        aerosol_absorption_optical_depth_lw=aerosol_lw_od,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1233:    if aerosol_sw_volc is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1234:        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1695:    (2009) AOD->CCN diagnostic (``forcing["aerosol_od"]``) so the radiation
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1814:        #   aerosol_od  : (ncol, nlev) per-layer aerosol optical depth
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1820:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1822:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1825:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1969:            # missing-aerosol_od raise is proxy-only, mirroring the
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1979:                    "nc_from_aerosol=True but no 'aerosol_od' was passed to "
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1988:                aerosol_od=(None if _aer_ext is None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2146:            aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2147:            aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2148:            aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2207:                aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2208:                aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2209:                aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3271:        # (see _make_hydrostatic_radiation): o3_vmr / aerosol_od are
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3277:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3279:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3282:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3445:            aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3446:            aerosol_lw_od=_aer_lw_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3447:            aerosol_sw_volc=_aer_volc_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3501:    # o3_vmr / aerosol_od / ghg_vmr) — the spectral combined dispatcher
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:550:    aerosol_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:571:    aerosol_od : jnp.ndarray, optional
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:582:        Proxy column AOD ``(ncol,)`` overriding the ``aerosol_od`` sum.
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:590:        if aerosol_od is None:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:592:                "activation scheme 'proxy' requires an aerosol_od field "
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:600:        return specified_nc_field(jnp.asarray(aerosol_od), target_shape, proxy_cfg,
packages/atmosphere/legoesm/atmosphere/physics/neural_physics.py:586:            aerosol_od,
packages/atmosphere/legoesm/atmosphere/physics/neural_physics.py:595:        del solar_weights, o3_vmr, aerosol_od
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:758:      # started threading per-step o3_vmr / aerosol_od (2026-06-10).
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1122:          aerosol_od_3d = jnp.clip(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1126:          aerosol_od_3d = None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1155:          aerosol_od_lw_3d = jnp.clip(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1160:          aerosol_od_lw_3d = None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1190:              aerosol_optical_depth=aerosol_od_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1234:              aerosol_absorption_optical_depth=aerosol_od_lw_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1269:          aerosol_absorption_optical_depth=aerosol_od_lw_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1291:          aerosol_optical_depth=aerosol_od_3d,
packages/tools/legoesm/forcing/external.py:2449:def get_aerosol_sw_volcanic_optics_at_time(config: AerosolConfig, day: float,
packages/tools/legoesm/forcing/surface_utils.py:810:    """Per-layer aerosol forcing for one day: ``(aerosol_od, ccn_aod,
packages/tools/legoesm/forcing/surface_utils.py:811:    aerosol_lw_od, aerosol_sw_volc)``, as the AMIP driver feeds radiation and
packages/tools/legoesm/forcing/surface_utils.py:814:    is off.  ``aerosol_sw_volc`` is the volcanic SW aerosol with its own
packages/tools/legoesm/forcing/surface_utils.py:817:    ``aerosol_od``; None when volcanic forcing is off or the volcanic file is a
packages/tools/legoesm/forcing/surface_utils.py:818:    legacy column ``aod`` file (which then stays in ``aerosol_od`` with the
packages/tools/legoesm/forcing/surface_utils.py:825:        get_aerosol_sw_volcanic_optics_at_time,
packages/tools/legoesm/forcing/surface_utils.py:830:    aerosol_sw_volc = None
packages/tools/legoesm/forcing/surface_utils.py:831:    aerosol_od = jnp.zeros((ncol, nlev), dtype=dtype)
packages/tools/legoesm/forcing/surface_utils.py:837:        # and kept OUT of the grey ``aerosol_od``.
packages/tools/legoesm/forcing/surface_utils.py:838:        volc_opt = get_aerosol_sw_volcanic_optics_at_time(_acfg, day, lat_col)
packages/tools/legoesm/forcing/surface_utils.py:847:            aerosol_sw_volc = jnp.moveaxis(placed, 0, -1).reshape(
packages/tools/legoesm/forcing/surface_utils.py:855:        aerosol_od = distribute_column_aod_to_layers(
packages/tools/legoesm/forcing/surface_utils.py:868:    # nlev) shape as ``aerosol_od``, default zeros so the SegmentForcing
packages/tools/legoesm/forcing/surface_utils.py:877:    aerosol_lw_od = jnp.zeros((ncol, nlev), dtype=dtype)
packages/tools/legoesm/forcing/surface_utils.py:884:            aerosol_lw_od = (place_stratospheric_aod_profile_to_layers(
packages/tools/legoesm/forcing/surface_utils.py:888:    return aerosol_od, ccn_aod, aerosol_lw_od, aerosol_sw_volc
ls: cannot access '.venv/bin/python': No such file or directory
rg: packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/gas_optics*: IO error for operation on packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/gas_optics*: No such file or directory (os error 2)

exec
/usr/bin/bash -lc "sed -n '520,760p' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model near operational. Every rule below mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming file:line as cause
  require showing that line run in configuration under test: print ENCLOSING
  FUNCTION (`awk` nearest `def` above it), confirm active lane/driver call it.
  FAILURE: blamed positivity clamps at `model_driver.py:10923` for century's
  water source; they live in `_run_per_step` while century run `_run_column`,
  which contain no moisture clamp at all. Fix nearly written for lane the run
  never touch. Same class as reading entry point instead of full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of reference's guards/scope conditions you kept and
  which dropped, with reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so
  column-conserving borrow applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing fake "accelerating dry bias" reported before being caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when feature removed.** `inspect.getsource(X)` assertion where X
  is delegating wrapper pass while proving nothing. FAILURE: asserted against
  `MPASPrimitiveEquationModel.step`; floors are in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read output tail.** Exit code without tool's
  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
  kills and timeouts can surface as success. FAILURE: reported regression suite
  green on exit-0 that was actually `Out Of Memory` mid-run. Quote decisive line
  when claiming suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify in code before
  writing it.** Plausible-sounding reason permanently hide real defect. FAILURE:
  classified `convective_buoyancy_death_memory` as "carried in SegmentCarry" (it
  not — leaf `BechtoldConfig.buoyancy_death_memory` exist and nothing map to
  it), asserted `SBMConfig.precip_efficiency` leaf that not exist, and credited
  `micro_substeps` to consumer that read `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print window next to number.** Differing spans = confound, not result.
  FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr"; matched
  windows gave +13.4. Extends existing controlled-comparison rule to derived
  rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV at
  0.1 kg/m2 over 8 days resolve only ~±4.6 kg/m2/yr — no reporting trend inside
  one quantum. Prefer fp64 from model state (checkpoints) over parsed log lines.
  Same class as throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** fp32 mismatch NOT a
  failure until re-run with x64; and *new* failure not yours until reproduced
  with your change stashed. Do both before reporting regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For controlled pair, COPY pinned checkpoint into each arm dir; never
  use `PREV_CKPT_DIR`-style newest-wins pointer while another run advancing.
  FAILURE (twice): arms exited instantly at "Already at/past target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF THEM.**
  Verify resolved paths in run log, not the flag you passed. FAILURE:
  `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016 SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** Code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" naming a limitation, guard, or
  missing feature MUST be re-verified in CURRENT code at point of use before
  repeated as finding — this repo routinely fix things without updating prose.
  Comment name module imposing guard → OPEN THAT MODULE. FAILURES (2026-07-30,
  three in one day): quoted `_run_column` "turbulent surface fluxes are
  intentionally NOT applied" comment and `AMIP.md` Known #3 to claim MPAS has no
  turbulence — MPAS turbulence path exist (`turbulence/integration.py`, Perot
  edge->cell) and run resolve `turbulence=louis` + `surface_bulk_scheme=coare3`;
  and doubted a cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; real clamp to
  lookup-table range is one call deeper in `rrtmgp/optics/cloud_optics.py`. Same
  class as blaming a line without proving its enclosing function runs.
- **BEFORE ATTRIBUTING A BIAS TO A COMPONENT, PROVE THE COMPONENT EXISTS AND
  RUNS IN THE CONFIG UNDER TEST.** ABSENT component and BADLY-TUNED one give
  SAME symptom, and every tuning arm against absent component return null.
  Enumerate, from RESOLVED config and LANE'S code path (not deck's comments):
  the scheme, its PROGNOSTIC STATE, its inputs. No prognostic state -> say so
  out loud; that usually the finding.
  Corollaries, each earned:
  (a) **A `false` on a component switch does not mean "the simpler version" —
  it may mean NOTHING.** Check what lane actually fall back to.
  (b) **A DECK COMMENT NAMING A FALLBACK IS A POINTER, NOT A FACT** (prose rule,
  applied to configs): grep named symbol in that lane's setup.
  (c) **When N different schemes for component A all fail to move a bias, STOP
  TUNING A.** Limiter upstream, or A's inputs wrong.
  (d) **A parameter wrong at BOTH ENDS of a physical range is a MISSING
  DEPENDENCE, not a mistuning** — thing it should depend on not read.
  FAILURE 2026-08-15, most expensive of campaign: MPAS AMIP deck set
  `use_multilayer_land: false` with comment "this deck runs the slab". There is
  NO slab land on MPAS lane (`slab_land_active` wired only on cube/general
  path), so runs had **no land surface model at all** — land T_sfc was nearest
  OCEAN's prescribed SST minus 6.5 K/km, land evaporation fixed 0.6 of
  saturation, no soil, no water store, no runoff, no stomata, no surface energy
  balance. Tropical deserts evaporated 3.5x observed and rainforest 0.73x FROM
  SAME CONSTANT (corollary d, unnoticed); sensible heat ~25 W/m2 on Sahara and
  Amazon alike. Waves 9-11 spent GPU-hours swapping FIVE convection schemes and
  dozen trigger knobs against tropical-land rain deficit no convection scheme
  could ever fix (corollary c, unnoticed). Interactive soil moisture, bucket
  hydrology, stomatal control are DEFAULTS for AMIP case, not options.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from unweighted `p_s` mean; AREA-WEIGHTED mass was invariant at
  983.493 hPa to 6 digits, i.e. defect did not exist. Habits carried from
  quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT IS
  CITED AS THE CAUSE.** X claimed to drive Y → change X by known factor, check Y
  respond as mechanism predict. FAILURE: proposed "damp-to-rest pumps mass
  convergence" (predict ~linear in sponge coefficient); quartering coefficient
  slowed growth only 1.5x, refuting it — fix would have shipped on false
  mechanism. Label every uncaught claim PLAUSIBLE; honest "cause unknown" is
  cheap, confident wrong cause buy a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below from
defect shipped in ONE session that prompted it, and every one caught by
adversarial reviewer rather than by me — i.e. each avoidable by reading two more
lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace data
  flow before writing it.** FAILURE: wrote "dropping this call now makes the
  test fail" about `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` not consumed until `d_sw5`. Sentence written
  from intent, not from call chain. Before asserting "X is covered by this
  test", name consumer of X and confirm it execute inside test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` give unconditional copy; it is
  copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of code
  with two aliasing behaviours, asserted as safe. Extends *Never infer an API*
  to STRENGTH of a guarantee, not just signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET A
  GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called interim helpers directly. Went
  into COMMIT MESSAGE, which cannot be edited after. Weaken to what you verified
  ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY IS
  NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` mean duogrid⇒bounded, and I wrote converse pair "unreachable" —
  inverting it and mislabelling legitimate regional/nested category as
  impossible. Quote the line next to restatement so direction checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print baseline value the
  perturbation multiplies BEFORE trusting result.** FAILURE: divg-corner probe
  scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0, so control was no-op
  and proved nothing; claim actually rested on predicted==actual identity. Say
  which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts differing in grid
  conventions AND in whether ext bundle existed, while asserting it isolated the
  exchange. Applies to fixtures, not just runs — controlled-comparison rule
  cover `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT, NOT
  ONLY THE TEST YOU WROTE.** Fixture edit = change to every test in the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout
  --`/`git restore`, `cp -f`, `rm -f`; never bare `cp`/`mv` for revert. FAILURE:
  bare `cp` hit interactive overwrite prompt and "revert" never applied — one
  `git status` short of reporting clean tree that was not clean. VERIFY EVERY
  REVERT with `git status --porcelain`.
- **A LOG-SCRAPING GUARD MUST EXCLUDE THE PROMPT/COMMAND IT ECHOES.** FAILURE:
  review wrapper grepped whole log for failure token its own prompt contained,
  so it "retried" every time and never printed exit status. Same class as test
  that cannot fail.
- **BUDGET THE REVIEW LOOP INTO THE WORK.** Adversarial reviewer found real
  defects in EACH of three rounds here; rounds 2 and 3 not ceremony. No
  presenting first-round implementation as finished, no treating "tests pass" as
  terminal condition — round-1 diff passed 401 tests while still containing
  silent fallback and false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
Gates above stop wrong CLAIMS about the model; these stop cheaper, more frequent
error — being wrong about CODE AND DATA IN FRONT OF YOU. Every rule below
mechanical, each cost full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** "Never infer
  an API" rule cover signatures; it also cover array SHAPE, AXIS ORDER, index
  base, padding convention. Before first index of any mesh/state field, print
  its `.shape`. FAILURE: indexed `mesh.cellsOnCell[c, k]` assuming
  `(nCells, 6)`; it is `(6, nCells)`, so probe raised `IndexError` on every
  method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. Real number produced by specific code path →
  GET IT BY CALLING THAT PATH.** Re-deriving lookalike from first principles
  silently answer different question. FAILURE: computed `max_degree` from 1-ring
  `cellsOnEdge` adjacency and reported it as "ppermute round count" — real
  exchange is halo-depth-aware, so proxy said 8 rounds for every method/rank
  count while real schedule said 12→14. Proxy could not even reproduce known
  answer — that the tell: **run proxy against a case whose real value you
  already know BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state which
  quantity it minimizes, confirm that quantity is measured bottleneck.**
  FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS minimize EDGE
  CUT, bottleneck is MAX_DEGREE, and measured they move OPPOSITELY (metis 19
  rounds @64 vs sfc 14, while metis has lower cut). "Better partitioner" is not
  a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
  against raw source first. FAILURE: regex over multi-line series lists attached
  subdiv-9's `9.60/11.48 ms` to `s8` label, and that mislabelled pair went into
  codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing test, grep
  for one already asserting same invariant. FAILURE: added two tests counting
  PCG reduction sites duplicating existing `test_reduction_count_halved`; codex
  had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG, AND
  SAY WHICH.** Assertion you just wrote is claim with no more standing than the
  code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant OFF; resolver is
  `!= "0"`, so `""` means ON — test was wrong, not the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep driver for
  decomposition/sharding the run would need.** FAILURE: scoped 128-GPU coupled
  ladder before finding `CoupledESM` contain three occurrences of "shard", all
  in comments — coupled ocean is replicated, so ladder would have measured
  unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT

exec
/usr/bin/bash -lc "which python; ls /glade/u/home/pg2328/legoesm/.venv/bin/python; rg -n 'g_point_to_bnd' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/gas_optics.py; sed -n '760,900p' CLAUDE.md" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
/glade/u/apps/derecho/25.10/opt/view/bin/python
ls: cannot access '/glade/u/home/pg2328/legoesm/.venv/bin/python': No such file or directory
341:  ibnd = lookup.g_point_to_bnd[igpt]
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  serial-vs-SPMD gap as "cadence PHASE disagreement"; predicates agree with ZERO
  offset and real cause was serial short-tail fallback. One `grep` of two
  predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* Gates above stop wrong CLAIMS; these stop wrong RUNS.
Simulation launched on hypothesis no measurement can refute = pure waste, and it
also cost WALL-CLOCK of queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job costing
  >1 GPU-hour, write down three things: (a) exact number the run will produce,
  (b) value that CONFIRMS and value that REFUTES, (c) why a cheaper offline/CPU
  test on an EXISTING checkpoint cannot answer it. Cannot fill all three -> DO
  NOT SUBMIT; run cheap test first.** Nearly every question asked so far
  (fluxes, tendencies, radii, momentum budgets, cloud optics) was answerable
  offline from saved checkpoint in minutes. FAILURE 2026-07-30: submitted two
  5-YEAR full-physics runs (8 h walltime each) while model had KNOWN unfixed +56
  W/m2 albedo error and no low-level circulation — five simulated years of
  broken climate, answering no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run full
  scorecard FIRST, work LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of session on hfls deficit (-40 W/m2) while dominant
  error was rsut (+56 W/m2) — and scorecard naming it was already sitting in run
  directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer ZERO
  questions, because no output attributable to any one of them. Multi-change
  config legitimate ONLY as deliberate new BASELINE labelled as such and never
  compared term-by-term against the old one. FAILURE 2026-07-30: one launch
  flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state largest outstanding scorecard term
  and why run still worth its GPU-hours. Fix big term, then extend. Short
  validation windows (days) are for "does it run and is new physics behaving";
  multi-year windows are for model that already pass.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** Hand-computed ratio that
  "matches" observed ratio is not evidence when calculation omit factors code
  actually applies. State it as arithmetic, or don't state it. FAILURE
  2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored sub-grid
  inhomogeneity factor radiation applies to cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing probe, check run
  directory for scorecard/manifest/diagnostic answering the question, and
  `scripts/validate/` for validator. Reading existing artifact cost seconds; new
  probe cost an hour and need its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule mechanical, from specific
failure in that one session. Root cause turned out to be one-line IC defect
findable offline in seconds; arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING ON
  IT — one line of arithmetic, before the conclusion.** Convert to quantity
  whose plausible span you know, compare. FAILURE: PGF probe returned 0.02
  m/s^2 and I built whole attribution on it; that value imply ~14 kg/m^3 density
  difference between ADJACENT 1-degree cells when entire ocean span ~6 kg/m^3.
  Check take seconds, number sat unused for hours, and running it immediately
  would have exposed real defect (cells initialised to T=0/S=0) before a single
  GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State instrument's resolution, confirm reported step is not
  merely its first sample. FAILURE: five one-variable arms all reported "BLOWUP
  at step 36" and I read that config-invariance as physics. `run_omip` set
  `block_size = max(1, diag_every)` and only test at block boundaries, so 36 was
  FIRST LOOK. At `--diag-every 1` answer was step 1. Corollary: arms run at
  coarse diagnostic stride prove only "this setting alone does not fix it",
  NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  same step can fail for different reasons. FAILURE: declared open North Pole
  "refuted" because capping did not move blowup step; states showed capping HAD
  removed that mode (|u|max 19.27 -> 0.0137 m/s) and merely uncovered second,
  unrelated one at same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read signature of every function a probe call,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported NaN "conservation defect"
  in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** Sentinel that is valid number of right dtype pass every finite/NaN
  guard downstream. FAILURE: `_interp_profile_to_z_coord` returned `0.0` for
  no-data column; `0.0` is not NaN, so caller's `where(isnan, fill, x)` never
  fired and 39032 wet cells entered model as fresh water at 0 degC. TWO
  COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one calling it "documented degenerate
  fallback" and one commenting "still no NaN leaks" — exact inversion of truth.
  Degenerate branch must return something → return NaN (or raise) so downstream
  guard can see it, and treat any test asserting magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run closure/redistribution check FIRST — it name the operator class
  instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted hypothesis I was about to spend a GPU arm on (dipole
  whose two-cell sum GREW is not a diffusion instability) and localised
  config-invariant source to single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance (git
  SHA + inputs + flags) stamped on every run. Inline heredoc probes not citable,
  cannot be re-run against changed model, and hide own bugs — both defects above
  came from uncommitted ones, and committing closure probe immediately surfaced
  two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In blown-up field, "worst
  cells are coastal / shallow / polar" describe where damage LANDED, usually
  downstream of single upstream defect. FAILURE: measured P(bad|coastal)=33% vs
  0.67% interior, real 50x enrichment, and treated it as coastal process; those
  were simply columns the IC defect had corrupted. Before citing spatial
  pattern, confirm snapshot is ORIGIN state (`snapshot_final.npz` stamp
  `_step = n_steps` regardless of when run stopped — verify with dedicated
  1-step run).

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.

exec
/usr/bin/bash -lc "rg -n 'def |aerosol|forcing' packages/coupler/legoesm/driver/physics_pipeline.py | rg 'step_unified|aerosol_sw|aerosol_lw'; sed -n '2490,2518p' packages/tools/legoesm/forcing/external.py; rg -n 'res6|subdiv.*6' configs | head -20; sed -n '1460,1508p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
2898:                               aerosol_lw_od_precomputed=None,
2899:                               aerosol_sw_volc_precomputed=None,
3321:            aerosol_lw_od_precomputed = _shard(aerosol_lw_od_precomputed)
3322:            aerosol_sw_volc_precomputed = _shard(aerosol_sw_volc_precomputed)
3337:            aerosol_lw_od_col=aerosol_lw_od_precomputed,
3339:            **({} if aerosol_sw_volc_precomputed is None
3340:               else {"aerosol_sw_volc_col": aerosol_sw_volc_precomputed}),
3458:    def build_step_unified(self, static_need_rad: bool | None = None,
3538:        def step_unified(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
3551:                         aerosol_lw_od=None,
3552:                         aerosol_sw_volc=None,
3607:                 solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
3633:                        aerosol_lw_od_precomputed=aerosol_lw_od,
3634:                        aerosol_sw_volc_precomputed=aerosol_sw_volc,
3719:                 solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
3790:                    solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
3844:                     aerosol_lw_od_col=None,
3850:                     cloud_fraction=None, aerosol_sw_volc_col=None):
3851:        del aerosol_lw_od_col, o3_top_vmr_col  # zero-radiation: no-ops
3852:        if aerosol_sw_volc_col is not None:
3902:                     aerosol_lw_od_col=None,
3908:                     cloud_fraction=None, aerosol_sw_volc_col=None):
3910:        del aerosol_lw_od_col, o3_top_vmr_col  # gray: no aerosol LW od / ozone
3911:        if aerosol_sw_volc_col is not None:
4035:                     aerosol_lw_od_col=None,
4042:                     aerosol_sw_volc_col=None):
4107:            aerosol_absorption_optical_depth_lw=aerosol_lw_od_col,
4108:            **({} if aerosol_sw_volc_col is None
4109:               else {"aerosol_volcanic_moments_sw": aerosol_sw_volc_col}),
    else:
        mid_anchored_v, prof_anchored_v = _cyclic_phase_anchor(
            mid_days_v, prof_v, first_date_v,
        )
        prof_t = _interp_monthly_cyclic(mid_anchored_v, prof_anchored_v, day)
    prof_t = prof_t * config.volcanic_scale  # (nlat, nlayer)

    if lat_grid is not None:
        # (nlat, nlayer) -> (*lat_grid.shape, nlayer); the driver hands a
        # raveled column-latitude grid, so flatten leading dims to
        # (ncol, nlayer).  The nlayer axis is the vertical PROFILE and is
        # intentionally NOT summed away (unlike the gray SW/column path).
        prof_col = _interp_zonal_to_grid(lat_v, prof_t, lat_grid)
        prof_col = jnp.reshape(
            jnp.clip(prof_col, 0.0, None), (-1, *prof_t.shape[1:]),
        )
        return prof_col, jnp.asarray(p_edges)

    return {
        "lat": lat_v,
        "aod_profile": np.clip(prof_t, 0.0, None),
        "p_edges": np.asarray(p_edges),
    }


# ==============================================================================
# Solar irradiance
# ==============================================================================

rg: configs: IO error for operation on configs: No such file or directory (os error 2)

      # Split every per-column array on its leading axis into
      # (n_block, column_chunk_size, ...).  This row-major reshape is
      # order-preserving and the reassembly reshape below is its exact
      # inverse, so no column is reordered — the exactness guarantee.
      def _is_col_array(x):
          return (hasattr(x, "shape") and getattr(x, "ndim", 0) >= 1
                  and x.shape[0] == ncol)

      def _split(x):
          return x.reshape(n_block, column_chunk_size, *x.shape[1:])

      mapped = {
          "T": _split(T), "p_full": _split(p_full), "p_half": _split(p_half),
          "sfc_temperature": _split(sfc_temperature), "q_v": _split(q_v),
          "cos_zenith": _split(cos_zenith),
      }
      # ``solar_spectral_fraction`` is a per-g-point vector (leading axis
      # n_gpt, NOT ncol) that must be broadcast to every block, never split.
      # Excluded by name so it stays static even in the degenerate
      # ncol == n_gpt_sw case where the shape heuristic alone would misfire
      # (splitting it → each block sees a short weight vector → solve_columns
      # raises on the wrong g-point length).
      _non_column_opt = ("solar_spectral_fraction",)
      static = {}
      for k, v in opt.items():
          if k not in _non_column_opt and _is_col_array(v):
              mapped[k] = _split(v)
          else:
              static[k] = v

      # Checkpointed per block: lax.map is a scan, so without this the
      # backward pass keeps every block's radiation activations at once and
      # the chunking buys compile time but no memory.  With it the backward
      # recomputes one block at a time, so reverse-mode scratch is bounded by
      # ``column_chunk_size`` columns instead of ``ncol`` -- the AMIP WB arm
      # needs it because max-random overlap hands the solver n_sub*ncol
      # sub-columns (measured 490 GiB of scratch at 8 sub-columns, T63/L32).
      # Forward-only callers are unaffected (checkpoint is a no-op there) and
      # the values are unchanged either way.  prevent_cse=False because this
      # body already runs inside a scan (same reason as the g-point block
      # loop in rte/two_stream.py).
      @functools.partial(
          jax.checkpoint,
          policy=jax.checkpoint_policies.nothing_saveable,
          prevent_cse=False,
      )
      def _one(chunk):
          return self.solve_columns(**chunk, **static)

exec
/usr/bin/bash -lc "python -c 'import jax, netCDF4; print(jax.__version__)'" in /glade/derecho/scratch/pg2328/wt_volc
 exited 1 in 0ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'

exec
/usr/bin/bash -lc "sed -n '3530,3645p' packages/coupler/legoesm/driver/physics_pipeline.py; rg -n 'g_point_to_bnd\\s*=' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp; sed -n '20170,20205p' packages/coupler/legoesm/driver/model_driver.py; sed -n '1770,1805p' packages/ml/legoesm/training/scale_build.py; rg -n 'res6|resolution: 6' scripts/run | head -10" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 27ms:
                    "turbulence='mynn25' needs build_step_unified("
                    "mynn_initflag=True) for the cold-start step and False "
                    f"for every other; got {mynn_initflag!r}")
        elif mynn_initflag is not None:
            raise ValueError(
                "mynn_initflag is the mynn25 cold-start switch; this "
                "pipeline does not run mynn25")

        def step_unified(need_rad, T, p_s, q_v, q_c, q_r, conv_prog, u, v,
                         sst, sic, lat, lon,
                         day_of_year, seconds_of_day, dt,
                         solar_weights, s_0,
                         o3_vmr, aerosol_od,
                         held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                         held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                         tau_equator=None, tau_pole=None,
                         sbm_tau_c=None, sbm_RH_ref=None,
                         C_H=pipeline.C_H, C_E=pipeline.C_E,
                         albedo_ice=pipeline.albedo_ice,
                         albedo_ocean=pipeline.albedo_ocean,
                         ghg_vmr_override=None,
                         aerosol_lw_od=None,
                         aerosol_sw_volc=None,
                         aerosol_ccn_aod=None,
                         o3_top_vmr=None,
                         T_land=None,
                         q_i=None, q_s=None, q_g=None,
                         N_c=None, N_r=None, N_i=None,
                         sfc_albedo_override=None,
                         sfc_T_override=None,
                         sfc_emissivity_override=None,
                         sfc_shflx_override=None,
                         sfc_lhflx_override=None,
                         sfc_evap_override=None,
                         sfc_taux_override=None, sfc_tauy_override=None,
                         sfc_lw_up=None, sfc_sw_up=None, sfc_sw_down=None,
                         land_frac=None, phis=None,
                         tke=None, qke=None, gwd_spectrum=None,
                         conv_precip=None, land_ml=None, w_land=None,
                         snow=None, land_ml_params=None, cloud_fraction=None,
                         conv_precip_frozen=None, ice_T_skin=None,
                         cell_area=None, dq_v_dt_dyn=None,
                         turb_dq_v_prev=None, evap_sfc_prev=None):

            if pipeline.ice_skin_prognostic and ice_T_skin is None:
                raise ValueError(
                    "ice_skin_prognostic=True but this lane's carry has no "
                    "ice_T_skin (#2106); the constant T_ice would run "
                    "silently. Seed the skin in the carry (pack_carry("
                    "ice_T_skin=...)).")
            if ice_T_skin is not None:
                # Melt-dependent sea-ice albedo on the carried skin (scalar
                # albedo_ice under "constant").
                from legoesm.ice.shortwave import sea_ice_albedo
                albedo_ice = sea_ice_albedo(
                    pipeline.sea_ice_albedo_scheme, albedo_ice,
                    T_skin=ice_T_skin, h_ice_m=pipeline.ice_thickness_m,
                    melt_width_K=pipeline.sea_ice_melt_ramp_width_K)

            def _skin_tail(skin, held, physics_out, sic, dt):
                # Same per-step update as the MPAS column lane: this step's
                # (fresh or held) radiative net + turbulent fluxes, consumed by
                # the NEXT step's surface blends.  A 5th return value ONLY when
                # the skin is on, so every 4-value caller is unchanged.
                if skin is None:
                    return ()
                from legoesm.forcing.surface_utils import advance_ice_skin
                new = advance_ice_skin(
                    skin, held[1], held[2], physics_out.shflx,
                    physics_out.lhflx, sic, dt_s=dt,
                    h_ice_m=pipeline.ice_thickness_m,
                    melt_cap_width_K=pipeline.ice_skin_melt_cap_width_K)
                return (new.astype(skin.dtype),)

            def _rad_branch(args):
                (T, p_s, q_v, q_c, q_r, conv_prog, u, v, sst, sic, lat, lon,
                 day_of_year, seconds_of_day, dt,
                 solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
                 held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                 held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
                 tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref,
                 C_H, C_E, albedo_ice, albedo_ocean,
                 ghg_vmr_override, T_land,
                 q_i, q_s, q_g, N_c, N_r, N_i,
                 sfc_albedo_override, sfc_T_override, sfc_emissivity_override,
                 sfc_shflx_override, sfc_lhflx_override,
                 sfc_taux_override, sfc_tauy_override, sfc_evap_override,
                 sfc_lw_up, sfc_sw_up, sfc_sw_down,
                 land_frac, phis,
                 tke, qke, gwd_spectrum,
                 conv_precip, land_ml, w_land, snow, land_ml_params,
                 cloud_fraction, conv_precip_frozen, ice_T_skin) = args
                # land_frac / phis exist for the LEARNED wrappers that share
                # this step signature (phase 2, part 2); the classical
                # pipeline does not consume them.
                del land_frac, phis

                (dT_dt_rad, sw_net_sfc, lw_net_sfc,
                 sw_up_toa, lw_up_toa, sw_down_toa, T_land_new, land_ml_new) = \
                    pipeline.compute_radiation_core(
                        T, p_s, q_v, sst, sic, lat, lon,
                        day_of_year, seconds_of_day,
                        solar_weights, s_0, o3_vmr, aerosol_od,
                        aerosol_lw_od_precomputed=aerosol_lw_od,
                        aerosol_sw_volc_precomputed=aerosol_sw_volc,
                        aerosol_ccn_aod=aerosol_ccn_aod,
                        o3_top_vmr=o3_top_vmr,
                        tau_equator=tau_equator, tau_pole=tau_pole,
                        albedo_ice=albedo_ice, albedo_ocean=albedo_ocean,
                        ghg_vmr_override=ghg_vmr_override,
                        q_c=q_c, q_i=q_i, N_c=N_c, N_i=N_i,
                        cloud_scheme=pipeline._cloud_scheme,
                        u=u, v=v, dt=dt, T_land=T_land,
                        sfc_albedo_override=sfc_albedo_override,
                        sfc_T_override=sfc_T_override,
                        sfc_emissivity_override=sfc_emissivity_override,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_base.py:295:  g_point_to_bnd = np.asarray([None] * dims['gpt'])
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_base.py:298:  g_point_to_bnd = np.array(g_point_to_bnd, dtype=jnp.int_)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_base.py:339:      g_point_to_bnd=jnp.array(g_point_to_bnd),
                "segment. Train with the flag off, or wire the conversion into "
                "the training carry first (#1028).  Note what training with "
                "the flag off costs once production turns it ON: the damped "
                "regime carries roughly a THIRD of the eddy amplitude, so a "
                "model fitted on it meets a distribution it never saw (GLM "
                "review) -- this refusal is fail-loud, not a resting place.")

        # Lat-lon C-grid runs carry their face winds between steps; a training
        # segment never enters run(), so it would integrate the cell-centre
        # (eddy-damped) path production no longer runs.  Refuse (approved hard
        # error) rather than fit a model to a different discretisation.
        from legoesm.driver.component_factory import driver_solver_name
        if driver_solver_name(self.config) == "latlon_cgrid_primitive_equations":
            raise NotImplementedError(
                "build_training_segment does not carry the lat-lon C-grid face "
                "winds production runs carry between steps, so it would train "
                "on the cell-centre path that re-projects the winds every step "
                "(most of the eddy kinetic energy gone within a day). Wire the "
                "face winds into the training carry first.")
        ctx = self._prepare_run_context(0, day, restore_carry=False)
        cfg = self.config
        DT = ctx["DT"]
        _sd = ctx["_sd"]
        _seg_lat = (self._physics_lat if self._physics_lat is not None
                    else self._grid_lat)
        _seg_lon = (self._physics_lon if self._physics_lon is not None
                    else self._grid_lon)

        # Un-jitted step (jit=False): a calibrator feeds a TRACED value into the
        # pipeline via an attribute the step reads (e.g. land_ml_params); a jitted
        # step would capture that tracer as a closure constant and leak it across
        # value_and_grad calls.  The step is inlined into run_segment's lax.scan.
        step_unified = self.physics.build_step_unified(
            static_need_rad=True, jit=False)

        run_segment = build_segment_fn(
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import (
        era5_to_latlon_carry,
        open_era5_zarr,
        regrid_2d_to_gaussian,
    era5_terrain_product,
)
    from legoesm.driver.compiled_segments import pack_forcing

    if getattr(cfg, "training_core", "latlon") == "spectral":
        return _load_era5_samples_spectral(
            cfg, yml, grid, sigma,
            rank=rank, nproc=nproc, host_resident=host_resident)

    global _WARNED_ERA5_FLUX_HOLD_FIXED

    era5_cfg = wb_era5_config(cfg, yml)
    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    snaps_per_day = 24 // era5_cfg.dt_hours
    roll_h = rollout_hours(cfg, yml)
    stride = int(roll_h) // era5_cfg.dt_hours

    # Opened once, like the state store — a per-sample remote zarr open would
    # dominate the load (see the spectral loader).
    cloud_ds = (open_era5_zarr(era5_cfg.cloud_zarr)
                if era5_cfg.load_cloud_condensate else None)
    # 2-D only, but keeps load_surface_fluxes / load_land_frac so the
    # sample-start slice carries the prescribed planes.
    sst_cfg = era5_cfg._replace(load_cloud_condensate=False)
    # Flux store opened once, only when fluxes are prescribed ("" = the
    # state store); a land fraction alone comes from the state store.
    flux_ds = None
    if sst_cfg.load_surface_fluxes:
        flux_ds = open_era5_zarr(sst_cfg.flux_zarr) if sst_cfg.flux_zarr else ds

exec
/usr/bin/bash -lc "ls /glade/work/pg2328; sed -n '280,304p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_base.py; sed -n '1810,1833p' packages/ml/legoesm/training/scale_build.py; rg -n 'res6|resolution.*6|subdiv.*6' scripts/validate/probe* scripts/run/run_amip* | head -20; sed -n '1,80p' tests/conftest.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 11ms:
amip_inputs
conda-envs
cpuperf_env.sh
env_backups
envs
headroom-venv
legoesm-base0918
legoesm-cpuperf
legoesm-duofix
legoesm-ladder
legoesm-license
legoesm-thomas
mpiwrapper
ollama
venvs
          idx_gases,
          strings['minor_gases_lower'],
          strings['scaling_gas_lower'],
      )
  )
  idx_minor_gases_upper, idx_scaling_gases_upper = (
      _create_rrtm_consistent_minor_gas_index(
          idx_gases,
          strings['minor_gases_upper'],
          strings['scaling_gas_upper'],
      )
  )
  # Decrement indices since RRTMGP was originally developed in a 1-based index
  # system.
  bnd_limits_gpt = np.asarray(tables['bnd_limits_gpt']) - 1
  g_point_to_bnd = np.asarray([None] * dims['gpt'])
  for i in range(dims['bnd']):
    g_point_to_bnd[bnd_limits_gpt[i, 0] : bnd_limits_gpt[i, 1] + 1] = i
  g_point_to_bnd = np.array(g_point_to_bnd, dtype=jnp.int_)
  minor_lower_gpt_lims = np.asarray(tables['minor_limits_gpt_lower']) - 1
  (
      minor_lower_bnd,
      minor_lower_bnd_start,
      minor_lower_bnd_end,
      minor_lower_gpt_shift,
    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)

    # Spectral forcing-dict name -> driver-path (pack_forcing) name for the
    # planes the two lanes name differently; the radiative planes and the
    # land fraction already share their names.
    _spec_to_driver = {
        "sfc_shf": "sfc_shflx_override",
        "sfc_lhf": "sfc_lhflx_override",
        "sfc_tau_x": "sfc_taux_override",
        "sfc_tau_y": "sfc_tauy_override",
    }

    def build(year, i_ic, i_tg, ds, cloud_ds, flux_ds, at):
        global _WARNED_ERA5_FLUX_HOLD_FIXED
        s_ic, s_tg, sst_src = _sample_inputs(
            era5_cfg, sst_cfg, i_ic, i_tg, ds, cloud_ds, flux_ds, at)
        tp = era5_terrain_product(s_ic, grid)
        _skin = bool(config.ice_skin_prognostic)
        ic = era5_to_latlon_carry(s_ic, grid, sigma, target_phis=tp,
                                  microphysics=_micro, ice_skin_prognostic=_skin,
                                  convection=config.convection)
        target = era5_to_latlon_carry(s_tg, grid, sigma, target_phis=tp,
                                      microphysics=_micro,
                                      ice_skin_prognostic=_skin,
scripts/run/run_amip_smoke_deck.py:32:        --resolution 16 --days 30 \\
scripts/run/run_amip_smoke_deck.py:38:        --auto-generate --resolution 16 --days 30
scripts/run/run_amip_smoke_deck.py:180:    parser.add_argument("--resolution", type=int, default=16,
scripts/run/run_amip_cross_grid.sh:118:    # ran at ``--resolution`` default (16), regardless of the
scripts/run/run_amip.py:6:        --dataset analytical --days 30 --resolution 16 --dt 600
scripts/run/run_amip.py:11:        --days 30 --resolution 16 --dt 600
scripts/run/run_amip.py:289:    parser.add_argument("--resolution", type=int, default=16)
scripts/run/run_amip.py:1871:                             "turnover resolution factor (cumastrn.F90:762-768;"
scripts/run/run_amip.py:2420:                             "subdivision-6 mesh), so it holds grid-scale "
scripts/run/run_amip.py:4544:    # Higher resolutions (e.g. C48 at dt=600) BLOWUP at day 25.
scripts/run/run_amip_1deg_latlon_mpi.sbatch:18:# Resolution: ``--resolution 180`` → 180 lat × 360 lon (= 1° at the
"""Shared test fixtures for legoESM."""

import functools
import os
import resource
import sys
from pathlib import Path

#: Cap the CPU affinity mask BEFORE JAX is imported (below).  XLA sizes its
#: Eigen thread pool from the affinity mask, so on a many-core login node a
#: single JAX process opens far more threads than ``ulimit -u`` allows once a
#: few run at once, and the pool constructor aborts the interpreter:
#:
#:   F env.cc:93] Check failed: ret == 0 (11 vs. 0) Thread tf_XLAPjRtCpuClient
#:   creation via pthread_create() failed.
#:     @ Eigen::ThreadPoolTempl<>::ThreadPoolTempl()
#:
#: (11 = EAGAIN.)  It surfaces as "Fatal Python error: Aborted" mid-run, which
#: reads like a test failure but is the process running out of thread slots.
#:
#: Measured on a 256-core Levante login node (``ulimit -u`` = 2048), threads
#: opened by one client for a trivial jitted matmul:
#:
#:     CPUs    4     8    16    32    64   256
#:     threads 29    45    77   141   237   621
#:
#: Uncapped that is 621 threads each, so FOUR concurrent clients (2484) are the
#: first count over the 2048 ceiling; five were observed to fail 5/5.  At 32
#: CPUs a client takes ~141, so ~14 fit.  Capping costs nothing measurable --
#: these tests are XLA-compile-bound, not intra-op-parallelism-bound
#: (tests/test_corner_div_damp_nh.py: 166.9 s on 256 CPUs, 166.3 s on 8,
#: 152.7 s on 32).
#:
#: Set ``LEGOESM_TEST_CPU_CAP`` to a positive integer to force a cap, or to 0
#: (or any value <= 0) to disable and inherit the machine's full affinity mask.
_DEFAULT_TEST_CPU_CAP = 32

#: Threads one XLA CPU client opens ~= _THREADS_PER_CPU * ncpu + _THREADS_BASE,
#: fitted to the table above (slope (621-29)/(256-4) = 2.35).  Used to size the
#: cap against the ACTUAL thread ceiling rather than trusting a fixed 32, which
#: is only evidenced safe for ulimit -u = 2048.
_THREADS_PER_CPU = 2.4
_THREADS_BASE = 20.0
#: Fraction of the thread ceiling the test session may claim, leaving room for
#: the user's other processes (a login node already carries ~50 threads).
_RLIMIT_SHARE = 0.5

#: Env vars every common launcher sets in a rank's environment.  Under MPI the
#: launcher (or the batch scheduler) owns placement: each rank already gets its
#: own binding, and clamping every rank to the SAME low-numbered CPUs here would
#: pile all of them onto one core set and serialise the run.  So the cap is
#: skipped for MPI ranks -- they are separate processes with a launcher-assigned
#: mask, which is exactly the placement this function would otherwise destroy.
#: MPI does not standardise a rank env var, so detection is BEST-EFFORT: this
#: covers the launchers in use here.  For an unlisted launcher, disable the cap
#: explicitly with ``LEGOESM_TEST_CPU_CAP=0``.
_MPI_RANK_ENV = (
    "OMPI_COMM_WORLD_RANK",     # Open MPI
    "PMI_RANK",                 # MPICH / Intel MPI
    "PMIX_RANK",                # PMIx
    "SLURM_PROCID",             # srun-launched
    "MV2_COMM_WORLD_RANK",      # MVAPICH2
    "MPI_RANKID",               # IBM Platform MPI
    "LAMRANK",                  # legacy LAM/MPI
)


def _is_mpi_rank() -> bool:
    """True when this interpreter was launched as a rank of an MPI job."""
    return any(v in os.environ for v in _MPI_RANK_ENV)


def _requested_cap() -> int | None:
    """The user's explicit ``LEGOESM_TEST_CPU_CAP``, or None when unset/garbage.

    A value <= 0 disables capping entirely and is returned as 0.
    """
    raw = os.environ.get("LEGOESM_TEST_CPU_CAP")
    if raw is None:
        return None

exec
/usr/bin/bash -lc "ls /glade/work/pg2328/venvs /glade/work/pg2328/conda-envs /glade/work/pg2328/envs; rg -n '"'^def |''^    def '"' packages/coupler/legoesm/driver/model_driver.py | tail -30; sed -n '560,615p' packages/atmosphere/legoesm/atmosphere/physics/neural_physics.py; sed -n '1,100p' packages/core/legoesm/core/precision.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
/glade/work/pg2328/conda-envs:
legoesm-gpu

/glade/work/pg2328/envs:
headroom

/glade/work/pg2328/venvs:
context-compress
12937:    def _fv3_duo_checkpoint_write(self, arrays: dict, mcfg, step: int,
12971:    def _fv3_duo_save_checkpoint(self, bundle: dict, step: int,
13016:    def _decode_fv3_duo_checkpoint_arrays(self, path: Path):
13138:    def _validate_fv3_duo_broadcast_source(self, src_flat, template,
13188:    def _fv3_duo_broadcast_leaf_spec(self, nm: str, template, q_shape,
13198:    def _broadcast_fv3_duo_bundle(self, template, hdr, src_flat):
13254:    def _load_fv3_duo_checkpoint(self, path: Path) -> tuple[int, float]:
13354:    def export_mpas_step(self, start_step: int = 0,
13389:    def _column_daily_forcing_active(self) -> bool:
13408:    def _run_column(self, start_step: int = 0, start_day: float | None = None) -> str:
17829:    def _refuse_stateful_physics_unthreaded(cfg) -> None:
17895:    def _run_spectral(self, start_step: int = 0, start_day: float | None = None) -> str:
18672:    def _save_lightweight_timeseries(
18835:    def _latlon_spmd_mesh(self):
18867:    def _latlon_spmd_physics_fn(self):
18896:    def _with_segment_bulk_surface(self, physics_fn, day_of):
18958:    def _reject_unconsumed_flux_override(self, lane: str) -> None:
18972:    def _reject_coupled_lane(self, lane: str, envelope: str,
19005:    def _run_compiled_latlon_spmd(self, start_step: int = 0,
19131:    def _run_operator_split_tiled_cube(self, start_step, start_day,
19135:    def _run_tiled_cube_spmd(self, start_step: int = 0,
19139:    def _operator_split_spmd_active(self) -> bool:
19157:    def _run_operator_split_spmd(self, start_step, start_day, mesh) -> str:
19618:    def _mynn_peel_pending(self, qke_key="qke", flag_key="mynn_initialized"):
19635:    def _prepare_run_context(self, start_step, start_day, restore_carry=False):
20057:    def _finalize_run(self, run_status, t_jit, t_start, n_steps_total,
20127:    def build_training_segment(self, n_steps: int, day: float = 0.0):
20306:    def _run_compiled(self, start_step: int = 0, start_day: float | None = None) -> str:
21326:    def _run_per_step(self, start_step: int = 0, start_day: float | None = None) -> str:
21889:    def save_results(self, run_status: str, jit_time: float, wall_time: float) -> None:
        Grid adapter for flatten/unflatten operations.

    Returns
    -------
    callable
        Supports both the legacy signature
        ``step_unified(..., q_r, u, v, ...)`` and the newer
        ``step_unified(..., q_r, conv_prog, u, v, ...)`` layout.
    """
    nlev = neural_physics.nlev

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = parse_step_unified_tail(args)
        (
            u,
            v,
            sst,
            sic,
            lat,
            lon,
            day_of_year,
            seconds_of_day,
            dt,
            solar_weights,
            s_0,
            o3_vmr,
            aerosol_od,
            held_dT_rad,
            held_sw_net_sfc,
            held_lw_net_sfc,
            held_sw_up_toa,
            held_lw_up_toa,
            held_sw_down_toa,
        ) = tail
        del need_rad, dt
        del solar_weights, o3_vmr, aerosol_od
        # Optional prescribed-input columns, built from the driver-path
        # kwargs EXACTLY like the spectral wrapper builds them from the
        # spectral forcing dict: static [sin(lat), cos(lat), land_frac,
        # orography in km] then the six surface fluxes in
        # SFC_FLUX_FORCING_KEYS order divided by SFC_FLUX_INPUT_NORMS (the
        # driver-path key names differ -- SFC_FLUX_STEP_UNIFIED_KEYS).
        # Every plane is flattened with the adapter, so this is
        # grid-agnostic (lat-lon, cubed sphere, MPAS).  A flag on with its
        # keyword missing is an error naming it, never silent zeros.
        # Flags off -> no extras, legacy path unchanged.
        if neural_physics.n_extra_dyn > 0:
            planes = []
            if neural_physics.n_static > 0:
                land_frac = kwargs.get("land_frac")
                phis = kwargs.get("phis")
                if land_frac is None:
                    raise ValueError(
                        "NeuralPhysics(spatial_embedding=True) requires "
                        "'land_frac' in the step_unified kwargs, got None")
                if phis is None:
"""Precision management for legoESM.

Provides a unified precision policy system that controls storage, compute,
accumulation, and control dtypes across all model components. Supports
FP32, FP64, and mixed-precision modes for GPU-optimized century-scale
climate simulations.

Three precision modes
---------------------
1. **FP32**: All computation in float32 (fastest, GPU-optimal)
2. **FP64**: Full scientific reference mode (most accurate)
3. **Mixed**: Storage/compute in float32, accumulation/solvers in float64
   (best balance of performance and stability)

Usage
-----
>>> from legoesm.core.precision import PrecisionPolicy, get_policy, cast, const
>>> policy = PrecisionPolicy.mixed()
>>> x_compute = cast(x, "tracer_advection", "compute")
>>> one = const(1.0, "pressure_gradient", "control")

References
----------
- Váňa et al. (2017): Single precision in weather forecasting.
- Klöwer et al. (2020): Number formats, error mitigation, and scope for
  16-bit arithmetics in weather and climate modeling.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Deferred import to break cycle: runtime/__init__.py imports
# runtime.precision, which re-exports from this module — so
# importing runtime.backend at module load time would mid-init
# this very file. Both helpers are only called from function
# bodies, so a lazy import is safe.
def _backend():
    from legoesm.runtime import backend as _b
    return _b


# ---------------------------------------------------------------------------
# Dtype parsing helper
# ---------------------------------------------------------------------------

_PRECISION_NAME_TO_DTYPE = {
    "float16": jnp.float16, "fp16": jnp.float16, "half": jnp.float16,
    "bfloat16": jnp.bfloat16, "bf16": jnp.bfloat16,
    "float32": jnp.float32, "fp32": jnp.float32, "single": jnp.float32,
    "float64": jnp.float64, "fp64": jnp.float64, "double": jnp.float64,
}


def parse_dtype(value, *, field_name: str, allow_none: bool = False):
    """Parse a precision config value into a JAX dtype."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} precision cannot be None")
    if value in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _PRECISION_NAME_TO_DTYPE:
            return _PRECISION_NAME_TO_DTYPE[key]
    raise ValueError(
        f"Unknown {field_name} precision {value!r}. "
        f"Use one of {sorted(_PRECISION_NAME_TO_DTYPE.keys())}."
    )


# ---------------------------------------------------------------------------
# Precision roles
# ---------------------------------------------------------------------------

class PrecisionPolicy(NamedTuple):
    """Global precision configuration.

    Attributes
    ----------
    storage : jnp.dtype
        Dtype for state arrays in memory. Determines GPU memory footprint.
    compute : jnp.dtype
        Dtype for arithmetic in kernels (matmuls, stencils, tendencies).
    accumulate : jnp.dtype
        Dtype for reductions (global sums, budget integrals, norms).
        Must be >= compute width to avoid catastrophic cancellation.
    control : jnp.dtype
        Dtype for solvers, implicit systems, and diagnostics that
        require tight tolerances (semi-implicit, barotropic, EOS).
    transcendentals : str
        ``"native"`` uses JAX/XLA math intrinsics. ``"libm"`` selects the
        CPU-only scalar system-libm callbacks used by NEMO certification.
    """
    storage: jnp.dtype = jnp.float32
    compute: jnp.dtype = jnp.float32

exec
/usr/bin/bash -lc "ls /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python /glade/work/pg2328/headroom-venv/bin/python; rg -n 'sys.path|packages' tests/conftest.py | head -15; sed -n '1,85p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; rg -n 'compute_dtype|compute_fp32' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; rg -n 'res6|40962|163842' docs/wb | head -10" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 13ms:
/glade/work/pg2328/conda-envs/legoesm-gpu/bin/python
/glade/work/pg2328/headroom-venv/bin/python
# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of a radiative transfer solver."""

import functools
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp import constants
from legoesm.atmosphere.physics.radiation.rrtmgp.config.radiative_transfer import (
    OpticsParameters,
    RRTMOptics as RRTMOpticsConfig,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import atmospheric_state
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import constants as optics_constants
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.lookup_volume_mixing_ratio import (
    LookupVolumeMixingRatio,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.optics import optics_factory
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "RRTMGP correlated-k radiation solver (solve_columns): builds the gas/"
        "cloud/aerosol optics per g-point and integrates the two-stream LW+SW "
        "radiative transfer to broadband fluxes and radiative heating rates."
    ),
    "inputs": {
        "T": "K", "p_full": "Pa", "p_half": "Pa", "sfc_temperature": "K",
        "q_v": "kg/kg", "cos_zenith": "1 (cos solar zenith angle)",
        "sfc_albedo": "1 (surface shortwave albedo)",
        "sfc_emissivity": "1 (surface longwave emissivity)",
        "o3_vmr": "mol/mol", "cloud_path_liq": "kg/m^2",
        "cloud_path_ice": "kg/m^2", "cloud_r_eff_liq": "m",
        "cloud_r_eff_ice": "m", "cloud_fraction": "1",
        "aerosol_optical_depth": "1",
    },
    "outputs": {
        "lw_flux_up": "W/m^2", "lw_flux_down": "W/m^2",
        "sw_flux_up": "W/m^2", "sw_flux_down": "W/m^2",
        "heating_rate": "K/s", "lw_heating_rate": "K/s",
        "sw_heating_rate": "K/s",
    },
    "sign_convention": (
        "heating_rate dT/dt>0 warms the layer; fluxes positive in their named "
        "direction (up/down); net radiative flux OUT of a layer cools it "
        "(heating_rate = -g/c_p * dF_net/dp); optical depth tau>=0, "
        "0<=ssa<=1, |g|<=1; cos_zenith>=0 for illuminated columns (nighttime SW "
        "zeroed). Photons enter/leave at TOA and the surface, so the column "
        "energy budget is OPEN (accounted, not conserved)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Pincus, Mlawer & Delamere (2019), JAMES, doi:10.1029/2019MS001621 "
        "(RRTMGP); swirl_jatmos two-stream port."
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_rrtmgp_stratosphere.py + "
        "tests/unit/test_physics_radiation.py: clear-sky LW cooling / SW heating "
        "profiles; differentiable in the continuous optical inputs (T, q_v, VMR, "
        "cloud paths), with only the integer band/g-point index selection static."
    ),
}

# ---------------------------------------------------------------------------
# Default RRTMGP data file paths (Zarr preferred, NetCDF fallback)
261:    waypoints).  Used only on the ``compute_fp32`` path.
377:              x64, getattr(config, "compute_fp32", False))
521:          if getattr(config, "compute_fp32", False):
524:              # ``compute_fp32=False`` because the fp32 RTE path still crashes
754:      # too: under ``compute_fp32`` a float64 ozone / aerosol column
docs/wb/reviews/2026-11-22_volcanic_optics_r1_glm.md:7:Adding `aerosol_volcanic_moments_sw` to `PER_COLUMN_SOLVER_KEYS` means MCICA duplicates the full moment tensor per subcolumn. At res6 (~10⁶ columns, 32 lev, 3×14 moments, float32) that is ~5.4 GB for one copy and `nsub`× that inside the subcolumn solve — versus ~130 MB for the old grey `(ncol, nlev)` field. The host-side interpolation (`surface_utils.py`) also materializes `(3·14, ncol, 70)` vmap inputs plus the `(ncol, 70, 3, 14)` source before remap (~11 GB peak at 10⁶ columns). This needs a measured memory budget or a band-looped/vmapped-with-lower-peak rewrite before production-resolution AMIP runs.
docs/wb/reviews/2026-11-22_volcanic_optics_r1_glm.md:26:**Approve with required follow-ups.** The physics, band ordering, energy accounting, and no-volcanic/legacy-file backward compatibility are correct and well tested; I found no silent drop path and no double counting. But **finding 1** (`GRID_SHAPED_FORCING_FIELDS` membership of a 4-D field — verify or remove before merge) and **finding 2** (subcolumn + host-interpolation memory at production resolution) must be resolved/measureed before this runs at res6; **finding 3** should be confirmed non-conditional on the grey aerosol being present.
docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md:27:(5) tests vacuous? (6) memory of (ncol,nsrc=70,3,14) host interpolation at res6; (7) any silent
docs/wb/reviews/2026-11-22_volcanic_optics_r1_codex.md:40:(5) tests vacuous? (6) memory of (ncol,nsrc=70,3,14) host interpolation at res6; (7) any silent
docs/wb/reviews/2026-11-22_volcanic_optics_r1_codex.md: WARNING: stopped searching binary file after match (found "\0" byte around offset 376507)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1728:/work/bd1083/b309178/diffESM/legoesm_pg/wt_pr_res6                                                                                 643454dd4 [feat/amip-res6-default]
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:2640:docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1728:/work/bd1083/b309178/diffESM/legoesm_pg/wt_pr_res6                                                                                 643454dd4 [feat/amip-res6-default]

exec
/usr/bin/bash -lc "rg -n 'res6|subdivision.*6|1_000_000' packages/grids scripts/run/run_amip.py | head -25; ls /glade/work/pg2328/conda-envs/legoesm-gpu/bin/pytest; rg -n 'PYTHONPATH|conda-envs|headroom' scripts/run/run_amip* | head -20; sed -n '1090,1140p' packages/coupler/legoesm/driver/compiled_segments.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 23ms:
rg: packages/grids: No such file or directory (os error 2)
scripts/run/run_amip.py:2420:                             "subdivision-6 mesh), so it holds grid-scale "
/glade/work/pg2328/conda-envs/legoesm-gpu/bin/pytest
scripts/run/run_amip_mpas_100yr_gpu.sbatch:73:export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
scripts/run/run_amip_latlon_restart_smoke.sbatch:49:export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
scripts/run/run_amip_1deg_latlon_mpi.sbatch:79:export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
    "sfc_taux_override", "sfc_tauy_override",
    "sfc_lw_up", "sfc_sw_up", "sfc_sw_down", "land_frac", "aerosol_ccn_aod",
    "o3_top_vmr", "aerosol_sw_volc",
)


def shard_forcing(forcing: SegmentForcing, device_config) -> SegmentForcing:
    """Commit the grid-shaped ``SegmentForcing`` leaves to the SAME device
    layout as the sharded state (SPMD sharding step 3 of the production
    cs_spmd design).

    ``run_segment``'s sharded JIT wrapper honours a committed
    ``NamedSharding`` on the build-time mesh per leaf and pins anything
    else REPLICATED (``_input_sharding``).  Replicated forcing means every
    device receives the FULL global forcing array at each segment
    boundary (host->device bandwidth and memory x n_devices) and GSPMD
    re-slices it against the face-sharded state inside the step.
    Committing the grid-shaped leaves with the state's own sharding rules
    (:func:`legoesm.parallel.mesh.shard_pytree` — face-first on
    cubed-sphere, lat-axis on lat-lon, tiling-aware) makes the transfer
    and memory per device ``1/n_devices`` and removes the reshard.

    No-ops (returns ``forcing`` unchanged) when: ``device_config`` is
    ``None`` (byte-identical single-device contract), the mesh has no
    face sharding (single device), or the run is mpi4jax-distributed
    (``is_distributed`` — forcing there is rank-local, never device-mesh
    sharded).  ``None`` optional fields and non-grid leaves pass through
    untouched; a grid-NAMED leaf whose shape does not match the grid rule
    (e.g. the ``(0,)`` o3 placeholder) is left alone by ``shard_pytree``'s
    shape gate.
    """
    if device_config is None or getattr(device_config, "is_distributed", False):
        return forcing
    if getattr(device_config, "face_sharding", None) is None:
        return forcing
    from legoesm.parallel.mesh import shard_pytree

    # Production cubed-sphere packs the 3-D radiation forcing FLATTENED to
    # ``(ncol=6*n*n, nlev)`` (``_precompute_external_forcing``), which the
    # leading-6 ``shard_pytree`` rule cannot see (codex 2026-07-01 Medium).
    # A flat face-major leaf shards ``P("face")`` on dim 0 — the exact
    # layout the driver already commits for flat carry leaves (see
    # ``_input_sharding``: "a flat [6*n*n, nlev] carry leaf the driver
    # placed on P('face')").  ``ncol % 6 == 0`` chunks land exactly one
    # face's columns per face-mesh slot; the face axis size divides 6 so
    # divisibility always holds.  Gate: cubed-sphere only, named grid
    # field, flat (shape[0] a positive non-6 multiple of 6).
    _flat_face = None
    if getattr(device_config, "grid_type", None) == "cubed_sphere" and \
            getattr(device_config, "mesh", None) is not None:
        from jax.sharding import NamedSharding, PartitionSpec

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c 'import jax, legoesm.forcing.external as x; print(jax.__version__, x.__file__)'" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 2283ms:
0.10.0 /glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py

exec
/usr/bin/bash -lc "ls src; cat legoesm.pth 2>/dev/null; rg --files packages | rg '/grids/(.*icos|.*factory|.*voron)' ; sed -n '1140,1180p' packages/coupler/legoesm/driver/compiled_segments.py; sed -n '1,70p' src/legoesm/__init__.py; rg -n '40962|163842|res6' docs/AMIP.md docs/atm* scripts/run/run_amip_mpas*" in /glade/derecho/scratch/pg2328/wt_volc
 exited 2 in 47ms:
legoesm
packages/core/legoesm/grids/voronoi_mesh_file.py
packages/core/legoesm/grids/factory.py
packages/core/legoesm/grids/voronoi.py
        from jax.sharding import NamedSharding, PartitionSpec
        _flat_face = NamedSharding(device_config.mesh, PartitionSpec("face"))

    updates = {}
    for name in GRID_SHAPED_FORCING_FIELDS:
        leaf = getattr(forcing, name)
        if leaf is None or not isinstance(leaf, (jax.Array, jnp.ndarray)):
            continue
        if (_flat_face is not None and leaf.ndim >= 1
                and leaf.shape[0] != 6 and leaf.shape[0] > 0
                and leaf.shape[0] % 6 == 0):
            # multiprocess_safe_device_put, NOT jax.device_put: under
            # multi-controller SPMD the cross-process bit-equality assert
            # in device_put trips on host-precomputed forcing (#693,
            # Levante 2-node receipt job 26030677).
            from legoesm.parallel.mesh import multiprocess_safe_device_put
            updates[name] = multiprocess_safe_device_put(leaf, _flat_face)
        else:
            updates[name] = shard_pytree(leaf, device_config)
    return forcing._replace(**updates) if updates else forcing


@dataclass(frozen=True)
class _SplitStepStatics:
    """Closure statics for the operator-split physics body, bundled so the
    single-rank physics + the shared finalizer live at MODULE SCOPE and can be
    reused by the lat-band-SPMD lane (which supplies its own sharded dynamics
    + mass fixer, then calls the SAME post-dynamics physics + tail — no
    duplication of the ~200-line write-back).  A plain frozen dataclass (NOT a
    NamedTuple) so it is never treated as a jax pytree: it is a closure constant
    captured by the scan body, never a scan carry/xs.  ``step_unified``,
    ``forcing`` and ``ghg_vmr_override`` depend on the per-segment forcing, so
    the bundle is built inside ``_make_single_step``."""
    step_unified: object
    forcing: object
    lat: object
    lon: object
    dt: object
    tau_equator: object
    tau_pole: object
    sbm_tau_c: object
sed: can't read src/legoesm/__init__.py: No such file or directory
rg: docs/AMIP.md: No such file or directory (os error 2)
docs/atmosphere/amip_bias_campaign_plan_2026-09.md:4:biharmonic moisture filter + zenith-angle ocean albedo; res6, 30 levels;
docs/atmosphere/cloud_cover_arms_prereg_2026-09.md:7:Base: `rhebc90_r6` (MPAS res6, production config, w7_eps_hi params), restart
docs/atmosphere/cloud_cover_arms_prereg_2026-09.md:70:Cost: 5 jobs × ~2.6 GPU-h (res6 ≈ 2.3 sim-days/h); +2 with `SPLIT_D=1`.

exec
/usr/bin/bash -lc "ls -l src/legoesm | head -10; rg -n 'n_cells|ncell|4\\*\\*|4 \\*\\*' packages/core/legoesm/grids/voronoi.py | head -12; cat pyproject.toml | head -30" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
total 96
-rw-r--r-- 1 pg2328 ncar 30042 Nov 22 14:56 cli.py
-rw-r--r-- 1 pg2328 ncar 25814 Nov 22 14:56 config.py
-rw-r--r-- 1 pg2328 ncar  2405 Nov 22 14:56 dycore_factory.py
-rw-r--r-- 1 pg2328 ncar  6851 Nov 22 14:56 experiment_registry.py
-rw-r--r-- 1 pg2328 ncar 10955 Nov 22 14:56 scaling_preflight.py
-rw-r--r-- 1 pg2328 ncar  8267 Nov 22 14:56 supported_matrix.py
1329:        n_cells = 10 * 4 ** subdivision_level + 2
1331:            f"subdivision_level={subdivision_level} would create {n_cells:.2e} "
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "legoesm"
version = "1.0.1"
description = "legoESM: A Differentiable Earth System Model in JAX"
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
requires-python = ">=3.11"
authors = [
    {name = "Pierre Gentine"},
]
keywords = ["earth-system-model", "jax", "differentiable", "weather", "climate"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]
dependencies = [
    # The substrate, carved into its own installable member (uv workspace).
    # Everything still imports as ``legoesm.<subpkg>`` via the shared PEP-420
    # namespace; this is what lets ``legoesm-atmosphere`` / ``-ocean`` / ... be
    # shipped independently on top of the same core (see FEDERATION.md).  Pinned

exec
/usr/bin/bash -lc 'PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c '"'import legoesm.forcing.external as x; print(x.__file__)'" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 1476ms:
/glade/u/home/pg2328/legoesm/packages/tools/legoesm/forcing/external.py

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c 'import sys,glob; sys.path[:0]=glob.glob(\"packages/*\")+[\"src\"]; import legoesm.forcing.external as e; print(e.__file__); import pytest; sys.exit(pytest.main([\"-q\",\"-p\",\"no:cacheprovider\",\"tests/unit/test_volcanic_sw_optics.py\",\"tests/unit/test_aerosol_band_ccn.py\"]))'" in /glade/derecho/scratch/pg2328/wt_volc
exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c 'import netCDF4; d=netCDF4.Dataset(\"/glade/work/pg2328/amip_inputs/volcanic_cmip6_from_cesm_1978-2014.nc\"); print({k:len(v) for k,v in d.dimensions.items()}); print([(k,d[k].dimensions,d[k].shape,str(d[k].dtype)) for k in (\"ext_sun\",\"omega_sun\",\"g_sun\")])'" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 322ms:
{'terrestrial_bands': 16, 'latitude': 36, 'altitude': 70, 'time': 444, 'solar_bands': 14, 'altitude_int': 71}
[('ext_sun', ('solar_bands', 'latitude', 'altitude', 'time'), (14, 36, 70, 444), 'float32'), ('omega_sun', ('solar_bands', 'latitude', 'altitude', 'time'), (14, 36, 70, 444), 'float32'), ('g_sun', ('solar_bands', 'latitude', 'altitude', 'time'), (14, 36, 70, 444), 'float32')]

exec
/usr/bin/bash -lc "sed -n '725,770p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; sed -n '3250,3330p' packages/coupler/legoesm/driver/physics_pipeline.py; rg -n 'aerosol_layer_fields\\(|get_aerosol_sw_volcanic_at_time' --glob '*.py'; sed -n '910,980p' packages/coupler/legoesm/driver/compiled_segments.py; rg -n 'def shard_pytree|ndim|shape' packages/core/legoesm/parallel/mesh.py | tail -20" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 3344ms:
          Fluxes and heating rates.
      """
      from legoesm.atmosphere.physics.radiation.output import RadiationOutput

      config = self._config
      ncol, nlev = T.shape

      # --- 0. Determine working dtype ---
      # The optics tables are loaded at whatever precision JAX was configured
      # with at load time (float32 if x64 off, float64 if x64 on).  Promote
      # all inputs to match so that table lookups, lax.cond branches, and
      # lax.scan carries have consistent dtypes throughout the solver.
      _table_dtype = self.optics_lib.gas_optics_lw.kmajor.dtype
      # McICA per-column table shift, seeded like CAM's mcica_subcol_gen
      # (clouds.subcolumns.cam_seed_shift), taken before the cast to the
      # table dtype.
      if mcica_cloud_fraction is None:
          _mcica_shift = None
      else:
          from legoesm.atmosphere.physics.clouds.subcolumns import (
              cam_seed_shift)
          _mcica_shift = cam_seed_shift(p_full)
      T = T.astype(_table_dtype)
      p_full = p_full.astype(_table_dtype)
      p_half = p_half.astype(_table_dtype)
      q_v = q_v.astype(_table_dtype)
      sfc_temperature = jnp.asarray(sfc_temperature).astype(_table_dtype)
      cos_zenith = jnp.asarray(cos_zenith).astype(_table_dtype)
      # Optional external-forcing arrays must match the working dtype
      # too: under ``compute_fp32`` a float64 ozone / aerosol column
      # (built by the driver under JAX x64) would re-promote the optical
      # depth and break the RTE ``lax.scan`` carry dtype — exactly the
      # PR #343 leak class, resurfaced when the MPAS/spectral paths
      # started threading per-step o3_vmr / aerosol_od (2026-06-10).
      if o3_vmr is not None:
          o3_vmr = jnp.asarray(o3_vmr).astype(_table_dtype)
      if aerosol_optical_depth is not None:
          aerosol_optical_depth = jnp.asarray(
              aerosol_optical_depth).astype(_table_dtype)
      if aerosol_absorption_optical_depth_lw is not None:
          aerosol_absorption_optical_depth_lw = jnp.asarray(
              aerosol_absorption_optical_depth_lw).astype(_table_dtype)
      if aerosol_volcanic_moments_sw is not None:
          aerosol_volcanic_moments_sw = jnp.asarray(
              aerosol_volcanic_moments_sw).astype(_table_dtype)
      if cloud_path_liq is not None:
            # of the RH grid-scale fraction.  Gated on the pipeline flag so a
            # non-clubb run passes None (byte-identical).  The carry is ALWAYS
            # flattened column form (ncol, nlev) — it is written from
            # ``turb_out.cloud_fraction`` (column layout) and carried as-is — so
            # reshape unconditionally to the local column shape ``T_col.shape``
            # (a no-op when already matching), exactly like the radiation
            # physics_fn sibling in radiation/integration.py.  (An earlier
            # ndim-conditional ``ad.flatten_3d`` branch was WRONG: flatten_3d
            # expects a 3D GRID array, not the 2D column carry, and misfired on
            # latlon where T is itself already column-shaped.)
            _cf_ovr = None
            if getattr(self, "_use_clubb_cloud_fraction", False) \
                    and cloud_fraction is not None:
                _cf_ovr = cloud_fraction.reshape(T_col.shape)
            from legoesm.atmosphere.physics.microphysics.config import (
                mg2_rel_ice_params,
            )
            _mg2_ice_dcs_m, _mg2_ice_slope_w = mg2_rel_ice_params(
                self.micro_config)
            cloud_props = compute_cloud_properties(
                T=T_col, p_full=p_full_col, q_v=q_v_col, dp=dp_col,
                config=cloud_config, q_cloud=q_c_col, q_ice=q_i_col,
                n_cloud=n_cloud_col, n_ice=n_ice_col,
                conv_precip=conv_precip_col,
                cloud_fraction_override=_cf_ovr,
                n_cloud_in_cloud=_nc_in_cloud,
                # CAM6 ice radius: the run's MG2 dcs + ice-slope width (same
                # helper and refusal as the MPAS radiation lane).
                mg2_ice_dcs_m=_mg2_ice_dcs_m,
                mg2_ice_slope_log_width=_mg2_ice_slope_w,
            )
            # ``to_rrtmg_kwargs`` builds the kwargs without
            # ``cloud_fraction`` (commit 4c9591bb, lost in AIMIP-#312
            # merge, restored iter-15) — see docstring for why.  iter-17
            # centralised the helper so the bug can't resurface at a
            # third call site.
            cloud_kwargs = cloud_props.to_rrtmg_kwargs()
            # Ice table size at the run's bulk ice density (rrtmgp rescales
            # r_eff by rho/500; exactly 1 at CAM's 500).
            cloud_kwargs["cloud_rho_ice"] = cloud_config.rho_cloud_ice

        # Issue #273 follow-up: when a column mesh is configured, place
        # every column-format input on the mesh's 'col' axis before
        # invoking the JIT'd radiation kernel.  Sharding propagates
        # through the kernel automatically because the column ops are
        # purely functional; the kernel itself is unchanged.
        if self.column_mesh is not None:
            from legoesm.parallel.column_shard import shard_columns
            n_dev = self.column_mesh.shape["col"]
            if ad.ncol % n_dev != 0:
                raise ValueError(
                    f"column_mesh requires ncol={ad.ncol} divisible by "
                    f"n_devices={n_dev}.  Pick an n_devices that divides "
                    f"the flattened column count, or disable "
                    f"shard_radiation_columns."
                )
            _shard = lambda x: (
                None if x is None else shard_columns(x, self.column_mesh)
            )
            T_col = _shard(T_col)
            p_full_col = _shard(p_full_col)
            p_half_col = _shard(p_half_col)
            q_v_col = _shard(q_v_col)
            T_sfc_col = _shard(T_sfc_col)
            lat_col = _shard(lat_col)
            lon_col = _shard(lon_col)
            albedo_col = _shard(albedo_col)
            emis_col = _shard(emis_col)
            open_ocean_frac_col = _shard(open_ocean_frac_col)
            o3_vmr_precomputed = _shard(o3_vmr_precomputed)
            aerosol_od_precomputed = _shard(aerosol_od_precomputed)
            aerosol_lw_od_precomputed = _shard(aerosol_lw_od_precomputed)
            aerosol_sw_volc_precomputed = _shard(aerosol_sw_volc_precomputed)
            o3_top_vmr = _shard(o3_top_vmr)
            if cloud_kwargs:
                cloud_kwargs = {k: (v if k == "cloud_rho_ice" else _shard(v))
                                for k, v in cloud_kwargs.items()}

        rad_out = self.radiation_fn(
            T_col, p_full_col, p_half_col, q_v_col,
            T_sfc_col, lat_col, lon_col,
packages/ml/legoesm/training/scale_build.py:1607:    per_year = [aerosol_layer_fields(
packages/tools/legoesm/forcing/surface_utils.py:808:def aerosol_layer_fields(acfg, day, lat_col, p_half_col, w_strat, *,
packages/coupler/legoesm/driver/model_driver.py:6160:         self._aerosol_sw_volc) = aerosol_layer_fields(

def ghg_array_to_dict(ghg_arr: jax.Array, ghg_keys: tuple[str, ...]) -> dict | None:
    """Reconstruct a GHG VMR dict from a flat array + key list.

    Parameters
    ----------
    ghg_arr : jax.Array, shape (n,)
    ghg_keys : tuple of str
        Species names in the same order used to build *ghg_arr*.

    Returns None when *ghg_keys* is empty (no GHG override).
    """
    if not ghg_keys:
        return None
    return {k: ghg_arr[i] for i, k in enumerate(ghg_keys)}


def pack_forcing(
    sst, sic, day_of_year, seconds_of_day,
    solar_weights, s_0, o3_vmr, aerosol_od,
    aerosol_lw_od=None,
    ghg_vmr=None,
    sfc_albedo_override=None,
    sfc_T_override=None,
    sfc_emissivity_override=None,
    sfc_shflx_override=None,
    sfc_lhflx_override=None,
    sfc_evap_override=None,
    sfc_taux_override=None,
    sfc_tauy_override=None,
    sfc_lw_up=None,
    sfc_sw_up=None,
    sfc_sw_down=None,
    land_frac=None,
    land_ml_params=None,
    aerosol_ccn_aod=None,
    o3_top_vmr=None,
    aerosol_sw_volc=None,
) -> SegmentForcing:
    """Pack per-segment forcing into a SegmentForcing pytree.

    Parameters
    ----------
    ghg_vmr : dict, jax.Array, or None
        GHG volume mixing ratios.  Accepts a dict (auto-converted via
        :func:`ghg_dict_to_array`), a pre-packed array, or None.
    sfc_albedo_override, sfc_T_override, sfc_emissivity_override : jax.Array or None
        Coupler-provided tile-blended surface albedo / skin temperature /
        emissivity for this segment (grid-shaped, like sst/sic).  ``None``
        (default) leaves the radiation's static internal blend untouched —
        byte-identical for AMIP / standalone runs.  The emissivity override
        carries the canopy's LAI-dependent eps_eff so the atmospheric LW
        boundary uses the same emissivity the land tile formed its LW_out with.
    sfc_albedo_override, sfc_T_override : jax.Array or None
        Coupler-provided tile-blended surface albedo / skin temperature for
        this segment (grid-shaped, like sst/sic).  ``None`` (default) leaves
        the radiation's static internal blend untouched — byte-identical for
        AMIP / standalone runs.
    sfc_shflx_override, sfc_lhflx_override : jax.Array or None
        Coupler-provided tile-blended sensible / latent heat flux [W/m2,
        positive UP] for this segment (grid-shaped, like sst/sic).  ``None``
        (default) leaves the atmosphere computing its own bulk surface fluxes —
        byte-identical for AMIP / standalone runs.  When present the atmosphere
        surface tendency consumes these instead, closing the air-sea budget.
    sfc_taux_override, sfc_tauy_override : jax.Array or None
        Prescribed surface momentum flux [Pa, stress ON THE ATMOSPHERE,
        opposite in sign to the wind] for this segment (grid-shaped, like
        sst/sic).  ``None`` (default) keeps the scheme's own / bulk surface
        drag.  When present the stress is the lower boundary condition of
        whatever turbulence scheme runs (folded into the kernel config), or
        the explicit lowest-layer momentum kick on the bulk-BL path.
754:                if config.tiling != (1, 1) and leaf.ndim < 3:
757:                if config.tiling != (1, 1) and leaf.ndim >= 3:
767:                    if leaf.shape[1] % tx != 0 or leaf.shape[2] % ty != 0:
784:            if leaf.ndim >= 2:
789:            # Spectral state arrays have shape (n_sh, nlev).
791:            if leaf.ndim == 2:
796:            if leaf.ndim == 1:
804:            if config.voronoi_dims is not None and leaf.ndim >= 1:
806:                if leaf.shape[0] in (nCells, nEdges):
853:    Pure / shape-only; no device or mesh needed.
857:    if not hasattr(arr, "shape") or arr.ndim < 1 or arr.shape[0] != N_FACES:
859:    if arr.ndim < 3:
867:    if arr.shape[1] == 4:
869:    a, b = arr.shape[1], arr.shape[2]
888:    inferring per-axis staggering from the shape.
900:    a, b = face_arr.shape[0], face_arr.shape[1]
939:    ha = face_arr.shape[0] - kt * nl
940:    hb = face_arr.shape[1] - kt * nl
943:            f"axes {face_arr.shape[:2]} are not kt*nl+2h padded for "
1089:            elif hasattr(val, "shape"):

exec
/usr/bin/bash -lc "rg -n 'GRID_SHAPED_FORCING_FIELDS|forcing\\._replace|forcing\\._asdict|forcing\\._fields' packages/coupler/legoesm/driver/compiled_segments.py packages/coupler/legoesm/driver/model_driver.py; sed -n '1850,1910p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; sed -n '440,510p' packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py; sed -n '3500,3540p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
packages/coupler/legoesm/driver/compiled_segments.py:876:    # A nested pytree, not a plain array; left out of GRID_SHAPED_FORCING_FIELDS
packages/coupler/legoesm/driver/compiled_segments.py:1086:GRID_SHAPED_FORCING_FIELDS = (
packages/coupler/legoesm/driver/compiled_segments.py:1144:    for name in GRID_SHAPED_FORCING_FIELDS:
packages/coupler/legoesm/driver/compiled_segments.py:1159:    return forcing._replace(**updates) if updates else forcing
        # Calendar time: prefer per-step TRACED forcing values (the MPAS
        # AMIP loop) over the static ``set_time`` closure.  The closure
        # cell is read at TRACE time inside the JIT'd dycore step, so a
        # multi-year MPAS run would otherwise integrate with the
        # insolation frozen at the initial day — no seasonal or diurnal
        # cycle (found 2026-06-10 while wiring CMIP6 forcing into the
        # standalone grid paths).
        _doy = _time["day_of_year"]
        _sod = _time["seconds_of_day"]
        if forcing is not None and forcing.get("day_of_year") is not None:
            _doy = forcing["day_of_year"]
        if forcing is not None and forcing.get("seconds_of_day") is not None:
            _sod = forcing["seconds_of_day"]
        # Transient solar (CMIP6 TSI + optional per-g-point spectral weights):
        # traced per-step forcing, same channel as T_sfc/o3/ghg above, so a
        # multi-year MPAS/standalone run follows the solar file WITHOUT
        # retracing (the coupled cube/lat-lon pipeline threads the equivalent
        # via SegmentForcing/current_s_0).  Absent keys -> the configured
        # static S_0 and the solver's default spectrum, byte-identical.
        _tsi_ext = forcing.get("tsi") if forcing is not None else None
        _ssf_ext = (forcing.get("solar_spectral_fraction")
                    if forcing is not None else None)

        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config,
            lon=lon,
            day_of_year=_doy,
            seconds_of_day=_sod,
        )
        if _tsi_ext is not None:
            # insolation is EXACTLY linear in S_0 in both the diurnal and
            # daily-mean branches of _compute_insolation, so a post-scale by
            # tsi/S_0_config is the transient-TSI application with no second
            # orbital computation.
            _S0_cfg = (radiation_config.rrtmgp.S_0
                       if radiation_config.scheme == "rrtmgp"
                       else radiation_config.gray.S_0)
            insol = insol * (_tsi_ext / _S0_cfg)

        # Flatten to column-major (ncol, nlev)
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = cos_sza.reshape(ncol) if cos_sza is not None else None

        # Per-column surface albedo (None -> the scalar config value).  A
        # scalar / 0-D forcing is passed through untouched so a uniform-albedo
        # caller keeps the legacy behaviour exactly.
        _alb_col = None
        if _alb_ext is not None:
            _alb_col = jnp.asarray(_alb_ext)
            if _alb_col.ndim > 0:
                if _alb_col.size != ncol:
                    raise ValueError(
                        f"forcing['sfc_albedo'] has {_alb_col.size} values "
                        f"but this grid has ncol={ncol} columns — refusing to "
                        f"broadcast a mismatched surface albedo into the "
        if a.ndim == 0:
            out[k] = v
            continue
        if a.shape[0] != ncol:
            raise ValueError(
                f"per-column solver kwarg {k!r} has leading dimension "
                f"{a.shape[0]}, expected ncol={ncol}; the subcolumn expansion "
                "cannot be applied safely"
            )
        out[k] = jnp.reshape(
            jnp.broadcast_to(a[None, ...], (n_sub,) + a.shape),
            (n_sub * ncol,) + a.shape[1:])
    return out


def average_output(result, n_sub: int, ncol: int):
    """Average every per-column field of a ``RadiationOutput`` over subcolumns.

    Fluxes and heating rates are averaged because the independent-column
    approximation IS the subcolumn-weighted mean of independent solutions.
    ``toa_insolation`` is identical across subcolumns, so averaging is a no-op
    on it rather than a special case.
    """
    return type(result)(*[
        None if f is None else average_over_subcolumns(f, n_sub, ncol)
        for f in result
    ])
    # Marker: consumes the per-step traced ``forcing`` dict (T_sfc /
    # o3_vmr / aerosol_od / ghg_vmr) — the spectral combined dispatcher
    # forwards ``forcing`` only to fns that advertise it.
    physics_fn._wants_forcing = True
    # Marker: READS ``phys_state`` (CLUBB cloud fraction + the CAM6 deepcu
    # carries) without writing a carry of its own, so the combined spectral
    # dispatcher forwards ``phys_state`` while keeping the single-return
    # contract.  Only set when the routing is active (byte-identical
    # otherwise: unmarked => not forwarded => RH cloud path unchanged).
    if use_clubb_cloud_fraction:
        physics_fn._wants_phys_state_ro = True
    return physics_fn


# _make_mpas_radiation is defined as an alias above (= _make_hydrostatic_radiation)


# Public promotions (CLAUDE.md cross-module private-import ratchet):
# these symbols are imported by sibling modules; expose a public alias
# so importers use the sanctioned public name (definitions keep the
# original underscore name for in-module callers).
compute_ozone_vmr = _compute_ozone_vmr

codex
The integration backend silently drops the field for gray and simple-LW radiation; the new refusal checks cover only the pipeline builders. The CPU tests are still running. The solver tests also use identical optics in every band, so they cannot detect a permuted g-point-to-band lookup.
exec
/usr/bin/bash -lc "sed -n '1230,1350p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; sed -n '19560,19610p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'forcing' packages/coupler/legoesm/driver/compiled_segments.py | tail -30; nl -ba packages/tools/legoesm/forcing/external.py | sed -n '1057,1088p'; nl -ba scripts/validate/probe_volcanic_sw_optics.py | sed -n '38,80p'" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
        solar_spectral_fraction=solar_spectral_fraction,
        **cloud_kwargs,
    )
    if aerosol_sw_volc is not None:
        _rad_kwargs["aerosol_volcanic_moments_sw"] = aerosol_sw_volc
    if cloud_config is not None:
        # One bulk ice density for the PSD radius and the optics size (CAM6).
        _rad_kwargs["cloud_rho_ice"] = cloud_config.rho_cloud_ice
    if clear_sky_toa:
        _rad_kwargs["clear_sky"] = True
    if o3_top_vmr is not None and o3_vmr_override is not None:
        _rad_kwargs["o3_top_vmr"] = o3_top_vmr
    # Maximum-random-overlap SUBCOLUMNS (opt-in).  The default path hands
    # every layer's GRID-MEAN water path to ONE homogeneous column, so cloud
    # spread thinly over many partly cloudy layers is solved as one deep
    # uniform cloud.  Measured against a Monte-Carlo independent-column
    # reference on a real model state, that costs ~30% of the cloud albedo
    # (0.3995 -> 0.2786) and ~18 W/m2 of OLR (219.2 -> 237.8, reference 239.0).
    # See clouds/subcolumns.py for the validation and for the closed-form
    # alternative that was measured and rejected.
    #
    # The kwargs are chosen BEFORE the solve so exactly ONE solve runs: an
    # earlier version solved the base column and then the subcolumns and threw
    # the base away, doubling the cost and compiling an extra shape.
    _ov = ("none" if cloud_props is None else
           getattr(cloud_config, "cloud_vertical_overlap_optics", "none"))
    _n_sub = 1
    if _ov == "max_random":
        from legoesm.atmosphere.physics.clouds import subcolumns as _sub
        _n_sub = int(getattr(cloud_config, "cloud_n_subcolumns",
                             _sub.N_SUBCOLUMNS_DEFAULT))
        _ncol = T.shape[0]
        # CAM draws different uniforms in every column; the per-column shift
        # (seeded from the bottom pressure, as for mcica) does the same, so a
        # layer with cf < 1/n_sub is cloudy in some columns instead of never.
        _shift = _sub.cam_seed_shift(p_full)
        _mask = _sub.generate_subcolumns(
            cloud_props.cloud_fraction, _n_sub, shift=_shift,
            width=float(cloud_config.cloud_subcolumn_mask_width))
        if cloud_config.cloud_optics_inhomogeneity == "two_region":
            # ecRad McICA gamma-PDF scaling of each cloudy cell's optical depth
            # (liquid and ice alike), replacing the mask in the paths.
            from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
                cam_interface_temperature)
            _od = _sub.ecrad_inhomogeneity_scaling(
                _mask, cloud_config.cloud_fsd, lat, p_half,
                cam_interface_temperature(T, p_full, p_half, sfc_temperature), shift=_shift)
            _lwp, _iwp = (
                jnp.reshape(_od * ic[None], (_n_sub * T.shape[0], T.shape[1]))
                for ic in _sub.in_cloud_paths(
                    cloud_props.cloud_fraction, cloud_props.lwp, cloud_props.iwp))
        else:
            _lwp, _iwp = _sub.subcolumn_paths(
                _mask, cloud_props.cloud_fraction, cloud_props.lwp, cloud_props.iwp)
        _rad_kwargs = _sub.expand_kwargs(_rad_kwargs, _n_sub, _ncol)
        _rad_kwargs["cloud_path_liq"] = _lwp
        _rad_kwargs["cloud_path_ice"] = _iwp
    elif _ov == "mcica":
        # CAM6 McICA: the same maximum-random generator, but ONE subcolumn per
        # g-point (drawn inside the solver) instead of n full solves.  The
        # solver takes IN-CLOUD paths (same floor as subcolumn_paths).
        from legoesm.atmosphere.physics.clouds import subcolumns as _sub
        (_rad_kwargs["cloud_path_liq"],
         _rad_kwargs["cloud_path_ice"]) = _sub.in_cloud_paths(
            cloud_props.cloud_fraction, cloud_props.lwp, cloud_props.iwp)
        _rad_kwargs["mcica_cloud_fraction"] = cloud_props.cloud_fraction
        _rad_kwargs["mcica_mask_width"] = float(cloud_config.cloud_subcolumn_mask_width)
        if cloud_props.cldfsnow is not None:
            # CAM6 snow (radiation.F90:928-993, 1057-1067): McICA samples
            # cldfprime = max(CLD, cldfsnow), and a cloudy subcolumn sees
            # (cldfsnow snow + CLD cloud)/cldfprime of each optical property.
            # Every optical property is linear in its path, so the weights go
            # on the in-cloud paths.
            _cld = cloud_props.cloud_fraction
            _cfp = jnp.maximum(_cld, cloud_props.cldfsnow)
            _pos = _cfp > 0.0
            _den = jnp.where(_pos, _cfp, 1.0)
            _w_cld = jnp.where(_pos, _cld / _den, 0.0)
            _w_snow = jnp.where(_pos, cloud_props.cldfsnow / _den, 0.0)
            _rad_kwargs["cloud_path_liq"] = _rad_kwargs["cloud_path_liq"] * _w_cld
            _rad_kwargs["cloud_path_ice"] = _rad_kwargs["cloud_path_ice"] * _w_cld
            _rad_kwargs["cloud_path_snow"] = cloud_props.icswp * _w_snow
            _rad_kwargs["cloud_snow_weight"] = _w_snow
            _rad_kwargs["cloud_d_eff_snow"] = cloud_props.des
            _rad_kwargs["snow_optics"] = _snow_optics_table(
                cloud_config.cam6_snow_optics_file)
            _rad_kwargs["mcica_cloud_fraction"] = _cfp
    elif _ov != "none":
        # Dispatch hardening: a typo must never silently run the legacy
        # single-column path (validated on the static config value).
        raise ValueError(
            f"unknown cloud_vertical_overlap_optics {_ov!r}; "
            "expected 'none', 'max_random' or 'mcica'"
        )
    if (cloud_props is not None and cloud_props.cldfsnow is not None
            and _ov != "mcica"):
        raise ValueError(
            "CloudConfig.cam6_snow_in_rad is wired for the CAM6 McICA "
            f"overlap only (cloud_vertical_overlap_optics='mcica'); got {_ov!r}.")

    # Column-chunk the rrtmgp solve when configured: the per-block body
    # compiles ONCE at ``column_chunk_size`` columns, capping the highly
    # super-linear rrtmgp XLA compile time at higher horizontal resolution.
    # Columns are physically independent, so this is numerically EXACT.
    _col_chunk = getattr(radiation_config.rrtmgp, "column_chunk_size", 0)
    if column_mesh is not None:
        if _col_chunk and _col_chunk > 0:
            raise ValueError(
                "column_mesh with rrtmgp column_chunk_size > 0 is not "
                "supported: the chunked solve scans over column blocks, so "
                "the column axis it would shard is a scan axis.")
        # Constrain the SOLVER's column axis (after the sub-column expansion):
        # a partition placed only on the physics inputs does not propagate into
        # the solver's working arrays, which keep full width on every device.
        from legoesm.atmosphere.physics.clouds.subcolumns import (
            PER_COLUMN_SOLVER_KEYS,
        )
        _cons = _column_constraint(column_mesh, _rad_kwargs["T"].shape[0])
        _rad_kwargs = {k: (_cons(v) if k in PER_COLUMN_SOLVER_KEYS else v)
                       for k, v in _rad_kwargs.items()}
    if _col_chunk and _col_chunk > 0:
                if _io_rank:
                    logger.info("operator-split SPMD run: %s (%.1fs)",
                                status, time.time() - t0)
                # Replicated verdict -> all ranks call in lockstep; the
                # helper's filesystem ops are root-gated internally.
                self._write_blowup_state(current_step, day)
                return status
            # current_step == seg_end_step now, so ``day`` is the segment-end day.
            self._current_day = day
            if self._segment_callback is not None:
                self._segment_callback(self, day, DT * seg_steps)
                # Fold any callback-applied change to the atm PROGNOSTIC state
                # back into the threaded sharded carry (a coupled/DA callback may
                # nudge state/moisture; _run_compiled picks this up by re-packing
                # from self.* each segment).  Re-shard the (possibly mutated)
                # replicated fields onto the carry's existing per-leaf sharding;
                # a no-op resharding of identical values when the callback does
                # not mutate.  The non-driver carry fields (held radiation, tke,
                # conv_prog) keep their threaded sharded values.
                if self._persistent_cgrid_active():
                    _u_fold = self.state.u_f.data
                    _v_fold = to_vface_lower(self.state.v_f.data)
                else:
                    _u_fold, _v_fold = self.state.u.data, self.state.v.data
                _fold = dict(
                    u=jax.device_put(_u_fold, carry.u.sharding),
                    v=jax.device_put(_v_fold, carry.v.sharding),
                    T=jax.device_put(self.state.T.data, carry.T.sharding),
                    p_s=jax.device_put(self.state.p_s.data, carry.p_s.sharding),
                    q_v=jax.device_put(self.q_v, carry.q_v.sharding),
                    q_c=jax.device_put(self.q_c, carry.q_c.sharding),
                    q_r=jax.device_put(self.q_r, carry.q_r.sharding))
                # Double-moment tracers: a DA/coupling callback may nudge
                # self.tracers[q_i…N_i]; fold those back too (same repack-from-self
                # _run_compiled does), else a double-moment run diverges after a
                # tracer-mutating callback (codex #789-F4).  Only carry-threaded
                # (non-None) fields are re-shardable.
                if isinstance(self.tracers, dict):
                    for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                        _cv, _sv = getattr(carry, _nm), self.tracers.get(_nm)
                        if _cv is not None and _sv is not None:
                            _fold[_nm] = jax.device_put(_sv, _cv.sharding)
                carry = carry._replace(**_fold)

        if _io_rank:
            # Rank-0 completion summary. self.state is the last segment's
            # REPLICATED gathered copy, so the max reductions are valid on any
            # rank; the selfspawn route-B gate greps this line + the finite
            # magnitudes to confirm the federation integrated without a hang.
            _maxT = float(jnp.abs(self.state.T.data).max())
            _maxU = float(jnp.abs(self.cell_centre_state().u.data).max())
3554:            _sharding_signature(forcing) if pin else (),
3559:            # ``(carry, forcing)`` with NO static ``n_steps`` arg, so they
3561:            # scan kinds take ``(carry, n_steps_static, forcing)``.  Either
3562:            # way the DYNAMIC args are exactly ``(carry, forcing)``, so the
3587:                # the ``(carry, n_steps_static, forcing)`` signature.
3590:                # ``(carry, forcing)`` host kinds already have exactly
3603:                    _sharding_tree(carry), _sharding_tree(forcing),
3613:                        forcing: SegmentForcing) -> SegmentCarry:
3626:        forcing : SegmentForcing
3627:            Per-segment external forcing (SST, SIC, solar, ozone, etc.).
3628:            Traced as a dynamic argument — changing forcing values does
3641:                return _get_sharded_jit("subcycled", True, carry, forcing)(
3642:                    carry, n_steps, forcing, lead,
3644:            return _run_subcycled_jit(carry, n_steps, forcing, lead)
3646:            return _get_sharded_jit("single", True, carry, forcing)(
3647:                carry, n_steps, forcing,
3649:        return _run_single_jit(carry, n_steps, forcing)
3652:                    forcing: SegmentForcing) -> SegmentCarry:
3678:            return _get_sharded_jit("single", False, carry, forcing)(
3679:                carry, n_steps, forcing,
3681:        return _run_single(carry, n_steps, forcing)
3689:    # ``.sharding`` off the committed carry/forcing — never re-derived).
3698:                                   forcing: SegmentForcing) -> SegmentCarry:
3701:                return _get_sharded_jit("norad_scan", True, carry, forcing)(
3702:                    carry, forcing,
3704:            return _run_norad_scan_jit(carry, forcing)
3707:                            forcing: SegmentForcing) -> SegmentCarry:
3710:                return _get_sharded_jit("rad", True, carry, forcing)(
3711:                    carry, forcing,
3713:            return _run_rad_jit(carry, forcing)
  1057	    tau = ext * dz_km[None, None, :, None]
  1058	    ssa = np.clip(omega, 0.0, 1.0)
  1059	    moments = np.stack([tau, tau * ssa, tau * ssa * np.clip(g, -1.0, 1.0)],
  1060	                       axis=-2)                       # (t, lat, alt, 3, band)
  1061	    # um -> cm-1: wn = 1e4 / wl.
  1062	    band_wn = np.stack([1.0e4 / np.maximum(wl1, wl2),
  1063	                        1.0e4 / np.minimum(wl1, wl2)], axis=-1)
  1064	    srt = np.argsort(band_wn[:, 0])
  1065	    band_wn = band_wn[srt]
  1066	    ref = np.asarray(_RRTMGP_SW_BAND_LIMITS_CM1)
  1067	    for a, b in _RRTMG_RRTMGP_EDGE_DIFF_CM1:
  1068	        ref = np.where(ref == a, b, ref)
  1069	    if band_wn.shape != ref.shape or not np.allclose(
  1070	            band_wn, ref, rtol=_VOLC_SW_EDGE_RTOL, atol=0.0):
  1071	        raise ValueError(
  1072	            f"Volcanic file {path!r} solar bands {band_wn.tolist()} [cm-1] do "
  1073	            "not map 1:1 onto the RRTMGP shortwave bands.")
  1074	    moments = moments[..., srt][:, :, ::-1].astype(np.float32)
  1075	    return mid_days, first_date, lat, moments, p_edges, band_wn
  1076	
  1077	
  1078	@lru_cache(maxsize=16)
  1079	def _load_volcanic_auto(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  1080	    """Dispatch ``path`` to the right volcanic loader.
  1081	
  1082	    Returns a ``(mid_days, lat, data)`` tuple (the anchor-free form of
  1083	    :func:`_load_monthly_zonal_anchored`) regardless of which on-disk
  1084	    schema the file follows.  See :func:`_load_volcanic_auto_anchored` for a
  1085	    variant that also returns the file's CF anchor (used by the
  1086	    multi-year non-cyclic dispatch in :func:`get_aerosol_at_time`).
  1087	    """
  1088	    mid_days, _first_date, lat, data = _load_volcanic_auto_anchored(path)
    38	    p_full = jnp.asarray(sig.pressure_at_full(p_s)).reshape(1, -1)
    39	    T = jnp.maximum(300.0 * (p_full / 1.0e5) ** 0.19, 200.0)
    40	    q_v = jnp.clip(0.018 * (p_full / 1.0e5) ** 3, 3.0e-6, None)
    41	    lat = jnp.array([0.0])
    42	    day = 365.0 * (year - 1979) + 30.4 * (month - 1) + 15.0
    43	    acfg = AerosolConfig(enabled=True, volcanic_enabled=True,
    44	                         volcanic_path=path, start_year=1979)
    45	    mom, p_edges, band_wn = get_aerosol_sw_volcanic_optics_at_time(acfg, day, lat)
    46	    w = stratospheric_layer_weight(p_half, lat, p_full, T)
    47	    nsrc = mom.shape[1]
    48	    flat = jnp.moveaxis(jnp.asarray(mom, jnp.float64).reshape(1, nsrc, 42), -1, 0)
    49	    placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
    50	                      in_axes=(0, None, None))(flat, jnp.asarray(p_edges), p_half) * w
    51	    volc = jnp.moveaxis(placed, 0, -1).reshape(1, -1, 3, 14)
    52	    wl = 1.0e4 / band_wn                       # cm-1 -> um, (14, 2)
    53	    wts = _planck_band_weights(wl[:, 1], wl[:, 0], _SOLAR_PLANCK_TEMP_K)
    54	    tau_gray = jnp.tensordot(volc[:, :, 0, :], jnp.asarray(wts / wts.sum()),
    55	                             axes=(2, 0))
    56	
    57	    lw, sw = gas_table_files("full")
    58	    cfg = RRTMGPConfig(lw_gas_file=lw, sw_gas_file=sw)
    59	    solver = RRTMGP.from_legoesm_config(cfg)
    60	    kw = dict(T=T, p_full=p_full, p_half=p_half,
    61	              sfc_temperature=jnp.array([300.0]), q_v=q_v,
    62	              cos_zenith=jnp.array([0.5]), sfc_albedo=jnp.array([0.07]))
    63	    arms = {
    64	        "none": solver.solve_columns(**kw),
    65	        "grey": solver.solve_columns(**kw, aerosol_optical_depth=tau_gray),
    66	        "file": solver.solve_columns(**kw, aerosol_optical_depth=jnp.zeros_like(tau_gray),
    67	                                     aerosol_volcanic_moments_sw=volc),
    68	    }
    69	    k100 = int(jnp.argmin(jnp.abs(p_half[0] - 1.0e4)))
    70	    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
    71	                         capture_output=True, text=True).stdout.strip()
    72	    print(f"file={path} {year}-{month:02d} lat=0 cosz=0.5 code={sha}")
    73	    tau_vis = float(volc[0, :, 0, 10].sum())
    74	    ssa_vis = float(volc[0, :, 1, 10].sum() / max(tau_vis, 1e-30))
    75	    print(f"volcanic column tau 16000-22650 cm-1 {tau_vis:.4f} ssa {ssa_vis:.4f};"
    76	          f" gray tau {float(tau_gray.sum()):.4f}")
    77	    for k, o in arms.items():
    78	        net = o.sw_flux_down[0] - o.sw_flux_up[0]
    79	        print(f"{k:5s} toa_up {float(o.sw_flux_up[0, 0]):8.3f}  sfc_down "
    80	              f"{float(o.sw_flux_down[0, -1]):8.3f}  abs_above_100hPa "

exec
/usr/bin/bash -lc "sed -n '958,998p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; rg -n 'class RadiationConfig|scheme:|simple_lw:' packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py; sed -n '1320,1340p' packages/core/legoesm/grids/voronoi.py; git diff --numstat HEAD" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 70ms:
    sfc_temperature: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    insolation: jnp.ndarray,
    cos_sza: jnp.ndarray | None = None,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
    open_ocean_frac: jnp.ndarray | None = None,
    q_cloud: jnp.ndarray | None = None,
    q_ice: jnp.ndarray | None = None,
    n_ice: jnp.ndarray | None = None,
    n_cloud: jnp.ndarray | None = None,
    n_cloud_in_cloud: bool = False,
    ghg_vmr_override: dict | None = None,
    f_day: jnp.ndarray | None = None,
    rrtmgp_solver=None,
    lon: jnp.ndarray | None = None,
    ml_ozone_coefs=None,
    o3_vmr_override: jnp.ndarray | None = None,
    o3_top_vmr: jnp.ndarray | None = None,
    aerosol_od: jnp.ndarray | None = None,
    aerosol_lw_od: jnp.ndarray | None = None,
    aerosol_sw_volc: jnp.ndarray | None = None,
    solar_spectral_fraction: jnp.ndarray | None = None,
    eccf: float | jnp.ndarray = 1.0,
    tsi: jnp.ndarray | None = None,
    cloud_fraction_override: jnp.ndarray | None = None,
    conv_precip: jnp.ndarray | None = None,
    conv_mass_flux_up: jnp.ndarray | None = None,
    conv_icwmr: jnp.ndarray | None = None,
    clear_sky_toa: bool = False,
    column_mesh=None,
    clubb_config=None,
    mg2_ice_dcs_m: float | None = None,
    mg2_ice_slope_log_width: float | None = None,
    q_snow: jnp.ndarray | None = None,
    n_snow: jnp.ndarray | None = None,
    n_cloud_specified: float | None = None,
):
    """Call configured radiation backend with a unified integration interface.

382:class RadiationConfig(NamedTuple):
390:        Active radiation scheme: "gray", "rrtmgp", "mc3d", "simple_lw" or
420:    scheme: str = "gray"
425:    simple_lw: SimpleLWConfig = SimpleLWConfig()
433:    cloud_scheme: str = "none"
        linearly, so the achieved refinement contrast grows with
        ``lloyd_iterations``; for strong/precise variable resolution prefer a
        JIGSAW-built mesh loaded via :func:`load_mpas_mesh`.

    Returns
    -------
    VoronoiMesh
    """
    if subdivision_level > _BIG_MESH_MAX_LEVEL:
        n_cells = 10 * 4 ** subdivision_level + 2
        raise ValueError(
            f"subdivision_level={subdivision_level} would create {n_cells:.2e} "
            f"cells — unsupported (hard cap {_BIG_MESH_MAX_LEVEL}; the scipy "
            f"SphericalVoronoi build path does not scale there). For higher "
            f"resolutions, use load_mpas_mesh() with a pre-built mesh file."
        )
    # Levels above the routine cap are CACHE-OR-PREWARM only (codex round-19
    # design): a valid cache hit is always admissible; a MISS is refused
    # unless this process is the designated prewarmer — otherwise an N-rank
    # MPI launch would have every rank silently rebuild for hours (build is
    # ~87 s/Lloyd-iteration at subdiv-8, ~4x that at 9). The prewarm path is
1	1	packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
18	0	packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
35	0	packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
28	3	packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
17	1	packages/coupler/legoesm/driver/compiled_segments.py
13	1	packages/coupler/legoesm/driver/model_driver.py
18	3	packages/coupler/legoesm/driver/physics_pipeline.py
7	2	packages/ml/legoesm/training/scale_build.py
161	45	packages/tools/legoesm/forcing/external.py
32	19	packages/tools/legoesm/forcing/surface_utils.py
89	0	scripts/validate/probe_volcanic_sw_optics.py
26	22	tests/unit/test_aerosol_band_ccn.py
1	1	tests/unit/test_cross_lane_forcing.py
242	0	tests/unit/test_volcanic_sw_optics.py
1	1	tests/unit/test_wb_amip_deck_params.py

exec
/usr/bin/bash -lc "sed -n '948,961p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; rg -n 'scheme ==|scheme in|none|mc3d' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | head -35; sed -n '300,355p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; rg -n 'def .*radiation|scheme.*none' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | tail -15" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
    from legoesm.atmosphere.physics.radiation.rrtmgp.optics.mitchell_ice_optics import (
        load_mitchell_ice_optics)
    return load_mitchell_ice_optics(path)


def _call_radiation_backend(
    radiation_config: RadiationConfig,
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    insolation: jnp.ndarray,
142:        # simple_lw / mc3d take no surface temperature or albedo override,
172:                   if radiation_config.scheme == "rrtmgp"
314:    if config.scheme == "gray":
316:    elif config.scheme == "simple_lw":
324:    elif config.scheme == "ecrad_ifs":
329:    elif config.scheme == "rrtmgp":
366:    S_0 = config.rrtmgp.S_0 if config.scheme == "rrtmgp" else config.gray.S_0
517:    if ozone_config.source == "none":
564:        f"Choose from 'standard', 'analytical', 'mls', 'none', 'ml'."
681:    if radiation_config.cloud_scheme != "none" and (
695:    if radiation_config.cloud_scheme == "none" or not getattr(
701:            "Morrison/MG2 microphysics config; none was passed to the "
1060:    if clear_sky_toa and radiation_config.scheme in ("simple_lw", "gray"):
1069:    if radiation_config.scheme == "simple_lw":
1075:    if radiation_config.scheme == "gray":
1161:    # with cloud_scheme="none" this branch never runs.
1164:    if radiation_config.cloud_scheme != "none":
1192:                                             "none") == "max_random"),
1254:    _ov = ("none" if cloud_props is None else
1255:           getattr(cloud_config, "cloud_vertical_overlap_optics", "none"))
1317:    elif _ov != "none":
1322:            "expected 'none', 'max_random' or 'mcica'"
1397:    to ``solve_columns`` (the ``cloud_scheme != "none"`` branch of
1414:    # mc3d shares the RRTMGP optics path (Phase 2b: it calls solve_columns for
1418:        radiation_config.scheme in ("rrtmgp", "mc3d")
1419:        and radiation_config.cloud_scheme != "none"
1429:            "or cloud_scheme='none' for a genuine clear-sky run."
1507:    if radiation_config.scheme == "mc3d" and model_type != "plane":
1509:            "scheme='mc3d' (3D Monte-Carlo ray tracing) is only wired for "
1543:    # Load heavy/static RRTMGP optics once outside model JIT traces. mc3d also
1546:    # mc3d used to fall back silently to gray optics, which also hid config
1549:    if radiation_config.scheme in ("rrtmgp", "mc3d"):
1565:    if radiation_config.ozone.source == "ml" and radiation_config.scheme == "rrtmgp":
1582:    if radiation_config.scheme == "ecrad_ifs":
1590:                "cloud_fraction_ifs (microphysics='ifs_cloudsc'); none is "
    Returns ``(_time, set_time)`` where *_time* is the mutable dict and
    *set_time* is a function that updates it in-place.
    """
    _time = {"day_of_year": 80.0, "seconds_of_day": 43200.0}  # coeff-ok: idealized time (spring equinox, noon)

    def set_time(day_of_year: float, seconds_of_day: float):
        _time["day_of_year"] = day_of_year
        _time["seconds_of_day"] = seconds_of_day

    return _time, set_time


def _get_radiation_fn(config: RadiationConfig):
    """Select the radiation backend based on config.scheme."""
    if config.scheme == "gray":
        return gray_radiation, config.gray
    elif config.scheme == "simple_lw":
        # Handled by its own branch in _call_radiation_backend: it needs cloud
        # condensate, density and heights, not the (T, p, lat, insolation)
        # signature the two-stream backends share.
        raise ValueError(
            "radiation scheme 'simple_lw' does not go through "
            "_get_radiation_fn; _call_radiation_backend handles it directly."
        )
    elif config.scheme == "ecrad_ifs":
        # Its own builder branch (between-call carry, RADHEATN every step).
        raise ValueError(
            "radiation scheme 'ecrad_ifs' does not go through "
            "_get_radiation_fn; the hydrostatic/MPAS physics_fn handles it.")
    elif config.scheme == "rrtmgp":
        from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
            rrtmgp_radiation,
        )
        return rrtmgp_radiation, config.rrtmgp
    else:
        raise ValueError(f"Unknown radiation scheme: {config.scheme!r}")


def _compute_insolation(
    lat: jnp.ndarray,
    config: RadiationConfig,
    lon: jnp.ndarray | None = None,
    day_of_year: float = 80.0,  # coeff-ok: idealized default (spring equinox)
    seconds_of_day: float = 43200.0,  # coeff-ok: idealized default (local noon)
) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
    """Compute TOA insolation and (optionally) cosine zenith angle.

    Returns
    -------
    insolation : jnp.ndarray
        TOA downward SW flux per column [W/m^2].
    cos_sza : jnp.ndarray or None
        Cosine of solar zenith angle (clipped >=0) per column.
        Only returned when ``config.diurnal_cycle`` is True.
    f_day : jnp.ndarray or None
        Daylight fraction per column.  Returned for non-diurnal daily-mean
1651:    if radiation_config.cloud_scheme == "none":
1660:def _make_hydrostatic_radiation(
1713:        radiation_config.cloud_scheme != "none"
1726:        radiation_config.cloud_scheme != "none"
1753:    # config variant for the second radiation pass.  ``cloud_scheme="none"``
1764:        radiation_config._replace(cloud_scheme="none", cloud_config=None)
2166:        # forcings, clouds OFF (``_clr_sky_cfg`` has cloud_scheme="none" and
2310:def _make_ecrad_ifs_step(radiation_config, full_call, column_mesh):
2430:def _make_nonhydrostatic_radiation(
2685:    # cloud scheme is 'none'). Only built when the RRTMGP solver is present.
2687:    if rrtmgp_solver is not None and radiation_config.cloud_scheme != "none":
2811:def _make_plane_radiation(
2997:def _make_mpas_nh_radiation(
3183:def _make_spectral_pe_radiation(
3226:        radiation_config.cloud_scheme != "none"

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'PY'
import sys,glob,os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
sys.path[:0]=glob.glob('packages/*')+['src']
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation import integration as i
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
kw=dict(T=jnp.array([[250.,280.]]),p_full=jnp.array([[25000.,75000.]]),p_half=jnp.array([[100.,50000.,100000.]]),sfc_temperature=jnp.array([290.]),lat=jnp.array([0.]),q_v=jnp.ones((1,2))*.001,insolation=jnp.array([300.]),aerosol_sw_volc=jnp.ones((1,2,3,14)))
for s in ('gray','simple_lw'):
 out=i._call_radiation_backend(RadiationConfig(scheme=s),**kw)
 print(s, 'accepted nonzero volcanic field; returned',type(out).__name__)
PY" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 5880ms:
gray accepted nonzero volcanic field; returned RadiationOutput
simple_lw accepted nonzero volcanic field; returned RadiationOutput

exec
/usr/bin/bash -lc "sed -n '680,752p' packages/tools/legoesm/forcing/surface_utils.py; sed -n '2940,3030p' packages/coupler/legoesm/driver/compiled_segments.py; sed -n '1570,1620p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; sed -n '2310,2335p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
    r = dt_s / C_areal
    T_new = (T_skin + r * (F_net_down_W_m2 + g_cond * T_freeze_K)) \
        / (1.0 + r * g_cond)
    if melt_cap_width_K == 0.0:
        T_new = jnp.clip(T_new, _ICE_SKIN_FLOOR_K, T_melt_surface_K)
    else:
        from legoesm.core.smooth import smooth_min
        T_new = smooth_min(jnp.maximum(T_new, _ICE_SKIN_FLOOR_K),
                           T_melt_surface_K, 1.0 / melt_cap_width_K)
    return jnp.where(sic > 0.0, T_new, T_freeze_K)


def advance_ice_skin(
    T_skin: jnp.ndarray,
    sw_net: jnp.ndarray,
    lw_net: jnp.ndarray,
    shflx: jnp.ndarray | None,
    lhflx: jnp.ndarray | None,
    sic: jnp.ndarray,
    dt_s: float,
    h_ice_m: float,
    melt_cap_width_K: float = 0.0,
) -> jnp.ndarray:
    """One step of the prognostic ice skin from a lane's surface fluxes.

    Grid-agnostic per-column physics shared by every lane: ``F_net_down =
    sw_net + lw_net - shflx - lhflx`` (sw/lw net [W/m^2, +into surface];
    turbulent fluxes [W/m^2, +upward]; an absent turbulent flux contributes
    nothing), then :func:`prognostic_ice_skin_temperature`.  Inputs are
    flattened to columns and the result has the shape of ``T_skin``.
    """
    shape = jnp.shape(T_skin)
    f_net = (jnp.asarray(sw_net).reshape(-1) + jnp.asarray(lw_net).reshape(-1))
    if shflx is not None:
        f_net = f_net - jnp.asarray(shflx).reshape(-1)
    if lhflx is not None:
        f_net = f_net - jnp.asarray(lhflx).reshape(-1)
    out = prognostic_ice_skin_temperature(
        jnp.asarray(T_skin).reshape(-1), f_net, jnp.asarray(sic).reshape(-1),
        dt_s=dt_s, h_ice_m=h_ice_m, melt_cap_width_K=melt_cap_width_K)
    return out.reshape(shape)

# NOTE: the model has OTHER tropopauses (clouds/cloud_fraction.py): the CAM6
# aist cloud switch (rhminis/rhmaxis) uses the analytic 25000 - 15000
# cos^2(lat) Pa surface (cam6_tropopause_analytic), and CLUBB's clubbtop uses
# CAM's chemistry tropopause (cam_chem_tropopause_level: twmo + polar rule,
# that surface as fallback).  This WMO locator places volcanic aerosol only;
# unifying them is an open follow-up.
def wmo_tropopause_pressure(p_full_col: jnp.ndarray, temp_col: jnp.ndarray,
                            p_bot: float = 5.0e4,
                            p_top: float = 5.0e3) -> jnp.ndarray:
    """WMO (1957) lapse-rate tropopause pressure [Pa] per column: the lowest
    full level in [p_top, p_bot] [Pa] whose lapse rate to the next level up
    is <= 2 K/km and whose mean lapse rate to every higher level within 2 km
    (and to 2 km itself) stays <= 2 K/km.
    NaN where no level qualifies -- incl. any NaN T in the column (it
    poisons z above and every within-2-km test below).
    ``p_full_col``/``temp_col`` (ncol, nlev) in either vertical order;
    heights by hypsometric integration of T.

    ponytail: the tropopause snaps to a full level (no sub-layer interpolation
    of the 2 K/km crossing as in CAM's twmo), i.e. it is located to within one
    layer -- all that the 1/0.5/0 layer weighting below resolves; dry T (not
    virtual) in the hypsometric dz (<0.1% at these heights); the within-2-km
    test loops over all nlev level offsets, O(nlev^2) per column (daily).
    """
    order = jnp.argsort(-p_full_col, axis=1)  # surface first
    p = jnp.take_along_axis(p_full_col, order, axis=1)
    temp = jnp.take_along_axis(temp_col, order, axis=1)
    dz = (constants.R_d / constants.g) * 0.5 * (temp[:, :-1] + temp[:, 1:]) \
        * jnp.log(p[:, :-1] / p[:, 1:])
    z = jnp.concatenate([jnp.zeros_like(dz[:, :1]),
                         jnp.cumsum(dz, axis=1)], axis=1)
        return _scan_no_rad(carry, tail)

    def _run_single(carry: SegmentCarry, n_steps: int,
                    forcing: SegmentForcing) -> SegmentCarry:
        """Legacy single-scan body.

        Used when ``step_unified_no_rad`` is not provided, when
        ``rad_update_steps <= 1`` (always-rad, no subcycle benefit
        beyond what the rad-only variant already gives), or when
        ``n_steps`` does not divide evenly by ``rad_update_steps``.
        """
        _step_fn = _make_single_step(forcing)
        if _resolve_checkpoint(gradient_checkpoint, n_steps):
            # Long differentiated rollout -> nested (sqrt-N) checkpointing so the
            # reverse-mode carry storage is O(sqrt(n_steps)), not O(n_steps) which
            # OOMs (#841).  Short segments keep the cheap per-step remat + one scan.
            if n_steps >= _NESTED_CKPT_STEPS:
                return _sqrt_checkpointed_scan(_step_fn, carry, n_steps)
            _step_fn = jax.checkpoint(_step_fn, prevent_cse=False)
        final_carry, _ = jax.lax.scan(_step_fn, carry, None, length=n_steps)
        return final_carry

    # ------------------------------------------------------------------
    # Un-fused radiation host kernels (ExperimentConfig.unfused_radiation;
    # PRODUCTION ``_run_compiled`` ONLY).  A ``jax.jit`` placed INSIDE a
    # ``lax.scan`` body is INLINED by XLA into ONE giant executable — so
    # even the cond-free ``_run_subcycled`` above still inlines the rrtmgp
    # branch and pays the ~3h XLA compile.  Lifting the radiation-cycle
    # loop to the HOST (Python) makes the no-rad dynamics+physics scan and
    # the rrtmgp radiation compile as TWO SEPARATE executables.  These two
    # functions are the host-callable halves; the driver alternates them
    # (``run_norad_scan`` × ``run_rad``) once per radiation cycle.  Both
    # are gated OFF by default: built only when ``step_unified_no_rad`` and
    # ``pipeline`` are supplied AND ``rad_update_steps > 1``.
    # ------------------------------------------------------------------
    _unfused_available = (
        step_unified_no_rad is not None
        and pipeline is not None
        and rad_update_steps > 1
    )

    def _run_norad_scan(carry: SegmentCarry,
                        forcing: SegmentForcing) -> SegmentCarry:
        """Advance exactly ``rad_update_steps`` no-rad steps as ONE scan.

        Pure dynamics+physics with HELD radiation (``step_unified_no_rad``
        — no fresh rrtmgp call), so this HLO contains NO rrtmgp.  Identical
        per-step body to ``_run_subcycled``'s ``body_no_rad``; only the
        compile boundary differs (host-level, not inlined in an outer
        scan).
        """
        body_no_rad = _make_single_step(forcing, step_fn=step_unified_no_rad)
        # This unfused no-rad scan has length rad_update_steps (called once per
        # radiation cycle); resolve the checkpoint policy against THAT length so
        # a long rad_update_steps still auto-checkpoints under default None.
        if _resolve_checkpoint(gradient_checkpoint, rad_update_steps):
            body_no_rad = jax.checkpoint(body_no_rad, prevent_cse=False)
        final_carry, _ = jax.lax.scan(
            body_no_rad, carry, None, length=rad_update_steps,
        )
        return final_carry

    def _run_rad(carry: SegmentCarry,
                 forcing: SegmentForcing) -> SegmentCarry:
        """Recompute FRESH held radiation + T_land from the carry state.

        Radiation ONLY (no dynamics/physics), so this HLO is rrtmgp-only.
        Calls the SAME ``pipeline.compute_radiation_core`` the fused
        ``_rad_branch`` calls (physics_pipeline.py), threading ``forcing``
        identically (so ``sfc_albedo_override`` / ``sfc_T_override`` /
        column-shard handling is byte-identical), and writes the 6 held
        flux fields + ``T_land`` back at owned indices.  Every other carry
        field passes through unchanged — the post-cycle dynamics+physics
        state is untouched here.

        Owned-face extract/write-back MIRROR ``_make_single_step``
        (compiled_segments.py): on the MPI replicated-dynamics path
        physics/radiation operate on owned columns only and write back at
        ``owned_face_ids``; on the single-rank path the whole grid is
        owned.
        """
        # Reconstruct GHG VMR dict from the forcing array + static keys,
        # exactly as ``_make_single_step`` does (keys captured in closure;
        # values dynamic).
        _ghg_vmr_override = ghg_array_to_dict(forcing.ghg_vmr, _ghg_keys)
        # #1028: the surface scheme inside radiation reads winds at cell
        # centres.  With persistent D-grid winds the carry holds corner arrays,
        # so take a read-only cell-centre view; off that lane these ARE
        # carry.u/carry.v and the calls below are byte-identical.
        _u_phys, _v_phys = (
            dgrid_to_center_vector(carry.u, carry.v) if persistent_dgrid
            )
        from legoesm.atmosphere.physics.radiation.ozone_ml import (
            load_ml_ozone_coefficients,
        )
        ml_ozone_coefs = load_ml_ozone_coefficients(
            radiation_config.ozone.ml_weights_path
        )

    if cloud_fraction_tracer is not None and model_type != "mpas":
        raise ValueError(
            "cloud_fraction_tracer (the advected IFS cloud fraction) is wired "
            f"on the MPAS lane only; got model_type={model_type!r}.")
    if radiation_config.scheme == "ecrad_ifs":
        if model_type != "mpas":
            raise ValueError(
                "radiation scheme 'ecrad_ifs' is wired on the MPAS column lane "
                f"only (vote L2a); got model_type={model_type!r}.")
        if cloud_fraction_tracer is None:
            raise ValueError(
                "radiation scheme 'ecrad_ifs' reads CLOUDSC's advected "
                "cloud_fraction_ifs (microphysics='ifs_cloudsc'); none is "
                "routed.")
        if not radiation_config.diurnal_cycle:
            raise ValueError(
                "radiation scheme 'ecrad_ifs' needs the instantaneous solar "
                "zenith angle (RadiationConfig.diurnal_cycle=True).")
    if model_type == "hydrostatic":
        return _make_hydrostatic_radiation(radiation_config, rrtmgp_solver,
                                            ml_ozone_coefs=ml_ozone_coefs,
                                            column_mesh=column_mesh,
                                            nc_from_aerosol=nc_from_aerosol,
                                            activation_config=activation_config,
                                            use_clubb_cloud_fraction=use_clubb_cloud_fraction,
                                            clubb_config=clubb_config,
                                            micro_leaf_config=micro_leaf_config)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_radiation(radiation_config, rrtmgp_solver,
                                               ml_ozone_coefs=ml_ozone_coefs)
    elif model_type == "plane":
        return _make_plane_radiation(radiation_config, rrtmgp_solver,
                                     ml_ozone_coefs=ml_ozone_coefs)
    elif model_type == "mpas_nh":
        return _make_mpas_nh_radiation(radiation_config, rrtmgp_solver,
                                       ml_ozone_coefs=ml_ozone_coefs)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_radiation(
            radiation_config, rrtmgp_solver,
            ml_ozone_coefs=ml_ozone_coefs,
            sfc_albedo_override=sfc_albedo_override,
            sfc_emissivity_override=sfc_emissivity_override,
            use_clubb_cloud_fraction=use_clubb_cloud_fraction,
def _make_ecrad_ifs_step(radiation_config, full_call, column_mesh):
    """The ecrad_ifs column step of the hydrostatic/MPAS physics_fn: the
    shared column preparation hands it T, pressures, skin T, mu0, surface
    albedo/emissivity and the condensate; it reads q_r, q_s, the land
    fraction, the carry and the seed time, runs
    :func:`~legoesm.atmosphere.physics.radiation.ecrad_ifs.lane.step` and
    packs the tendencies.  Every missing input raises (vote S6a)."""
    from legoesm.atmosphere.physics.convection.integration import (
        land_fraction_for_columns)
    from legoesm.atmosphere.physics.radiation.ecrad_ifs import lane as _lane
    cfg = radiation_config.ecrad_ifs
    _rcfg = radiation_config.rrtmgp

    def run(state, grid_or_mesh, phys_state, forcing, shape_3d, shape_2d, *,
            ncol, nlev, T_col, p_full_col, p_half_col, T_sfc_col, lat_col,
            lon_col, cos_sza_col, alb_col, ocn_frac_col, emis_col, q_v_col,
            q_cloud_col, q_ice_col, cf_col, eccf, tsi, o3_vmr_col, ghg_vmr):
        _need = ("q_v", "q_c", "q_i", "q_r", "q_s", "cloud_fraction_ifs")
        _missing = [n for n in _need
                    if state.tracers is None or n not in state.tracers]
        if _missing:
            raise ValueError(
                f"radiation 'ecrad_ifs' needs the tracers {_need}; missing "
                f"{_missing}.")
        if cos_sza_col is None:
            raise ValueError("radiation 'ecrad_ifs' needs cos(solar zenith).")

exec
/usr/bin/bash -lc "rg -n 'def place_stratospheric_aod_profile_to_layers' packages/tools/legoesm/forcing/surface_utils.py; sed -n '570,637p' packages/tools/legoesm/forcing/surface_utils.py; rg -n '"'^def load_era5_samples|def _load_era5'"' packages/ml/legoesm/training/scale_build.py; sed -n '1955,1990p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; git status --porcelain | tail -8" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 220ms:
494:def place_stratospheric_aod_profile_to_layers(
    return jnp.clip(_cdf_at(p_hi) - _cdf_at(p_lo), 0.0, None)


# --- prognostic sea-ice skin temperature (Semtner 1976 zero-layer + thermal
#     inertia; conductivity Untersteiner 1961 via constants.k_ice_default) ---

# Numerics floor for the skin update [K]: below the coldest observed polar
# surface (~185 K is the terrestrial record, Antarctic plateau) — a guard
# against a pathological flux spike, never active in normal operation.
_ICE_SKIN_FLOOR_K = 185.0


def prognostic_ice_skin_temperature(
    T_skin: jnp.ndarray,
    F_net_down_W_m2: jnp.ndarray,
    sic: jnp.ndarray,
    dt_s: float,
    h_ice_m: float,
    T_freeze_K: float = constants.T_freeze_ocean,
    k_ice_W_m_K: float = constants.k_ice_default,
    T_melt_surface_K: float = constants.T_freeze,
    melt_cap_width_K: float = 0.0,
) -> jnp.ndarray:
    """Advance a prescribed-ice AMIP skin temperature one integration step.

    Semtner (1976) zero-layer thermodynamics — conductive flux through a
    climatological ice slab of thickness ``h_ice_m`` toward the seawater
    freezing point ``T_freeze_K`` at the BASE (ice/ocean interface) — plus
    the slab's half thermal inertia (``C = rho_ice * c_pi * h/2``), which
    turns the diagnostic zero-layer balance into a stably integrable
    prognostic skin:

        C dT_s/dt = F_net_down(atm) + (k_i / h) * (T_base - T_s)

    Signs (surface conventions of the exported ``_sfc_diag`` fluxes):
    ``F_net_down_W_m2 = sw_net_sfc + lw_net_sfc - shflx - lhflx`` — sw/lw net
    positive INTO the surface, turbulent fluxes positive UPWARD out of it, so
    ``F_net_down`` is the net energy gain of the skin from the atmosphere.
    Winter polar night: F_net_down < 0, the skin cools below ``T_freeze_K``
    until conduction from the ocean balances the loss — the equilibrium
    ``T_s = T_base + F_net_down * h / k_i`` of the classic zero-layer model.

    Discretization: backward-Euler in the CONDUCTIVE term (unconditionally
    stable for any ``dt_s``; the atmospheric flux is the lagged explicit
    forcing, refreshed by the physics each step):

        T_new = (T_s + (dt/C)*(F_net + (k_i/h)*T_base)) / (1 + (dt/C)*k_i/h)

    The FULL coupled system (skin + atmosphere recomputing F_net at the new
    skin) is only CONDITIONALLY stable, and the threshold depends on WHEN the
    atmosphere sees the new skin.  If the caller refreshes the surface anchor
    at the SAME cadence as this advance — the MPAS driver re-blends T_sfc every
    model step — the per-step amplification is ``|1 - r*lambda|/(1 + r*g)`` for
    a surface-flux feedback ``lambda = -dF_net/dT_s > 0`` (LW + sensible),
    ``r = dt/C``, ``g = k_i/h``; at the short model step ``dt_s = DT`` this
    keeps ``r*lambda << 1``, stable and monotone for any realistic ``lambda``.
    If instead the anchor is HELD while the skin advances sub-cadence (e.g.
    per-step advance under a once-daily anchor), the effective map reverts to
    the coarse-cadence thresholds (order tens of W/m^2/K, tightening as h
    shrinks) and the melt cap only BOUNDS — does not cure — a cap/floor
    oscillation.  Advance and anchor-refresh cadence MUST match.

    Bounds: melt cap ``T_new <= T_melt_surface_K`` — the FRESH-ICE SURFACE
    melting point (``constants.T_freeze``, i.e. 0 C), NOT the basal seawater
    freezing point ``T_freeze_K`` (``constants.T_freeze_ocean``).  Distinct
    boundary conditions: conduction is toward the warmer seawater-freezing
    base, the surface melts at 0 C (mirrors ``ice/sea_ice.py`` ``T_base`` vs
    ``T_melt_surface``).  A melting skin sheds energy at 0 C — Semtner's melt
1656:def _load_era5_samples_spectral(cfg, yml, grid, sigma, *,
1761:def load_era5_samples(cfg, yml, grid, sigma, *,
        # fill; fail fast if the coupling is configured but no aerosol field
        # was threaded.  Done BEFORE the optional column-shard below so the
        # overridden field shards with the rest.  Only ``rrtmgp`` consumes a
        # droplet number (gray ignores ``n_cloud`` entirely), so the override
        # is gated on the scheme — gray runs the microphysics fill (second
        # indirect effect) but skips this radiation-only first-indirect path
        # instead of computing a droplet field the gray optics would discard.
        _nc_in_cloud = False
        if nc_from_aerosol and radiation_config.scheme == "rrtmgp":
            # Route through the SAME activation dispatch as the microphysics
            # N_c fill (proxy | arg) so the first (radiation r_eff) and second
            # (autoconversion) indirect effects see ONE droplet number per
            # step.  activation_config=None -> default proxy, byte-identical
            # to the previous direct specified_nc_field call.  The
            # missing-aerosol_od raise is proxy-only, mirroring the
            # microphysics guard (ARG needs no AOD).
            from legoesm.atmosphere.physics.microphysics.arg_activation import (  # noqa: E501
                ActivationConfig,
                activated_nc_field,
            )
            _act_cfg = (activation_config if activation_config is not None
                        else ActivationConfig())
            if _act_cfg.scheme == "proxy" and _aer_ext is None:
                raise ValueError(
                    "nc_from_aerosol=True but no 'aerosol_od' was passed to "
                    "the radiation physics_fn via forcing — enable external "
                    "aerosol forcing (--aerosol-forcing external) or disable "
                    "--aerosol-ccn."
                )
            _aer_num = (forcing.get("aerosol_number")
                        if forcing is not None else None)
            n_cloud_col = activated_nc_field(
                _act_cfg, (ncol, nlev),
                aerosol_od=(None if _aer_ext is None
                            else jnp.asarray(_aer_ext)),
                T=T_col, p=p_full_col,
 A tests/unit/test_volcanic_sw_optics.py
 M tests/unit/test_wb_amip_deck_params.py
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_prompt.md

exec
/usr/bin/bash -lc "sed -n '1798,1815p' packages/ml/legoesm/training/scale_build.py; sed -n '20127,20150p' packages/coupler/legoesm/driver/model_driver.py; sed -n '494,570p' packages/tools/legoesm/forcing/surface_utils.py; rg -n 'volcanic' packages/ml/legoesm/training/scale_build.py | head -15; sed -n '2480,2493p' packages/tools/legoesm/forcing/external.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 78ms:
    # 2-D only, but keeps load_surface_fluxes / load_land_frac so the
    # sample-start slice carries the prescribed planes.
    sst_cfg = era5_cfg._replace(load_cloud_condensate=False)
    # Flux store opened once, only when fluxes are prescribed ("" = the
    # state store); a land fraction alone comes from the state store.
    flux_ds = None
    if sst_cfg.load_surface_fluxes:
        flux_ds = open_era5_zarr(sst_cfg.flux_zarr) if sst_cfg.flux_zarr else ds
    _micro = wb_ic_microphysics(cfg.mode, yml)

    config = build_latlon_config(cfg, yml)
    driver = _driver_for_ctx(config)
    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)

    # Spectral forcing-dict name -> driver-path (pack_forcing) name for the
    # planes the two lanes name differently; the radiative planes and the
    # land fraction already share their names.
    _spec_to_driver = {
    def build_training_segment(self, n_steps: int, day: float = 0.0):
        """Build a DIFFERENTIABLE single-segment forward for parameter calibration.

        Returns ``(run_segment_raw, carry0, forcing)``:
          * ``run_segment_raw`` — the NON-JIT, non-donating ``.raw`` segment
            (buffer donation conflicts with reverse-mode AD), composable inside
            ``eqx.filter_value_and_grad`` / ``jax.grad``.
          * ``carry0`` — the initial ``SegmentCarry`` (multilayer land seeded in
            ``setup`` when ``use_multilayer_land``; rides ``carry0.land_ml``).
          * ``forcing`` — the ``SegmentForcing`` for ``day`` (prescribed SST/SIC,
            solar, ozone, aerosol).

        The caller runs ``final = run_segment_raw(carry0, n_steps, forcing)``.
        Trainable LAND params reach the tile by assigning a *traced*
        ``LandSurfaceParams`` to ``self.physics.land_ml_params`` BEFORE the call
        inside the loss closure — the segment reads that attribute at call time,
        so gradients flow back to the params (no hot-loop refactor needed; see
        ``scripts/run/train_coupled_land_era5.py``).  ``gradient_checkpoint`` is
        forced on to bound the reverse-mode memory of the unrolled scan.

        Reuses ``_prepare_run_context`` (the same step_unified / forcing / carry
        seeds as ``_run_compiled``) so the training forward is byte-faithful to a
        production segment.  Single-rank / single-device only (the carry has no
        SPMD/MPI partition spec)."""
def place_stratospheric_aod_profile_to_layers(
    aod_profile: jnp.ndarray,
    p_edges: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Conservatively remap a fixed-pressure-edge AOD profile onto model layers.

    Volcanic stratospheric aerosol is supplied as a per-layer absorption
    optical depth on the forcing file's OWN (altitude -> pressure) grid.  This
    bins it onto the model's pressure layers by fractional pressure overlap,
    so the aerosol lands at its true stratospheric pressure -- unlike
    :func:`distribute_column_aod_to_layers`, which spreads a single column AOD
    by full-column pressure mass and thus dumps ~90 % of a stratospheric layer
    into the troposphere.

    Method: the cumulative-OD curve is sampled at each model half level and the
    per-layer OD is the difference of the cumulative at the layer's two edges.
    The cumulative is interpolated in LOG-pressure, because the source OD is
    ``ext*dz`` -- uniform in geometric height (~ log-pressure) within a source
    layer, NOT uniform in pressure; a linear-pressure CDF would misallocate ~1 %
    of a 0.5 km source layer's OD across a cutting model interface (Codex review
    iter-1).  The remap is exact at source-layer edges and conserves the total
    column OD that falls within the model's pressure range (OD above the model
    top or below the surface edge is dropped -- it cannot be represented).

    The op set (searchsorted / gather / clip / min / max) is piecewise-linear:
    a.e.-differentiable in both ``aod_profile`` and ``p_half_col`` with
    well-defined one-sided gradients (kinks at source/model edges), and
    JIT/vmap-safe.

    Sign/units: optical depth is >= 0 and additive; pressure [Pa].

    Parameters
    ----------
    aod_profile : array, shape (ncol, nsrc)
        Per-source-layer absorption optical depth [-] (>= 0).
    p_edges : array, shape (nsrc+1,)
        Source-layer pressure edges [Pa], ASCENDING (index 0 = top / lowest
        pressure), aligned so ``aod_profile[:, j]`` occupies
        ``[p_edges[j], p_edges[j+1]]``.
    p_half_col : array, shape (ncol, nlev+1)
        Model half-level pressures [Pa] (index 0 = top).

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer absorption optical depth [-].
    """
    pe = jnp.asarray(p_edges)                                   # (nsrc+1,)
    aod = jnp.clip(jnp.asarray(aod_profile), 0.0, None)         # (ncol, nsrc)
    # Cumulative OD from the top: cum[:, k] = OD in source layers 0..k-1, so
    # cum aligns to the nsrc+1 source edges ``pe``.
    cum = jnp.concatenate(
        [jnp.zeros((aod.shape[0], 1), aod.dtype), jnp.cumsum(aod, axis=1)],
        axis=1,
    )                                                          # (ncol, nsrc+1)
    # Interpolate the CDF in log-pressure (source OD is uniform in geometric
    # height ~ log-pressure within a layer).
    l_edges = jnp.log(jnp.clip(pe, 1.0e-12, None))             # (nsrc+1,)

    ph = jnp.asarray(p_half_col)
    p_lo = jnp.minimum(ph[:, :-1], ph[:, 1:])                  # (ncol, nlev)
    p_hi = jnp.maximum(ph[:, :-1], ph[:, 1:])

    def _cdf_at(p):
        # Per-column log-pressure interp of the shared-edge cumulative at
        # pressures ``p``; clamps to [0, total] outside [pe[0], pe[-1]].
        lp = jnp.log(jnp.clip(p, 1.0e-12, None))
        idx = jnp.clip(jnp.searchsorted(l_edges, lp) - 1, 0, pe.shape[0] - 2)
        l0 = l_edges[idx]
        l1 = l_edges[idx + 1]
        frac = jnp.clip((lp - l0) / jnp.clip(l1 - l0, 1.0e-12, None), 0.0, 1.0)
        c0 = jnp.take_along_axis(cum, idx, axis=1)
        c1 = jnp.take_along_axis(cum, idx + 1, axis=1)
        return c0 + frac * (c1 - c0)

    return jnp.clip(_cdf_at(p_hi) - _cdf_at(p_lo), 0.0, None)
548:    "volcanic_aerosol_file", "volcanic_aerosol_lw",
1560:    lw = cl.get("volcanic_aerosol_lw", False)
1568:            f"classical.volcanic_aerosol_lw must be a YAML boolean, got {lw!r}")
1569:    volc = str(cl.get("volcanic_aerosol_file", "") or "")
1575:        volcanic_enabled=bool(volc), volcanic_path=volc,
1576:        volcanic_lw_enabled=(lw and bool(volc)),
        except ValueError:
            # File has no ext_earth / ext_sun band — treat as "no source".
            return None

    # Interpolate the (ntime, nlat, nlayer) profile in time -> (nlat, nlayer).
    if len(mid_days_v) > 12:
        file_day_v = _simday_to_file_day(day, config.start_year, first_date_v)
        prof_t = _interp_monthly_noncyclic(
            mid_days_v, prof_v, file_day_v, config.volcanic_path,
            config.hold_outside_file_years)
    else:
        mid_anchored_v, prof_anchored_v = _cyclic_phase_anchor(
            mid_days_v, prof_v, first_date_v,
        )

exec
/usr/bin/bash -lc "nl -ba tests/unit/test_volcanic_sw_optics.py | sed -n '140,172p'; nl -ba packages/tools/legoesm/forcing/external.py | sed -n '1650,1668p'; sed -n '736,750p' packages/coupler/legoesm/driver/physics_pipeline.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
   140	    ncol, nlev = kw["T"].shape
   141	    tau = jnp.zeros((ncol, nlev, 14)).at[:, layers, :].set(jnp.asarray(tau_b))
   142	    return jnp.stack([tau, tau * jnp.asarray(ssa_b),
   143	                      tau * jnp.asarray(ssa_b) * jnp.asarray(g_b)], axis=2)
   144	
   145	
   146	def _solver():
   147	    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
   148	    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
   149	    cfg = RRTMGPConfig()
   150	    return RRTMGP.from_legoesm_config(cfg), cfg
   151	
   152	
   153	def test_grey_moments_reproduce_grey_path():
   154	    """Volcanic tau with the grey ssa/g == the same tau in aerosol_od."""
   155	    solver, cfg = _solver()
   156	    kw = _columns()
   157	    volc = _volc_moments(kw, 0.05, cfg.aerosol_ssa, cfg.aerosol_g)
   158	    a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
   159	    kw_grey = {**kw, "aerosol_optical_depth":
   160	               kw["aerosol_optical_depth"] + volc[:, :, 0, 0]}
   161	    b = solver.solve_columns(**kw_grey)
   162	    for f in ("sw_flux_up", "sw_flux_down", "sw_heating_rate"):
   163	        np.testing.assert_allclose(getattr(a, f), getattr(b, f), rtol=1e-10,
   164	                                   atol=1e-10)
   165	
   166	
   167	def test_file_optics_change_sw_and_close_budget():
   168	    """Non-absorbing volcanic sulfate (ssa 1) absorbs less SW and reflects
   169	    more than the same optical depth at the grey ssa 0.93; both close the SW
   170	    column budget (TOA net - surface net = sum of layer heating * cp dp / g)."""
   171	    from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rc
   172	    solver, cfg = _solver()
  1650	        x0 = lat_src[idx]
  1651	        x1 = lat_src[idx + 1]
  1652	        # Avoid division by zero on duplicated source points.
  1653	        dx = np.where(x1 > x0, x1 - x0, 1.0)
  1654	        w = np.where(x1 > x0, (lat_deg - x0) / dx, 0.0)
  1655	        # Clamp at the endpoints (matches np.interp's flat extrapolation).
  1656	        w = np.clip(w, 0.0, 1.0)
  1657	        # Reshape ``field`` to (nsrc, n_cols) and gather along leading.
  1658	        trailing_shape = field.shape[1:]
  1659	        n_cols = int(np.prod(trailing_shape))
  1660	        field_flat = np.asarray(field).reshape(nsrc, n_cols)
  1661	        f0 = field_flat[idx]                    # (n_lat, n_cols)
  1662	        f1 = field_flat[idx + 1]                # (n_lat, n_cols)
  1663	        result_flat = (1.0 - w[:, None]) * f0 + w[:, None] * f1
  1664	    return jnp.array(result_flat).reshape((*lat_grid.shape, *field.shape[1:]))
  1665	
  1666	
  1667	def _interp_vertical(field_plev: jnp.ndarray, plev_src: np.ndarray,
  1668	                      p_target: jnp.ndarray) -> jnp.ndarray:
        from legoesm.atmosphere.physics.turbulence.surface_layer import (
            surface_fluxes_at_lowest_level,
        )
        # SAME lowest-full-level density the turbulence path feeds its surface
        # layer (rho_col_phys[:, -1] = p_full/(R_d*T)), NOT the legacy
        # _land_surface_bulk value — the law must see identical inputs.
        rho_low = self.sigma_coord.pressure_at_full(p_s)[..., -1] / (constants.R_d * T_air)
        if self.surface_tiled:
            cfg = self._land_tile_surface_cfg()
        else:
            cfg = self.turbulence_config.surface
            if T_sfc_ocean is None:
                raise ValueError(
                    "unified non-tiled land fluxes need T_sfc_ocean (the "
                    "pre-land ocean/ice surface-temperature blend): the "

 succeeded in 157874ms:
/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/external.py
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/derecho/scratch/pg2328/wt_volc
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 30 items

tests/unit/test_volcanic_sw_optics.py .................                  [ 56%]
tests/unit/test_aerosol_band_ccn.py .............                        [100%]

=============================== warnings summary ===============================
tests/unit/test_volcanic_sw_optics.py: 19 warnings
tests/unit/test_aerosol_band_ccn.py: 6 warnings
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:54: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable(k, "f8", dims)[:] = v

tests/unit/test_volcanic_sw_optics.py::test_legacy_column_file_is_not_an_optics_source
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:100: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01

tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
tests/unit/test_aerosol_band_ccn.py::test_volcanic_sw_above_tropopause_and_out_of_ccn
tests/unit/test_aerosol_band_ccn.py::test_ccn_file_flows_from_cli_to_ccn
tests/unit/test_aerosol_band_ccn.py::test_ccn_file_flows_from_cli_to_ccn
tests/unit/test_aerosol_band_ccn.py::test_driver_volcanic_cut_at_state_tropopause
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_aerosol_band_ccn.py:55: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", dims)[:] = np.broadcast_to(band, shape)

tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_aerosol_band_ccn.py:91: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", ("time", "lnwl", "lat"))[:] = np.broadcast_to(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 30 passed, 33 warnings in 153.09s (0:02:33) ==================

codex
All 30 selected tests have reached completion; I’m waiting for pytest’s final summary. The direct CPU check confirmed that gray and simple-LW accept a nonzero volcanic field without raising. At res6’s 40,962 columns, host interpolation materializes a roughly 0.90 GiB float64 result plus several large temporaries; peak memory remains unmeasured.
exec
/usr/bin/bash -lc "rg -n '"'^def _simday_to_file_day'"' packages/tools/legoesm/forcing/external.py; sed -n '1560,1582p' packages/ml/legoesm/training/scale_build.py; sed -n '895,925p' packages/tools/legoesm/forcing/external.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
1521:def _simday_to_file_day(sim_day: float, start_year: int,
    lw = cl.get("volcanic_aerosol_lw", False)
    if (not isinstance(years, list) or len(years) != 2
            or not all(isinstance(y, int) for y in years)
            or years[0] > years[1]):
        raise ValueError("classical.aerosol_climatology_years must be "
                         f"[first, last] integer years, got {years!r}")
    if not isinstance(lw, bool):
        raise ValueError(
            f"classical.volcanic_aerosol_lw must be a YAML boolean, got {lw!r}")
    volc = str(cl.get("volcanic_aerosol_file", "") or "")
    acfg = AerosolConfig(
        enabled=True, source="climatology", path=str(cl["aerosol_file"]),
        ccn_path=str(cl.get("aerosol_ccn_file", "") or ""),
        use_reference_if_missing=True,
        reference_aod_550=float(cl.get("aerosol_reference_aod", 0.03)),
        volcanic_enabled=bool(volc), volcanic_path=volc,
        volcanic_lw_enabled=(lw and bool(volc)),
        hold_outside_file_years=False)
    return acfg, tuple(range(years[0], years[1] + 1)), (lw and bool(volc))


def wb_aerosol_forcing(spec, grid, sigma, carry, time_ns):
    """``aerosol_od`` / ``aerosol_ccn_aod`` / ``aerosol_lw_od`` for one sample:
        raise ValueError(f"No altitude coordinate in volcanic file {path!r}")
    alt_km = np.asarray(ds[alt_name].values, dtype=np.float64)

    if "lat" in ds:
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lat_name = "lat"
    elif "latitude" in ds:
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lat_name = "latitude"
    else:
        ds.close()
        raise ValueError(f"No 'lat'/'latitude' in volcanic file {path!r}")

    if "time" in dims:
        time_name = "time"
        mid_days, first_date = _read_time_axis(ds)
    elif "month" in dims:
        time_name = "month"
        nm = ext.shape[dims.index("month")]
        mid_days = np.array([15.5 + 30.4375 * m for m in range(nm)])
        first_date = None
    else:
        ds.close()
        raise ValueError(f"No time/month dimension in volcanic file {path!r}")

    # B3: Planck-emission-weighted collapse of the terrestrial-band axis.
    band_name = next(
        (d for d in dims
         if d not in (time_name, lat_name, alt_name)),
        None,
    )

codex
1. **MAJOR — Training silently loses volcanic SW.** [model_driver.py:20284](/glade/derecho/scratch/pg2328/wt_volc/packages/coupler/legoesm/driver/model_driver.py:20284) and [scale_build.py:1856](/glade/derecho/scratch/pg2328/wt_volc/packages/ml/legoesm/training/scale_build.py:1856): `build_training_segment` and the lat-lon ERA5 sample loader pack `aerosol_od` without `aerosol_sw_volc`. Preprocessing has already removed volcanic extinction from `aerosol_od`, so these training forwards receive **no volcanic SW forcing**, unlike production.

2. **MAJOR — Non-RRTMGP refusal is incomplete.** [integration.py:1069](/glade/derecho/scratch/pg2328/wt_volc/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1069): simple-LW and gray return before consuming or rejecting `aerosol_sw_volc`. This affects integration-based hydrostatic/MPAS and spectral callers. Direct CPU calls confirmed both accept a nonzero volcanic field without raising. The pipeline-builder tests miss these dispatches.

3. **MINOR — Solver tests cannot detect incorrect band selection.** [test_volcanic_sw_optics.py:157](/glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:157): solver fixtures give every band identical moments. Reversing the volcanic band lookup therefore preserves equivalence, closure, and gradient results. The loader test checks sorting separately, leaving the connection to `g_point_to_bnd` unprotected. Add a distinct-band test with a deliberately wrong mapping.

4. **MINOR — Probe samples an approximate date while printing a calendar month.** [probe_volcanic_sw_optics.py:42](/glade/derecho/scratch/pg2328/wt_volc/scripts/validate/probe_volcanic_sw_optics.py:42): `365*(year-1979) + 30.4*(month-1)` omits leap days and actual month lengths. Later-year transient volcanic measurements sample a different date from the advertised mid-month.

**VERDICT: HOLD** for findings 1–2. Both selected test modules passed: **30 passed**. Band sorting, zero-based g-point mapping, moment mixing, CCN exclusion, and production RRTMGP forwarding look correct on inspection.

**UNVERIFIED:** distributed execution, compute-fp32 runtime, and peak memory. At res6’s 40,962 columns, host interpolation alone produces approximately **0.90 GiB** of float64 output plus large temporaries; the float32 cache does not prevent that promotion.
tokens used
113,308
1. **MAJOR — Training silently loses volcanic SW.** [model_driver.py:20284](/glade/derecho/scratch/pg2328/wt_volc/packages/coupler/legoesm/driver/model_driver.py:20284) and [scale_build.py:1856](/glade/derecho/scratch/pg2328/wt_volc/packages/ml/legoesm/training/scale_build.py:1856): `build_training_segment` and the lat-lon ERA5 sample loader pack `aerosol_od` without `aerosol_sw_volc`. Preprocessing has already removed volcanic extinction from `aerosol_od`, so these training forwards receive **no volcanic SW forcing**, unlike production.

2. **MAJOR — Non-RRTMGP refusal is incomplete.** [integration.py:1069](/glade/derecho/scratch/pg2328/wt_volc/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1069): simple-LW and gray return before consuming or rejecting `aerosol_sw_volc`. This affects integration-based hydrostatic/MPAS and spectral callers. Direct CPU calls confirmed both accept a nonzero volcanic field without raising. The pipeline-builder tests miss these dispatches.

3. **MINOR — Solver tests cannot detect incorrect band selection.** [test_volcanic_sw_optics.py:157](/glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:157): solver fixtures give every band identical moments. Reversing the volcanic band lookup therefore preserves equivalence, closure, and gradient results. The loader test checks sorting separately, leaving the connection to `g_point_to_bnd` unprotected. Add a distinct-band test with a deliberately wrong mapping.

4. **MINOR — Probe samples an approximate date while printing a calendar month.** [probe_volcanic_sw_optics.py:42](/glade/derecho/scratch/pg2328/wt_volc/scripts/validate/probe_volcanic_sw_optics.py:42): `365*(year-1979) + 30.4*(month-1)` omits leap days and actual month lengths. Later-year transient volcanic measurements sample a different date from the advertised mid-month.

**VERDICT: HOLD** for findings 1–2. Both selected test modules passed: **30 passed**. Band sorting, zero-based g-point mapping, moment mixing, CCN exclusion, and production RRTMGP forwarding look correct on inspection.

**UNVERIFIED:** distributed execution, compute-fp32 runtime, and peak memory. At res6’s 40,962 columns, host interpolation alone produces approximately **0.90 GiB** of float64 output plus large temporaries; the float32 cache does not prevent that promotion.
ics — conductive flux through a
    climatological ice slab of thickness ``h_ice_m`` toward the seawater
    freezing point ``T_freeze_K`` at the BASE (ice/ocean interface) — plus
    the slab's half thermal inertia (``C = rho_ice * c_pi * h/2``), which
    turns the diagnostic zero-layer balance into a stably integrable
    prognostic skin:

        C dT_s/dt = F_net_down(atm) + (k_i / h) * (T_base - T_s)

    Signs (surface conventions of the exported ``_sfc_diag`` fluxes):
    ``F_net_down_W_m2 = sw_net_sfc + lw_net_sfc - shflx - lhflx`` — sw/lw net
    positive INTO the surface, turbulent fluxes positive UPWARD out of it, so
    ``F_net_down`` is the net energy gain of the skin from the atmosphere.
    Winter polar night: F_net_down < 0, the skin cools below ``T_freeze_K``
    until conduction from the ocean balances the loss — the equilibrium
    ``T_s = T_base + F_net_down * h / k_i`` of the classic zero-layer model.

    Discretization: backward-Euler in the CONDUCTIVE term (unconditionally
    stable for any ``dt_s``; the atmospheric flux is the lagged explicit
    forcing, refreshed by the physics each step):

        T_new = (T_s + (dt/C)*(F_net + (k_i/h)*T_base)) / (1 + (dt/C)*k_i/h)

    The FULL coupled system (skin + atmosphere recomputing F_net at the new
    skin) is only CONDITIONALLY stable, and the threshold depends on WHEN the
    atmosphere sees the new skin.  If the caller refreshes the surface anchor
    at the SAME cadence as this advance — the MPAS driver re-blends T_sfc every
    model step — the per-step amplification is ``|1 - r*lambda|/(1 + r*g)`` for
    a surface-flux feedback ``lambda = -dF_net/dT_s > 0`` (LW + sensible),
    ``r = dt/C``, ``g = k_i/h``; at the short model step ``dt_s = DT`` this
    keeps ``r*lambda << 1``, stable and monotone for any realistic ``lambda``.
    If instead the anchor is HELD while the skin advances sub-cadence (e.g.
    per-step advance under a once-daily anchor), the effective map reverts to
    the coarse-cadence thresholds (order tens of W/m^2/K, tightening as h
    shrinks) and the melt cap only BOUNDS — does not cure — a cap/floor
    oscillation.  Advance and anchor-refresh cadence MUST match.

    Bounds: melt cap ``T_new <= T_melt_surface_K`` — the FRESH-ICE SURFACE
    melting point (``constants.T_freeze``, i.e. 0 C), NOT the basal seawater
    freezing point ``T_freeze_K`` (``constants.T_freeze_ocean``).  Distinct
    boundary conditions: conduction is toward the warmer seawater-freezing
    base, the surface melts at 0 C (mirrors ``ice/sea_ice.py`` ``T_base`` vs
    ``T_melt_surface``).  A melting skin sheds energy at 0 C — Semtner's melt
    branch, whose meltwater bookkeeping a prescribed-ice run does not carry.
    ``_ICE_SKIN_FLOOR_K`` guards a pathological cold spike.  Open water
    (``sic <= 0``) snaps to ``T_freeze_K`` (seawater freezing) so a cell
    freezing later starts from the freezing point, not a stale skin.

    Limitations. (1) Single-tile: ``F_net_down`` is the BLENDED-cell surface
    flux (computed at ``sic*T_skin + (1-sic)*sst``), used here as the ice-tile
    forcing.  Exact only at ``sic=1``; at marginal ice the true ice-tile
    LW/turbulent flux differs — a documented closure error, most consequential
    during seasonal advance/retreat, second-order because the skin enters the
    anchor weighted by ``sic``.  (2) ``h_ice_m`` is a single global
    climatological thickness (no Arctic~2 m / Antarctic~1 m asymmetry).
    (3) Differentiability: the interior is smooth (``dT_new/dF = r/(1+r*g)``,
    exact), but ``clip`` has zero gradient AT the two caps and ``where(sic>0)``
    zeroes the flux gradient over open water — the physically correct
    dead-gradient of a saturated cap / inactive mask, not a defect.

    Parameters
    ----------
    T_skin : (ncol,) current skin temperature [K].
    F_net_down_W_m2 : (ncol,) net downward atmospheric energy flux [W/m^2].
    sic : (ncol,) sea-ice concentration [0-1].
    dt_s : integration step [s] (the model step DT in the MPAS AMIP loop).
    h_ice_m : climatological slab thickness [m] (tunable; ~2 m Arctic mean).
    T_freeze_K : basal seawater freezing point [K] (conduction base +
        open-water snap).
    k_ice_W_m_K : ice thermal conductivity [W/m/K].
    T_melt_surface_K : fresh-ice surface melting point [K] (upper cap).
    melt_cap_width_K : width [K] of the smooth melt cap.  ``0`` = the hard
        ``clip`` cap; ``> 0`` = ``smooth_min(T_new, T_melt_surface_K,
        sharpness=1/width)`` (LogSumExp, never above the melting point), so
        the skin keeps a nonzero gradient at the cap.  KNOWN DIFFERENCE vs the
        hard cap: a melting skin sits slightly below 0 C, by about
        ``width * ln(1 / (1 - exp(-dT/width)))`` for a per-step warming dT
        (measured per case in tests/unit/test_ice_skin_smooth_cap.py).

    Returns
    -------
            # shared with the periodic path.
            _ckpt = ((getattr(self, "_checkpoint_callback", None)
                      if self._mpi_rank is None else None)
                     or self.save_checkpoint)
            # A COMPLETED run has NO in-progress month to resume, and
            # ``diagnostics.save`` above already finalized every bucket to
            # NetCDF WITHOUT popping — so the terminal checkpoint must carry no
            # CMOR sidecar, or a run-EXTENDING restart would re-append every
            # already-written month (duplicate time coords).  The flag rides
            # through whichever ``_ckpt`` callback writes the sidecar.
            self._suppress_cmor_sidecar = True
            try:
                _ckpt(n_steps_total, START_DAY + N_DAYS)
            finally:
                self._suppress_cmor_sidecar = False

        # Issue #275 fix A lifecycle: restore the halo backend captured
        # at activation time so subsequent drivers / tests in the same
        # process see a clean ``"local"`` (or pre-existing MPI) state.
        self._restore_halo_backend()

        return run_status

    # ==================================================================
    # Differentiable training entry (parameter calibration)
    # ==================================================================

    def build_training_segment(self, n_steps: int, day: float = 0.0):
        """Build a DIFFERENTIABLE single-segment forward for parameter calibration.

        Returns ``(run_segment_raw, carry0, forcing)``:
          * ``run_segment_raw`` — the NON-JIT, non-donating ``.raw`` segment
            (buffer donation conflicts with reverse-mode AD), composable inside
            ``eqx.filter_value_and_grad`` / ``jax.grad``.
          * ``carry0`` — the initial ``SegmentCarry`` (multilayer land seeded in
            ``setup`` when ``use_multilayer_land``; rides ``carry0.land_ml``).
          * ``forcing`` — the ``SegmentForcing`` for ``day`` (prescribed SST/SIC,
            solar, ozone, aerosol).

        The caller runs ``final = run_segment_raw(carry0, n_steps, forcing)``.
        Trainable LAND params reach the tile by assigning a *traced*
        ``LandSurfaceParams`` to ``self.physics.land_ml_params`` BEFORE the call
        inside the loss closure — the segment reads that attribute at call time,
        so gradients flow back to the params (no hot-loop refactor needed; see
        ``scripts/run/train_coupled_land_era5.py``).  ``gradient_checkpoint`` is
        forced on to bound the reverse-mode memory of the unrolled scan.

        Reuses ``_prepare_run_context`` (the same step_unified / forcing / carry
        seeds as ``_run_compiled``) so the training forward is byte-faithful to a
        production segment.  Single-rank / single-device only (the carry has no
        SPMD/MPI partition spec)."""
        from legoesm.driver.compiled_segments import (
            pack_carry, tiedtke_lag_seeds, build_segment_fn, pack_forcing,
        )
        if (self._device_config is not None
                and getattr(self._device_config, "is_distributed", False)):
            raise NotImplementedError(
                "build_training_segment is single-rank / single-device only "
                "(the SegmentCarry has no partition spec for SPMD/MPI sharding)."
            )
        # #1028: the persistent-D conversion happens in ``run()``, which a
    # nowhere to put one, so seeding it would make the forecast carry
    # structurally different from the initial condition it is scored against.
    # Carrying turbulence energy across steps on this path is separate work.
    _micro = wb_ic_microphysics(cfg.mode, yml)
    _aerosol = wb_aerosol_spec(yml) if cfg.mode == "physics" else None


    def build(year, i_ic, i_tg, ds, cloud_ds, flux_ds, at):
        s_ic, s_tg, sst_src = _sample_inputs(
            era5_cfg, sst_cfg, i_ic, i_tg, ds, cloud_ds, flux_ds, at)
        # ERA5-derived terrain (no elevation file on the training lane)
        tp = era5_terrain_product(s_ic, grid)
        ic = era5_to_spectral_carry(s_ic, grid, sigma, target_phis=tp, microphysics=_micro)
        target = era5_to_spectral_carry(s_tg, grid, sigma, target_phis=tp, microphysics=_micro)
        # times stays the REMOTE calendar: i_ic is always a remote index
        forcing = build_spectral_forcing(sst_src, grid, times[i_ic], year,
                                         aerosol=_aerosol, sigma=sigma,
                                         ic_carry=ic)
        return (ic, target, forcing)

    mine = list(_sharded_indices(
        cfg, yml, times, snaps_per_day, stride, rank, nproc))
    src = (ds, cloud_ds, flux_ds, int)
    local = _wb_local_stores(cfg, yml, era5_cfg, sst_cfg, times,
                             snaps_per_day, stride, rank, nproc)
    if local is not None and mine:
        _, i0, t0 = mine[0]
        _assert_same_sample(
            _sample_inputs(era5_cfg, sst_cfg, i0, t0, *src),
            _sample_inputs(era5_cfg, sst_cfg, i0, t0, *local))
        src = local
    samples = []
    for year, i_ic, i_tg in mine:
        sample = build(year, i_ic, i_tg, *src)
        if not samples:
            validate_carry_holds_scheme(
                sample[0], _micro, context=f"WB {cfg.mode} arm initial condition")
        samples.append(_sample_to_host(sample) if host_resident else sample)
    return samples


def load_era5_samples(cfg, yml, grid, sigma, *,
                      rank=0, nproc=1, host_resident=False):
    """(ic, target, forcing) samples over the train windows, on the lat-lon grid.

    #1286: ``nproc > 1`` builds ONLY ``rank``'s contiguous shard (fix B);
"""Shared test fixtures for legoESM."""

import functools
import os
import resource
import sys
from pathlib import Path

#: Cap the CPU affinity mask BEFORE JAX is imported (below).  XLA sizes its
#: Eigen thread pool from the affinity mask, so on a many-core login node a
#: single JAX process opens far more threads than ``ulimit -u`` allows once a
#: few run at once, and the pool constructor aborts the interpreter:
#:
#:   F env.cc:93] Check failed: ret == 0 (11 vs. 0) Thread tf_XLAPjRtCpuClient
#:   creation via pthread_create() failed.
#:     @ Eigen::ThreadPoolTempl<>::ThreadPoolTempl()
#:
#: (11 = EAGAIN.)  It surfaces as "Fatal Python error: Aborted" mid-run, which
#: reads like a test failure but is the process running out of thread slots.
#:
#: Measured on a 256-core Levante login node (``ulimit -u`` = 2048), threads
#: opened by one client for a trivial jitted matmul:
#:
#:     CPUs    4     8    16    32    64   256
#:     threads 29    45    77   141   237   621
#:
#: Uncapped that is 621 threads each, so FOUR concurrent clients (2484) are the
#: first count over the 2048 ceiling; five were observed to fail 5/5.  At 32
#: CPUs a client takes ~141, so ~14 fit.  Capping costs nothing measurable --
#: these tests are XLA-compile-bound, not intra-op-parallelism-bound
#: (tests/test_corner_div_damp_nh.py: 166.9 s on 256 CPUs, 166.3 s on 8,
#: 152.7 s on 32).
#:
#: Set ``LEGOESM_TEST_CPU_CAP`` to a positive integer to force a cap, or to 0
#: (or any value <= 0) to disable and inherit the machine's full affinity mask.
_DEFAULT_TEST_CPU_CAP = 32

#: Threads one XLA CPU client opens ~= _THREADS_PER_CPU * ncpu + _THREADS_BASE,
#: fitted to the table above (slope (621-29)/(256-4) = 2.35).  Used to size the
#: cap against the ACTUAL thread ceiling rather than trusting a fixed 32, which
#: is only evidenced safe for ulimit -u = 2048.
_THREADS_PER_CPU = 2.4
_THREADS_BASE = 20.0
#: Fraction of the thread ceiling the test session may claim, leaving room for
#: the user's other processes (a login node already carries ~50 threads).
_RLIMIT_SHARE = 0.5

#: Env vars every common launcher sets in a rank's environment.  Under MPI the
#: launcher (or the batch scheduler) owns placement: each rank already gets its
#: own binding, and clamping every rank to the SAME low-numbered CPUs here would
#: pile all of them onto one core set and serialise the run.  So the cap is
#: skipped for MPI ranks -- they are separate processes with a launcher-assigned
#: mask, which is exactly the placement this function would otherwise destroy.
#: MPI does not standardise a rank env var, so detection is BEST-EFFORT: this
#: covers the launchers in use here.  For an unlisted launcher, disable the cap
#: explicitly with ``LEGOESM_TEST_CPU_CAP=0``.
_MPI_RANK_ENV = (
    "OMPI_COMM_WORLD_RANK",     # Open MPI
    "PMI_RANK",                 # MPICH / Intel MPI
    "PMIX_RANK",                # PMIx
    "SLURM_PROCID",             # srun-launched
    "MV2_COMM_WORLD_RANK",      # MVAPICH2
    "MPI_RANKID",               # IBM Platform MPI
    "LAMRANK",                  # legacy LAM/MPI
)


def _is_mpi_rank() -> bool:
    """True when this interpreter was launched as a rank of an MPI job."""
    return any(v in os.environ for v in _MPI_RANK_ENV)


def _requested_cap() -> int | None:
    """The user's explicit ``LEGOESM_TEST_CPU_CAP``, or None when unset/garbage.

    A value <= 0 disables capping entirely and is returned as 0.
    """
    raw = os.environ.get("LEGOESM_TEST_CPU_CAP")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(0, value)


def _xdist_worker_count() -> int:
    """How many pytest-xdist workers share this machine (1 when not under xdist)."""
    try:
diff --git a/tests/unit/test_cross_lane_forcing.py b/tests/unit/test_cross_lane_forcing.py
index d436c2a65b..5fd911e129 100644
--- a/tests/unit/test_cross_lane_forcing.py
+++ b/tests/unit/test_cross_lane_forcing.py
@@ -75,7 +75,7 @@ def _patch_readers(monkeypatch):
         assert np.shape(lon_grid) == np.shape(lat_grid)
         return _aod(np.asarray(lat_grid), day, ccn)
     monkeypatch.setattr(ext, "get_aerosol_at_time", _aer)
-    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_at_time", lambda c, day, lat_grid=None: None)
+    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_optics_at_time", lambda c, day, lat_grid=None: None)
     monkeypatch.setattr(ext, "get_aerosol_lw_at_time", lambda c, day, lat_grid=None: (
         np.full((np.size(lat_grid), 2), 0.01), np.array([2000.0, 3000.0, 4000.0])))
     monkeypatch.setattr(ext, "get_ghg_at_time", lambda c, day: dict(GHG))
diff --git a/tests/unit/test_wb_amip_deck_params.py b/tests/unit/test_wb_amip_deck_params.py
index 5d4822e2db..be92dbd847 100644
--- a/tests/unit/test_wb_amip_deck_params.py
+++ b/tests/unit/test_wb_amip_deck_params.py
@@ -409,7 +409,7 @@ def test_aerosol_climatology_averages_the_years_on_the_noleap_clock(
         return np.full(np.shape(lat_grid), 1e-3 * (c.start_year - 2000)
                        + (0.5 if band == "vis" else 0.0))
     monkeypatch.setattr(ext, "get_aerosol_at_time", _aod)
-    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_at_time",
+    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_optics_at_time",
                         lambda c, day, lat_grid=None: None)
     monkeypatch.setattr(ext, "get_aerosol_lw_at_time",
                         lambda c, day, lat_grid=None: None)

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=packages/core:packages/atmosphere:packages/coupler:packages/tools:packages/ml:packages/land:packages/ocean:packages/ice /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c 'import legoesm.forcing.external as e; import netCDF4; print(e.__file__); d=netCDF4.Dataset(\"/glade/work/pg2328/amip_inputs/volcanic_cmip6_from_cesm_1978-2014.nc\"); print(d.variables[\"ext_sun\"].shape); print(d.variables[\"ext_sun\"].dimensions)'" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 1705ms:
/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/external.py
(14, 36, 70, 444)
('solar_bands', 'latitude', 'altitude', 'time')

exec
/usr/bin/bash -lc "sed -n '494,570p' packages/tools/legoesm/forcing/surface_utils.py; sed -n '19580,19620p' packages/coupler/legoesm/driver/model_driver.py; sed -n '290,300p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_base.py; rg -n 'monkeypatch|sys.path|legoesm.__path__' tests/conftest.py; rg -n 'aerosol_od' packages --glob '*.py' | tail -65" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 6ms:
def place_stratospheric_aod_profile_to_layers(
    aod_profile: jnp.ndarray,
    p_edges: jnp.ndarray,
    p_half_col: jnp.ndarray,
) -> jnp.ndarray:
    """Conservatively remap a fixed-pressure-edge AOD profile onto model layers.

    Volcanic stratospheric aerosol is supplied as a per-layer absorption
    optical depth on the forcing file's OWN (altitude -> pressure) grid.  This
    bins it onto the model's pressure layers by fractional pressure overlap,
    so the aerosol lands at its true stratospheric pressure -- unlike
    :func:`distribute_column_aod_to_layers`, which spreads a single column AOD
    by full-column pressure mass and thus dumps ~90 % of a stratospheric layer
    into the troposphere.

    Method: the cumulative-OD curve is sampled at each model half level and the
    per-layer OD is the difference of the cumulative at the layer's two edges.
    The cumulative is interpolated in LOG-pressure, because the source OD is
    ``ext*dz`` -- uniform in geometric height (~ log-pressure) within a source
    layer, NOT uniform in pressure; a linear-pressure CDF would misallocate ~1 %
    of a 0.5 km source layer's OD across a cutting model interface (Codex review
    iter-1).  The remap is exact at source-layer edges and conserves the total
    column OD that falls within the model's pressure range (OD above the model
    top or below the surface edge is dropped -- it cannot be represented).

    The op set (searchsorted / gather / clip / min / max) is piecewise-linear:
    a.e.-differentiable in both ``aod_profile`` and ``p_half_col`` with
    well-defined one-sided gradients (kinks at source/model edges), and
    JIT/vmap-safe.

    Sign/units: optical depth is >= 0 and additive; pressure [Pa].

    Parameters
    ----------
    aod_profile : array, shape (ncol, nsrc)
        Per-source-layer absorption optical depth [-] (>= 0).
    p_edges : array, shape (nsrc+1,)
        Source-layer pressure edges [Pa], ASCENDING (index 0 = top / lowest
        pressure), aligned so ``aod_profile[:, j]`` occupies
        ``[p_edges[j], p_edges[j+1]]``.
    p_half_col : array, shape (ncol, nlev+1)
        Model half-level pressures [Pa] (index 0 = top).

    Returns
    -------
    aod_layers : array, shape (ncol, nlev)
        Layer absorption optical depth [-].
    """
    pe = jnp.asarray(p_edges)                                   # (nsrc+1,)
    aod = jnp.clip(jnp.asarray(aod_profile), 0.0, None)         # (ncol, nsrc)
    # Cumulative OD from the top: cum[:, k] = OD in source layers 0..k-1, so
    # cum aligns to the nsrc+1 source edges ``pe``.
    cum = jnp.concatenate(
        [jnp.zeros((aod.shape[0], 1), aod.dtype), jnp.cumsum(aod, axis=1)],
        axis=1,
    )                                                          # (ncol, nsrc+1)
    # Interpolate the CDF in log-pressure (source OD is uniform in geometric
    # height ~ log-pressure within a layer).
    l_edges = jnp.log(jnp.clip(pe, 1.0e-12, None))             # (nsrc+1,)

    ph = jnp.asarray(p_half_col)
    p_lo = jnp.minimum(ph[:, :-1], ph[:, 1:])                  # (ncol, nlev)
    p_hi = jnp.maximum(ph[:, :-1], ph[:, 1:])

    def _cdf_at(p):
        # Per-column log-pressure interp of the shared-edge cumulative at
        # pressures ``p``; clamps to [0, total] outside [pe[0], pe[-1]].
        lp = jnp.log(jnp.clip(p, 1.0e-12, None))
        idx = jnp.clip(jnp.searchsorted(l_edges, lp) - 1, 0, pe.shape[0] - 2)
        l0 = l_edges[idx]
        l1 = l_edges[idx + 1]
        frac = jnp.clip((lp - l0) / jnp.clip(l1 - l0, 1.0e-12, None), 0.0, 1.0)
        c0 = jnp.take_along_axis(cum, idx, axis=1)
        c1 = jnp.take_along_axis(cum, idx + 1, axis=1)
        return c0 + frac * (c1 - c0)

    return jnp.clip(_cdf_at(p_hi) - _cdf_at(p_lo), 0.0, None)
                    _u_fold = self.state.u_f.data
                    _v_fold = to_vface_lower(self.state.v_f.data)
                else:
                    _u_fold, _v_fold = self.state.u.data, self.state.v.data
                _fold = dict(
                    u=jax.device_put(_u_fold, carry.u.sharding),
                    v=jax.device_put(_v_fold, carry.v.sharding),
                    T=jax.device_put(self.state.T.data, carry.T.sharding),
                    p_s=jax.device_put(self.state.p_s.data, carry.p_s.sharding),
                    q_v=jax.device_put(self.q_v, carry.q_v.sharding),
                    q_c=jax.device_put(self.q_c, carry.q_c.sharding),
                    q_r=jax.device_put(self.q_r, carry.q_r.sharding))
                # Double-moment tracers: a DA/coupling callback may nudge
                # self.tracers[q_i…N_i]; fold those back too (same repack-from-self
                # _run_compiled does), else a double-moment run diverges after a
                # tracer-mutating callback (codex #789-F4).  Only carry-threaded
                # (non-None) fields are re-shardable.
                if isinstance(self.tracers, dict):
                    for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
                        _cv, _sv = getattr(carry, _nm), self.tracers.get(_nm)
                        if _cv is not None and _sv is not None:
                            _fold[_nm] = jax.device_put(_sv, _cv.sharding)
                carry = carry._replace(**_fold)

        if _io_rank:
            # Rank-0 completion summary. self.state is the last segment's
            # REPLICATED gathered copy, so the max reductions are valid on any
            # rank; the selfspawn route-B gate greps this line + the finite
            # magnitudes to confirm the federation integrated without a hang.
            _maxT = float(jnp.abs(self.state.T.data).max())
            _maxU = float(jnp.abs(self.cell_centre_state().u.data).max())
            logger.info(
                "operator-split SPMD run: %s, %d steps, %d device(s) / %d "
                "process(es), max|T|=%.3f max|u|=%.3f (%.1fs)",
                status, n_steps_total - start_step, n_dev,
                jax.process_count(), _maxT, _maxU, time.time() - t0)
        return status

    def _mynn_peel_pending(self, qke_key="qke", flag_key="mynn_initialized"):
        """turbulence='mynn25': True when this run still owes its cold-start
        (initflag) step, from the host flag carried in ``carry_aux`` and the
      )
  )
  # Decrement indices since RRTMGP was originally developed in a 1-based index
  # system.
  bnd_limits_gpt = np.asarray(tables['bnd_limits_gpt']) - 1
  g_point_to_bnd = np.asarray([None] * dims['gpt'])
  for i in range(dims['bnd']):
    g_point_to_bnd[bnd_limits_gpt[i, 0] : bnd_limits_gpt[i, 1] + 1] = i
  g_point_to_bnd = np.array(g_point_to_bnd, dtype=jnp.int_)
  minor_lower_gpt_lims = np.asarray(tables['minor_limits_gpt_lower']) - 1
  (
170:    """Undo ``jax-mps``'s broken monkeypatch of ``jax.nn.gelu`` / ``dot_product_attention``.
172:    The installed ``jax-mps`` plugin (Apple Metal) monkeypatches ``jax.nn.gelu``
270:    # Plain os.environ, not monkeypatch: requesting monkeypatch here would
284:def allow_version_drift(monkeypatch):
287:    monkeypatch.setenv("LEGOESM_ALLOW_VERSION_DRIFT", "1")
packages/coupler/legoesm/driver/physics_pipeline.py:3790:                    solar_weights, s_0, o3_vmr, aerosol_od, aerosol_lw_od,
packages/coupler/legoesm/driver/physics_pipeline.py:3840:                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
packages/coupler/legoesm/driver/physics_pipeline.py:3898:                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
packages/coupler/legoesm/driver/physics_pipeline.py:4031:                     albedo_col, emis_col, o3_vmr_col, aerosol_od_col,
packages/coupler/legoesm/driver/physics_pipeline.py:4106:            aerosol_optical_depth=aerosol_od_col,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:550:    aerosol_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:571:    aerosol_od : jnp.ndarray, optional
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:582:        Proxy column AOD ``(ncol,)`` overriding the ``aerosol_od`` sum.
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:590:        if aerosol_od is None:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:592:                "activation scheme 'proxy' requires an aerosol_od field "
packages/atmosphere/legoesm/atmosphere/physics/microphysics/arg_activation.py:600:        return specified_nc_field(jnp.asarray(aerosol_od), target_shape, proxy_cfg,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:978:    aerosol_od: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1037:    aerosol_od : jnp.ndarray or None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1228:        aerosol_optical_depth=aerosol_od,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1695:    (2009) AOD->CCN diagnostic (``forcing["aerosol_od"]``) so the radiation
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1814:        #   aerosol_od  : (ncol, nlev) per-layer aerosol optical depth
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1820:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1969:            # missing-aerosol_od raise is proxy-only, mirroring the
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1979:                    "nc_from_aerosol=True but no 'aerosol_od' was passed to "
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1988:                aerosol_od=(None if _aer_ext is None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2146:            aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2207:                aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3271:        # (see _make_hydrostatic_radiation): o3_vmr / aerosol_od are
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3277:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3445:            aerosol_od=_aer_ext,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3501:    # o3_vmr / aerosol_od / ghg_vmr) — the spectral combined dispatcher
packages/coupler/legoesm/driver/model_driver.py:6159:        (aerosol_od, _ccn_aod, self._aerosol_lw_od,
packages/coupler/legoesm/driver/model_driver.py:6168:        return o3_vmr, aerosol_od, ghg_vmr
packages/coupler/legoesm/driver/model_driver.py:13395:        Aerosol-CCN specified-Nc fill needs ``forcing["aerosol_od"]`` in the
packages/coupler/legoesm/driver/model_driver.py:13399:        never thread aerosol_od and the microphysics fill would raise its
packages/coupler/legoesm/driver/model_driver.py:13628:        # prescribed ``forcing["aerosol_od"]`` and the warm-rain closures match
packages/coupler/legoesm/driver/model_driver.py:13632:        # and by the factories' fail-fast (no aerosol_od => raise).
packages/coupler/legoesm/driver/model_driver.py:15543:        # ``forcing["o3_vmr"|"aerosol_od"|"ghg_vmr"]`` and forwards them
packages/coupler/legoesm/driver/model_driver.py:16476:                        _forcing_daily["aerosol_od"] = _aer
packages/coupler/legoesm/driver/model_driver.py:16485:                        # absent; the volcanic SW is then in aerosol_od).
packages/coupler/legoesm/driver/model_driver.py:16488:                        # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
packages/coupler/legoesm/driver/model_driver.py:18466:                        _ext_daily = {"o3_vmr": _o3, "aerosol_od": _aer}
packages/coupler/legoesm/driver/model_driver.py:18474:                        # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
packages/coupler/legoesm/driver/model_driver.py:19409:        o3_vmr, aerosol_od = ctx["o3_vmr"], ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:19432:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:19445:                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:20017:        o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:20052:            "o3_vmr": o3_vmr, "aerosol_od": aerosol_od, "ghg_vmr": ghg_vmr,
packages/coupler/legoesm/driver/model_driver.py:20284:            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
packages/coupler/legoesm/driver/model_driver.py:20366:        aerosol_od = ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:20614:            # (s_0/ghg_vmr/o3_vmr/aerosol_od/solar_weights) are traced arrays of
packages/coupler/legoesm/driver/model_driver.py:20635:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:20661:                o3_vmr=o3_vmr, aerosol_od=aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:20927:                        aerosol_od=_g(forcing.aerosol_od),
packages/coupler/legoesm/driver/model_driver.py:21426:        aerosol_od = ctx["aerosol_od"]
packages/coupler/legoesm/driver/model_driver.py:21503:            o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:21548:                o3_vmr, aerosol_od,
packages/coupler/legoesm/driver/model_driver.py:21666:                o3_vmr, aerosol_od, ghg_vmr = self._precompute_external_forcing(
packages/coupler/legoesm/driver/model_driver.py:21694:                    o3_vmr, aerosol_od,
packages/atmosphere/legoesm/atmosphere/physics/neural_physics.py:586:            aerosol_od,
packages/atmosphere/legoesm/atmosphere/physics/neural_physics.py:595:        del solar_weights, o3_vmr, aerosol_od
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:758:      # started threading per-step o3_vmr / aerosol_od (2026-06-10).
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1122:          aerosol_od_3d = jnp.clip(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1126:          aerosol_od_3d = None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1155:          aerosol_od_lw_3d = jnp.clip(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1160:          aerosol_od_lw_3d = None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1190:              aerosol_optical_depth=aerosol_od_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1234:              aerosol_absorption_optical_depth=aerosol_od_lw_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1269:          aerosol_absorption_optical_depth=aerosol_od_lw_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1291:          aerosol_optical_depth=aerosol_od_3d,

exec
/usr/bin/bash -lc "sed -n '950,1000p' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py; sed -n '1770,1815p' packages/ml/legoesm/training/scale_build.py; sed -n '1,100p' packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py; rg -n '"'^class PrecisionPolicy|compute_fp32|def set_policy'"' packages/core/legoesm/core/precision.py; sed -n '556,610p' packages/atmosphere/legoesm/atmosphere/physics/neural_physics.py; sed -n '181,201p' tests/conftest.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
    return load_mitchell_ice_optics(path)


def _call_radiation_backend(
    radiation_config: RadiationConfig,
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    insolation: jnp.ndarray,
    cos_sza: jnp.ndarray | None = None,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
    open_ocean_frac: jnp.ndarray | None = None,
    q_cloud: jnp.ndarray | None = None,
    q_ice: jnp.ndarray | None = None,
    n_ice: jnp.ndarray | None = None,
    n_cloud: jnp.ndarray | None = None,
    n_cloud_in_cloud: bool = False,
    ghg_vmr_override: dict | None = None,
    f_day: jnp.ndarray | None = None,
    rrtmgp_solver=None,
    lon: jnp.ndarray | None = None,
    ml_ozone_coefs=None,
    o3_vmr_override: jnp.ndarray | None = None,
    o3_top_vmr: jnp.ndarray | None = None,
    aerosol_od: jnp.ndarray | None = None,
    aerosol_lw_od: jnp.ndarray | None = None,
    aerosol_sw_volc: jnp.ndarray | None = None,
    solar_spectral_fraction: jnp.ndarray | None = None,
    eccf: float | jnp.ndarray = 1.0,
    tsi: jnp.ndarray | None = None,
    cloud_fraction_override: jnp.ndarray | None = None,
    conv_precip: jnp.ndarray | None = None,
    conv_mass_flux_up: jnp.ndarray | None = None,
    conv_icwmr: jnp.ndarray | None = None,
    clear_sky_toa: bool = False,
    column_mesh=None,
    clubb_config=None,
    mg2_ice_dcs_m: float | None = None,
    mg2_ice_slope_log_width: float | None = None,
    q_snow: jnp.ndarray | None = None,
    n_snow: jnp.ndarray | None = None,
    n_cloud_specified: float | None = None,
):
    """Call configured radiation backend with a unified integration interface.

    Parameters
    ----------
    import jax.numpy as jnp

    from legoesm.training.era5_to_state import (
        era5_to_latlon_carry,
        open_era5_zarr,
        regrid_2d_to_gaussian,
    era5_terrain_product,
)
    from legoesm.driver.compiled_segments import pack_forcing

    if getattr(cfg, "training_core", "latlon") == "spectral":
        return _load_era5_samples_spectral(
            cfg, yml, grid, sigma,
            rank=rank, nproc=nproc, host_resident=host_resident)

    global _WARNED_ERA5_FLUX_HOLD_FIXED

    era5_cfg = wb_era5_config(cfg, yml)
    ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    snaps_per_day = 24 // era5_cfg.dt_hours
    roll_h = rollout_hours(cfg, yml)
    stride = int(roll_h) // era5_cfg.dt_hours

    # Opened once, like the state store — a per-sample remote zarr open would
    # dominate the load (see the spectral loader).
    cloud_ds = (open_era5_zarr(era5_cfg.cloud_zarr)
                if era5_cfg.load_cloud_condensate else None)
    # 2-D only, but keeps load_surface_fluxes / load_land_frac so the
    # sample-start slice carries the prescribed planes.
    sst_cfg = era5_cfg._replace(load_cloud_condensate=False)
    # Flux store opened once, only when fluxes are prescribed ("" = the
    # state store); a land fraction alone comes from the state store.
    flux_ds = None
    if sst_cfg.load_surface_fluxes:
        flux_ds = open_era5_zarr(sst_cfg.flux_zarr) if sst_cfg.flux_zarr else ds
    _micro = wb_ic_microphysics(cfg.mode, yml)

    config = build_latlon_config(cfg, yml)
    driver = _driver_for_ctx(config)
    ctx = driver._prepare_run_context(0, config.start_day, restore_carry=False)

    # Spectral forcing-dict name -> driver-path (pack_forcing) name for the
    # planes the two lanes name differently; the radiative planes and the
    # land fraction already share their names.
    _spec_to_driver = {
"""Configuration for atmospheric radiation schemes.

Provides configuration NamedTuples for:
1. Gray two-stream radiation (Frierson et al. 2006 / O'Gorman & Schneider 2008)
2. RRTMGP correlated-k radiation (via jax-rrtmgp)
3. Top-level RadiationConfig that selects the active scheme.

References
----------
- Frierson, D. M. W., Held, I. M., & Zurita-Gotor, P. (2006).
  A Gray-Radiation Aquaplanet Moist GCM. Part I: Static Stability
  and Eddy Scale. J. Atmos. Sci., 63, 2548-2566.
- O'Gorman, P. A. & Schneider, T. (2008). The Hydrological Cycle
  over a Wide Range of Climates Simulated with an Idealized GCM.
  J. Climate, 21, 3815-3832.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs.config import EcradIFSConfig
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.simple_lw import (
    SimpleLWConfig,
)

if TYPE_CHECKING:
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.radiation.solar import OrbitalParameters


# Machine-readable tunable/fixed split for the radiation scheme configs.
# Reviewed 2026-06-13 (placeholders replaced with physically-correct
# classifications):
#   * gray tau_equator/tau_pole were PRIMARY trained knobs until 2026-08-11;
#     gray radiation is no longer trained at all, so every gray param is now
#     tier 0 (never selected) with its bounds retained for the LES feedback
#     loop's per-column promotion clamp;
#   * surface albedo (the AIMIP-trained blended-surface knob that reaches the
#     heating through both solvers) -> tier 1;
#   * other closure knobs (LW/SW optical-depth shape, ozone profile,
#     emissivity) -> tier 2;
#   * RRTMGP gas concentrations and bulk aerosol optics are read into the
#     optics-cache key (any flip rebuilds the solver instance, see
#     RRTMGP._instance_cache_key) and are NOT meant to be trained -> tier 3.
__param_spec__ = {
    "GrayRadiationConfig": {
        "scheme_key": "atm.rad.GrayRadiationConfig",
        "excluded": {
            "sfc_emissivity": "physics: surface boundary emissivity is a domain boundary condition, not a sigmoid-tunable closure (fix via config)",
        },
        "params": {
        # Gray radiation is NOT trainable: every knob below sits at
        # tunable_tier 0, which build_trainable_params never selects (it takes
        # 1 <= tier <= level). The classical model trains RRTMGP instead —
        # there, only the surface albedo and emissivity are tunable. Bounds are
        # retained deliberately: promotable_params.py promotes gray_tau_equator
        # / gray_tau_pole to per-column fields in the LES feedback loop, and
        # feedback.param_field_bounds clamps that diagnosis to these ranges.
        # User directive 2026-08-11.
        "tau_equator": {"units": "1", "bounds": (2.0, 15.0), "tunable_tier": 0, "transform": "sigmoid", "category": "optical_depth", "reference": "Frierson et al. (2006)", "shape": None, "legacy_name": "tau_equator"},
        "tau_pole": {"units": "1", "bounds": (0.5, 5.0), "tunable_tier": 0, "transform": "sigmoid", "category": "optical_depth", "reference": "Frierson et al. (2006)", "shape": None, "legacy_name": "tau_pole"},
        "linear_frac": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0, "transform": "sigmoid", "category": "longwave", "reference": "O'Gorman & Schneider (2008)", "shape": None},
        "tau_moist_coeff": {"units": "m2/kg", "bounds": (0.0, 0.05), "tunable_tier": 0, "transform": "sigmoid", "category": "longwave", "reference": "Frierson et al. (2006)", "shape": None},
        "lw_diff_factor": {"units": "1", "bounds": (1.0, 2.0), "tunable_tier": 0, "transform": "sigmoid", "category": "longwave", "reference": "Fu & Liou (1992) diffusivity factor", "shape": None},
        "sw_tau_0": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0, "transform": "sigmoid", "category": "shortwave", "reference": "Frierson et al. (2006)", "shape": None},
        "sw_exponent": {"units": "1", "bounds": (0.5, 6.0), "tunable_tier": 0, "transform": "sigmoid", "category": "shortwave", "reference": "gray radiation scheme default", "shape": None},
        "sfc_albedo": {"units": "1", "bounds": (0.0, 1.0), "tunable_tier": 0, "transform": "sigmoid", "category": "albedo", "reference": "gray radiation scheme default", "shape": None},
        "obliquity": {"units": "degree", "bounds": (0.0, 90.0), "tunable_tier": 0, "transform": "sigmoid", "category": "solar", "reference": "gray radiation scheme default", "shape": None},
        },
    },
    "OzoneProfileConfig": {
        "scheme_key": "atm.rad.OzoneProfileConfig",
        "excluded": {
        },
        "params": {
            "p_peak_hPa": {"units": "hPa", "bounds": (1.0, 100.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ozone_profile", "reference": "US Standard Atmosphere 1976 / analytical ozone profile default", "shape": None},
            "o3_max_vmr": {"units": "mol/mol", "bounds": (1.0e-06, 2.0e-05), "tunable_tier": 2, "transform": "sigmoid", "category": "ozone_profile", "reference": "US Standard Atmosphere 1976 / analytical ozone profile default", "shape": None},
            "sigma_logp": {"units": "1", "bounds": (0.3, 4.0), "tunable_tier": 2, "transform": "sigmoid", "category": "ozone_profile", "reference": "analytical ozone profile default", "shape": None},
        },
    },
    "RRTMGPConfig": {
        "scheme_key": "atm.rad.RRTMGPConfig",
        "excluded": {
            # RRTMGP optics/solver instances are cached keyed on these Python
            # config floats (rrtmgp.py _optics_cache_key / _instance_cache_key).
            # A traced trainable leaf here would break the Python hash or rebuild
            # the solver per parameter value -- NOT trainable until RRTMGP
            # consumes them as per-call traced inputs. Fixed (tier 0).
            "co2_ppmv": "RRTMGP optics-cache key (Python-hashed); not trainable until consumed as a traced input",
            "ch4_ppbv": "RRTMGP optics-cache key (Python-hashed); not trainable until consumed as a traced input",
            "n2o_ppbv": "RRTMGP optics-cache key (Python-hashed); not trainable until consumed as a traced input",
            "aerosol_ssa": "RRTMGP instance-cache key (Python-hashed); not trainable until consumed as a traced input",
            "aerosol_g": "RRTMGP instance-cache key (Python-hashed); not trainable until consumed as a traced input",
            # sfc_albedo/sfc_emissivity are also folded into _instance_cache_key
            # via _hashable() (which does float(x)); a traced leaf would fail the
            # hash. The TRAINABLE surface-albedo path is the coupler's legacy-8
            # albedo_ice/albedo_ocean -> albedo_col (a per-call traced input), not
80:class PrecisionPolicy(NamedTuple):
173:def set_policy(policy: PrecisionPolicy) -> None:
    ----------
    neural_physics : NeuralPhysics
        Trained (or untrained) Equinox column MLP.
    adapter : ColumnAdapter
        Grid adapter for flatten/unflatten operations.

    Returns
    -------
    callable
        Supports both the legacy signature
        ``step_unified(..., q_r, u, v, ...)`` and the newer
        ``step_unified(..., q_r, conv_prog, u, v, ...)`` layout.
    """
    nlev = neural_physics.nlev

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = parse_step_unified_tail(args)
        (
            u,
            v,
            sst,
            sic,
            lat,
            lon,
            day_of_year,
            seconds_of_day,
            dt,
            solar_weights,
            s_0,
            o3_vmr,
            aerosol_od,
            held_dT_rad,
            held_sw_net_sfc,
            held_lw_net_sfc,
            held_sw_up_toa,
            held_lw_up_toa,
            held_sw_down_toa,
        ) = tail
        del need_rad, dt
        del solar_weights, o3_vmr, aerosol_od
        # Optional prescribed-input columns, built from the driver-path
        # kwargs EXACTLY like the spectral wrapper builds them from the
        # spectral forcing dict: static [sin(lat), cos(lat), land_frac,
        # orography in km] then the six surface fluxes in
        # SFC_FLUX_FORCING_KEYS order divided by SFC_FLUX_INPUT_NORMS (the
        # driver-path key names differ -- SFC_FLUX_STEP_UNIFIED_KEYS).
        # Every plane is flattened with the adapter, so this is
        # grid-agnostic (lat-lon, cubed sphere, MPAS).  A flag on with its
        # keyword missing is an error naming it, never silent zeros.
        # Flags off -> no extras, legacy path unchanged.
        if neural_physics.n_extra_dyn > 0:
            planes = []
            if neural_physics.n_static > 0:
                land_frac = kwargs.get("land_frac")
                phis = kwargs.get("phis")
    """
    try:
        from jax_plugins.mps import ops as _mps_ops
    except Exception:
        return
    # The patch is applied LAZILY on first backend init; force it so the plugin
    # has saved the originals (``_gelu_original`` / ``_sdpa_original``) before we
    # read them back.
    jnp.zeros(())
    import jax.nn as _jnn
    from jax._src.nn import functions as _nnf

    original_gelu = getattr(_mps_ops, "_gelu_original", None)
    if original_gelu is not None and getattr(_jnn.gelu, "_mps_patched", False):
        _jnn.gelu = original_gelu
        _nnf.gelu = original_gelu
    original_sdpa = getattr(_mps_ops, "_sdpa_original", None)
    if original_sdpa is not None and getattr(
        _jnn.dot_product_attention, "_mps_patched", False
    ):
        _jnn.dot_product_attention = original_sdpa

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu PYTHONPATH=packages/core:packages/atmosphere:packages/coupler:packages/tools:packages/ml:packages/land:packages/ocean:packages/ice /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'PY'
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as j
from legoesm.atmosphere.physics.radiation.integration import _call_radiation_backend
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
T=j.array([[220.,260.,285.]])
p=j.array([[5000.,30000.,80000.]])
ph=j.array([[1000.,10000.,50000.,100000.]])
kw=dict(T=T,p_full=p,p_half=ph,sfc_temperature=j.array([290.]),lat=j.array([0.]),q_v=j.ones_like(T)*.001,insolation=j.array([400.]))
for scheme in ['gray','simple_lw']:
    a=_call_radiation_backend(RadiationConfig(scheme=scheme),**kw)
    b=_call_radiation_backend(RadiationConfig(scheme=scheme),**kw,aerosol_sw_volc=j.ones((1,3,3,14)))
    eq=all(bool(j.array_equal(x,y)) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
    print(scheme,'accepted volcanic moments; output identical:',eq,flush=True)
PY" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 7262ms:
gray accepted volcanic moments; output identical: True
simple_lw accepted volcanic moments; output identical: True

exec
/usr/bin/bash -lc "nl -ba packages/tools/legoesm/forcing/external.py | sed -n '1645,1664p'; nl -ba packages/tools/legoesm/forcing/surface_utils.py | sed -n '833,855p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '20278,20292p'; nl -ba packages/ml/legoesm/training/scale_build.py | sed -n '1851,1863p'; nl -ba tests/unit/test_volcanic_sw_optics.py | sed -n '135,162p'; sed -n '80,171p' packages/core/legoesm/core/precision.py; rg -n '_aerosol_active =' packages/coupler/legoesm/driver/model_driver.py; git status --short" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 43ms:
  1645	        result_flat = np.broadcast_to(
  1646	            field.reshape(1, -1), (lat_deg.size, np.prod(field.shape[1:])),
  1647	        ).copy()
  1648	    else:
  1649	        idx = np.clip(np.searchsorted(lat_src, lat_deg) - 1, 0, nsrc - 2)
  1650	        x0 = lat_src[idx]
  1651	        x1 = lat_src[idx + 1]
  1652	        # Avoid division by zero on duplicated source points.
  1653	        dx = np.where(x1 > x0, x1 - x0, 1.0)
  1654	        w = np.where(x1 > x0, (lat_deg - x0) / dx, 0.0)
  1655	        # Clamp at the endpoints (matches np.interp's flat extrapolation).
  1656	        w = np.clip(w, 0.0, 1.0)
  1657	        # Reshape ``field`` to (nsrc, n_cols) and gather along leading.
  1658	        trailing_shape = field.shape[1:]
  1659	        n_cols = int(np.prod(trailing_shape))
  1660	        field_flat = np.asarray(field).reshape(nsrc, n_cols)
  1661	        f0 = field_flat[idx]                    # (n_lat, n_cols)
  1662	        f1 = field_flat[idx + 1]                # (n_lat, n_cols)
  1663	        result_flat = (1.0 - w[:, None]) * f0 + w[:, None] * f1
  1664	    return jnp.array(result_flat).reshape((*lat_grid.shape, *field.shape[1:]))
   833	        _acfg = acfg
   834	        # Volcanic SW with the file's own per-band extinction / SSA / g:
   835	        # each moment is placed at its true stratospheric pressure (linear
   836	        # remap + tropopause weight, so mixing the remapped moments is exact)
   837	        # and kept OUT of the grey ``aerosol_od``.
   838	        volc_opt = get_aerosol_sw_volcanic_optics_at_time(_acfg, day, lat_col)
   839	        if volc_opt is not None:
   840	            mom, p_edges_v, _band_wn = volc_opt          # (ncol, nsrc, 3, nb)
   841	            nsrc, n_m, n_b = mom.shape[1:]
   842	            flat = jnp.moveaxis(jnp.asarray(mom).reshape(ncol, nsrc, n_m * n_b),
   843	                                -1, 0)                   # (3*nb, ncol, nsrc)
   844	            placed = jax.vmap(place_stratospheric_aod_profile_to_layers,
   845	                              in_axes=(0, None, None))(
   846	                flat, jnp.asarray(p_edges_v), p_half_col) * w_strat
   847	            aerosol_sw_volc = jnp.moveaxis(placed, 0, -1).reshape(
   848	                ncol, nlev, n_m, n_b).astype(dtype)
   849	        # A legacy column-only volcanic file (no ext_sun) stays in the column.
   850	        _trop_cfg = _acfg._replace(volcanic_enabled=False)
   851	        aerosol_col = get_aerosol_at_time(
   852	            _acfg if volc_opt is None else _trop_cfg,
   853	            day, lat_grid=lat_col, lon_grid=lon_col,
   854	        )
   855	        aerosol_od = distribute_column_aod_to_layers(
 20278	            sst = jnp.full(ctx["shape_2d"], 290.0, dtype=_sd)
 20279	            sic = jnp.zeros(ctx["shape_2d"], dtype=_sd)
 20280	        forcing = pack_forcing(
 20281	            sst=jnp.asarray(sst), sic=jnp.asarray(sic),
 20282	            day_of_year=day_of_year, seconds_of_day=seconds_of_day,
 20283	            solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
 20284	            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
 20285	            aerosol_ccn_aod=getattr(self, "_aerosol_ccn_aod", None),
 20286	            o3_top_vmr=getattr(self, "_o3_top_vmr", None),
 20287	            ghg_vmr=ctx["ghg_vmr"],
 20288	            # NOTE: transient cover is deliberately NOT injected here.  The training
 20289	            # segment returns run_segment.raw (un-jitted), so the land calibration
 20290	            # differentiates w.r.t. the ``pipe.land_ml_params`` ATTRIBUTE (read fresh
 20291	            # each trace — see test_build_training_segment_land_gradient); a non-None
 20292	            # land_ml_params forcing leaf would SHADOW that attribute and break the
  1851	                    / np.timedelta64(1, "D"))
  1852	        forcing = pack_forcing(
  1853	            sst=sst, sic=jnp.zeros_like(sst),
  1854	            day_of_year=jnp.asarray(doy), seconds_of_day=jnp.asarray(0.0),
  1855	            solar_weights=ctx["solar_weights"], s_0=ctx["current_s_0"],
  1856	            o3_vmr=ctx["o3_vmr"], aerosol_od=ctx["aerosol_od"],
  1857	            o3_top_vmr=getattr(driver, "_o3_top_vmr", None),
  1858	            **plane_kwargs)
  1859	        return (ic, target, forcing)
  1860	
  1861	    mine = list(_sharded_indices(
  1862	        cfg, yml, times, snaps_per_day, stride, rank, nproc))
  1863	    src = (ds, cloud_ds, flux_ds, int)
   135	                cos_zenith=jnp.array([0.6, 0.3])[:ncol],
   136	                aerosol_optical_depth=jnp.full((ncol, nlev), 0.01))
   137	
   138	
   139	def _volc_moments(kw, tau_b, ssa_b, g_b, layers=slice(1, 4)):
   140	    ncol, nlev = kw["T"].shape
   141	    tau = jnp.zeros((ncol, nlev, 14)).at[:, layers, :].set(jnp.asarray(tau_b))
   142	    return jnp.stack([tau, tau * jnp.asarray(ssa_b),
   143	                      tau * jnp.asarray(ssa_b) * jnp.asarray(g_b)], axis=2)
   144	
   145	
   146	def _solver():
   147	    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
   148	    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
   149	    cfg = RRTMGPConfig()
   150	    return RRTMGP.from_legoesm_config(cfg), cfg
   151	
   152	
   153	def test_grey_moments_reproduce_grey_path():
   154	    """Volcanic tau with the grey ssa/g == the same tau in aerosol_od."""
   155	    solver, cfg = _solver()
   156	    kw = _columns()
   157	    volc = _volc_moments(kw, 0.05, cfg.aerosol_ssa, cfg.aerosol_g)
   158	    a = solver.solve_columns(**kw, aerosol_volcanic_moments_sw=volc)
   159	    kw_grey = {**kw, "aerosol_optical_depth":
   160	               kw["aerosol_optical_depth"] + volc[:, :, 0, 0]}
   161	    b = solver.solve_columns(**kw_grey)
   162	    for f in ("sw_flux_up", "sw_flux_down", "sw_heating_rate"):
class PrecisionPolicy(NamedTuple):
    """Global precision configuration.

    Attributes
    ----------
    storage : jnp.dtype
        Dtype for state arrays in memory. Determines GPU memory footprint.
    compute : jnp.dtype
        Dtype for arithmetic in kernels (matmuls, stencils, tendencies).
    accumulate : jnp.dtype
        Dtype for reductions (global sums, budget integrals, norms).
        Must be >= compute width to avoid catastrophic cancellation.
    control : jnp.dtype
        Dtype for solvers, implicit systems, and diagnostics that
        require tight tolerances (semi-implicit, barotropic, EOS).
    transcendentals : str
        ``"native"`` uses JAX/XLA math intrinsics. ``"libm"`` selects the
        CPU-only scalar system-libm callbacks used by NEMO certification.
    """
    storage: jnp.dtype = jnp.float32
    compute: jnp.dtype = jnp.float32
    accumulate: jnp.dtype = jnp.float32
    control: jnp.dtype = jnp.float32
    transcendentals: str = "native"

    @staticmethod
    def fp32(*, transcendentals: str = "native") -> PrecisionPolicy:
        """All-float32 mode. Fastest, GPU-optimal."""
        return PrecisionPolicy(
            storage=jnp.float32,
            compute=jnp.float32,
            accumulate=jnp.float32,
            control=jnp.float32,
            transcendentals=transcendentals,
        )

    @staticmethod
    def fp64(*, transcendentals: str = "native") -> PrecisionPolicy:
        """All-float64 mode. Full scientific reference."""
        return PrecisionPolicy(
            storage=jnp.float64,
            compute=jnp.float64,
            accumulate=jnp.float64,
            control=jnp.float64,
            transcendentals=transcendentals,
        )

    @staticmethod
    def mixed(*, transcendentals: str = "native") -> PrecisionPolicy:
        """Mixed-precision mode.

        Storage and compute in float32 for GPU bandwidth.
        Accumulation and control in float64 for stability.
        """
        return PrecisionPolicy(
            storage=jnp.float32,
            compute=jnp.float32,
            accumulate=jnp.float64,
            control=jnp.float64,
            transcendentals=transcendentals,
        )

    @staticmethod
    def mixed_fp64_storage(*, transcendentals: str = "native") -> PrecisionPolicy:
        """Mixed mode with float64 storage, float32 compute.

        State arrays are stored in float64 for maximum precision in
        I/O, checkpointing, and long-run accumulation.  Compute kernels
        run in float32 for GPU throughput.  Accumulation and control
        remain float64.
        """
        return PrecisionPolicy(
            storage=jnp.float64,
            compute=jnp.float32,
            accumulate=jnp.float64,
            control=jnp.float64,
            transcendentals=transcendentals,
        )


# Module-level recommended dtypes: maps (module_name, role) -> role override.
# When a module has a specific precision requirement, it overrides the global
# policy for that role. None means "use the global policy default".
_MODULE_OVERRIDES: dict[str, dict[str, str | None]] = {}

# The active global policy.
_ACTIVE_POLICY: list[PrecisionPolicy] = [PrecisionPolicy.fp32()]


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------
5840:        self._aerosol_active = (cfg.radiation in ("rrtmg", "rrtmgp")
 M packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
 M packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
 M packages/coupler/legoesm/driver/compiled_segments.py
 M packages/coupler/legoesm/driver/model_driver.py
 M packages/coupler/legoesm/driver/physics_pipeline.py
 M packages/ml/legoesm/training/scale_build.py
 M packages/tools/legoesm/forcing/external.py
 M packages/tools/legoesm/forcing/surface_utils.py
 A scripts/validate/probe_volcanic_sw_optics.py
 M tests/unit/test_aerosol_band_ccn.py
 M tests/unit/test_cross_lane_forcing.py
 A tests/unit/test_volcanic_sw_optics.py
 M tests/unit/test_wb_amip_deck_params.py
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_r1_prompt.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_codex.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_glm.md
?? docs/wb/reviews/2026-11-22_volcanic_optics_vote_prompt.md

exec
/usr/bin/bash -lc "rg -n 'set_policy|PrecisionPolicy|compute_fp32' tests/conftest.py packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py; sed -n '720,767p' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py; rg -n 'compute_dtype|_table_dtype' packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py | head -18; git diff HEAD --stat" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 81ms:
packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py:301:    compute_fp32: bool = False
          before it, as CAM does.  ``None`` => 1.

      Returns
      -------
      RadiationOutput
          Fluxes and heating rates.
      """
      from legoesm.atmosphere.physics.radiation.output import RadiationOutput

      config = self._config
      ncol, nlev = T.shape

      # --- 0. Determine working dtype ---
      # The optics tables are loaded at whatever precision JAX was configured
      # with at load time (float32 if x64 off, float64 if x64 on).  Promote
      # all inputs to match so that table lookups, lax.cond branches, and
      # lax.scan carries have consistent dtypes throughout the solver.
      _table_dtype = self.optics_lib.gas_optics_lw.kmajor.dtype
      # McICA per-column table shift, seeded like CAM's mcica_subcol_gen
      # (clouds.subcolumns.cam_seed_shift), taken before the cast to the
      # table dtype.
      if mcica_cloud_fraction is None:
          _mcica_shift = None
      else:
          from legoesm.atmosphere.physics.clouds.subcolumns import (
              cam_seed_shift)
          _mcica_shift = cam_seed_shift(p_full)
      T = T.astype(_table_dtype)
      p_full = p_full.astype(_table_dtype)
      p_half = p_half.astype(_table_dtype)
      q_v = q_v.astype(_table_dtype)
      sfc_temperature = jnp.asarray(sfc_temperature).astype(_table_dtype)
      cos_zenith = jnp.asarray(cos_zenith).astype(_table_dtype)
      # Optional external-forcing arrays must match the working dtype
      # too: under ``compute_fp32`` a float64 ozone / aerosol column
      # (built by the driver under JAX x64) would re-promote the optical
      # depth and break the RTE ``lax.scan`` carry dtype — exactly the
      # PR #343 leak class, resurfaced when the MPAS/spectral paths
      # started threading per-step o3_vmr / aerosol_od (2026-06-10).
      if o3_vmr is not None:
          o3_vmr = jnp.asarray(o3_vmr).astype(_table_dtype)
      if aerosol_optical_depth is not None:
          aerosol_optical_depth = jnp.asarray(
              aerosol_optical_depth).astype(_table_dtype)
      if aerosol_absorption_optical_depth_lw is not None:
          aerosol_absorption_optical_depth_lw = jnp.asarray(
              aerosol_absorption_optical_depth_lw).astype(_table_dtype)
      if aerosol_volcanic_moments_sw is not None:
737:      _table_dtype = self.optics_lib.gas_optics_lw.kmajor.dtype
747:      T = T.astype(_table_dtype)
748:      p_full = p_full.astype(_table_dtype)
749:      p_half = p_half.astype(_table_dtype)
750:      q_v = q_v.astype(_table_dtype)
751:      sfc_temperature = jnp.asarray(sfc_temperature).astype(_table_dtype)
752:      cos_zenith = jnp.asarray(cos_zenith).astype(_table_dtype)
760:          o3_vmr = jnp.asarray(o3_vmr).astype(_table_dtype)
763:              aerosol_optical_depth).astype(_table_dtype)
766:              aerosol_absorption_optical_depth_lw).astype(_table_dtype)
769:              aerosol_volcanic_moments_sw).astype(_table_dtype)
771:          cloud_path_liq = jnp.asarray(cloud_path_liq).astype(_table_dtype)
773:          cloud_path_ice = jnp.asarray(cloud_path_ice).astype(_table_dtype)
776:              cloud_path_liq_lw).astype(_table_dtype)
779:              cloud_path_ice_lw).astype(_table_dtype)
781:          cloud_r_eff_liq = jnp.asarray(cloud_r_eff_liq).astype(_table_dtype)
783:          cloud_r_eff_ice = jnp.asarray(cloud_r_eff_ice).astype(_table_dtype)
790:                  dtype=_table_dtype)
 .../atmosphere/physics/clouds/subcolumns.py        |   2 +-
 .../atmosphere/physics/radiation/integration.py    |  18 ++
 .../atmosphere/physics/radiation/rrtmgp/rrtmgp.py  |  35 +++
 .../physics/radiation/rrtmgp/rte/two_stream.py     |  31 ++-
 .../coupler/legoesm/driver/compiled_segments.py    |  18 +-
 packages/coupler/legoesm/driver/model_driver.py    |  14 +-
 .../coupler/legoesm/driver/physics_pipeline.py     |  21 +-
 packages/ml/legoesm/training/scale_build.py        |   9 +-
 packages/tools/legoesm/forcing/external.py         | 206 ++++++++++++++----
 packages/tools/legoesm/forcing/surface_utils.py    |  51 +++--
 scripts/validate/probe_volcanic_sw_optics.py       |  89 ++++++++
 tests/unit/test_aerosol_band_ccn.py                |  48 ++--
 tests/unit/test_cross_lane_forcing.py              |   2 +-
 tests/unit/test_volcanic_sw_optics.py              | 242 +++++++++++++++++++++
 tests/unit/test_wb_amip_deck_params.py             |   2 +-
 15 files changed, 689 insertions(+), 99 deletions(-)

codex
Gray and simple-LW integration both accept nonzero volcanic moments and return identical outputs to runs without them. At res6, the host interpolation also creates a float64 result of about 0.90 GiB before remapping, despite the float32 cache. The focused tests are still running.
exec
/usr/bin/bash -lc "git diff HEAD -- tests/unit/test_volcanic_sw_optics.py | tail -45; rg -n 'aerosol|forcing' packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/integration.py | head -30; sed -n '350,365p' packages/atmosphere/legoesm/atmosphere/physics/radiation/config.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
+        volc = _volc_moments(kw, 0.08 * scale, 0.99, 0.7)
+        return jnp.sum(solver.solve_columns(
+            **kw, aerosol_volcanic_moments_sw=volc).sw_flux_up[:, 0])
+
+    eager = toa_up(1.0)
+    np.testing.assert_allclose(jax.jit(toa_up)(1.0), eager, rtol=1e-12)
+    g = jax.grad(toa_up)(1.0)
+    assert np.isfinite(g) and g > 0.0
+    fd = (toa_up(1.0 + 1e-5) - toa_up(1.0 - 1e-5)) / 2e-5
+    np.testing.assert_allclose(g, fd, rtol=1e-5)
+
+
+def test_wrong_band_count_raises():
+    solver, _ = _solver()
+    kw = _columns()
+    with pytest.raises(ValueError, match="trailing shape"):
+        solver.solve_columns(**kw, aerosol_volcanic_moments_sw=jnp.zeros(
+            kw["T"].shape + (3, 13)))
+
+
+@pytest.mark.parametrize("builder", ["_build_gray_radiation_fn",
+                                     "_build_none_radiation_fn"])
+def test_non_rrtmgp_radiation_refuses(builder):
+    from legoesm.driver import physics_pipeline as pp
+    from legoesm.driver.config import ExperimentConfig
+    fn, _ = getattr(pp, builder)(ExperimentConfig(radiation="gray"))
+    kw = _columns()
+    ncol, nlev = kw["T"].shape
+    args = (kw["T"], kw["p_full"], kw["p_half"], kw["q_v"],
+            kw["sfc_temperature"], jnp.zeros(ncol), jnp.zeros(ncol), 180.0,
+            43200.0, jnp.full((ncol,), 0.1), jnp.ones(ncol),
+            jnp.zeros((ncol, nlev)), jnp.zeros((ncol, nlev)), jnp.zeros(0))
+    with pytest.raises(ValueError, match="volcanic SW"):
+        fn(*args, aerosol_sw_volc_col=jnp.zeros((ncol, nlev, 3, 14)))
+
+
+def test_probe_runs_on_synthetic_file(tmp_path, capsys):
+    import importlib.util
+    spec = importlib.util.spec_from_file_location(
+        "probe", _REPO / "scripts/validate/probe_volcanic_sw_optics.py")
+    probe = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(probe)
+    probe.main(volc_sw_file(tmp_path / "v.nc", omega=np.full(14, 1.0)), 1979, 1)
+    out = capsys.readouterr().out
+    assert "file-grey" in out and "ssa 1.0000" in out
rg: packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/integration.py: IO error for operation on packages/atmosphere/legoesm/atmosphere/physics/radiation/mc3d/integration.py: No such file or directory (os error 2)
        - ``"none"``: zero ozone (disables ozone absorption entirely).
        - ``"ml"``: machine-learning ridge regression predictor of Ma et al.
          (UKESM-trained, per-gridpoint T -> O3 column).  Requires
          ``ml_weights_path`` to point at a directory of NetCDF weights;
          see :mod:`legoesm.atmosphere.physics.radiation.ozone_ml`.
    p_peak_hPa : float
        Peak pressure [hPa] for the analytical profile (default 30.0).
    o3_max_vmr : float
        Maximum ozone VMR at the profile peak (default 8.0e-6 = 8 ppmv).
    sigma_logp : float
        Width of the Gaussian in log-pressure space (default 1.5).
    lat_dependence : bool
        If True, scale analytical ozone by ``1 + 0.5 * sin²(lat)``
        (default True).  Only used when ``source="analytical"``.
    ml_weights_path : str or None
        Directory of NetCDF coefficient files for ``source="ml"``.  Must

exec
/usr/bin/bash -lc "rg -n 'mc3d|optical_field_only|forcing=' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | head -70; rg -n 'sw_optical_field_only|aerosol_optical_depth|forcing.get\\(\"aerosol' packages/atmosphere --glob '*.py' | tail -55; sed -n '1650,1685p' packages/coupler/legoesm/driver/compiled_segments.py; sed -n '18395,18483p' packages/coupler/legoesm/driver/model_driver.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 0ms:
142:        # simple_lw / mc3d take no surface temperature or albedo override,
1414:    # mc3d shares the RRTMGP optics path (Phase 2b: it calls solve_columns for
1418:        radiation_config.scheme in ("rrtmgp", "mc3d")
1507:    if radiation_config.scheme == "mc3d" and model_type != "plane":
1509:            "scheme='mc3d' (3D Monte-Carlo ray tracing) is only wired for "
1543:    # Load heavy/static RRTMGP optics once outside model JIT traces. mc3d also
1546:    # mc3d used to fall back silently to gray optics, which also hid config
1549:    if radiation_config.scheme in ("rrtmgp", "mc3d"):
1779:                   forcing=None, phys_state=None) -> HydrostaticTendencies:
2626:def _mc3d_plane_heating(
2663:    from legoesm.atmosphere.physics.radiation.mc3d import plane_adapter
2666:    mc_cfg = radiation_config.mc3d
2674:            "scheme='mc3d' requires an instantaneous solar zenith angle: set "
2707:    # NOTE: the plane mc3d path is AEROSOL-FREE (no aerosol_optical_depth is
2718:            sw_optical_field_only=True, **cloud_kwargs,
2731:            from legoesm.atmosphere.physics.radiation.mc3d import mie as _mie
2779:                lw_optical_field_only=True, **cloud_kwargs,
2921:        if radiation_config.scheme == "mc3d":
2923:            dT_dt = _mc3d_plane_heating(
3241:        sim_time_seconds=0.0, forcing=None, phys_state=None,
3489:            sim_time_seconds=0.0, forcing=None, phys_state=None,
packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py:415:    "aerosol_optical_depth", "aerosol_absorption_optical_depth_lw",
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:783:            _aer_od = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:789:                forcing.get("aerosol_number") if forcing is not None else None)
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:804:                    ccn_aod=(forcing.get("aerosol_ccn_aod")
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1040:        RRTMGP solver as ``aerosol_optical_depth``.  Ignored by gray
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1228:        aerosol_optical_depth=aerosol_od,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1820:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1822:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1825:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1984:            _aer_num = (forcing.get("aerosol_number")
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1993:                ccn_aod=(forcing.get("aerosol_ccn_aod")
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2707:    # NOTE: the plane mc3d path is AEROSOL-FREE (no aerosol_optical_depth is
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2718:            sw_optical_field_only=True, **cloud_kwargs,
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3277:        _aer_ext = forcing.get("aerosol_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3279:            forcing.get("aerosol_lw_od") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:3282:            forcing.get("aerosol_sw_volc") if forcing is not None else None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py:34:        "aerosol_optical_depth": "1",
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py:111:    aerosol_optical_depth: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py:139:        aerosol_optical_depth=aerosol_optical_depth,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:50:        "cloud_fraction": "1", "aerosol_optical_depth": "1",
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:629:    aerosol_optical_depth: Array | None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:656:      igpt, optics_lib, sw_optical_props, aerosol_optical_depth,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:665:    aerosol_optical_depth: Array | None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:678:  if aerosol_optical_depth is None and aerosol_volcanic_moments is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:679:    aerosol_optical_depth = jnp.zeros_like(sw_optical_props['optical_depth'])
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:680:  if aerosol_optical_depth is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:697:    tau_aer = jnp.maximum(aerosol_optical_depth, 0.0)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:766:    aerosol_optical_depth: Array | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:790:        cloud_fraction, aerosol_optical_depth,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:879:    aerosol_optical_depth: Array | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:919:    aerosol_optical_depth: Optional aerosol optical depth per layer (same
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:1022:          cloud_fraction, aerosol_optical_depth,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py:1036:      _aer = (aerosol_optical_depth, aerosol_single_scattering_albedo,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:54:        "aerosol_optical_depth": "1",
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:604:      aerosol_optical_depth: jnp.ndarray | None = None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:608:      sw_optical_field_only: bool = False,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:670:      aerosol_optical_depth : jnp.ndarray | None
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:687:          ``aerosol_optical_depth`` as a second component.  ``None`` = absent.
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:761:      if aerosol_optical_depth is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:762:          aerosol_optical_depth = jnp.asarray(
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:763:              aerosol_optical_depth).astype(_table_dtype)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:806:                  or sw_optical_field_only or lw_optical_field_only):
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:845:                  and not (sw_optical_field_only or lw_optical_field_only))
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:878:          aerosol_optical_depth = _zero(aerosol_optical_depth)
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1078:          if sw_optical_field_only or lw_optical_field_only:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1121:      if aerosol_optical_depth is not None:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1123:              _add_halos(aerosol_optical_depth[:, None, ::-1]), 0.0, None,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1178:      # default path (sw_optical_field_only=False) is byte-identical.
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1179:      if clear_sky and (sw_optical_field_only or lw_optical_field_only):
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1184:      if sw_optical_field_only:
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1190:              aerosol_optical_depth=aerosol_od_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1291:          aerosol_optical_depth=aerosol_od_3d,
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1417:      if opt.get("sw_optical_field_only") or opt.get("lw_optical_field_only"):
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py:1420:              "sw_optical_field_only / lw_optical_field_only return a "
                if phys_out.snow is not None
                else carry.snow)

    # --- Clear-sky TOA second radiation pass (#843) ---
    # Mirror the all-sky held refresh: feed the SAME post-dynamics state
    # step_unified's all-sky radiation saw (T_new / p_s_new / moist q_v,
    # u_new / v_new), but with clouds OFF (cloud_scheme="none", no
    # condensate/number tracers).  Held between radiation refreshes exactly
    # like held_sw_up_toa.  "hold" mode (feature off, or the no-rad subcycle
    # variant) runs NO extra RRTMGP, so it stays byte-identical + keeps
    # body_no_rad HLOs rrtmgp-free.
    def _compute_fresh_clr():
        (_c0, _c1, _c2, _sw_clr, _lw_clr, _c5, _c6, _c7) = \
            statics.pipeline.compute_radiation_core(
                T_new, p_s_new, moist["q_v"],
                statics.forcing.sst, statics.forcing.sic,
                statics.lat, statics.lon, doy_step, sod_step,
                statics.forcing.solar_weights, statics.forcing.s_0,
                statics.forcing.o3_vmr, statics.forcing.aerosol_od,
                aerosol_lw_od_precomputed=statics.forcing.aerosol_lw_od,
                aerosol_sw_volc_precomputed=statics.forcing.aerosol_sw_volc,
                aerosol_ccn_aod=statics.forcing.aerosol_ccn_aod,
                o3_top_vmr=statics.forcing.o3_top_vmr,
                tau_equator=statics.tau_equator, tau_pole=statics.tau_pole,
                albedo_ice=_skin_albedo(statics.pipeline, statics.albedo_ice,
                                        carry.ice_T_skin),
                albedo_ocean=statics.albedo_ocean,
                ghg_vmr_override=statics.ghg_vmr_override,
                q_c=None, q_i=None, N_c=None, N_i=None,
                cloud_scheme="none",
                u=u_cc, v=v_cc, dt=statics.dt,
                T_land=carry.T_land, land_ml=carry.land_ml,
                land_ml_params=statics.forcing.land_ml_params,
                sfc_albedo_override=statics.forcing.sfc_albedo_override,
                sfc_T_override=statics.forcing.sfc_T_override,
                sfc_emissivity_override=statics.forcing.sfc_emissivity_override,

        def _couple(n):
            """Hand the coupler the ``n`` steps since the last boundary (#2290).

            Called at the END of the step that closes a day or the run.
            """
            _sw_net, _lw_net = _spectral_sfc_net_rad(
                self.state, float(daily_forcing_bucket(self._current_day)))
            if not isinstance(self._carry_aux, dict):
                self._carry_aux = {}
            self._carry_aux["held_sw_net_sfc"] = _sw_net
            self._carry_aux["held_lw_net_sfc"] = _lw_net
            # Surface precipitation export [kg/m^2/s, +down] (#2277): the
            # segment-MEAN rate over the steps since the last boundary (the
            # compiled lanes' segment_accum_to_rate semantics).  Its frozen
            # part is None when a precipitating module does not publish one
            # -- the coupler then refuses (require_surface_radiation_aux).
            # The dry legacy path precipitates nothing: an exact zero.
            if _full_physics:
                if not _sp_seg["ok"] or _sp_seg["p"] is None:
                    raise RuntimeError(
                        "coupled spectral lane: a step since the last "
                        "coupling boundary published no surface "
                        "precipitation (model._sfc_precip is None: the "
                        "step path does not integrate it, e.g. the "
                        "semi-implicit RK); run the driver's explicit "
                        "ssp_rk54 or leapfrog_si integrator.")
                _seg_s = _sp_seg["n"] * DT
                self._carry_aux["seg_precip"] = (
                    _sp_seg["p"] / _seg_s).reshape(shape_2d)
                self._carry_aux["seg_precip_frozen"] = (
                    None if _sp_seg["frozen"] is None
                    else (_sp_seg["frozen"] / _seg_s).reshape(shape_2d))
                _sp_seg.update(_sp_acc_new())
            else:
                self._carry_aux["seg_precip"] = jnp.zeros_like(_sw_net)
                # nothing falls, nothing frozen (F29: the coupler needs it)
                self._carry_aux["seg_precip_frozen"] = jnp.zeros_like(_sw_net)
            # The segment's real length: a full day, or the steps a restart
            # link or the run's last partial day actually took.
            self._segment_callback(self, self._current_day, n * DT)

        for step in range(start_step, n_steps_total):
            self._current_day = START_DAY + (step + 1) * DT / 86400.0

            if _full_physics:
                # Traced forcing for the unified pipeline: SST/SIC-blend
                # surface anchor + per-step calendar time + (daily) the
                # external CMIP6 ozone/aerosol/GHG fields.
                sst_step, sic_step = self.get_sst_sic(self._current_day)
                if sst_step.ndim == 1:
                    sst_step = jnp.broadcast_to(sst_step[:, None], shape_2d)
                    sic_step = jnp.broadcast_to(sic_step[:, None], shape_2d)
                _T_sfc_step = blend_surface_temperature(
                    sst_step, sic_step, T_ice).reshape(-1)
                # Bucket on the step START time, as _run_column does (_current_day
                # is end-of-step: the day flipped one step early; gridaudit).
                _fd_int = daily_forcing_bucket(START_DAY + step * DT / 86400.0)
                if (_ext_forcing or _solar_ext) and _fd_int != _last_ext_day:
                    _ext_daily = {}
                    if _ext_forcing:
                        _f_now = spectral_pe_to_grid(
                            self.state, self.grid, self.sigma)
                        # Sample at the CANONICAL day boundary, not the first
                        # step entering the day — a restart link's first step
                        # lands mid-day (same bug class as the MPAS loop; see
                        # daily_forcing_bucket / FIX_RESTART_TIME).
                        _o3, _aer, _ghg = self._precompute_external_forcing(
                            float(_fd_int), _f_now['p_s'], _lat_2d_loop,
                            temp=_f_now['T'], lon=_lon_2d_loop,
                        )
                        _ext_daily = {"o3_vmr": _o3, "aerosol_od": _aer}
                        # Volcanic LONGWAVE aerosol (gap #9): only when active
                        # (omitted ⇒ None ⇒ RRTMGP no-op ⇒ byte-identical).
                        _aer_lw = getattr(self, "_aerosol_lw_od", None)
                        if self._aerosol_lw_active and _aer_lw is not None:
                            _ext_daily["aerosol_lw_od"] = _aer_lw
                        if getattr(self, "_aerosol_sw_volc", None) is not None:
                            _ext_daily["aerosol_sw_volc"] = self._aerosol_sw_volc
                        # Tropospheric visible-band CCN AOD (absent => sum(aerosol_od)).
                        if getattr(self, "_aerosol_ccn_aod", None) is not None:
                            _ext_daily["aerosol_ccn_aod"] = self._aerosol_ccn_aod
                        # Overhead (above-model-top) ozone for the RRTMGP overhead layer.
                        if getattr(self, "_o3_top_vmr", None) is not None:
                            _ext_daily["o3_top_vmr"] = self._o3_top_vmr
                        if _ghg is not None:
                            _ext_daily["ghg_vmr"] = {
                                k: jnp.asarray(v) for k, v in _ghg.items()
                            }

exec
/usr/bin/bash -lc "ps -u pg2328 -o pid,etime,%cpu,rss,args | rg 'pytest.*test_volcanic_sw_optics.py|python -c import jax' | head -8; sed -n '1550,1582p' packages/ml/legoesm/training/scale_build.py; sed -n '5840,5850p' packages/coupler/legoesm/driver/model_driver.py" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 307ms:
2775763       02:26  108 3405692 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -p no:cacheprovider tests/unit/test_volcanic_sw_optics.py tests/unit/test_aerosol_band_ccn.py -q
2776973       01:40  136 3174216 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -c import sys,glob; sys.path[:0]=glob.glob("packages/*")+["src"]; import legoesm.forcing.external as e; print(e.__file__); import pytest; sys.exit(pytest.main(["-q","-p","no:cacheprovider","tests/unit/test_volcanic_sw_optics.py","tests/unit/test_aerosol_band_ccn.py"]))
2780475       00:00 20.0  3072 /usr/bin/bash -c ps -u pg2328 -o pid,etime,%cpu,rss,args | rg 'pytest.*test_volcanic_sw_optics.py|python -c import jax' | head -8; sed -n '1550,1582p' packages/ml/legoesm/training/scale_build.py; sed -n '5840,5850p' packages/coupler/legoesm/driver/model_driver.py
2780488       00:00  0.0  5444 rg pytest.*test_volcanic_sw_optics.py|python -c import jax
    The AMIP production aerosol, as its driver builds it, but on a
    calendar-month CLIMATOLOGY over ``aerosol_climatology_years`` [first,
    last]: the files end 2014 and the WB windows are 2015-2020 (owner
    decision 2026-10-16)."""
    cl = dict(yml.get("classical", {}) or {})
    if not cl.get("aerosol_file"):
        return None
    from legoesm.forcing.external import AerosolConfig

    years = cl.get("aerosol_climatology_years")
    lw = cl.get("volcanic_aerosol_lw", False)
    if (not isinstance(years, list) or len(years) != 2
            or not all(isinstance(y, int) for y in years)
            or years[0] > years[1]):
        raise ValueError("classical.aerosol_climatology_years must be "
                         f"[first, last] integer years, got {years!r}")
    if not isinstance(lw, bool):
        raise ValueError(
            f"classical.volcanic_aerosol_lw must be a YAML boolean, got {lw!r}")
    volc = str(cl.get("volcanic_aerosol_file", "") or "")
    acfg = AerosolConfig(
        enabled=True, source="climatology", path=str(cl["aerosol_file"]),
        ccn_path=str(cl.get("aerosol_ccn_file", "") or ""),
        use_reference_if_missing=True,
        reference_aod_550=float(cl.get("aerosol_reference_aod", 0.03)),
        volcanic_enabled=bool(volc), volcanic_path=volc,
        volcanic_lw_enabled=(lw and bool(volc)),
        hold_outside_file_years=False)
    return acfg, tuple(range(years[0], years[1] + 1)), (lw and bool(volc))


def wb_aerosol_forcing(spec, grid, sigma, carry, time_ns):
    """``aerosol_od`` / ``aerosol_ccn_aod`` / ``aerosol_lw_od`` for one sample:
        self._aerosol_active = (cfg.radiation in ("rrtmg", "rrtmgp")
                                and cfg.aerosol_forcing == "external")
        # Volcanic stratospheric LONGWAVE aerosol (gap #9): active only on a
        # gas-radiation scheme (gray ignores aerosol) with the LW switch and
        # a volcanic file present.  Default OFF ⇒ no LW aerosol path.
        self._aerosol_lw_active = (
            cfg.radiation in ("rrtmg", "rrtmgp")
            and bool(cfg.volcanic_aerosol_lw)
            and bool(cfg.volcanic_aerosol_file)
        )
        self._aerosol_lw_od = None

exec
/usr/bin/bash -lc "nl -ba tests/unit/test_volcanic_sw_optics.py | sed -n '207,245p'; nl -ba packages/tools/legoesm/forcing/external.py | sed -n '2450,2469p'; git diff HEAD --numstat" in /glade/derecho/scratch/pg2328/wt_volc
 succeeded in 81ms:
   207	    np.testing.assert_allclose(g, fd, rtol=1e-5)
   208	
   209	
   210	def test_wrong_band_count_raises():
   211	    solver, _ = _solver()
   212	    kw = _columns()
   213	    with pytest.raises(ValueError, match="trailing shape"):
   214	        solver.solve_columns(**kw, aerosol_volcanic_moments_sw=jnp.zeros(
   215	            kw["T"].shape + (3, 13)))
   216	
   217	
   218	@pytest.mark.parametrize("builder", ["_build_gray_radiation_fn",
   219	                                     "_build_none_radiation_fn"])
   220	def test_non_rrtmgp_radiation_refuses(builder):
   221	    from legoesm.driver import physics_pipeline as pp
   222	    from legoesm.driver.config import ExperimentConfig
   223	    fn, _ = getattr(pp, builder)(ExperimentConfig(radiation="gray"))
   224	    kw = _columns()
   225	    ncol, nlev = kw["T"].shape
   226	    args = (kw["T"], kw["p_full"], kw["p_half"], kw["q_v"],
   227	            kw["sfc_temperature"], jnp.zeros(ncol), jnp.zeros(ncol), 180.0,
   228	            43200.0, jnp.full((ncol,), 0.1), jnp.ones(ncol),
   229	            jnp.zeros((ncol, nlev)), jnp.zeros((ncol, nlev)), jnp.zeros(0))
   230	    with pytest.raises(ValueError, match="volcanic SW"):
   231	        fn(*args, aerosol_sw_volc_col=jnp.zeros((ncol, nlev, 3, 14)))
   232	
   233	
   234	def test_probe_runs_on_synthetic_file(tmp_path, capsys):
   235	    import importlib.util
   236	    spec = importlib.util.spec_from_file_location(
   237	        "probe", _REPO / "scripts/validate/probe_volcanic_sw_optics.py")
   238	    probe = importlib.util.module_from_spec(spec)
   239	    spec.loader.exec_module(probe)
   240	    probe.main(volc_sw_file(tmp_path / "v.nc", omega=np.full(14, 1.0)), 1979, 1)
   241	    out = capsys.readouterr().out
   242	    assert "file-grey" in out and "ssa 1.0000" in out
  2450	                                           lat_grid: jnp.ndarray):
  2451	    """SHORTWAVE volcanic aerosol with its OWN per-band optics from the file.
  2452	
  2453	    Returns ``(moments_col, p_edges, band_wn)``: ``moments_col``
  2454	    ``(ncol, nsrc, 3, nband)`` per-file-layer ``(tau, tau*omega,
  2455	    tau*omega*g)`` (see :func:`_load_volcanic_sw_optics_profile`), ``p_edges``
  2456	    ``(nsrc+1,)`` [Pa] ascending, ``band_wn`` ``(nband, 2)`` [cm-1].  None
  2457	    when volcanic forcing is off or the file carries no ``omega_sun`` /
  2458	    ``g_sun`` / band edges (the caller then keeps the gray path).
  2459	    """
  2460	    if not config.volcanic_enabled or not config.volcanic_path:
  2461	        return None
  2462	    loaded = _load_volcanic_sw_optics_profile(config.volcanic_path)
  2463	    if loaded is None:
  2464	        return None
  2465	    out = _volcanic_profile_at_time(config, day, lat_grid, "sw_optics",
  2466	                                    loaded=loaded[:4] + (loaded[4],))
  2467	    return out[0], out[1], loaded[5]
  2468	
  2469	
1	1	packages/atmosphere/legoesm/atmosphere/physics/clouds/subcolumns.py
18	0	packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
35	0	packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
28	3	packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
17	1	packages/coupler/legoesm/driver/compiled_segments.py
13	1	packages/coupler/legoesm/driver/model_driver.py
18	3	packages/coupler/legoesm/driver/physics_pipeline.py
7	2	packages/ml/legoesm/training/scale_build.py
161	45	packages/tools/legoesm/forcing/external.py
32	19	packages/tools/legoesm/forcing/surface_utils.py
89	0	scripts/validate/probe_volcanic_sw_optics.py
26	22	tests/unit/test_aerosol_band_ccn.py
1	1	tests/unit/test_cross_lane_forcing.py
242	0	tests/unit/test_volcanic_sw_optics.py
1	1	tests/unit/test_wb_amip_deck_params.py

 succeeded in 196740ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/derecho/scratch/pg2328/wt_volc
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 30 items

tests/unit/test_volcanic_sw_optics.py .................                  [ 56%]
tests/unit/test_aerosol_band_ccn.py .............                        [100%]

=============================== warnings summary ===============================
tests/unit/test_volcanic_sw_optics.py: 19 warnings
tests/unit/test_aerosol_band_ccn.py: 6 warnings
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:54: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable(k, "f8", dims)[:] = v

tests/unit/test_volcanic_sw_optics.py::test_legacy_column_file_is_not_an_optics_source
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:100: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", ("time", "lat"))[:] = 0.01

tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
tests/unit/test_aerosol_band_ccn.py::test_volcanic_sw_above_tropopause_and_out_of_ccn
tests/unit/test_aerosol_band_ccn.py::test_ccn_file_flows_from_cli_to_ccn
tests/unit/test_aerosol_band_ccn.py::test_ccn_file_flows_from_cli_to_ccn
tests/unit/test_aerosol_band_ccn.py::test_driver_volcanic_cut_at_state_tropopause
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_aerosol_band_ccn.py:55: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", dims)[:] = np.broadcast_to(band, shape)

tests/unit/test_aerosol_band_ccn.py::test_solar_weighted_and_visible_band_collapse
  /glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_aerosol_band_ccn.py:91: DeprecationWarning: Setting the shape on a NumPy array has been deprecated in NumPy 2.5.
  As an alternative, you can create a new view using np.reshape (with copy=False if needed).
    ds.createVariable("aod", "f8", ("time", "lnwl", "lat"))[:] = np.broadcast_to(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 30 passed, 33 warnings in 177.53s (0:02:57) ==================

codex
1. **MAJOR — Training silently loses volcanic SW forcing.** [model_driver.py:20280](/glade/derecho/scratch/pg2328/wt_volc/packages/coupler/legoesm/driver/model_driver.py:20280) and [scale_build.py:1852](/glade/derecho/scratch/pg2328/wt_volc/packages/ml/legoesm/training/scale_build.py:1852) construct `SegmentForcing` without `aerosol_sw_volc`. With a CMIP6 volcanic file, preprocessing now removes volcanic extinction from `aerosol_od`, so `build_training_segment` and lat-lon ERA5 training receive **no volcanic SW aerosol**. Forward the separately computed field in both constructors.

2. **MAJOR — Integration bypasses the required non-RRTMGP refusal.** [integration.py:1069](/glade/derecho/scratch/pg2328/wt_volc/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1069) returns through `simple_lw` or `gray` before checking `aerosol_sw_volc`. Both branches accepted nonzero moments and produced outputs identical to the absent-field case in a direct check. The refusal tests exercise different, pipeline-specific builders. Add the guard before these returns.

3. **MINOR — Band-selection tests cannot detect a wrong g-point mapping.** [test_volcanic_sw_optics.py:153](/glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:153): solver tests supply identical moments across all 14 bands. Selecting band zero for every g-point would still satisfy them; the loader’s varying-band test never reaches radiation. Add a varying-band or single-band injection test against an independent expected mapping.

4. **MINOR — Float32 caching does not bound interpolation memory.** [external.py:1661](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/external.py:1661), reached from [surface_utils.py:838](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/surface_utils.py:838), gathers two full source profiles and interpolates with float64 weights. At res6, the `(40962,70,3,14)` float64 result alone is **0.90 GiB**, with additional full-sized temporaries before remapping. Chunking this interpolation would bound the daily-update allocation.

**VERDICT: HOLD.** Two MAJOR forcing-path omissions remain. The 30 focused tests passed, including closure and gradient checks, but do not cover those failures.
tokens used
116,183
1. **MAJOR — Training silently loses volcanic SW forcing.** [model_driver.py:20280](/glade/derecho/scratch/pg2328/wt_volc/packages/coupler/legoesm/driver/model_driver.py:20280) and [scale_build.py:1852](/glade/derecho/scratch/pg2328/wt_volc/packages/ml/legoesm/training/scale_build.py:1852) construct `SegmentForcing` without `aerosol_sw_volc`. With a CMIP6 volcanic file, preprocessing now removes volcanic extinction from `aerosol_od`, so `build_training_segment` and lat-lon ERA5 training receive **no volcanic SW aerosol**. Forward the separately computed field in both constructors.

2. **MAJOR — Integration bypasses the required non-RRTMGP refusal.** [integration.py:1069](/glade/derecho/scratch/pg2328/wt_volc/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:1069) returns through `simple_lw` or `gray` before checking `aerosol_sw_volc`. Both branches accepted nonzero moments and produced outputs identical to the absent-field case in a direct check. The refusal tests exercise different, pipeline-specific builders. Add the guard before these returns.

3. **MINOR — Band-selection tests cannot detect a wrong g-point mapping.** [test_volcanic_sw_optics.py:153](/glade/derecho/scratch/pg2328/wt_volc/tests/unit/test_volcanic_sw_optics.py:153): solver tests supply identical moments across all 14 bands. Selecting band zero for every g-point would still satisfy them; the loader’s varying-band test never reaches radiation. Add a varying-band or single-band injection test against an independent expected mapping.

4. **MINOR — Float32 caching does not bound interpolation memory.** [external.py:1661](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/external.py:1661), reached from [surface_utils.py:838](/glade/derecho/scratch/pg2328/wt_volc/packages/tools/legoesm/forcing/surface_utils.py:838), gathers two full source profiles and interpolates with float64 weights. At res6, the `(40962,70,3,14)` float64 result alone is **0.90 GiB**, with additional full-sized temporaries before remapping. Chunking this interpolation would bound the daily-update allocation.

**VERDICT: HOLD.** Two MAJOR forcing-path omissions remain. The 30 focused tests passed, including closure and gradient checks, but do not cover those failures.
