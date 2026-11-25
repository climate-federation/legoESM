Reading additional input from stdin...
OpenAI Codex v0.166.0
--------
workdir: /glade/u/home/pg2328/legoesm
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: medium
reasoning summaries: none
session id: 01a202d0-7e4d-7682-9e57-8fe13d2536a7
--------
user
READ-ONLY adversarial review of CLAIMS (no code edits). Repo: /glade/u/home/pg2328/legoesm (legoESM, differentiable ESM in JAX). Reference sources: /glade/derecho/scratch/pg2328/cesm_src/{CAM (cam_cesm2_1_rel_60), CLM5 (release-clm5.0.34), CICE}.

Context: production AMIP deck config/amip/amip_production.yaml (CAM6 physics on MPAS res6, two-leaf multilayer land, prescribed SST/SIC). 1979 one-year run vs ERA5/CERES: global tas -1.12 K, cold concentrated over NH high-latitude land (Siberia, Canada), Antarctica, Greenland, Tibet; TOA albedo +0.10..+0.15 over Siberia/Canada and Arctic Ocean; evaporation -0.24 mm/d, precip -0.43 mm/d; cloud cover +40% over Antarctica (also in CMIP6 MMM).

CLAIMS to refute or confirm (each: CONFIRMED / REFUTED / PLAUSIBLE, with file:line evidence; say which parts you verified in code). Proposed actions follow each; challenge them too.

C1 (READ-OFF-CODE + offline probe scripts/tmp/probe_snow_albedo_canopy.py): with land_canopy_snow_masking unset (default False, packages/coupler/legoesm/driver/config.py:1424) every snow-covered land column, including boreal evergreen forest, exports the snowy-ground albedo (~0.74-0.79 broadband, multilayer_land.py:913-930), vs CLM5 canopy-over-snow ~0.10-0.30. This is the main driver of the boreal cold bias. Proposed: turn masking on in production.
C2 (probe): masking ON gives evergreen spruce 0.14, darker than CLM5 with intercepted canopy snow (fcansno 0.5 -> 0.16, 1.0 -> 0.30), because canopy_masked_snow_albedo fixes fcansno=0 (canopy/radiative_transfer.py:630-650). Proposed: port CLM5 canopy snow (snocan -> fcansno, CanopyHydrologyMod/SurfaceAlbedoMod) so masking is faithful. Check: does our land carry a canopy snow store at all? What does CLM5 0.34 actually compute for fcansno?
C3 (probe): production land_snow_tau_days 150 + snow_age_activation_K 5000 keeps snow at 0.735 after 90 cold days + 10 melting days; calibration's 3.674 d gives 0.51. Spring snow too bright -> later melt -> cold. Proposed: user decision on the value (no code).
C4 (READ-OFF-CODE): MPAS lane radiation (integration.py ~641-650, ~2294) passes only q_c, q_i to RRTMG; CAM6 adds MG2 snow to SW+LW optics with cldfsnow and cldfprime=max(cld,cldfsnow) (CAM rrtmg/radiation.F90 ~888-986, micro_mg_cam.F90 ~2309). Missing snow -> less downwelling LW at the surface in polar winter -> cold. Proposed: port radiatively active snow.
C5 (READ-OFF-CODE): land tas diagnostic (diagnostics.py ~875-888) uses ocean drag law and treats lowest model level (~60 m on L32) as 10 m; biased WARM in stable conditions, so it hides, not causes, cold bias. Proposed: fix the diagnostic to CLM-style 2 m from land roughness at true level height.
C6: sea-ice albedo flat 0.80 (albedo_ice) vs CICE prescribed-ice -> Arctic albedo excess. Proposed: lower priority.

Also: anything bigger we missed that would cool land by >1 K annually? Rank C1-C6 by expected impact on annual land tas. Under 500 words.
codex
I’ll check the production configuration, trace each claim through the active code paths, and compare the relevant CAM/CLM/CICE source. I’ll keep the review read-only and separate verified behavior from unproven climate attribution.
exec
/usr/bin/bash -lc 'cat CLAUDE.md' in /glade/u/home/pg2328/legoesm
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
/usr/bin/bash -lc 'git status --short; git worktree list' in /glade/u/home/pg2328/legoesm
 succeeded in 1158ms:
?? .claude/.headroom_wrap_settings.lock
?? c1.pbs
?? config/amip/amip_canopy_snow_mask_ab.yaml
?? data/les_cases/AYOTTE/
?? data/les_cases/GABLS3/
?? data/les_cases/GABLS4/
?? data/les_cases/PROVENANCE_2026-10-15.md
?? docs/datasets.json
?? docs/wb/reviews/2026-11-22_cold_bias_claims_codex.md
?? docs/wb/reviews/2026-11-22_cold_bias_claims_prompt.md
?? scripts/cluster/les_suite/derecho_les_case.pbs
?? scripts/cluster/les_suite/les_timing.o7827219
/glade/u/home/pg2328/legoesm                                                                                                                        8750d0005d [perf/amip-cam6-derecho-2026-09-26]
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
/glade/derecho/scratch/pg2328/wt_ab_premerge                                                                                                        576824fa0e (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_treat                                                                                                           ed2c8ead3a (detached HEAD)
/glade/derecho/scratch/pg2328/wt_latlon_tracer                                                                                                      5117047230 [fix/latlon-tracer-vertical-conservation]
/glade/derecho/scratch/pg2328/wt_probe_efa650f75e                                                                                                   efa650f75e (detached HEAD)
/glade/derecho/scratch/pg2328/wt_probe_fc6a65ca38                                                                                                   fc6a65ca38 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/ab_base                                                                                              6d4bc64109 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a1d4c1231007d5d0b                                                                              8232c3c49d [les/campaign-stage1]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a318fc5ce2a972ffc                                                                              8d09705873 [crm-mpi-port]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a4cf45c870727d40b                                                                              da10384494 [les/cosci-harness]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a62b01bad95c93693                                                                              a8f5ac7d2f [les/gabls3]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a874b7fc139d79083                                                                              8b49890830 [worktree-agent-a874b7fc139d79083]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-abcd9a518a3f50e9a                                                                              27b522e5dc [worktree-agent-abcd9a518a3f50e9a]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ad0003e1ed1298b1e                                                                              3d3537422b [les/campaign-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ae58ae57a5e97ee45                                                                              12367e6ef2 [diag/crm-les-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-afbef113e02d249fe                                                                              75bf36b1a7 [les/campaign-stage3b]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-fp32-tracers                                                                                     be5b35cdfa [perf/crm-fp32-tracers]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-les-scaling                                                                                      43d9f3a14d [perf/crm-les-mpi-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-mpi-gaps                                                                                         f3068db97e [fix/crm-mpi-gaps]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-rad-blocking                                                                                     8669277fd4 [perf/crm-rad-blocking]
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_base                                                                                              7759a33780 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_fuse                                                                                              b70807a971 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2                                                                                             6d9e51da13 [perf/les-plane-fft]
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
/usr/bin/bash -lc "sed -n '300,780p' CLAUDE.md" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
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

exec
/usr/bin/bash -lc "rg --files | rg '(multilayer_land.py|radiative_transfer.py|integration.py|diagnostics.py|probe_snow_albedo_canopy.py)"'$'"'; cat config/amip/amip_production.yaml" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
evaluations/headline_diagnostics.py
scripts/run/global_overturning/plot_5deg_rerun_diagnostics.py
scripts/run/global_overturning/plot_idealized_diagnostics.py
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py
scripts/run/s2s/plot_coupling_diagnostics.py
packages/coupler/legoesm/driver/diagnostics.py
packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py
packages/land/legoesm/land/carbon_diagnostics.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py
packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/config/radiative_transfer.py
scripts/matrix/ocean_test_matrix/energy_diagnostics.py
packages/land/legoesm/land/canopy/radiative_transfer.py
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
packages/land/legoesm/land/multilayer_land.py
packages/ocean/legoesm/ocean/diagnostics.py
packages/core/legoesm/parallel/scaling_diagnostics.py
packages/core/legoesm/timestepping/integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/fesom_integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/integration.py
packages/ocean/legoesm/ocean/physics/convection/integration.py
packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py
packages/ocean/legoesm/ocean/physics/lateral_mixing/integration.py
tests/run/test_correction_e2e_integration.py
tests/run/test_cmip_compare_integration.py
tests/run/test_offline_amip_forcing_integration.py
scripts/plot/plot_les_diagnostics.py
scripts/plot/plot_amip_cmor_diagnostics.py
scripts/validate/fv3_native/ic_parity_diagnostics.py
tests/ocean/unit/test_eddy_diagnostics.py
tests/unit/test_ifs_cloudsc_integration.py
tests/unit/test_run_omip_sharded_diagnostics.py
tests/land/unit/test_carbon_diagnostics.py
tests/land/unit/test_multilayer_land.py
scripts/validate/ocean_fidelity/mle_psi_diagnostics.py
tests/land/test_train_multilayer_land.py
tests/unit/test_ryf_pipeline_integration.py
tests/land/integration/test_canopy_sif_integration.py
tests/ocean/unit/test_ocean_diagnostics.py
tests/ocean/unit/test_tke_integration.py
tests/unit/test_plot_amip_cmor_diagnostics.py
tests/ocean/unit/test_tracer_time_integration.py
tests/ocean/unit/test_eke_3d_integration.py
tests/ocean/unit/test_vmix_mpas_integration.py
tests/ocean/unit/test_longterm_diagnostics.py
tests/ocean/fidelity/test_mitgcm_integration.py
tests/unit/test_rce_diagnostics.py
tests/ocean/unit/test_catke_integration.py
tests/unit/test_ensemble_diagnostics.py
tests/unit/test_land_ml_checkpoint_integration.py
tests/evaluations/test_headline_diagnostics.py
# =============================================================================
# legoESM AMIP — AUTHORITATIVE production configuration (single source of truth)
# =============================================================================
# Run with:
#   run_amip.py --config config/amip/amip_production.yaml \
#       <machine path flags> --days N --output DIR
#
# Keys are run_amip argument DESTS (set as defaults via --config); any explicit
# CLI flag still overrides a key here.  Machine-specific PATHS (ERA5 IC,
# SST/SIC, solar/ozone/GHG/aerosol/volcanic, topography) are deliberately NOT
# pinned here — they live in config/amip/amip_production.sh (env-overridable)
# so this physics config stays machine-independent and reproducible across
# deployments.
#
# THE ATMOSPHERIC PHYSICS SUITE IS CAM6 (user decision 2026-09-23).  The lane
# (MPAS res6 Voronoi, prescribed SST, two-leaf multilayer land, external
# forcings) is unchanged from the previous production deck; the atmospheric
# parameterizations, the vertical grid and the physics cadence are CAM6
# (CESM2.1, cam_cesm2_1_rel_60, f09 namelist) wherever a faithful port exists.
# The outgoing configuration is kept RUNNABLE and unchanged at
# config/amip/amip_sundqvist_l36.yaml, so every earlier result reproduces; the
# decks that used to `include: amip_production.yaml` now include that file,
# because they document results measured against it.
#
# THIS DEFAULT IS ADOPTED ON FAITHFULNESS TO CAM6, NOT ON DEMONSTRATED SKILL.
# The 60-day CAM6 arm is NOT finished and has NOT yet been scored against
# observations.  Over its first twenty-five days it sits about 25 W/m2 out of
# top-of-atmosphere energy balance, against about 2 W/m2 for the Sundqvist/L36
# arm, because its clouds reflect far too little sunlight.  The cause is known
# and is NOT yet fixed: cloud water is being rimed onto falling ice.  The
# supporting measurement, on the two arms' day-5 states: 71% of the liquid
# deficit lies below freezing, 70% of that cold deficit is accounted for by
# the snow + graupel increase, and graupel more than doubles column-wide.
#
# EVIDENCE AND PAIR RECORD live in the RUN TREE, which is deliberately NOT in
# this repository (runtime outputs are gitignored):
#   /work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/_cam6/PREREGISTRATION.md
# on the Levante deployment.  It records the pair (cam6_base60 = the
# Sundqvist/L36 deck, cam6_suite60 = this one; 60-day cold starts, scored
# days 31-60 vs ERA5), the resolved-config diff, and every measurement cited
# above.  A reader without that tree should treat the numbers here as
# unverifiable rather than as repository facts.
#
# NOT YET CAM6 (run 2+): MG2 microphysics (Morrison mg flavour stands in),
# ridge-based orographic drag (needs the ridge-statistics dataset), MAM4
# aerosols (licence decision), CLUBB tuning table.
#
# EVERY scientific choice is spelled out below even where it equals a code
# default: a default is not a record.  In particular the two numerical levers
# (mpas_div_damp4_scale, mpas_vert_advection_scheme: sb) keep their OFF code
# defaults and are selected HERE.
# ---- grid / dycore ----------------------------------------------------------
grid_type: voronoi
resolution: 6               # 40962 cells (~1.1 deg) ~ CAM6 f09 (0.9x1.25)
nlev: 32                    # CAM6: the L32 hybrid table (cam_vcoords_L32_c180105)
vertical_coord: cam_l32     # CAM6: hyai/hybi verbatim, p_top 2.255 hPa
discretization: mpas
dt: 112.5
precision: fp64
corner_fill: avg                     # cube-vertex halo fill; inert on MPAS,
                                     # pinned so the run records it
a_h_scale: 0.375
mpas_sponge_del2_top_layers: 2       # index-based sponge: 2 layers on L32 span
                                     # 2.3-10.2 hPa (the Sundqvist/L36 deck: 3 layers on
                                     # sigma-36 = 2.0-7.6 hPa).  CAM6 f09 FV has
                                     # NO top sponge (div24del2flag=4 = uniform
                                     # del4 divergence damping; the tanh top
                                     # profile lives only in the ldel2/ldiv2
                                     # branches) -- 2 layers is the closest
                                     # offered setting (user 2026-09-21)
mpas_sponge_del2_top_factor: 8.0
mpas_vert_advection_scheme: sb       # conservative Simmons-Burridge flux form
                                     # of the hybrid theta transport (van Leer
                                     # is sigma-only on this lane).  The
                                     # advective upwind form violates the
                                     # discrete product rule: measured column
                                     # residual 3.8e-2 of the interfacial
                                     # exchange on this L32 table against
                                     # 6.7e-16 conservative.  USER-APPROVED
                                     # 2026-09-23; the CODE default stays
                                     # 'upwind' and this deck selects it.
mpas_vert4_flux_form_tracers: []     # vapour takes the log form below instead
mpas_vert4_log_form_tracers: [q_v]   # vertical del4 filter on ln(q_v + 1e-7), water-weighted
                                     # mass-conserving flux.  The flux form [q_v] removed 8-10 %/day
                                     # of UTLS water offline; one 60-day pair vs it (2026-10-14,
                                     # PLAUSIBLE): Feb ClimateEval median RMSE ratio 0.97, 200 hPa
                                     # T RMSE -26..-32 %.  USER-APPROVED 2026-10-15; the CODE
                                     # default moved to this too (same day).
mpas_div_damp4_scale: 1.0            # CAM6: fv_div24del2flag=4 selects ldiv4,
                                     # 4th-order DIVERGENCE damping at every
                                     # level (cd_core.F90 tau4 = 0.01/dt times
                                     # the squared cell area).  1.0 = CAM's
                                     # nondimensional RATE, not CAM's behaviour
                                     # (CAM applies it per acoustic substep,
                                     # per cell); 7.235e15 m^4/s here.  RATE
                                     # PENDING USER APPROVAL.  Our del2/del4
                                     # are VECTOR Laplacians and damp balanced
                                     # flow too; this is the curl-free-only
                                     # term CAM relies on, and the L32 table's
                                     # 14 pure-pressure top layers have no way
                                     # to absorb divergence into their own mass.
                                     # MEASURED, physics off, from the day-5
                                     # state: alone it cuts the 24 hPa
                                     # hot-patch drift from 3.14 to 1.96 K/h,
                                     # a 38% reduction -- real but NOT
                                     # sufficient.  WITH the conservative
                                     # transport above the drift REVERSES to
                                     # -1.10 K/h, which is why this deck
                                     # carries both or neither.
energy_fixer: true                   # CAM6 check_energy_fix analogue (review
                                     # 2026-10-10 F15, by vote): the energy each
                                     # dynamics step loses (sponge/del4/div4 KE,
                                     # numerics) returns as a uniform heating
                                     # before physics, and the energy the dry-
                                     # mass fixer moves (outside that window)
                                     # returns too (2026-10-14 F15 V8: 0.065
                                     # W/m2 at res6).  Named, not inherited
                                     # (the code default is also true).

# ---- physics cadence ---------------------------------------------------------
physics_update_steps: 16    # CAM6: physics every 1800 s (16 x 112.5 s); the
                            # held-physics steps re-apply the last tendencies
rad_update_steps: 32        # CAM6 f09: radiation hourly (iradsw = -1)
cld_macmic_num_steps: 3     # CAM6 namelist_defaults_cam.xml:1254 (mg2 + clubb_sgs,
                            # dtime 1800): CLUBB + microphysics sub-cycled 3 x
                            # at 600 s inside each physics step (tphysbc macmic)

# ---- radiation / clouds / microphysics --------------------------------------
radiation: rrtmg            # rrtmgp builder (CAM6 uses RRTMG; same k-lineage)
rrtmgp_overhead_layer: true # CAM RRTMG extra layer above the model top (ozone
                            # column above p_top absorbs; ~2 W/m2 leaves budget)
# Saturation curve the cloud-cover diagnosis measures RH against.  NAMED, not
# inherited: the code default is "mixed_phase" and that is the value this deck
# wants, but the default moved on 2026-09-17 and a default is not a record.
# The preserved pre-CAM6 deck pins "liquid" for the same reason.
cloud_saturation_scheme: mixed_phase
clouds: cam6_clubb          # CAM6: liquid = CLUBB PDF fraction, ice = CAM aist,
                            # deep-convective = clubb_intr deepcu from the
                            # scheme's updraft mass flux + in-cloud condensate
use_clubb_cloud_fraction: true   # routes the CLUBB cloud-fraction carry to
                                 # radiation (required by cam6_clubb)
cloud_cam6_rhmini: 0.80     # CAM6 aist ramp (cldfrc2m.F90:846-856), namelist
cloud_cam6_rhmaxi: 1.0      # cldfrc2m_rhmini/rhmaxi/rhminis/rhmaxis under
cloud_cam6_rhminis: 1.0     # clubb_sgs (namelist_defaults_cam.xml:1521-1533);
cloud_cam6_rhmaxis: 1.0     # = the code defaults, now recorded (user 2026-09-30)
microphysics: morrison      # mg flavour (MG2 port = run 2)
morrison_flavor: mg         # NAMED, equals the code default.  Resolves the
                            # MG rows (resolve_morrison_flavor), incl. since
                            # 2026-10-19 (oracle audit, user decision) CAM6
                            # MG2 rain: slope bounds 1/[500, 20] um (was
                            # 1/[2800, 20] um) and Beheng self-collection
                            # -8*nric*qric*rho on in-precip rain (was SAM
                            # SB2001 with breakup).
morrison_do_graupel: true   # NAMED, equals the code default: this deck's
                            # measured behaviour.  CAM6's MG2 has NO graupel
                            # (eight constituents, micro_mg_cam.F90:139); our
                            # port rimes cloud water onto graupel, ~half the
                            # cold-sector cloud-water sink on the day-5 state.
                            # The graupel-off arm is this deck plus
                            # --no-morrison-do-graupel; this value stays
                            # until the user decides (2026-09-24).
morrison_warm_rain_scheme: kk2000_cam6  # CAM6 MG2 law (user decision
                                   # 2026-10-19, oracle audit; was kk2000, the
                                   # SAM/E3SM 1350*qc^2.47*Nc^-1.79 form, 0.003-
                                   # 13x MG2): 13.5*var_coef(relvar,2.47)*
                                   # qc^2.47*Nc^-1.1 (micro_mg_utils.F90), with
                                   # per-cell relvar = rcm^2/min(1, rcp2) from
                                   # CLUBB (clubb_intr.F90:2416-2425).  Named
                                   # explicitly; the code default (unset)
                                   # resolves to the same law under mg.
morrison_autocon_fact: 1.0         # kk2000 autoconversion multiplier (1 = unscaled)
morrison_accre_enhan_fact: 1.0     # kk2000 accretion multiplier (MG2 accre_enhan)
morrison_warm_rain_incloud: true   # CAM6 MG2 in-cloud warm rain (user decision
                                   # 2026-09-30): rates on qc/lcldm, qr/lcldm,
                                   # lcldm = max(ast, 1e-4), ast = max(CLUBB
                                   # liquid, aist ice-stratus) as CAM6
                                   # (clubb_intr.F90:2575, micro_mg_cam.F90:
                                   # 1810), times lcldm (micro_mg2_0.F90:878,
                                   # :1224-1322).  Grid-mean arm:
                                   # --no-morrison-warm-rain-incloud.
                                   # Since 2026-10-19 (oracle audit, user
                                   # decision) it is the whole CAM6 MG2 rain
                                   # mode: in_cloud precip_frac, rain
                                   # evaporation only in the precipitating
                                   # clear air, rain sedimenting the post-
                                   # source state, two-sided number re-adjust.
                                   # Since 2026-10-19 also CAM6 MG2 TWO-MOMENT
                                   # SNOW (user decision): the run carries the
                                   # snow number N_s (tracer slot [9]); MG2
                                   # riming, Bergeron onto snow, sublimation,
                                   # self-aggregation, melt above +2 C, fall
                                   # speeds, snow sedimenting the post-source
                                   # state.
morrison_ice_snow_d_auto: 500.0e-6 # CAM6 MG2 micro_mg_dcs = 500 um ice->snow
                                   # size (namelist_defaults_cam.xml:1244;
                                   # user decision 2026-10-19, was 250 um).
morrison_sed_cfl_substeps: true  # MG2 micro_mg2_0.F90 sedimentation loop:
                                 # nstep = 1 + int(max V dt/dz) per column,
                                 # so precipitation crosses >1 layer per 600 s.
                                 # Rain and snow sediment the POST-source state
                                 # as MG2 (:2204, via warm_rain_incloud); ice/
                                 # graupel still sediment PRE-source pools.
morrison_sed_cfl_substeps_strict: false # the abort path is an equinox error_if,
                                        # i.e. a host callback, and the GPU lane
                                        # runs JAX_PLATFORMS=cuda with no CPU
                                        # device to place it on (this killed the
                                        # first 60-day arm at day 2).  The
                                        # required sub-step count is still
                                        # reported as a diagnostic.
                                 # (L32 at 600 s needs ~90 for graupel; a clamp
                                 # would silently slow the fall)
hard_saturation_adjustment: false  # CAM6 clubb_do_liqsupersat = .false.
                                   # (namelist_defaults_cam.xml:1225): no
                                   # supersaturation removal after the physics
                                   # step.  On this lane the flag is the
                                   # POST-STEP drain (model_driver
                                   # _mpas_hard_saturation_poststep), which is
                                   # our counterpart of CAM's post-CLUBB
                                   # "iceadj" block (clubb_intr.F90:2258).
                                   # Was true; user table 2026-09-24, row 2.
cloud_partial_coverage_optics: none
cloud_optics_inhomogeneity: constant  # CAM6: RRTMG applies NO in-cloud
                                       # inhomogeneity factor (none anywhere
                                       # in src/physics/rrtmg/).  "constant"
                                       # with factor 1.0 is the no-thinning
                                       # path (chi = 1, byte-identical).  Was
                                       # two_region: on day-15 states it
                                       # removed 52% of the liquid in run 1
                                       # and 42% in run 3, so the water
                                       # reaching radiation stayed ~14.5 g/m2
                                       # across the liquid hand-off.  User
                                       # decision 2026-09-25, for run 4.
cloud_inhomogeneity_factor: 1.0        # the "constant" scheme's scalar: 1.0
                                       # = homogeneous, no reduction.  Set
                                       # explicitly so the no-thinning row
                                       # rests on no default.
cloud_fsd: 1.0                         # INERT under "constant" (consumed by
                                       # two_region only); kept explicit so
                                       # the old value is on record.
cloud_vertical_overlap_optics: mcica  # CAM6 McICA (user 2026-09-25): each
                                     # g-point sees its own max-random cloud
                                     # subcolumn in ONE solve (was max_random:
                                     # 8 full solves, ~8.5x the radiation cost).
                                     # Synthetic 512-column check vs a 64-subcolumn
                                     # reference: bias within sampling noise,
                                     # per-column scatter ~1.8x the 8-subcolumn one.
cloud_n_subcolumns: 8                # read only by max_random; inert under mcica
rrtmgp_gpoint_batch_size: 0          # CPU: one g-point at a time (code default 16 is
                                     # for GPU). Measured 2026-10-06, res6 16 ranks x 8
                                     # cores: radiation step 13.8 s at 16, 7.9 s at 0
                                     # (~139 s per simulated day); fluxes differ only by
                                     # summation order (<= 6.1e-4 W/m2).
rrtmgp_use_scan: true                # scan the RRTMGP column recurrence (code default:
                                     # unrolled on CPU, scan on GPU, so GPU is unchanged).
                                     # Measured 2026-10-29, this deck, MPAS res6 CPU: step
                                     # -4% at 1 node, -7% at 8; round-off only (fp32 fluxes
                                     # <= 1.1e-7 of max after one step), dry mass unchanged,
                                     # 127-step divergence (0.66 K) matched by a 1e-14 T
                                     # perturbation control (0.27 K).  User 2026-10-30.
hard_sat_ice_curve: false          # the ice-curve blend of that same
                                   # post-step drain; inert once the drain
                                   # is off, set explicitly so no row rests
                                   # on a default.  User table row 3.
homogeneous_ice_nucleation: true
aerosol_ccn: true                  # droplet number diagnosed from the prescribed
                                   # aerosol (user decision 2026-11-01, #2216,
                                   # reverses 2026-10-24): Jan 1979 30-day arm,
                                   # global rsut bias vs CERES-EBAF -7.99 W/m2
                                   # (prognostic) -> +0.78 (this setting).
morrison_prognostic_droplet_number: false  # CAM6 MG2 prognostic droplet number
                                   # OFF until its droplet budget closes (#2216);
                                   # the prognostic arm is
                                   # amip_production_prognostic_nc.yaml.
                                   # cam6_aerosol_dir below is read only by it.
morrison_mg2_drop_early_reset_log_width: 0.01  # NAMED, equals the code default:
                                   # RULE 5 width of MG2's early droplet size
                                   # reset (autoconversion + snow riming on
                                   # this deck's specified droplet number; 0 =
                                   # MG2's hard clip; user 2026-11-01, #2216)
morrison_mg2_drop_sed_reset_log_width: 0.01  # NAMED, equals the code default:
                                   # RULE 5 width of MG2's droplet size reset in
                                   # the sedimentation fall speeds (read only by
                                   # the prognostic droplet number, so inert on
                                   # this deck; 0 = MG2's hard clip; #2216)
cam6_aerosol_dir: /work/bb1596/b309178/diffESM/legoesm_ap/data/cam6_optics
energy_consistent_moisture_clip: true

# ---- convection ---------------------------------------------------------------
zm_land_fraction: required  # ZM reads the land mask (c0_lnd vs c0_ocn); "none" = aquaplanet only
convection: zhang_mcfarlane # CAM6: faithful zm_conv port (f09 mg2+clubb namelist:
                            # c0_lnd 0.0075, c0_ocn 0.03, ke 5e-6 (dyn=fv),
                            # tau 3600,
                            # capelmt 70, tiedke_add 0.5, momcu = momcd = 0.7,
                            # num_cin 1); scheme defaults carry those values.
                            # ZM convective momentum transport (momtran) is ON
                            # and reaches the MPAS winds (mpas_cmt resolves True;
                            # the Sundqvist/L36 deck's Bechtold runs CMT
                            # off, lane default)
convective_rain_to_surface: true
convective_cloud: false     # refused by cam6_clubb (deepcu is CAM6's version)
# Bechtold/IFS settings written explicitly (inert here: convection is ZM).
bechtold_use_ifs_ascent: false         # faithful OpenIFS chain off
bechtold_ifs_shallow_nosupply: zero    # no sub-cloud supply -> shallow flux ramps to ~0 (OpenIFS: first_guess)
# ZM tunables written explicitly (values = the scheme defaults = the CAM6 f09
# namelist above), so a campaign arm's change is one visible row.
zm_c0_lnd: 0.0075           # rain conversion over land [1/m]
zm_c0_ocn: 0.03             # rain conversion over ocean [1/m]
zm_ke: 5.0e-6               # rain evaporation efficiency
zm_dmpdz: -1.0e-3           # dilute-plume entrainment [1/m]
zm_tau: 3600.0              # CAPE consumption timescale [s]
zm_capelmt: 70.0            # deep-convection CAPE threshold [J/kg]
# ZM detrainment smoothing widths (RULE 5; values = the scheme defaults; 0 =
# CAM's hard switch), read by the CLUBB macmic sub-step's detrainment booking.
zm_detrain_phase_width_K: 0.1              # liquid/ice ramp corners [K]
zm_detrain_number_width_kgkg_s: 1.0e-12    # max(0, dlf) in the number source [kg/kg/s]
zm_transport_cloud_tracers: true  # ZM convtran of carried cloud/precip water + numbers (CAM6)

# ---- turbulence ---------------------------------------------------------------
turbulence: clubb
clubb_prognostic: true
clubb_liquid_partition: true       # CAM6 clubb_intr: the closure's diagnosed
                                   # cloud liquid is handed to the host as a
                                   # tendency, and the Morrison saturation
                                   # adjustment is switched off, both
                                   # condensation and (since 2026-10-19, user
                                   # decision) cloud-water evaporation (MG2
                                   # allow_sed_supersat = .false. under CLUBB,
                                   # micro_mg_cam.F90:668-672).  One flag,
                                   # both halves.  User table row 1.
# The partition REFUSES an active radiative condensate floor (it would be
# imposed on top of the liquid the closure now supplies, making clouds opaque
# twice).  Both floors default to a non-zero scheme value when unset, so they
# are pinned to 0 here.  Not in the user's three-row table: REQUIRED BY ROW 1.
cloud_q_c_diagnostic: 0.0
conv_cloud_condensate: 0.0
# CLUBB upper domain limit = CAM6's trop_cloud_top_press 1.D2 (100 Pa;
# cam_cesm2_1_rel_60 namelist_defaults_cam.xml:1295).  User decision
# 2026-10-20 ("use cam6 rule"), was 15000 Pa.  Offline 1/3-day A/B on a
# day-105 production checkpoint: stable, sparse stratospheric CLUBB cloud
# (scripts/validate/oracle/clubb_glue/top_press_scm_ab.py).
clubb_trop_cloud_top_press: 100.0
# CLUBB tunables written explicitly (values = the port's CLUBBParams defaults,
# the CAM6 clubb namelist).  clubb_c_K10h (0.3) is read by the eddy-scalar
# rtm/thlm diffusion (CAM l_do_expldiff_rtm_thlm) but has no deck key, so the
# CLUBBParams default applies.
clubb_c14: 2.2
clubb_c8: 4.2
clubb_c11: 0.7
clubb_c11b: 0.35
clubb_gamma_coef: 0.308
clubb_gamma_coefb: 0.32
clubb_beta: 2.4
clubb_c_k10: 0.5

# ---- gravity-wave drag ---------------------------------------------------------
# CAM6 f09 L32 (non-WACCM, nlev < 60) runs OROGRAPHIC drag ONLY: build-namelist
# sets use_gw_front=false and use_gw_convect_dp=false (bld/build-namelist ~3230;
# the f09 frontal/Beres coefficient rows apply to WACCM / >=60-level builds).
# So no non-orographic member here (the Sundqvist/L36 deck carries the
# e3sm_cam
# background source; that row is dropped on purpose).
gravity_wave_drag: mcfarlane            # orographic member = CAM6 gw_oro port
                                        # (cam_cesm2_1_rel_60, oracle-matched;
                                        # user decision 2026-10-19): effgw_oro
                                        # 0.125 x landfrac, sgh/sqrt(landfrac),
                                        # interface N with n2min 5e-5, no launch
                                        # clip, tndmax 400 m/s/day, limiter
                                        # remainder carried down, gw eddy
                                        # diffusion of dse + tracers, critical-
                                        # level offset 0 (CAM c = 0; follow-up
                                        # 2026-10-20, was 0.5 m/s).  These are
                                        # the McFarlaneConfig code defaults (no
                                        # deck rows exist for them, except
                                        # effgw_oro below); the fv3duo deck
                                        # inherits this row.
mcfarlane_effgw_oro: 0.125              # CAM6 effgw_oro (namelist_defaults_cam
                                        # non-WACCM L32); explicit so the run
                                        # records it (#1712).
hines_launch_p: 70000.0                 # inert (hines not selected); kept equal
                                        # to the Sundqvist/L36 deck.  The e3sm_cam_*
                                        # rows
                                        # also resolve (to run_amip defaults) but
                                        # are dead while e3sm_cam is not in the
                                        # selector; re-pin them from the CAM6
                                        # namelist before any run that adds it.

# ---- surface ------------------------------------------------------------------
surface_bulk_scheme: large_yeager_cesm  # CAM6/CESM shr_flux_atmOcn (LY04)
# (surface_gustiness_zi: 300 removed, user 2026-10-19: inert on this scheme --
# CESM2.1 runs gust_fac = 0, and the large_yeager_cesm law never reads it.)
surface_tiled: false
surface_z0_land: 0.1
mpas_land_lapse_K_per_km: 6.5
mpas_land_beta: 0.6
mpas_ice_skin_prognostic: true
albedo_ice: 0.80
# Melt-dependent sea-ice albedo on the prognostic skin (#2106, user decision
# 2026-10-24 after the July 1979 A/B: Arctic clear-sky reflected SW -19.2
# W/m2).  Named explicitly although "auto" resolves to it here.
sea_ice_albedo_scheme: maykut_untersteiner
sea_ice_melt_ramp_width_K: 1.0
ice_skin_melt_cap_width_K: 0.01
dynamic_albedo: true
snow_albedo_feedback: true
land_snow_tau_days: 150.0
snow_age_activation_K: 5000.0

# ---- land (unchanged) -----------------------------------------------------------
use_multilayer_land: true
land_update_seconds: 1800.0  # CAM6: CLM steps once per 1800 s coupling
                             # interval (user 2026-09-25).  = 16 x 112.5 s,
                             # forced with the interval-mean atmosphere; the
                             # offline LMIP driver also runs the tile at 1800 s.
                             # Was 300 (-> 337.5 s actual, every 3 steps).
multilayer_n_layers: 10
multilayer_soil_depth: 3.0
# Soil freeze/thaw ON (user 2026-09-26), as in CLM5 and as the land IC spin-up
# ran; the land calibration's tables were fitted with it off.
land_soil_freeze_thaw: true
# Snowpack: explicit bulk (one SWE reservoir), as every prior production run.
# Moving production to the layered pack (bulk -> layered) is an OPEN user decision.
land_snow_scheme: bulk
# Frozen soil impedes water flow, CLM5 form 10**(-e * ice fraction), e = 6
# (user 2026-09-28).  Active because freeze/thaw is on.
land_soil_ice_impedance_exponent: 6.0
# Soil-water numerics (user 2026-10-14): modified van Genuchten air-entry head [m].
land_soil_vg_air_entry_m: -0.02
# One-layer snow thermal node OFF (record only; A/B switch added 2026-10-03,
# user decision: off until the Sep-Dec arm verdict).
land_snow_insulation: false
land_soil_moisture_init_frac: 0.25
land_stomatal_beta: false
land_calibrated_physics: true
land_surface_scheme: two_leaf
land_canopy_max_iters: 10   # NEW BASELINE 2026-10-05 (user): A/B vs 60 = 4.2x/2.7x faster, land LH -1.5/-3.4 %; see fv3_duo_gaps/q_review/deck_ledgers.md
# Pinned at the current values (b0 stressed, no interception) so the deck
# records them; changing either is a labelled land baseline, not a default flip.
land_canopy_stress_b0: true
land_canopy_interception: false
# Two-leaf canopy solver smoothing widths, named so every run records them
# (user 2026-09-30): canopy-air RH <= 1 cap and stable zeta <= 0.5 cap.
land_canopy_rh_cap_smoothing_width: 0.01
land_canopy_zeta_cap_smoothing_width: 0.05
# Above-canopy Monin-Obukhov fixed-point iterations: 5 -> 10 (user 2026-09-30;
# 5 left unstable columns up to 11% off in aerodynamic resistance).
land_canopy_most_n_iters: 10
mpas_land_beta_soil: true
# Surface drag over land from the land model's own roughness / canopy drag, not
# the ocean bulk law (user 2026-10-03: "a major default"; Dec 2001 45-70N land:
# the boundary layer received 0.04 Pa vs land model 0.23 vs ERA5 0.26).
mpas_land_stress_from_land: true
# Coastal cells: the non-land fraction's bulk fluxes on the ocean/ice surface,
# not on the land-blended T_sfc/q_sfc (#1320; CESM computes atm-ocn fluxes on
# ocean points only).  User 2026-10-19.
ocean_flux_on_ocean_surface: true
mpas_land_params_refresh: true  # seasonal LAI / canopy height / wet-soil
                                # albedo rebuilt every land step, as the
                                # offline calibration runs the tables (user
                                # 2026-09-29; before, January LAI all year)
land_soil_init: aridity
# Land starting state: the res-6 spun-up soil water/snow (soil_ic_mpas6.npz) with
# soil TEMPERATURE replaced by ERA5 1979-01-01 00 UTC stl1-4 on non-glacier land
# (glaciers keep the spin-up). Built by scripts/data/regrid_land_ic.py
# --era5-soil-t; provenance is in the file's metadata. data/ is gitignored:
# stage the file in the launch directory.
land_ic: data/lmip_soil_ic/soil_ic_mpas6_era5T_19790101.npz

# ---- moisture smoothing / clamps (unchanged) -----------------------------------
mpas_qv_smooth_del2_m2s: 0.0
mpas_qv_smooth_del4_m4s: 3.6421e14
mpas_conservative_tracer_clamp: true

# ---- forcing / IC / output (unchanged) ------------------------------------------
diurnal_cycle: true
orbital_insolation: true
ic: era5
dataset: custom
sst_var: tosbcs
sst_offset: 0.0
sic_var: siconcbcs
sic_scale: 0.01
solar_source: spectral_file
solar_tsi_var: TSI
solar_spectral_var: SSI_frac
solar_spectral_band_order: rrtmg_sw   # explicit: SSI_frac is in RRTMG-SW band order (adapt_cmip6_solar.py); 'auto' only inferred it
ozone_forcing: external
ghg_forcing: external
aerosol_forcing: external
aerosol_reference_aod: 0.03  # 550 nm background AOD of the built-in reference
                             # aerosol field, read only when no aerosol file is
                             # given (the production launchers give one). Written
                             # out so it rests on no default (user decision
                             # 2026-10-15); equals the run_amip default.
# A date outside a multi-year forcing file's years raises (explicit: the
# end-record hold is an opt-in for deliberate experiments only; F37).
forcing_hold_outside_file_years: false
start_year: 1979
diag_days: 1
checkpoint_days: 30         # run control, carried from the outgoing production
                            # deck (not a CAM6 choice)
monthly_means: true
cmip_output: true
clear_sky_diag: true

exec
/usr/bin/bash -lc "ls /glade/derecho/scratch/pg2328/cesm_src; rg --files /glade/derecho/scratch/pg2328/cesm_src | rg '(SurfaceAlbedoMod|CanopyHydrologyMod|radiation.F90|micro_mg_cam.F90|ice_shortwave)'" in /glade/u/home/pg2328/legoesm
 succeeded in 67ms:
CAM
CICE
CLM5
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src_clm40/biogeophys/SurfaceAlbedoMod.F90
/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/simple/radiation.F90
/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/cam/micro_mg_cam.F90
/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/spcam/crm/CLUBB/crmx_parameters_radiation.F90
/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/camrt/radiation.F90
/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90

exec
/usr/bin/bash -lc "rg --files packages | rg '(multilayer_land.py|radiative_transfer.py|integration.py|diagnostics.py)"'$'"'; rg -n 'fcansno|snocan|frac_sno' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/{SurfaceAlbedoMod,CanopyHydrologyMod}.F90; rg -n 'snow_mask|fcansno|snow_age|snow_tau|canopy_snow|snocan' packages/land/legoesm/land/{multilayer_land.py,canopy/radiative_transfer.py}; cat scripts/tmp/probe_snow_albedo_canopy.py" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py
packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py
packages/ocean/legoesm/ocean/diagnostics.py
packages/atmosphere/legoesm/atmosphere/dynamics/crm/rce_diagnostics.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py
packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py
packages/core/legoesm/parallel/scaling_diagnostics.py
packages/coupler/legoesm/driver/diagnostics.py
packages/atmosphere/legoesm/atmosphere/physics/radiation/rrtmgp/config/radiative_transfer.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/fesom_integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/integration.py
packages/core/legoesm/timestepping/integration.py
packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py
packages/land/legoesm/land/carbon_diagnostics.py
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
packages/ocean/legoesm/ocean/physics/convection/integration.py
packages/ocean/legoesm/ocean/physics/lateral_mixing/integration.py
packages/land/legoesm/land/multilayer_land.py
packages/land/legoesm/land/canopy/radiative_transfer.py
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:190:     real(r8) :: snocanmx                                     ! maximum allowed snow on canopy [mm equiv. water]
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:194:     real(r8) :: qflx_snocanfall(bounds%begp:bounds%endp)     ! rate of excess canopy snow falling off canopy
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:243:          snocan               => waterstate_inst%snocan_patch            , & ! Output: [real(r8) (:)   ]  canopy snow (mm H2O)             
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:250:          frac_sno_eff         => waterstate_inst%frac_sno_eff_col        , & ! Output: [real(r8) (:)   ]  eff. fraction of ground covered by snow (0 to 1)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:251:          frac_sno             => waterstate_inst%frac_sno_col            , & ! Output: [real(r8) (:)   ]  fraction of ground covered by snow (0 to 1)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:295:             qflx_snocanfall(p) = 0._r8      ! rate of just snow canopy fall
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:329:                      snocanmx = 60._r8*dewmx(p) * (elai(p) + esai(p))  ! 6*(LAI+SAI)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:347:                      snocan(p) = max(0._r8, snocan(p) + dtime*forc_snow(c)*fpisnow)    
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:359:                   qflx_snocanfall(p) = 0._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:373:                         xsnorun = (snocan(p) - snocanmx)/dtime
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:375:                            qflx_snocanfall(p) = xsnorun
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:376:                            snocan(p) = snocanmx
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:379:                      qflx_candrip(p) = qflx_snocanfall(p) + qflx_liqcanfall(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:380:                      h2ocan(p) = snocan(p) + liqcan(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:404:             snocan(p)            = 0._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:406:             qflx_snocanfall(p)    = 0._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:425:                   if (snocan(p) > 0._r8) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:426:                      qflx_snotempunload(p) = max(0._r8,snocan(p)*(forc_t(c)-270.15_r8)/1.87e5_r8) 
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:427:                      qflx_snowindunload(p) = 0.5_r8*snocan(p)*forc_wind(g)/1.56e5_r8 
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:429:                      if ( snounload(p) > snocan(p) ) then ! Limit unloading to snow in canopy
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:430:                         snounload(p) = snocan(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:431:                         write(iulog,"(A,I2.2,A,ES13.4E2)") "snocan",p,": ",snocan(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:433:                      snocan(p) = snocan(p) - snounload(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:436:                   qflx_prec_grnd_snow(p) = qflx_through_snow(p) + qflx_snocanfall(p)  + snounload(p)/dtime
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:531:                frac_sno(c) = 1. - (acos(min(1._r8,(2.*smr - 1._r8)))/rpi)**(n_melt(c))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:537:                fsno_new = 1._r8 - (1._r8 - tanh(accum_factor*newsnow(c)))*(1._r8 - frac_sno(c))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:538:                frac_sno(c) = fsno_new
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:542:                     / (0.5*(cos(rpi*(1._r8-max(frac_sno(c),1e-6_r8))**(1./n_melt(c)))+1._r8))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:550:                if (frac_sno(c) > 0._r8)then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:551:                   snow_depth(c)=snow_depth(c) + newsnow(c)/(bifall(c) * frac_sno(c))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:564:                   frac_sno(c) = tanh(snow_depth(c)/(2.5_r8*zlnd* &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:568:                   frac_sno(c)=min(frac_sno(c),h2osno(c))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:573:             ! initialize frac_sno and snow_depth when no snow present initially
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:577:                frac_sno(c) = tanh(accum_factor*newsnow(c))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:582:                     / (0.5*(cos(rpi*(1._r8-max(frac_sno(c),1e-6_r8))**(1./n_melt(c)))+1._r8))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:585:                ! update snow_depth and h2osno to be consistent with frac_sno, z_avg
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:587:                   snow_depth(c)=z_avg/frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:595:                      frac_sno(c) = tanh(snow_depth(c)/(2.5_r8*zlnd* &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:602:                frac_sno(c) = 0._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:616:          ! set frac_sno_eff variable
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:619:                frac_sno_eff(c) = frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:621:                frac_sno_eff(c) = 1._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:624:             frac_sno_eff(c) = 1._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:637:          if (snl(c) == 0 .and. frac_sno(c)*snow_depth(c) >= 0.01_r8) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:664:       ! update surface water fraction (this may modify frac_sno)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:704:          snocan         => waterstate_inst%snocan_patch          , & ! Output: [real(r8) (:)   ]  canopy snow (mm H2O)             
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:708:          fcansno        => waterstate_inst%fcansno_patch         , & ! Output: [real(r8) (:) ]  fraction of canopy that is snow covered (0 to 1) 
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:724:                   if (snocan(p) > 0._r8) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:726:                      fcansno(p) = ((dewmxi/(vegt*6.0_r8*10.0_r8))*snocan(p))**0.15_r8 ! must match snocanmx 
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:727:                      fcansno(p) = min (fcansno(p),1.0_r8)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:729:                      fcansno(p) = 0._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:732:                   fcansno(p) = 0._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:736:                fcansno(p) = 0._r8
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:782:          frac_sno     => waterstate_inst%frac_sno_col     , & ! Output: [real(r8) (:)   ] fraction of ground covered by snow (0 to 1)       
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:783:          frac_sno_eff => waterstate_inst%frac_sno_eff_col , & ! Output: [real(r8) (:)   ] eff. fraction of ground covered by snow (0 to 1)  
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:830:                if (frac_sno(c) > (1._r8 - frac_h2osfc(c)) .and. h2osno(c) > 0) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:833:                      frac_h2osfc(c) = max(1.0_r8 - frac_sno(c),0.01_r8)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:834:                      frac_sno(c) = 1.0_r8 - frac_h2osfc(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:836:                      frac_sno(c) = 1.0_r8 - frac_h2osfc(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:838:                   frac_sno_eff(c)=frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:296:          frac_sno      =>    waterstate_inst%frac_sno_col        , & ! Input:  [real(r8)  (:)   ]  fraction of ground covered by snow (0 to 1)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:659:             albgrd(c,ib) = albsod(c,ib)*(1._r8-frac_sno(c)) + albsnd(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:660:             albgri(c,ib) = albsoi(c,ib)*(1._r8-frac_sno(c)) + albsni(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:665:                albgrd_bc(c,ib) = albsod(c,ib)*(1.-frac_sno(c)) + albsnd_bc(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:666:                albgri_bc(c,ib) = albsoi(c,ib)*(1.-frac_sno(c)) + albsni_bc(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:670:                   albgrd_oc(c,ib) = albsod(c,ib)*(1.-frac_sno(c)) + albsnd_oc(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:671:                   albgri_oc(c,ib) = albsoi(c,ib)*(1.-frac_sno(c)) + albsni_oc(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:675:                albgrd_dst(c,ib) = albsod(c,ib)*(1.-frac_sno(c)) + albsnd_dst(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:676:                albgri_dst(c,ib) = albsoi(c,ib)*(1.-frac_sno(c)) + albsni_dst(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:679:                albgrd_pur(c,ib) = albsod(c,ib)*(1.-frac_sno(c)) + albsnd_pur(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:680:                albgri_pur(c,ib) = albsoi(c,ib)*(1.-frac_sno(c)) + albsni_pur(c,ib)*frac_sno(c)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:689:                   flx_absdv(c,i) = flx_absd_snw(c,i,ib)*frac_sno(c) + &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:690:                        ((1.-frac_sno(c))*(1-albsod(c,ib))*(flx_absd_snw(c,i,ib)/(1.-albsnd(c,ib))))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:691:                   flx_absiv(c,i) = flx_absi_snw(c,i,ib)*frac_sno(c) + &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:692:                        ((1.-frac_sno(c))*(1-albsoi(c,ib))*(flx_absi_snw(c,i,ib)/(1.-albsni(c,ib))))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:694:                   flx_absdn(c,i) = flx_absd_snw(c,i,ib)*frac_sno(c) + &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:695:                        ((1.-frac_sno(c))*(1-albsod(c,ib))*(flx_absd_snw(c,i,ib)/(1.-albsnd(c,ib))))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:696:                   flx_absin(c,i) = flx_absi_snw(c,i,ib)*frac_sno(c) + &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:697:                        ((1.-frac_sno(c))*(1-albsoi(c,ib))*(flx_absi_snw(c,i,ib)/(1.-albsni(c,ib))))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1205:          fcansno      =>    waterstate_inst%fcansno_patch       , & ! Input:  [real(r8) (:)   ]  fraction of canopy that is snow-covered (0 to 1) 
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1328:                tmp0 =   (1._r8-fcansno(p))*omegal        + fcansno(p)*omegas(ib)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1329:                tmp1 = ( (1._r8-fcansno(p))*omegal*betadl + fcansno(p)*omegas(ib)*betads ) / tmp0
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1330:                tmp2 = ( (1._r8-fcansno(p))*omegal*betail + fcansno(p)*omegas(ib)*betais ) / tmp0
packages/land/legoesm/land/canopy/radiative_transfer.py:498:                           alb_ground, fcansno=0.0, omega_snow=0.0):
packages/land/legoesm/land/canopy/radiative_transfer.py:505:    orientation index, ``fcansno`` the snow-covered canopy fraction with
packages/land/legoesm/land/canopy/radiative_transfer.py:532:    om = (1.0 - fcansno) * omegal + fcansno * omega_snow
packages/land/legoesm/land/canopy/radiative_transfer.py:533:    betad = ((1.0 - fcansno) * omegal * betadl
packages/land/legoesm/land/canopy/radiative_transfer.py:534:             + fcansno * omega_snow * _CLM_BETA_SNOW) / om
packages/land/legoesm/land/canopy/radiative_transfer.py:535:    betai = ((1.0 - fcansno) * omegal * betail
packages/land/legoesm/land/canopy/radiative_transfer.py:536:             + fcansno * omega_snow * _CLM_BETA_SNOW) / om
packages/land/legoesm/land/canopy/radiative_transfer.py:650:    (fcansno = 0): a bare-branch bound."""
packages/land/legoesm/land/multilayer_land.py:64:from legoesm.land.snow_budget import update_snow_age
packages/land/legoesm/land/multilayer_land.py:481:    (``config.canopy_snow_masking``): wraps the unmasked ``snowy_fn`` (soil band
packages/land/legoesm/land/multilayer_land.py:488:            "canopy_snow_masking needs per-column SAI_dom, hbot_dom and pft_dom in the "
packages/land/legoesm/land/multilayer_land.py:552:    snow_age = state.snow_age    # (ncol,)
packages/land/legoesm/land/multilayer_land.py:633:    if config.canopy_snow_masking and (
packages/land/legoesm/land/multilayer_land.py:639:            "canopy_snow_masking is implemented only for the two-leaf canopy with "
packages/land/legoesm/land/multilayer_land.py:642:    if config.canopy_snow_masking and config.snow_scheme == "layered":
packages/land/legoesm/land/multilayer_land.py:648:            "canopy_snow_masking is not supported with snow_scheme='layered' yet: "
packages/land/legoesm/land/multilayer_land.py:919:                lat, snow, snow_age, config.land_albedo,
packages/land/legoesm/land/multilayer_land.py:922:            if config.canopy_snow_masking:
packages/land/legoesm/land/multilayer_land.py:1045:            snow_age=snow_age,
packages/land/legoesm/land/multilayer_land.py:1125:                _bands_eff, state.snow_age_bands, _base, _cover_fn, _alb_fn,
packages/land/legoesm/land/multilayer_land.py:1147:            state.snow_bands, state.snow_age_bands, ice_bands_in, T_surface,
packages/land/legoesm/land/multilayer_land.py:1151:            snow_age_activation_K=config.land_albedo.snow_age_activation_K,
packages/land/legoesm/land/multilayer_land.py:1155:        snow_age_bands_new = band_step.snow_age_bands
packages/land/legoesm/land/multilayer_land.py:1157:        snow_age_new = band_step.snow_age
packages/land/legoesm/land/multilayer_land.py:1168:        snow_age_new = None          # set after the final solve
packages/land/legoesm/land/multilayer_land.py:1171:        snow_age_bands_new = state.snow_age_bands
packages/land/legoesm/land/multilayer_land.py:1178:        snow_new, snow_age_new, snow_melt = update_snow(
packages/land/legoesm/land/multilayer_land.py:1179:            snow, snow_age, T_surface, precip_snow_eff, dt,
packages/land/legoesm/land/multilayer_land.py:1188:            snow_age_activation_K=config.land_albedo.snow_age_activation_K,
packages/land/legoesm/land/multilayer_land.py:1191:        snow_age_bands_new = state.snow_age_bands
packages/land/legoesm/land/multilayer_land.py:1558:        snow_age_new = update_snow_age(
packages/land/legoesm/land/multilayer_land.py:1559:            snow_new, snow_age, precip_snow_eff, dt, T_snow=T_surface,
packages/land/legoesm/land/multilayer_land.py:1560:            age_activation_K=config.land_albedo.snow_age_activation_K)
packages/land/legoesm/land/multilayer_land.py:1780:        snow_age=_match(snow_age_new, state.snow_age),
packages/land/legoesm/land/multilayer_land.py:1785:        snow_age_bands=_match(snow_age_bands_new, state.snow_age_bands),
packages/land/legoesm/land/multilayer_land.py:1846:                snow_bands_new, snow_age_bands_new, _base_new,
packages/land/legoesm/land/multilayer_land.py:1871:                lat, snow_new, snow_age_new, config.land_albedo,
packages/land/legoesm/land/multilayer_land.py:1874:            if config.canopy_snow_masking:
packages/land/legoesm/land/multilayer_land.py:1890:                lat, snow_new, snow_age_new, config.land_albedo,
packages/land/legoesm/land/multilayer_land.py:2407:        snow_age_bands = jnp.zeros((ncol, _nb), dtype=_bdt)
packages/land/legoesm/land/multilayer_land.py:2410:        snow_bands = snow_age_bands = ice_bands = None
packages/land/legoesm/land/multilayer_land.py:2429:        snow_age=jnp.zeros(ncol),
packages/land/legoesm/land/multilayer_land.py:2433:        snow_age_bands=snow_age_bands,
"""Probe: exported land albedo of snowy boreal columns, production land albedo
functions, canopy snow masking off/on x snow ageing (production 150 d with the
5000 K temperature clock vs the land calibration's 3.674 d calendar clock).
Mirrors the production call (multilayer_land.py ~918): land_albedo without
cos_zenith on each band, optionally wrapped in canopy_masked_snow_albedo.
Reference: CLM5 two-stream WITH intercepted canopy snow (fcansno 0.5, 1.0)."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.land.canopy import radiative_transfer as rt
from legoesm.land.clm_surface_map import biophysics_lmip_albedo_scalars
from legoesm.land.snow_budget import update_snow_age
from legoesm.surface_albedo import LandAlbedoConfig, land_albedo, snow_cover_fraction

# name, pft, LAI, SAI, htop, hbot, snow-free (vis, nir)  -- CLM5 boreal spring values (approx.)
COLS = [
    ("NET boreal (spruce)", 2, 2.0, 0.6, 17.0, 8.5, 0.05, 0.22),
    ("NDT boreal (larch)", 3, 0.0, 0.6, 14.0, 7.0, 0.06, 0.22),
    ("BDT boreal (birch)", 8, 0.0, 0.8, 20.0, 11.5, 0.06, 0.24),
    ("BDS boreal shrub", 11, 0.1, 0.3, 0.5, 0.1, 0.08, 0.26),
    ("C3 arctic grass", 12, 0.1, 0.2, 0.5, 0.01, 0.08, 0.28),
    ("bare soil", 0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.25),
]
SWE = 120.0          # kg/m2
COSZ, SWD = 0.4, 400.0
DAY = 86400.0

scal = biophysics_lmip_albedo_scalars()
arms = {
    "prod 150d+T": (150.0 * DAY, 5000.0),
    "calib 3.67d": (scal["tau_snow_decay"], 0.0),
}


def age_after(act, schedule):
    age = jnp.zeros(1)
    swe = jnp.full(1, SWE)
    for ndays, T in schedule:
        for _ in range(ndays):
            age = update_snow_age(swe, age, jnp.zeros(1), DAY, T_snow=jnp.full(1, T),
                                  age_activation_K=act)
    return age


stages = {
    "fresh": [],
    "90d at 255K": [(90, 255.0)],
    "+10d melting": [(90, 255.0), (10, constants.T_freeze)],
}

n = len(COLS)
pft = jnp.asarray([c[1] for c in COLS], float)
lai = jnp.asarray([c[2] for c in COLS]); sai = jnp.asarray([c[3] for c in COLS])
htop = jnp.asarray([c[4] for c in COLS]); hbot = jnp.asarray([c[5] for c in COLS])
base = [jnp.asarray([c[6] for c in COLS]), jnp.asarray([c[7] for c in COLS])]
swe = jnp.full(n, SWE); cz = jnp.full(n, COSZ); lat = jnp.full(n, np.deg2rad(60.0))
pd, pf, nd, nf, _ = rt.split_sw_components(jnp.full(n, SWD), cz)
fdif = (pf / (pd + pf), nf / (nd + nf))

print(f"SWE {SWE} kg/m2, cosz {COSZ}; broadband = rt.broadband_albedo(vis, nir)")
for arm, (tau, act) in arms.items():
    la = LandAlbedoConfig()._replace(**{**scal, "tau_snow_decay": tau,
                                        "snow_age_activation_K": act})
    fs = snow_cover_fraction(swe, la)
    for st, sched in stages.items():
        age = jnp.full(n, float(age_after(act, sched)[0]))
        out_off, out_on = [], []
        for ib in (0, 1):
            snowy = land_albedo(lat, swe, age, la, base_albedo=base[ib])
            out_off.append(snowy)
            out_on.append(rt.canopy_masked_snow_albedo(
                base[ib], snowy, ib, lai, sai, htop, hbot, pft, fs,
                swe / constants.rho_snow_land, cz, fdif[ib]))
        off = np.asarray(rt.broadband_albedo(*out_off))
        on = np.asarray(rt.broadband_albedo(*out_on))
        print(f"\n[{arm} | {st}] age={float(age[0])/DAY:.1f} melt-equiv days")
        for i, c in enumerate(COLS):
            print(f"  {c[0]:22s} off {off[i]:.3f}  on {on[i]:.3f}")

# CLM5 reference with canopy snow on the spruce column, ground = fresh-snow bands
xl_t, rhol_t, taul_t, rhos_t, taus_t = (np.asarray(t) for t in rt._clm5_pft_optics())
print("\nCLM5 two-stream NET boreal, fully exposed (VAI 2.6), snowy ground vis/nir=0.95/0.65:")
for fc in (0.0, 0.5, 1.0):
    bands = []
    for ib, ag in ((0, 0.95), (1, 0.65)):
        albd, albi = rt.clm5_two_stream_albedo(
            jnp.asarray(2.6), jnp.asarray(2.0 / 2.6), rhol_t[2, ib], taul_t[2, ib],
            rhos_t[2, ib], taus_t[2, ib], xl_t[2], jnp.asarray(COSZ), jnp.asarray(ag),
            fcansno=fc, omega_snow=rt._CLM_OMEGA_SNOW[ib])
        bands.append(fdif[ib][0] * albi + (1 - fdif[ib][0]) * albd)
    print(f"  fcansno {fc}: broadband {float(rt.broadband_albedo(*bands)):.3f}")

exec
/usr/bin/bash -lc "rg -n 'albedo_ice|maykut|sea_ice_albedo_scheme|land_canopy_snow_masking|land_snow_tau_days' packages/coupler/legoesm/driver/{config.py,model_driver.py} packages/core/legoesm/surface_albedo.py; rg --files packages | rg '(snow_budget|surface_albedo|multilayer.*types|land.*state).py'; rg -n 'albedo|shortwave' /glade/derecho/scratch/pg2328/cesm_src/CICE/cicecore/drivers/cesm/* 2>/dev/null | head -25" in /glade/u/home/pg2328/legoesm
exec
/usr/bin/bash -lc "nl -ba packages/coupler/legoesm/driver/diagnostics.py | sed -n '790,930p'; rg -n 'q_s|q_c|q_i|snow|def ' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | tail -65; nl -ba /glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90 | sed -n '875,995p'; nl -ba /glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/cam/micro_mg_cam.F90 | sed -n '2290,2330p'" in /glade/u/home/pg2328/legoesm
 succeeded in 12ms:
packages/coupler/legoesm/driver/config.py:674:def resolve_sea_ice_albedo_scheme(cfg) -> str:
packages/coupler/legoesm/driver/config.py:677:    ``"auto"`` resolves to ``"maykut_untersteiner"`` when the run carries the
packages/coupler/legoesm/driver/config.py:681:    scheme = getattr(cfg, "sea_ice_albedo_scheme", "auto")
packages/coupler/legoesm/driver/config.py:684:    return ("maykut_untersteiner" if getattr(cfg, "ice_skin_prognostic", False)
packages/coupler/legoesm/driver/config.py:1018:    # with albedo_ice 0.80, polar clear-sky reflected SW bias -15.7 -> -2.7
packages/coupler/legoesm/driver/config.py:1020:    land_snow_tau_days: float | None = 150.0
packages/coupler/legoesm/driver/config.py:1424:    land_canopy_snow_masking: bool = False
packages/coupler/legoesm/driver/config.py:1537:    # DEFAULT moved 0.65 -> 0.80 (user 2026-09-15), see land_snow_tau_days.
packages/coupler/legoesm/driver/config.py:1538:    albedo_ice: float = 0.80
packages/coupler/legoesm/driver/config.py:2147:    #   "auto" (default)      -> resolved per run by resolve_sea_ice_albedo_scheme:
packages/coupler/legoesm/driver/config.py:2148:    #     "maykut_untersteiner" when the lane carries the prognostic ice skin,
packages/coupler/legoesm/driver/config.py:2151:    #   "constant"            -> albedo_ice everywhere, every season.
packages/coupler/legoesm/driver/config.py:2152:    #   "maykut_untersteiner" -> ice.shortwave.maykut_untersteiner_albedo on the
packages/coupler/legoesm/driver/config.py:2153:    #     prognostic ice skin, recomputed every step: albedo_ice when the skin
packages/coupler/legoesm/driver/config.py:2156:    sea_ice_albedo_scheme: str = "auto"
packages/coupler/legoesm/driver/config.py:2752:        if self.land_canopy_snow_masking and not (
packages/coupler/legoesm/driver/config.py:2756:                "land_canopy_snow_masking requires use_multilayer_land, "
packages/coupler/legoesm/driver/config.py:2759:        if self.land_canopy_snow_masking and self.land_snow_scheme == "layered":
packages/coupler/legoesm/driver/config.py:2761:                "land_canopy_snow_masking is not supported with land_snow_scheme="
packages/coupler/legoesm/driver/config.py:4037:        _sia = ("auto", "constant", "maykut_untersteiner")
packages/coupler/legoesm/driver/config.py:4038:        if self.sea_ice_albedo_scheme not in _sia:
packages/coupler/legoesm/driver/config.py:4040:                f"sea_ice_albedo_scheme={self.sea_ice_albedo_scheme!r} "
packages/coupler/legoesm/driver/config.py:4042:        if (self.sea_ice_albedo_scheme == "maykut_untersteiner"
packages/coupler/legoesm/driver/config.py:4045:                "sea_ice_albedo_scheme='maykut_untersteiner' reads the "
packages/coupler/legoesm/driver/config.py:4056:                and resolve_sea_ice_albedo_scheme(self)
packages/coupler/legoesm/driver/config.py:4057:                != "maykut_untersteiner"):
packages/coupler/legoesm/driver/config.py:4061:                "sea_ice_albedo_scheme='maykut_untersteiner' (the override "
packages/coupler/legoesm/driver/config.py:4724:            ("land_snow_tau_days", 0.5, 400.0),
packages/coupler/legoesm/driver/config.py:5326:            albedo_ice=amip_cfg.albedo_ice,
packages/coupler/legoesm/driver/config.py:5525:            albedo_ice=self.albedo_ice,
packages/coupler/legoesm/driver/model_driver.py:2030:    _tau_d = getattr(config, "land_snow_tau_days", None)
packages/coupler/legoesm/driver/model_driver.py:2042:    _csm = bool(config.land_canopy_snow_masking)
packages/coupler/legoesm/driver/model_driver.py:3172:            resolve_ice_skin_prognostic, resolve_sea_ice_albedo_scheme)
packages/coupler/legoesm/driver/model_driver.py:3185:        upd["sea_ice_albedo_scheme"] = resolve_sea_ice_albedo_scheme(cfg)
packages/coupler/legoesm/driver/model_driver.py:3189:                    upd["sea_ice_albedo_scheme"], "ON" if skin else "OFF", why)
packages/coupler/legoesm/driver/model_driver.py:14206:        _albedo_ice = float(cfg.albedo_ice)
packages/coupler/legoesm/driver/model_driver.py:14207:        from legoesm.driver.config import resolve_sea_ice_albedo_scheme
packages/coupler/legoesm/driver/model_driver.py:14209:        _sia_scheme = resolve_sea_ice_albedo_scheme(cfg)
packages/coupler/legoesm/driver/model_driver.py:14210:        if _sia_scheme not in ("constant", "maykut_untersteiner"):
packages/coupler/legoesm/driver/model_driver.py:14212:                f"sea_ice_albedo_scheme={_sia_scheme!r} unknown; choose "
packages/coupler/legoesm/driver/model_driver.py:14213:                "'auto', 'constant' or 'maykut_untersteiner'.")
packages/coupler/legoesm/driver/model_driver.py:14214:        _ice_alb_melt = _sia_scheme == "maykut_untersteiner"
packages/coupler/legoesm/driver/model_driver.py:14218:                "sea_ice_albedo_scheme='maykut_untersteiner' needs "
packages/coupler/legoesm/driver/model_driver.py:14225:                _sia_scheme, _albedo_ice,
packages/coupler/legoesm/driver/model_driver.py:14284:                _albedo_ocean, _albedo_ice, _alb_lo, _alb_hi,
packages/coupler/legoesm/driver/model_driver.py:14290:                _albedo_ocean, _albedo_ice)
packages/coupler/legoesm/driver/model_driver.py:17787:                        sic_step.reshape(-1), _sp_f_land, float(cfg.albedo_ice),
packages/coupler/legoesm/driver/model_driver.py:18575:            albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
packages/coupler/legoesm/driver/model_driver.py:19221:            albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
packages/coupler/legoesm/driver/model_driver.py:20112:            albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
packages/coupler/legoesm/driver/model_driver.py:20416:                albedo_ice=cfg.albedo_ice,
packages/coupler/legoesm/driver/model_driver.py:21157:                        albedo_ice=cfg.albedo_ice, albedo_ocean=cfg.albedo_ocean,
packages/core/legoesm/surface_albedo.py
packages/land/legoesm/land/snow_budget.py
packages/land/legoesm/land/state.py
packages/land/legoesm/land/canopy/state.py

exec
/usr/bin/bash -lc "nl -ba packages/land/legoesm/land/canopy/radiative_transfer.py | sed -n '625,735p'; nl -ba packages/land/legoesm/land/multilayer_land.py | sed -n '900,945p;1850,1905p'; rg -n 'canopy_water|canopy_snow|snocan|intercep|snow_insulation' packages/land/legoesm/land/{multilayer_land.py,state.py,types.py}; nl -ba /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90 | sed -n '690,743p'" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
   625	        return jnp.where(e < _CLM_VAI_CUT, 0.0, e)
   626	
   627	    return _exposed(LAI), _exposed(SAI)
   628	
   629	
   630	def canopy_masked_snow_albedo(alb_snowfree, alb_snowy, band, LAI, SAI, htop,
   631	                              hbot, pft_index, f_snow, snow_depth, cosz,
   632	                              f_diffuse):
   633	    """Column albedo of one band with the canopy hiding the snow (CLM5).
   634	
   635	    DELTA form: the snow-free column albedo is left exactly as supplied and only
   636	    the snow increment is passed through the canopy,
   637	
   638	        alb = alb_snowfree + TS(elai, esai; alb_snowy) - TS(tlai, tsai; alb_snowfree)
   639	
   640	    with TS the CLM5 two-stream column albedo (:func:`clm5_two_stream_albedo`)
   641	    as the step's light sees it, ``f_diffuse * albi + (1 - f_diffuse) * albd``
   642	    (beam albedo at ``cosz``; the model carries one albedo per band, and the
   643	    canopy RT reflects that one value off both beam and diffuse light, so the
   644	    blend makes the reflected total exact for this step's split), elai/esai the
   645	    CLM5 snow-buried leaf/stem area and ``alb_snowy`` the ground albedo with
   646	    snow already blended on (the unmasked model value).  No exposed plant area
   647	    (none, or all of it buried) -> ``alb_snowy`` exactly; no snow ->
   648	    ``alb_snowfree`` exactly.  ``band`` 0 =
   649	    visible, 1 = near-infrared.  Intercepted canopy snow is not represented
   650	    (fcansno = 0): a bare-branch bound."""
   651	    xl_t, rhol_t, taul_t, rhos_t, taus_t = (jnp.asarray(t) for t in _clm5_pft_optics())
   652	    ip = jnp.clip(jnp.round(pft_index).astype(jnp.int32), 0, _N_CLM_PFT - 1)
   653	    elai, esai = clm5_exposed_area(LAI, SAI, htop, hbot, pft_index, f_snow,
   654	                                   snow_depth)
   655	
   656	    def _ts(lai, sai, ground):
   657	        vai = lai + sai
   658	        f_leaf = lai / jnp.maximum(vai, _CLM_MPE)
   659	        albd, albi = clm5_two_stream_albedo(
   660	            vai, f_leaf, rhol_t[ip, band], taul_t[ip, band], rhos_t[ip, band],
   661	            taus_t[ip, band], xl_t[ip], cosz, ground)
   662	        return f_diffuse * albi + (1.0 - f_diffuse) * albd
   663	
   664	    tlai = jnp.where(LAI < _CLM_VAI_CUT, 0.0, LAI)
   665	    tsai = jnp.where(SAI < _CLM_VAI_CUT, 0.0, SAI)
   666	    # Parenthesised so a zero increment returns alb_snowfree bit-exactly.  No
   667	    # EXPOSED plant area (none at all, or all buried) returns alb_snowy
   668	    # bit-exactly, the unmasked model value: CTSM puts elai + esai = 0 in its
   669	    # non-vegetated albedo filter (SurfaceAlbedoMod 805-814).
   670	    masked = alb_snowfree + (_ts(elai, esai, alb_snowy)
   671	                             - _ts(tlai, tsai, alb_snowfree))
   672	    return jnp.where(elai + esai > 0.0, masked, alb_snowy)
   673	
   674	
   675	# ---------------------------------------------------------------------------
   676	# Longwave radiative transfer
   677	# ---------------------------------------------------------------------------
   678	
   679	@jax.jit
   680	def canopy_longwave_rt(
   681	    LAI: jax.Array,
   682	    CI: jax.Array,
   683	    SZA: jax.Array,
   684	    Ts: jax.Array,
   685	    Tf_Sun: jax.Array,
   686	    Tf_Sh: jax.Array,
   687	    La: jax.Array,
   688	    epsf: float | jax.Array,
   689	    epss: float | jax.Array,
   690	) -> CanopyLWOutput:
   691	    """Absorbed longwave radiation by sunlit/shaded leaves and soil.
   692	
   693	    Two-big-leaf longwave transfer with Beer-law extinction against the
   694	    **effective LAI** ``L_eff = LAI * CI`` (clumping correction, consistent
   695	    with the shortwave routine), following Ryu et al. (2011) / CABLE / CLM.
   696	
   697	    Sunlit and shaded leaves absorb sky and soil longwave through separate
   698	    depth-weighted two-stream kernels (``W_*_sky``, ``W_*_soil``) and emit
   699	    with their own temperature.  By Kirchhoff / detailed balance the same
   700	    kernels weight the canopy emission reaching the soil:
   701	
   702	        canopy LW reaching soil = W_sun_soil * Lf_Sun + W_sh_soil * Lf_Sh
   703	
   704	    which is exact (under the kd kernel) when ``Tf_Sun != Tf_Sh`` and reduces
   705	    to the bulk ``W_tot * Lf`` form in the isothermal limit.  This replaces
   706	    the earlier ``(1 - exp(-kd LAI)) (Ls + La - 2 Lf_Sh)`` shaded form, whose
   707	    ``2 Lf_Sh`` self-emission term spuriously warmed shaded leaves when
   708	    ``Tf_Sun > Tf_Sh`` over hot soil.  The canopy is always fully coupled to
   709	    the soil (no decoupling option).
   710	
   711	    Conservative gray-body emissivity (β = 1/2 isotropic leaf scatter): each
   712	    surface ABSORBS ``eps * incident`` (Kirchhoff: absorptivity = emissivity)
   713	    and REFLECTS ``(1 - eps)``.  The diffuse canopy stream splits into absorbed
   714	    ``a_c = eps_f * W_tot``, back-scattered ``r_c = (1-eps_f) W_tot / 2`` and
   715	    transmitted ``t_c`` (``a_c + r_c + t_c = 1``), and the canopy<->ground
   716	    interreflection is summed in closed form ``1/(1 - rho * r_c)`` (one division;
   717	    ``rho * r_c <= 6e-4`` for physical eps).  This is EXACTLY energy-conserving
   718	    (``ALW_Sun + ALW_Sh + ALW_Soil = La - LW_out``) and gives zero net flux at
   719	    isothermal equilibrium for ANY eps_f, eps_s; it reduces exactly to the
   720	    previous near-black scheme at eps_f = eps_s = 1 (then r_c = 0, t_c = gap_LW).
   721	    ``LW_out`` is the true top-of-canopy upward LW (emission + reflection) for the
   722	    coupler and the LST diagnostic.  Ported from DifferBESS; full derivation in
   723	    docs/canopy_longwave_graybody_conservation.md §5.
   724	
   725	    Parameters
   726	    ----------
   727	    LAI     : (ncol,) leaf area index [m2/m2] (one-sided, nominal)
   728	    CI      : (ncol,) clumping index [-]; effective LAI for radiation = LAI*CI
   729	    SZA     : (ncol,) solar zenith angle [degrees]
   730	    Ts      : (ncol,) soil surface temperature [K]
   731	    Tf_Sun  : (ncol,) sunlit leaf temperature [K]
   732	    Tf_Sh   : (ncol,) shaded leaf temperature [K]
   733	    La      : (ncol,) incoming atmospheric longwave [W m-2]
   734	    epsf    : leaf emissivity (scalar or array)
   735	    epss    : soil emissivity (scalar or array)
   900	        # band albedos are the snow-free soil-colour background, so without
   901	        # this the land absorbed sunlight through ~0.15 while the atmosphere
   902	        # reflected the exported snow-aged 0.52 on the same cell -- nothing
   903	        # reconciled them (energy created on snow-covered tundra, lost on
   904	        # glacier).  Snow is layered on each band's own base, so snow-free
   905	        # columns absorb exactly as before and the calibrated glacier bands
   906	        # survive; the prognostic dry-soil brightening the old export added
   907	        # is NOT applied here because the bands carry the CTSM soil-colour
   908	        # moisture dependence (rewet above) and both reviewers flagged the
   909	        # double count.  Absorption this step is exactly (1 - broadband of
   910	        # these bands) * sw_down in daylight (the RT's own low-light fallback,
   911	        # sw_down < 1 W/m2, is the only exception).  With snow feedback on, the
   912	        # EXPORT (post-step block below) is the same construction on the
   913	        # post-step snow and soil water, i.e. what the NEXT step absorbs with,
   914	        # so the hand-off to the next radiation call is exact; within one step
   915	        # they differ by that step's snow and top-layer water change.
   916	        if (config.snow_albedo_feedback and lat is not None
   917	                and lp is not None and hasattr(lp, "ALB_VIS")):
   918	            _band = lambda a: compute_land_albedo(
   919	                lat, snow, snow_age, config.land_albedo,
   920	                base_albedo=jnp.broadcast_to(a, T_surface.shape))
   921	            _vis, _nir = _band, _band
   922	            if config.canopy_snow_masking:
   923	                _vis, _nir = (_canopy_masked_band(_band, lp, LAI_override, snow,
   924	                                                  config, ib, forcing)
   925	                              for ib in (0, 1))
   926	            lp = lp._replace(ALB_VIS=_vis(lp.ALB_VIS), ALB_NIR=_nir(lp.ALB_NIR))
   927	            _alpha_applied = broadband_albedo(lp.ALB_VIS, lp.ALB_NIR)
   928	        surface_out = compute_two_leaf_canopy_fluxes(
   929	            T_soil_top=T_surface,
   930	            forcing=forcing,
   931	            canopy_config=config.surface_scheme,
   932	            land_config=config,
   933	            canopy_params=lp,
   934	            w_frac_rz=w_frac_rz,
   935	            wind_speed=wind_speed,
   936	            wind_dir_x=wind_dir_x,
   937	            wind_dir_y=wind_dir_y,
   938	            soil_thermal_fn=_soil_thermal_cb,
   939	            dt=dt,
   940	            TgC_override=TgC_override,
   941	            LAI_override=LAI_override,
   942	            fwet=_fwet_pre,
   943	            # Warm start: the previous step's last converged canopy solution
   944	            # (None on a state that does not carry the cache -> cold start).
   945	            canopy_seed=state.canopy_x,
  1850	                       else jnp.asarray(config.land_albedo.snow_cover_scale)[:, None]),
  1851	                    0.0, 1.0),
  1852	                lambda a: snow_albedo(a, config.land_albedo, cos_zenith=_cz_new),
  1853	                ice_bands=ice_bands_new, cfg=bands)
  1854	        else:
  1855	            alpha_bands_new = jnp.broadcast_to(
  1856	                jnp.reshape(albedo_land_post, (-1, 1)), (ncol, bands.band_dz.shape[-1]))
  1857	        band_rad_new = band_net_radiation(
  1858	            T_surface_new, alpha_bands_new, forcing.sw_down, forcing.lw_down,
  1859	            emissivity, bands)
  1860	        alpha_new = band_rad_new.alpha_eff
  1861	        lw_up_new = band_rad_new.lw_up_agg
  1862	    else:
  1863	        if _alpha_applied is not None:
  1864	            # Same bands, same snow layering, POST-step snow and soil water:
  1865	            # the export feeds the NEXT radiation call, whose canopy will
  1866	            # absorb with the post-step state (codex).  Absorption this step
  1867	            # used the pre-step bands (``_alpha_applied``); the two differ only
  1868	            # by one step's snow and top-layer water change.
  1869	            _lp_new = rewet_soil_bands(_lp_soil, richards_out.theta_new[:, 0])
  1870	            _band_new = lambda a: compute_land_albedo(
  1871	                lat, snow_new, snow_age_new, config.land_albedo,
  1872	                base_albedo=jnp.broadcast_to(a, T_surface_new.shape))
  1873	            _vis_new, _nir_new = _band_new, _band_new
  1874	            if config.canopy_snow_masking:
  1875	                _vis_new, _nir_new = (
  1876	                    _canopy_masked_band(_band_new, _lp_new, LAI_override,
  1877	                                        snow_new, config, ib, forcing)
  1878	                    for ib in (0, 1))
  1879	            alpha_new = broadband_albedo(_vis_new(_lp_new.ALB_VIS),
  1880	                                         _nir_new(_lp_new.ALB_NIR))
  1881	        elif getattr(_lp_soil, "ALB_VIS_DRY", None) is not None:
  1882	            # Two-leaf without snow layering: the soil bands at the post-step
  1883	            # water, i.e. what the next step absorbs with (same hand-off as
  1884	            # above, minus snow).
  1885	            _lp_new = rewet_soil_bands(_lp_soil, richards_out.theta_new[:, 0])
  1886	            alpha_new = broadband_albedo(_lp_new.ALB_VIS, _lp_new.ALB_NIR)
  1887	        elif config.snow_albedo_feedback and lat is not None:
  1888	            # Per-cell base albedo (CLM PFT / trainable), consistent with the SEB.
  1889	            alpha_new = compute_land_albedo(
  1890	                lat, snow_new, snow_age_new, config.land_albedo,
  1891	                base_albedo=jnp.broadcast_to(albedo_land_post, T_surface_new.shape))
  1892	        else:
  1893	            alpha_new = surface_out.albedo
  1894	        # lw_up recomputed with post-step surface T and surface scheme's
  1895	        # effective emissivity (canopy RT vs scalar land emissivity).
  1896	        _, _, lw_up_new = surface_radiation_fluxes(
  1897	            forcing.sw_down, forcing.lw_down, T_skin_new, alpha_new,
  1898	            emissivity,
  1899	        )
  1900	
  1901	    # --- Post-step q_surface ---
  1902	    # Reuse the same Audit #6 / Iter-65 range floor as the pre-step branch
  1903	    # above (now inside root_zone_beta_soil) so degenerate PFT cells cannot
  1904	    # blow up beta_soil_new propagating into the q_surface reported back to
  1905	    # the atmosphere.  Computed UNCONDITIONALLY (not folded into the
rg: packages/land/legoesm/land/types.py: No such file or directory (os error 2)
packages/land/legoesm/land/state.py:72:    # interception scheme (``config.interception``; two-leaf / SimpleSEB path).
packages/land/legoesm/land/state.py:73:    # ``None`` (legacy / interception off) carries no store.  The CLM-ML canopy
packages/land/legoesm/land/state.py:99:    # it does not.  Present iff ``config.thermal.snow_insulation``; ``None``
packages/land/legoesm/land/multilayer_land.py:92:from legoesm.land.canopy.interception import (
packages/land/legoesm/land/multilayer_land.py:93:    intercept_rain,
packages/land/legoesm/land/multilayer_land.py:94:    max_canopy_water,
packages/land/legoesm/land/multilayer_land.py:95:    wetted_fraction as interception_wetted_fraction,
packages/land/legoesm/land/multilayer_land.py:481:    (``config.canopy_snow_masking``): wraps the unmasked ``snowy_fn`` (soil band
packages/land/legoesm/land/multilayer_land.py:488:            "canopy_snow_masking needs per-column SAI_dom, hbot_dom and pft_dom in the "
packages/land/legoesm/land/multilayer_land.py:585:    _snow_node = bool(config.thermal.snow_insulation)
packages/land/legoesm/land/multilayer_land.py:589:                "config.thermal.snow_insulation is on but state.T_snow is None; "
packages/land/legoesm/land/multilayer_land.py:596:                "config.thermal.snow_insulation (one-layer snow thermal node) is "
packages/land/legoesm/land/multilayer_land.py:601:                "config.thermal.snow_insulation is the one-layer thermal node of "
packages/land/legoesm/land/multilayer_land.py:633:    if config.canopy_snow_masking and (
packages/land/legoesm/land/multilayer_land.py:639:            "canopy_snow_masking is implemented only for the two-leaf canopy with "
packages/land/legoesm/land/multilayer_land.py:642:    if config.canopy_snow_masking and config.snow_scheme == "layered":
packages/land/legoesm/land/multilayer_land.py:648:            "canopy_snow_masking is not supported with snow_scheme='layered' yet: "
packages/land/legoesm/land/multilayer_land.py:879:        # the wet-leaf evaporation INSIDE the canopy energy balance (interception
packages/land/legoesm/land/multilayer_land.py:881:        # ``None`` when interception is off — the canopy then runs the pure-
packages/land/legoesm/land/multilayer_land.py:884:        if config.interception is not None and state.W_canopy is not None:
packages/land/legoesm/land/multilayer_land.py:890:            _fwet_pre = interception_wetted_fraction(
packages/land/legoesm/land/multilayer_land.py:891:                state.W_canopy, _pai_i, config.interception)
packages/land/legoesm/land/multilayer_land.py:922:            if config.canopy_snow_masking:
packages/land/legoesm/land/multilayer_land.py:1367:    # --- Canopy interception, phase 1: intercept rain into the store ----------
packages/land/legoesm/land/multilayer_land.py:1373:    # latent stream).  See land/canopy/interception.py.
packages/land/legoesm/land/multilayer_land.py:1376:    # never intercepts water it cannot store (which would leak): the store is
packages/land/legoesm/land/multilayer_land.py:1377:    # allocated at init / restored from restart when interception is configured,
packages/land/legoesm/land/multilayer_land.py:1379:    _do_intercept = (config.interception is not None
packages/land/legoesm/land/multilayer_land.py:1385:    if _do_intercept:
packages/land/legoesm/land/multilayer_land.py:1391:        _W_int, _throughfall = intercept_rain(
packages/land/legoesm/land/multilayer_land.py:1393:            config.interception)
packages/land/legoesm/land/multilayer_land.py:1402:        _cap = max_canopy_water(_pai, config.interception)
packages/land/legoesm/land/multilayer_land.py:1451:    # not raw precip, since interception removed the intercepted part upstream.
packages/land/legoesm/land/multilayer_land.py:1491:    # --- Canopy interception, phase 2: deplete the store by the wet-leaf flux --
packages/land/legoesm/land/multilayer_land.py:1494:    # interception-loss latent flux, NOT re-labelled transpiration).  That water
packages/land/legoesm/land/multilayer_land.py:1502:    if _do_intercept:
packages/land/legoesm/land/multilayer_land.py:1791:        # out (updated on the two-leaf interception path, else carried), a
packages/land/legoesm/land/multilayer_land.py:1792:        # ``None`` in stays ``None``.  ``_do_intercept`` already requires the
packages/land/legoesm/land/multilayer_land.py:1793:        # store to exist, so an active interception step never introduces the
packages/land/legoesm/land/multilayer_land.py:1874:            if config.canopy_snow_masking:
packages/land/legoesm/land/multilayer_land.py:2436:        # Dry canopy at start; carried only when interception is configured.
packages/land/legoesm/land/multilayer_land.py:2437:        W_canopy=(jnp.zeros(ncol) if config.interception is not None else None),
packages/land/legoesm/land/multilayer_land.py:2449:        T_snow=(T_soil[:, 0] if config.thermal.snow_insulation else None),
   690	     !
   691	     ! !LOCAL VARIABLES:
   692	     integer  :: fp,p             ! indices
   693	     real(r8) :: vegt             ! lsai
   694	     real(r8) :: dewmxi           ! inverse of maximum allowed dew [1/mm]
   695	     !-----------------------------------------------------------------------
   696	
   697	     associate(                                              & 
   698	          frac_veg_nosno => canopystate_inst%frac_veg_nosno_patch , & ! Input:  [integer (:)]  fraction of veg not covered by snow (0/1 now) [-]
   699	          dewmx          => canopystate_inst%dewmx_patch          , & ! Input:  [real(r8) (:) ]  Maximum allowed dew [mm]                
   700	          elai           => canopystate_inst%elai_patch           , & ! Input:  [real(r8) (:) ]  one-sided leaf area index with burying by snow
   701	          esai           => canopystate_inst%esai_patch           , & ! Input:  [real(r8) (:) ]  one-sided stem area index with burying by snow
   702	
   703	          h2ocan         => waterstate_inst%h2ocan_patch          , & ! Input:  [real(r8) (:) ]  total canopy water (mm H2O)             
   704	          snocan         => waterstate_inst%snocan_patch          , & ! Output: [real(r8) (:)   ]  canopy snow (mm H2O)             
   705	          liqcan         => waterstate_inst%liqcan_patch          , & ! Output: [real(r8) (:)   ]  canopy liquid (mm H2O)
   706	
   707	          fwet           => waterstate_inst%fwet_patch            , & ! Output: [real(r8) (:) ]  fraction of canopy that is wet (0 to 1) 
   708	          fcansno        => waterstate_inst%fcansno_patch         , & ! Output: [real(r8) (:) ]  fraction of canopy that is snow covered (0 to 1) 
   709	          fdry           => waterstate_inst%fdry_patch              & ! Output: [real(r8) (:) ]  fraction of foliage that is green and dry [-] (new)
   710	          )
   711	
   712	       ! Set status of snowveg_flag
   713	       snowveg_onrad = IsSnowvegFlagOnRad()
   714	
   715	       do fp = 1,numf
   716	          p = filter(fp)
   717	          if (frac_veg_nosno(p) == 1) then
   718	             if (h2ocan(p) > 0._r8) then
   719	                vegt    = frac_veg_nosno(p)*(elai(p) + esai(p))
   720	                dewmxi  = 1.0_r8/dewmx(p)
   721	                fwet(p) = ((dewmxi/vegt)*h2ocan(p))**0.666666666666_r8
   722	                fwet(p) = min (fwet(p),maximum_leaf_wetted_fraction)   ! Check for maximum limit of fwet
   723	                if (snowveg_onrad) then
   724	                   if (snocan(p) > 0._r8) then
   725	                      dewmxi  = 1.0_r8/dewmx(p)
   726	                      fcansno(p) = ((dewmxi/(vegt*6.0_r8*10.0_r8))*snocan(p))**0.15_r8 ! must match snocanmx 
   727	                      fcansno(p) = min (fcansno(p),1.0_r8)
   728	                   else
   729	                      fcansno(p) = 0._r8
   730	                   end if
   731	                else
   732	                   fcansno(p) = 0._r8
   733	                end if
   734	             else
   735	                fwet(p) = 0._r8
   736	                fcansno(p) = 0._r8
   737	             end if
   738	             fdry(p) = (1._r8-fwet(p))*elai(p)/(elai(p)+esai(p))
   739	          else
   740	             fwet(p) = 0._r8
   741	             fdry(p) = 0._r8
   742	          end if
   743	       end do

 succeeded in 34ms:
   790	        # column is only valid for a PURE-sigma grid, where p = sigma*p_s makes
   791	        # the level sigmas identical in every column.  On the (default) hybrid
   792	        # grid p = A p_ref + B p_s, so the bracketing levels genuinely differ
   793	        # per column and that shortcut selects the wrong pair over terrain.
   794	        idx_hi = np.sum(p_model[..., None] < plev_target, axis=-2)
   795	        idx_hi = np.clip(idx_hi, 1, p_model.shape[-1] - 1)
   796	        idx_lo = idx_hi - 1
   797	
   798	        f_lo = np.take_along_axis(field_np, idx_lo, axis=-1)
   799	        f_hi = np.take_along_axis(field_np, idx_hi, axis=-1)
   800	        lp_lo = np.take_along_axis(log_p_model, idx_lo, axis=-1)
   801	        lp_hi = np.take_along_axis(log_p_model, idx_hi, axis=-1)
   802	
   803	        log_pt = log_plev.reshape((1,) * p_s_np.ndim + (n_target,))
   804	        denom = np.where(lp_hi == lp_lo, 1.0, lp_hi - lp_lo)
   805	        alpha = np.clip((log_pt - lp_lo) / denom, 0.0, 1.0)
   806	        return f_lo + alpha * (f_hi - f_lo)
   807	
   808	    def _tas_2m(self, state, q_v, sst, sic, T_ice, u_low=None, v_low=None,
   809	                *, T_land=None, q_land=None, land_fraction=None):
   810	        """2 m air temperature for CMIP ``tas`` from the MOST surface-layer
   811	        similarity profile (interpolate the lowest model level down to 2 m).
   812	        Returns the lowest-level T when the surface inputs (SST / sigma) are
   813	        unavailable — e.g. a prescribed-SST run that does not pass SST here.
   814	
   815	        ``T_land`` / ``land_fraction`` (keyword-only) make the diagnostic
   816	        LAND-AWARE.  Without them the surface is the blended SST/sea-ice skin
   817	        with a saturated humidity EVERYWHERE, so the published ``tas`` over
   818	        land is the neighbouring ocean's temperature rather than the model's
   819	        land — the reason a February Arctic land bias could not be read from
   820	        this field.  There is no below-canopy 2 m profile in this code, so the
   821	        land branch uses the same MOST profile anchored on the land skin; do
   822	        not read the result as a sub-canopy screen temperature.
   823	
   824	        ``u_low`` / ``v_low`` override the lowest-level winds (lets the MPAS
   825	        path pass the reconstructed CELL winds from
   826	        ``reconstruct_cell_velocity`` instead of the edge-normal ``state.u``,
   827	        which is not cell-collocated); default reads them off ``state``."""
   828	        T_low = state.T.data[..., -1]
   829	        if sst is None or self.sigma_full is None:
   830	            return T_low
   831	        import jax.numpy as jnp
   832	        from legoesm import constants
   833	        from legoesm.thermo import saturation_mixing_ratio
   834	        from legoesm.core.bulk_flux import compute_most_fluxes
   835	        u_low = state.u.data[..., -1] if u_low is None else u_low
   836	        v_low = state.v.data[..., -1] if v_low is None else v_low
   837	        q_low = q_v[..., -1] if q_v is not None else jnp.zeros_like(T_low)
   838	        p_s = state.p_s.data
   839	        p_low = jnp.asarray(self._p_full(p_s))[..., -1]
   840	        rho_low = p_low / (constants.R_d * T_low)
   841	        # Similarity profile matched to the experiment's surface fluxes:
   842	        # SAME bulk scheme (constant -> the historical coare3 stand-in, see
   843	        # __init__) and SAME stable-branch selector (codex 2026-08-02: a
   844	        # large_yeager or grachev/gryanik run otherwise published a
   845	        # coare3-native-default tas).  The 2 m value is set by stability, so
   846	        # gustiness is left scheme-native here (pre-existing choice).
   847	        def _profile_to_2m(T_sfc, q_sfc, *, ocean: bool):
   848	            """One surface's 2 m temperature.
   849	
   850	            MERGE NOTE (#1773), and it is a physics decision, not a textual
   851	            one. ``main`` added a ``large_yeager_cesm`` arm that returns CESM
   852	            ``shr_flux_atmOcn``'s own ``tref`` instead of the MOST profile;
   853	            this branch split the diagnostic into a LAND surface and a SEA
   854	            surface. ``shr_flux_atmOcn`` is the atmosphere-OCEAN coupler law
   855	            (the module's own docstring: "SAM ocean surface fluxes"), with
   856	            Charnock roughness and a saturated surface, so handing it a land
   857	            skin would re-create the exact artifact this branch exists to
   858	            remove -- land ``tas`` computed as if the surface were ocean.
   859	
   860	            So the CESM arm applies to the SEA leg ONLY; the land leg keeps
   861	            the MOST profile on every scheme. Review (GLM) flagged the
   862	            alternative -- dispatching for both legs -- as a third scheme
   863	            neither side tested.
   864	            """
   865	            if ocean and self.tas_profile_scheme == "large_yeager_cesm":
   866	                # CESM shr_flux_atmOcn's own ``tref`` diagnostic (2 m).
   867	                from legoesm.core.bulk_flux import compute_sam_oceflx_fluxes
   868	                *_, T_2m = compute_sam_oceflx_fluxes(
   869	                    u_low, v_low, T_low, q_low, T_sfc, q_sfc, rho_low,
   870	                    z_bot=10.0, variant="cesm", return_2m=True,
   871	                    t_bot=T_low,
   872	                )
   873	                return T_2m
   874	            *_, T_2m = compute_most_fluxes(
   875	                u_low, v_low, T_low, q_low, T_sfc, q_sfc, rho_low,
   876	                scheme=(self.tas_profile_scheme
   877	                        if self.tas_profile_scheme != "large_yeager_cesm"
   878	                        else "large_yeager"),
   879	                return_2m=True,
   880	                stability_scheme=self.surface_stability_scheme,
   881	            )
   882	            return T_2m
   883	
   884	        T_sfc_sea = blend_surface_temperature(sst, sic, T_ice)
   885	        q_sfc_sea = saturation_mixing_ratio(T_sfc_sea, p_s)
   886	        if T_land is None or land_fraction is None:
   887	            # Legacy path, bit-identical: ONE profile anchored on the
   888	            # ocean/sea-ice skin with a saturated surface.  Over land that is
   889	            # the neighbouring ocean's temperature and a saturated surface the
   890	            # land does not have, so a caller that owns a land skin should pass
   891	            # it — see the land branch below.
   892	            return _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)
   893	
   894	        # Land and sea get their OWN surface, then combine by land fraction.
   895	        # Both branches keep the same bulk scheme and stability selector, so
   896	        # this introduces no second convention — only the missing surface.
   897	        # ``q_land`` is a surface mixing ratio; without one the land surface is
   898	        # treated as saturated, which is what the legacy path assumed
   899	        # everywhere and is an OVERESTIMATE of surface humidity over dry or
   900	        # frozen ground.
   901	        q_sfc_land = (saturation_mixing_ratio(T_land, p_s) if q_land is None
   902	                      else q_land)
   903	        T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False)
   904	        T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)
   905	        f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),
   906	                          0.0, 1.0)
   907	        return f_land * T_2m_land + (1.0 - f_land) * T_2m_sea
   908	
   909	    def _cloud_config_for(self, cloud_fraction):
   910	        """The collector's cloud config, with the cloud-fraction override over
   911	        the WHOLE column when an advected prognostic fraction (IFS CLOUDSC)
   912	        is handed in, as radiation applies it (votes VC24, VC40)."""
   913	        if (cloud_fraction is None
   914	                or getattr(self._cloud_config, "scheme", None) == "cam6_clubb"):
   915	            return self._cloud_config
   916	        return self._cloud_config._replace(clubb_cf_override_p_min_pa=0.0)
   917	
   918	    def _cam6_cloud_kwargs(self, cloud_fraction, conv_mass_flux_up,
   919	                           conv_icwmr, lat_deg, p_s, ncol, nlev):
   920	        """Extra ``compute_cloud_properties`` kwargs for cloud_scheme
   921	        'cam6_clubb': the CLUBB cloud fraction + deep-convection carries, lat
   922	        [rad] and interface pressures.  For any other scheme: the advected
   923	        IFS cloud fraction as the override when one is handed in (VC40), else
   924	        empty (byte-identical call); a cam6 run without the carry raises here rather than
   925	        silently scoring an RH cloud cover the radiation never used.
   926	
   927	        SAMPLING: the carry handed in is the one written by THIS step's
   928	        physics (CLUBB's current PDF fraction), applied to the current state;
   929	        the radiation call read the carry as it stood at its last update
   930	        (one physics step behind, more under radiation subcycling).  clt/clivi
1554:def _make_hydrostatic_radiation(
1665:    def physics_fn(state, grid_or_mesh, sigma_coord,
1824:        q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col = (
1964:            if q_cloud_col is not None:
1965:                q_cloud_col = shard_columns(q_cloud_col, column_mesh)
1966:            if q_ice_col is not None:
1967:                q_ice_col = shard_columns(q_ice_col, column_mesh)
1994:                q_cloud_col=q_cloud_col, q_ice_col=q_ice_col, cf_col=_cf_ovr,
2011:            q_cloud=q_cloud_col,
2012:            q_ice=q_ice_col,
2069:                q_cloud=None,
2070:                q_ice=None,
2151:def _ecrad_ifs_sources(full_call, tsi, eccf, o3_vmr, ghg_vmr, dtype):
2181:def _make_ecrad_ifs_step(radiation_config, full_call, column_mesh):
2184:    albedo/emissivity and the condensate; it reads q_r, q_s, the land
2194:    def run(state, grid_or_mesh, phys_state, forcing, shape_3d, shape_2d, *,
2197:            q_cloud_col, q_ice_col, cf_col, eccf, tsi, o3_vmr_col, ghg_vmr):
2198:        _need = ("q_v", "q_c", "q_i", "q_r", "q_s", "cloud_fraction_ifs")
2226:        q_r, q_s = _col(state.tracers["q_r"]), _col(state.tracers["q_s"])
2236:            q_r, q_s, alb, alb_dir, emis, land = (
2238:                for x in (q_r, q_s, alb, alb_dir, emis, land))
2271:            lat=lat_col, lon=lon_col, q=q_v_col, q_liquid=q_cloud_col,
2272:            q_ice=q_ice_col, q_rain=q_r, q_snow=q_s, cloud_fraction=cf_col,
2301:def _make_nonhydrostatic_radiation(
2313:    def physics_fn(
2382:        # Cloud condensate from tracer slots 1 (q_c) and 3 (q_i).
2383:        q_cloud_col = None
2384:        q_ice_col = None
2386:            q_cloud_col = jnp.clip(
2390:            q_ice_col = jnp.clip(
2394:        # canonical 9-slot double-moment layout (q_v,q_c,q_r,q_i,q_s,q_g,
2432:            q_cloud=q_cloud_col,
2433:            q_ice=q_ice_col,
2496:def _mc3d_plane_heating(
2507:    q_cloud_col,
2508:    q_ice_col,
2570:            config=cloud_config, q_cloud=q_cloud_col, q_ice=q_ice_col,
2681:def _make_plane_radiation(
2699:    def physics_fn(
2751:        # NH plane state stores water vapor in tracer slot 0; q_c at 1,
2752:        # q_i at 3 — mirroring the cubed-sphere NH tracer layout used
2761:        q_cloud_col = None
2762:        q_ice_col = None
2764:            q_cloud_col = jnp.clip(
2768:            q_ice_col = jnp.clip(
2774:        # (q_v,q_c,q_r,q_i,q_s,q_g,N_c,N_r,N_i) shared by ALL double-moment
2795:                q_cloud_col, q_ice_col, n_cloud_col, n_ice_col,
2806:                q_cloud=q_cloud_col, q_ice=q_ice_col,
2866:def _make_mpas_nh_radiation(
2885:    def physics_fn(
2939:        #   [0] q_v   [1] q_c   [2] q_r   [3] q_i
2941:        # have access to q_c (slot 1) AND q_i (slot 3); otherwise a
2955:                f"at least 4 slots (q_v, q_c, q_r, q_i); MPAS NH "
2965:        q_cloud_col = None
2966:        q_ice_col = None
2968:            q_cloud_col = jnp.clip(
2972:            q_ice_col = jnp.clip(
2998:            q_cloud=q_cloud_col, q_ice=q_ice_col,
3051:def _make_spectral_pe_radiation(
3106:    def _physics_fn_core(
3234:        q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col = (
3246:        def _to_col(x):
3297:            q_cloud=q_cloud_col,
3298:            q_ice=q_ice_col,
3350:        def physics_fn(
   875	   Nnite = 0
   876	   do i = 1, ncol
   877	      if ( coszrs(i) > 0.0_r8 ) then
   878	         Nday = Nday + 1
   879	         IdxDay(Nday) = i
   880	      else
   881	         Nnite = Nnite + 1
   882	         IdxNite(Nnite) = i
   883	      end if
   884	   end do
   885	
   886	   ! Associate pointers to physics buffer fields
   887	   itim_old = pbuf_old_tim_idx()
   888	   if (cldfsnow_idx > 0) then
   889	      call pbuf_get_field(pbuf, cldfsnow_idx, cldfsnow, start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
   890	   endif
   891	   call pbuf_get_field(pbuf, cld_idx, cld, start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
   892	
   893	   call pbuf_get_field(pbuf, qrs_idx, qrs)
   894	   call pbuf_get_field(pbuf, qrl_idx, qrl)
   895	
   896	   call pbuf_get_field(pbuf, fsnt_idx, fsnt)
   897	   call pbuf_get_field(pbuf, fsds_idx, fsds)
   898	   call pbuf_get_field(pbuf, fsns_idx, fsns)
   899	   call pbuf_get_field(pbuf, flns_idx, flns)
   900	   call pbuf_get_field(pbuf, flnt_idx, flnt)
   901	
   902	   if (spectralflux) then
   903	      call pbuf_get_field(pbuf, su_idx, su)
   904	      call pbuf_get_field(pbuf, sd_idx, sd)
   905	      call pbuf_get_field(pbuf, lu_idx, lu)
   906	      call pbuf_get_field(pbuf, ld_idx, ld)
   907	   end if
   908	
   909	   !  For CRM, make cloud equal to input observations:
   910	   if (scm_crm_mode .and. have_cld) then
   911	      do k = 1, pver
   912	         cld(:ncol,k)= cldobs(k)
   913	      end do
   914	   end if
   915	
   916	   ! Find tropopause height if needed for diagnostic output
   917	   if (hist_fld_active('FSNR') .or. hist_fld_active('FLNR')) then
   918	      call tropopause_find(state, troplev, tropP=p_trop, primary=TROP_ALG_HYBSTOB, backup=TROP_ALG_CLIMATE)
   919	   endif
   920	
   921	   if (dosw .or. dolw) then
   922	
   923	      ! construct an RRTMG state object
   924	      r_state = rrtmg_state_create( state, cam_in )
   925	
   926	      call t_startf('cldoptics')
   927	
   928	      if (cldfsnow_idx > 0) then
   929	         do k = 1, pver
   930	            do i = 1, ncol
   931	               cldfprime(i,k) = max(cld(i,k), cldfsnow(i,k))
   932	            end do
   933	         end do
   934	      else
   935	         cldfprime(:ncol,:) = cld(:ncol,:)
   936	      end if
   937	      
   938	      
   939	      if (dosw) then
   940	
   941	         if (oldcldoptics) then
   942	            call ec_ice_optics_sw(state, pbuf, ice_tau, ice_tau_w, ice_tau_w_g, ice_tau_w_f, oldicewp=.false.)
   943	            call slingo_liq_optics_sw(state, pbuf, liq_tau, liq_tau_w, liq_tau_w_g, liq_tau_w_f, oldliqwp=.false.)
   944	         else
   945	            select case (icecldoptics)
   946	            case ('ebertcurry')
   947	               call  ec_ice_optics_sw(state, pbuf, ice_tau, ice_tau_w, ice_tau_w_g, ice_tau_w_f, oldicewp=.true.)
   948	            case ('mitchell')
   949	               call get_ice_optics_sw(state, pbuf, ice_tau, ice_tau_w, ice_tau_w_g, ice_tau_w_f)
   950	            case default
   951	               call endrun('iccldoptics must be one either ebertcurry or mitchell')
   952	            end select
   953	
   954	            select case (liqcldoptics)
   955	            case ('slingo')
   956	               call slingo_liq_optics_sw(state, pbuf, liq_tau, liq_tau_w, liq_tau_w_g, liq_tau_w_f, oldliqwp=.true.)
   957	            case ('gammadist')
   958	               call get_liquid_optics_sw(state, pbuf, liq_tau, liq_tau_w, liq_tau_w_g, liq_tau_w_f)
   959	            case default
   960	               call endrun('liqcldoptics must be either slingo or gammadist')
   961	            end select
   962	         end if
   963	
   964	         cld_tau(:,:ncol,:)     =  liq_tau(:,:ncol,:)     + ice_tau(:,:ncol,:)
   965	         cld_tau_w(:,:ncol,:)   =  liq_tau_w(:,:ncol,:)   + ice_tau_w(:,:ncol,:)
   966	         cld_tau_w_g(:,:ncol,:) =  liq_tau_w_g(:,:ncol,:) + ice_tau_w_g(:,:ncol,:)
   967	         cld_tau_w_f(:,:ncol,:) =  liq_tau_w_f(:,:ncol,:) + ice_tau_w_f(:,:ncol,:)
   968	 
   969	         if (cldfsnow_idx > 0) then
   970	            ! add in snow
   971	            call get_snow_optics_sw(state, pbuf, snow_tau, snow_tau_w, snow_tau_w_g, snow_tau_w_f)
   972	            do i = 1, ncol
   973	               do k = 1, pver
   974	
   975	                  if (cldfprime(i,k) > 0._r8) then
   976	
   977	                     c_cld_tau(:,i,k)     = ( cldfsnow(i,k)*snow_tau(:,i,k) &
   978	                                             + cld(i,k)*cld_tau(:,i,k) )/cldfprime(i,k)
   979	
   980	                     c_cld_tau_w(:,i,k)   = ( cldfsnow(i,k)*snow_tau_w(:,i,k)  &
   981	                                             + cld(i,k)*cld_tau_w(:,i,k) )/cldfprime(i,k)
   982	
   983	                     c_cld_tau_w_g(:,i,k) = ( cldfsnow(i,k)*snow_tau_w_g(:,i,k) &
   984	                                             + cld(i,k)*cld_tau_w_g(:,i,k) )/cldfprime(i,k)
   985	
   986	                     c_cld_tau_w_f(:,i,k) = ( cldfsnow(i,k)*snow_tau_w_f(:,i,k) &
   987	                                             + cld(i,k)*cld_tau_w_f(:,i,k) )/cldfprime(i,k)
   988	                  else
   989	                     c_cld_tau(:,i,k)     = 0._r8
   990	                     c_cld_tau_w(:,i,k)   = 0._r8
   991	                     c_cld_tau_w_g(:,i,k) = 0._r8
   992	                     c_cld_tau_w_f(:,i,k) = 0._r8
   993	                  end if
   994	               end do
   995	            end do
  2290	
  2291	   do k = top_lev, pver
  2292	      do i = 1, ncol
  2293	         ! Limits for in-cloud mixing ratios consistent with MG microphysics
  2294	         ! in-cloud mixing ratio maximum limit of 0.005 kg/kg
  2295	         icimrst(i,k)   = min( state_loc%q(i,k,ixcldice) / max(mincld,icecldf(i,k)),0.005_r8 )
  2296	         icwmrst(i,k)   = min( state_loc%q(i,k,ixcldliq) / max(mincld,liqcldf(i,k)),0.005_r8 )
  2297	         icinc(i,k)     = state_loc%q(i,k,ixnumice) / max(mincld,icecldf(i,k)) * &
  2298	              state_loc%pmid(i,k) / (287.15_r8*state_loc%t(i,k))
  2299	         icwnc(i,k)     = state_loc%q(i,k,ixnumliq) / max(mincld,liqcldf(i,k)) * &
  2300	              state_loc%pmid(i,k) / (287.15_r8*state_loc%t(i,k))
  2301	         ! Calculate micro_mg_cam cloud water paths in each layer
  2302	         ! Note: uses stratiform cloud fraction!
  2303	         iciwpst(i,k)   = min(state_loc%q(i,k,ixcldice)/max(mincld,ast(i,k)),0.005_r8) * state_loc%pdel(i,k) / gravit
  2304	         iclwpst(i,k)   = min(state_loc%q(i,k,ixcldliq)/max(mincld,ast(i,k)),0.005_r8) * state_loc%pdel(i,k) / gravit
  2305	
  2306	         ! ------------------------------ !
  2307	         ! Adjust cloud fraction for snow !
  2308	         ! ------------------------------ !
  2309	         cldfsnow(i,k) = cld(i,k)
  2310	         ! If cloud and only ice ( no convective cloud or ice ), then set to 0.
  2311	         if( ( cldfsnow(i,k) .gt. 1.e-4_r8 ) .and. &
  2312	            ( concld(i,k)   .lt. 1.e-4_r8 ) .and. &
  2313	            ( state_loc%q(i,k,ixcldliq) .lt. 1.e-10_r8 ) ) then
  2314	            cldfsnow(i,k) = 0._r8
  2315	         end if
  2316	         ! If no cloud and snow, then set to 0.25
  2317	         if( ( cldfsnow(i,k) .le. 1.e-4_r8 ) .and. ( qsout(i,k) .gt. 1.e-6_r8 ) ) then
  2318	            cldfsnow(i,k) = 0.25_r8
  2319	         end if
  2320	         ! Calculate in-cloud snow water path
  2321	         icswp(i,k) = qsout(i,k) / max( mincld, cldfsnow(i,k) ) * state_loc%pdel(i,k) / gravit
  2322	      end do
  2323	   end do
  2324	
  2325	   ! Calculate cloud fraction for prognostic precip sizes.
  2326	   if (micro_mg_version > 1) then
  2327	      ! Cloud fraction for purposes of precipitation is maximum cloud
  2328	      ! fraction out of all the layers that the precipitation may be
  2329	      ! falling down from.
  2330	      cldmax(:ncol,:) = max(mincld, ast(:ncol,:))

codex
C6’s flat-albedo premise is refuted: the current production deck selects melt-dependent sea-ice albedo. C1’s unmasked snow export is verified, but code inspection alone cannot establish it as the main cause of the annual cold bias.
exec
/usr/bin/bash -lc "nl -ba /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90 | sed -n '712,740p'; nl -ba packages/land/legoesm/land/state.py | sed -n '20,108p'; nl -ba packages/land/legoesm/land/multilayer_land.py | sed -n '655,740p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '2020,2068p;14270,14294p'; rg -n 'def compute_most_fluxes|z_ref:|z_ref=|T_2m|tref|def.*large_yeager' packages/core/legoesm/core/bulk_flux.py" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
   712	       ! Set status of snowveg_flag
   713	       snowveg_onrad = IsSnowvegFlagOnRad()
   714	
   715	       do fp = 1,numf
   716	          p = filter(fp)
   717	          if (frac_veg_nosno(p) == 1) then
   718	             if (h2ocan(p) > 0._r8) then
   719	                vegt    = frac_veg_nosno(p)*(elai(p) + esai(p))
   720	                dewmxi  = 1.0_r8/dewmx(p)
   721	                fwet(p) = ((dewmxi/vegt)*h2ocan(p))**0.666666666666_r8
   722	                fwet(p) = min (fwet(p),maximum_leaf_wetted_fraction)   ! Check for maximum limit of fwet
   723	                if (snowveg_onrad) then
   724	                   if (snocan(p) > 0._r8) then
   725	                      dewmxi  = 1.0_r8/dewmx(p)
   726	                      fcansno(p) = ((dewmxi/(vegt*6.0_r8*10.0_r8))*snocan(p))**0.15_r8 ! must match snocanmx 
   727	                      fcansno(p) = min (fcansno(p),1.0_r8)
   728	                   else
   729	                      fcansno(p) = 0._r8
   730	                   end if
   731	                else
   732	                   fcansno(p) = 0._r8
   733	                end if
   734	             else
   735	                fwet(p) = 0._r8
   736	                fcansno(p) = 0._r8
   737	             end if
   738	             fdry(p) = (1._r8-fwet(p))*elai(p)/(elai(p)+esai(p))
   739	          else
   740	             fwet(p) = 0._r8
    20	    snow_depth: Field      # Snow water equivalent [kg/m2]
    21	    snow_age: Field        # Time since last snowfall [s]
    22	    runoff: jax.Array | None = None  # Surface runoff [kg/m2/s]
    23	    # 30-day exponential moving average of near-surface air temperature
    24	    # in [°C] — feeds the Leuning Vcmax growth-temperature acclimation
    25	    # term when the two-leaf canopy surface scheme is active.  ``None``
    26	    # by default; populated by ``init_*_land_state`` and advanced each
    27	    # step.  Independent of any externally prescribed
    28	    # ``CanopyLandParams.TgC`` (which takes precedence when both are set).
    29	    TgC: jax.Array | None = None
    30	
    31	
    32	class MultiLayerLandState(NamedTuple):
    33	    """Multi-layer land state (Task 8).
    34	
    35	    2D fields have shape (ncol,).
    36	    3D fields have shape (ncol, n_soil_layers).
    37	    """
    38	    T_soil: jax.Array          # Soil temperature [K], (ncol, n_soil_layers)
    39	    psi_soil: jax.Array        # Soil matric potential [m], (ncol, n_soil_layers)
    40	    theta_soil: jax.Array      # Volumetric water content [m3/m3], (ncol, n_soil_layers)
    41	    runoff_surface: jax.Array  # Surface runoff [kg/m2/s], (ncol,)
    42	    runoff_subsurface: jax.Array  # Subsurface runoff [kg/m2/s], (ncol,)
    43	    snow_depth: jax.Array      # Snow water equivalent [kg/m2], (ncol,)
    44	    snow_age: jax.Array        # Time since last snowfall [s], (ncol,)
    45	    # 30-day exponential moving average of near-surface air temperature
    46	    # in [°C] — see ``LandState.TgC``.  Optional; ``None`` by default.
    47	    TgC: jax.Array | None = None
    48	    # Surface ponding depth [m], (ncol,) — a coupled surface cell (ParFlow/CliMA
    49	    # overland store): excess precip ponds, infiltrates on later steps, and
    50	    # overflows to runoff above ``pond_max``.  ``None`` (legacy) is zero ponding.
    51	    surface_water: jax.Array = None
    52	    # Sub-grid elevation-band SWE [kg/m2], (ncol, n_bands) — present iff the
    53	    # elevation-band snow scheme is enabled (``config.elev_bands``); ``None``
    54	    # (legacy) runs the single cell-mean snowpack.  ``snow_depth`` always carries
    55	    # the area-weighted aggregate for diagnostics/restart compatibility.
    56	    snow_bands: jax.Array = None
    57	    # Per-band snow age [s], (ncol, n_bands) — companion to ``snow_bands`` (fresh
    58	    # snow rejuvenates only its own band's albedo while perennial firn keeps aging).
    59	    # ``None`` (legacy) uses the single cell-mean ``snow_age`` clock.
    60	    snow_age_bands: jax.Array = None
    61	    # Sub-grid elevation-band firn/glacier-ice reservoir [kg/m2], (ncol, n_bands) —
    62	    # companion to ``snow_bands`` (gap 4): seasonal snow above the snow cap firnifies
    63	    # into this store, which discharges slowly (delayed glacier outflow) and exposes
    64	    # dark ablation ice where the seasonal snow melts off.  ``None`` (legacy / bands
    65	    # off) has no perennial-ice reservoir.
    66	    ice_bands: jax.Array = None
    67	    # Prognostic state for the CLM-ML-JAX multilayer canopy scheme.
    68	    # Holds the ``mlcanopy_type`` instance carried forward between steps.
    69	    # ``None`` when the CLM-ML canopy scheme is not active.
    70	    canopy_state: Any | None = None
    71	    # Intercepted canopy water store [kg m-2], (ncol,) — the shared canopy
    72	    # interception scheme (``config.interception``; two-leaf / SimpleSEB path).
    73	    # ``None`` (legacy / interception off) carries no store.  The CLM-ML canopy
    74	    # keeps its own internal ``h2ocan`` store inside ``canopy_state`` and does
    75	    # NOT use this field.
    76	    W_canopy: jax.Array | None = None
    77	    # Warm-start cache for the two-leaf canopy Newton closure: the last
    78	    # CONVERGED solution per column, ``(ncol, 6)``, NaN where a column has never
    79	    # converged (which the canopy reads as "cold start").  NOT a prognostic
    80	    # variable — it carries no physics, only the seed the iteration starts from
    81	    # (the solve's adjoint returns a zero cotangent for it).  A seed outside the
    82	    # physical box is discarded for a cold start (``canopy_state_admissible``),
    83	    # since a far-off seed can reach a spurious root.  Present iff the surface
    84	    # scheme is the two-leaf canopy; ``None`` otherwise, which restores the
    85	    # cold-start-every-step behaviour exactly.  Appended last (positional-ABI).
    86	    canopy_x: Any | None = None
    87	    # Layered snowpack (``MultiLayerLandConfig.snow_scheme == "layered"``,
    88	    # ``legoesm.land.snow_column``): per-layer ice and liquid mass [kg/m2],
    89	    # temperature [K] and density [kg/m3], ``(ncol, n_snow_layers)``, top layer
    90	    # first.  ``snow_depth`` stays the TOTAL pack water (sum of ice + liquid), so
    91	    # albedo, snow cover and the latent partition read it unchanged.  ``None``
    92	    # for the bulk snowpack.
    93	    snow_ice_layers: jax.Array | None = None
    94	    snow_liq_layers: jax.Array | None = None
    95	    snow_T_layers: jax.Array | None = None
    96	    snow_rho_layers: jax.Array | None = None
    97	    # One-layer snow thermal node temperature [K], (ncol,): the SNOW-SURFACE
    98	    # temperature where the column holds snow, the top-soil temperature where
    99	    # it does not.  Present iff ``config.thermal.snow_insulation``; ``None``
   100	    # (default) keeps the legacy heat-free snow bucket.  Appended last.
   101	    T_snow: jax.Array | None = None
   655	            # radiative closure (two-leaf Kelvin RT, or CLM-ML's multilayer canopy
   656	            # RT — both carry their own T_rad/lw_up) would be inconsistent.  Reject
   657	            # rather than silently apply incompatible closures.  Applies to BOTH
   658	            # canopy schemes (the guard predates CLMMLCanopyConfig, which needs the
   659	            # same treatment as TwoLeafCanopyConfig).
   660	            raise ValueError(
   661	                "config.elev_bands (sub-grid snow bands) is not supported with the "
   662	                "TwoLeafCanopyConfig or CLMMLCanopyConfig surface scheme — the "
   663	                "banded radiation would override the canopy radiative closure.  "
   664	                "Use SimpleSEBConfig with bands.")
   665	        if state.snow_bands is None:
   666	            raise ValueError(
   667	                "config.elev_bands is set but state.snow_bands is None; initialise "
   668	                "the state with init_multilayer_land_state(config=...) so the banded "
   669	                "SWE field exists.")
   670	        ice_bands_in = (jnp.zeros_like(state.snow_bands) if state.ice_bands is None
   671	                        else state.ice_bands)
   672	        snowfall_bands = band_precip_snow(
   673	            forcing.T_lowest, forcing.precip_total, forcing.precip_snow, bands)
   674	        precip_snow_eff = jnp.mean(snowfall_bands, axis=-1)   # equal-area bands
   675	    else:
   676	        precip_snow_eff = forcing.precip_snow
   677	
   678	    # --- Layered snowpack (``snow_scheme == "layered"``, snow_column.py) ---
   679	    # Snowfall enters the pack at the START of the step, the pack is remapped and
   680	    # compacted, and the surface schemes see a skin temperature and a ground
   681	    # emissivity blended by the PHYSICAL snow-covered fraction f (unscaled
   682	    # snow_cover_fraction; only the albedo blend applies snow_cover_scale).
   683	    # The pack and the soil are then solved as ONE implicit column (the canopy
   684	    # Picard callback below and the final solve), with f*G entering the pack top
   685	    # and (1-f)*G the soil top; the pack's thickness lies on the covered area.
   686	    if config.snow_scheme not in ("bulk", "layered"):
   687	        raise ValueError(
   688	            f"Unknown MultiLayerLandConfig.snow_scheme {config.snow_scheme!r}; "
   689	            "expected 'bulk' or 'layered'.")
   690	    layered = config.snow_scheme == "layered"
   691	    if layered:
   692	        if bands is not None:
   693	            raise ValueError(
   694	                "snow_scheme='layered' is not supported with config.elev_bands "
   695	                "(the banded snowpack is a separate bulk scheme).")
   696	        if isinstance(config.surface_scheme, CLMMLCanopyConfig):
   697	            raise ValueError(
   698	                "snow_scheme='layered' is not supported with the CLM-ML canopy "
   699	                "scheme (it runs its own soil/snow boundary).")
   700	        if state.snow_T_layers is None:
   701	            raise ValueError(
   702	                "snow_scheme='layered' needs the snow-layer state; build the state "
   703	                "with init_multilayer_land_state(config=...) or seed_snow_layers().")
   704	        scc = config.snow_column
   705	        pack = SnowColumnState(swe_ice=state.snow_ice_layers,
   706	                               swe_liq=state.snow_liq_layers,
   707	                               T=state.snow_T_layers, density=state.snow_rho_layers)
   708	        # Enthalpy [J/m^2, relative to ice at T_freeze] carried INTO the pack by
   709	        # mass this step (snowfall, frost, rain) minus that leaving it (sublimated
   710	        # ice, drainage) -- the closure term reported as snow_advected_heat.
   711	        snow_advected_heat = (precip_snow_eff * dt * constants.c_pi
   712	                              * (jnp.minimum(forcing.T_lowest, constants.T_freeze)
   713	                                 - constants.T_freeze))
   714	        # CLM5 fresh-snow density and wind-drift compaction use the lowest-level
   715	        # wind speed (CTSM forc_wind), not the floored exchange wind below.
   716	        _wind_air = jnp.sqrt(jnp.maximum(
   717	            forcing.u_lowest ** 2 + forcing.v_lowest ** 2, 1e-12))   # AD-safe at calm
   718	        pack = snow_add_mass(pack, precip_snow_eff * dt, forcing.T_lowest,
   719	                             rho_fresh=new_snow_bulk_density(forcing.T_lowest, _wind_air))
   720	        pack = snow_remap_compact(pack, dt, _wind_air)
   721	        # Pack-top temperature the surface fluxes are evaluated at (the Robin
   722	        # linearisation point; sublimation and rain modify the pack afterwards).
   723	        T_pack_top_ref = pack.T[:, 0]
   724	        # PHYSICAL snow cover (CTSM frac_sno): the unscaled cover fraction.  The
   725	        # per-PFT snow_cover_scale is an albedo brightness calibration and stays
   726	        # in the albedo blend only (user 2026-10-03 / 2026-10-06).
   727	        f_snow = snow_cover_fraction(total_water(pack), config.land_albedo)
   728	        T_surface = _snow_skin(pack, T_soil[:, 0], f_snow)
   729	
   730	    # Smooth wind speed floor.
   731	    wind_speed = jnp.sqrt(
   732	        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + U_min ** 2)
   733	    wind_dir_x = forcing.u_lowest / jnp.maximum(wind_speed, 1e-6)
   734	    wind_dir_y = forcing.v_lowest / jnp.maximum(wind_speed, 1e-6)
   735	
   736	    # --- Snow phase (iter-68 — consistent with simple_seb's flux calc) ---
   737	    # Reuse the same warm-surface-snowfall gate as simple_seb.py so
   738	    # downstream latent-mass partition (sublimation vs soil evap) is
   739	    # consistent with the L_eff that produced the demand.  Ported from
   740	    # main during the jianing/land ↔ main sync 2026-06-03.  Uses the banded
  2020	    # calibration fitted only the calendar-clock scalars and never saw this
  2021	    # field, so it cannot be overwritten by it.
  2022	    _act = getattr(config, "snow_age_activation_K", None)
  2023	    if _act is not None:
  2024	        cfg = cfg._replace(
  2025	            land_albedo=cfg.land_albedo._replace(
  2026	                snow_age_activation_K=float(_act)))
  2027	    # Snow-age e-folding time, same placement and for the same reason: the
  2028	    # calibration re-apply above sets tau_snow_decay, so an override has to
  2029	    # land after it or it is silently discarded.
  2030	    _tau_d = getattr(config, "land_snow_tau_days", None)
  2031	    if _tau_d is not None:
  2032	        cfg = cfg._replace(
  2033	            land_albedo=cfg.land_albedo._replace(
  2034	                tau_snow_decay=float(_tau_d) * 86400.0))
  2035	        logger.info("  land snow-albedo age e-folding: %.3g days "
  2036	                    "(overrides the calibration)", float(_tau_d))
  2037	    # Soil freeze/thaw, same placement: the bake rebuilds ``thermal`` with
  2038	    # the switch at its library default, so it must be set after it.
  2039	    _ft = bool(config.land_soil_freeze_thaw)
  2040	    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=_ft))
  2041	    logger.info("  land soil freeze/thaw: %s", "ON" if _ft else "off")
  2042	    _csm = bool(config.land_canopy_snow_masking)
  2043	    cfg = cfg._replace(canopy_snow_masking=_csm)
  2044	    logger.info("  land canopy snow masking (trees hide snow): %s",
  2045	                "ON" if _csm else "off")
  2046	    # Snowpack scheme + snow emissivity, same placement (after the bake).
  2047	    cfg = cfg._replace(
  2048	        snow_scheme=str(config.land_snow_scheme),
  2049	        snow_column=cfg.snow_column._replace(
  2050	            emissivity_snow=float(config.land_snow_emissivity)))
  2051	    logger.info("  land snowpack: %s (snow emissivity %.3f)",
  2052	                cfg.snow_scheme, cfg.snow_column.emissivity_snow)
  2053	    from legoesm.land.multilayer_land import FINAL_THERMAL_SUBSTEPS
  2054	    logger.info("  land soil thermal sub-steps per land step: %d",
  2055	                FINAL_THERMAL_SUBSTEPS if _ft else 1)
  2056	    _e_ice = float(config.land_soil_ice_impedance_exponent)
  2057	    cfg = cfg._replace(
  2058	        richards=cfg.richards._replace(ice_impedance_exponent=_e_ice))
  2059	    logger.info("  land soil ice impedance exponent: %g (active only with "
  2060	                "freeze/thaw ON)", _e_ice)
  2061	    if not _ft and _e_ice != type(config)._field_defaults[
  2062	            "land_soil_ice_impedance_exponent"]:
  2063	        logger.warning("  land soil ice impedance exponent set but soil "
  2064	                       "freeze/thaw is off: it has no effect")
  2065	    # Soil-water numerics, same placement: the bake rebuilds ``hydraulics``
  2066	    # from the CLM map with library defaults, so set them after it.
  2067	    cfg = cfg._replace(
  2068	        hydraulics=cfg.hydraulics._replace(
 14270	            # silently (as a heating error, not a crash); refuse it here.
 14271	            if not bool(jnp.all(jnp.isfinite(_albedo_land_static))):
 14272	                raise ValueError(
 14273	                    "land surface albedo contains non-finite values — refusing "
 14274	                    "to hand a NaN surface boundary condition to radiation.")
 14275	            _alb_lo = float(jnp.min(_albedo_land_static))
 14276	            _alb_hi = float(jnp.max(_albedo_land_static))
 14277	            if not (0.0 <= _alb_lo and _alb_hi <= 1.0):
 14278	                raise ValueError(
 14279	                    f"land surface albedo out of physical range "
 14280	                    f"[{_alb_lo:.3f}, {_alb_hi:.3f}] — must lie in [0, 1].")
 14281	            logger.info(
 14282	                "  Surface albedo: ocean=%.3f ice=%.3f land=[%.3f,%.3f] "
 14283	                "mean=%.3f (f_land mean=%.3f)",
 14284	                _albedo_ocean, _albedo_ice, _alb_lo, _alb_hi,
 14285	                float(jnp.mean(_albedo_land_static)),
 14286	                float(jnp.mean(_f_land_cells)))
 14287	        elif _sfc_albedo_on:
 14288	            logger.info(
 14289	                "  Surface albedo: ocean=%.3f ice=%.3f (no land fraction)",
 14290	                _albedo_ocean, _albedo_ice)
 14291	
 14292	        # ---- Interactive multilayer (Richards) land tile — MPAS port -------
 14293	        # Phase-1 coupling contract (tasks/mpas_land_port.md): the land is
 14294	        # stepped OUTSIDE the jitted atmosphere step, once per dt, forced by
324:def large_yeager_neutral_cd(
1180:def large_yeager_widths(config) -> dict:
1346:def compute_most_fluxes(
1354:    z_ref=10.0,
1590:        ONLY when ``return_convergence=True`` (appended last, after ``T_2m``
1931:            T_2m = T_sfc - (theta_star / KAPPA) * denom_d
1952:            T_2m = T_atm - _coare["ts"] / KAPPA * prf
1956:        T_2m = jnp.clip(T_2m, lo, hi)
1958:            return (tau_x, tau_y, shflx, lhflx, u_star, T_2m, most_residual
1960:        return (tau_x, tau_y, shflx, lhflx, u_star, T_2m) + _wind_out
2130:      re-derived here).  With ``return_2m=True`` the CESM ``tref`` 2 m
2330:    # CESM ``tref`` (shr_flux_atmOcn diagnostics block): re-evaluate the
2331:    # stability function at ztref, integrate the theta profile from zbot down
2332:    # to ztref, then the 0.01 K/m potential-to-temperature correction.
2339:    tref = theta_atm - delt * fac - 0.01 * z_diag  # coeff-ok: CESM 0.01 K/m theta->T
2340:    return (tau_x, tau_y, shflx, lhflx, u_star, tref) + _wind_out

exec
/usr/bin/bash -lc "nl -ba packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | sed -n '600,675p;1805,1835p;1980,2025p'; rg -n 'cldfsnow_idx|snow_optics_lw|snow_tau' /glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90; rg -n 'snow|tau_snow|activation' packages/core/legoesm/surface_albedo.py packages/land/legoesm/land/snow_budget.py | head -70; rg -n 'tas_profile_scheme|tas.*2m|_tas|T_land=' packages/coupler/legoesm/driver/diagnostics.py | tail -30" in /glade/u/home/pg2328/legoesm
 succeeded in 0ms:
   600	    radii in ``compute_cloud_properties`` so RRTMGP gets droplet-number-aware
   601	    r_eff (not a fixed constant). They are ``None`` unless the state's tracer
   602	    dict carries ``"N_c"`` / ``"N_i"`` — single-moment schemes and
   603	    specified-Nc Morrison (``dopredictNc=.false.``) leave them absent, so those
   604	    fall back to the constant ``config.r_eff_liq`` / ``r_eff_ice`` (unchanged
   605	    behaviour). UNIT NOTE (corrected, #1715): the ``N_c`` CARRY is stored
   606	    per-MASS [#/kg] (checkpoint stamp ``number_convention = per_mass``) and is
   607	    converted to the per-VOLUME [#/m³] number ``compute_cloud_properties``
   608	    expects by multiplying with moist air density in
   609	    ``_extract_tracer_columns`` below; ``N_i`` is per-MASS and passed raw.
   610	    This docstring previously asserted the opposite convention for ``N_c``
   611	    ("passed RAW"), contradicting the conversion the function actually
   612	    performs — the same wrong prose that let the pipeline entry pass the
   613	    carry raw (#1715).
   614	    """
   615	    if dtype is None:
   616	        # Try to infer dtype from state.T or state.T_hat.  Fall back to
   617	        # the configured compute precision rather than hardcoding fp64
   618	        # — the latter would force radiation to allocate fp64 zeros on
   619	        # Metal/fp32 backends, which then upcast the whole RRTMGP
   620	        # column to fp64 via dtype promotion.
   621	        if hasattr(state, "T"):
   622	            dtype = state.T.data.dtype
   623	        elif hasattr(state, "T_hat"):
   624	            # Spectral state stores complex T_hat; the corresponding
   625	            # real-valued T column should match the spectral grid's
   626	            # real dtype.
   627	            dtype = state.T_hat.data.real.dtype
   628	        else:
   629	            dtype = get_policy().compute
   630	    T_col_shape = (ncol, nlev)
   631	    q_v_col = jnp.zeros(T_col_shape, dtype=dtype)
   632	    q_cloud_col = None
   633	    q_ice_col = None
   634	    n_cloud_col = None
   635	    n_ice_col = None
   636	
   637	    tracers = getattr(state, "tracers", None)
   638	    if tracers is not None:
   639	        if "q_v" in tracers:
   640	            _qv_raw = tracers["q_v"]
   641	            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
   642	            q_v_col = _qv_data.reshape(ncol, nlev)
   643	        if "q_c" in tracers:
   644	            _qc_raw = tracers["q_c"]
   645	            _qc_data = _qc_raw.data if hasattr(_qc_raw, "data") else _qc_raw
   646	            q_cloud_col = jnp.maximum(_qc_data.reshape(ncol, nlev), 0.0)
   647	        if "q_i" in tracers:
   648	            _qi_raw = tracers["q_i"]
   649	            _qi_data = _qi_raw.data if hasattr(_qi_raw, "data") else _qi_raw
   650	            q_ice_col = jnp.maximum(_qi_data.reshape(ncol, nlev), 0.0)
   651	        # Double-moment NUMBER columns (Morrison / Seifert-Beheng) → M2005 PSD
   652	        # r_eff.  Both are STORED per MASS [#/kg], so a dycore's mass-mixing-
   653	        # ratio advection is the right operator for them; the PSD formulas want
   654	        # N_c per VOLUME [#/m³], so convert here.  N_i is used per-mass and
   655	        # passes through.  Absent ⇒ None ⇒ constant-r_eff fallback.
   656	        if "N_c" in tracers:
   657	            if rho_col is None:
   658	                raise ValueError(
   659	                    "N_c is stored per MASS [#/kg]; converting it to the "
   660	                    "per-VOLUME number the cloud-optics PSD expects needs the "
   661	                    "air density. Pass rho_col to _extract_tracer_columns."
   662	                )
   663	            _nc_raw = tracers["N_c"]
   664	            _nc_data = _nc_raw.data if hasattr(_nc_raw, "data") else _nc_raw
   665	            n_cloud_col = jnp.maximum(
   666	                _nc_data.reshape(ncol, nlev) * rho_col, 0.0)
   667	        if "N_i" in tracers:
   668	            _ni_raw = tracers["N_i"]
   669	            _ni_data = _ni_raw.data if hasattr(_ni_raw, "data") else _ni_raw
   670	            n_ice_col = jnp.maximum(_ni_data.reshape(ncol, nlev), 0.0)
   671	
   672	    return q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col
   673	
   674	
   675	def _get_grid_lat_lon(grid_or_mesh, shape_2d):
  1805	        if _ocn_frac_ext is not None:
  1806	            _ocn_frac_col = jnp.asarray(_ocn_frac_ext)
  1807	            if _ocn_frac_col.size != ncol:
  1808	                raise ValueError(
  1809	                    f"forcing['open_ocean_frac'] has {_ocn_frac_col.size} "
  1810	                    f"values but this grid has ncol={ncol} columns.")
  1811	            _ocn_frac_col = _ocn_frac_col.reshape(ncol)
  1812	            if forcing.get("sfc_sw_up") is not None:
  1813	                raise ValueError(
  1814	                    "forcing carries both a prescribed surface shortwave "
  1815	                    "(sfc_sw_up) and open_ocean_frac; the prescribed albedo "
  1816	                    "would be overwritten on the direct beam.")
  1817	        # Land-coupled radiating surface (multilayer land, U11-1); a
  1818	        # prescribed LW below still wins.
  1819	        T_sfc_col, _emis_col = resolve_land_radiative_bc(
  1820	            forcing, radiation_config, ncol, T_sfc_col, _emis_col)
  1821	        T_sfc_col, _alb_col, _emis_col = resolve_prescribed_radiative_bc(
  1822	            forcing, radiation_config, ncol, T_sfc_col, _alb_col, _emis_col)
  1823	
  1824	        q_v_col, q_cloud_col, q_ice_col, n_cloud_col, n_ice_col = (
  1825	            _extract_tracer_columns(
  1826	                state, ncol, nlev,
  1827	                rho_col=_rho_for_number_conversion(
  1828	                    state, ncol, nlev, T_col, p_full_col))
  1829	        )
  1830	
  1831	        # Aerosol-CCN droplet number for the cloud-optics PSD (Twomey first
  1832	        # indirect effect): override the (dead-zeros / constant-r_eff) N_c
  1833	        # with the per-column Andreae (2009) AOD->CCN diagnostic so the
  1834	        # radiation effective radius is consistent with the microphysics
  1835	        # specified-Nc fill.  Mirrors the coupled physics_pipeline radiation
  1980	            if _aer_lw_ext is not None:
  1981	                _aer_lw_ext = shard_columns(_aer_lw_ext, column_mesh)
  1982	            if _cf_ovr is not None:
  1983	                _cf_ovr = shard_columns(_cf_ovr, column_mesh)
  1984	            if _conv_precip_col is not None:
  1985	                _conv_precip_col = shard_columns(_conv_precip_col, column_mesh)
  1986	
  1987	        if _ecrad:
  1988	            return _ecrad_ifs_physics(
  1989	                state, grid_or_mesh, phys_state, forcing, shape_3d, shape_2d,
  1990	                ncol=ncol, nlev=nlev, T_col=T_col, p_full_col=p_full_col,
  1991	                p_half_col=p_half_col, T_sfc_col=T_sfc_col, lat_col=lat_col,
  1992	                lon_col=lon_col, cos_sza_col=cos_sza_col, alb_col=_alb_col,
  1993	                ocn_frac_col=_ocn_frac_col, emis_col=_emis_col, q_v_col=q_v_col,
  1994	                q_cloud_col=q_cloud_col, q_ice_col=q_ice_col, cf_col=_cf_ovr,
  1995	                eccf=eccf, tsi=_tsi_ext, o3_vmr_col=_o3_ext, ghg_vmr=_ghg_ext)
  1996	
  1997	        rad_out = _call_radiation_backend(
  1998	            radiation_config=radiation_config,
  1999	            eccf=eccf,
  2000	            T=T_col,
  2001	            p_full=p_full_col,
  2002	            p_half=p_half_col,
  2003	            sfc_temperature=T_sfc_col,
  2004	            lat=lat_col,
  2005	            q_v=q_v_col,
  2006	            insolation=insol_col,
  2007	            cos_sza=cos_sza_col,
  2008	            sfc_albedo_override=_alb_col,
  2009	            open_ocean_frac=_ocn_frac_col,
  2010	            sfc_emissivity_override=_emis_col,
  2011	            q_cloud=q_cloud_col,
  2012	            q_ice=q_ice_col,
  2013	            n_cloud=n_cloud_col,
  2014	            n_cloud_in_cloud=_nc_in_cloud,
  2015	            n_ice=n_ice_col,
  2016	            f_day=f_day_col,
  2017	            rrtmgp_solver=rrtmgp_solver,
  2018	            lon=lon_col,
  2019	            ml_ozone_coefs=ml_ozone_coefs,
  2020	            o3_vmr_override=_o3_ext,
  2021	            o3_top_vmr=_o3_top_ext,
  2022	            aerosol_od=_aer_ext,
  2023	            aerosol_lw_od=_aer_lw_ext,
  2024	            ghg_vmr_override=_ghg_ext,
  2025	            cloud_fraction_override=_cf_ovr,
140:integer :: cldfsnow_idx = 0 
380:   cldfsnow_idx = pbuf_get_index('CLDFSNOW',errcode=err)
430:   if (cldfsnow_idx > 0) then
798:   real(r8) :: snow_tau    (nswbands,pcols,pver) ! snow extinction optical depth
799:   real(r8) :: snow_tau_w  (nswbands,pcols,pver) ! snow single scattering albedo * tau
800:   real(r8) :: snow_tau_w_g(nswbands,pcols,pver) ! snow assymetry parameter * tau * w
801:   real(r8) :: snow_tau_w_f(nswbands,pcols,pver) ! snow forward scattered fraction * tau * w
831:   real(r8) :: gb_snow_tau(pcols,pver) ! grid-box mean snow_tau
888:   if (cldfsnow_idx > 0) then
889:      call pbuf_get_field(pbuf, cldfsnow_idx, cldfsnow, start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
928:      if (cldfsnow_idx > 0) then
969:         if (cldfsnow_idx > 0) then
971:            call get_snow_optics_sw(state, pbuf, snow_tau, snow_tau_w, snow_tau_w_g, snow_tau_w_f)
977:                     c_cld_tau(:,i,k)     = ( cldfsnow(i,k)*snow_tau(:,i,k) &
980:                     c_cld_tau_w(:,i,k)   = ( cldfsnow(i,k)*snow_tau_w(:,i,k)  &
983:                     c_cld_tau_w_g(:,i,k) = ( cldfsnow(i,k)*snow_tau_w_g(:,i,k) &
986:                     c_cld_tau_w_f(:,i,k) = ( cldfsnow(i,k)*snow_tau_w_f(:,i,k) &
1008:         if (cldfsnow_idx > 0) then
1009:            rd%snow_icld_vistau(:ncol,:) = snow_tau(idx_sw_diag,:ncol,:)
1021:            if (cldfsnow_idx > 0) then
1057:         if (cldfsnow_idx > 0) then
1196:         gb_snow_tau(:,:) = 0._r8
1198:         if (cldfsnow_idx > 0) then
1202:                     gb_snow_tau(i,k) = snow_tau(rrtmg_sw_cloudsim_band,i,k)*cldfsnow(i,k)
1219:               snow_tau_in=gb_snow_tau, snow_emis_in=gb_snow_lw)
1349:   if (cldfsnow_idx > 0) then
packages/land/legoesm/land/snow_budget.py:6:    M = min(snow, max(0, Q_net * dt / L_f))
packages/land/legoesm/land/snow_budget.py:22:# Degree-day-style snow melt rate default [kg/m2/s/K] (scheme default).  Used as
packages/land/legoesm/land/snow_budget.py:28:def update_snow(
packages/land/legoesm/land/snow_budget.py:29:    snow: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:30:    snow_age: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:32:    precip_snow: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:36:    snow_melt_rate: float = _SNOW_MELT_RATE_DEFAULT,
packages/land/legoesm/land/snow_budget.py:37:    T_snow_melt: float = constants.T_freeze,
packages/land/legoesm/land/snow_budget.py:38:    snow_age_activation_K: float,
packages/land/legoesm/land/snow_budget.py:40:    """Update snow depth and age, returning the melt amount.
packages/land/legoesm/land/snow_budget.py:44:    snow : jnp.ndarray
packages/land/legoesm/land/snow_budget.py:45:        Current snow depth [kg/m2].
packages/land/legoesm/land/snow_budget.py:46:    snow_age : jnp.ndarray
packages/land/legoesm/land/snow_budget.py:47:        Current snow age [s].
packages/land/legoesm/land/snow_budget.py:50:    precip_snow : jnp.ndarray
packages/land/legoesm/land/snow_budget.py:59:            melt = min(snow_available, max(0, Q_net * dt / L_f))
packages/land/legoesm/land/snow_budget.py:63:    snow_melt_rate : float
packages/land/legoesm/land/snow_budget.py:64:        Degree-day snowmelt rate coefficient [kg/m2/s/K above T_melt].
packages/land/legoesm/land/snow_budget.py:66:    T_snow_melt : float
packages/land/legoesm/land/snow_budget.py:67:        Temperature above which snow melts [K].
packages/land/legoesm/land/snow_budget.py:68:    snow_age_activation_K : float
packages/land/legoesm/land/snow_budget.py:69:        Snow-age metamorphism activation temperature [K], from
packages/land/legoesm/land/snow_budget.py:70:        ``LandAlbedoConfig.snow_age_activation_K``; 0.0 = calendar clock.
packages/land/legoesm/land/snow_budget.py:75:    (snow_new, snow_age_new, snow_melt)
packages/land/legoesm/land/snow_budget.py:76:        Updated snow depth [kg/m2], snow age [s], and amount of snow
packages/land/legoesm/land/snow_budget.py:78:        * add ``snow_melt`` to the liquid-water budget, and
packages/land/legoesm/land/snow_budget.py:79:        * subtract ``snow_melt * L_f / (C * d)`` from the surface
packages/land/legoesm/land/snow_budget.py:82:    # Accumulation from snowfall
packages/land/legoesm/land/snow_budget.py:83:    snow_accum = precip_snow * dt  # kg/m2
packages/land/legoesm/land/snow_budget.py:84:    snow_available = snow + snow_accum
packages/land/legoesm/land/snow_budget.py:88:        # Only melt when T_sfc >= T_snow_melt AND Q_net > 0
packages/land/legoesm/land/snow_budget.py:89:        above_freezing = T_sfc >= T_snow_melt
packages/land/legoesm/land/snow_budget.py:91:        snow_melt = jnp.where(above_freezing, energy_melt, 0.0)
packages/land/legoesm/land/snow_budget.py:92:        snow_melt = jnp.minimum(snow_melt, snow_available)
packages/land/legoesm/land/snow_budget.py:95:        melt_rate = snow_melt_rate * jnp.maximum(T_sfc - T_snow_melt, 0.0)
packages/land/legoesm/land/snow_budget.py:96:        snow_melt = jnp.minimum(melt_rate * dt, snow_available)
packages/land/legoesm/land/snow_budget.py:98:    snow_new = jnp.maximum(snow_available - snow_melt, 0.0)
packages/land/legoesm/land/snow_budget.py:100:    snow_age_new = update_snow_age(
packages/land/legoesm/land/snow_budget.py:101:        snow_new, snow_age, precip_snow, dt,
packages/land/legoesm/land/snow_budget.py:102:        T_snow=T_sfc, age_activation_K=snow_age_activation_K)
packages/land/legoesm/land/snow_budget.py:104:    return snow_new, snow_age_new, snow_melt
packages/land/legoesm/land/snow_budget.py:107:def metamorphism_rate(T_snow: jnp.ndarray, activation_K: float) -> jnp.ndarray:
packages/land/legoesm/land/snow_budget.py:108:    """Temperature scaling of snow grain growth, BATS/CLM form.
packages/land/legoesm/land/snow_budget.py:111:    of Dickinson et al. (1993) BATS with activation temperature ``A`` [K]
packages/land/legoesm/land/snow_budget.py:113:    snow cools: 0.08 at 240 K, 0.03 at 230 K, so cold dry polar snow ages ~12-30
packages/land/legoesm/land/snow_budget.py:114:    times more slowly than melting snow.  Bounded to (0, 1]: snow never ages
packages/land/legoesm/land/snow_budget.py:118:    ``activation_K = 0`` returns exactly 1.0 (the temperature-independent clock).
packages/land/legoesm/land/snow_budget.py:120:    if activation_K == 0.0:
packages/land/legoesm/land/snow_budget.py:121:        return jnp.ones_like(T_snow)
packages/land/legoesm/land/snow_budget.py:122:    T = jnp.maximum(T_snow, 1.0)                       # guard the 1/T at T -> 0
packages/land/legoesm/land/snow_budget.py:124:        jnp.exp(activation_K * (1.0 / constants.T_freeze - 1.0 / T)), 0.0, 1.0)
packages/land/legoesm/land/snow_budget.py:127:def update_snow_age(
packages/land/legoesm/land/snow_budget.py:128:    snow_new: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:129:    snow_age: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:130:    precip_snow: jnp.ndarray,
packages/land/legoesm/land/snow_budget.py:133:    T_snow: jnp.ndarray | None = None,
packages/land/legoesm/land/snow_budget.py:134:    age_activation_K: float,
packages/land/legoesm/land/snow_budget.py:136:    """Snow-age clock: mass-weighted grain-age mixing of fresh + existing snow.
packages/land/legoesm/land/snow_budget.py:138:    Shared by the cell-mean budget (:func:`update_snow`) and the elevation-band
packages/land/legoesm/land/snow_budget.py:139:    scheme (``snow_bands``), which keeps a single cell-level age for the albedo
packages/land/legoesm/land/snow_budget.py:142:    Fresh snowfall must NOT hard-reset the albedo age to zero on a trace flurry: the
packages/land/legoesm/land/snow_budget.py:143:    old ``precip_snow > 1e-10`` FULL reset drove the albedo to its bright maximum on
packages/land/legoesm/land/snow_budget.py:145:    grain age mixes by MASS (CLM/BATS): existing snow ages by ``dt`` while fresh snow
packages/land/legoesm/land/snow_budget.py:148:        new_age = (snow_age + dt) * swe_old / (swe_old + fresh_swe)
packages/land/legoesm/land/snow_budget.py:150:    with ``fresh_swe = precip_snow*dt`` [kg/m2] the fresh snow added this step and
packages/land/legoesm/land/snow_budget.py:151:    ``swe_old = snow_new - fresh_swe`` [kg/m2] the pre-existing snow surviving this
packages/land/legoesm/land/snow_budget.py:153:    ``snow_new`` net of the fresh addition).  Trace snow on a deep aged pack barely
packages/land/legoesm/land/snow_budget.py:154:    moves the age (``swe_old >> fresh_swe``); fresh snow on bare/thin ground drives
packages/land/legoesm/land/snow_budget.py:156:    to 0 once the pack has fully melted (``snow_new == 0``).
packages/land/legoesm/land/snow_budget.py:158:    # Units: fresh_swe, swe_old, denom all [kg/m2]; snow_age, dt [s]; ratio is
311:        # MOST selectors for the ``tas`` 2 m profile (``_tas_2m``): the SAME
323:        self.produces_tas = surface_bulk_scheme != "none"
324:        self.tas_profile_scheme = (
447:                track_extremes={"tas"} if self.produces_tas else set(),
808:    def _tas_2m(self, state, q_v, sst, sic, T_ice, u_low=None, v_low=None,
809:                *, T_land=None, q_land=None, land_fraction=None):
865:            if ocean and self.tas_profile_scheme == "large_yeager_cesm":
876:                scheme=(self.tas_profile_scheme
877:                        if self.tas_profile_scheme != "large_yeager_cesm"
1090:            temperature: ``tas = tas_2m(instant) + (t_low_mean - T_low)``,
1322:            if self.produces_tas:
1323:                tas_field = self._tas_2m(state, q_v, sst, sic, T_ice)
1709:        if self._spatial_daily is None or not self.produces_tas:
1967:        # ascending, ~1.0 at the surface — see _interp_to_plev19 / _tas_2m).
2055:        if not self.produces_tas:
2614:        if restored and not self.produces_tas:

exec
/usr/bin/bash -lc "rg -n -i 'snowveg_flag|IsSnowvegFlagOnRad|snowveg_onrad' /glade/derecho/scratch/pg2328/cesm_src/CLM5/{src,bld} | head -65; nl -ba packages/core/legoesm/core/bulk_flux.py | sed -n '1346,1378p;1910,1960p'; rg -n 'z0t|z0m|large_yeager' packages/core/legoesm/core/bulk_flux.py | head -40; nl -ba packages/core/legoesm/surface_albedo.py | sed -n '260,415p'; rg -n '_tas_2m|T_land=|tas_field' packages/coupler/legoesm/driver/model_driver.py | tail -25; rg -n 'snow_insulation|snow_depth' packages/land/legoesm/land/soil_thermal.py | head -30; rg --files /glade/derecho/scratch/pg2328/cesm_src/CICE | head -12" in /glade/u/home/pg2328/legoesm
 succeeded in 613ms:
/glade/derecho/scratch/pg2328/cesm_src/CLM5/bld/namelist_files/namelist_definition_clm4_5.xml:912:<entry id="snowveg_flag" type="char*10" category="clm_physics"
/glade/derecho/scratch/pg2328/cesm_src/CLM5/bld/CLMBuildNamelist.pm:1995:    add_default($opts, $nl_flags->{'inputdata_rootdir'}, $definition, $defaults, $nl, 'snowveg_flag', 'phys'=>$nl_flags->{'phys'} );
/glade/derecho/scratch/pg2328/cesm_src/CLM5/bld/namelist_files/namelist_defaults_clm4_5.xml:304:<snowveg_flag phys="clm4_5" >OFF</snowveg_flag>
/glade/derecho/scratch/pg2328/cesm_src/CLM5/bld/namelist_files/namelist_defaults_clm4_5.xml:305:<snowveg_flag phys="clm5_0" >ON_RAD</snowveg_flag>
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:39:  use CanopyHydrologyMod    , only : IsSnowvegFlagOn, IsSnowvegFlagOnRad
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:67:  logical, private :: snowveg_on     = .false.                ! snowveg_flag = 'ON'
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:68:  logical, private :: snowveg_onrad  = .true.                 ! snowveg_flag = 'ON_RAD'
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1133:      ! Set status of snowveg_flag
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1135:      snowveg_onrad = IsSnowvegFlagOnRad()
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1257:         if (snowveg_on .or. snowveg_onrad) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:37:  public :: IsSnowvegFlagOff       ! Returns true if snowveg_flag is OFF
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:38:  public :: IsSnowvegFlagOn        ! Returns true if snowveg_flag is ON
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:39:  public :: IsSnowvegFlagOnRad     ! Returns true if snowveg_flag is ON_RAD
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:51:  logical, private :: snowveg_off    = .false.  ! snowveg_flag = 'OFF'
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:52:  logical, private :: snowveg_on     = .false.  ! snowveg_flag = 'ON'
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:53:  logical, private :: snowveg_onrad  = .true.   ! snowveg_flag = 'ON_RAD'
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:55:  character(len= 10), public           :: snowveg_flag = 'ON_RAD'
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:89:         snowveg_flag
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:113:       snowveg_onrad = IsSnowvegFlagOnRad()
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:116:       write(iulog,*) 'snowveg_onrad = ',snowveg_onrad
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:117:       if (snowveg_off .or. snowveg_on .or. snowveg_onrad) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:118:          write(iulog,*) 'snowveg_flag = ',snowveg_flag
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:120:          call endrun(msg="snowveg_flag is set incorrectly (not ON, ON_RAD, or OFF)"//errmsg(sourcefile, __LINE__))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:128:    call shr_mpi_bcast(snowveg_flag, mpicom)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:276:       ! Set status of snowveg_flag
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:278:       snowveg_onrad = IsSnowvegFlagOnRad()
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:328:                   if (snowveg_on .or. snowveg_onrad) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:343:                   if (snowveg_on .or. snowveg_onrad) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:365:                   if (snowveg_on .or. snowveg_onrad) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:421:                if (snowveg_on .or. snowveg_onrad) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:712:       ! Set status of snowveg_flag
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:713:       snowveg_onrad = IsSnowvegFlagOnRad()
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:723:                if (snowveg_onrad) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:867:     ! Return true if snowveg_flag is OFF
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:874:     IsSnowvegFlagOff = (trim(snowveg_flag) == 'OFF')
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:889:     ! Return true if snowveg_flag is ON
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:896:     IsSnowvegFlagOn = (trim(snowveg_flag) == 'ON')
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:903:   ! !IROUTINE: IsSnowvegFlagOnRad
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:907:   logical function IsSnowvegFlagOnRad( )
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:911:     ! Return true if snowveg_flag is ON_RAD
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:918:     IsSnowvegFlagOnRad = (trim(snowveg_flag) == 'ON_RAD')
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:920:   end function IsSnowvegFlagOnRad
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:30:  use CanopyHydrologyMod, only : IsSnowvegFlagOn, IsSnowvegFlagOnRad
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:70:  logical, private :: snowveg_onrad  = .true.   ! snowveg_flag = 'ON_RAD'
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1298:   ! Set status of snowveg_flag
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1299:   snowveg_onrad = IsSnowvegFlagOnRad()
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1319:          if ( lSFonly .or. ( (.not. snowveg_onrad) .and. (t_veg(p) > tfrz) ) ) then
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/SurfaceAlbedoMod.F90:1327:             if (snowveg_onrad) then
  1346	def compute_most_fluxes(
  1347	    u_rel,
  1348	    v_rel,
  1349	    T_atm,
  1350	    q_atm,
  1351	    T_sfc,
  1352	    q_sfc,
  1353	    rho,
  1354	    z_ref=10.0,
  1355	    z_t=None,
  1356	    z_q=None,
  1357	    z0_init=1e-4,
  1358	    scheme="coare3",
  1359	    n_iter=5,
  1360	    charnock=0.011,
  1361	    L_latent=None,
  1362	    thermo_convention="legoesm",
  1363	    gustiness_w_zi=None,
  1364	    gustiness_beta=1.25,
  1365	    return_2m=False,
  1366	    z_diag=2.0,
  1367	    max_exchange_coeff=None,
  1368	    stability_scheme="dyer1974",
  1369	    return_convergence=False,
  1370	    *,
  1371	    return_wind_speed=False,
  1372	    unstable_gamma=_DYER_UNSTABLE_GAMMA,
  1373	    stable_beta=_DYER_STABLE_BETA,
  1374	    z0h_z0_ratio=_Z0H_Z0_RATIO_DEFAULT,
  1375	    coare_neutral_width=COARE_NEUTRAL_WIDTH,
  1376	    coare_charnock_ramp_width=COARE_CHARNOCK_RAMP_WIDTH,
  1377	    coare_gust_onset_width=COARE_GUST_ONSET_WIDTH,
  1378	    large_yeager_neutral_width=NCAR_NEUTRAL_WIDTH,
  1910	    # ``return_wind_speed``: the speed THIS solver put in its flux formulas
  1911	    # (U_eff incl. gustiness/floor, or AeroBulk's Ub), appended LAST, for a
  1912	    # closure that must pair it with this u* (WRF MYNN's ust**2/wspd drag).
  1913	    _wind_out = (U_eff_final,) if return_wind_speed else ()
  1914	
  1915	    if return_2m:
  1916	        # Air temperature at the diagnostic height (default 2 m) from the
  1917	        # converged MOST similarity profile: T(z) = T_sfc − (θ*/κ)·[ln(z/z0t) −
  1918	        # ψ_h(z/L)], which reduces to T_atm at z=z_t.  Over a warm ocean the
  1919	        # lowest model level (~100 m at nlev=20) reads colder than 2 m, so the
  1920	        # raw lowest-level "tas" exaggerates the cold/air-sea-gap bias — this
  1921	        # gives the physically correct CMIP 2 m value.  Recompute 1/L from the
  1922	        # converged scales (the iteration carries scales, not L).
  1923	        if _coare is None:
  1924	            u_star_safe = jnp.maximum(u_star, 1e-6)
  1925	            theta_v_star = theta_star + _vT_coef * T_atm * q_star_val
  1926	            inv_L = -KAPPA * G * theta_v_star / (u_star_safe ** 2 * T_v)
  1927	            zeta_d = jnp.clip(z_diag * inv_L, -10.0, 10.0)
  1928	            _psi_h_d = psi_h(zeta_d, stability_scheme,
  1929	                             unstable_gamma=unstable_gamma, stable_beta=stable_beta)
  1930	            denom_d = jnp.log(z_diag / jnp.maximum(z0_t, 1e-12)) - _psi_h_d
  1931	            T_2m = T_sfc - (theta_star / KAPPA) * denom_d
  1932	        else:
  1933	            # coare3 / large_yeager: AeroBulk's own height shift (the t_zu
  1934	            # update of TURB_COARE3P0 / TURB_NCAR) applied from z_t to z_diag
  1935	            # with the solver's last-iteration 1/L and theta*, zeta clipped as
  1936	            # in that routine: T(z_t) = T_atm exactly.
  1937	            if scheme == "coare3":
  1938	                _zmax = _AB_COARE_ZETA_ABS_MAX
  1939	
  1940	                def _ph(z):
  1941	                    return psi_h_coare(z, stability_scheme, coare_neutral_width)
  1942	            else:
  1943	                _zmax = _AB_NCAR_ZETA_ABS_MAX
  1944	
  1945	                def _ph(z):
  1946	                    return psi_h(z, stability_scheme, unstable_gamma=unstable_gamma,
  1947	                                 stable_beta=stable_beta)
  1948	
  1949	            def _zeta(z):
  1950	                return jnp.clip(z * _coare["inv_L"], -_zmax, _zmax)
  1951	            prf = jnp.log(z_t / z_diag) + _ph(_zeta(z_diag)) - _ph(_zeta(z_t))
  1952	            T_2m = T_atm - _coare["ts"] / KAPPA * prf
  1953	        # Guard against profile extrapolation outside [T_atm, T_sfc].
  1954	        lo = jnp.minimum(T_atm, T_sfc)
  1955	        hi = jnp.maximum(T_atm, T_sfc)
  1956	        T_2m = jnp.clip(T_2m, lo, hi)
  1957	        if return_convergence:
  1958	            return (tau_x, tau_y, shflx, lhflx, u_star, T_2m, most_residual
  1959	                    ) + _wind_out
  1960	        return (tau_x, tau_y, shflx, lhflx, u_star, T_2m) + _wind_out
13:3. ``large_yeager`` — Large & Yeager 2004/2009 (CORE/OMIP) as implemented by
20:large_yeager's neutral Stanton switches (first guess and loop) and 33 m/s drag
22:``coare_gust_onset_width``, ``large_yeager_neutral_width``,
23:``large_yeager_dthv_width``, ``large_yeager_cd_cutoff_width``; 0 = the
24:oracle).  The three large_yeager widths are fields of every component config
26::func:`large_yeager_widths` (the 2 m diagnostic still uses the defaults).  coare3's dt/dq sign floors stay hard as in AeroBulk (division guard,
32:The ``large_yeager`` path is the OMIP-2 protocol bulk formula (Griffies
91:#   "large_yeager" — Large & Yeager 2009 (OMIP)
92:#   "large_yeager_cesm" — CESM/CIME ``shr_flux_atmOcn`` (Large & Pond 1981/82
97:    "constant", "most", "coare3", "large_yeager", "large_yeager_cesm",
171:# and thereby STRENGTHENS the sensible/latent flux.  COARE 3.0 and large_yeager
324:def large_yeager_neutral_cd(
721:_AB_FG_CH10 = _f32(0.00115)       # neutral 10 m Stanton number for z0t
722:_AB_FG_Z_MIN = _f32(1.0e-8)       # first-guess z0 / z0t floor [m]
730:_AB_COARE_Z_MIN = _f32(1.0e-9)    # z0 / z0t floor [m]
1026:    z0t = 10.0 / jnp.exp(KAPPA / (_AB_FG_CH10 * ((jnp.log(10.0) - log_z0) / KAPPA)))
1027:    z0t = jnp.clip(jnp.abs(z0t), _AB_FG_Z_MIN, 1.0)
1028:    log_z0t = jnp.log(z0t)
1030:    cc_ri = KAPPA * KAPPA / (Cd * (jnp.log(z_t) - log_z0t)) * rib
1040:    tmp = KAPPA / (log_zu - log_z0t - psih_u)
1076:    positive-downward sign, z0t, 1/L) and the last-iteration relative u* change.
1087:    z0t = jnp.zeros_like(us)
1097:        us, ts, qs, t_zu, q_zu, dt, dq, log_z0, _z0t, _Ub, _inv_L, _resid = c
1111:        z0t = jnp.minimum(_AB_COARE_Z0T_MAX,
1113:        z0t = jnp.clip(jnp.abs(z0t), _AB_COARE_Z_MIN, 1.0)
1115:        tmp = KAPPA / _floor(log_zu - jnp.log(z0t) - psih_u)
1131:        return (us_new, ts, qs, t_zu, q_zu, dt, dq, log_z0, z0t, Ub, inv_L, resid)
1133:    carry = (us, ts, qs, t_zu, q_zu, dt, dq, log_z0, z0t, Ub, inv_L, resid)
1140:    us, ts, qs, t_zu, q_zu, dt, dq, _log_z0, z0t, Ub, inv_L, resid = jax.lax.fori_loop(
1147:                z0t=z0t, inv_L=inv_L, resid=resid)
1174:    "large_yeager_neutral_width": NCAR_NEUTRAL_WIDTH,
1175:    "large_yeager_dthv_width": NCAR_DTHV_WIDTH,
1176:    "large_yeager_cd_cutoff_width": NCAR_CD_CUTOFF_WIDTH,
1180:def large_yeager_widths(config) -> dict:
1378:    large_yeager_neutral_width=NCAR_NEUTRAL_WIDTH,
1379:    large_yeager_dthv_width=NCAR_DTHV_WIDTH,
1380:    large_yeager_cd_cutoff_width=NCAR_CD_CUTOFF_WIDTH,
1419:        ``coare3`` / ``large_yeager``: AeroBulk's first guesses have no
1422:        ``"constant"``, ``"most"``, ``"coare3"`` or ``"large_yeager"``.  The
1425:        ``coare3``/``large_yeager`` evolve the roughness in the iteration.
   260	    snow_depth : jnp.ndarray
   261	        Snow water equivalent [kg/m2].
   262	    config : LandAlbedoConfig
   263	
   264	    Returns
   265	    -------
   266	    f_snow : jnp.ndarray
   267	        Snow cover fraction [0-1].
   268	    """
   269	    return jnp.tanh(snow_depth / jnp.maximum(config.snow_depth_crit, 1e-6))
   270	
   271	
   272	def ground_snow_cover(
   273	    snow_depth: jnp.ndarray,
   274	    config: LandAlbedoConfig = LandAlbedoConfig(),
   275	) -> jnp.ndarray:
   276	    """Snow-covered fraction of the GROUND as the land albedo blend uses it:
   277	    :func:`snow_cover_fraction` times the per-cell ``snow_cover_scale`` (when
   278	    set), clipped to [0, 1]: the cover the albedo BLEND uses.  The canopy snow
   279	    burial deliberately uses the unscaled :func:`snow_cover_fraction` instead
   280	    (the scale is a brightness calibration, not snow area; user 2026-10-03)."""
   281	    f_snow = snow_cover_fraction(snow_depth, config)
   282	    if config.snow_cover_scale is not None:
   283	        f_snow = jnp.clip(f_snow * jnp.asarray(config.snow_cover_scale), 0.0, 1.0)
   284	    return f_snow
   285	
   286	
   287	def dry_soil_brightening(
   288	    theta_top: jnp.ndarray,
   289	    config: LandAlbedoConfig = LandAlbedoConfig(),
   290	) -> jnp.ndarray:
   291	    """Dry-soil albedo increment (Oleson et al. 2013, CLM).
   292	
   293	    ``delta = boost * clip(1 - theta_top / theta_ref, 0, 1)`` — the snow-free soil
   294	    albedo rises by up to ``soil_dry_albedo_boost`` as the top-layer volumetric water
   295	    ``theta_top`` falls to zero, and vanishes once ``theta_top >= soil_dry_albedo_ref``.
   296	    This is what makes a dry desert (theta ~ 0.05) bright while moist bare soil / tundra
   297	    (theta ~ 0.3) stays dark.
   298	
   299	    Parameters
   300	    ----------
   301	    theta_top : jnp.ndarray
   302	        Top soil-layer volumetric water content [m3/m3].
   303	    config : LandAlbedoConfig
   304	
   305	    Returns
   306	    -------
   307	    delta : jnp.ndarray
   308	        Additive albedo increment in [0, soil_dry_albedo_boost].
   309	    """
   310	    ref = jnp.maximum(config.soil_dry_albedo_ref, 1e-6)
   311	    return config.soil_dry_albedo_boost * jnp.clip(1.0 - theta_top / ref, 0.0, 1.0)
   312	
   313	
   314	def land_albedo(
   315	    lat: jnp.ndarray,
   316	    snow_depth: jnp.ndarray,
   317	    snow_age: jnp.ndarray,
   318	    config: LandAlbedoConfig = LandAlbedoConfig(),
   319	    base_albedo: jnp.ndarray | None = None,
   320	    f_snow_override: jnp.ndarray | None = None,
   321	    snow_contrib_override: jnp.ndarray | None = None,
   322	    cos_zenith: jnp.ndarray | None = None,
   323	) -> jnp.ndarray:
   324	    """Compute land surface albedo with snow feedback.
   325	
   326	    alpha_land = alpha_veg * (1 - f_snow) + alpha_snow * f_snow
   327	
   328	    Parameters
   329	    ----------
   330	    lat : jnp.ndarray
   331	        Latitude in radians.
   332	    snow_depth : jnp.ndarray
   333	        Snow water equivalent [kg/m2].
   334	    snow_age : jnp.ndarray
   335	        Time since last snowfall [s].
   336	    config : LandAlbedoConfig
   337	    base_albedo : jnp.ndarray or None
   338	        Per-cell snow-free base albedo (e.g. CLM PFT map); ``None`` uses the
   339	        latitude-band vegetation albedo.
   340	    f_snow_override : jnp.ndarray or None
   341	        Pre-aggregated snow-cover fraction (e.g. the area-weighted sub-grid
   342	        elevation-band cover from ``land.snow_bands``).  ``None`` computes the
   343	        cover from ``snow_depth`` with the standard Niu & Yang curve.
   344	    snow_contrib_override : jnp.ndarray or None
   345	        Pre-aggregated snow-albedo contribution ``mean_k[alpha_snow(age_k)*f_k]``
   346	        from the sub-grid elevation bands (``land.snow_bands.band_snow_albedo``),
   347	        where each band uses its OWN age-decayed snow albedo.  When supplied (with
   348	        ``f_snow_override``) the cell snow term is this pre-aggregate instead of a
   349	        single ``alpha_snow(snow_age) * f_snow`` — so perennial firn and fresh snow
   350	        contribute at their own ages.  ``None`` uses the single-age snow albedo.
   351	
   352	    Returns
   353	    -------
   354	    alpha : jnp.ndarray
   355	        Land surface albedo.
   356	    """
   357	    # Snow-free base: a per-cell map (e.g. CLM PFT albedo) if supplied, else the
   358	    # latitude-band vegetation albedo.  Snow albedo always blends on top, so a real
   359	    # albedo map and the snow feedback coexist (bright deserts AND bright ice sheets).
   360	    alpha_veg = (land_vegetation_albedo(lat, config) if base_albedo is None
   361	                 else base_albedo)
   362	    # Per-cell snow_cover_scale (forest canopies hide ground snow, scale<1; open
   363	    # tundra whitens faster, scale>1) is applied ONLY when this function computes
   364	    # the cover itself: a caller that supplies f_snow_override (the elevation-band
   365	    # path) must scale each BAND's cover before aggregating — post-aggregate
   366	    # scaling can push the snow contribution above alpha_snow at scale>1 on
   367	    # saturated bands (codex).
   368	    f_snow = (ground_snow_cover(snow_depth, config) if f_snow_override is None
   369	              else f_snow_override)
   370	    if snow_contrib_override is not None:
   371	        # Banded path: each elevation band already blended its own age-decayed snow
   372	        # albedo; the aggregate snow contribution replaces alpha_snow * f_snow.
   373	        return alpha_veg * (1.0 - f_snow) + snow_contrib_override
   374	    alpha_snow = snow_albedo(snow_age, config, cos_zenith=cos_zenith)
   375	    # Snow cannot darken a surface below its own snow-free albedo: the aged
   376	    # value is a melting/dirty-snow asymptote, and on an ice sheet (band base
   377	    # 0.82 VIS / 0.62 NIR) the snow IS the surface.  No-op wherever the base
   378	    # is darker than aged snow (soil, vegetation), i.e. everywhere but
   379	    # glacier and other bright bases.
   380	    alpha_snow = jnp.maximum(alpha_snow, alpha_veg)
   381	    return alpha_veg * (1.0 - f_snow) + alpha_snow * f_snow
   382	
   383	
   384	# =========================================================================
   385	# Sea ice albedo
   386	# =========================================================================
   387	
   388	def ice_albedo(
   389	    T_ice: jnp.ndarray,
   390	    config: IceAlbedoConfig = IceAlbedoConfig(),
   391	) -> jnp.ndarray:
   392	    """Compute sea ice albedo as a function of ice surface temperature.
   393	
   394	    Linearly interpolates between cold and warm albedo values.
   395	    When T_ice < T_freeze - T_transition_width: alpha = alpha_ice_cold.
   396	    When T_ice >= T_freeze: alpha = alpha_ice_warm.
   397	
   398	    Parameters
   399	    ----------
   400	    T_ice : jnp.ndarray
   401	        Ice surface temperature [K].
   402	    config : IceAlbedoConfig
   403	
   404	    Returns
   405	    -------
   406	    alpha : jnp.ndarray
   407	        Sea ice albedo.
   408	    """
   409	    T_cold = config.T_freeze - config.T_transition_width
   410	    frac = jnp.clip(
   411	        (T_ice - T_cold) / config.T_transition_width, 0.0, 1.0
   412	    )
   413	    return config.alpha_ice_cold * (1.0 - frac) + config.alpha_ice_warm * frac
   414	
   415	
10937:                        T_land=jnp.asarray(_tas_land_T).reshape(-1),
10947:                tas = diag._tas_2m(
14916:                "columns solved), cadence every %d step(s) (DT_land=%.1f s)",
18609:            T_land=ctx["T_land"], w_land=ctx["w_land"],
19275:            T_land=ctx["T_land"], w_land=ctx["w_land"], snow=ctx.get("snow"),
20151:            T_land=ctx["T_land"], w_land=ctx["w_land"],
20630:                T_land=T_land,
20815:                        T_land=_g(getattr(carry, "T_land", None)),
21444:                T_land=T_land, w_land=w_land, snow=snow, **_dm_step_in,
21589:                    T_land=T_land, w_land=w_land, snow=snow, **_dm_step_in,
170:    snow_insulation: bool = False
730:      ``snow_swe/f`` as the pack empties: ``snow_depth_crit`` for the tanh
/glade/derecho/scratch/pg2328/cesm_src/CICE/README.md
/glade/derecho/scratch/pg2328/cesm_src/CICE/DistributionPolicy.pdf
/glade/derecho/scratch/pg2328/cesm_src/CICE/CONTRIBUTING.md
/glade/derecho/scratch/pg2328/cesm_src/CICE/codecov.yml
/glade/derecho/scratch/pg2328/cesm_src/CICE/cice.setup
/glade/derecho/scratch/pg2328/cesm_src/CICE/COPYRIGHT.pdf
/glade/derecho/scratch/pg2328/cesm_src/CICE/LICENSE.pdf
/glade/derecho/scratch/pg2328/cesm_src/CICE/doc/make.bat
/glade/derecho/scratch/pg2328/cesm_src/CICE/doc/pyproject.toml
/glade/derecho/scratch/pg2328/cesm_src/CICE/doc/Makefile
/glade/derecho/scratch/pg2328/cesm_src/CICE/doc/PDF/BL_NCAR2007.pdf
/glade/derecho/scratch/pg2328/cesm_src/CICE/doc/PDF/LAUR-95-1146.pdf

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/surface_albedo.py | sed -n '195,252p'; nl -ba packages/land/legoesm/land/snow_budget.py | sed -n '158,181p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '10900,10965p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | sed -n '125,205p'; nl -ba /glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90 | sed -n '1035,1078p'; rg -n 'CLDFSNOW|register.*snow' /glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/cam/micro_mg_cam.F90; rg -n 'snow_cover_scale|tau_snow_decay|alpha_snow' packages/land/legoesm/land/clm_surface_map.py | tail -25; rg -n 'maykut_untersteiner_albedo|ice_albedo_on_skin' packages/coupler/legoesm/driver/model_driver.py packages/ice/legoesm/ice/shortwave.py; rg --files /glade/derecho/scratch/pg2328/cesm_src/CICE | rg '(prescribed|shortwave).*F90'" in /glade/u/home/pg2328/legoesm
 succeeded in 36ms:
   195	    cos_zenith: jnp.ndarray | None = None,
   196	) -> jnp.ndarray:
   197	    """Compute snow albedo as a function of snow age (and optionally solar zenith).
   198	
   199	    Diffuse (age-decayed) part::
   200	
   201	        alpha_diffuse = alpha_snow_min + (alpha_snow_max - alpha_snow_min) * exp(-t/tau)
   202	
   203	    the exponential age decay standing in for grain-size metamorphism (fresh snow
   204	    bright, aged/firn snow darker).  When ``cos_zenith`` is supplied, the BATS /
   205	    Dickinson (1983) solar-zenith brightening (as in CLM) is added — snow is brighter
   206	    at a low sun (high zenith), the leading control on the polar ice-sheet shortwave
   207	    budget that an overhead-sun albedo misses::
   208	
   209	        alpha = alpha_diffuse + factor * f(mu) * (1 - alpha_diffuse)
   210	        f(mu) = max( (1/b) * ((1 + b) / (1 + 2 b mu) - 1), 0 ),  mu = clip(cos_zenith, 0, 1)
   211	
   212	    ``f`` is 0 at overhead sun (mu = 1) and rises to 1 at grazing incidence (mu = 0),
   213	    and the denominator ``1 + 2 b mu >= 1`` is never zero — differentiable everywhere.
   214	
   215	    Parameters
   216	    ----------
   217	    snow_age : jnp.ndarray
   218	        Time since last snowfall [s].
   219	    config : LandAlbedoConfig
   220	    cos_zenith : jnp.ndarray or None
   221	        Cosine of the solar zenith angle.  ``None`` disables the zenith brightening
   222	        (returns the diffuse albedo — the legacy behaviour).
   223	
   224	    Returns
   225	    -------
   226	    alpha_snow : jnp.ndarray
   227	        Snow albedo.
   228	    """
   229	    alpha_diffuse = config.alpha_snow_min + (
   230	        config.alpha_snow_max - config.alpha_snow_min
   231	    ) * jnp.exp(-snow_age / config.tau_snow_decay)
   232	    if cos_zenith is None:
   233	        return alpha_diffuse
   234	    mu = jnp.clip(cos_zenith, 0.0, 1.0)
   235	    b = config.snow_zenith_b
   236	    f_zen = jnp.maximum((1.0 / b) * ((1.0 + b) / (1.0 + 2.0 * b * mu) - 1.0), 0.0)
   237	    return alpha_diffuse + config.snow_zenith_factor * f_zen * (1.0 - alpha_diffuse)
   238	
   239	
   240	def snow_cover_fraction(
   241	    snow_depth: jnp.ndarray,
   242	    config: LandAlbedoConfig = LandAlbedoConfig(),
   243	) -> jnp.ndarray:
   244	    """Compute snow cover fraction from snow water equivalent.
   245	
   246	    Niu & Yang (2007) / CLM-family SATURATING form::
   247	
   248	        f_snow = tanh(SWE / snow_depth_crit)
   249	
   250	    A thin snowpack already masks most of the surface, so ``f_snow`` rises steeply and
   251	    reaches ~1 by ~3x ``snow_depth_crit`` while still giving light transient snow a
   252	    partial cover.  The previous linear ``min(1, SWE/crit)`` was too gradual: a
   158	    # Units: fresh_swe, swe_old, denom all [kg/m2]; snow_age, dt [s]; ratio is
   159	    # dimensionless in [0, 1], so new_age stays in [0, snow_age + dt] -- age never
   160	    # grows past the aged clock and never goes negative (monotone, positivity kept).
   161	    fresh_swe = jnp.maximum(precip_snow, 0.0) * dt           # kg/m2 fresh this step
   162	    swe_old = jnp.maximum(snow_new - fresh_swe, 0.0)         # kg/m2 old snow surviving melt
   163	    denom = swe_old + fresh_swe                              # kg/m2 (== snow_new where >=0)
   164	    # The clock accumulates METAMORPHIC EXPOSURE, not calendar time: the
   165	    # increment is dt scaled by the grain-growth rate at the snow temperature
   166	    # (BATS/CLM).  Cold dry snow therefore stays near its fresh albedo for
   167	    # months, while melting snow darkens on the same timescale as before.  The
   168	    # state stays in SECONDS of melting-point-equivalent age, so restarts and
   169	    # every downstream albedo consumer are unchanged in meaning.
   170	    # ``age_activation_K = 0`` gives rate 1 (the calendar clock).
   171	    _rate = (1.0 if (T_snow is None or age_activation_K == 0.0)
   172	             else metamorphism_rate(T_snow, age_activation_K))
   173	    aged = snow_age + dt * _rate                             # existing snow ages
   174	    # Guard the divide (denom>0 whenever snow_new>0; the snow_new==0 rows are masked
   175	    # out below, so the where only prevents a 0/0 NaN gradient on those dead rows).
   176	    mixed = aged * swe_old / jnp.where(denom > 0.0, denom, 1.0)
   177	    # If all snow has melted, reset age to zero
   178	    return jnp.where(snow_new > 0.0, mixed, 0.0)
 10900	        # this path (``get_sst_sic`` set for a radiation+SST run) — matches
 10901	        # the cube-path collect() ``tas`` instead of a bare lowest-level
 10902	        # proxy.  Uses the RECONSTRUCTED cell winds (``state.u`` is
 10903	        # edge-normal on MPAS, not cell-collocated).  A failure falls back
 10904	        # to the lowest model level (logged once) so a tas-only glitch never
 10905	        # drops the whole CMOR feed.
 10906	        tas = None
 10907	        ts = None
 10908	        sic = None
 10909	        sst = None
 10910	        _get_sst_sic = getattr(self, "get_sst_sic", None)
 10911	        if _get_sst_sic is not None:
 10912	            try:
 10913	                _sst, _sic = _get_sst_sic(day)
 10914	                _sst = jnp.asarray(_sst).reshape(-1)
 10915	                _sic = jnp.asarray(_sic).reshape(-1)
 10916	                sic = _sic
 10917	                sst = _sst
 10918	                # Report tas off the SAME ice surface the radiation +
 10919	                # turbulence saw: the per-cell prognostic skin when the
 10920	                # feature is on, else the constant T_ice.  Otherwise the
 10921	                # scorecard's 2 m extrapolation uses a 271.35 K ice surface
 10922	                # while the model cooled the skin (codex-1 finding 3).
 10923	                _tas_ice = getattr(self.config, "T_ice", None)
 10924	                if (getattr(self.config, "ice_skin_prognostic", False)
 10925	                        and getattr(self, "_ice_T_skin", None) is not None):
 10926	                    _tas_ice = self._ice_T_skin
 10927	                # Give the diagnostic the LAND surface too where the
 10928	                # interactive tile has produced one, else its profile is
 10929	                # anchored on the ocean/ice skin over land as well and the
 10930	                # published land tas is not the model's land at all.
 10931	                _tas_land_T = getattr(self, "_land_T_skin_last", None)
 10932	                _tas_land_q = getattr(self, "_land_qsfc_last", None)
 10933	                _tas_f_land = getattr(self, "_f_land", None)
 10934	                _tas_kw = {}
 10935	                if _tas_land_T is not None and _tas_f_land is not None:
 10936	                    _tas_kw = dict(
 10937	                        T_land=jnp.asarray(_tas_land_T).reshape(-1),
 10938	                        q_land=(None if _tas_land_q is None
 10939	                                else jnp.asarray(_tas_land_q).reshape(-1)),
 10940	                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1))
 10941	                from legoesm.forcing.surface_utils import blend_surface_temperature
 10942	                ts = blend_surface_temperature(_sst, _sic, _tas_ice)
 10943	                if _tas_land_T is not None and _tas_f_land is not None:
 10944	                    f_land = jnp.clip(jnp.asarray(_tas_f_land).reshape(-1), 0.0, 1.0)
 10945	                    ts = (f_land * jnp.asarray(_tas_land_T).reshape(-1)
 10946	                          + (1.0 - f_land) * ts)
 10947	                tas = diag._tas_2m(
 10948	                    state, q_v, _sst, _sic, _tas_ice,
 10949	                    u_low=u_east[..., -1], v_low=v_north[..., -1], **_tas_kw)
 10950	            except Exception as exc:
 10951	                if not getattr(self, "_logged_tas2m_fallback", False):
 10952	                    logger.warning(
 10953	                        "  CMOR tas: 2 m MOST calc failed (%s); using the "
 10954	                        "lowest model level as the tas proxy.", exc)
 10955	                    self._logged_tas2m_fallback = True
 10956	                tas = None
 10957	        return (state.T.data[..., -1] if tas is None else tas), ts, sic, sst
 10958	
 10959	    def _mpas_cmip_native_kwargs(self, day: float, diag,
 10960	                                 u_override=None, *, state_only=False) -> dict:
 10961	        """Build the NATIVE (rank-local) cell-field kwargs for
 10962	        :meth:`DiagnosticCollector.feed_cmip_accumulators_native`.
 10963	
 10964	        PURE and COLLECTIVE-FREE: reads ``self.state`` / the physics flux
 10965	        export and returns host/device arrays whose leading axis is the
   125	        x = jnp.asarray(x)
   126	        if x.ndim == 0:
   127	            return jnp.broadcast_to(x, (ncol,))
   128	        if x.ndim > 2 or x.size != ncol:
   129	            raise ValueError(
   130	                "a prescribed radiative surface plane must be a scalar, "
   131	                f"(ncol,) or (n_lat, n_lon) with ncol={ncol}; got shape "
   132	                f"{tuple(x.shape)}.")
   133	        return x.reshape(ncol)
   134	
   135	    lw_up = forcing.get("sfc_lw_up")
   136	    sw_up = forcing.get("sfc_sw_up")
   137	    sw_dn = forcing.get("sfc_sw_down")
   138	    if lw_up is None and sw_up is None and sw_dn is None:
   139	        return T_sfc_col, alb_col, emis_col
   140	    if radiation_config.scheme not in ("rrtmgp", "gray"):
   141	        # simple_lw / mc3d take no surface temperature or albedo override,
   142	        # so the prescribed boundary condition would be silently ignored.
   143	        raise ValueError(
   144	            "prescribed radiative surface fluxes (sfc_lw_up / sfc_sw_up "
   145	            "/ sfc_sw_down) are only consumed by the rrtmgp and gray "
   146	            f"schemes; radiation scheme is {radiation_config.scheme!r}.")
   147	    if (sw_up is None) != (sw_dn is None):
   148	        raise ValueError(
   149	            "the prescribed surface shortwave boundary condition needs BOTH "
   150	            "'sfc_sw_up' and 'sfc_sw_down' in the radiation forcing; got "
   151	            f"only {'sfc_sw_up' if sw_up is not None else 'sfc_sw_down'!r}.")
   152	    if lw_up is not None:
   153	        _gray_e = radiation_config.gray.sfc_emissivity
   154	        try:
   155	            _gray_e_is_one = bool(np.all(np.asarray(_gray_e) == 1.0))
   156	        except (TypeError, jax.errors.TracerArrayConversionError):
   157	            _gray_e_is_one = False   # traced: cannot be shown to be 1
   158	        if radiation_config.scheme != "rrtmgp" and not _gray_e_is_one:
   159	            raise ValueError(
   160	                "forcing['sfc_lw_up'] prescribes the upwelling LW as "
   161	                "sigma*T_rad^4 (emissivity 1), but the gray radiation config "
   162	                f"has sfc_emissivity={_gray_e!r} and the gray backend takes "
   163	                "no per-call emissivity; set it to 1.0 or use rrtmgp.")
   164	        _lw = jnp.maximum(_col(lw_up), _PRESCRIBED_LW_UP_FLOOR_W_M2)
   165	        T_sfc_col = (_lw / constants.sigma_sb) ** 0.25  # coeff-ok: exact quartic root inverting sigma_sb*T^4
   166	        emis_col = 1.0  # coeff-ok: emissivity exactly 1 by construction
   167	    if sw_up is not None:
   168	        own = alb_col
   169	        if own is None:
   170	            own = (radiation_config.rrtmgp.sfc_albedo
   171	                   if radiation_config.scheme == "rrtmgp"
   172	                   else radiation_config.gray.sfc_albedo)
   173	        own_c = jnp.broadcast_to(jnp.asarray(own).reshape(-1), (ncol,))
   174	        _up = _col(sw_up)
   175	        _dn = _col(sw_dn)
   176	        alb_era5 = jnp.clip(
   177	            _up / jnp.maximum(_dn, _PRESCRIBED_ALBEDO_MIN_SW_DOWN_W_M2),
   178	            0.0, 1.0)  # coeff-ok: albedo is a dimensionless fraction in [0, 1]
   179	        # nan_to_num on the computed branch: jnp.where propagates NaN
   180	        # cotangents from the untaken branch.
   181	        alb_col = jnp.where(_dn >= _PRESCRIBED_ALBEDO_MIN_SW_DOWN_W_M2,
   182	                            jnp.nan_to_num(alb_era5, nan=0.0), own_c)
   183	    return T_sfc_col, alb_col, emis_col
   184	
   185	
   186	def resolve_land_radiative_bc(forcing, radiation_config, ncol,
   187	                              T_sfc_col, emis_col):
   188	    """Apply the land-coupled radiating surface, per column.
   189	
   190	    Reads the optional traced forcing pair ``T_sfc_rad`` [K] and
   191	    ``sfc_emissivity`` [1]: the area blend of the surface tiles' radiating
   192	    pairs, ``eps * sigma * T_sfc_rad**4 = sum_i f_i * eps_i * sigma *
   193	    T_rad_i**4`` with ``eps = sum_i f_i * eps_i`` (the coupler's tile-blend
   194	    rule, ``coupler/tile_fractions.py``).  The solver's upward LW
   195	    ``eps*sigma*T^4 + (1 - eps)*La`` is then the area sum of the tiles' own
   196	    grey-body LW, i.e. the LW the land balanced its energy with (tier-5
   197	    U11-1).  ``forcing["T_sfc"]`` stays the TURBULENT surface temperature.
   198	
   199	    Identity when both keys are absent.  Refuses one key without the other
   200	    (a half-applied blend) and a scheme that takes no per-column emissivity.
   201	    A prescribed ``sfc_lw_up`` applied afterwards still wins.
   202	    """
   203	    if forcing is None:
   204	        return T_sfc_col, emis_col
   205	    T_rad = forcing.get("T_sfc_rad")
  1035	            select case (icecldoptics)
  1036	            case ('ebertcurry')
  1037	               call ec_ice_get_rad_props_lw(state, pbuf, ice_lw_abs, oldicewp=.true.)
  1038	            case ('mitchell')
  1039	               call ice_cloud_get_rad_props_lw(state, pbuf, ice_lw_abs)
  1040	            case default
  1041	               call endrun('iccldoptics must be one either ebertcurry or mitchell')
  1042	            end select
  1043	
  1044	            select case (liqcldoptics)
  1045	            case ('slingo')
  1046	               call slingo_liq_get_rad_props_lw(state, pbuf, liq_lw_abs, oldliqwp=.true.)
  1047	            case ('gammadist')
  1048	               call liquid_cloud_get_rad_props_lw(state, pbuf, liq_lw_abs)
  1049	            case default
  1050	               call endrun('liqcldoptics must be either slingo or gammadist')
  1051	            end select
  1052	
  1053	            cld_lw_abs(:,:ncol,:) = liq_lw_abs(:,:ncol,:) + ice_lw_abs(:,:ncol,:)
  1054	
  1055	         end if
  1056	
  1057	         if (cldfsnow_idx > 0) then
  1058	
  1059	            ! add in snow
  1060	            call snow_cloud_get_rad_props_lw(state, pbuf, snow_lw_abs)
  1061	
  1062	            do i = 1, ncol
  1063	               do k = 1, pver
  1064	                  if (cldfprime(i,k) > 0._r8) then
  1065	                     c_cld_lw_abs(:,i,k) = ( cldfsnow(i,k)*snow_lw_abs(:,i,k) &
  1066	                                            + cld(i,k)*cld_lw_abs(:,i,k) )/cldfprime(i,k)
  1067	                  else
  1068	                     c_cld_lw_abs(:,i,k) = 0._r8
  1069	                  end if
  1070	               end do
  1071	            end do
  1072	         else
  1073	            c_cld_lw_abs(:,:ncol,:) = cld_lw_abs(:,:ncol,:)
  1074	         end if
  1075	
  1076	      end if   ! if (dolw)
  1077	
  1078	      call t_stopf('cldoptics')
473:   call pbuf_add_field('CLDFSNOW ',  'physpkg',dtype_r8,(/pcols,pver,dyn_time_lvls/), cldfsnow_idx)
531:      call pbuf_register_subcol('SADSNOW',     'micro_mg_cam_register', sadsnow_idx)
551:      call pbuf_register_subcol('CLDFSNOW ',   'micro_mg_cam_register', cldfsnow_idx)
564:      call pbuf_register_subcol('LS_REFFSNOW', 'micro_mg_cam_register', ls_reffsnow_idx)
758:   call addfld ('CLDFSNOW',   (/ 'lev' /), 'A', '1',        'Cloud fraction adjusted for snow'                        )
2931:   call outfld('CLDFSNOW',    cldfsnow,    psetcols, lchnk, avg_subcol_field=use_subcol_microp)
69:TUNED_SNOW_ALBEDO_MAX = 0.8388    # LandAlbedoConfig.alpha_snow_max
173:    per-column ``snow_cover_scale`` map built from the surfdata, and replacing
177:        "alpha_snow_max": _BIOPHYS_LMIP_SNOW_ALBEDO_MAX,
178:        "alpha_snow_min": _BIOPHYS_LMIP_SNOW_ALBEDO_MIN,
180:        "tau_snow_decay": _BIOPHYS_LMIP_SNOW_TAU_S,
235:# implies (>1); consumed as LandAlbedoConfig.snow_cover_scale (PFT-weighted per
661:        snow_cover_scale=(
666:        alpha_snow_max=TUNED_SNOW_ALBEDO_MAX_MULTILAYER,
667:        alpha_snow_min=TUNED_SNOW_ALBEDO_MIN_MULTILAYER,
669:        tau_snow_decay=TUNED_SNOW_TAU_DAYS_MULTILAYER * 86400.0,  # days -> s
packages/ice/legoesm/ice/shortwave.py:113:def maykut_untersteiner_albedo(
packages/ice/legoesm/ice/shortwave.py:196:    :func:`maykut_untersteiner_albedo` on the prognostic skin with
packages/ice/legoesm/ice/shortwave.py:206:        return maykut_untersteiner_albedo(
packages/ice/legoesm/ice/shortwave.py:342:    # Sign/formula walk (matches maykut_untersteiner_albedo): skin T is clamped
packages/ice/legoesm/ice/shortwave.py:456:        alpha = maykut_untersteiner_albedo(T_sfc, h_ice)
/glade/derecho/scratch/pg2328/cesm_src/CICE/cicecore/drivers/mapl/geos/ice_prescribed_mod.F90
/glade/derecho/scratch/pg2328/cesm_src/CICE/cicecore/drivers/mct/cesm1/ice_prescribed_mod.F90
/glade/derecho/scratch/pg2328/cesm_src/CICE/cicecore/drivers/nuopc/cmeps/ice_prescribed_mod.F90

exec
/usr/bin/bash -lc "nl -ba packages/land/legoesm/land/clm_surface_map.py | sed -n '130,185p;225,255p'; rg -n 'ALB_VIS|absorbed|alpha|sw_up|broadband' packages/land/legoesm/land/canopy/surface.py | head -45; rg -n 'def _make|mpas' packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py | head -25; nl -ba packages/ice/legoesm/ice/shortwave.py | sed -n '176,215p'; rg -n 'albedo|albice|albsno' /glade/derecho/scratch/pg2328/cesm_src/CICE/cicecore/drivers/mct/cesm1/ice_prescribed_mod.F90 | head -30; rg -n 't_ref2m|t_ref|temp2m|2 m|2m' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/{CanopyFluxesMod,BareGroundFluxesMod}.F90 | tail -20; nl -ba config/amip/amip_production.yaml | sed -n '285,354p'" in /glade/u/home/pg2328/legoesm
 succeeded in 44ms:
   130	_BIOPHYS_LMIP_PFT_THETA_FC = (
   131	    0.1706, 0.2249, 0.2351, 0.2253, 0.2890, 0.2492, 0.2918, 0.2676, 0.2599,
   132	    0.2635, 0.1995, 0.2045, 0.2164, 0.2171, 0.2193, 0.2123, 0.2469)  # const-ok: baked per-PFT calibration table (LMIP biophysics 2-deg)
   133	
   134	# Scalar snow / soil albedo settings of the SAME calibration. These are
   135	# LandAlbedoConfig field values; the offline LMIP consumes them through
   136	# ``lmip_config`` (physics.albedo), and the coupled path applies them in
   137	# ``land.config.apply_biophysics_lmip_two_leaf``.
   138	_BIOPHYS_LMIP_SNOW_ALBEDO_MAX = 0.8077    # [-] fresh-snow albedo
   139	_BIOPHYS_LMIP_SNOW_ALBEDO_MIN = 0.5207    # [-] aged-snow albedo
   140	_BIOPHYS_LMIP_SNOW_DCRIT = 15.4229        # [kg/m2] Niu-Yang half-cover SWE
   141	_BIOPHYS_LMIP_SNOW_TAU_S = 317424.96      # [s] snow-age decay (3.674 days)
   142	_BIOPHYS_LMIP_SOIL_DRY_BOOST = 0.1458     # [-] dry-soil brightening
   143	_BIOPHYS_LMIP_GLACIER_ALBEDO = (0.8178, 0.6178)   # [-] (visible, near-infrared)
   144	
   145	
   146	def biophysics_lmip_pft_root_params() -> dict:
   147	    """Per-PFT root / soil-moisture tables of the biophysics LMIP calibration.
   148	
   149	    Shape and ordering match every other ``pft_root_params`` consumer
   150	    (``build_canopy_land_params``, ``make_step_land_params_updater``): length-17
   151	    CLM5 order, applied at the column's DOMINANT PFT.
   152	    """
   153	    return {
   154	        "root_depth": _BIOPHYS_LMIP_PFT_ROOT_DEPTH_M,
   155	        "theta_wp": _BIOPHYS_LMIP_PFT_THETA_WP,
   156	        "theta_fc": _BIOPHYS_LMIP_PFT_THETA_FC,
   157	    }
   158	
   159	
   160	def biophysics_lmip_glacier_albedo() -> tuple:
   161	    """(visible, near-infrared) ice albedo of the same calibration.
   162	
   163	    Passed to the canopy parameter builder as ``glacier_alb``; without it the
   164	    builder takes its 0.70 / 0.50 default and the calibrated pair is inert.
   165	    """
   166	    return _BIOPHYS_LMIP_GLACIER_ALBEDO
   167	
   168	
   169	def biophysics_lmip_albedo_scalars() -> dict:
   170	    """``LandAlbedoConfig`` scalar fields of the biophysics LMIP calibration.
   171	
   172	    SCALARS ONLY, deliberately: the coupled ``land_albedo`` also carries the
   173	    per-column ``snow_cover_scale`` map built from the surfdata, and replacing
   174	    the whole object would throw that away.
   175	    """
   176	    return {
   177	        "alpha_snow_max": _BIOPHYS_LMIP_SNOW_ALBEDO_MAX,
   178	        "alpha_snow_min": _BIOPHYS_LMIP_SNOW_ALBEDO_MIN,
   179	        "snow_depth_crit": _BIOPHYS_LMIP_SNOW_DCRIT,
   180	        "tau_snow_decay": _BIOPHYS_LMIP_SNOW_TAU_S,
   181	        "soil_dry_albedo_boost": _BIOPHYS_LMIP_SOIL_DRY_BOOST,
   182	    }
   183	
   184	# per-PFT roughness length z0 [m] (drives the MOST surface exchange -> tall forests
   185	# rough ~1-2 m, grass/crop/bare smooth ~0.02-0.23 m).  Calibrated under MOST (the
   225	# conductance absorbing aerodynamic/humidity error; GLM review 2026-08-18).  Do NOT
   226	# reuse as photosynthetic capacity in the carbon cycle, and do NOT enable coupled
   227	# without a controlled A/B (#741 over-transpiration, predicted worse here).  Coupled
   228	# consumption is TRIPLE-gated: stomata.enabled AND carbon.scheme=="differland" AND a
   229	# prescribed carbon state, none of which AMIP sets by default.  Index 0 = bare soil.
   230	_TUNED_PFT_VCMAX_MULTILAYER = (0.00, 77.59, 57.59, 71.93, 52.12, 71.03, 42.98, 73.50, 62.42, 47.42, 67.29, 55.74, 63.28, 67.13, 28.99, 72.05, 58.87)
   231	_TUNED_PFT_G1_MULTILAYER = (1.000, 10.657, 8.395, 10.587, 9.276, 10.161, 9.896, 11.037, 10.713, 5.599, 10.608, 9.307, 8.592, 8.653, 6.095, 9.895, 7.483)
   232	_TUNED_PFT_LCMA_MULTILAYER = (60.92, 88.43, 83.77, 85.68, 53.69, 59.19, 60.97, 67.12, 62.24, 33.43, 52.47, 70.27, 46.52, 52.60, 38.34, 53.64, 46.91)
   233	# Per-PFT snow-cover masking scale (v7, canopy snow burial): forests hide ground
   234	# snow (<1), open tundra/shrub/crop whitens faster than the global snow_depth_crit
   235	# implies (>1); consumed as LandAlbedoConfig.snow_cover_scale (PFT-weighted per
   236	# cell).  Closed the NH>55 mean albedo bias (-0.028 -> +0.003).
   237	TUNED_PFT_SNOWMASK_MULTILAYER = (0.922, 0.885, 0.988, 1.027, 0.988, 0.998, 1.009, 0.979, 1.025, 1.087, 1.120, 1.048, 1.042, 0.990, 1.063, 0.993, 1.112)
   238	# btran needs theta_fc > theta_wp per PFT; PFT-weighting (a convex combination) then
   239	# preserves the ordering for every mixed cell, so the stress range never inverts.
   240	assert all(fc > wp for wp, fc in zip(_TUNED_PFT_WP_MULTILAYER, _TUNED_PFT_FC_MULTILAYER)), \
   241	    "multilayer plant theta_fc must exceed theta_wp for every PFT"
   242	# every baked per-PFT tuple must have exactly _N_PFT entries (a wrong-length paste is
   243	# the bake's main footgun -> a load-time tripwire instead of a deep matmul error).
   244	assert all(len(t) == _N_PFT for t in (
   245	    _TUNED_PFT_ALBEDO_MULTILAYER, _TUNED_PFT_EMISSIVITY_MULTILAYER,
   246	    _TUNED_PFT_ROOT_DEPTH_MULTILAYER, _TUNED_PFT_Z0_MULTILAYER, _TUNED_PFT_CH_MULTILAYER,
   247	    _TUNED_PFT_KSCALE_MULTILAYER, _TUNED_PFT_CSCALE_MULTILAYER,
   248	    _TUNED_PFT_WP_MULTILAYER, _TUNED_PFT_FC_MULTILAYER,
   249	    _TUNED_PFT_VCMAX_MULTILAYER, _TUNED_PFT_G1_MULTILAYER,
   250	    _TUNED_PFT_LCMA_MULTILAYER, TUNED_PFT_SNOWMASK_MULTILAYER)), \
   251	    f"every _TUNED_PFT_*_MULTILAYER tuple must have {_N_PFT} entries"
   252	
   253	# Per-variant lookup: snow-free per-PFT (albedo, emissivity, root_depth) columns +
   254	# glacier ice base albedo.  The slab W_max is reused for both (W_max is the bucket
   255	# store of the 1-layer slab; the Richards multilayer ignores it -> it gets no
rg: packages/land/legoesm/land/canopy/surface.py: IO error for operation on packages/land/legoesm/land/canopy/surface.py: No such file or directory (os error 2)
10:- "mpas"         : MPASPrimitiveEquationModel (Voronoi mesh, sigma coords)
272:def _make_T_sfc_override_cell():
296:def _make_time_state():
1360:        One of "hydrostatic", "nonhydrostatic", "plane", "mpas_nh",
1361:        "spectral_pe", "mpas".
1415:            "hydrostatic", "mpas", "spectral_pe"):
1418:            f"model_type='hydrostatic'/'mpas'/'spectral_pe', got {model_type!r}.  Extend the "
1429:            "hydrostatic", "mpas"):
1432:            "rsutcs/rlutcs) is only wired for model_type 'hydrostatic'/'mpas', "
1472:    if cloud_fraction_tracer is not None and model_type != "mpas":
1477:        if model_type != "mpas":
1505:    elif model_type == "mpas_nh":
1506:        return _make_mpas_nh_radiation(radiation_config, rrtmgp_solver,
1518:    elif model_type == "mpas":
1519:        return _make_mpas_radiation(radiation_config, rrtmgp_solver,
1532:            f"'mpas_nh', 'spectral_pe', 'mpas'."
1554:def _make_hydrostatic_radiation(
2181:def _make_ecrad_ifs_step(radiation_config, full_call, column_mesh):
2294:_make_mpas_radiation = _make_hydrostatic_radiation
2301:def _make_nonhydrostatic_radiation(
2681:def _make_plane_radiation(
2866:def _make_mpas_nh_radiation(
3051:def _make_spectral_pe_radiation(
3377:# _make_mpas_radiation is defined as an alias above (= _make_hydrostatic_radiation)
   176	    # the open-water value (1 mm ice would be darker than the ocean it sits in).
   177	    # Floor at albedo_ocean so thin ice asymptotes to open water, never darker.
   178	    return jnp.maximum(alpha_regime * thickness_factor, albedo_ocean)
   179	
   180	
   181	# ==============================================================================
   182	# Delta-Eddington two-band albedo (Briegleb & Light 2007 surrogate)
   183	# ==============================================================================
   184	
   185	
   186	def sea_ice_albedo(
   187	    scheme: str,
   188	    albedo_ice: float,
   189	    T_skin=None,
   190	    h_ice_m: float | None = None,
   191	    melt_width_K: float = _T_MELT_WIDTH_K,
   192	):
   193	    """Sea-ice albedo a lane's radiation sees, by resolved scheme.
   194	
   195	    ``"constant"`` -> the scalar ``albedo_ice``; ``"maykut_untersteiner"`` ->
   196	    :func:`maykut_untersteiner_albedo` on the prognostic skin with
   197	    ``albedo_ice`` as the cold end.  Unknown schemes raise.
   198	    """
   199	    if scheme == "constant":
   200	        return albedo_ice
   201	    if scheme == "maykut_untersteiner":
   202	        if T_skin is None or h_ice_m is None:
   203	            raise ValueError(
   204	                "sea_ice_albedo('maykut_untersteiner') needs the prognostic "
   205	                "ice skin temperature and slab thickness.")
   206	        return maykut_untersteiner_albedo(
   207	            T_skin, h_ice_m, albedo_cold_bare=albedo_ice, T_width=melt_width_K)
   208	    raise ValueError(
   209	        f"sea_ice_albedo scheme={scheme!r} unknown; choose 'constant' or "
   210	        "'maykut_untersteiner' (resolve 'auto' first).")
   211	
   212	def _band_albedo_snow(
   213	    melt_fraction: jnp.ndarray,
   214	    h_snow: jnp.ndarray,
   215	    *,
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:402:            call dis_coiS(tc_ref2m(p), rh_ref2m(p), wbt_ref2m(p), discomf_index_ref2mS(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:404:               call Wet_Bulb(t_ref2m(p), vap_ref2m(p), forc_pbot(c), rh_ref2m(p), q_ref2m(p), &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:405:                               teq_ref2m(p), ept_ref2m(p), wb_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:406:               call dis_coi(tc_ref2m(p), wb_ref2m(p), discomf_index_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:407:               call THIndex(tc_ref2m(p), wb_ref2m(p), thic_ref2m(p), thip_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:408:               call SwampCoolEff(tc_ref2m(p), wb_ref2m(p), swmp80_ref2m(p), swmp65_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:412:              wbt_ref2m_r(p)            = wbt_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:413:              nws_hi_ref2m_r(p)         = nws_hi_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:414:              appar_temp_ref2m_r(p)     = appar_temp_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:415:              swbgt_ref2m_r(p)          = swbgt_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:416:              humidex_ref2m_r(p)        = humidex_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:417:              discomf_index_ref2mS_r(p) = discomf_index_ref2mS(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:419:                 teq_ref2m_r(p)            = teq_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:420:                 ept_ref2m_r(p)            = ept_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:421:                 wb_ref2m_r(p)             = wb_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:422:                 discomf_index_ref2m_r(p)  = discomf_index_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:423:                 thic_ref2m_r(p)           = thic_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:424:                 thip_ref2m_r(p)           = thip_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:425:                 swmp80_ref2m_r(p)         = swmp80_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:426:                 swmp65_ref2m_r(p)         = swmp65_ref2m(p)
   285	cam6_aerosol_dir: /work/bb1596/b309178/diffESM/legoesm_ap/data/cam6_optics
   286	energy_consistent_moisture_clip: true
   287	
   288	# ---- convection ---------------------------------------------------------------
   289	zm_land_fraction: required  # ZM reads the land mask (c0_lnd vs c0_ocn); "none" = aquaplanet only
   290	convection: zhang_mcfarlane # CAM6: faithful zm_conv port (f09 mg2+clubb namelist:
   291	                            # c0_lnd 0.0075, c0_ocn 0.03, ke 5e-6 (dyn=fv),
   292	                            # tau 3600,
   293	                            # capelmt 70, tiedke_add 0.5, momcu = momcd = 0.7,
   294	                            # num_cin 1); scheme defaults carry those values.
   295	                            # ZM convective momentum transport (momtran) is ON
   296	                            # and reaches the MPAS winds (mpas_cmt resolves True;
   297	                            # the Sundqvist/L36 deck's Bechtold runs CMT
   298	                            # off, lane default)
   299	convective_rain_to_surface: true
   300	convective_cloud: false     # refused by cam6_clubb (deepcu is CAM6's version)
   301	# Bechtold/IFS settings written explicitly (inert here: convection is ZM).
   302	bechtold_use_ifs_ascent: false         # faithful OpenIFS chain off
   303	bechtold_ifs_shallow_nosupply: zero    # no sub-cloud supply -> shallow flux ramps to ~0 (OpenIFS: first_guess)
   304	# ZM tunables written explicitly (values = the scheme defaults = the CAM6 f09
   305	# namelist above), so a campaign arm's change is one visible row.
   306	zm_c0_lnd: 0.0075           # rain conversion over land [1/m]
   307	zm_c0_ocn: 0.03             # rain conversion over ocean [1/m]
   308	zm_ke: 5.0e-6               # rain evaporation efficiency
   309	zm_dmpdz: -1.0e-3           # dilute-plume entrainment [1/m]
   310	zm_tau: 3600.0              # CAPE consumption timescale [s]
   311	zm_capelmt: 70.0            # deep-convection CAPE threshold [J/kg]
   312	# ZM detrainment smoothing widths (RULE 5; values = the scheme defaults; 0 =
   313	# CAM's hard switch), read by the CLUBB macmic sub-step's detrainment booking.
   314	zm_detrain_phase_width_K: 0.1              # liquid/ice ramp corners [K]
   315	zm_detrain_number_width_kgkg_s: 1.0e-12    # max(0, dlf) in the number source [kg/kg/s]
   316	zm_transport_cloud_tracers: true  # ZM convtran of carried cloud/precip water + numbers (CAM6)
   317	
   318	# ---- turbulence ---------------------------------------------------------------
   319	turbulence: clubb
   320	clubb_prognostic: true
   321	clubb_liquid_partition: true       # CAM6 clubb_intr: the closure's diagnosed
   322	                                   # cloud liquid is handed to the host as a
   323	                                   # tendency, and the Morrison saturation
   324	                                   # adjustment is switched off, both
   325	                                   # condensation and (since 2026-10-19, user
   326	                                   # decision) cloud-water evaporation (MG2
   327	                                   # allow_sed_supersat = .false. under CLUBB,
   328	                                   # micro_mg_cam.F90:668-672).  One flag,
   329	                                   # both halves.  User table row 1.
   330	# The partition REFUSES an active radiative condensate floor (it would be
   331	# imposed on top of the liquid the closure now supplies, making clouds opaque
   332	# twice).  Both floors default to a non-zero scheme value when unset, so they
   333	# are pinned to 0 here.  Not in the user's three-row table: REQUIRED BY ROW 1.
   334	cloud_q_c_diagnostic: 0.0
   335	conv_cloud_condensate: 0.0
   336	# CLUBB upper domain limit = CAM6's trop_cloud_top_press 1.D2 (100 Pa;
   337	# cam_cesm2_1_rel_60 namelist_defaults_cam.xml:1295).  User decision
   338	# 2026-10-20 ("use cam6 rule"), was 15000 Pa.  Offline 1/3-day A/B on a
   339	# day-105 production checkpoint: stable, sparse stratospheric CLUBB cloud
   340	# (scripts/validate/oracle/clubb_glue/top_press_scm_ab.py).
   341	clubb_trop_cloud_top_press: 100.0
   342	# CLUBB tunables written explicitly (values = the port's CLUBBParams defaults,
   343	# the CAM6 clubb namelist).  clubb_c_K10h (0.3) is read by the eddy-scalar
   344	# rtm/thlm diffusion (CAM l_do_expldiff_rtm_thlm) but has no deck key, so the
   345	# CLUBBParams default applies.
   346	clubb_c14: 2.2
   347	clubb_c8: 4.2
   348	clubb_c11: 0.7
   349	clubb_c11b: 0.35
   350	clubb_gamma_coef: 0.308
   351	clubb_gamma_coefb: 0.32
   352	clubb_beta: 2.4
   353	clubb_c_k10: 0.5
   354	

codex
C2’s missing canopy-snow store is verified. CLM5’s default `ON_RAD` branch derives snow cover from stored canopy snow with a 0.15 exponent. The supplied probe compares against our translated two-stream routine, so it does not independently validate CLM agreement. C5’s height and drag-law defects are verified; the claimed warm-bias sign remains unproven.
exec
/usr/bin/bash -lc "rg -n 'snow_insulation|T_snow|snow_thermal' packages/land/legoesm/land/multilayer_land.py | head -30; nl -ba packages/land/legoesm/land/multilayer_land.py | sed -n '752,802p'; rg -n 't_ref2m\\(p\\) =' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/{CanopyFluxesMod,BareGroundFluxesMod}.F90; rg -n 'sea_ice_albedo\\(' packages/coupler/legoesm/driver/model_driver.py; nl -ba config/amip/amip_production.yaml | sed -n '390,442p'; nl -ba scripts/tmp/probe_snow_albedo_canopy.py | sed -n '27,65p;100,115p'; git status --porcelain -uno" in /glade/u/home/pg2328/legoesm
 succeeded in 18ms:
72:    snow_thermal_props,
585:    _snow_node = bool(config.thermal.snow_insulation)
587:        if state.T_snow is None:
589:                "config.thermal.snow_insulation is on but state.T_snow is None; "
596:                "config.thermal.snow_insulation (one-layer snow thermal node) is "
601:                "config.thermal.snow_insulation is the one-layer thermal node of "
609:        snow, T_snow0, _store_melt, _store_heat = melt_snow_node_excess(
610:            snow, state.T_snow)
612:        T_snow0 = jnp.where(snow > 0.0, T_snow0, T_soil[:, 0])
621:        T_surface = _node_skin(T_snow0, T_soil[:, 0], f_snow)
831:                C_s, coeff_s, rb_s = snow_thermal_props(pack, scc, f_snow, config.land_albedo.snow_depth_crit)
842:                    T_snow0, snow, T_soil, theta, grid,
1150:            cfg=bands, T_snow_melt=config.T_snow_melt,
1187:            T_snow_melt=config.T_snow_melt,
1534:        C_s, coeff_s, rb_s = snow_thermal_props(pack, scc, f_snow, config.land_albedo.snow_depth_crit)
1540:            T_snow_top_ref=T_pack_top_ref)
1559:            snow_new, snow_age, precip_snow_eff, dt, T_snow=T_surface,
1730:            _T_start = jnp.where(snow > 0.0, T_snow0,
1732:            T_snow_new, T_soil_new = solve_snow_node_soil_thermal(
1746:        T_snow_new = None
1813:        T_snow=_match(T_snow_new, state.T_snow),
1828:            T_snow_new, T_surface_new,
2086:def _node_skin(T_snow, T_soil_top, f_snow):
2087:    """Snow-node skin ``f*min(T_snow, T_freeze) + (1-f)*T_soil_top`` (a pack's
2089:    return (f_snow * jnp.minimum(T_snow, constants.T_freeze)
2449:        T_snow=(T_soil[:, 0] if config.thermal.snow_insulation else None),
2462:    if state.T_snow is None:
2464:    return _node_skin(state.T_snow, state.T_soil[:, 0],
2470:    """Give a state that lacks it a snow-node temperature ``T_snow``.
2478:    T_snow = jnp.where(state.snow_depth > 0.0,
   752	    _w_ground_snow = f_snow if _partial_cover else has_snow.astype(T_surface.dtype)
   753	    _L_s_T = latent_heat_sublimation(T_surface)
   754	
   755	    # --- Root distribution and per-layer moisture stress (shared) ---
   756	    theta_r = config.hydraulics.theta_r
   757	    z_centers = grid.z_node
   758	
   759	    # ``root_depth``, ``theta_wp``, ``theta_fc`` may be scalars (from the
   760	    # config) or per-column arrays (from ``LandSurfaceParams``).  Handle
   761	    # both by promoting to (ncol,) shape.
   762	    def _to_ncol(v):
   763	        arr = jnp.asarray(v)
   764	        if arr.ndim == 0:
   765	            return jnp.broadcast_to(arr, (ncol,))
   766	        return arr
   767	
   768	    root_depth_c = _to_ncol(root_depth)
   769	    # Root-zone stress uses the PLANT wilting point (transpiration extraction
   770	    # limit); the soil wilting point ``theta_wp`` is the soil-column reference.
   771	    theta_wp_c   = _to_ncol(theta_wp_plant)
   772	    theta_fc_c   = _to_ncol(theta_fc)
   773	
   774	    root_frac = jnp.exp(-z_centers[None, :] / root_depth_c[:, None])
   775	    root_frac = root_frac / jnp.sum(root_frac, axis=-1, keepdims=True)
   776	
   777	    # --- Moisture availability from root-zone water content ---
   778	    # Root-zone moisture stress via the shared helper (single source of truth;
   779	    # see root_zone_beta_soil for the audit-#6 / iter-65 theta_fc-theta_wp
   780	    # range floor).  theta_wp_c / theta_fc_c are already promoted to (ncol,),
   781	    # so use the spatial (per-column) path.  beta_root (per-layer) is reused
   782	    # below for the root-water-uptake sink partition.  (#823 keeps this
   783	    # root_zone_beta_soil path — numerically identical to main's canonical
   784	    # root_zone_moisture_stress, which stays available for the coupler land
   785	    # tile via land_tile_beta_soil — because the soil-C-spin-up + carbon audit
   786	    # downstream reuse the inline root_frac / theta_wp_c / theta_fc_c / w_frac_rz.)
   787	    beta_soil, beta_root = root_zone_beta_soil(
   788	        theta_avail, root_frac, theta_wp_c, theta_fc_c, config.beta_min,
   789	        spatial=True,
   790	    )
   791	    # Root-zone-integrated wetness (vegetation-cover proxy), reused downstream
   792	    # for the transpiration/bare-soil evaporation partition (f_veg).  Same
   793	    # reduction the helper applies internally for beta_soil.
   794	    w_frac_rz = jnp.clip(jnp.sum(root_frac * beta_root, axis=-1), 0.0, 1.0)
   795	
   796	    # Top-layer effective saturation S_top = (theta0 - theta_r)/(theta_sat - theta_r),
   797	    # used by BOTH surface schemes' bare-soil-evaporation resistance.  Compute it in
   798	    # the (ncol, nlayers) layer space THEN take layer 0, so a per-column (ncol, 1)
   799	    # theta_r / theta_sat (spatial hydraulics, e.g. the coupled land tile) broadcasts
   800	    # against the 2-D theta rather than outer-producting against the 1-D theta[:, 0]
   801	    # to (ncol, ncol) — the shape bug behind the coupled multilayer-land driver crash.
   802	    # .astype(theta.dtype): a per-column theta_r/theta_sat may be float64 while the
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/BareGroundFluxesMod.F90:378:         t_ref2m(p) = thm(p) + temp1(p)*dth(p)*(1._r8/temp12m(p) - 1._r8/temp1(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1182:         t_ref2m(p) = thm(p) + temp1(p)*dth(p)*(1._r8/temp12m(p) - 1._r8/temp1(p))
14224:            return sea_ice_albedo(
   390	# CESM2.1 runs gust_fac = 0, and the large_yeager_cesm law never reads it.)
   391	surface_tiled: false
   392	surface_z0_land: 0.1
   393	mpas_land_lapse_K_per_km: 6.5
   394	mpas_land_beta: 0.6
   395	mpas_ice_skin_prognostic: true
   396	albedo_ice: 0.80
   397	# Melt-dependent sea-ice albedo on the prognostic skin (#2106, user decision
   398	# 2026-10-24 after the July 1979 A/B: Arctic clear-sky reflected SW -19.2
   399	# W/m2).  Named explicitly although "auto" resolves to it here.
   400	sea_ice_albedo_scheme: maykut_untersteiner
   401	sea_ice_melt_ramp_width_K: 1.0
   402	ice_skin_melt_cap_width_K: 0.01
   403	dynamic_albedo: true
   404	snow_albedo_feedback: true
   405	land_snow_tau_days: 150.0
   406	snow_age_activation_K: 5000.0
   407	
   408	# ---- land (unchanged) -----------------------------------------------------------
   409	use_multilayer_land: true
   410	land_update_seconds: 1800.0  # CAM6: CLM steps once per 1800 s coupling
   411	                             # interval (user 2026-09-25).  = 16 x 112.5 s,
   412	                             # forced with the interval-mean atmosphere; the
   413	                             # offline LMIP driver also runs the tile at 1800 s.
   414	                             # Was 300 (-> 337.5 s actual, every 3 steps).
   415	multilayer_n_layers: 10
   416	multilayer_soil_depth: 3.0
   417	# Soil freeze/thaw ON (user 2026-09-26), as in CLM5 and as the land IC spin-up
   418	# ran; the land calibration's tables were fitted with it off.
   419	land_soil_freeze_thaw: true
   420	# Snowpack: explicit bulk (one SWE reservoir), as every prior production run.
   421	# Moving production to the layered pack (bulk -> layered) is an OPEN user decision.
   422	land_snow_scheme: bulk
   423	# Frozen soil impedes water flow, CLM5 form 10**(-e * ice fraction), e = 6
   424	# (user 2026-09-28).  Active because freeze/thaw is on.
   425	land_soil_ice_impedance_exponent: 6.0
   426	# Soil-water numerics (user 2026-10-14): modified van Genuchten air-entry head [m].
   427	land_soil_vg_air_entry_m: -0.02
   428	# One-layer snow thermal node OFF (record only; A/B switch added 2026-10-03,
   429	# user decision: off until the Sep-Dec arm verdict).
   430	land_snow_insulation: false
   431	land_soil_moisture_init_frac: 0.25
   432	land_stomatal_beta: false
   433	land_calibrated_physics: true
   434	land_surface_scheme: two_leaf
   435	land_canopy_max_iters: 10   # NEW BASELINE 2026-10-05 (user): A/B vs 60 = 4.2x/2.7x faster, land LH -1.5/-3.4 %; see fv3_duo_gaps/q_review/deck_ledgers.md
   436	# Pinned at the current values (b0 stressed, no interception) so the deck
   437	# records them; changing either is a labelled land baseline, not a default flip.
   438	land_canopy_stress_b0: true
   439	land_canopy_interception: false
   440	# Two-leaf canopy solver smoothing widths, named so every run records them
   441	# (user 2026-09-30): canopy-air RH <= 1 cap and stable zeta <= 0.5 cap.
   442	land_canopy_rh_cap_smoothing_width: 0.01
    27	SWE = 120.0          # kg/m2
    28	COSZ, SWD = 0.4, 400.0
    29	DAY = 86400.0
    30	
    31	scal = biophysics_lmip_albedo_scalars()
    32	arms = {
    33	    "prod 150d+T": (150.0 * DAY, 5000.0),
    34	    "calib 3.67d": (scal["tau_snow_decay"], 0.0),
    35	}
    36	
    37	
    38	def age_after(act, schedule):
    39	    age = jnp.zeros(1)
    40	    swe = jnp.full(1, SWE)
    41	    for ndays, T in schedule:
    42	        for _ in range(ndays):
    43	            age = update_snow_age(swe, age, jnp.zeros(1), DAY, T_snow=jnp.full(1, T),
    44	                                  age_activation_K=act)
    45	    return age
    46	
    47	
    48	stages = {
    49	    "fresh": [],
    50	    "90d at 255K": [(90, 255.0)],
    51	    "+10d melting": [(90, 255.0), (10, constants.T_freeze)],
    52	}
    53	
    54	n = len(COLS)
    55	pft = jnp.asarray([c[1] for c in COLS], float)
    56	lai = jnp.asarray([c[2] for c in COLS]); sai = jnp.asarray([c[3] for c in COLS])
    57	htop = jnp.asarray([c[4] for c in COLS]); hbot = jnp.asarray([c[5] for c in COLS])
    58	base = [jnp.asarray([c[6] for c in COLS]), jnp.asarray([c[7] for c in COLS])]
    59	swe = jnp.full(n, SWE); cz = jnp.full(n, COSZ); lat = jnp.full(n, np.deg2rad(60.0))
    60	pd, pf, nd, nf, _ = rt.split_sw_components(jnp.full(n, SWD), cz)
    61	fdif = (pf / (pd + pf), nf / (nd + nf))
    62	
    63	print(f"SWE {SWE} kg/m2, cosz {COSZ}; broadband = rt.broadband_albedo(vis, nir)")
    64	for arm, (tau, act) in arms.items():
    65	    la = LandAlbedoConfig()._replace(**{**scal, "tau_snow_decay": tau,

codex
**C1 is the strongest candidate; none of these claims establishes annual temperature attribution.** Verified below by source inspection; probe outputs and the historical run’s resolved configuration were not independently reproduced.

- **C1 — CONFIRMED mechanism; PLAUSIBLE “main driver.”** The default is false, and the active two-leaf path applies snow albedo to both bands before exporting their broadband blend ([config.py:1424](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/config.py:1424), [multilayer_land.py:916](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/multilayer_land.py:916), [export:1869](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/multilayer_land.py:1869)). “Every snow-covered column” overstates this: partial cover and existing cover scaling matter. Recommend a controlled masking comparison before a production flip; neither the CLM albedo range nor causal dominance follows from this code.

- **C2 — CONFIRMED omission; numerical comparison unverified.** Masking uses `fcansno=0` and a **delta correction**, not CLM’s absolute column albedo ([radiative_transfer.py:630](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/canopy/radiative_transfer.py:630)). Production has interception off; its state has optional canopy water but no separate snow store ([state.py:71](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/state.py:71)). CLM5 defaults to `ON_RAD`; under its vegetation/water guards, `fcansno=min[1,(snocan/(60·dewmx·vegt))^0.15]` ([CanopyHydrologyMod.F90:717](/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:717)). Porting only this formula would be incomplete: interception, unloading, phase changes and budgets belong together. The probe’s “CLM reference” calls our translation.

- **C3 — PLAUSIBLE mechanism; exact numbers unsupported.** Production overrides the calibration’s decay time ([model_driver.py:2030](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:2030)). However, the probe changes **both** decay time and activation temperature ([probe:32](/glade/u/home/pg2328/legoesm/scripts/tmp/probe_snow_albedo_canopy.py:32)); it cannot isolate either. Current calibrated aged-snow albedo is 0.5207, also challenging the quoted 0.51 ([clm_surface_map.py:138](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/clm_surface_map.py:138)). Retain the user decision; calibration alone does not establish fidelity.

- **C4 — CONFIRMED missing snow optics; PLAUSIBLE cooling.** MPAS feeds only cloud liquid/ice to radiation ([integration.py:2011](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2011)). CAM explicitly combines snow with clouds in SW and LW ([radiation.F90:969](/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90:969), [LW:1057](/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90:1057)). Support the fidelity port, but annual warming is unproven: shortwave effects and inversions complicate the sign.

- **C5 — CONFIRMED diagnostic defect; PLAUSIBLE warm bias.** Land selects ocean `large_yeager` and inherits the 10-m reference height ([diagnostics.py:874](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/diagnostics.py:874), [bulk_flux.py:1354](/glade/u/home/pg2328/legoesm/packages/core/legoesm/core/bulk_flux.py:1354)). Fix using land exchange variables, displacement and actual height; true-height adjustment alone does not guarantee CLM screen temperature.

- **C6 — REFUTED flat-0.80 premise.** Production explicitly selects melt-dependent albedo ([deck:395](/glade/u/home/pg2328/legoesm/config/amip/amip_production.yaml:395)). Remaining CICE fidelity differences require separate assessment.

**Tentative physical-impact ranking:** C1 > C3 ≈ C4 > C2 > C6; C5 changes reported tas directly. Biggest omitted candidate: bulk snow with insulation disabled ([deck:422](/glade/u/home/pg2328/legoesm/config/amip/amip_production.yaml:422)). Its seasonal sign and >1-K annual impact remain unverified.
tokens used
94,651
**C1 is the strongest candidate; none of these claims establishes annual temperature attribution.** Verified below by source inspection; probe outputs and the historical run’s resolved configuration were not independently reproduced.

- **C1 — CONFIRMED mechanism; PLAUSIBLE “main driver.”** The default is false, and the active two-leaf path applies snow albedo to both bands before exporting their broadband blend ([config.py:1424](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/config.py:1424), [multilayer_land.py:916](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/multilayer_land.py:916), [export:1869](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/multilayer_land.py:1869)). “Every snow-covered column” overstates this: partial cover and existing cover scaling matter. Recommend a controlled masking comparison before a production flip; neither the CLM albedo range nor causal dominance follows from this code.

- **C2 — CONFIRMED omission; numerical comparison unverified.** Masking uses `fcansno=0` and a **delta correction**, not CLM’s absolute column albedo ([radiative_transfer.py:630](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/canopy/radiative_transfer.py:630)). Production has interception off; its state has optional canopy water but no separate snow store ([state.py:71](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/state.py:71)). CLM5 defaults to `ON_RAD`; under its vegetation/water guards, `fcansno=min[1,(snocan/(60·dewmx·vegt))^0.15]` ([CanopyHydrologyMod.F90:717](/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyHydrologyMod.F90:717)). Porting only this formula would be incomplete: interception, unloading, phase changes and budgets belong together. The probe’s “CLM reference” calls our translation.

- **C3 — PLAUSIBLE mechanism; exact numbers unsupported.** Production overrides the calibration’s decay time ([model_driver.py:2030](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/model_driver.py:2030)). However, the probe changes **both** decay time and activation temperature ([probe:32](/glade/u/home/pg2328/legoesm/scripts/tmp/probe_snow_albedo_canopy.py:32)); it cannot isolate either. Current calibrated aged-snow albedo is 0.5207, also challenging the quoted 0.51 ([clm_surface_map.py:138](/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/clm_surface_map.py:138)). Retain the user decision; calibration alone does not establish fidelity.

- **C4 — CONFIRMED missing snow optics; PLAUSIBLE cooling.** MPAS feeds only cloud liquid/ice to radiation ([integration.py:2011](/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py:2011)). CAM explicitly combines snow with clouds in SW and LW ([radiation.F90:969](/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90:969), [LW:1057](/glade/derecho/scratch/pg2328/cesm_src/CAM/src/physics/rrtmg/radiation.F90:1057)). Support the fidelity port, but annual warming is unproven: shortwave effects and inversions complicate the sign.

- **C5 — CONFIRMED diagnostic defect; PLAUSIBLE warm bias.** Land selects ocean `large_yeager` and inherits the 10-m reference height ([diagnostics.py:874](/glade/u/home/pg2328/legoesm/packages/coupler/legoesm/driver/diagnostics.py:874), [bulk_flux.py:1354](/glade/u/home/pg2328/legoesm/packages/core/legoesm/core/bulk_flux.py:1354)). Fix using land exchange variables, displacement and actual height; true-height adjustment alone does not guarantee CLM screen temperature.

- **C6 — REFUTED flat-0.80 premise.** Production explicitly selects melt-dependent albedo ([deck:395](/glade/u/home/pg2328/legoesm/config/amip/amip_production.yaml:395)). Remaining CICE fidelity differences require separate assessment.

**Tentative physical-impact ranking:** C1 > C3 ≈ C4 > C2 > C6; C5 changes reported tas directly. Biggest omitted candidate: bulk snow with insulation disabled ([deck:422](/glade/u/home/pg2328/legoesm/config/amip/amip_production.yaml:422)). Its seasonal sign and >1-K annual impact remain unverified.
